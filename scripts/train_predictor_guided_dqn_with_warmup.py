import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from src.modules import DuelingDQN, ReplayBuffer
from src.utils import set_seed, normalize_obs, process_reward, 
from student_client import create_student_gym_env
import joblib
import logging

# 日志配置
logging.basicConfig(filename='train_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# 加载 predictor
health_predictor = joblib.load('checkpoints/linear.joblib')
    
def warmup_replay_buffer(env, buffer, num_episodes=50, max_steps=300):
    def strong_policy(obs, repair_cnt):
        prob = health_predictor.predict_proba(obs)[0, 1]
        # prob是 0 1 2 轮死掉的概率，越大越不健康
        if prob >= 0.4:
            if repair_cnt < 4:
                return 1 # repair
            else:
                return 2 # 卖掉
        else:
            return 0 # do nothing
    print(f"[Warmup] Collecting {num_episodes} episodes with strong policy...")
    for ep in range(num_episodes):
        obs, info = env.reset()
        obs = normalize_obs(np.array(obs, dtype=np.float32))
        done = False
        t = 0
        repair_cnt = 0
        while not done and t < max_steps:
            action = strong_policy(obs, repair_cnt)
            if action == 1:
                repair_cnt += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward = process_reward(reward)
            next_obs = normalize_obs(np.array(next_obs, dtype=np.float32))
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            t += 1
        print(f"[Warmup] Episode {ep+1}/{num_episodes}, steps={t}")
    print(f"[Warmup] Buffer size after warmup: {len(buffer)}")


def predictor_guided_epsilon_greedy(obs, q_values, epsilon=0.6, danger_threshold=0.4):
    if np.random.rand() > epsilon:
        print(f"      Exploitation: obs={obs}, q_values={q_values}")
        return int(np.argmax(q_values))
    else:
        p_fail = health_predictor.predict_proba(obs.reshape(1, -1))[0, 1]
        print(f"      Predictor-guided exploration:  q_values={q_values}, p_fail={p_fail:.3f}")
         # 探索时的分布：低风险偏0，高风险偏1；卖掉很少发生，除非极高风险
        if p_fail < danger_threshold:
            probs = np.array([0.70, 0.25, 0.05])
        elif p_fail < 0.95:  # very_danger = 0.95
            probs = np.array([0.20, 0.75, 0.05])
        else:
            probs = np.array([0.05, 0.70, 0.25])  # 极高风险才明显提高卖掉
        return int(np.random.choice([0,1,2], p=probs))

if __name__ == '__main__':
    from dotenv import load_dotenv
    
    set_seed(42)
    load_dotenv()
    env = create_student_gym_env()
    obs_dim = 9
    act_dim = env.action_space.n
    buffer = ReplayBuffer(2000)
    device = torch.device('cpu')
    policy_net = DuelingDQN(obs_dim, act_dim).to(device)
    target_net = DuelingDQN(obs_dim, act_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    def sync_target():
        target_net.load_state_dict(policy_net.state_dict())

    num_episodes = 3000
    batch_size = 64
    gamma = 0.999
    target_update = 20
    epsilon_start = 0.6
    epsilon_end = 0.05
    epsilon_decay = 200

    resume_checkpoint = os.environ.get('RESUME_CKPT', None)

    start_ep = 0
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        buffer.load(checkpoint['buffer_path'])
        start_ep = checkpoint['epoch']
        print(f"Resumed model/optimizer from {resume_checkpoint}, start from episode {start_ep}")
    else:
        warmup_replay_buffer(env, buffer, num_episodes=200)



    for ep in range(num_episodes):
        try:
            obs, info = env.reset()
        except Exception as e:
            logging.error(f"Reset failed at episode {ep+1}: {e}")
            print(f"Reset failed at episode {ep+1}: {e}")
            continue

        obs = normalize_obs(np.array(obs, dtype=np.float32))
        total_reward = 0
        done = False
        t = 0
        step_logs = []
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * ep / epsilon_decay)

            # Obs
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                q_values = policy_net(obs_tensor).cpu().numpy()[0]

            # Action
            action = predictor_guided_epsilon_greedy(obs, q_values, epsilon=epsilon)

            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                logging.error(f"Step failed at episode {ep+1}, step {t}: {e}")
                print(f"Step failed at episode {ep+1}, step {t}: {e}")
                break

            # Reward
            reward = process_reward(reward)
            next_obs = normalize_obs(np.array(next_obs, dtype=np.float32))
            done = terminated or truncated

            # Store transition
            buffer.push(obs, action, reward, next_obs, done)

            step_log = f"Episode {ep+1}, Step {t}, Action: {action}, Epsilon: {epsilon:.3f}, Reward: {reward:.2f}, Done: {done}"
            step_logs.append(step_log)
            logging.info(step_log)
            print(f"    {step_log}")

            # Update
            obs = next_obs
            total_reward += reward

            # Train
            t += 1
            if len(buffer) >= batch_size:
                s, a, r, s_, d = buffer.sample(batch_size)
                s = torch.tensor(s, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.long, device=device)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                s_ = torch.tensor(s_, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)
                q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(s_).max(1)[0]
                    q_target = r + gamma * q_next * (1 - d)
                loss = loss_fn(q, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if ep % target_update == 0:
            sync_target()
        
        # 保存模型和优化器
        torch.save({
            'epoch': ep + 1,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': total_reward,
            'epsilon': epsilon,
            'buffer_path': f'checkpoints/DDQN_bufferwarmup/buffer_ep{ep+1}.pkl'
        }, f'checkpoints/DDQN_bufferwarmup/buffer_warmup_checkpoint_ep{ep+1}.pth')
        buffer.save(f'checkpoints/DDQN_bufferwarmup/buffer_ep{ep+1}.pkl')

        episode_log = f"Episode {ep+1} finished, Total Reward: {total_reward:.2f}, Steps: {t}, Epsilon: {epsilon:.3f}"
        logging.info(episode_log)
        print(episode_log)
