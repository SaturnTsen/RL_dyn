import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from utils import normalize_obs, process_reward
from student_client import create_student_gym_env
import joblib
import logging

# 日志配置
logging.basicConfig(filename='train_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# 加载 predictor
health_predictor = joblib.load('data/linear.joblib')

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

class DuelingDQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        feat = self.feature(x)
        value = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, d):
        # 强约束：外部必须传 (1, 9)
        assert isinstance(s, np.ndarray) and isinstance(s_, np.ndarray)
        assert s.shape == (1, 9), f"s shape must be (1, 9), got {s.shape}"
        assert s_.shape == (1, 9), f"s_ shape must be (1, 9), got {s_.shape}"

        # buffer 内部仍然存 (1, 9)
        self.buffer.append((s, int(a), float(r), s_, bool(d)))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, s_, d = zip(*[self.buffer[i] for i in idx])

        # s: (batch, 1, 9) -> (batch, 9)
        s = np.concatenate(s, axis=0)
        s_ = np.concatenate(s_, axis=0)

        # 最终强校验
        assert s.shape == (batch_size, 9), f"s shape wrong: {s.shape}"
        assert s_.shape == (batch_size, 9), f"s_ shape wrong: {s_.shape}"

        return (
            s,
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            s_,
            np.asarray(d, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from dotenv import load_dotenv

if __name__ == '__main__':
    set_seed(42)
    load_dotenv()
    env = create_student_gym_env()
    obs_dim = env.observation_space.shape[0]
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

    num_episodes = 300
    batch_size = 64
    gamma = 0.999
    target_update = 20
    epsilon_start = 0.95
    epsilon_end = 0.05
    epsilon_decay = 200

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
            
            print(f"    Step {t}, Action: {action}, Epsilon: {epsilon:.3f}, Reward: {reward:.2f}, Done: {done}")
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
            'epsilon': epsilon
        }, f'checkpoint_ep{ep+1}.pth')
        logging.info(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
        print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
