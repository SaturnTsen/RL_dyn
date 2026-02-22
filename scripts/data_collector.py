import os
import sys
sys.path.append('./src')

import numpy as np
from dotenv import load_dotenv
from student_client import create_student_gym_env

load_dotenv()

all_observations = []
all_rewards = []
all_steps = []

env = create_student_gym_env()
env.reset()

def main(save_interval=10, save_path='raw_data', save_id='episode_data', num_episodes=50):
    print(f"\nStarting data collection for {num_episodes} episodes...")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        episode_observations = []
        episode_rewards = []
        episode_steps = []

        obs, info = env.reset()

        obs = obs.reshape(-1, 9)
        # n_obs = (obs - obs_mean) / np.sqrt(obs_var + 1e-8)
        episode_observations.append(obs)
        episode_rewards.append(0)
        episode_steps.append(info['step'])

        done = False

        while not done:

            obs, reward, terminated, truncated, info = env.step(0)

            obs = obs.reshape(-1, 9)
            # n_obs = (obs - obs_mean) / np.sqrt(obs_var + 1e-8)

            episode_observations.append(obs)
            episode_rewards.append(reward)
            episode_steps.append(info['step'])

            done = terminated or truncated

            print(f" Obs shape={obs.shape} Reward={reward:.2f}, Total Episode Reward={sum(episode_rewards):.2f}")

        all_observations.append(episode_observations)
        all_rewards.append(episode_rewards)
        all_steps.append(episode_steps)

        print(f"🏁 Episode {episode + 1} ended. Total Reward: {sum(episode_rewards):.2f}")

        if (episode + 1) % save_interval == 0:
            print(f"\n--- Saving data after {episode + 1} episodes ---")
            # save the collected data
            data_dir = save_path
            os.makedirs(data_dir, exist_ok=True)
            observations_path = os.path.join(data_dir, f'{save_id}_observations.npy')
            rewards_path = os.path.join(data_dir, f'{save_id}_rewards.npy')
            steps_path = os.path.join(data_dir, f'{save_id}_steps.npy')

            np.save(observations_path, np.array(all_observations, dtype=object))
            np.save(rewards_path, np.array(all_rewards, dtype=object))
            np.save(steps_path, np.array(all_steps, dtype=object))
            print(f"Data saved to {data_dir}/{save_id}_observations.npy, {data_dir}/{save_id}_rewards.npy, and {data_dir}/{save_id}_steps.npy")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect episodes of data from the environment.")
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes to collect')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval (in episodes) to save collected data')
    parser.add_argument('--save_path', type=str, default='raw_data', help='Directory to save collected data')
    parser.add_argument('--save_id', type=str, default='episode_data', help='Base name for saved data files')
    args = parser.parse_args()
    main(num_episodes=args.num_episodes, save_interval=args.save_interval, save_path=args.save_path, save_id=args.save_id)