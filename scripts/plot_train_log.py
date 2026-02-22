import matplotlib.pyplot as plt
import re

def parse_log(logfile):
    episodes = []
    rewards = []
    epsilons = []
    with open(logfile, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r"Episode (\d+), Reward: ([\-\d\.]+), Epsilon: ([\d\.]+)", line)
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
                epsilons.append(float(match.group(3)))
    return episodes, rewards, epsilons

if __name__ == '__main__':
    logfile = 'train_log.txt'
    episodes, rewards, epsilons = parse_log(logfile)
    plt.figure(figsize=(10,4))
    plt.plot(episodes, rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Dueling DQN with Predictor-Guided Exploration')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,4))
    plt.plot(episodes, epsilons, label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.legend()
    plt.show()
