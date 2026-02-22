import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
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
