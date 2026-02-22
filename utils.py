import numpy as np
import os
import sys
import numpy as np
sys.path.append('./src')

def period_aggregate(obs, period=10, agg="mean"):
    """
    obs: (T, 9)
    返回: (1, 9)
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1, 9)
    if agg == "mean":
        return obs.mean(axis=0, keepdims=True)
    elif agg == "sum":
        return obs.sum(axis=0, keepdims=True)
    elif agg == "last":
        return obs[-1:].reshape(1, -1)
    else:
        raise ValueError(f"Unknown agg={agg}")
    
def compute_period_stats(episodes, period=10, agg="mean", obs_key="observations"):
    ps = []
    for ep in episodes:
        for obs in ep[obs_key]:
            P = period_aggregate(obs, period=period, agg=agg)
            if len(P) > 0:
                ps.append(P)
    X = np.concatenate(ps, axis=0)  # (sum_periods, 9)
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    return mean, var


obs_mean = np.array([7.9053516e+02, 1.9228922e+04, 3.3525293e+02, 1.1191241e+03, 3.7180111e-01, 1.3477822e+06, 3.9543977e+03, 0.0000000e+00, 9.5033712e+00])
obs_var = np.array([1.53039703e+01, 1.37877227e+04, 1.24639764e-01, 1.58602977e+00, 6.52246754e-08, 2.10554672e+08, 4.33407402e+00, 1.00000000e+00, 6.01958632e-02])
reward_scale = 1e-3

def normalize_obs(obs, period=10, agg="mean"):
    """
    obs: (T, 9)
    返回: (1, 9)
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1, 9)
    if agg == "mean":
        obs = (obs - obs_mean) / (np.sqrt(obs_var) + 1e-8)
        return obs.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown agg={agg}")

def process_reward(reward):
    return reward * reward_scale

def process_episode(episode):
    return {
        "steps": episode["steps"],
        "rewards": [process_reward(r) for r in episode["rewards"]],
        "health": list(range(len(episode["steps"]) - 1, -1, -1)),
        "observations": [normalize_obs(obs) for obs in episode["observations"]]
    }

def load_episodes():
    episodes = np.load("data/episodes.npz", allow_pickle=True)
    episodes = episodes["episodes"].tolist()
    return [process_episode(episode) for episode in episodes]