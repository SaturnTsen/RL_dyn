import numpy as np

from pathlib import Path

# This script checks the collected data for missing files. It looks for files with the pattern

root = Path("raw_data")

pairs = []
missing = []

for obs in root.glob("*_observations.npy"):
    prefix = obs.name.removesuffix("_observations.npy")
    rew = root / f"{prefix}_rewards.npy"
    steps = root / f"{prefix}_steps.npy"

    if rew.exists() and steps.exists():
        pairs.append((obs, rew, steps))
    else:
        missing.append({
            "id": prefix,
            "observations": obs.exists(),
            "rewards": rew.exists(),
            "steps": steps.exists(),
        })

print(f"found pairs: {len(pairs)}")
print(f"missing: {len(missing)}")

all_obs = []
all_rew = []
all_steps = []

for obs_file, rew_file, steps_file in pairs:
    obs = np.load(obs_file, allow_pickle=True)
    rew = np.load(rew_file, allow_pickle=True)
    steps = np.load(steps_file, allow_pickle=True)
    assert len(obs) == len(rew) == len(steps), f"Length mismatch in {obs_file}"
    all_obs.extend(obs)
    all_rew.extend(rew)
    all_steps.extend(steps)

# Save the combined data into a single .npz file
save_path = "data/episodes.npz"
episodes = []
for obs, rew, steps in zip(all_obs, all_rew, all_steps):
    if abs(rew[-1] + 1000.0) < 1e-3: # clean polluted data
        episodes.append({
            "observations": obs,
            "rewards": rew,
            "steps": steps,
        })
    else:
        print(f"Skipping episode with final reward {rew[-1]}")

np.savez(save_path, episodes=episodes)