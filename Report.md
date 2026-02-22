## Environment

The environment has a nine-dimensional continuous state space and three discrete actions:
(0) continue (immediate positive reward)
(1) repair (immediate negative reward, potential long-term benefit)
(2) sell (terminate episode, positive reward)

## Difficulties and what does not work

The difficulty of the challenge lies in  (1) vastly different numerical scales of the observation state, which led to unstable learning and poor convergence when fed directly into the network. (2) an extreme action asymmetry, i.e. 1 = immediate penalty; 2 = termination. This creates a strong bias toward short-sighted policies, and simply training a DQN yields reward ~2k per episode, and a PPO yields reward <3k per episode. This STRONGLY hinders the eps-greedy exploration, as uniformly random exploration rarely samples intermittent repair decisions (even though it is almost sure in the limit). More specifically, the agent simply learns to avoid any repair or sell actions, and thus almost never learns the long-term benefits of repair actions (unless we use a very high ε for an intolerably long training time).

## About the observation variables: A linear health predictor

To gain a useful notion of “lifetime,” I collected 500 episodes under a pure action 0 only policy, and trained a supervised health predictor to classify whether failure would occur within the next three steps. Surprisingly, a simple logistic regression achieved an AUC of ~0.95, indicating that near-term failure risk is highly predictable from the current observation. Using this predictor alone as a heuristic controller already improved performance to ~4.5k.

ROC-AUC: 0.9558
dim 0: -1.387    dim 1: -1.733   dim 2: 0.356    dim 3: -0.687
dim 4: -0.690    dim 5: -1.190   dim 6: 0.603    dim 7: 0.000 
dim 8: 0.643
Also, one practical issue we noticed is that the 7-th observation dimension is always zero across all collected data. I suspect it may be a server-side issue rather than a meaningful state variable.

## RL implementation: Efficiency improvement via predictor-guided exploration and dueling decomposition

To further improve efficiency while allowing the RL agent to autonomously discover a feasible policy, I integrated the predictor into exploration via a health-guided, non-uniform ε-greedy sampler (inspired by Metropolis–Hastings intuition). For value learning, I used a Dueling DQN with a shallow two-layer MLP + experience replay. The dueling decomposition serves as a variance-reduction mechanism (control variate) which stabilizes learning.

1. Exploitation: With probability 1−ϵ, the agent exploits by selecting argmax⁡_𝑎 𝑄(𝑠,𝑎). 
2. Exploration: it samples actions from a distribution conditioned by our health predictor:
        if p_fail < danger_threshold:
            probs = np.array([0.70, 0.25, 0.05])
        elif p_fail < 0.95:  # very_danger = 0.95
            probs = np.array([0.20, 0.75, 0.05])
        else:
            probs = np.array([0.05, 0.70, 0.25]) 

## Results

As training progressed, the Q-function began to learn the long-term value of repair actions. The log shows the emergence of a state-dependent ordering of Q-values, where states having high P(failure) predicted by the health predictor also had higher Q-values for repair actions compared to continue actions (sometimes even negative Q value predicted for continue). And when P(failure) is very high (>0.95), the Q-value for both continue and repair actions are lower than sell actions. We can conclude that the agent successfully learned to predict the long-term consequences of actions. However, the training curve is very noisy, and it takes a VERY LONG time to converge to a stable policy.