# utils/buffer.py

"""
Rollout Buffer with GAE computation
"""

import numpy as np
from typing import Tuple


class RolloutBuffer:
    """Store rollout data and calculate GAE"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
    
    def add(self, obs, action, reward, done, value, log_prob):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def get(self):
        return {
            'observations': np.array(self.observations, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.bool_),
            'values': np.array(self.values, dtype=np.float32),
            'log_probs': np.array(self.log_probs, dtype=np.float32)
        }
    
    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the GAE advantage function and returns
        
        Returns:
            advantages: The value of the objective function
            returns: Discount rewards
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_not_done = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_value * next_not_done - values[t]
            else:
                next_not_done = 1.0 - dones[t]
                delta = rewards[t] + gamma * values[t + 1] * next_not_done - values[t]
            
            gae = delta + gamma * gae_lambda * next_not_done * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def __len__(self):
        return len(self.observations)
