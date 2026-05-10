# utils/metrics.py

"""
Metrics tracking for training evaluation
"""

import numpy as np
from collections import deque
from typing import Dict, List


class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_scores = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.values = deque(maxlen=window_size)
        
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def reset_episode(self):
        """Reset episode tracking"""
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def add_step(self, reward: float, done: bool):
        """Add a step to the current episode"""
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.reset_episode()
    
    def add_score(self, score: float):
        """Add episode score"""
        self.episode_scores.append(score)
    
    def add_loss(self, loss: float):
        """Add loss value"""
        self.losses.append(loss)
    
    def add_value(self, value: float):
        """Add value estimate"""
        self.values.append(value)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        metrics = {}
        
        if len(self.episode_rewards) > 0:
            metrics['mean_reward'] = np.mean(self.episode_rewards)
            metrics['max_reward'] = np.max(self.episode_rewards)
            metrics['min_reward'] = np.min(self.episode_rewards)
            metrics['std_reward'] = np.std(self.episode_rewards)
        
        if len(self.episode_lengths) > 0:
            metrics['mean_length'] = np.mean(self.episode_lengths)
        
        if len(self.episode_scores) > 0:
            metrics['mean_score'] = np.mean(self.episode_scores)
            metrics['max_score'] = np.max(self.episode_scores)
        
        if len(self.losses) > 0:
            metrics['mean_loss'] = np.mean(self.losses)
        
        if len(self.values) > 0:
            metrics['mean_value'] = np.mean(self.values)
        
        metrics['current_reward'] = self.current_episode_reward
        metrics['current_length'] = self.current_episode_length
        
        return metrics
    
    def clear(self):
        """Clear all tracked metrics"""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_scores.clear()
        self.losses.clear()
        self.values.clear()
        self.reset_episode()
