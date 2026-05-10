# env/augmentation_wrapper.py

"""
Data augmentation wrapper for training under color blindness
"""

import gymnasium as gym
import numpy as np


class AugmentationWrapper(gym.ObservationWrapper):
    """
    Apply data augmentations to observations
    Helps improve generalization under color blindness
    """
    
    def __init__(self, env, color_jitter=0.2, random_shift=4, random_noise=0.02):
        super().__init__(env)
        self.color_jitter = color_jitter
        self.random_shift = random_shift
        self.random_noise = random_noise
        self._augment_prob = 0.5
    
    def observation(self, obs):
        """Apply random augmentations during training"""
        if np.random.random() > self._augment_prob:
            return obs
    
        # Color jitter for RGB images
        if self.color_jitter > 0 and len(obs.shape) == 3:
            if obs.shape[-1] == 3:  # RGB image
                # Add independent brightness jitter to each channel
                for c in range(3):
                    factor = np.random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
                    obs[:, :, c] = obs[:, :, c] * factor
            elif obs.shape[-1] == 1:  # Grayscale
                factor = np.random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
                obs = obs * factor
    
        # Random shift
        if self.random_shift > 0:
            h_shift = np.random.randint(-self.random_shift, self.random_shift + 1)
            w_shift = np.random.randint(-self.random_shift, self.random_shift + 1)
            obs = self._random_shift(obs, h_shift, w_shift)
    
        # Random noise
        if self.random_noise > 0:
            noise = np.random.normal(0, self.random_noise, obs.shape)
            obs = obs + noise
    
        return np.clip(obs, 0, 1)
    
    def _random_shift(self, obs, h_shift, w_shift):
        """Shift the image by given amounts"""
        if len(obs.shape) == 3:
            if h_shift > 0:
                obs = np.roll(obs, h_shift, axis=0)
                obs[:h_shift] = 0
            elif h_shift < 0:
                obs = np.roll(obs, h_shift, axis=0)
                obs[h_shift:] = 0
            
            if w_shift > 0:
                obs = np.roll(obs, w_shift, axis=1)
                obs[:, :w_shift] = 0
            elif w_shift < 0:
                obs = np.roll(obs, w_shift, axis=1)
                obs[:, w_shift:] = 0
        return obs
