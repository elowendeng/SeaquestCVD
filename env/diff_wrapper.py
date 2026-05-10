# env/diff_wrapper.py

"""
Difference Map Wrapper
During training, provide the original image along with the difference image, allowing the model to learn how to compensate for color information.
"""

import gymnasium as gym
import numpy as np
import cv2


class DiffMapTrainingWrapper(gym.ObservationWrapper):
    """
    Train using the original image + the difference image
    
    Principle:
    - Input: Original image (with full color)
    - Output: Original image + Difference image (concatenated together)
    - Difference image = |Original image - Color blindness simulation image|
    
    This model can learn "which color information will be lost when someone is color-blind"
    """
    
    def __init__(self, env, cvd_type='deutan', severity=1.0, use_original=True):
        super().__init__(env)
        self.cvd_type = cvd_type
        self.severity = severity
        self.use_original = use_original
        
        from env.cvd_simulation import simulate_cvd
        self.simulate_cvd = simulate_cvd
        
        # Update the observation space
        old_shape = env.observation_space.shape
        if len(old_shape) == 3:
            h, w, c = old_shape
            # Original image (c) + Difference image (c) = 2c channel
            self.observation_space = gym.spaces.Box(
                low=0, high=1,
                shape=(h, w, c * 2),
                dtype=np.float32
            )
    
    def observation(self, obs):
        """
        Input: Original image (already resized and normalized)
        Output: Original image + Difference image
        """
        # Ensure it is of type float32 and normalized
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0
        
        # Extract the RGB section
        if obs.shape[-1] >= 3:
            rgb = obs[:, :, :3]
        else:
            rgb = obs
        
        # Generate color blindness chart
        cvd_input = (rgb * 255).astype(np.uint8)
        cvd_img = self.simulate_cvd(cvd_input, self.cvd_type, self.severity)
        cvd_img = cvd_img.astype(np.float32) / 255.0
        
        # Calculate the difference map
        diff_map = np.abs(rgb - cvd_img)
        
        # Difference map enhancement (optional: enlarge the differences to make it easier for the model to learn)
        # diff_map = np.clip(diff_map * 2, 0, 1)
        
        # Merge the original image and the difference image
        if self.use_original:
            combined = np.concatenate([rgb, diff_map], axis=-1)
        else:
            combined = np.concatenate([cvd_img, diff_map], axis=-1)
        
        # Handling the situation of stacked frames
        num_channels = obs.shape[-1]
        if num_channels > 3:
            # For multi-frame stacking, only modify the first group of RGB values, and use the same difference image for the other frames.
            result = obs.copy()
            result[:, :, :3] = rgb
            result[:, :, 3:6] = diff_map
            return result
        else:
            return combined


class SimpleDiffWrapper(gym.ObservationWrapper):
    """
    Simplified version: Provide the difference graph as additional information
    Output: CVD graph + Difference graph (6-channel single frame)
    """
    
    def __init__(self, env, cvd_type='deutan', severity=1.0):
        super().__init__(env)
        self.cvd_type = cvd_type
        self.severity = severity
        
        from env.cvd_simulation import simulate_cvd
        self.simulate_cvd = simulate_cvd
        
        old_shape = env.observation_space.shape
        if len(old_shape) == 3:
            h, w, c = old_shape
            # Output 6 channels: 3 (CVD) + 3 (difference map)
            self.observation_space = gym.spaces.Box(
                low=0, high=1,
                shape=(h, w, 6),  # Single frame with 6 channels
                dtype=np.float32
            )
    
    def observation(self, obs):
        # Ensure it is of type float32 and normalized.
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0
        
        # Extract RGB (if multi-frame stacking, take the first 3 channels)
        if obs.shape[-1] >= 3:
            rgb = obs[:, :, :3]
        else:
            rgb = obs
        
        # Generate color blindness chart
        cvd_input = (rgb * 255).astype(np.uint8)
        cvd_img = self.simulate_cvd(cvd_input, self.cvd_type, self.severity)
        cvd_img = cvd_img.astype(np.float32) / 255.0
        
        # Calculate the difference map
        diff_map = np.abs(rgb - cvd_img)
        
        # Return CVD graph + difference graph (6 channels)
        result = np.concatenate([cvd_img, diff_map], axis=-1)
        
        # If there are multiple frames stacked in the input, processing is required.
        if obs.shape[-1] > 3:
            # For the stacked frames, only modify the first group, and copy the other frames.
            result_full = obs.copy()
            result_full[:, :, :6] = result
            # The subsequent frames remain unchanged.
            for i in range(6, obs.shape[-1], 3):
                result_full[:, :, i:i+3] = cvd_img
                if i + 3 < obs.shape[-1]:
                    result_full[:, :, i+3:i+6] = diff_map
            return result_full
        
        return result
