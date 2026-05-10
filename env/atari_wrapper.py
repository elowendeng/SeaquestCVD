# env/atari_wrapper.py

"""
Atari environment wrappers for Seaquest with CVD simulation
"""

import gymnasium as gym
import numpy as np
import cv2
from gymnasium import ObservationWrapper, Wrapper
from collections import deque
from typing import Optional, Literal, Dict

# Register the ALE environment
try:
    import ale_py
    gym.register_envs(ale_py)
    print(":) Registered ALE environments")
except ImportError:
    print("! Warning: ale_py not installed")
except Exception as e:
    print(f"! Warning: Failed to register ALE: {e}")

from env.cvd_simulation import simulate_cvd, CVDType
from env.augmentation_wrapper import AugmentationWrapper

CVDType = Literal['normal', 'protan', 'deutan', 'tritan']


class FrameSkip(Wrapper):
    """Frame skipping acceleration"""
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ResizeAndNormalize(ObservationWrapper):
    """Resize and normalize (while preserving colors)"""
    
    def __init__(self, env, size: int = 84, normalize: bool = True):
        super().__init__(env)
        self.size = size
        self.normalize = normalize
        
        # Output space: (size, size, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=1 if normalize else 255,
            shape=(size, size, 3),
            dtype=np.float32 if normalize else np.uint8
        )
    
    def observation(self, obs):
        # obs: (210, 160, 3) - Original Atari resolution
        obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.normalize:
            obs = obs.astype(np.float32) / 255.0
        return obs


class KeepOriginalResolution(ObservationWrapper):
    """Maintain the original resolution and only normalize"""
    
    def __init__(self, env, normalize: bool = True):
        super().__init__(env)
        self.normalize = normalize
        
        # original resolution: 210(height) x 160(width)
        self.observation_space = gym.spaces.Box(
            low=0, high=1 if normalize else 255,
            shape=(210, 160, 3),
            dtype=np.float32 if normalize else np.uint8
        )
    
    def observation(self, obs):
        # obs: (210, 160, 3) Original Atari resolution
        if self.normalize:
            obs = obs.astype(np.float32) / 255.0
        return obs


class FrameStack(ObservationWrapper):
    """Stacked frames - along the channel dimension"""
    
    def __init__(self, env, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        
        # update observation space: (H, W, C * n_stack)
        old_shape = env.observation_space.shape
        new_shape = (old_shape[0], old_shape[1], old_shape[2] * n_stack)

        old_low = env.observation_space.low
        old_high = env.observation_space.high
        
        # If it is a scalar, simply repeat it
        # If it is an array, repeat it along the last dimension.
        if np.isscalar(old_low):
            new_low = old_low
            new_high = old_high
        else:
            # Repeat n_stack times along the last dimension
            new_low = np.repeat(old_low, n_stack, axis=-1)
            new_high = np.repeat(old_high, n_stack, axis=-1)
        
        self.observation_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        self.frames.append(obs)
        while len(self.frames) < self.n_stack:
            self.frames.append(obs)

        stacked = np.concatenate(list(self.frames), axis=-1)
        return stacked.astype(self.observation_space.dtype)
        # return np.concatenate(list(self.frames), axis=-1)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self.observation(obs), info


class CVDWrapper(ObservationWrapper):
    """Color-blind simulation wrapper"""
    
    def __init__(self, env, cvd_type: CVDType, severity: float = 1.0):
        super().__init__(env)
        self.cvd_type = cvd_type
        self.severity = severity
    
    def observation(self, obs):
        if self.cvd_type == 'normal' or self.severity == 0:
            return obs

        # Check if it is normalized
        is_normalized = obs.max() <= 1.0
        
        if is_normalized:
            img = (obs * 255).astype(np.uint8)
        else:
            img = obs.astype(np.uint8)

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3 and img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        
        # Apply CVD simulation
        result = simulate_cvd(img, self.cvd_type, self.severity)
        if is_normalized:
            result = result.astype(np.float32) / 255.0   
        return result


def make_env(
    cvd_type: CVDType = 'normal',
    severity: float = 1.0,
    train_mode: bool = True,
    frame_skip: int = 4,
    frame_stack: int = 4,
    use_augmentation: bool = False,
    use_diff_training: bool = False,
    diff_training_mode: str = 'original',
    render_mode: Optional[str] = None,
    max_episode_steps: int = 2000,
    keep_original_resolution: bool = False
):
    """
    Unified environment creation function
    
    Args:
        cvd_type: Type of color blindness
        severity: Severity level: 0 - 1
        train_mode: True = Training mode (84x84), False = Evaluation mode
        frame_skip: Frame skipping count
        frame_stack: Stack frame count
        use_augmentation: Whether to use data augmentation
        use_diff_training: Whether to use the difference map for training
        diff_training_mode: 'original' = Original image + Difference image, 'cvd' = Color-blind map + Difference map
        render_mode: Rendering mode
        max_episode_steps: Maximum number of steps
        keep_original_resolution: Whether maintain the original resolution of 210x160 (effective when in evaluation mode)
    """
    # Try multiple environment names
    env_names = [
        "ALE/Seaquest-v5",           # New version ale-py
        "SeaquestNoFrameskip-v4",    # Classic version
        "Seaquest-v4",               # Standard version
        "SeaquestNoFrameskip-v0",    # Old version
        "Seaquest-v0"                # The oldest version
    ]
    
    env = None
    used_name = None
    
    for name in env_names:
        try:
            env = gym.make(name, render_mode=render_mode)
            used_name = name
            print(f"  ✓ Using environment: {name}")
            break
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
            continue
    
    if env is None:
        raise RuntimeError(
            "Could not load Seaquest environment.\n"
            f"Tried: {env_names}\n"
            "Please ensure ROMs are installed."
        )
    
    # skip frame
    env = FrameSkip(env, skip=frame_skip)
    
    if train_mode:
        # training mode: 84x84
        env = ResizeAndNormalize(env, size=84, normalize=True)
    else:
        # Evaluation/Recording Mode
        if keep_original_resolution:
            # Maintain the original resolution: 210x160
            env = KeepOriginalResolution(env, normalize=True)
        else:
            # Default usage: 160x160
            env = ResizeAndNormalize(env, size=160, normalize=True)
    
    # Difference map training
    if use_diff_training:
        from env.diff_wrapper import DiffMapTrainingWrapper, SimpleDiffWrapper
        if diff_training_mode == 'original':
            # Original+Diff mode
            env = DiffMapTrainingWrapper(env, cvd_type, severity, use_original=True)
        elif diff_training_mode == 'cvd':
            # CVD+Diff mode
            env = SimpleDiffWrapper(env, cvd_type, severity)
        else:
            # Default usage Original+Diff
            env = DiffMapTrainingWrapper(env, cvd_type, severity, use_original=True)
    # Standard color blindness simulation
    elif cvd_type != 'normal':
        env = CVDWrapper(env, cvd_type, severity)
    
    # Data augmentation (for training only)
    if use_augmentation and train_mode:
        env = AugmentationWrapper(env, color_jitter=0.2, random_shift=4)
    
    # Frame stacking
    env = FrameStack(env, n_stack=frame_stack)
    # Time limit
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
