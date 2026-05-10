# utils/video_recorder.py

import numpy as np
import cv2
import os
from typing import Optional, Literal
from datetime import datetime
import jax


class VideoRecorder:
    """
    High-quality video recorder preserving original game resolution
    Supports standard (12ch) and diff training (24ch) inputs
    """
    
    def __init__(self, video_dir: str, fps: int = 30, 
                 quality: Literal['lossless', 'high', 'medium'] = 'high',
                 colorize: bool = True,
                 use_colormap: str = 'jet',
                 output_scale: float = 2.0):
        self.video_dir = video_dir
        self.fps = fps
        self.colorize = colorize
        self.use_colormap = use_colormap
        self.quality = quality
        self.output_scale = output_scale
        
        os.makedirs(video_dir, exist_ok=True)
        self.frames = []
        
        self._quality_settings = {
            'lossless': {
                'fourcc': cv2.VideoWriter_fourcc(*'FFV1'),
                'extension': '.avi',
                'suffix': '_lossless',
                'description': 'Lossless compression'
            },
            'high': {
                'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
                'extension': '.mp4',
                'suffix': '',
                'description': 'High quality H.264'
            },
            'medium': {
                'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
                'extension': '.mp4',
                'suffix': '_medium',
                'description': 'Medium quality'
            }
        }
    
    def _get_colormap(self) -> int:
        colormaps = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'ocean': cv2.COLORMAP_OCEAN,
            'plasma': cv2.COLORMAP_PLASMA,
        }
        return colormaps.get(self.use_colormap, cv2.COLORMAP_JET)
    
    def _colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert grayscale frame to color"""
        frame = frame.copy()

        if len(frame.shape) == 3:
            if frame.shape[-1] == 12:
                frame = frame[:, :, :3]
            elif frame.shape[-1] == 24:
                # Difference map training: Select the first 3 channels (RGB or CVD images)
                frame = frame[:, :, :3]
            elif frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.shape[-1] == 1:
                frame = frame[:, :, 0]
        
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        if len(frame.shape) == 3:
            frame = frame[:, :, 0]
        
        colored = cv2.applyColorMap(frame, self._get_colormap())
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored
    
    def _resize_multichannel(self, img, target_size):
        """
        Efficiently resize multi-channel images (supporting any number of channels)
        
        Args:
            img: Input image (H, W, C)
            target_size: Target size (width, height)
        
        Returns:
            resized: Output image (target_height, target_width, C)
        """
        h, w, c = img.shape
        target_w, target_h = target_size
        
        # Resize each channel separately
        resized_channels = [
            cv2.resize(img[:, :, i], (target_w, target_h), interpolation=cv2.INTER_AREA)
            for i in range(c)
        ]
        return np.stack(resized_channels, axis=-1)
    
    def _resize_multichannel_fast(self, img, target_size):
        """
        More efficient resizing of multi-channel images
        """
        h, w, c = img.shape
        target_w, target_h = target_size
        
        img_merged = img.reshape(h, w * c)
        img_resized = cv2.resize(img_merged, (target_w * c, target_h), interpolation=cv2.INTER_AREA)
        return img_resized.reshape(target_h, target_w, c)
    
    def _extract_rgb_from_obs(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the RGB images for display from the observations 

        Support multiple input formats:
        - (H, W, 3): Directly return
        - (H, W, 12): Standard 4-frame stacking, take the RGB of the middle frame
        - (H, W, 24): Difference image 4-frame stacking, take the RGB part of the middle frame
        """
        if len(frame.shape) == 2:
            # Grayscale image
            if self.colorize:
                return self._colorize_frame(frame)
            else:
                return np.stack([frame, frame, frame], axis=-1)
        
        channels = frame.shape[-1]
        
        # RGB image
        if channels == 3:
            return frame
        
        # Standard training: 4 frames × 3 channels = 12 channels
        if channels == 12:
            # Select the middle frame (the 2nd frame, with indices 3-6)
            rgb_frame = frame[:, :, 3:6]
            return rgb_frame
        
        # Difference map training: 4 frames × 6 channels = 24 channels
        # Each frame has 6 channels = 3 (RGB/CVD) + 3 (difference image)
        if channels == 24:
            # Extract the RGB/CVD part (the first three channels) of the middle frame
            # The starting index of the intermediate frame: (4 frames stacked, each frame 6 channels) -> In the middle is the 2nd frame, with indices 6 to 9
            rgb_frame = frame[:, :, 6:9]  # The RGB/CVD section of the second frame
            return rgb_frame
        
        # Other situation: Attempt to access the first three channels
        if channels > 3:
            return frame[:, :, :3]
        
        return frame
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame: convert to RGB 3-channel for display"""
        
        # Convert to uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Extract the RGB image
        rgb_frame = self._extract_rgb_from_obs(frame)
        
        # Make sure it is a 3-channel.
        if len(rgb_frame.shape) == 2:
            if self.colorize:
                rgb_frame = self._colorize_frame(rgb_frame)
            else:
                rgb_frame = np.stack([rgb_frame, rgb_frame, rgb_frame], axis=-1)
        elif len(rgb_frame.shape) == 3 and rgb_frame.shape[-1] == 1:
            if self.colorize:
                rgb_frame = self._colorize_frame(rgb_frame)
            else:
                rgb_frame = np.concatenate([rgb_frame, rgb_frame, rgb_frame], axis=-1)
        
        # Zoom output
        if self.output_scale != 1.0:
            h, w = rgb_frame.shape[:2]
            new_h, new_w = int(h * self.output_scale), int(w * self.output_scale)
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return rgb_frame
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame (supports 3, 12, or 24 channels)"""
        processed = self._process_frame(frame)
        self.frames.append(processed)
    
    def save(self, filename: str = None, quality_override: str = None) -> str:
        """Save video"""
        if not self.frames:
            print("No frames to save")
            return ""
        
        quality_mode = quality_override or self.quality
        settings = self._quality_settings[quality_mode]
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}{settings['suffix']}"
        
        ext = settings['extension']
        if not filename.endswith(ext):
            filename = filename.rsplit('.', 1)[0] + ext
        
        video_path = os.path.join(self.video_dir, filename)
        
        h, w = self.frames[0].shape[:2]
        
        out = cv2.VideoWriter(video_path, settings['fourcc'], self.fps, (w, h))
        
        if not out.isOpened():
            fallback = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fallback, self.fps, (w, h))
        
        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"\nVideo saved: {video_path}")
        print(f"   Resolution: {w} x {h}")
        print(f"   Frames: {len(self.frames)}")
        print(f"   Size: {size_mb:.2f} MB")
        
        return video_path
    
    def save_png_sequence(self, filename: str) -> str:
        """Save frames as PNG sequence"""
        seq_dir = os.path.join(self.video_dir, f"{filename}_png_sequence")
        os.makedirs(seq_dir, exist_ok=True)
        
        for i, frame in enumerate(self.frames):
            frame_path = os.path.join(seq_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        total_frames = len(self.frames)
        print(f"\nPNG sequence saved to: {seq_dir}/")
        print(f"   Total frames: {total_frames}")
        
        return seq_dir
    
    def clear(self):
        """Clear all frames"""
        self.frames = []
    
    def record_episode(self, env, agent, max_steps: int = 5000, 
                       deterministic: bool = True, verbose: bool = True,
                       use_random_actions: bool = False) -> dict:
        """Record an episode"""
        self.clear()
    
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        key = jax.random.PRNGKey(0)
    
        if verbose:
            print("Recording started...")
    
        while not done and episode_length < max_steps:
            self.add_frame(obs)
        
            if use_random_actions:
                action = env.action_space.sample()
            else:
                if deterministic:
                    # Resize observation for network input
                    if obs.shape[0] != 84 or obs.shape[1] != 84:
                        obs_resized = self._resize_multichannel(obs, (84, 84))
                        action = agent.get_action_deterministic(obs_resized)
                    else:
                        action = agent.get_action_deterministic(obs)
                else:
                    key, subkey = jax.random.split(key)
                    action, _, _ = agent.get_action_and_value(obs, subkey)
        
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
            if verbose and episode_length % 200 == 0:
                print(f"  Frame {episode_length}, reward: {episode_reward:.1f}")
            if done or truncated:
                self.add_frame(obs)
                break
    
        if verbose:
            print(f"Completed: {episode_length} frames, reward: {episode_reward:.1f}")
    
        return {'reward': episode_reward, 'length': episode_length, 'frames': len(self.frames)}
