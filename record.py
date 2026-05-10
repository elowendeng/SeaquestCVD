# record.py

"""
Record videos - Supports normal, deutan, and diff models
Using the trained best model, with the original resolution of 210x160
"""

import os
import argparse
import cv2
import numpy as np
from env import make_env
from agents import PPOAgent
from utils.video_recorder import VideoRecorder


def parse_args():
    parser = argparse.ArgumentParser(description='Record final video with best model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (best.eqx)')
    parser.add_argument('--cvd-type', type=str, default='deutan',
                       choices=['normal', 'deutan', 'protan', 'tritan'],
                       help='CVD type for video recording')
    parser.add_argument('--severity', type=float, default=1.0,
                       help='CVD severity (0-1)')
    parser.add_argument('--output-dir', type=str, default='results/final_videos',
                       help='Output directory for videos')
    parser.add_argument('--output-scale', type=float, default=2.0,
                       help='Output scale for video (2.0 = 320x420 from 160x210)')
    parser.add_argument('--max-steps', type=int, default=3000,
                       help='Max frames per recorded video')
    parser.add_argument('--num-episodes', type=int, default=1,
                       help='Number of episodes to record')
    parser.add_argument('--colorize', action='store_true', default=True,
                       help='Colorize grayscale frames')
    parser.add_argument('--frame-stack', type=int, default=4,
                       help='Number of frames to stack')
    parser.add_argument('--use-diff-training', action='store_true', default=False,
                       help='Use diff training mode (matches checkpoint type)')
    parser.add_argument('--diff-training-mode', type=str, default='original',
                       choices=['original', 'cvd'],
                       help='Diff training mode used in training')
    return parser.parse_args()


def resize_multichannel(img, target_size):
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
    
    # Resize each channel separately (OpenCV only supports 1/3/4 channels)
    resized_channels = [
        cv2.resize(img[:, :, i], (target_w, target_h), interpolation=cv2.INTER_AREA)
        for i in range(c)
    ]
    # Stacked channels
    return np.stack(resized_channels, axis=-1)


def extract_rgb_for_display(obs, diff_training_mode=None):
    """
    Extract the RGB images for display from the observations
    
    Args:
    obs: Environmental observation
    diff_training_mode: 'original', 'cvd', or None (normal training)
    
    Returns:
    RGB images are used for display.
    """
    if diff_training_mode is None:
        # Normal training mode: Select the first 3 channels (single frame) directly
        if obs.shape[-1] >= 3:
            return obs[:, :, :3]
        return obs
    
    # Difference map training mode: obs shape = (H, W, 24) = 4 frames × 6 channels
    if diff_training_mode == 'original':
        # Original+Diff: The first three channels are RGB.
        return obs[:, :, :3]
    elif diff_training_mode == 'cvd':
        # CVD+Diff: The first three channels are CVD simulation images.
        return obs[:, :, :3]  # Display the CVD simulation diagram
    else:
        return obs[:, :, :3]


def record_video(args, agent, output_path):
    """Record a single video"""
    
    # Create the environment based on the model type
    if args.use_diff_training:
        # Difference graph model: During training, use the diff wrapper; the same applies when recording.
        print(f"  Using diff training mode: {args.diff_training_mode}")
        env = make_env(
            cvd_type=args.cvd_type,
            severity=args.severity,
            train_mode=False,
            frame_stack=args.frame_stack,
            keep_original_resolution=True,
            use_diff_training=True,
            diff_training_mode=args.diff_training_mode
        )
    else:
        # Ordinary model
        env = make_env(
            cvd_type=args.cvd_type,
            severity=args.severity,
            train_mode=False,
            frame_stack=args.frame_stack,
            keep_original_resolution=True,
            use_diff_training=False
        )
    
    recorder = VideoRecorder(
        args.output_dir, fps=30, quality='high',
        colorize=args.colorize, output_scale=args.output_scale
    )
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_score = 0
    episode_length = 0
    done = False
    
    print(f"  Recording: {output_path}")
    print(f"  Resolution: 210x160 → {int(160*args.output_scale)}x{int(210*args.output_scale)}")
    print(f"  Obs shape: {obs.shape}")  # (210, 160, 12)

    while not done and episode_length < args.max_steps:
        # Extract the RGB image for display
        display_frame = extract_rgb_for_display(obs, args.diff_training_mode if args.use_diff_training else None)
        recorder.add_frame(display_frame)
        
        # Zooming observation is used for network input.
        obs_resized = resize_multichannel(obs, (84, 84))
        # obs_resized shape: (84, 84, C) where C=12 or 24

        if len(obs_resized.shape) == 3:
            obs_for_network = np.transpose(obs_resized, (2, 0, 1))  # HWC -> CHW
        else:
            obs_for_network = obs_resized
        
        action = agent.get_action_deterministic(obs_for_network)
        obs, reward, done, truncated, info = env.step(action)

        episode_reward += reward
        episode_length += 1
        
        if episode_length % 500 == 0:
            print(f"    Frame {episode_length}, reward: {episode_reward:.1f}")
        
        if done or truncated:
            display_frame = extract_rgb_for_display(obs, args.diff_training_mode if args.use_diff_training else None)
            recorder.add_frame(display_frame)
            break
    
    video_path = recorder.save(os.path.basename(output_path))
    env.close()
    
    print(f"  ✓ Saved: {video_path}")
    print(f"    Frames: {episode_length}, Score: {episode_score:.0f}, Reward: {episode_reward:.2f}")
    
    return episode_score


def auto_detect_model_type(checkpoint_path):
    """
    Automatically detect the model type from the checkpoint path
    """
    if 'diff_model' in checkpoint_path:
        return True, 'original'  # Default to the original mode
    elif 'deutan_model' in checkpoint_path:
        return False, None
    elif 'normal_model' in checkpoint_path:
        return False, None
    else:
        # Try to determine from the file name
        if 'diff' in checkpoint_path.lower():
            return True, 'original'
        return False, None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Automatically detect the model type
    if not args.use_diff_training:
        auto_diff, auto_mode = auto_detect_model_type(args.checkpoint)
        if auto_diff:
            args.use_diff_training = True
            args.diff_training_mode = auto_mode
            print(f"  Auto-detected: diff model with mode={args.diff_training_mode}")
    
    print("=" * 60)
    print("Recording Final Videos with Best Model")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {'Diff (' + args.diff_training_mode + ')' if args.use_diff_training else 'Standard'}")
    print(f"CVD type: {args.cvd_type}, Severity: {args.severity}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)
    
    # Try to read the correct input_channels from the checkpoint.
    import pickle
    try:
        with open(args.checkpoint, 'rb') as f:
            checkpoint_data = pickle.load(f)
        loaded_network = checkpoint_data['network']
        input_channels = loaded_network.conv1.weight.shape[1]
        print(f"  Detected input channels from checkpoint: {input_channels}")
    except Exception as e:
        if args.use_diff_training:
            input_channels = args.frame_stack * 6
        else:
            input_channels = args.frame_stack * 3
        print(f"  Using calculated input channels: {input_channels}")
    
    # When creating an agent, the number of channels in the checkpoint must be used.
    from agents import PPOAgent
    agent = PPOAgent(
        action_dim=18, 
        input_channels=input_channels,
        seed=42
    )
    
    # load checkpoint
    try:
        agent.load(args.checkpoint)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        raise
    
    # record video
    for episode in range(args.num_episodes):
        model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        suffix = f"_diff_{args.diff_training_mode}" if args.use_diff_training else ""
        output_path = os.path.join(
            args.output_dir, 
            f"{args.cvd_type}_{model_name}{suffix}_ep{episode+1}.mp4"
        )
        score = record_video(args, agent, output_path)
    
    print("\n" + "=" * 60)
    print("✓ All final videos recorded!")
    print(f"  Videos saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
