# train.py

"""
Unified Training Script - Supports normal, deutan, and diff models
"""

import os
import sys
import warnings

if os.environ.get('JAX_PLATFORMS') is None:
    os.environ['JAX_PLATFORMS'] = 'cpu'

# Disable XLA automatic optimization
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
# Disable TensorFlow/JAX logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress Python warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import pickle
import cv2
from datetime import datetime

# Print GPU information
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvd-type', type=str, default='normal', choices=['normal', 'deutan'])
    parser.add_argument('--severity', type=float, default=1.0)
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--num-steps', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--frame-stack', type=int, default=4)
    parser.add_argument('--use-augmentation', action='store_true', default=False)
    parser.add_argument('--use-diff-training', action='store_true', default=False,
                       help='Use difference map training')
    parser.add_argument('--diff-training-mode', type=str, default='original',
                       choices=['original', 'cvd'],
                       help='Diff training mode: original (original+diff) or cvd (cvd+diff)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--record-video-interval', type=int, default=100_000,
                       help='Record video every N steps (0 to disable)')
    parser.add_argument('--video-length', type=int, default=2000,
                       help='Max frames per recorded video')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory (overrides auto-detection)')
    return parser.parse_args()


def get_output_dir(args):
    """Automatically obtain the output directory based on the model type"""
    if args.output_dir:
        return args.output_dir
    
    if args.use_diff_training:
        if args.diff_training_mode == 'original':
            return "results/diff_original_model"
        else:
            return "results/diff_cvd_model"
    else:
        if args.cvd_type == 'normal':
            return "results/normal_model"
        else:
            return "results/deutan_model"


def resize_multichannel(img, target_size):
    """Efficiently resize multi-channel images"""
    h, w, c = img.shape
    target_w, target_h = target_size
    resized_channels = [
        cv2.resize(img[:, :, i], (target_w, target_h), interpolation=cv2.INTER_AREA)
        for i in range(c)
    ]
    return np.stack(resized_channels, axis=-1)


def record_training_video(agent, output_path, cvd_type='normal', max_steps=2000, output_scale=2.0, 
                          use_diff_training=False, diff_training_mode='original'):
    """Record videos of the training process"""
    from utils.video_recorder import VideoRecorder
    from env import make_env as create_env
    
    if use_diff_training:
        record_env = create_env(
            cvd_type=cvd_type,
            severity=1.0 if cvd_type == 'deutan' else 0.0,
            train_mode=False,
            frame_stack=4,
            keep_original_resolution=True,
            use_diff_training=True,
            diff_training_mode=diff_training_mode
        )
    else:
        record_env = create_env(
            cvd_type=cvd_type,
            severity=1.0 if cvd_type == 'deutan' else 0.0,
            train_mode=False,
            frame_stack=4,
            keep_original_resolution=True
        )
    
    recorder = VideoRecorder(
        os.path.dirname(output_path), fps=30, quality='high',
        colorize=True, output_scale=output_scale
    )
    
    obs, _ = record_env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    
    print(f"    Recording training video to: {output_path}")
    print(f"    Obs shape: {obs.shape}")
    
    while not done and episode_length < max_steps:
        if use_diff_training:
            display_frame = obs[:, :, :3]
        else:
            display_frame = obs
        recorder.add_frame(display_frame)
        
        obs_resized = resize_multichannel(obs, (84, 84))
        if len(obs_resized.shape) == 3:
            obs_for_network = np.transpose(obs_resized, (2, 0, 1))
        else:
            obs_for_network = obs_resized
        
        action = agent.get_action_deterministic(obs_for_network)
        obs, reward, done, truncated, info = record_env.step(action)
        episode_reward += reward
        episode_length += 1
        
        if episode_length % 500 == 0:
            print(f"    Frame {episode_length}, reward: {episode_reward:.1f}")
        
        if done or truncated:
            if use_diff_training:
                display_frame = obs[:, :, :3]
            else:
                display_frame = obs
            recorder.add_frame(display_frame)
            break
    
    recorder.save(os.path.basename(output_path))
    record_env.close()
    
    print(f"    Video recorded: {episode_length} frames, reward={episode_reward:.2f}")
    return episode_reward


def train():
    args = parse_args()
    
    results_dir = get_output_dir(args)
    checkpoint_dir = f"{results_dir}/checkpoints"
    log_dir = f"{results_dir}/logs"
    video_dir = f"{results_dir}/videos"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Output directory: {results_dir}")
    
    if args.use_diff_training:
        if args.diff_training_mode == 'original':
            print("EXPERIMENT: Training with Original + Difference Map")
            print("   Training Input: Original RGB + Difference Map (24 channels)")
            print("   Test Input:     Original RGB or CVD + Difference Map (24 channels)")
        else:
            print("EXPERIMENT: Training with CVD + Difference Map")
            print("   Training Input: CVD Image + Difference Map (24 channels)")
            print("   Test Input:     CVD Image + Difference Map (24 channels)")
    elif args.cvd_type == 'normal':
        print("EXPERIMENT: Training on Original Images")
        print("   Training Input: Original RGB (12 channels)")
        print("   Test Input:     Original RGB or CVD (12 channels)")
    else:
        print("EXPERIMENT: Training on CVD Images")
        print("   Training Input: CVD-simulated RGB (12 channels)")
        print("   Test Input:     CVD (12 channels)")
    print("=" * 60)
    
    from env import make_env
    
    env = make_env(
        cvd_type=args.cvd_type if not args.use_diff_training else 'normal',
        severity=args.severity if args.cvd_type == 'deutan' else 0.0,
        train_mode=True,
        frame_skip=4,
        frame_stack=args.frame_stack,
        use_augmentation=args.use_augmentation and (args.cvd_type == 'deutan' or args.use_diff_training),
        use_diff_training=args.use_diff_training,
        diff_training_mode=args.diff_training_mode
    )
    
    print(f"  Actual input shape: {env.observation_space.shape}")
    print(f"  Environment channels: {env.observation_space.shape[-1]}")

    if args.use_diff_training:
        input_channels = args.frame_stack * 6
    else:
        input_channels = args.frame_stack * 3
    
    print(f"  Expected input channels: {input_channels}")
    
    from agents import PPOAgent
    
    agent = PPOAgent(
        action_dim=env.action_space.n,
        input_channels=input_channels,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    actual_channels = agent.network.conv1.weight.shape[1]
    print(f"  Network actual channels: {actual_channels}")
    assert actual_channels == input_channels, \
        f"Channel mismatch! Network has {actual_channels}, but expected {input_channels}"
    
    from utils.buffer import RolloutBuffer
    
    buffer = RolloutBuffer(args.num_steps)
    obs, _ = env.reset()
    
    episode_rewards = []
    episode_lengths = []
    current_reward = 0
    current_length = 0
    
    global_step = 0
    num_iterations = args.total_timesteps // args.num_steps
    key = jax.random.PRNGKey(args.seed)
    
    best_reward = -float('inf')
    last_video_step = 0
    
    log_file = open(f"{log_dir}/training_log.txt", 'w')
    log_file.write("step,avg_reward,avg_length,policy_loss,value_loss\n")
    
    for iteration in tqdm(range(num_iterations), desc="Training"):
        buffer.clear()
        
        for _ in range(args.num_steps):
            key, subkey = jax.random.split(key)
            action, log_prob, value = agent.get_action_and_value(obs, subkey)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            buffer.add(obs, action, reward, done, value, log_prob)
            
            current_reward += reward
            current_length += 1
            obs = next_obs
            global_step += 1
            
            if done:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                current_reward = 0
                current_length = 0
                obs, _ = env.reset()
        
        advantages, returns = buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        
        data = buffer.get()
        losses = agent.update(
            data['observations'],
            data['actions'],
            data['log_probs'],
            advantages,
            returns,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size
        )
        
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            
            if iteration % 50 == 0:
                print(f"\nStep {global_step}: Reward={avg_reward:.2f}, Length={avg_length:.1f}, Loss={losses['total_loss']:.4f}")
                log_file.write(f"{global_step},{avg_reward:.2f},{avg_length:.1f},{losses['policy_loss']:.4f},{losses['value_loss']:.4f}\n")
                log_file.flush()
            
            if iteration > 50:
                if best_reward == -float('inf') or avg_reward > best_reward:
                    best_reward = avg_reward
                    agent.save(f"{checkpoint_dir}/best.eqx")
                    print(f"  ✓ New best model! Reward: {best_reward:.2f}")
            
            if args.record_video_interval > 0:
                if global_step - last_video_step >= args.record_video_interval and global_step > 0:
                    last_video_step = global_step
                    video_path = os.path.join(video_dir, f"step_{global_step}_reward_{avg_reward:.0f}.mp4")
                    print(f"\n  Recording training video at step {global_step}...")
                    record_training_video(
                        agent, video_path, 
                        cvd_type=args.cvd_type, 
                        max_steps=args.video_length, 
                        output_scale=2.0,
                        use_diff_training=args.use_diff_training,
                        diff_training_mode=args.diff_training_mode
                    )
        
        if iteration % 500 == 0 and iteration > 0:
            agent.save(f"{checkpoint_dir}/checkpoint_{global_step}.eqx")
    
    agent.save(f"{checkpoint_dir}/final.eqx")
    
    final_video_path = os.path.join(video_dir, f"final_step_{global_step}.mp4")
    print(f"\n  Recording final video...")
    record_training_video(
        agent, final_video_path, 
        cvd_type=args.cvd_type, 
        max_steps=args.video_length * 2, 
        output_scale=2.0,
        use_diff_training=args.use_diff_training,
        diff_training_mode=args.diff_training_mode
    )
    
    log_file.close()
    env.close()
    
    print(f"\n✓ Training completed! Best reward: {best_reward:.2f}")
    print(f"  Model saved to: {checkpoint_dir}/best.eqx")
    print(f"  Videos saved to: {video_dir}/")


if __name__ == "__main__":
    train()
