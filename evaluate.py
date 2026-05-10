# evaluate.py

"""
Evaluation script - Supports standard (12ch) and diff (24ch) models
"""

import os
import sys
import warnings
import csv
import pickle
from datetime import datetime

os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import cv2

from env import make_env
from agents import PPOAgent
from env.cvd_simulation import simulate_cvd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--cvd-type', type=str, default='deutan', 
                       choices=['normal', 'deutan', 'protan', 'tritan'],
                       help='CVD type for evaluation')
    parser.add_argument('--severity', type=float, default=1.0,
                       help='CVD severity (0-1)')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--frame-stack', type=int, default=4,
                       help='Number of frames to stack')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom name for this evaluation run')
    parser.add_argument('--use-diff-training', action='store_true', default=False,
                       help='Use diff training mode (for diff model evaluation)')
    parser.add_argument('--diff-training-mode', type=str, default='original',
                       choices=['original', 'cvd'],
                       help='Diff training mode used in training (original: [I,diff], cvd: [CVD,diff])')
    parser.add_argument('--test-on-original', action='store_true', default=False,
                       help='For diff_original model only: if True, test on [I, diff] (Exp 3a); '
                            'if False, test on [CVD, diff] (Exp 3b)')
    return parser.parse_args()


def get_model_output_dir(checkpoint_path):
    """Automatically obtain the model output directory"""
    checkpoints_dir = os.path.dirname(checkpoint_path)
    model_dir = os.path.dirname(checkpoints_dir)
    eval_dir = os.path.join(model_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def detect_model_type(checkpoint_path):
    """Detect the model type (standard 12-channel or difference 24-channel)"""
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        channels = data['network'].conv1.weight.shape[1]
        if channels == 24:
            return True, 'original'
        else:
            return False, None
    except:
        return False, None


def resize_multichannel(img, target_size):
    """Resize multi-channel image to target size (width, height)"""
    h, w, c = img.shape
    target_w, target_h = target_size
    resized_channels = [
        cv2.resize(img[:, :, i], (target_w, target_h), interpolation=cv2.INTER_AREA)
        for i in range(c)
    ]
    return np.stack(resized_channels, axis=-1)


def extract_rgb_and_diff(obs, cvd_type, severity, frame_stack=4):
    """
    Extract original RGB and compute diff map from environment observation.
    
    Args:
        obs: Environment output (12ch or 24ch)
        cvd_type: Type of CVD to simulate
        severity: CVD severity
        frame_stack: Number of stacked frames
    
    Returns:
        original_rgb: Original RGB image (H, W, 3)
        diff_map: Difference map |I - CVD(I)|
        cvd_rgb: CVD simulated image
    """
    h, w, c = obs.shape
    channels_per_frame = c // frame_stack
    
    # Extract original RGB from first frame
    if channels_per_frame == 3:
        original_rgb = obs[:, :, :3]
    elif channels_per_frame == 6:
        original_rgb = obs[:, :, :3]
    else:
        original_rgb = obs[:, :, :3]
    
    # Convert to uint8 for CVD simulation
    if original_rgb.dtype != np.uint8:
        if original_rgb.max() <= 1.0:
            original_rgb_uint8 = (original_rgb * 255).astype(np.uint8)
        else:
            original_rgb_uint8 = original_rgb.astype(np.uint8)
    else:
        original_rgb_uint8 = original_rgb.copy()
    
    # Apply CVD simulation
    cvd_rgb_uint8 = simulate_cvd(original_rgb_uint8, cvd_type, severity)
    cvd_rgb = cvd_rgb_uint8.astype(np.float32) / 255.0
    
    # Calculate diff map
    diff_map = np.abs(original_rgb - cvd_rgb)
    
    return original_rgb, diff_map, cvd_rgb


def process_obs_for_original_test(obs, cvd_type, severity, frame_stack=4):
    """
    Exp 3a: Test diff_original model on ORIGINAL vision.
    Training:  [I, |I-CVD(I)|]
    Test:      [I, |I-CVD(I)|]  (SAME as training!)
    """
    original_rgb, diff_map, _ = extract_rgb_and_diff(obs, cvd_type, severity, frame_stack)
    
    test_single_frame = np.concatenate([original_rgb, diff_map], axis=-1)
    test_obs = np.concatenate([test_single_frame] * frame_stack, axis=-1)
    
    return test_obs.astype(np.float32)


def process_obs_for_cvd_test(obs, cvd_type, severity, frame_stack=4):
    """
    Exp 3b & 3c: Test diff model on CVD vision.
    Training (3b): [I, diff] -> Test: [CVD, diff]
    Training (3c): [CVD, diff] -> Test: [CVD, diff]
    """
    _, diff_map, cvd_rgb = extract_rgb_and_diff(obs, cvd_type, severity, frame_stack)
    
    test_single_frame = np.concatenate([cvd_rgb, diff_map], axis=-1)
    test_obs = np.concatenate([test_single_frame] * frame_stack, axis=-1)
    
    return test_obs.astype(np.float32)


def evaluate():
    args = parse_args()
    
    # Auto-detect model type
    is_diff_model, detected_mode = detect_model_type(args.checkpoint)
    
    if not args.use_diff_training and is_diff_model:
        args.use_diff_training = True
        args.diff_training_mode = detected_mode
        print(f"  Auto-detected: diff model with {args.diff_training_mode} mode")
    elif args.use_diff_training and not is_diff_model:
        print(f"  Warning: Specified diff model but checkpoint appears to be standard")
    
    output_dir = get_model_output_dir(args.checkpoint)
    
    if args.run_name is None:
        run_name = f"{args.cvd_type}_sev{args.severity}"
    else:
        run_name = args.run_name
    
    print("=" * 60)
    print(f"Evaluating model: {args.checkpoint}")
    print(f"Model type: {'Diff (' + args.diff_training_mode + ')' if args.use_diff_training else 'Standard'}")
    print(f"CVD type: {args.cvd_type}, Severity: {args.severity}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    if args.use_diff_training:
        if args.diff_training_mode == 'original':
            if args.test_on_original:
                print(f"  Exp 3a: Testing diff_original model on ORIGINAL vision")
                print(f"  Test input: [I, diff] - Same as training! (Ideal condition)")
                use_original_test = True
            else:
                print(f"  Exp 3b: Testing diff_original model on CVD vision")
                print(f"  Test input: [CVD(I), diff] - Different from training (Generalization)")
                use_original_test = False
        else:  # cvd mode
            print(f"  Exp 3c: Testing diff_cvd model on CVD vision")
            print(f"  Test input: [CVD(I), diff] - Same as training!")
            use_original_test = False
        
        # Create environment that outputs original RGB (12ch)
        env = make_env(
            cvd_type='normal',
            severity=0.0,
            train_mode=True,
            frame_stack=args.frame_stack,
            use_diff_training=False
        )
        input_channels = args.frame_stack * 6
        cvd_type_for_processing = args.cvd_type
        severity_for_processing = args.severity
        
        print(f"  Environment outputs: {env.observation_space.shape} (12ch, original RGB)")
        
        if args.diff_training_mode == 'original' and args.test_on_original:
            print(f"  Processing: [I, diff] (Exp 3a)")
        else:
            print(f"  Processing: [CVD(I), diff] (Exp 3b/3c)")
    
    else:
        # Standard model evaluation (12ch)
        if args.cvd_type == 'normal' and args.severity == 0.0:
            print(f"  Exp 1a: Testing normal model on ORIGINAL vision")
        elif args.cvd_type != 'normal':
            print(f"  Exp 1b/2: Testing model on CVD vision")
        else:
            print(f"  Testing standard model")
        
        env = make_env(
            cvd_type=args.cvd_type,
            severity=args.severity,
            train_mode=True,
            frame_stack=args.frame_stack,
            use_diff_training=False
        )
        input_channels = args.frame_stack * 3
    
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Input channels: {input_channels}")
    
    # Create Agent
    agent = PPOAgent(
        action_dim=env.action_space.n,
        input_channels=input_channels,
        seed=42
    )
    agent.load(args.checkpoint)
    print("  Model loaded successfully")
    
    rewards = []
    scores = []
    lengths = []
    
    print(f"\n  Running {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_score = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 5000:
            
            if args.use_diff_training:
                if args.diff_training_mode == 'original' and args.test_on_original:
                    # Exp 3a: Test on [I, diff]
                    obs_for_network = process_obs_for_original_test(
                        obs, cvd_type_for_processing, severity_for_processing,
                        frame_stack=args.frame_stack
                    )
                else:
                    # Exp 3b or 3c: Test on [CVD, diff]
                    obs_for_network = process_obs_for_cvd_test(
                        obs, cvd_type_for_processing, severity_for_processing,
                        frame_stack=args.frame_stack
                    )
                
                # Resize to 84x84 if needed
                if obs_for_network.shape[0] != 84 or obs_for_network.shape[1] != 84:
                    obs_for_network = resize_multichannel(obs_for_network, (84, 84))
            else:
                # Standard model: use environment output directly
                obs_for_network = obs
                if obs_for_network.shape[0] != 84 or obs_for_network.shape[1] != 84:
                    obs_for_network = resize_multichannel(obs_for_network, (84, 84))
            
            # Ensure CHW format for network
            if len(obs_for_network.shape) == 3:
                obs_for_network = np.transpose(obs_for_network, (2, 0, 1))
            
            action = agent.get_action_deterministic(obs_for_network)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_score += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        rewards.append(episode_reward)
        scores.append(episode_score)
        lengths.append(episode_length)
        print(f"  Episode {episode+1:3d}: Score={episode_score:6.0f}, Reward={episode_reward:7.2f}, Length={episode_length:4d}")
    
    results = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'scores': scores,
        'rewards': rewards,
        'lengths': lengths
    }
    
    print("\n" + "=" * 60)
    print(f"Results over {args.num_episodes} episodes:")
    print(f"  Mean Score: {results['mean_score']:.0f} ± {results['std_score']:.0f}")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print("=" * 60)
    
    # Determine model type string for saving
    if args.use_diff_training:
        if args.diff_training_mode == 'original':
            if args.test_on_original:
                model_type = 'diff_original_exp3a'
            else:
                model_type = 'diff_original_exp3b'
        else:
            model_type = 'diff_cvd_exp3c'
    else:
        if args.cvd_type == 'normal' and args.severity == 0.0:
            model_type = 'normal_exp1a'
        elif args.cvd_type != 'normal':
            if 'deutan_model' in args.checkpoint:
                model_type = 'deutan_exp2'
            else:
                model_type = 'normal_exp1b'
        else:
            model_type = 'standard'
    
    save_results(args, results, output_dir, run_name, model_type)
    
    env.close()
    print(f"\n✓ Evaluation completed!")


def save_results(args, results, output_dir, run_name, model_type):
    """Save evaluation results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = run_name or f"{args.cvd_type}_sev{args.severity}_{timestamp}"
    
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    # txt_path = os.path.join(output_dir, f"{base_name}_{model_type}.txt")
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    
    model_name = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))
    
    row = {
        'timestamp': timestamp,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'model_type': model_type,
        'checkpoint': args.checkpoint,
        'cvd_type': args.cvd_type,
        'severity': args.severity,
        'num_episodes': args.num_episodes,
        'mean_score': f"{results['mean_score']:.2f}",
        'std_score': f"{results['std_score']:.2f}",
        'mean_reward': f"{results['mean_reward']:.2f}",
        'std_reward': f"{results['std_reward']:.2f}",
        'mean_length': f"{results['mean_length']:.1f}",
        'std_length': f"{results['std_length']:.1f}",
        'max_score': results['max_score'],
        'min_score': results['min_score'],
        'scores': str(results['scores']),
        'rewards': str(results['rewards']),
        'lengths': str(results['lengths'])
    }
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"  Results appended to: {csv_path}")
    
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"CVD type: {args.cvd_type}, Severity: {args.severity}\n")
        f.write(f"Episodes: {args.num_episodes}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}\n")
        f.write(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}\n")
        f.write(f"Max Score: {results['max_score']:.0f}\n")
        f.write(f"Min Score: {results['min_score']:.0f}\n")
        f.write("-" * 60 + "\n")
        f.write("Individual Episodes:\n")
        for i, (score, reward, length) in enumerate(zip(results['scores'], results['rewards'], results['lengths'])):
            f.write(f"  Episode {i+1:3d}: Score={score:6.0f}, Reward={reward:7.2f}, Length={length:4d}\n")
        f.write("=" * 60 + "\n")
    
    print(f"  Summary saved to: {txt_path}")
    
    return csv_path, txt_path


if __name__ == "__main__":
    evaluate()
