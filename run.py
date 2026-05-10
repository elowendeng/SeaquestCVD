# run.py

"""
One-click to run the entire experiment
Now runs 5 experiments with correct training-test configurations:
1a. Normal model (train on I, test on I)
1b. Normal model (train on I, test on CVD)
2. Deutan model (train on CVD, test on CVD)
3a. Diff Original model (train on [I, diff], test on [I, diff])
3b. Diff Original model (train on [I, diff], test on [CVD, diff])
3c. Diff CVD model (train on [CVD, diff], test on [CVD, diff])
"""

import subprocess
import os
import sys
import argparse
import time
import shutil
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Run complete experiments for CVD Atari project')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['debug', 'quick', 'medium', 'full'],
                       help='Experiment mode: debug(10k), quick(100k), medium(500k), full(1M)')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training, only run evaluation')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--skip-visualize', action='store_true',
                       help='Skip visualization after evaluation')
    parser.add_argument('--record-final-video', action='store_true',
                       help='Record final videos using best models after training')
    parser.add_argument('--video-severity', type=float, default=1.0,
                       help='CVD severity for final video')
    parser.add_argument('--record-video-interval', type=int, default=10000,
                       help='Record training video every N steps (default: 10000)')
    parser.add_argument('--video-length', type=int, default=2000,
                       help='Max frames per recorded video (default: 2000)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean existing results before training')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    return parser.parse_args()


def get_timesteps(mode: str) -> int:
    steps = {
        'debug': 10_000,
        'quick': 100_000,
        'medium': 500_000,
        'full': 1_000_000
    }
    return steps.get(mode, 100_000)


def get_mode_description(mode: str) -> str:
    descriptions = {
        'debug': 'Debug Mode (10k steps)',
        'quick': 'Quick Mode (100k steps)',
        'medium': 'Medium Mode (500k steps)',
        'full': 'Full Mode (1M steps)'
    }
    return descriptions.get(mode, 'Unknown Mode')


def run_command(cmd, description, dry_run=False):
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print(f"Command: {cmd}")
    print("-" * 70)
    
    if dry_run:
        print("[DRY RUN] Command would be executed here")
        return True
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n! Warning: Command failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True


def clean_directories():
    dirs_to_clean = [
        "results/normal_model",
        "results/deutan_model",
        "results/diff_original_model",
        "results/diff_original_model_2",  # For 3b (same training, different test)
        "results/diff_cvd_model",
        "results/comparison",
    ]
    
    print("\n" + "=" * 70)
    print("Cleaning existing results...")
    print("=" * 70)
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"  ✓ Removed {dir_path}/")
        else:
            print(f"  - {dir_path}/ (not exists)")
    
    print("Clean complete!")


def setup_directories():
    print("\n" + "=" * 70)
    print("Setting up directory structure...")
    print("=" * 70)
    
    directories = [
        "results/normal_model/checkpoints",
        "results/normal_model/logs",
        "results/normal_model/videos",
        "results/normal_model/evaluation",
        "results/deutan_model/checkpoints",
        "results/deutan_model/logs",
        "results/deutan_model/videos",
        "results/deutan_model/evaluation",
        "results/diff_original_model/checkpoints",
        "results/diff_original_model/logs",
        "results/diff_original_model/videos",
        "results/diff_original_model/evaluation",
        "results/diff_cvd_model/checkpoints",
        "results/diff_cvd_model/logs",
        "results/diff_cvd_model/videos",
        "results/diff_cvd_model/evaluation",
        "results/comparison"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")


def check_dependencies():
    print("\n" + "=" * 70)
    print("Checking dependencies...")
    print("=" * 70)
    
    modules_to_check = [
        ('jax', 'jax'),
        ('equinox', 'equinox'),
        ('gymnasium', 'gymnasium'),
        ('cv2', 'opencv-python'),
        ('yaml', 'pyyaml'),
        ('tqdm', 'tqdm'),
        ('ale_py', 'ale-py'),
        ('matplotlib', 'matplotlib'),
    ]
    
    missing = []
    for module_name, package_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError:
            print(f"  ✗ {module_name} (missing: pip install {package_name})")
            missing.append(package_name)
    
    try:
        from daltonlens import simulate
        print("  ✓ daltonlens")
    except ImportError:
        print("  ✗ daltonlens (missing: pip install daltonlens)")
        missing.append('daltonlens')
    
    if missing:
        print(f"\n!!! Missing dependencies: {', '.join(missing)}")
        print("  Please run: pip install -r requirements.txt")
        return False
    
    print("\n:) All dependencies satisfied!")
    return True


def check_models_exist():
    normal_exists = os.path.exists("results/normal_model/checkpoints/best.eqx")
    deutan_exists = os.path.exists("results/deutan_model/checkpoints/best.eqx")
    diff_original_exists = os.path.exists("results/diff_original_model/checkpoints/best.eqx")
    diff_cvd_exists = os.path.exists("results/diff_cvd_model/checkpoints/best.eqx")
    return normal_exists, deutan_exists, diff_original_exists, diff_cvd_exists


def print_summary(args, total_steps):
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Mode:              {args.mode} - {get_mode_description(args.mode)}")
    print(f"  Total steps:       {total_steps:,}")
    print(f"  Skip training:     {'Yes' if args.skip_train else 'No'}")
    print(f"  Skip evaluation:   {'Yes' if args.skip_eval else 'No'}")
    print(f"  Record final video:{'Yes' if args.record_final_video else 'No'}")
    print(f"  Record interval:   {args.record_video_interval:,} steps")
    print(f"  Video length:      {args.video_length} frames")
    print(f"  Clean results:     {'Yes' if args.clean else 'No'}")
    print(f"  GPU:               {'Enabled' if args.gpu else 'Disabled'}")
    print(f"  Dry run:           {'Yes' if args.dry_run else 'No'}")
    print("=" * 70)


def print_completion_summary(start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)
    print(f"  Total time: {hours}h {minutes}m {seconds}s")
    
    print("\nModel checkpoints:")
    for model in ['normal_model', 'deutan_model', 'diff_original_model', 'diff_cvd_model']:
        path = f"results/{model}/checkpoints/best.eqx"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✓ {model}: {path} ({size:.1f} KB)")
        else:
            print(f"  ✗ {model}: not found")
    
    print("\nComparison directory (results/comparison/):")
    if os.path.exists("results/comparison"):
        for f in os.listdir("results/comparison"):
            if f.endswith('.png') or f.endswith('.mp4'):
                print(f"  - {f}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("  View results:")
    print("    ls results/comparison/")
    print("\n  Clean up and re-run:")
    print("    python run.py --clean --mode quick")
    print("=" * 70)


def run_visualization(args, dry_run=False):
    """Run visualization to generate charts"""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    viz_cmd = "python visualize.py"
    
    success = run_command(
        viz_cmd,
        "Generating training curves and CVD effect images",
        dry_run=dry_run
    )
    
    if not success and not dry_run:
        print("!!! Visualization failed")
        return False
    
    return True


def record_final_videos(args, dry_run=False):
    """Record final comparison videos"""
    print("\n" + "=" * 70)
    print("RECORDING FINAL VIDEOS")
    print("=" * 70)
    
    results = []
    output_dir = "results/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    normal_best = "results/normal_model/checkpoints/best.eqx"
    deutan_best = "results/deutan_model/checkpoints/best.eqx"
    diff_original_best = "results/diff_original_model/checkpoints/best.eqx"
    diff_cvd_best = "results/diff_cvd_model/checkpoints/best.eqx"
    
    # 1. Normal model on original vision
    if os.path.exists(normal_best):
        print("\n  [1/6] Recording: Normal Model on Original Vision")
        cmd = (f"python record.py --checkpoint {normal_best} --cvd-type normal "
               f"--severity 0.0 --output-dir {output_dir} "
               f"--output-scale 2.0 --num-episodes 1")
        success = run_command(cmd, "Normal Model on Original", dry_run=dry_run)
        results.append(("Normal Model (Original)", success))
    
    # 2. Normal model on CVD
    if os.path.exists(normal_best):
        print("\n  [2/6] Recording: Normal Model on CVD")
        cmd = (f"python record.py --checkpoint {normal_best} --cvd-type deutan "
               f"--severity {args.video_severity} --output-dir {output_dir} "
               f"--output-scale 2.0 --num-episodes 1")
        success = run_command(cmd, "Normal Model on CVD", dry_run=dry_run)
        results.append(("Normal Model (CVD)", success))
    
    # 3. Deutan model on CVD
    if os.path.exists(deutan_best):
        print("\n  [3/6] Recording: Deutan Model on CVD")
        cmd = (f"python record.py --checkpoint {deutan_best} --cvd-type deutan "
               f"--severity {args.video_severity} --output-dir {output_dir} "
               f"--output-scale 2.0 --num-episodes 1")
        success = run_command(cmd, "Deutan Model on CVD", dry_run=dry_run)
        results.append(("Deutan Model (CVD)", success))
    
    # 4. Diff Original model on original vision (3a)
    if os.path.exists(diff_original_best):
        print("\n  [4/6] Recording: Diff Original Model on Original Vision (Exp 3a)")
        cmd = (f"python record.py --checkpoint {diff_original_best} --cvd-type normal "
               f"--severity 0.0 --output-dir {output_dir} "
               f"--use-diff-training --diff-training-mode original "
               f"--output-scale 2.0 --num-episodes 1")
        success = run_command(cmd, "Diff Original Model on Original", dry_run=dry_run)
        results.append(("Diff Original Model (Original)", success))
    
    # 5. Diff Original model on CVD (3b)
    if os.path.exists(diff_original_best):
        print("\n  [5/6] Recording: Diff Original Model on CVD (Exp 3b)")
        cmd = (f"python record.py --checkpoint {diff_original_best} --cvd-type deutan "
               f"--severity {args.video_severity} --output-dir {output_dir} "
               f"--use-diff-training --diff-training-mode original "
               f"--output-scale 2.0 --num-episodes 1")
        success = run_command(cmd, "Diff Original Model on CVD", dry_run=dry_run)
        results.append(("Diff Original Model (CVD)", success))
    
    # 6. Diff CVD model on CVD (3c)
    if os.path.exists(diff_cvd_best):
        print("\n  [6/6] Recording: Diff CVD Model on CVD (Exp 3c)")
        cmd = (f"python record.py --checkpoint {diff_cvd_best} --cvd-type deutan "
               f"--severity {args.video_severity} --output-dir {output_dir} "
               f"--use-diff-training --diff-training-mode cvd "
               f"--output-scale 2.0 --num-episodes 1")
        success = run_command(cmd, "Diff CVD Model on CVD", dry_run=dry_run)
        results.append(("Diff CVD Model (CVD)", success))
    
    print("\n" + "=" * 70)
    print("FINAL VIDEO RECORDING SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  [{status}] {name}")
    print("=" * 70)
    
    return True


def main():
    args = parse_args()
    total_steps = get_timesteps(args.mode)
    start_time = time.time()
    
    if args.gpu:
        os.environ['JAX_PLATFORMS'] = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_autotune_level=0'
        print("✓ GPU mode enabled")
    else:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
        print("✓ CPU mode enabled")
    
    import jax
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  JAX devices: {jax.devices()}")
    
    print("=" * 70)
    print(" " * 20 + "SEAQUEST COLORBLIND PPO EXPERIMENTS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not check_dependencies():
        print("\nT^T Please install missing dependencies and try again.")
        sys.exit(1)
    
    normal_exists, deutan_exists, diff_original_exists, diff_cvd_exists = check_models_exist()
    if (normal_exists or deutan_exists or diff_original_exists or diff_cvd_exists) and not args.clean and not args.skip_train:
        print("\n!!! Existing models found")
        response = input("Continue without cleaning? (y/N): ")
        if response.lower() != 'y':
            print("Aborted. Use --clean to remove existing results.")
            sys.exit(0)
    
    if args.clean:
        clean_directories()
    
    setup_directories()
    print_summary(args, total_steps)
    
    if args.mode == 'full' and not args.dry_run and not args.skip_train:
        response = input("\n!!! Full mode will take several hours. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # ========== Training phase ==========
    if not args.skip_train:
        common_args = f"--record-video-interval {args.record_video_interval} --video-length {args.video_length}"
        aug_flag = "--use-augmentation" if args.mode not in ['debug', 'quick'] else ""

        # Experiment 1: Normal Visual Training - results/normal_model
        run_command(f"python train.py --cvd-type normal --total-timesteps {total_steps} {common_args}",
                   "Exp 1: Training Normal Vision Model", dry_run=args.dry_run)

        # Experiment 2: Deuteranopia Training - results/deutan_model
        run_command(f"python train.py --cvd-type deutan --total-timesteps {total_steps} {common_args} {aug_flag}",
                   "Exp 2: Training Deuteranopia Model", dry_run=args.dry_run)

        # Experiment 3a & 3b: Original + Difference Map - results/diff_original_model
        # (Same training, will be evaluated in two ways)
        run_command(f"python train.py --cvd-type normal --use-diff-training --diff-training-mode original "
                    f"--output-dir results/diff_original_model "
                    f"--total-timesteps {total_steps} {common_args} {aug_flag}",
                   "Exp 3a/3b: Training with Original + Difference Map", dry_run=args.dry_run)

        # Experiment 3c: CVD + Difference Map - results/diff_cvd_model
        run_command(f"python train.py --cvd-type normal --use-diff-training --diff-training-mode cvd "
                    f"--output-dir results/diff_cvd_model "
                    f"--total-timesteps {total_steps} {common_args} {aug_flag}",
                   "Exp 3c: Training with CVD + Difference Map", dry_run=args.dry_run)
    
    # ========== Evaluation stage ==========
    if not args.skip_eval:
        normal_exists, deutan_exists, diff_original_exists, diff_cvd_exists = check_models_exist()
        
        # Exp 1a: Normal model on original vision
        if normal_exists:
            run_command("python evaluate.py --checkpoint results/normal_model/checkpoints/best.eqx "
                       "--cvd-type normal --severity 0.0 --num-episodes 10 "
                       "--run-name exp1a_normal_on_original",
                       "Exp 1a: Normal Model on Original Vision", dry_run=args.dry_run)
        # python evaluate.py --checkpoint results/normal_model/checkpoints/best.eqx --cvd-type normal --severity 0.0 --num-episodes 10 --run-name exp1a_normal_on_original
        
        # Exp 1b: Normal model on CVD
        if normal_exists:
            run_command("python evaluate.py --checkpoint results/normal_model/checkpoints/best.eqx "
                       "--cvd-type deutan --severity 1.0 --num-episodes 10 "
                       "--run-name exp1b_normal_on_cvd",
                       "Exp 1b: Normal Model on CVD", dry_run=args.dry_run)
        # python evaluate.py --checkpoint results/normal_model/checkpoints/best.eqx --cvd-type deutan --severity 1.0 --num-episodes 10 --run-name exp1b_normal_on_cvd
        
        # Exp 2: Deutan model on CVD
        if deutan_exists:
            run_command("python evaluate.py --checkpoint results/deutan_model/checkpoints/best.eqx "
                       "--cvd-type deutan --severity 1.0 --num-episodes 10 "
                       "--run-name exp2_deutan_on_cvd",
                       "Exp 2: Deutan Model on CVD", dry_run=args.dry_run)
        # python evaluate.py --checkpoint results/deutan_model/checkpoints/best.eqx --cvd-type deutan --severity 1.0 --num-episodes 10 --run-name exp2_deutan_on_cvd
        
        # Exp 3a: Diff Original model on original vision
        if diff_original_exists:
            run_command("python evaluate.py --checkpoint results/diff_original_model/checkpoints/best.eqx "
                       "--cvd-type normal --severity 0.0 --num-episodes 10 "
                       "--use-diff-training --diff-training-mode original "
                       "--run-name exp3a_diff_original_on_original",
                       "Exp 3a: Diff Original Model on Original Vision", dry_run=args.dry_run)
        # python evaluate.py --checkpoint results/diff_original_model/checkpoints/best.eqx --cvd-type normal --severity 0.0 --num-episodes 10 --use-diff-training --diff-training-mode original --run-name exp3a_diff_original_on_original
        
        # Exp 3b: Diff Original model on CVD
        if diff_original_exists:
            run_command("python evaluate.py --checkpoint results/diff_original_model/checkpoints/best.eqx "
                       "--cvd-type deutan --severity 1.0 --num-episodes 10 "
                       "--use-diff-training --diff-training-mode original "
                       "--run-name exp3b_diff_original_on_cvd",
                       "Exp 3b: Diff Original Model on CVD (Generalization)", dry_run=args.dry_run)
        # python evaluate.py --checkpoint results/diff_original_model/checkpoints/best.eqx --cvd-type deutan --severity 1.0 --num-episodes 10 --use-diff-training --diff-training-mode original --run-name exp3b_diff_original_on_cvd
        
        # Exp 3c: Diff CVD model on CVD
        if diff_cvd_exists:
            run_command("python evaluate.py --checkpoint results/diff_cvd_model/checkpoints/best.eqx "
                       "--cvd-type deutan --severity 1.0 --num-episodes 10 "
                       "--use-diff-training --diff-training-mode cvd "
                       "--run-name exp3c_diff_cvd_on_cvd",
                       "Exp 3c: Diff CVD Model on CVD", dry_run=args.dry_run)
        # python evaluate.py --checkpoint results/diff_cvd_model/checkpoints/best.eqx --cvd-type deutan --severity 1.0 --num-episodes 10 --use-diff-training --diff-training-mode cvd --run-name exp3c_diff_cvd_on_cvd
    
    # ========== Record final videos ==========
    if args.record_final_video and not args.dry_run:
        record_final_videos(args, dry_run=args.dry_run)
    
    # ========== Visualization ==========
    if not args.skip_visualize:
        run_visualization(args, dry_run=args.dry_run)
    
    print_completion_summary(start_time)


if __name__ == "__main__":
    main()


# python run.py --mode full --clean --record-final-video --gpu --record-video-interval 100_000
