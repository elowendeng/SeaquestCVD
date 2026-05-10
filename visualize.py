# visualize.py

"""
Functions:
1. Generate a comparison chart of training curves (plotting FOUR models together)
2. Generate a visual representation of CVD effect
3. Generate performance comparison bar chart for all four models (only latest evaluation)
"""

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from env import make_env, simulate_cvd


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize trained policies')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Base results directory')
    return parser.parse_args()


def plot_training_curves(results_dir, output_dir):
    """
    Draw a comparison chart of training curves for FOUR models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    
    models_config = {
        'normal': {
            'name': 'normal_model',
            'label': 'Normal Model (train on RGB)',
            'color': '#2ecc71',
            'linestyle': '-',
            'linewidth': 2.5
        },
        'deutan': {
            'name': 'deutan_model',
            'label': 'Deutan Model (train on CVD)',
            'color': '#e74c3c',
            'linestyle': '-',
            'linewidth': 2.5
        },
        'diff_original': {
            'name': 'diff_original_model',
            'label': 'Diff Original Model (train on [I, |I-CVD|])',
            'color': '#3498db',
            'linestyle': '--',
            'linewidth': 2.5
        },
        'diff_cvd': {
            'name': 'diff_cvd_model',
            'label': 'Diff CVD Model (train on [CVD, |I-CVD|])',
            'color': '#9b59b6',
            'linestyle': ':',
            'linewidth': 2.5
        }
    }
    
    has_data = False
    
    for key, config in models_config.items():
        log_path = os.path.join(results_dir, config['name'], 'logs', 'training_log.txt')
        
        if os.path.exists(log_path):
            try:
                data = np.loadtxt(log_path, delimiter=',', skiprows=1)
                if len(data.shape) == 2 and len(data) > 0:
                    steps = data[:, 0] / 1_000_000
                    rewards = data[:, 1]
                    
                    window = min(50, len(rewards) // 20) if len(rewards) > 30 else 1
                    if len(rewards) > window:
                        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        steps_smoothed = steps[window-1:]
                        plt.plot(steps_smoothed, smoothed, 
                                color=config['color'], 
                                linestyle=config['linestyle'],
                                linewidth=config['linewidth'], 
                                label=config['label'])
                    else:
                        plt.plot(steps, rewards, 
                                color=config['color'],
                                linestyle=config['linestyle'],
                                linewidth=config['linewidth'], 
                                label=config['label'])
                    
                    print(f"  Loaded {config['name']} log: {len(data)} entries, final reward: {rewards[-1]:.2f}")
                    has_data = True
                else:
                    print(f"  No valid data in {config['name']} log")
            except Exception as e:
                print(f"  Warning: Could not plot {log_path}: {e}")
        else:
            print(f"  Log file not found: {log_path}")
    
    if not has_data:
        print("  No training log files found")
        plt.close()
        return None
    
    plt.xlabel('Training Steps (millions)', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Training Curves Comparison: Four Experiments', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Training curves saved to {output_path}")
    
    return output_path


def visualize_cvd_effect(output_dir):
    """Visualize the comparison chart of different degrees of color blindness"""
    os.makedirs(output_dir, exist_ok=True)
    
    env = make_env(
        cvd_type='normal',
        severity=0.0,
        train_mode=False,
        frame_stack=1
    )
    
    obs, _ = env.reset()
    
    if len(obs.shape) == 3 and obs.shape[-1] == 3:
        frame = obs
    elif len(obs.shape) == 3 and obs.shape[-1] > 3:
        frame = obs[:, :, :3]
    else:
        frame = obs
    
    env.close()
    
    severities = [0.0, 0.3, 0.6, 1.0]
    titles = ['Normal Vision', 'Mild (0.3)', 'Moderate (0.6)', 'Complete (1.0)']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (severity, title) in enumerate(zip(severities, titles)):
        if frame.max() <= 1.0:
            img = (frame * 255).astype(np.uint8)
        else:
            img = frame.astype(np.uint8)
        
        cvd_img = simulate_cvd(img, 'deutan', severity)
        cvd_img = cvd_img.astype(np.float32) / 255.0
        
        axes[i].imshow(cvd_img)
        axes[i].set_title(title, fontsize=14)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'cvd_effect.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ CVD effect visualization saved to {output_path}")
    
    return output_path


def get_latest_evaluation_file(eval_dir):
    """
    Get the latest evaluation file in a directory (by modification time).
    Returns filepath or None if not found.
    """
    if not os.path.exists(eval_dir):
        return None
    
    txt_files = []
    for filename in os.listdir(eval_dir):
        if filename.endswith('.txt') and not filename.startswith('evaluation'):
            filepath = os.path.join(eval_dir, filename)
            mtime = os.path.getmtime(filepath)
            txt_files.append((mtime, filepath, filename))
    
    if not txt_files:
        return None
    
    # Sort by modification time (newest first)
    txt_files.sort(reverse=True)
    return txt_files[0][1]


def parse_evaluation_file(filepath):
    """
    Parse evaluation file to extract model_type, mean_score, std_score.
    Returns (model_type, mean_score, std_score) or (None, None, None).
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            model_type = None
            mean_score = None
            std_score = None
            
            for line in content.split('\n'):
                if 'Model Type:' in line:
                    model_type = line.split(':')[1].strip()
                if 'Mean Score:' in line:
                    parts = line.split('±')
                    mean_score = float(parts[0].split(':')[1].strip())
                    std_score = float(parts[1].strip())
            
            return model_type, mean_score, std_score
    except Exception as e:
        print(f"  Warning: Could not parse {filepath}: {e}")
        return None, None, None


def plot_experiment_comparison_bar(results_dir, output_dir):
    """
    Draw a bar chart comparing the performance of all models on CVD test.
    Uses ONLY the LATEST evaluation file for each model type.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define each experiment with its model directory and expected model_type
    # We'll match by reading the model_type from file content
    model_dirs = ['normal_model', 'deutan_model', 'diff_original_model', 'diff_cvd_model']
    
    # Mapping from model_type to display name and color
    display_config = {
        'normal_exp1a': {'display_name': 'Normal Model (on Original)', 'color': '#2ecc71', 'order': 1},
        'normal_exp1b': {'display_name': 'Normal Model (on CVD)', 'color': '#2ecc71', 'order': 2},
        'deutan_exp2': {'display_name': 'Deutan Model (on CVD)', 'color': '#e74c3c', 'order': 3},
        'diff_original_exp3a': {'display_name': 'Diff Original (on Original)', 'color': '#3498db', 'order': 4},
        'diff_original_exp3b': {'display_name': 'Diff Original (on CVD)', 'color': '#3498db', 'order': 5},
        'diff_cvd_exp3c': {'display_name': 'Diff CVD (on CVD)', 'color': '#9b59b6', 'order': 6},
    }
    
    # Store results keyed by model_type
    results_dict = {}
    
    for model_dir in model_dirs:
        eval_dir = os.path.join(results_dir, model_dir, 'evaluation')
        latest_file = get_latest_evaluation_file(eval_dir)
        
        if latest_file:
            model_type, mean_score, std_score = parse_evaluation_file(latest_file)
            if model_type and mean_score is not None:
                results_dict[model_type] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'filename': os.path.basename(latest_file)
                }
                print(f"  Loaded {model_type}: {mean_score:.0f} ± {std_score:.0f} (from {os.path.basename(latest_file)})")
    
    if not results_dict:
        print("  No evaluation data found, skipping bar chart")
        return None
    
    # Build data for bar chart (only include entries that have display config)
    scores = []
    stds = []
    labels = []
    colors = []
    
    # Sort by order
    sorted_model_types = sorted(
        [mt for mt in results_dict.keys() if mt in display_config],
        key=lambda x: display_config[x]['order']
    )
    
    for model_type in sorted_model_types:
        config = display_config[model_type]
        data = results_dict[model_type]
        scores.append(data['mean_score'])
        stds.append(data['std_score'])
        labels.append(config['display_name'])
        colors.append(config['color'])
    
    if not scores:
        print("  No evaluation data found for configured experiments")
        return None
    
    # Draw bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.6
    
    bars = ax.bar(x, scores, width, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5,
                  yerr=stds, capsize=5, error_kw={'linewidth': 1.5})
    
    y_max = max(scores) if scores else 100
    for bar, score, std in zip(bars, scores, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(5, y_max * 0.02),
               f'{score:.0f}±{std:.0f}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Performance Comparison on Color-Blind Test Environment', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=10, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add baseline reference (Normal on Original)
    if 'normal_exp1a' in results_dict:
        baseline_score = results_dict['normal_exp1a']['mean_score']
        ax.axhline(y=baseline_score, color='#2ecc71', linestyle='--', alpha=0.5, 
                   label=f'Normal on Original Baseline: {baseline_score:.0f}')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Performance comparison bar chart saved to {output_path}")
    
    return output_path


def main():
    args = parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        frame_stack = config['environment']['frame_stack']
    else:
        frame_stack = 4
        print(f"!!! Config not found, using frame_stack={frame_stack}")
    
    output_dir = os.path.join(args.results_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION: Model Comparison and Analysis")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Frame stack: {frame_stack}")
    print("=" * 70)
    
    print("\n[1/3] Plotting training curves (4 models)...")
    plot_training_curves(args.results_dir, output_dir)
    
    print("\n[2/3] Visualizing CVD effect...")
    visualize_cvd_effect(output_dir)
    
    print("\n[3/3] Plotting performance comparison bar chart...")
    plot_experiment_comparison_bar(args.results_dir, output_dir)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll files saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - training_curves.png      (Training curves of 4 models)")
    print("  - cvd_effect.png           (CVD simulation effect)")
    print("  - performance_comparison.png (Bar chart of test performance)")
    print("=" * 70)


if __name__ == "__main__":
    main()
