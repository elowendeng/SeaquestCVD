# utils/setup_directories.py

"""
Setup directory structure for the project - ALL UNDER results/
Simplified for unified training (normal and deutan models)
"""

import os

def setup_directories():
    """Create all necessary directories under results/"""
    
    # Base results directory
    base_dir = "results"
    
    directories = [
        # Normal model
        f'{base_dir}/normal_model/checkpoints',
        f'{base_dir}/normal_model/logs',
        f'{base_dir}/normal_model/videos',
        f'{base_dir}/normal_model/evaluation',
        
        # Deuteranopia model
        f'{base_dir}/deutan_model/checkpoints',
        f'{base_dir}/deutan_model/logs',
        f'{base_dir}/deutan_model/videos',
        f'{base_dir}/deutan_model/evaluation',
        
        # Diff model
        f'{base_dir}/diff_original_model/checkpoints',
        f'{base_dir}/diff_original_model/logs',
        f'{base_dir}/diff_original_model/videos',
        f'{base_dir}/diff_original_model/evaluation',
        
        f'{base_dir}/diff_cvd_model/checkpoints',
        f'{base_dir}/diff_cvd_model/logs',
        f'{base_dir}/diff_cvd_model/videos',
        f'{base_dir}/diff_cvd_model/evaluation',
        
        # Comparison results
        f'{base_dir}/comparison'
    ]
    
    print("Creating directory structure under results/...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    print("\nAll directories created successfully!")


if __name__ == "__main__":
    setup_directories()
