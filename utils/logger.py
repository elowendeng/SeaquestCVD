# utils/logger.py

"""
Logging utilities for training metrics
"""

import os
import json
from datetime import datetime
from typing import Dict, Any
import numpy as np


class Logger:
    """Simple logger for training metrics"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_path, exist_ok=True)
        
        self.metrics = []
        self.start_time = datetime.now()
    
    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics at a given step"""
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(entry)
        
        # Print to console
        print(f"\nStep {step}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    def save(self):
        """Save metrics to file"""
        save_path = os.path.join(self.log_path, 'metrics.json')
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Also save as numpy for easy loading
        np_path = os.path.join(self.log_path, 'metrics.npy')
        np.save(np_path, self.metrics)
    
    def load(self):
        """Load metrics from file"""
        load_path = os.path.join(self.log_path, 'metrics.json')
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                self.metrics = json.load(f)
        return self.metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.metrics:
            return {}
        
        summary = {
            'total_steps': self.metrics[-1]['step'],
            'duration': str(datetime.now() - self.start_time),
            'experiment': self.experiment_name
        }
        
        # Get final metrics
        if self.metrics:
            final_metrics = {k: v for k, v in self.metrics[-1].items() if k not in ['step', 'timestamp']}
            summary['final'] = final_metrics
        
        return summary
