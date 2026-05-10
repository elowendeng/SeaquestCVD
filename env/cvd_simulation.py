# env/cvd_simulation.py

"""
Color Vision Deficiency Simulation using daltonlens library
Based on Brettel (1997) method - Professional implementation
"""

import numpy as np
from typing import Literal
import warnings

warnings.filterwarnings("ignore")

# Use daltonlens library
try:
    from daltonlens import simulate
    DALTONLENS_AVAILABLE = True
except ImportError:
    DALTONLENS_AVAILABLE = False
    print("! Warning: daltonlens not installed, run: pip install daltonlens")

CVDType = Literal['normal', 'protan', 'deutan', 'tritan']


class BrettelSimulator:
    """
    Color blindness simulator using daltonlens library
    Implements the standard algorithm from Brettel (1997) paper
    """
    
    def __init__(self):
        if not DALTONLENS_AVAILABLE:
            raise ImportError("Please install daltonlens: pip install daltonlens")
        
        # Create Brettel simulator
        self.simulator = simulate.Simulator_Brettel1997()
        
        # Type mapping
        self._type_map = {
            'protan': simulate.Deficiency.PROTAN,
            'deutan': simulate.Deficiency.DEUTAN,
            'tritan': simulate.Deficiency.TRITAN,
        }
    
    def simulate(
        self, 
        image: np.ndarray, 
        cvd_type: CVDType, 
        severity: float = 1.0
    ) -> np.ndarray:
        """
        Apply color blindness simulation
        
        Args:
            image: RGB image (H, W, 3), range [0, 255]
            cvd_type: 'normal', 'protan', 'deutan', 'tritan'
            severity: Severity 0-1
        
        Returns:
            Simulated RGB image
        """
        if cvd_type == 'normal' or severity == 0:
            return image
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img = (image * 255).astype(np.uint8)
            else:
                img = image.astype(np.uint8)
        else:
            img = image.copy()
        
        # Get deficiency type
        deficiency = self._type_map.get(cvd_type)
        if deficiency is None:
            return image
        
        # Apply simulation
        try:
            simulated = self.simulator.simulate_cvd(
                img, 
                deficiency, 
                severity=severity
            )
        except Exception as e:
            print(f"CVD simulation failed: {e}")
            return image
        
        return simulated
    
    def simulate_batch(
        self, 
        images: np.ndarray, 
        cvd_type: CVDType, 
        severity: float = 1.0
    ) -> np.ndarray:
        """Batch processing"""
        if cvd_type == 'normal' or severity == 0:
            return images
        
        results = []
        for i in range(images.shape[0]):
            results.append(self.simulate(images[i], cvd_type, severity))
        
        return np.stack(results, axis=0)
    
    def get_cvd_types(self) -> list:
        return ['normal', 'protan', 'deutan', 'tritan']


# Singleton simulator for convenience
_SIMULATOR = None

def get_simulator():
    global _SIMULATOR
    if _SIMULATOR is None:
        _SIMULATOR = BrettelSimulator()
    return _SIMULATOR

def simulate_cvd(image, cvd_type, severity=1.0):
    """Convenience function - uses singleton simulator"""
    return get_simulator().simulate(image, cvd_type, severity)
