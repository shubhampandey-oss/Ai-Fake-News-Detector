"""
Deterministic Execution Module

Ensures reproducible predictions by setting seeds across all libraries.
This is CRITICAL for production-grade ML systems.

Same input → Same output, every time.
"""

import os
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np

# Try to import torch (optional for classical models)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import sklearn
try:
    from sklearn.utils import check_random_state
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Global state tracking
_seeds_set = False
_current_seed = None


def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for all libraries to ensure deterministic behavior.
    
    This function should be called ONCE at application startup.
    
    Args:
        seed: Random seed value (default: 42)
    """
    global _seeds_set, _current_seed
    
    # Python's random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Use deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (TypeError, AttributeError):
            # Older PyTorch versions
            pass
    
    _seeds_set = True
    _current_seed = seed


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """
    Get a NumPy RandomState object for reproducible randomness.
    
    Args:
        seed: Optional seed (uses global seed if not provided)
        
    Returns:
        numpy.random.RandomState object
    """
    if seed is None:
        seed = _current_seed if _current_seed is not None else 42
    return np.random.RandomState(seed)


@contextmanager
def ensure_reproducibility(seed: int = 42):
    """
    Context manager to ensure reproducible execution within a block.
    
    Usage:
        with ensure_reproducibility(42):
            # All operations here will be deterministic
            result = model.predict(data)
    
    Args:
        seed: Random seed value
    """
    # Save current states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    
    if TORCH_AVAILABLE:
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    
    try:
        # Set seeds
        set_all_seeds(seed)
        yield
    finally:
        # Restore states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        
        if TORCH_AVAILABLE:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def is_deterministic_mode() -> bool:
    """
    Check if deterministic mode is enabled.
    
    Returns:
        True if seeds have been set
    """
    return _seeds_set


def get_current_seed() -> Optional[int]:
    """
    Get the current global seed.
    
    Returns:
        Current seed value or None if not set
    """
    return _current_seed


class DeterministicInference:
    """
    Wrapper class for deterministic model inference.
    
    Ensures the model is in eval mode and uses deterministic settings.
    """
    
    def __init__(self, model, seed: int = 42):
        """
        Initialize deterministic inference wrapper.
        
        Args:
            model: The ML model to wrap
            seed: Random seed for reproducibility
        """
        self.model = model
        self.seed = seed
        self._original_training = None
    
    def __enter__(self):
        """Enter deterministic inference mode."""
        set_all_seeds(self.seed)
        
        # If PyTorch model, set to eval mode
        if TORCH_AVAILABLE and hasattr(self.model, 'eval'):
            if hasattr(self.model, 'training'):
                self._original_training = self.model.training
            self.model.eval()
        
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit deterministic inference mode."""
        # Restore training mode if it was changed
        if TORCH_AVAILABLE and self._original_training is not None:
            if self._original_training:
                self.model.train()
        return False


# Initialize with default seed on import
# This ensures the module is always in a deterministic state
set_all_seeds(42)
