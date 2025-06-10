# fusioncat/profiles/shapes.py
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def parabolic_profile(r_grid: np.ndarray, core_val: float, edge_val: float, alpha: float) -> np.ndarray:
    """
    Generates a parabolic-like profile on a given radial grid.
    
    Formula: f(r) = (core - edge) * (1 - r^2)^alpha + edge
    
    Args:
        r_grid: 1D array of normalized radial points (from 0 to 1).
        core_val: The value at the center (r=0).
        edge_val: The value at the edge (r=1).
        alpha: The peaking factor. Higher alpha means a more peaked profile.
    
    Returns:
        A 1D numpy array of the profile values.
    """
    return (core_val - edge_val) * (1 - r_grid**2)**alpha + edge_val 