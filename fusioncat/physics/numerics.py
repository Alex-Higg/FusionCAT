"""
Contains custom, Numba-JIT compatible numerical routines needed for
the physics solvers.
"""
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def gradient_1d(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Calculates the gradient of a 1D array using a second-order
    central difference scheme. This is a Numba-compatible replacement
    for np.gradient().

    The endpoints are handled with first-order forward/backward differences.
    """
    grad = np.zeros_like(y)
    
    # Forward difference for the first point
    grad[0] = (y[1] - y[0]) / dx
    
    # Central difference for the interior points
    for i in range(1, len(y) - 1):
        grad[i] = (y[i+1] - y[i-1]) / (2 * dx)
        
    # Backward difference for the last point
    grad[-1] = (y[-1] - y[-2]) / dx
    
    return grad 