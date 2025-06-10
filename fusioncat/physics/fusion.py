# fusioncat/physics/fusion.py
import astropy.units as u
import numpy as np
from numba import jit
from ..utils.fuels import Fuel

@jit(nopython=True, cache=True)
def _reactivity_dt_jit(T_keV: float) -> float:
    """
    Numba-JIT compiled kernel for D-T reactivity. This is the single,
    high-performance source of truth for this physics calculation.
    Takes temperature in keV and returns reactivity in m^3/s.
    Formula from H.-S. Bosch, G.M. Hale, Nuclear Fusion, Vol. 32 (1992).
    """
    # This function is designed to be called by other Numba functions.
    if T_keV <= 0.2:  # Fit is not accurate at very low T
        return 0.0

    # Coefficients for the analytical fit
    BG = 34.3827
    C = [1.17302E-9, 1.51361E-2, 7.51886E-2, 4.60643E-3, 1.35E-2, -1.06750E-4, 1.366E-6]
    
    # Numerically stable calculation for the theta parameter
    theta_denom_term1 = C[3] + T_keV * C[5]
    theta_denom_term2 = 1 + T_keV * theta_denom_term1
    
    # Avoid division by zero if the denominator is zero
    if theta_denom_term2 == 0:
        return 0.0
        
    theta_num_term1 = C[4] + T_keV * C[6]
    theta_num_term2 = C[2] + T_keV * theta_num_term1
    
    theta = T_keV / (1 - (T_keV * theta_num_term2) / theta_denom_term2)
    
    if theta <= 0:
        return 0.0
    
    xi = BG / (theta**0.5)
    sigma_v_cm3_s = C[0] * theta**(-2/3) * np.exp(-xi)
    
    # Convert from cm^3/s to m^3/s for SI consistency
    return sigma_v_cm3_s / 1e6


def calculate_reactivity(fuel: Fuel, T_i: u.Quantity) -> u.Quantity[u.m**3 / u.s]:
    """
    User-facing wrapper for reactivity calculations. This function handles
    Astropy Units and calls the appropriate JIT-compiled kernel.
    """
    T_keV = T_i.to_value(u.keV)

    if fuel.name == 'D-T':
        # Call the JIT kernel for the actual calculation
        reactivity_val = _reactivity_dt_jit(T_keV)
        return reactivity_val * u.m**3 / u.s
    else:
        # Placeholder for future fuel cycle implementations
        raise NotImplementedError(f"Reactivity calculation for {fuel.name} has not yet been implemented.")


def calculate_fusion_power(
    n_i: u.Quantity, T_i: u.Quantity, V: u.Quantity,
    fuel: Fuel, ratio: float
) -> tuple[u.Quantity, u.Quantity]:
    """Calculates total fusion power and charged particle power."""
    sigma_v = calculate_reactivity(fuel, T_i)
    n1, n2 = n_i * ratio, n_i * (1 - ratio)
    factor = 0.5 if fuel.reactants[0] == fuel.reactants[1] else 1.0
    power_density = factor * n1 * n2 * sigma_v * fuel.energy_per_reaction
    p_fusion_total = (power_density * V).to(u.W)
    p_charged_particles = p_fusion_total * fuel.charged_particle_fraction
    return p_fusion_total, p_charged_particles