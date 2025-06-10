# fusioncat/physics/fusion.py
import astropy.units as u
import numpy as np
from numba import jit
from ..utils.fuels import Fuel

@jit(nopython=True, cache=True)
def _reactivity_dt_jit(T_keV: float) -> float:
    """
    Numba-JIT compiled kernel for D-T reactivity.
    Takes temperature in keV and returns reactivity in m^3/s.
    This is the single source of truth for this physics calculation.
    """
    if T_keV <= 0:
        return 0.0

    # This is the simple and robust analytical fit from the NRL Plasma Formulary (2019).
    # The formula itself gives cm^3/s, so we convert at the end.
    sigma_v_cm3_s = 5.0e-12 * np.exp(-19.94 / T_keV**(1/3))
    
    # Convert from cm^3/s to m^3/s by dividing by 1e6
    sigma_v_m3_s = sigma_v_cm3_s / 1e6
    
    return sigma_v_m3_s


def calculate_reactivity(fuel: Fuel, T_i: u.Quantity) -> u.Quantity[u.m**3 / u.s]:
    """
    User-facing wrapper for reactivity calculations. Handles units and calls
    the appropriate JIT-compiled kernel based on the fuel type.
    """
    T_keV = T_i.to_value(u.keV)

    if fuel.name == 'D-T':
        reactivity_val = _reactivity_dt_jit(T_keV)
        return reactivity_val * u.m**3 / u.s
    elif fuel.name == 'D-He3':
        # Analytical fit from NRL Plasma Formulary (2019), page 46.
        A5 = 2.52e-10
        B5 = 85.3
        sigma_v_cm3_s = A5 * T_keV**(-2/3) * np.exp(-B5 / T_keV**(1/3))
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
    else:
        # The other fuel reactivities can be implemented here using similar simple fits.
        raise NotImplementedError(f"Reactivity calculation for {fuel.name} is not yet implemented.")


def calculate_fusion_power(
    n_i: u.Quantity, T_i: u.Quantity, V: u.Quantity,
    fuel: Fuel, ratio: float
) -> tuple[u.Quantity, u.Quantity]:
    """Calculates total fusion power by calling the main reactivity function."""
    sigma_v = calculate_reactivity(fuel, T_i)
    n1, n2 = n_i * ratio, n_i * (1 - ratio)
    factor = 0.5 if fuel.reactants[0] == fuel.reactants[1] else 1.0
    power_density = factor * n1 * n2 * sigma_v * fuel.energy_per_reaction
    p_fusion_total = (power_density * V).to(u.W)
    p_charged_particles = p_fusion_total * fuel.charged_particle_fraction
    return p_fusion_total, p_charged_particles 