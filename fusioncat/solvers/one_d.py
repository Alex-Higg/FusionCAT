# fusioncat/solvers/one_d.py
"""
The 1D steady-state, two-temperature power balance solver.
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
from numba import jit

# Import the JIT-compatible physics kernels
from ..physics.numerics import gradient_1d
from ..physics.fusion import _reactivity_dt_jit

from ..utils.fuels import Fuel

@jit(nopython=True, cache=True)
def _solve_1d_balance_jit(
    r_grid: np.ndarray, T_i_profile_K: np.ndarray, T_e_profile_K: np.ndarray,
    n_i_profile: np.ndarray, chi_i: float, chi_e: float, fuel_name_is_dt: bool,
    fuel_energy: float, charged_fraction: float, alpha_frac_ions: float
) -> tuple:
    """
    Numba-JIT compiled kernel for the 1D power balance calculation.
    This now calls the central reactivity kernel for its physics.
    """
    num_points = len(r_grid)
    dr = r_grid[1] - r_grid[0] if num_points > 1 else 1.0
    
    # Output arrays
    P_fusion_profile = np.zeros(num_points)
    P_brems_profile = np.zeros(num_points)
    P_ie_exchange_profile = np.zeros(num_points)
    
    k_B = 1.380649e-23
    e_charge = 1.60217663e-19
    C_B = 1.69e-38

    for i in range(num_points):
        Ti_K, Te_K, ni = T_i_profile_K[i], T_e_profile_K[i], n_i_profile[i]
        ne = ni

        # --- FUSION POWER ---
        # Call the single, verified reactivity kernel
        if fuel_name_is_dt:
            Ti_keV = Ti_K * k_B / e_charge / 1000.0
            sigma_v = _reactivity_dt_jit(Ti_keV)
            P_fusion_profile[i] = 0.5 * ni * ni * sigma_v * fuel_energy
        
        # --- BREMSSTRAHLUNG & EXCHANGE ---
        P_brems_profile[i] = C_B * ne * ni * Te_K**0.5
        log_lambda = 24 - 0.5 * np.log(ne) + 1.5 * np.log(Te_K)
        p_ie_nrl = (1.7e-40 * ne*ni*log_lambda*(Ti_K-Te_K)) / (2.5 * Te_K**1.5)
        P_ie_exchange_profile[i] = p_ie_nrl

    # --- TRANSPORT ---
    grad_Ti = gradient_1d(T_i_profile_K, dr)
    grad_Te = gradient_1d(T_e_profile_K, dr)
    q_i = -n_i_profile * chi_i * grad_Ti * k_B
    q_e = -n_i_profile * chi_e * grad_Te * k_B # Use n_i for both as per some models
    
    # Handle r=0 case safely for divergence calculation
    rq_i = r_grid * q_i
    rq_e = r_grid * q_e
    P_transport_ions = gradient_1d(rq_i, dr) / r_grid
    P_transport_electrons = gradient_1d(rq_e, dr) / r_grid
    if r_grid[0] == 0:
        P_transport_ions[0] = 2 * q_i[0] / dr if dr > 0 else 0.0
        P_transport_electrons[0] = 2 * q_e[0] / dr if dr > 0 else 0.0

    # --- POWER BALANCE ---
    f_alpha_i, f_alpha_e = 0.2, 0.8 # Simple partition for D-T
    p_alpha_i = P_fusion_profile * charged_fraction * f_alpha_i
    p_alpha_e = P_fusion_profile * charged_fraction * f_alpha_e
    
    P_heat_ions_profile = P_transport_ions + P_ie_exchange_profile - p_alpha_i
    P_heat_electrons_profile = P_transport_electrons + P_brems_profile - P_ie_exchange_profile - p_alpha_e

    return (
        P_fusion_profile, P_brems_profile, P_ie_exchange_profile,
        P_transport_ions, P_transport_electrons,
        P_heat_ions_profile, P_heat_electrons_profile
    )


def solve_steady_state_1d(
    radius_grid: u.Quantity, T_i_profile: u.Quantity, T_e_profile: u.Quantity,
    n_i_profile: u.Quantity, chi_i: u.Quantity, chi_e: u.Quantity, fuel: Fuel
):
    """
    User-facing wrapper for the 1D solver. Handles unit conversions.
    """
    r_grid_si = radius_grid.to_value(u.m)
    Ti_K_si = T_i_profile.to_value(u.K, equivalencies=u.temperature_energy())
    Te_K_si = T_e_profile.to_value(u.K, equivalencies=u.temperature_energy())
    ni_si = n_i_profile.to_value(u.m**-3)
    chi_i_si = chi_i.to_value(u.m**2 / u.s)
    chi_e_si = chi_e.to_value(u.m**2 / u.s)
    fuel_energy_si = fuel.energy_per_reaction.to_value(u.J)

    profiles = _solve_1d_balance_jit(
        r_grid_si, Ti_K_si, Te_K_si, ni_si,
        chi_i_si, chi_e_si, fuel.name == 'D-T', fuel_energy_si,
        fuel.charged_particle_fraction
    )
    
    power_density_unit = u.W / u.m**3
    results_with_units = [p * power_density_unit for p in profiles]
    
    return tuple(results_with_units) 