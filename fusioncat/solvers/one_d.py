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
from ..physics.fusion import _reactivity_dt_jit, _reactivity_dd_jit, _reactivity_dhe3_jit, _reactivity_pb11_jit

from ..utils.fuels import Fuel

@jit(nopython=True, cache=True)
def _solve_1d_balance_jit(
    r_grid, T_i_profile_K, T_e_profile_K, n_i_profile, chi_i, chi_e,
    fuel_name, fuel_energy, charged_fraction, alpha_frac_ions
) -> tuple:
    """
    Numba-JIT compiled kernel for the 1D power balance calculation.
    This now calls the central reactivity kernel for its physics.
    """
    num_points = len(r_grid)
    dr = r_grid[1] - r_grid[0] if num_points > 1 else 1.0
    
    # Output arrays
    P_fusion = np.zeros(num_points)
    P_brems = np.zeros(num_points)
    P_ie = np.zeros(num_points)
    
    k_B = 1.380649e-23
    e_charge = 1.60217663e-19
    C_B = 1.69e-38

    for i in range(num_points):
        Ti_K, Te_K, ni = T_i_profile_K[i], T_e_profile_K[i], n_i_profile[i]
        ne = ni
        Ti_keV = Ti_K * k_B / e_charge / 1000.0
        
        sigma_v = 0.0
        if fuel_name == 1: sigma_v = _reactivity_dt_jit(Ti_keV)
        elif fuel_name == 2: sigma_v = _reactivity_dd_jit(Ti_keV)
        elif fuel_name == 3: sigma_v = _reactivity_dhe3_jit(Ti_keV)
        elif fuel_name == 4: sigma_v = _reactivity_pb11_jit(Ti_keV)
        
        P_fusion[i] = 0.5 * ni * ni * sigma_v * fuel_energy
        P_brems[i] = C_B * ne * ni * Te_K**0.5
        log_lambda = 24 - 0.5 * np.log(ne) + 1.5 * np.log(Te_K)
        P_ie[i] = (1.7e-40*ne*ni*log_lambda*(Ti_K-Te_K))/(2.5*Te_K**1.5)

    # --- TRANSPORT ---
    grad_Ti = gradient_1d(T_i_profile_K, dr)
    grad_Te = gradient_1d(T_e_profile_K, dr)
    q_i = -n_i_profile * chi_i * grad_Ti * k_B
    q_e = -n_i_profile * chi_e * grad_Te * k_B # Use n_i for both as per some models
    
    # Handle r=0 case safely for divergence calculation
    r_safe = r_grid.copy()
    r_safe[0] = 1e-9
    P_trans_i = gradient_1d(r_safe * q_i, dr) / r_safe
    P_trans_e = gradient_1d(r_safe * q_e, dr) / r_safe
    if r_grid[0] == 0:
        P_trans_i[0] = 2 * q_i[0] / dr if dr > 0 else 0.0
        P_trans_e[0] = 2 * q_e[0] / dr if dr > 0 else 0.0

    # --- POWER BALANCE ---
    p_alpha = P_fusion * charged_fraction
    p_alpha_i = p_alpha * alpha_frac_ions
    p_alpha_e = p_alpha * (1.0 - alpha_frac_ions)
    
    P_heat_i = P_trans_i + P_ie - p_alpha_i
    P_heat_e = P_trans_e + P_brems - P_ie - p_alpha_e

    return (
        P_fusion, P_brems, P_ie,
        P_trans_i, P_trans_e,
        P_heat_i, P_heat_e
    )


def solve_steady_state_1d(
    radius_grid: u.Quantity, T_i_profile: u.Quantity, T_e_profile: u.Quantity,
    n_i_profile: u.Quantity, chi_i: u.Quantity, chi_e: u.Quantity, fuel: Fuel,
    alpha_frac_ions: float
):
    """
    User-facing wrapper for the 1D solver. Handles unit conversions.
    """
    fuel_map = {'D-T': 1, 'D-D': 2, 'D-He3': 3, 'p-B11': 4}
    profiles = _solve_1d_balance_jit(
        radius_grid.to_value(u.m), T_i_profile.to_value(u.K, equivalencies=u.temperature_energy()),
        T_e_profile.to_value(u.K, equivalencies=u.temperature_energy()), n_i_profile.to_value(u.m**-3),
        chi_i.to_value(u.m**2 / u.s), chi_e.to_value(u.m**2 / u.s),
        fuel_map.get(fuel.name, -1), fuel.energy_per_reaction.to_value(u.J),
        fuel.charged_particle_fraction, alpha_frac_ions
    )
    
    power_density_unit = u.W / u.m**3
    results_with_units = [p * power_density_unit for p in profiles]
    
    return tuple(results_with_units) 