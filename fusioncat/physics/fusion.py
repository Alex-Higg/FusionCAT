# fusioncat/physics/fusion.py
import astropy.units as u
import numpy as np
from numba import jit
from ..utils.fuels import Fuel

@jit(nopython=True, cache=True)
def _reactivity_dt_jit(T_keV: float) -> float:
    if T_keV < 0.2: return 0.0
    BG = 34.3827
    C = np.array([1.17302E-9, 1.51361E-2, 7.51886E-2, 4.60643E-3, 1.35E-2, -1.06750E-4, 1.366E-6])
    theta = T_keV / (1.0 - T_keV * (C[2] + T_keV * (C[4] + T_keV * C[6])) / (1.0 + T_keV * (C[3] + T_keV * C[5])))
    if theta <= 0: return 0.0
    xi = BG / (theta**0.5)
    val = C[0] * theta**(-2/3) * np.exp(-xi)
    return val / 1e6 # cm^3/s to m^3/s

@jit(nopython=True, cache=True)
def _reactivity_dd_jit(T_keV: float) -> float:
    if T_keV <= 0: return 0.0
    A = np.array([5.36e-12, 5.62e-12]); B = np.array([65.85, 64.63])
    term1 = A[0] * T_keV**(-2/3) * np.exp(-B[0] / T_keV**(1/3))
    term2 = A[1] * T_keV**(-2/3) * np.exp(-B[1] / T_keV**(1/3))
    return (term1 + term2) / 1e6

@jit(nopython=True, cache=True)
def _reactivity_dhe3_jit(T_keV: float) -> float:
    if T_keV <= 0: return 0.0
    A, B = 5.51e-10, 89.87
    val = A * T_keV**(-2/3) * np.exp(-B / T_keV**(1/3))
    return val / 1e6

@jit(nopython=True, cache=True)
def _reactivity_pb11_jit(T_keV: float) -> float:
    if T_keV <= 10.0: return 0.0
    T100 = T_keV / 100.0
    term1 = 4.86e-23 / (T_keV**0.5 * T100**0.5) * np.exp(-14.5 / T100**0.5)
    term2 = 8.75e-24 / (1 + (T_keV/50.0)**2)
    return term1 + term2 # Already in m^3/s

def calculate_reactivity(fuel: Fuel, T_i: u.Quantity, reactivity_enhancement_factor: float = 1.0) -> u.Quantity:
    T_keV = T_i.to_value(u.keV)
    val = 0.0
    if fuel.name == 'D-T': val = _reactivity_dt_jit(T_keV)
    elif fuel.name == 'D-D': val = _reactivity_dd_jit(T_keV)
    elif fuel.name == 'D-He3': val = _reactivity_dhe3_jit(T_keV)
    elif fuel.name == 'p-B11': val = _reactivity_pb11_jit(T_keV) * reactivity_enhancement_factor
    else: raise NotImplementedError(f"Reactivity for {fuel.name} not implemented.")
    return val * u.m**3 / u.s

def calculate_fusion_power(n_i, T_i, V, fuel, ratio, reactivity_enhancement_factor=1.0):
    sigma_v = calculate_reactivity(fuel, T_i, reactivity_enhancement_factor)
    n1, n2 = n_i * ratio, n_i * (1 - ratio)
    factor = 0.5 if fuel.reactants[0] == fuel.reactants[1] else 1.0
    power_density = factor * n1 * n2 * sigma_v * fuel.energy_per_reaction
    p_fusion_total = (power_density * V).to(u.W)
    p_charged_particles = p_fusion_total * fuel.charged_particle_fraction
    return p_fusion_total, p_charged_particles