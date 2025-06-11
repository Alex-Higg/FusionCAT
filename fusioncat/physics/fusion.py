# fusioncat/physics/fusion.py
import astropy.units as u
import numpy as np
from ..utils.fuels import Fuel

def calculate_reactivity(fuel: Fuel, T_i: u.Quantity, reactivity_enhancement_factor: float = 1.0) -> u.Quantity:
    T_keV = T_i.to_value(u.keV)
    if T_keV <= 0: return 0 * u.m**3 / u.s
    
    sigma_v_cm3_s = 0.0

    if fuel.name == 'D-T':
        # Formula from H.-S. Bosch, G.M. Hale, Nuclear Fusion, Vol. 32, No. 4 (1992)
        BG = 34.3827; C = [1.17302E-9, 1.51361E-2, 7.51886E-2, 4.60643E-3, 1.35E-2, -1.06750E-4, 1.366E-6]
        theta = T_keV / (1. - T_keV * (C[2] + T_keV * (C[4] + T_keV * C[6])) / (1. + T_keV * (C[3] + T_keV * C[5])))
        if theta <= 0: return 0 * u.m**3 / u.s
        xi = BG / (theta**0.5)
        sigma_v_cm3_s = C[0] * theta**(-2/3) * np.exp(-xi)
    elif fuel.name == 'D-D':
        # Sum of analytical fits for both D-D branches, NRL Plasma Formulary (2019)
        A1, B1 = 5.36e-12, 65.85; A2, B2 = 5.62e-12, 64.63
        term1 = A1 * T_keV**(-2/3) * np.exp(-B1 / T_keV**(1/3)); term2 = A2 * T_keV**(-2/3) * np.exp(-B2 / T_keV**(1/3))
        sigma_v_cm3_s = term1 + term2
    elif fuel.name == 'D-He3':
        # Analytical fit from NRL Plasma Formulary (2019).
        A, B = 5.51e-10, 89.87
        sigma_v_cm3_s = A * T_keV**(-2/3) * np.exp(-B / T_keV**(1/3))
    elif fuel.name == 'p-B11':
        # Analytical fit from W. M. Nevins & R. Swain, Nuclear Fusion, Vol. 40, No. 4 (2000)
        T100 = T_keV / 100.0; B0, B1, B2, B3 = -10.4, -1.05, 3.84, -0.48
        C_vals = [8.95e-23, 1.37e-23, 6.42e-26]
        T_log = np.log(T_keV)
        sigma_v_m3_s = C_vals[0]*np.exp(B0+B1*T_log+B2*T_log**2+B3*T_log**3) + C_vals[1]*T_keV**(-0.75)*np.exp(-25.7/T_keV**0.25) + C_vals[2]
    else:
        raise NotImplementedError(f"Reactivity for {fuel.name} not implemented.")

    reactivity = (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
    if fuel.name == 'p-B11':
        reactivity *= reactivity_enhancement_factor
    return reactivity

def calculate_fusion_power(n_i, T_i, V, fuel, ratio, reactivity_enhancement_factor=1.0):
    sigma_v = calculate_reactivity(fuel, T_i, reactivity_enhancement_factor)
    n1, n2 = n_i * ratio, n_i * (1 - ratio)
    factor = 0.5 if fuel.reactants[0] == fuel.reactants[1] else 1.0
    effective_energy = ((4.03 + 3.27) / 2 * u.MeV).to(u.J) if fuel.name == 'D-D' else fuel.energy_per_reaction
    power_density = factor * n1 * n2 * sigma_v * effective_energy
    p_fusion_total = (power_density * V).to(u.W)
    p_charged_particles = p_fusion_total * fuel.charged_particle_fraction
    return p_fusion_total, p_charged_particles