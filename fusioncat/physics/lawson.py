# fusioncat/physics/lawson.py
import astropy.units as u; import numpy as np
from ..packages.plasmapy_bridge import get_particle_mass

def calculate_triple_product(n, T, tau_E):
    return (n * T * tau_E).to(u.s * u.m**-3 * u.keV)

def calculate_coulomb_logarithm(n_e, T_e):
    if n_e == 0 or T_e == 0: return 0.0
    lambda_de = 743.4 * (T_e.to_value(u.eV)**0.5) / (n_e.to_value(u.m**-3)**0.5)
    return np.log(lambda_de)

def calculate_ion_electron_exchange(ion_populations, T_i, n_e, T_e, V, fuel):
    log_lambda_ie = calculate_coulomb_logarithm(n_e, T_e)
    sum_term = 0.0 / (u.s * u.m**3)
    # This loop correctly sums the contribution from each ion species
    for (n_s, Z_s), reactant_name in zip(ion_populations, fuel.reactants):
        m_s = get_particle_mass(reactant_name)
        sum_term += n_s * Z_s**2 / m_s
    
    prefactor = 1.8e-19 * (u.J * u.m**3 / u.s)
    power_density = prefactor * n_e * sum_term * log_lambda_ie * (T_i - T_e) / T_e**1.5
    return (power_density.to(u.W / u.m**3) * V).to(u.W)