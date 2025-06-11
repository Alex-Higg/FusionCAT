# fusioncat/physics/lawson.py
import astropy.units as u
import numpy as np
from ..packages.plasmapy_bridge import get_particle_mass

def calculate_triple_product(n, T, tau_E):
    """Calculates the fusion triple product."""
    return (n * T * tau_E).to(u.s * u.m**-3 * u.keV)

def calculate_coulomb_logarithm(n_e, T_e):
    """Calculates the electron-ion Coulomb logarithm."""
    if n_e.value <= 0 or T_e.value <= 0: return 1.0  # Return a neutral value
    # Using a common approximation from NRL Plasma Formulary
    lambda_de = 7.434e8 * (T_e.to_value(u.eV)**0.5) / (n_e.to_value(u.m**-3)**0.5)
    return np.log(lambda_de) if lambda_de > 1 else 1.0

def calculate_ion_electron_exchange(ion_populations, T_i, n_e, T_e, V, fuel):
    """
    Calculates power transfer from ions to electrons via collisions.
    Positive value means ions are heating electrons.
    Citation: W.M. Stacey, "Fusion Plasma Physics", 4th ed., pg. 90, Eq. 4.102
    """
    log_lambda_ie = calculate_coulomb_logarithm(n_e, T_e)
    
    # Calculate the sum over ion species for the formula
    sum_term = 0.0 / (u.kg * u.m**3)
    for i, (n_s, Z_s) in enumerate(ion_populations):
        # Get ion mass from our bridge. This requires reactant names in the fuel object.
        m_s = get_particle_mass(fuel.reactants[i])
        sum_term += n_s * Z_s**2 / m_s
        
    # Prefactor C_ei from Stacey's book, in SI units
    C_ei = 1.8e-19 * u.J * u.m**3 / u.s
    
    # The formula is P_ie = C_ei * n_e * sum(n_j*Z_j^2/m_j) * ln(Lambda) * (T_i - T_e) / T_e^(3/2)
    # where temperatures are in Joules.
    T_i_J = T_i.to(u.J, equivalencies=u.temperature_energy())
    T_e_J = T_e.to(u.J, equivalencies=u.temperature_energy())

    if T_e_J.value == 0: return 0.0 * u.W

    power_density = C_ei * n_e * sum_term * log_lambda_ie * (T_i_J - T_e_J) / (T_e_J**1.5)
    
    return (power_density.to(u.W / u.m**3) * V).to(u.W)