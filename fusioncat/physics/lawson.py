# fusioncat/physics/lawson.py
import astropy.units as u
import astropy.constants as const
import numpy as np
from ..utils.fuels import Fuel

def calculate_triple_product(n: u.Quantity, T: u.Quantity, tau_E: u.Quantity) -> u.Quantity:
    """Calculates the fusion triple product (n * T * tau_E)."""
    return (n * T * tau_E).to(u.m**-3 * u.keV * u.s)

def calculate_coulomb_logarithm(n_e: u.Quantity, T_e: u.Quantity) -> float:
    """Calculates the electron-ion Coulomb logarithm."""
    # Using a common approximation from NRL Formulary
    lambda_de = 743.4 * (T_e.to_value(u.eV)**0.5) / (n_e.to_value(u.m**-3)**0.5)
    return np.log(lambda_de)

def calculate_ion_electron_exchange(
    n_i: u.Quantity, T_i: u.Quantity, n_e: u.Quantity, T_e: u.Quantity, V: u.Quantity, fuel: Fuel
) -> u.Quantity[u.W]:
    """
    Calculates the power transferred from ions to electrons via Coulomb collisions.
    Positive value means ions are heating electrons.
    Citation: NRL Plasma Formulary (2019), pg. 34.
    """
    # Using a simplified formula assuming one dominant ion species for mass
    ion_mass_number = 2.5 # Average for D-T
    ion_charge_number = 1.0

    log_lambda_ie = calculate_coulomb_logarithm(n_e, T_e)
    
    # Power density in W/m^3
    T_i_K = T_i.to_value(u.K, equivalencies=u.temperature_energy())
    T_e_K = T_e.to_value(u.K, equivalencies=u.temperature_energy())
    p_ie_density = (1.7e-40 * n_e.to_value(u.m**-3) * n_i.to_value(u.m**-3) * ion_charge_number**2 * log_lambda_ie * (T_e_K - T_i_K) /
                    (ion_mass_number * T_e_K**1.5)
                   ) * u.W / u.m**3
    # Total power is density * volume
    p_ie_total = p_ie_density * V.to(u.m**3)
    return p_ie_total.to(u.W) 