# fusioncat/physics/lawson.py
import astropy.units as u
import astropy.constants as const
import numpy as np
from ..utils.fuels import Fuel

def calculate_triple_product(n: u.Quantity, T: u.Quantity, tau_E: u.Quantity) -> u.Quantity:
    """
    Calculates the fusion triple product (n * T * tau_E).

    The triple product is a key figure of merit in fusion research,
    representing the conditions required for ignition.

    Parameters
    ----------
    n : astropy.units.Quantity
        The ion density.
    T : astropy.units.Quantity
        The ion temperature.
    tau_E : astropy.units.Quantity
        The energy confinement time.

    Returns
    -------
    astropy.units.Quantity
        The triple product in units of m^-3 * keV * s.
    """
    return (n * T * tau_E).to(u.m**-3 * u.keV * u.s)

def calculate_coulomb_logarithm(n_e: u.Quantity, T_e: u.Quantity) -> float:
    """
    Calculates the electron-ion Coulomb logarithm (log Lambda).

    This value is a factor in formulas for collisional processes in plasmas.

    Source
    ------
    A common approximation found in many plasma physics texts, including
    the NRL Plasma Formulary.

    Parameters
    ----------
    n_e : astropy.units.Quantity
        The electron density.
    T_e : astropy.units.Quantity
        The electron temperature.

    Returns
    -------
    float
        The dimensionless Coulomb logarithm.
    """
    # Using a common approximation from NRL Formulary
    lambda_de = 743.4 * (T_e.to_value(u.eV)**0.5) / (n_e.to_value(u.m**-3)**0.5)
    return np.log(lambda_de)

def calculate_ion_electron_exchange(
    n_i: u.Quantity, T_i: u.Quantity, n_e: u.Quantity, T_e: u.Quantity, V: u.Quantity, fuel: Fuel
) -> u.Quantity[u.W]:
    """
    Calculates power transferred from ions to electrons via Coulomb collisions.

    A positive value indicates net power transfer from ions to electrons (Ti > Te).

    Source
    ------
    NRL Plasma Formulary (2019), pg. 34. The implementation uses a simplified
    formula assuming a single dominant ion species for mass calculation.

    Parameters
    ----------
    n_i : astropy.units.Quantity
        The ion density.
    T_i : astropy.units.Quantity
        The ion temperature.
    n_e : astropy.units.Quantity
        The electron density.
    T_e : astropy.units.Quantity
        The electron temperature.
    V : astropy.units.Quantity
        The plasma volume.
    fuel : Fuel
        The fusion fuel, used to determine average ion mass.

    Returns
    -------
    astropy.units.Quantity
        The total power exchanged between ions and electrons in Watts.
    """
    # Using a simplified formula assuming one dominant ion species for mass
    ion_mass_number = 2.5 # Average for D-T
    ion_charge_number = 1.0

    log_lambda_ie = calculate_coulomb_logarithm(n_e, T_e)
    
    # This prefactor is based on the NRL formulary for SI units.
    prefactor = 4.8e-34 
    
    # Power density in W/m^3
    T_i_K = T_i.to_value(u.K, equivalencies=u.temperature_energy())
    T_e_K = T_e.to_value(u.K, equivalencies=u.temperature_energy())
    
    # Corrected term to (Ti - Te) so that Ti > Te yields a positive power transfer.
    p_ie_density = (prefactor * n_e.to_value(u.m**-3) * n_i.to_value(u.m**-3) * ion_charge_number**2 * log_lambda_ie * (T_i_K - T_e_K) /
                    (ion_mass_number * T_e_K**1.5)
                   ) * u.W / u.m**3
    # Total power is density * volume
    p_ie_total = p_ie_density * V.to(u.m**3)
    return p_ie_total.to(u.W) 