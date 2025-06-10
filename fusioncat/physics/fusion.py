# fusioncat/physics/fusion.py
import astropy.units as u
import numpy as np
from ..utils.fuels import Fuel

def calculate_reactivity(fuel: Fuel, T_i: u.Quantity) -> u.Quantity[u.m**3 / u.s]:
    """
    Calculates fusion reactivity <sigma*v> using a simple, robust analytical formula.
    """
    T_keV = T_i.to_value(u.keV)

    if fuel.name == 'D-T':
        # This is the simplified analytical fit from the NRL Plasma Formulary (2019), page 46.
        # It is robust and accurate for the temperature range of interest.
        # <sigma*v> is in cm^3/s.
        sigma_v_cm3_s = 5.0e-12 * np.exp(-19.94 / T_keV**(1/3))
        
        # Convert to SI units (m^3/s) for consistency.
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
    elif fuel.name == 'D-D':
        # Analytical fit from NRL Plasma Formulary (2019), page 46, for the two D-D branches combined.
        # This gives the total D-D reactivity.
        A_values = [5.666e-12, 5.857e-12]
        B_values = [68.75, 67.82]
        sigma_v_cm3_s = A_values[0] * T_keV**(-2/3) * np.exp(-B_values[0] / T_keV**(1/3)) + \
                        A_values[1] * T_keV**(-2/3) * np.exp(-B_values[1] / T_keV**(1/3))
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
        
    elif fuel.name == 'D-He3':
        # Analytical fit from NRL Plasma Formulary (2019), page 46.
        A5 = 2.52e-10
        B5 = 85.3
        sigma_v_cm3_s = A5 * T_keV**(-2/3) * np.exp(-B5 / T_keV**(1/3))
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
        
    elif fuel.name == 'p-B11':
        # Analytical fit from W. M. Nevins & R. Swain, Nuclear Fusion, Vol. 40, No. 4 (2000)
        T_100 = T_keV / 100.0
        sigma_v_m3_s = (3.1e-21 * T_100**-0.5 * np.exp(-0.2*np.pi/T_100**0.5)) / (1 + (T_keV/50.0)**2)
        return sigma_v_m3_s * u.m**3 / u.s
    else:
        # A simple placeholder for other fuels for now to ensure they don't cause errors
        # This part should be updated with robust fits like the D-T one
        raise NotImplementedError(f"Reactivity calculation for {fuel.name} is not yet implemented.")

def calculate_fusion_power(n_i, T_i, V, fuel, ratio):
    """Calculates total fusion power and the power in charged particles."""
    sigma_v = calculate_reactivity(fuel, T_i)
    n1, n2 = n_i * ratio, n_i * (1 - ratio)
    factor = 0.5 if fuel.reactants[0] == fuel.reactants[1] else 1.0
    power_density = factor * n1 * n2 * sigma_v * fuel.energy_per_reaction
    p_fusion_total = (power_density * V).to(u.W)
    p_charged_particles = p_fusion_total * fuel.charged_particle_fraction
    return p_fusion_total, p_charged_particles 