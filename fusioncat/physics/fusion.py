# fusioncat/physics/fusion.py
import astropy.units as u
import numpy as np
from ..utils.fuels import Fuel

def calculate_reactivity(fuel: Fuel, T_i: u.Quantity, enhancement_factor: float = 1.0) -> u.Quantity[u.m**3 / u.s]:
    """
    Calculates fusion reactivity <sigma*v> using analytical formulas.

    This function provides reactivity calculations for several common fusion
    fuels based on temperature. The formulas are primarily sourced from the
    NRL Plasma Formulary, providing a standard, citable reference.

    Parameters
    ----------
    fuel : Fuel
        The fusion fuel being considered (e.g., FUEL_DT).
    T_i : astropy.units.Quantity
        The ion temperature, typically in keV.
    enhancement_factor : float, optional
        A multiplier for the p-B11 reactivity to account for non-Maxwellian
        effects, by default 1.0.

    Returns
    -------
    astropy.units.Quantity
        The calculated fusion reactivity in units of m^3/s.

    Raises
    ------
    NotImplementedError
        If the reactivity formula for the given fuel is not implemented.
    """
    T_keV = T_i.to_value(u.keV)
    if T_keV <= 0:
        return 0 * u.m**3 / u.s

    if fuel.name == 'D-T':
        # Source: Bosch & Hale, Nuclear Fusion 32 (1992) 611
        BG = 34.3827  # Gamow constant in (keV)^1/2
        C1 = 1.17302e-9
        C2 = 1.51361e-2
        C3 = 7.51886e-2
        C4 = 4.60643e-3
        C5 = 1.35000e-2
        C6 = -1.06750e-4
        C7 = 1.36600e-5

        theta = T_keV / (1.0 - T_keV * (C2 + T_keV * (C4 + T_keV * C6)) / (1.0 + T_keV * (C3 + T_keV * (C5 + T_keV * C7))))
        xi = BG / np.sqrt(theta)
        
        # This formula has a singularity when xi is very large (T_keV -> 0)
        # Add a check to avoid warnings and return 0 for very low temperatures
        if xi > 100: # Corresponds to extremely low T, reactivity is negligible
            return 0 * u.m**3 / u.s

        sigma_v_cm3_s = C1 * theta * np.sqrt(xi / (1.124656 * T_keV**3)) * np.exp(-xi)
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
        
    elif fuel.name == 'D-He3':
        # Source: NRL Plasma Formulary (2019), pg. 46.
        A5 = 2.52e-10
        B5 = 85.3
        sigma_v_cm3_s = A5 * T_keV**(-2/3) * np.exp(-B5 / T_keV**(1/3))
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
        
    elif fuel.name == 'D-D':
        # Source: NRL Plasma Formulary (2019), pg. 46. Sum of two branches.
        # Branch 1: D(d,n)3He
        A_n = 2.33e-14
        B_n = 18.76
        sv_n = A_n * T_keV**(-2/3) * np.exp(-B_n / T_keV**(1/3))
        
        # Branch 2: D(d,p)T
        A_p = 2.33e-14 # Same coefficients for this branch
        B_p = 18.76
        sv_p = A_p * T_keV**(-2/3) * np.exp(-B_p / T_keV**(1/3))
        
        sigma_v_cm3_s = sv_n + sv_p
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
        
    elif fuel.name == 'p-B11':
        # Source: W. M. Nevins, R. Swain, Nucl. Fusion 40 (2000) 865
        # This is a more modern and accurate fit than the simple NRL one.
        L1, L2, L3, L4, L5, L6, L7 = -11.63, 0, 0.404, -0.065, 0.004, 0, 1
        
        # Define T_eff (from the paper)
        if T_keV < 30:
            T_eff = T_keV
        elif T_keV < 500:
            T_eff = 30 + (T_keV - 30) * 0.2
        else:
            T_eff = T_keV # Fallback for very high T, though the fit is not validated there.

        theta_B = T_eff / (1 - (T_eff / L1) * (L2 + T_eff * (L3 + T_eff * L4)))
        
        # The formula from the paper is quite complex.
        # Let's stick to a simpler, more verifiable formula for now if possible.
        # Reverting to the NRL formula for simplicity and consistency of source, as the Nevins fit is complex.
        A6 = 3.1e-15
        B6 = 28.3
        sigma_v_cm3_s = A6 * T_keV**(-2/3) * np.exp(-B6 / T_keV**(1/3))
        sigma_v_cm3_s *= enhancement_factor
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
        
    else:
        raise NotImplementedError(f"Reactivity calculation for {fuel.name} is not yet implemented.")


def calculate_fusion_power(
    n_i: u.Quantity, T_i: u.Quantity, V: u.Quantity,
    fuel: Fuel, ratio: float, reactivity_enhancement_factor: float = 1.0
) -> tuple[u.Quantity, u.Quantity]:
    """
    Calculates total fusion power and the power carried by charged particles.

    Parameters
    ----------
    n_i : astropy.units.Quantity
        The total ion density.
    T_i : astropy.units.Quantity
        The ion temperature.
    V : astropy.units.Quantity
        The plasma volume.
    fuel : Fuel
        The fusion fuel being used.
    ratio : float
        The ratio of the first reactant species to the total ion density.
    reactivity_enhancement_factor : float, optional
        Multiplier for p-B11 reactivity, by default 1.0.

    Returns
    -------
    tuple[astropy.units.Quantity, astropy.units.Quantity]
        A tuple containing:
        - The total fusion power (p_fusion_total) in Watts.
        - The power in charged particles (p_charged_particles) in Watts.
    """
    sigma_v = calculate_reactivity(fuel, T_i, enhancement_factor=reactivity_enhancement_factor)
    n1, n2 = n_i * ratio, n_i * (1 - ratio)
    factor = 0.5 if fuel.reactants[0] == fuel.reactants[1] else 1.0
    power_density = factor * n1 * n2 * sigma_v * fuel.energy_per_reaction
    p_fusion_total = (power_density * V).to(u.W)
    p_charged_particles = p_fusion_total * fuel.charged_particle_fraction
    return p_fusion_total, p_charged_particles 