# fusioncat/physics/fusion.py
import astropy.units as u
import numpy as np
from ..utils.fuels import Fuel

def calculate_reactivity(
    fuel: Fuel,
    T_i: u.Quantity,
    reactivity_enhancement_factor: float = 1.0
) -> u.Quantity[u.m**3 / u.s]:
    """
    Calculates fusion reactivity <sigma*v> using cited analytical formulas.
    Includes an optional enhancement factor for p-B11 studies.
    """
    T_keV = T_i.to_value(u.keV)
    if T_keV <= 0:
        return 0 * u.m**3 / u.s

    # Select the formula based on the fuel name
    if fuel.name == 'D-T':
        # Formula from H.-S. Bosch, G.M. Hale, Nuclear Fusion, Vol. 32, No. 4 (1992)
        BG = 34.3827
        C = [1.17302E-9, 1.51361E-2, 7.51886E-2, 4.60643E-3, 1.35E-2, -1.06750E-4, 1.366E-6]
        theta = T_keV / (1 - T_keV * (C[2] + T_keV * (C[4] + T_keV * C[6])) / 
                         (1 + T_keV * (C[3] + T_keV * C[5])))
        if theta <= 0: return 0 * u.m**3 / u.s
        xi = BG / (theta**0.5)
        sigma_v_cm3_s = C[0] * theta**(-2/3) * np.exp(-xi)
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)

    elif fuel.name == 'D-D':
        # Analytical fit from NRL Plasma Formulary (2019), accounting for both branches.
        g1 = 5.666e-12 * T_keV**(-2/3) * np.exp(-68.75 / T_keV**(1/3))
        g2 = 5.857e-12 * T_keV**(-2/3) * np.exp(-67.82 / T_keV**(1/3))
        sigma_v_cm3_s = g1 + g2
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
        
    elif fuel.name == 'D-He3':
        # Analytical fit from NRL Plasma Formulary (2019).
        sigma_v_cm3_s = 2.52e-10 * T_keV**(-2/3) * np.exp(-85.3 / T_keV**(1/3))
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s)
        
    elif fuel.name == 'p-B11':
        # Analytical fit from Nevins & Swain, Nuclear Fusion (2000)
        T100 = T_keV / 100
        term1 = 3.1e-15 / (T_keV * T100**0.5) * np.exp(-0.2 * np.pi / T100**0.5)
        term2 = 8e-16 / (1 + (T_keV/50)**2)
        sigma_v_cm3_s = term1 + term2
        # Apply user-defined enhancement factor for non-thermal studies
        return (sigma_v_cm3_s * u.cm**3 / u.s).to(u.m**3 / u.s) * reactivity_enhancement_factor
    else:
        raise NotImplementedError(f"Reactivity calculation for {fuel.name} is not yet implemented.")

def calculate_fusion_power(
    n_i, T_i, V, fuel, ratio, reactivity_enhancement_factor=1.0
):
    """Calculates total fusion power and the power in charged particles."""
    sigma_v = calculate_reactivity(fuel, T_i, reactivity_enhancement_factor)
    n1, n2 = n_i * ratio, n_i * (1 - ratio)
    factor = 0.5 if fuel.reactants[0] == fuel.reactants[1] else 1.0
    
    # For D-D, we need to use the total effective energy of both branches
    if fuel.name == 'D-D':
        effective_energy = (4.03 + 3.27) * u.MeV / 2
    else:
        effective_energy = fuel.energy_per_reaction

    power_density = factor * n1 * n2 * sigma_v * effective_energy
    p_fusion_total = (power_density * V).to(u.W)
    p_charged_particles = p_fusion_total * fuel.charged_particle_fraction
    return p_fusion_total, p_charged_particles 