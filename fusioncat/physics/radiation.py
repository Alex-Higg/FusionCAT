# fusioncat/physics/radiation.py
import astropy.units as u
import astropy.constants as const

def get_ion_species(charge_map: dict[str, float]) -> tuple[list[float], list[float]]:
    """
    Helper function to get ion species densities and charges from a charge map.
    """
    densities = []
    charges = []
    for species, fraction in charge_map.items():
        if species != 'e-':  # Skip electrons
            densities.append(fraction)
            charges.append(1.0)  # Assuming singly charged ions
    return densities, charges

def calculate_bremsstrahlung_power(
    n_e: u.Quantity, T_e: u.Quantity, charge_map: dict[str, float]
) -> u.Quantity[u.W]:
    """
    Calculates total Bremsstrahlung power using the NRL Plasma Formulary formula.
    Citation: NRL Plasma Formulary (2019), pg. 60.
    """
    T_e_keV = T_e.to_value(u.keV)
    n_e_19 = n_e.to_value(1e19 * u.m**-3)
    
    # Get ion species information
    ion_densities, ion_charges = get_ion_species(charge_map)
    
    # Sum over all ion species
    Z_eff = sum(n * Z**2 for n, Z in zip(ion_densities, ion_charges))
    
    # Power density in W/m^3
    p_br_density = 5.35e-37 * n_e_19**2 * Z_eff * T_e_keV**0.5 * u.W / u.m**3
    
    return p_br_density

def calculate_synchrotron_power(
    n_e: u.Quantity, T_e: u.Quantity, B: u.Quantity,
    major_radius: u.Quantity, minor_radius: u.Quantity,
    wall_reflectivity: float = 0.9
) -> u.Quantity[u.W]:
    """
    Calculates synchrotron power loss using the Trubnikov formula for a torus.
    Citation: B.A. Trubnikov, in Reviews of Plasma Physics, Vol. 7 (1979).
    This is a standard formula for synchrotron emission in a tokamak-like geometry.
    """
    T_e_keV = T_e.to_value(u.keV)
    n_e_19 = n_e.to_value(1e19 * u.m**-3)
    a_m = minor_radius.to_value(u.m)
    B_T = B.to_value(u.T)

    # Formula gives power density in W/m^3
    p_synch_density = (
        6.14e3 * n_e_19 * B_T**2 * T_e_keV * (1 + T_e_keV / 204)
    ) * u.W / u.m**3

    # Total power is density * volume
    volume = 2 * (3.14159)**2 * major_radius * minor_radius**2
    p_synch_total = p_synch_density * volume

    return p_synch_total.to(u.W) * (1 - wall_reflectivity) 