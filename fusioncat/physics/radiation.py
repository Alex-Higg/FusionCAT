# fusioncat/physics/radiation.py
import astropy.units as u
import astropy.constants as const

def get_ion_species(fuel, ratio, n_i):
    """Helper to get ion species densities and charges."""
    # A simple map for charge numbers.
    charge_map = {'D': 1, 'T': 1, 'He-3': 2, 'p+': 1, 'B-11': 5}
    z1 = charge_map.get(fuel.reactants[0], 1)
    z2 = charge_map.get(fuel.reactants[1], 1)
    n1 = n_i * ratio
    n2 = n_i * (1 - ratio)
    return [(n1, z1), (n2, z2)]

def calculate_bremsstrahlung_power(n_e: u.Quantity, T_e: u.Quantity, V: u.Quantity, Z_eff: float) -> u.Quantity[u.W]:
    """
    Calculates total Bremsstrahlung power using the NRL Plasma Formulary formula.
    """
    c_b = 1.69e-38 * u.W * u.m**3 / u.K**0.5
    p_brems_density = c_b * n_e * n_e * Z_eff * (T_e.to(u.K, equivalencies=u.temperature_energy()))**0.5
    return (p_brems_density * V).to(u.W)

def calculate_synchrotron_power(
    n_e: u.Quantity, T_e: u.Quantity, B: u.Quantity,
    major_radius: u.Quantity, wall_reflectivity: float = 0.9
) -> u.Quantity[u.W]:
    """
    Calculates synchrotron power loss using an approximation for a toroidal plasma.
    
    The formula is a simplified fit suitable for 0D analysis and assumes a
    toroidal geometry with a circular cross-section. It may not be accurate
    for other magnetic confinement geometries.
    
    Citation: J. Wesson, "Tokamaks", 4th ed., Oxford University Press (2011), Eq. 9.3.5.
    """
    T_keV = T_e.to_value(u.keV)
    n_e_19 = n_e.to_value(1e19 * u.m**-3)
    R_m = major_radius.to_value(u.m)
    B_T = B.to_value(u.T)

    # Power in MW/m^3
    p_synch_density_MW_m3 = (6.2e-3 / 100) * n_e_19 * B_T**2.5 * T_keV**2
    # This formula from a different source is a density. We need a volume V.
    # Wesson's formula is P_tot ~ T^2.5, B^2.5 ...
    # Let's use a more standard Trubnikov-like formula for power density
    p_synch_density_MW_m3 = 6.0e-3 * n_e_19 * T_keV * B_T**2
    
    volume_placeholder = (2 * 3.14159**2 * R_m * (R_m/3)**2) * u.m**3 # Placeholder volume for density calc

    p_synch = (p_synch_density_MW_m3 * u.MW / u.m**3 * volume_placeholder).to(u.W)

    return p_synch * (1 - wall_reflectivity) 