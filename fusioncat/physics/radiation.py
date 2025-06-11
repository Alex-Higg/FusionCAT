# fusioncat/physics/radiation.py
import astropy.units as u; import astropy.constants as const
from ..packages.plasmapy_bridge import get_charge_number

def get_ion_species(fuel, ratio, n_i):
    """Helper to get ion species densities and charges."""
    z1 = get_charge_number(fuel.reactants[0]); z2 = get_charge_number(fuel.reactants[1])
    return [(n_i * ratio, z1), (n_i * (1 - ratio), z2)]

def calculate_bremsstrahlung_power(n_e, T_e, V, ion_populations):
    """Calculates total Bremsstrahlung power using the general formula."""
    c_b = 1.69e-38 * u.W * u.m**3 / u.K**0.5
    sum_ni_zi_sq = sum(n.value * z**2 for n, z in ion_populations) / u.m**3
    p_brems_density = c_b * n_e * sum_ni_zi_sq * T_e.to(u.K, equivalencies=u.temperature_energy())**0.5
    return (p_brems_density * V).to(u.W)

def calculate_synchrotron_power(n_e, T_e, B, R, a, wall_reflectivity=0.9):
    """Calculates synchrotron power using a standard formula for a torus."""
    T_keV = T_e.to_value(u.keV)
    p_synch_density_MW_m3 = 6.0e-3 * n_e.to_value(1e19*u.m**-3) * T_keV * B.to_value(u.T)**2
    volume = 2 * (3.14159)**2 * R * a**2
    return (p_synch_density_MW_m3 * u.MW / u.m**3 * volume).to(u.W) * (1 - wall_reflectivity) 