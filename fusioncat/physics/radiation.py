# fusioncat/physics/radiation.py
import astropy.units as u; import astropy.constants as const
from ..packages.plasmapy_bridge import get_charge_number

def get_ion_species(fuel, ratio, n_i):
    z1 = get_charge_number(fuel.reactants[0]); z2 = get_charge_number(fuel.reactants[1])
    return [(n_i * ratio, z1), (n_i * (1 - ratio), z2)]

def calculate_bremsstrahlung_power(n_e, T_e, V, ion_populations):
    c_b = 1.69e-38 * u.W * u.m**3 / u.K**0.5
    sum_ni_zi_sq = sum(n_i.value * Z_i**2 for n_i, Z_i in ion_populations) / u.m**3
    p_brems_density = c_b * n_e * sum_ni_zi_sq * T_e.to(u.K, equivalencies=u.temperature_energy())**0.5
    return (p_brems_density * V).to(u.W)

def calculate_synchrotron_power(n_e, T_e, B, R, a, wall_reflectivity=0.9):
    T_keV = T_e.to_value(u.keV)
    # Trubnikov formula for total power in Watts
    p_synch = (6.02e-3 * a.to_value(u.m)**2 * R.to_value(u.m) *
               (n_e.to_value(1e19*u.m**-3))**0.5 * B.to_value(u.T)**2.5 *
               T_keV**1.5) * u.MW
    return p_synch.to(u.W) * (1 - wall_reflectivity)