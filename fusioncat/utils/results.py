# fusioncat/utils/results.py
from dataclasses import dataclass
import astropy.units as u

@dataclass(frozen=True)
class ZeroDResults:
    """Holds results from a 0D analysis."""
    ion_temperature: u.Quantity
    electron_temperature: u.Quantity
    fusion_power: u.Quantity
    charged_particle_power: u.Quantity
    bremsstrahlung_power: u.Quantity
    synchrotron_power: u.Quantity
    ion_confinement_loss: u.Quantity
    electron_confinement_loss: u.Quantity
    total_loss_power: u.Quantity
    required_heating_power: u.Quantity
    fusion_gain_q: float
    ion_electron_exchange_power: u.Quantity
    triple_product: u.Quantity

@dataclass(frozen=True)
class OneDResults:
    """Holds all profile results from a 1D analysis."""
    radius_grid: u.Quantity
    T_i_profile: u.Quantity
    T_e_profile: u.Quantity
    n_i_profile: u.Quantity
    n_e_profile: u.Quantity
    fusion_power_profile: u.Quantity
    bremsstrahlung_power_profile: u.Quantity
    ion_heat_flux_profile: u.Quantity # This should be transport power
    electron_heat_flux_profile: u.Quantity # This should be transport power
    ion_electron_exchange_profile: u.Quantity
    ion_heating_profile: u.Quantity
    electron_heating_profile: u.Quantity 