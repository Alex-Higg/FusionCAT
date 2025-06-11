# fusioncat/utils/results.py
from dataclasses import dataclass
import astropy.units as u

@dataclass(frozen=True)
class ZeroDResults:
    fusion_power: u.Quantity[u.W]
    charged_particle_power: u.Quantity[u.W]
    bremsstrahlung_power: u.Quantity[u.W]
    synchrotron_power: u.Quantity[u.W]
    ion_confinement_loss: u.Quantity[u.W]
    electron_confinement_loss: u.Quantity[u.W]
    total_loss_power: u.Quantity[u.W]
    required_heating_power: u.Quantity[u.W]
    fusion_gain_q: float
    triple_product: u.Quantity
    ion_temperature: u.Quantity[u.keV]
    electron_temperature: u.Quantity[u.keV]
    ion_electron_exchange_power: u.Quantity[u.W]
    ash_fraction: float
    fuel_burnup_fraction: float 