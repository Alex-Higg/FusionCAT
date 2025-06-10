# fusioncat/utils/results.py
from dataclasses import dataclass
import astropy.units as u

@dataclass(frozen=True)
class ZeroDResults:
    fusion_power: u.Quantity[u.W]
    charged_particle_power: u.Quantity[u.W]
    bremsstrahlung_power: u.Quantity[u.W]
    synchrotron_power: u.Quantity[u.W]
    confinement_loss_power: u.Quantity[u.W]
    total_loss_power: u.Quantity[u.W]
    required_heating_power: u.Quantity[u.W]
    fusion_gain_q: float
    triple_product: u.Quantity 