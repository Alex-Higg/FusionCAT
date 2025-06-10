# fusioncat/utils/fuels.py
from dataclasses import dataclass
from typing import Sequence, Tuple
import astropy.units as u

@dataclass(frozen=True)
class Fuel:
    """Represents a fusion fuel with its properties."""
    name: str
    reactants: Sequence[str]
    energy_per_reaction: u.Quantity
    charged_particle_fraction: float
    alpha_heating_fractions: Tuple[float, float] = (0.5, 0.5)  # (ion fraction, electron fraction)
    citation: str = ""

# Define common fusion fuels
FUEL_DT = Fuel(
    name="D-T",
    reactants=["D", "T"],
    energy_per_reaction=17.6 * u.MeV,
    charged_particle_fraction=0.2,  # 3.5 MeV / 17.6 MeV
    alpha_heating_fractions=(0.5, 0.5),  # 50-50 split between ions and electrons
    citation="NRL Plasma Formulary (2019)"
)

FUEL_DD = Fuel(
    name="D-D",
    reactants=["D", "D"],
    energy_per_reaction=4.03 * u.MeV,
    charged_particle_fraction=0.66,  # 2.66 MeV / 4.03 MeV
    alpha_heating_fractions=(0.5, 0.5),
    citation="NRL Plasma Formulary (2019)"
)

FUEL_DHE3 = Fuel(
    name="D-He3",
    reactants=["D", "He3"],
    energy_per_reaction=18.3 * u.MeV,
    charged_particle_fraction=0.95,  # 17.4 MeV / 18.3 MeV
    alpha_heating_fractions=(0.5, 0.5),
    citation="NRL Plasma Formulary (2019)"
)

FUEL_PB11 = Fuel(
    name="p-B11",
    reactants=["p", "B11"],
    energy_per_reaction=8.7 * u.MeV,
    charged_particle_fraction=1.0,  # All energy in charged particles
    alpha_heating_fractions=(0.5, 0.5),
    citation="NRL Plasma Formulary (2019)"
) 