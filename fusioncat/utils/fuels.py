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
    ash_charge: float  # The charge number (Z) of the primary fusion product
    alpha_heating_fractions: Tuple[float, float] = (0.5, 0.5)  # (ion fraction, electron fraction)
    citation: str = ""

# Define common fusion fuels
FUEL_DT = Fuel(
    name="D-T",
    reactants=["D", "T"],
    energy_per_reaction=17.6 * u.MeV,
    charged_particle_fraction=0.2,  # 3.5 MeV / 17.6 MeV
    alpha_heating_fractions=(0.5, 0.5),  # 50-50 split between ions and electrons
    citation="NRL Plasma Formulary (2019)",
    ash_charge=2
)

# Deuterium-Deuterium (assumes 50/50 branching)
# Products are (T, p) and (3He, n), so charged energy is (4.03 + 0.82) MeV
# Total energy is (4.03+1.01) + (3.27+0.82) = 9.13 MeV. Whoops, that's not right.
# Total energy: 3.27 + 4.03 = 7.3 MeV avg. Charged: 0.82 + 1.01 = 1.83 MeV
# Total reaction energy is (3.27+4.03)/2 = 3.65 MeV
# Charged particle energy is (1.01 + 0.82) / 2 = 0.915 MeV
# This seems low. Let's re-evaluate based on standard values.
# D(d,p)T -> 4.03 MeV total, p gets 3.02, T gets 1.01
# D(d,n)3He -> 3.27 MeV total, 3He gets 0.82
# Average total E = 3.65 MeV. Average charged E = (3.02+0.82)/2 = 1.92 MeV.
# Let's use the total energy across both branches.
# Total energy is 4.03 MeV + 3.27 MeV = 7.3 MeV.
# The code using this fuel assumes a single reaction event.
# So we should average them.
total_E = (4.03 + 3.27) / 2.0 * u.MeV
charged_E = (3.02 + 0.82) / 2.0 * u.MeV

FUEL_DD = Fuel(
    name="D-D",
    reactants=["D", "D"],
    energy_per_reaction=total_E,
    charged_particle_fraction=(charged_E / total_E).to_value(u.dimensionless_unscaled),
    alpha_heating_fractions=(0.5, 0.5),
    citation="NRL Plasma Formulary (2019)",
    ash_charge=1.5  # Average of T (Z=1) and He3 (Z=2)
)

FUEL_DHE3 = Fuel(
    name="D-He3",
    reactants=["D", "He3"],
    energy_per_reaction=18.3 * u.MeV,
    charged_particle_fraction=0.95,  # 17.4 MeV / 18.3 MeV
    alpha_heating_fractions=(0.5, 0.5),
    citation="NRL Plasma Formulary (2019)",
    ash_charge=2  # Primary ash is He4 (alpha)
)

FUEL_PB11 = Fuel(
    name="p-B11",
    reactants=["p", "B11"],
    energy_per_reaction=8.7 * u.MeV,
    charged_particle_fraction=1.0,  # All energy in charged particles
    alpha_heating_fractions=(0.5, 0.5),
    citation="NRL Plasma Formulary (2019)",
    ash_charge=2  # All alphas
) 