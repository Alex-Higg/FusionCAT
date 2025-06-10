# fusioncat/utils/fuels.py
"""
Defines structured data for different fusion fuels and dynamically calculates
reaction energies using the PlasmaPy API.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence
import astropy.units as u
from ..packages.plasmapy_bridge import get_nuclear_reaction_energy

@dataclass(frozen=True)
class Fuel:
    """
    Represents a fusion fuel cycle with its reactants, products, and properties.
    """
    name: str
    reactants: Sequence[str]
    products: Sequence[str]
    charged_particle_fraction: float
    citation: str
    energy_per_reaction: u.Quantity[u.J] = field(init=False)

    def __post_init__(self):
        """Calculate reaction energy if not provided."""
        try:
            energy = get_nuclear_reaction_energy(reactants=self.reactants, products=self.products)
            object.__setattr__(self, 'energy_per_reaction', energy.to(u.J))
        except Exception as e:
            # Provide a fallback for reactions PlasmaPy might not know, like D-D avg.
            if self.name == 'D-D':
                object.__setattr__(self, 'energy_per_reaction', (3.65 * u.MeV).to(u.J))
            else:
                raise e

NRL_FORMULARY_CITATION = "J.D. Huba, NRL Plasma Formulary (2019)."

# Define standard fuel cycles
FUEL_DT = Fuel(
    name='D-T', reactants=('D+', 'T+'), products=('alpha', 'n'),
    charged_particle_fraction=3.52 / 17.59, citation=NRL_FORMULARY_CITATION
)

FUEL_DD = Fuel(
    name='D-D', reactants=('D+', 'D+'), products=('T+', 'p+'),
    charged_particle_fraction=0.66, citation=f"Branching ratios from {NRL_FORMULARY_CITATION}"
)

FUEL_DHE3 = Fuel(
    name='D-He3', reactants=('D+', 'He-3 2+'), products=('alpha', 'p+'),
    charged_particle_fraction=1.0, citation=NRL_FORMULARY_CITATION
)

FUEL_PB11 = Fuel(
    name='p-B11', reactants=('p+', 'B-11 5+'), products=('alpha', 'alpha', 'alpha'),
    charged_particle_fraction=1.0, citation="W. M. Nevins & R. Swain, Nuclear Fusion, Vol. 40, No. 4 (2000)"
) 