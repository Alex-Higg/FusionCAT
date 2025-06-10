# fusioncat/utils/fuels.py
"""
Defines structured data for different fusion fuels. This version dynamically
calculates reaction energies using the stable plasmapy.particles API.
"""
from dataclasses import dataclass, field
from typing import Sequence
import astropy.units as u

# This is a stable, verified import.
from plasmapy.particles.nuclear import nuclear_reaction_energy

@dataclass(frozen=True)
class Fuel:
    """Represents a fusion fuel cycle with cited physical constants."""
    name: str
    reactants: Sequence[str]
    products: Sequence[str]
    # This will be calculated automatically
    energy_per_reaction: u.Quantity[u.J] = field(init=False)
    charged_particle_fraction: float
    citation: str

    def __post_init__(self):
        """
        Calculates the reaction energy (Q-value) dynamically after the
        object is created, using PlasmaPy. This avoids hard-coding values.
        """
        energy = nuclear_reaction_energy(reactants=self.reactants, products=self.products)
        # The __setattr__ is needed because the dataclass is frozen.
        object.__setattr__(self, 'energy_per_reaction', energy.to(u.J))

# --- Definitions of Supported Fuel Cycles ---

NRL_FORMULARY_CITATION = "J.D. Huba, NRL Plasma Formulary (2019)."

FUEL_DT = Fuel(
    name='D-T',
    reactants=('D', 'T'),
    products=('alpha', 'n'),
    charged_particle_fraction=3.52 / 17.59,
    citation=NRL_FORMULARY_CITATION
)

FUEL_DD = Fuel(
    name='D-D',
    # We define it by the neutron branch; the reactivity formula will account for both.
    reactants=('D', 'D'),
    products=('He-3', 'n'),
    # Note: The effective energy and charged fraction for D-D are handled
    # directly in the reactivity and power calculation.
    charged_particle_fraction=0.66, # Effective fraction of total energy
    citation=f"Branching ratios from {NRL_FORMULARY_CITATION}"
)

FUEL_DHE3 = Fuel(
    name='D-He3',
    reactants=('D', 'He-3'),
    products=('alpha', 'p'),
    charged_particle_fraction=1.0,
    citation=NRL_FORMULARY_CITATION
)

FUEL_PB11 = Fuel(
    name='p-B11',
    reactants=('p+', 'B-11'),
    products=(f"3 alpha"),
    charged_particle_fraction=1.0,
    citation="W. M. Nevins & R. Swain, Nuclear Fusion, Vol. 40, No. 4 (2000)"
) 