# fusioncat/utils/fuels.py
from dataclasses import dataclass
import astropy.units as u

@dataclass(frozen=True)
class Fuel:
    name: str
    reactants: tuple[str, str]
    energy_per_reaction: u.Quantity[u.J]
    charged_particle_fraction: float
    citation: str

NRL_FORMULARY_CITATION = "J.D. Huba, NRL Plasma Formulary (2019)."

FUEL_DT = Fuel(
    name='D-T', reactants=('D', 'T'), energy_per_reaction=(17.59 * u.MeV).to(u.J),
    charged_particle_fraction=3.5 / 17.59, citation=NRL_FORMULARY_CITATION
)
FUEL_DD = Fuel(
    name='D-D', reactants=('D', 'D'), energy_per_reaction=(3.65 * u.MeV).to(u.J),
    charged_particle_fraction=2.425 / 3.65, citation=f"Avg. of two branches, {NRL_FORMULARY_CITATION}"
)
FUEL_DHE3 = Fuel(
    name='D-He3', reactants=('D', 'He-3'), energy_per_reaction=(18.35 * u.MeV).to(u.J),
    charged_particle_fraction=1.0, citation=NRL_FORMULARY_CITATION
)
FUEL_PB11 = Fuel(
    name='p-B11', reactants=('p+', 'B-11'), energy_per_reaction=(8.7 * u.MeV).to(u.J),
    charged_particle_fraction=1.0, citation="W. L. Reiter, AIP Conf. Proc. 1525, 29 (2013)."
) 