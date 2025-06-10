"""
This module acts as a dedicated "bridge" to the PlasmaPy library.
It is the ONLY file in our project that should import directly from plasmapy.
This isolates the dependency, making future updates much easier to manage.
"""
from plasmapy.particles.nuclear import nuclear_reaction_energy
from plasmapy.particles import charge_number

# Re-export the stable functions we need under our own namespace.
get_nuclear_reaction_energy = nuclear_reaction_energy
get_charge_number = charge_number 