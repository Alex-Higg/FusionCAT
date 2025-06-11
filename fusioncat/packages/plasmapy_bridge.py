"""
This module is the ONLY file that imports from plasmapy. It provides a stable
internal interface, protecting our library from external API changes.
"""
from plasmapy.particles.nuclear import nuclear_reaction_energy
from plasmapy.particles import charge_number

# Re-export the functions we need
get_nuclear_reaction_energy = nuclear_reaction_energy
get_charge_number = charge_number 