import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT
from fusioncat.utils.results import ZeroDResults

# 1. Initialize the fusion concept
concept = FusionConcept(name="MyFirstTokamak")

# 2. Set the plasma and machine parameters
#    Note the use of astropy.units to ensure physical consistency.
concept.set_parameters(
    ion_temperature=20 * u.keV,
    electron_temperature=18 * u.keV,
    ion_density=1.5e20 / u.m**3,
    ion_confinement_time=3.0 * u.s,
    electron_confinement_time=2.8 * u.s,
    particle_confinement_time=5.0 * u.s, # Key parameter for ash buildup
    volume=520 * u.m**3,
    fuel=FUEL_DT,
    magnetic_field=5.3 * u.T,
    major_radius=3.1 * u.m,
    minor_radius=1.1 * u.m
)

# 3. Run the 0D analysis
results: ZeroDResults = concept.run_0d_analysis()

# 4. Print the key results in a user-friendly format
print(f"--- Results for {concept.name} ---")
print(f"Fusion Power: {results.fusion_power.to(u.MW):.2f}")
print(f"Required Heating Power: {results.required_heating_power.to(u.MW):.2f}")
print(f"Plasma Q: {results.fusion_gain_q:.2f}")
print("-" * 20)
print(f"Ash Fraction: {results.ash_fraction:.3f}")
print(f"Fuel Burn-up Fraction: {results.fuel_burnup_fraction:.3f}")
print("-" * 20)
print("Power Losses:")
print(f"  - Ion Transport: {results.ion_confinement_loss.to(u.MW):.2f}")
print(f"  - Electron Transport: {results.electron_confinement_loss.to(u.MW):.2f}")
print(f"  - Bremsstrahlung: {results.bremsstrahlung_power.to(u.MW):.2f}")
print(f"  - Synchrotron: {results.synchrotron_power.to(u.MW):.2f}") 