# examples/phase3_1d_analysis.py
"""
An example script demonstrating the Phase 3 capabilities of FusionCAT:
a 1D, two-temperature, steady-state power balance analysis.
"""
import astropy.units as u
import numpy as np
from scipy.integrate import simpson

from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT
from fusioncat.plotting import plot_1d_profiles

def integrate_profile(r_grid: u.Quantity, profile: u.Quantity, major_radius: u.Quantity) -> u.Quantity:
    """
    Helper function to integrate a power density profile (W/m^3) over
    a toroidal volume to get total power (W). This function is now
    fully unit-aware.
    """
    # --- THIS IS THE CORRECTED SECTION ---
    # 1. Strip units to get plain NumPy arrays for SciPy
    y_values = profile.value
    x_values = r_grid.value
    
    # 2. Define the geometric part of the integrand, dV/dr = 4 * pi^2 * r * R
    geometric_factor = 4 * np.pi**2 * r_grid.value * major_radius.value
    
    # 3. Perform the numerical integration on plain numbers
    # The integrand is P(r) * dV/dr, so the integral over dr gives total P
    integral_value = simpson(y=(y_values * geometric_factor), x=x_values)
    
    # 4. Determine the final unit and re-attach it to the result
    # Unit of integral = unit(integrand) * unit(dr)
    # unit(integrand) = (W/m^3) * m * m = W/m
    # unit(dr) = m
    # Final unit = (W/m) * m = W
    integrand_unit = profile.unit * r_grid.unit * major_radius.unit
    result_unit = integrand_unit * r_grid.unit
    
    return integral_value * result_unit.decompose()
    # ------------------------------------


def main():
    print("--- Running 1D Power Balance Analysis for an Example Tokamak ---")
    
    reactor = FusionConcept(name="Example 1D D-T Reactor")
    reactor_params = {
        "ion_temperature": 25 * u.keV, "electron_temperature": 22 * u.keV,
        "ion_density": 1.5e20 / u.m**3, "fuel": FUEL_DT,
        "ion_diffusivity": 0.6 * u.m**2 / u.s, "electron_diffusivity": 1.0 * u.m**2 / u.s,
        "minor_radius": 2.0 * u.m, "major_radius": 6.2 * u.m,
        "volume": 830 * u.m**3, "magnetic_field": 5.3 * u.T,
    }
    reactor.set_parameters(**reactor_params)
    
    results = reactor.run_1d_analysis(num_points=101, T_alpha=2.0, n_alpha=0.5)

    # Calculate Integrated Results
    total_fusion_power = integrate_profile(results.radius_grid, results.fusion_power_profile, reactor_params['major_radius'])
    total_ion_heating = integrate_profile(results.radius_grid, results.ion_heating_profile, reactor_params['major_radius'])
    total_electron_heating = integrate_profile(results.radius_grid, results.electron_heating_profile, reactor_params['major_radius'])
    
    # This logic is now correct because integrate_profile returns the correct units
    required_ion_heating_val = max(0.0, total_ion_heating.to_value(u.W))
    required_ion_heating = required_ion_heating_val * u.W
    
    required_electron_heating_val = max(0.0, total_electron_heating.to_value(u.W))
    required_electron_heating = required_electron_heating_val * u.W

    total_required_heating = required_ion_heating + required_electron_heating
    
    q_value = total_fusion_power / total_required_heating if total_required_heating > 0 else float('inf')

    print("\n" + "="*60)
    print(f"Integrated 1D Analysis Results for: {reactor.name}")
    print("="*60)
    print(f"Total Fusion Power:           {total_fusion_power.to(u.MW):.2f}")
    print(f"Total Required External Heating: {total_required_heating.to(u.MW):.2f}")
    print(f"  --> Required Ion Heating:      {required_ion_heating.to(u.MW):.2f} (raw value: {total_ion_heating.to(u.MW):.2f})")
    print(f"  --> Required Electron Heating: {required_electron_heating.to(u.MW):.2f} (raw value: {total_electron_heating.to(u.MW):.2f})")
    print(f"Global Fusion Gain (Q):       {q_value:.2f}")
    print("="*60)

    print("Generating profile plots...")
    plot_1d_profiles(results, save_path="1d_profiles.png")


if __name__ == "__main__":
    main() 