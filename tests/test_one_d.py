# tests/test_one_d.py
import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT
from fusioncat.profiles.shapes import parabolic_profile
from fusioncat.plotting import plot_1d_profiles

@pytest.fixture
def configured_1d_concept():
    """
    Provides a fully configured FusionConcept object ready for 1D analysis.
    This fixture is used by multiple tests to avoid code duplication.
    """
    concept = FusionConcept(name="Test 1D Reactor")
    concept.set_parameters(
        # --- Core Physics Parameters ---
        ion_temperature=25 * u.keV,      # Peak (core) temperature
        electron_temperature=22 * u.keV,   # Peak (core) temperature
        ion_density=1.5e20 / u.m**3,     # Peak (core) density
        fuel=FUEL_DT,
        
        # --- Confinement Performance Parameters ---
        ion_diffusivity=0.5 * u.m**2 / u.s,
        electron_diffusivity=0.8 * u.m**2 / u.s,

        # --- Machine Geometry Parameters ---
        minor_radius=2.0 * u.m,
        major_radius=6.0 * u.m,
        volume=500 * u.m**3, # Note: Solver uses profiles, this is for 0D context
        magnetic_field=5.5 * u.T,
    )
    return concept

def test_parabolic_profile_shape():
    """Tests that the parabolic profile function behaves as expected."""
    r_grid = np.linspace(0, 1, 11)
    core_val = 10.0
    edge_val = 1.0
    
    profile = parabolic_profile(r_grid, core_val, edge_val, alpha=1.0)
    
    assert profile[0] == pytest.approx(core_val) # Check core value
    assert profile[-1] == pytest.approx(edge_val) # Check edge value
    assert profile[5] < core_val and profile[5] > edge_val # Check midpoint

def test_1d_solver_execution_and_shapes(configured_1d_concept):
    """
    This is a 'smoke test'. It ensures the 1D solver runs without crashing
    and that the output arrays have the correct dimensions.
    """
    num_points = 51
    results = configured_1d_concept.run_1d_analysis(num_points=num_points)
    
    assert results.radius_grid.shape == (num_points,)
    assert results.T_i_profile.shape == (num_points,)
    assert results.fusion_power_profile.shape == (num_points,)
    assert results.ion_heating_profile.shape == (num_points,)
    
    # Check that units are correct
    assert results.fusion_power_profile.unit == u.W / u.m**3
    assert results.T_i_profile.unit == u.keV

def test_1d_solver_physics_sanity(configured_1d_concept):
    """
    Checks if the output profiles follow physically sensible trends.
    """
    results = configured_1d_concept.run_1d_analysis()

    # Fusion power should be highest at the core (r=0) where T is max
    assert np.argmax(results.fusion_power_profile) == 0
    
    # Fusion power should be near zero at the edge where T is low
    assert u.isclose(results.fusion_power_profile[-1], 0 * u.W / u.m**3, atol=1e-9 * u.W / u.m**3)
    
    # Ion temperature profile should be peaked at the core
    assert np.argmax(results.T_i_profile) == 0

def test_plotting_1d_smoke_test(configured_1d_concept):
    """
    Smoke test to ensure the 1D plotting function can be called
    with a valid results object without raising an error.
    """
    results = configured_1d_concept.run_1d_analysis()
    try:
        plt.ion() # Prevent plot from blocking the test suite
        plot_1d_profiles(results)
    finally:
        plt.ioff()
        plt.close('all') 