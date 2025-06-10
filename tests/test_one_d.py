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
    """Provides a fully configured FusionConcept object for 1D analysis."""
    concept = FusionConcept(name="Test 1D Reactor")
    concept.set_parameters(
        ion_temperature=25 * u.keV, electron_temperature=22 * u.keV,
        ion_density=1.5e20 / u.m**3, fuel=FUEL_DT,
        ion_diffusivity=0.5 * u.m**2 / u.s, electron_diffusivity=0.8 * u.m**2 / u.s,
        minor_radius=2.0 * u.m, major_radius=6.0 * u.m,
        volume=500 * u.m**3, magnetic_field=5.5 * u.T,
    )
    return concept

def test_parabolic_profile_shape():
    """Tests that the parabolic profile function behaves as expected."""
    r_grid = np.linspace(0, 1, 11)
    profile = parabolic_profile(r_grid, 10.0, 1.0, alpha=1.0)
    assert profile[0] == pytest.approx(10.0)
    assert profile[-1] == pytest.approx(1.0)

def test_1d_solver_execution_and_shapes(configured_1d_concept):
    """Smoke test to ensure the 1D solver runs and produces correct shapes."""
    num_points = 51
    results = configured_1d_concept.run_1d_analysis(num_points=num_points)
    assert results.radius_grid.shape == (num_points,)
    assert results.fusion_power_profile.shape == (num_points,)
    assert results.fusion_power_profile.unit == u.W / u.m**3

def test_1d_solver_physics_sanity(configured_1d_concept):
    """Checks if the output profiles follow physically sensible trends."""
    results = configured_1d_concept.run_1d_analysis()
    assert np.argmax(results.fusion_power_profile.value) == 0
    assert u.isclose(results.fusion_power_profile[-1], 0 * u.W / u.m**3, atol=1e-9 * u.W / u.m**3)

def test_plotting_1d_smoke_test(configured_1d_concept):
    """Smoke test for the 1D plotting function."""
    results = configured_1d_concept.run_1d_analysis()
    try:
        plt.ion()
        plot_1d_profiles(results)
    finally:
        plt.ioff()
        plt.close('all') 