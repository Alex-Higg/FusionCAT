# tests/test_library.py
import pytest; import astropy.units as u
from fusioncat.core import FusionConcept; from fusioncat.utils.fuels import FUEL_DT, FUEL_DD, FUEL_DHE3, FUEL_PB11
from fusioncat.physics.fusion import calculate_reactivity

def test_dt_reactivity():
    expected = 4.33e-22 * u.m**3 / u.s
    assert u.isclose(calculate_reactivity(FUEL_DT, 20 * u.keV), expected, rtol=0.01)

def test_dd_reactivity():
    expected = 1.15e-23 * u.m**3 / u.s
    assert u.isclose(calculate_reactivity(FUEL_DD, 50 * u.keV), expected, rtol=0.01)

def test_dhe3_reactivity():
    expected = 5.2e-23 * u.m**3 / u.s
    assert u.isclose(calculate_reactivity(FUEL_DHE3, 80 * u.keV), expected, rtol=0.1)

def test_pb11_reactivity():
    T = 300 * u.keV
    base = calculate_reactivity(FUEL_PB11, T)
    enhanced = calculate_reactivity(FUEL_PB11, T, reactivity_enhancement_factor=2.0)
    assert u.isclose(enhanced, 2.0 * base)

@pytest.fixture
def concept_0d():
    concept = FusionConcept("Test 0D"); concept.set_parameters(
        ion_temperature=20*u.keV, electron_temperature=18*u.keV, ion_density=1.5e20/u.m**3,
        ion_confinement_time=2.0*u.s, electron_confinement_time=2.0*u.s, volume=500*u.m**3,
        fuel=FUEL_DT, magnetic_field=5*u.T, major_radius=3*u.m, minor_radius=1*u.m)
    return concept

def test_0d_analysis_runs(concept_0d):
    results = concept_0d.run_0d_analysis()
    assert results.fusion_gain_q > 1.0

@pytest.fixture
def concept_1d():
    concept = FusionConcept("Test 1D"); concept.set_parameters(
        ion_temperature=25*u.keV, electron_temperature=22*u.keV, ion_density=1.5e20/u.m**3,
        fuel=FUEL_DT, ion_diffusivity=0.5*u.m**2/u.s, electron_diffusivity=0.8*u.m**2/u.s,
        minor_radius=2.0*u.m, major_radius=6.0*u.m, volume=500*u.m**3, magnetic_field=5.5*u.T)
    return concept

def test_1d_analysis_runs(concept_1d):
    results = concept_1d.run_1d_analysis()
    assert results.fusion_power_profile.unit == u.W / u.m**3
    assert results.fusion_power_profile.shape == (101,) 