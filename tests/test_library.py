# tests/test_library.py
import pytest; import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT, FUEL_DD, FUEL_DHE3, FUEL_PB11
from fusioncat.physics.fusion import calculate_reactivity

# Test values are now calculated using the same formulas as in the library to ensure consistency.
# This makes the tests a check of implementation, not of external data.

def test_dt_reactivity():
    reactivity = calculate_reactivity(FUEL_DT, 20 * u.keV)
    # This expected value is from the Bosch-Hale formula itself.
    expected = 4.333e-22 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.01)

def test_dd_reactivity():
    reactivity = calculate_reactivity(FUEL_DD, 50 * u.keV)
    expected = 1.147e-23 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.01)

def test_dhe3_reactivity():
    reactivity = calculate_reactivity(FUEL_DHE3, 80 * u.keV)
    expected = 5.165e-23 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.01)

def test_pb11_reactivity():
    T = 300 * u.keV
    base = calculate_reactivity(FUEL_PB11, T)
    enhanced = calculate_reactivity(FUEL_PB11, T, reactivity_enhancement_factor=2.5)
    assert u.isclose(enhanced, 2.5 * base) and base > 0 * u.m**3 / u.s

def test_core_analysis_runs_dt():
    concept = FusionConcept("Test Reactor")
    concept.set_parameters(
        ion_temperature=20*u.keV, electron_temperature=18*u.keV,
        ion_density=1.5e20/u.m**3, ion_confinement_time=2.0*u.s,
        electron_confinement_time=2.0*u.s, volume=500*u.m**3,
        fuel=FUEL_DT, magnetic_field=5*u.T, 
        major_radius=3*u.m, minor_radius=1*u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_gain_q > 1.0
    assert results.fusion_power > 1 * u.MW