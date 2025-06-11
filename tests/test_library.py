# tests/test_library.py
import pytest; import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT, FUEL_DD, FUEL_DHE3, FUEL_PB11
from fusioncat.physics.fusion import calculate_reactivity

def test_dt_reactivity():
    reactivity = calculate_reactivity(FUEL_DT, 20 * u.keV)
    expected = 4.33e-22 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.01)

def test_dd_reactivity():
    reactivity = calculate_reactivity(FUEL_DD, 50 * u.keV)
    expected = 1.15e-23 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.02)

def test_dhe3_reactivity():
    reactivity = calculate_reactivity(FUEL_DHE3, 80 * u.keV)
    expected = 5.16e-23 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.02)

def test_pb11_reactivity():
    base = calculate_reactivity(FUEL_PB11, 300 * u.keV)
    enhanced = calculate_reactivity(FUEL_PB11, 300 * u.keV, reactivity_enhancement_factor=2.0)
    assert u.isclose(enhanced, 2.0 * base)

def test_core_analysis_runs():
    concept = FusionConcept("Test Reactor")
    concept.set_parameters(
        ion_temperature=20*u.keV, electron_temperature=18*u.keV,
        ion_density=1.5e20/u.m**3, ion_confinement_time=2.0*u.s,
        electron_confinement_time=2.0*u.s, volume=500*u.m**3,
        fuel=FUEL_DT, magnetic_field=5*u.T, major_radius=3*u.m, minor_radius=1*u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_gain_q > 1.0 