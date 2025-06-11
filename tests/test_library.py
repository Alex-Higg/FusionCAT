# tests/test_library.py
import pytest; import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT, FUEL_DD, FUEL_DHE3, FUEL_PB11
from fusioncat.physics.fusion import calculate_reactivity

def test_dt_reactivity():
    """Checks the D-T reactivity against the known data point."""
    reactivity = calculate_reactivity(FUEL_DT, 20 * u.keV)
    # The interpolation is based on this exact data point from NRL Formulary
    expected = 4.33e-22 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.01)

@pytest.mark.parametrize("temp, expected_cm3_s", [
    (10 * u.keV, 1.08e-17),
    (50 * u.keV, 8.58e-17),
    (100 * u.keV, 8.44e-17),
])
def test_dt_reactivity_across_temperatures(temp, expected_cm3_s):
    """Tests the D-T reactivity interpolation at multiple points."""
    reactivity = calculate_reactivity(FUEL_DT, temp)
    expected = expected_cm3_s * u.cm**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.1)

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
    """Tests that the full 0D analysis runs and produces sensible physical values."""
    concept = FusionConcept("Test Reactor")
    concept.set_parameters(
        ion_temperature=20*u.keV, electron_temperature=18*u.keV,
        ion_density=1.5e20/u.m**3, ion_confinement_time=2.0*u.s,
        electron_confinement_time=2.0*u.s, volume=500*u.m**3,
        fuel=FUEL_DT, magnetic_field=5*u.T, 
        major_radius=3*u.m, minor_radius=1*u.m
    )
    results = concept.run_0d_analysis()
    
    # Physics Sanity Checks
    assert results.fusion_power > 1 * u.MW, "Fusion power should be significant."
    assert results.fusion_gain_q > 1.0, "Q-value should be greater than 1 for these parameters."
    # Since T_i > T_e, ions should be heating electrons, so P_ie should be positive.
    assert results.ion_electron_exchange_power > 0 * u.W, "Ions should be heating electrons."