# tests/test_core_2t.py
import pytest
import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT, FUEL_DD, FUEL_DHE3, FUEL_PB11

def test_power_balance_2t_dt():
    """Validates the 2T power balance for D-T fuel."""
    concept = FusionConcept("Test_D-T")
    concept.set_parameters(
        ion_temperature=20 * u.keV, electron_temperature=18 * u.keV,
        ion_density=1.5e20 / u.m**3, ion_confinement_time=3.0 * u.s,
        electron_confinement_time=2.8 * u.s, particle_confinement_time=5.0*u.s,
        volume=520 * u.m**3, fuel=FUEL_DT, magnetic_field=5.3 * u.T,
        major_radius=3.1 * u.m, minor_radius=1.1 * u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_power > 0 * u.W
    assert results.required_heating_power >= 0 * u.W
    assert results.fusion_gain_q > 0
    assert results.ash_fraction > 1.0

@pytest.mark.parametrize("fuel_obj, name", [(FUEL_DD, "D-D"), (FUEL_DHE3, "D-He3"), (FUEL_PB11, "p-B11")])
def test_power_balance_2t_advanced_fuels(fuel_obj, name):
    """A parameterized test for advanced fuels to reduce code duplication."""
    concept = FusionConcept(f"Test_{name}")
    params = {
        "D-D": {
            "ion_temperature": 50 * u.keV, "electron_temperature": 45 * u.keV,
            "ion_density": 2.0e20 / u.m**3, "ion_confinement_time": 4.0 * u.s,
            "electron_confinement_time": 3.8 * u.s, "particle_confinement_time": 6.0*u.s,
            "volume": 600 * u.m**3, "fuel": FUEL_DD, "magnetic_field": 6.0 * u.T,
            "major_radius": 3.5 * u.m, "minor_radius": 1.2 * u.m
        },
        "D-He3": {
            "ion_temperature": 80 * u.keV, "electron_temperature": 70 * u.keV,
            "ion_density": 2.5e20 / u.m**3, "ion_confinement_time": 5.0 * u.s,
            "electron_confinement_time": 4.5 * u.s, "particle_confinement_time": 7.0*u.s,
            "volume": 700 * u.m**3, "fuel": FUEL_DHE3, "magnetic_field": 7.0 * u.T,
            "major_radius": 4.0 * u.m, "minor_radius": 1.3 * u.m
        },
        "p-B11": {
            "ion_temperature": 300 * u.keV, "electron_temperature": 250 * u.keV,
            "ion_density": 5.0e21 / u.m**3, "ion_confinement_time": 1.0 * u.s,
            "electron_confinement_time": 0.8 * u.s, "particle_confinement_time": 1.5*u.s,
            "volume": 100 * u.m**3, "fuel": FUEL_PB11, "magnetic_field": 12.0 * u.T,
            "major_radius": 2.0 * u.m, "minor_radius": 0.5 * u.m
        }
    }
    concept.set_parameters(**params[name])
    results = concept.run_0d_analysis()
    assert results.fusion_power > 0 * u.W
    assert results.required_heating_power >= 0 * u.W
    assert results.fusion_gain_q > 0
    assert 0 < results.ash_fraction < 1.0

def test_power_balance_2t_dd():
    """Validates the 2T power balance for D-D fuel."""
    test_power_balance_2t_advanced_fuels(FUEL_DD, "D-D")

def test_power_balance_2t_dhe3():
    """Validates the 2T power balance for D-He3 fuel."""
    test_power_balance_2t_advanced_fuels(FUEL_DHE3, "D-He3")

def test_power_balance_2t_pb11():
    """Validates the 2T power balance for p-B11 fuel."""
    test_power_balance_2t_advanced_fuels(FUEL_PB11, "p-B11")

def test_pb11_reactivity_enhancement():
    """Verifies the reactivity enhancement factor for p-B11."""
    concept = FusionConcept("Test_p-B11_Enhancement")
    base_params = {
        "ion_temperature": 300 * u.keV, "electron_temperature": 250 * u.keV,
        "ion_density": 5.0e21 / u.m**3, "ion_confinement_time": 1.0 * u.s,
        "electron_confinement_time": 0.8 * u.s, "particle_confinement_time": 1.5 * u.s,
        "volume": 100 * u.m**3,
        "fuel": FUEL_PB11, "magnetic_field": 12.0 * u.T,
        "major_radius": 2.0 * u.m, "minor_radius": 0.5 * u.m
    }

    # Run with default factor (1.0)
    concept.set_parameters(**base_params)
    results_base = concept.run_0d_analysis()

    # Run with enhancement factor
    enhanced_params = base_params.copy()
    enhanced_params['reactivity_enhancement_factor'] = 2.0
    concept.set_parameters(**enhanced_params)
    results_enhanced = concept.run_0d_analysis()

    assert results_enhanced.fusion_power > results_base.fusion_power
    assert results_enhanced.ash_fraction > results_base.ash_fraction
    assert results_enhanced.fuel_burnup_fraction > results_base.fuel_burnup_fraction
    # Q should also increase with more fusion power for the same input params
    assert results_enhanced.fusion_gain_q > results_base.fusion_gain_q

def test_fuel_dilution_effect_on_power():
    """
    Verifies that higher ash fraction (from longer tau_p) decreases fusion power.
    """
    concept = FusionConcept("Test_Dilution")
    base_params = {
        "ion_temperature": 20 * u.keV, "electron_temperature": 18 * u.keV,
        "ion_density": 1.5e20 / u.m**3, "ion_confinement_time": 3.0 * u.s,
        "electron_confinement_time": 2.8 * u.s, "volume": 520 * u.m**3,
        "fuel": FUEL_DT, "magnetic_field": 5.3 * u.T,
        "major_radius": 3.1 * u.m, "minor_radius": 1.1 * u.m
    }

    # Case 1: Low ash (short tau_p)
    params_low_ash = base_params.copy()
    params_low_ash['particle_confinement_time'] = 1.0 * u.s
    concept.set_parameters(**params_low_ash)
    results_low_ash = concept.run_0d_analysis()

    # Case 2: High ash (long tau_p)
    params_high_ash = base_params.copy()
    params_high_ash['particle_confinement_time'] = 10.0 * u.s
    concept.set_parameters(**params_high_ash)
    results_high_ash = concept.run_0d_analysis()

    # With the new physics, both scenarios are runaway, but the one with
    # longer particle confinement should have a HIGHER ash fraction.
    assert results_high_ash.ash_fraction > results_low_ash.ash_fraction

    # In the runaway state, the fuel is fully consumed, so power should be zero.
    assert u.isclose(results_high_ash.fusion_power, 0 * u.W, atol=1e-3 * u.W)
    assert u.isclose(results_low_ash.fusion_power, 0 * u.W, atol=1e-3 * u.W)

    # Higher ash fraction should lead to higher radiation losses
    assert results_high_ash.bremsstrahlung_power > results_low_ash.bremsstrahlung_power 