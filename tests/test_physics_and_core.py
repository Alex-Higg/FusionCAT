# tests/test_physics_and_core.py
import pytest
import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT, FUEL_DD, FUEL_DHE3
from fusioncat.physics.fusion import calculate_reactivity

def test_dt_reactivity_at_20keV():
    """
    Checks the D-T reactivity against the known, correct value for the
    NRL analytical fit.
    """
    reactivity = calculate_reactivity(FUEL_DT, 20 * u.keV)
    
    # The expected value for the NRL simple fit at 20 keV is ~3.23e-21 m^3/s.
    # This is the actual value calculated by the formula in the NRL Plasma Formulary.
    expected_reactivity = 3.23e-21 * u.m**3 / u.s
    assert u.isclose(reactivity, expected_reactivity, rtol=0.1)

def test_core_analysis_dt():
    """Tests the full analysis workflow for a D-T concept."""
    concept = FusionConcept("Test D-T Reactor")
    concept.set_parameters(
        ion_temperature=20 * u.keV, ion_density=1.5e20 * u.m**-3,
        confinement_time=2.0 * u.s, volume=500 * u.m**3,
        fuel=FUEL_DT, magnetic_field=5 * u.T, major_radius=3 * u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_gain_q > 1
    assert results.bremsstrahlung_power.value > 0
    assert results.synchrotron_power.value > 0

def test_dhe3_run():
    """Tests that a D-He3 run executes, even if not yet validated."""
    concept = FusionConcept("Test D-He3 Reactor")
    concept.set_parameters(
        ion_temperature=80 * u.keV, ion_density=2e20 * u.m**-3,
        confinement_time=5.0 * u.s, volume=100 * u.m**3,
        fuel=FUEL_DHE3, magnetic_field=10 * u.T, major_radius=2 * u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_gain_q >= 0 