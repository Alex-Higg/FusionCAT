# tests/test_dd_catalyzed_cycle.py
import pytest
import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DD

def test_catalyzed_dd_power_enhancement():
    """
    Verifies that the catalyzed D-D cycle produces more power than the
    simple D-D cycle due to the burning of T and He3 products.
    """
    concept = FusionConcept("Test_Catalyzed_DD")
    base_params = {
        "ion_temperature": 50 * u.keV, "electron_temperature": 45 * u.keV,
        "ion_density": 2.0e20 / u.m**3, "ion_confinement_time": 4.0 * u.s,
        "electron_confinement_time": 3.8 * u.s, "particle_confinement_time": 6.0*u.s,
        "volume": 600 * u.m**3, "fuel": FUEL_DD, "magnetic_field": 6.0 * u.T,
        "major_radius": 3.5 * u.m, "minor_radius": 1.2 * u.m
    }

    # Case 1: Simple D-D cycle
    simple_params = base_params.copy()
    simple_params['cycle'] = 'simple'
    concept.set_parameters(**simple_params)
    results_simple = concept.run_0d_analysis()
    power_simple = results_simple.fusion_power

    # Case 2: Catalyzed D-D cycle
    catalyzed_params = base_params.copy()
    catalyzed_params['cycle'] = 'catalyzed_dd'
    concept.set_parameters(**catalyzed_params)
    results_catalyzed = concept.run_0d_analysis()
    power_catalyzed = results_catalyzed.fusion_power

    # The catalyzed cycle should produce significantly more power
    assert power_catalyzed > power_simple

    # The required heating should be different
    assert results_catalyzed.required_heating_power != results_simple.required_heating_power

    # Ash fraction should be non-zero and plausible
    assert 0 < results_catalyzed.ash_fraction < 1 