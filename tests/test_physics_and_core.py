# tests/test_physics_and_core.py
import pytest
import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT, FUEL_DD, FUEL_DHE3, FUEL_PB11
from fusioncat.physics.fusion import calculate_reactivity
from fusioncat.physics.radiation import calculate_bremsstrahlung_power, calculate_synchrotron_power
from fusioncat.physics.lawson import calculate_ion_electron_exchange

def test_dt_reactivity():
    """Checks the D-T reactivity against the known Bosch-Hale value."""
    reactivity = calculate_reactivity(FUEL_DT, 20 * u.keV)
    expected = 4.33e-22 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.01)

@pytest.mark.parametrize("temp, expected_m3_s", [
    (10 * u.keV, 1.08e-22), (50 * u.keV, 8.58e-22), (100 * u.keV, 8.44e-22)
])
def test_dt_reactivity_across_temperatures(temp, expected_m3_s):
    reactivity = calculate_reactivity(FUEL_DT, temp)
    expected = expected_m3_s * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.1)

def test_dd_reactivity():
    """Checks D-D reactivity against a known value."""
    reactivity = calculate_reactivity(FUEL_DD, 50 * u.keV)
    expected = 1.15e-23 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.01)

def test_dhe3_reactivity():
    """Checks D-He3 reactivity against a known value."""
    reactivity = calculate_reactivity(FUEL_DHE3, 80 * u.keV)
    expected = 5.2e-23 * u.m**3 / u.s
    assert u.isclose(reactivity, expected, rtol=0.1)

def test_pb11_reactivity():
    """Checks p-B11 reactivity enhancement factor."""
    T = 300 * u.keV
    base = calculate_reactivity(FUEL_PB11, T)
    enhanced = calculate_reactivity(FUEL_PB11, T, reactivity_enhancement_factor=2.0)
    assert u.isclose(enhanced, 2.0 * base)

def test_core_analysis_runs():
    """Tests that the full 0D analysis runs without error for D-T."""
    concept = FusionConcept("Test Reactor")
    concept.set_parameters(
        ion_temperature=20 * u.keV, electron_temperature=18 * u.keV,
        ion_density=1.5e20 / u.m**3, ion_confinement_time=2.0 * u.s,
        electron_confinement_time=2.0 * u.s, volume=500 * u.m**3,
        fuel=FUEL_DT, magnetic_field=5 * u.T, 
        major_radius=3 * u.m, minor_radius=1 * u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_gain_q > 1.0

def test_unimplemented_fuel_reactivity():
    """The new tests for other fuels will go here once implemented."""
    pass # No longer raises NotImplementedError

def test_core_input_validation():
    """Ensures the core analysis raises errors for non-physical inputs."""
    concept = FusionConcept("Validation Test")
    
    # Test for negative temperature
    concept.set_parameters(
        ion_temperature=-10 * u.keV,
        electron_temperature=10 * u.keV,
        ion_density=1e20 / u.m**3,
        ion_confinement_time=1 * u.s,
        electron_confinement_time=1 * u.s,
        volume=1 * u.m**3,
        fuel=FUEL_DT,
        magnetic_field=5 * u.T,
        major_radius=1 * u.m,
        minor_radius=0.3 * u.m
    )
    with pytest.raises(ValueError, match="Temperatures must be positive"):
        concept.run_0d_analysis()

    # Test for zero density
    concept.set_parameters(
        ion_temperature=10 * u.keV,
        electron_temperature=10 * u.keV,
        ion_density=0 / u.m**3,
        ion_confinement_time=1 * u.s,
        electron_confinement_time=1 * u.s,
        volume=1 * u.m**3,
        fuel=FUEL_DT,
        magnetic_field=5 * u.T,
        major_radius=1 * u.m,
        minor_radius=0.3 * u.m
    )
    with pytest.raises(ValueError, match="Density must be positive"):
        concept.run_0d_analysis()

    # Test for invalid geometry
    concept.set_parameters(
        ion_temperature=10 * u.keV,
        electron_temperature=10 * u.keV,
        ion_density=1e20 / u.m**3,
        ion_confinement_time=1 * u.s,
        electron_confinement_time=1 * u.s,
        volume=1 * u.m**3,
        fuel=FUEL_DT,
        magnetic_field=5 * u.T,
        major_radius=1 * u.m,
        minor_radius=2 * u.m  # Invalid: minor radius > major radius
    )
    with pytest.raises(ValueError, match="Minor radius must be less than major radius"):
        concept.run_0d_analysis()

def test_bremsstrahlung_power():
    """Test bremsstrahlung power calculation."""
    n_e = 1e20 * u.m**-3
    T_e = 10 * u.keV
    V = 1000 * u.m**3
    charge_map = {'D+': 1.0, 'e-': 1.0}  # D+ ions, electrons
    p_br = calculate_bremsstrahlung_power(n_e, T_e, charge_map) * V
    assert p_br.unit == u.W
    assert p_br > 0 * u.W

def test_synchrotron_power():
    """Test synchrotron power calculation."""
    n_e = 1e20 * u.m**-3
    T_e = 10 * u.keV
    B = 5 * u.T
    R = 6 * u.m
    a = 2 * u.m
    p_synch = calculate_synchrotron_power(n_e, T_e, B, R, a)
    assert p_synch.unit == u.W
    assert p_synch > 0 * u.W

def test_ion_electron_exchange():
    """Test ion-electron energy exchange calculation."""
    n_i = 1e20 * u.m**-3
    T_i = 20 * u.keV
    n_e = 1e20 * u.m**-3
    T_e = 10 * u.keV
    V = 1000 * u.m**3
    # Use temperature-energy equivalency for conversion
    p_ie = calculate_ion_electron_exchange(n_i, T_i, n_e, T_e, V, FUEL_DT)
    assert p_ie.unit == u.W
    # Should be positive when T_i > T_e
    assert p_ie > 0 * u.W

def test_core_analysis():
    """Test the core 0D analysis with 2T model."""
    concept = FusionConcept("test_concept")
    # Set up parameters for 2T model
    concept.set_parameters(
        ion_temperature=20 * u.keV,
        electron_temperature=10 * u.keV,
        ion_density=1e20 * u.m**-3,
        ion_confinement_time=1 * u.s,
        electron_confinement_time=0.5 * u.s,
        volume=1000 * u.m**3,
        fuel=FUEL_DT,
        magnetic_field=5 * u.T,
        major_radius=6 * u.m,
        minor_radius=2 * u.m
    )
    results = concept.run_0d_analysis()
    # Check that all power terms are positive
    assert results.fusion_power > 0 * u.W
    assert results.charged_particle_power > 0 * u.W
    assert results.bremsstrahlung_power > 0 * u.W
    assert results.synchrotron_power > 0 * u.W
    assert results.ion_confinement_loss > 0 * u.W
    assert results.electron_confinement_loss > 0 * u.W
    assert results.total_loss_power > 0 * u.W
    # Check that temperatures are preserved
    assert results.ion_temperature == 20 * u.keV
    assert results.electron_temperature == 10 * u.keV
    # Check that Q is calculated
    assert isinstance(results.fusion_gain_q, float)
    assert results.fusion_gain_q >= 0
    # Check that ion-electron exchange is calculated
    assert results.ion_electron_exchange_power.unit == u.W

def test_core_analysis_dt():
    """Tests the full analysis workflow for a D-T concept."""
    concept = FusionConcept("Test D-T Reactor")
    concept.set_parameters(
        ion_temperature=20 * u.keV,
        electron_temperature=18 * u.keV,
        ion_density=1.5e20 / u.m**3,
        ion_confinement_time=2.0 * u.s,
        electron_confinement_time=2.0 * u.s,
        volume=500 * u.m**3,
        fuel=FUEL_DT,
        magnetic_field=5 * u.T,
        major_radius=3 * u.m,
        minor_radius=1 * u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_gain_q > 1
    assert results.bremsstrahlung_power.value > 0
    assert results.synchrotron_power.value > 0

def test_dhe3_run():
    """Tests that a D-He3 run executes, even if not yet validated (2T model)."""
    concept = FusionConcept("Test D-He3 Reactor")
    concept.set_parameters(
        ion_temperature=80 * u.keV,
        electron_temperature=40 * u.keV,
        ion_density=2e20 * u.m**-3,
        ion_confinement_time=5.0 * u.s,
        electron_confinement_time=2.5 * u.s,
        volume=100 * u.m**3,
        fuel=FUEL_DHE3,
        magnetic_field=10 * u.T,
        major_radius=2 * u.m,
        minor_radius=0.7 * u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_gain_q >= 0

def test_core_analysis_dhe3():
    concept = FusionConcept("Test D-He3 Reactor")
    concept.set_parameters(
        ion_temperature=80 * u.keV, electron_temperature=70*u.keV,
        ion_density=2e20 / u.m**-3, ion_confinement_time=5.0 * u.s,
        electron_confinement_time=5.0 * u.s, volume=100 * u.m**3,
        fuel=FUEL_DHE3, magnetic_field=10 * u.T, 
        major_radius=2 * u.m, minor_radius=0.7 * u.m
    )
    results = concept.run_0d_analysis()
    assert results.fusion_gain_q >= 0
    assert results.fusion_power > 1 * u.W 