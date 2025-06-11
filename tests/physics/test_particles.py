# tests/physics/test_particles.py
import pytest
import astropy.units as u
from fusioncat.physics.particles import solve_steady_state_composition
from fusioncat.utils.fuels import FUEL_DT, FUEL_DD
import numpy as np

def test_solver_runs_without_error():
    """
    Tests that the solver runs to completion without raising errors for a
    typical set of parameters.
    """
    params = {
        "n_i_total": 1.5e20 / u.m**3,
        "T_i": 20 * u.keV,
        "tau_p": 5.0 * u.s,
        "fuel": FUEL_DT
    }
    try:
        ash_frac, burnup_frac, z_eff = solve_steady_state_composition(**params)
        assert np.isfinite(ash_frac)
        assert np.isfinite(burnup_frac)
        assert np.isfinite(z_eff)
    except Exception as e:
        pytest.fail(f"Solver raised an unexpected exception: {e}")

def test_no_ash_at_zero_temp():
    """
    Ensures that at T=0, where reactivity is zero, the ash fraction is zero.
    """
    params = {
        "n_i_total": 1.5e20 / u.m**3,
        "T_i": 0 * u.keV,
        "tau_p": 5.0 * u.s,
        "fuel": FUEL_DT
    }
    ash_frac, burnup_frac, z_eff = solve_steady_state_composition(**params)
    assert u.isclose(ash_frac, 0.0)
    assert u.isclose(burnup_frac, 0.0)
    assert u.isclose(z_eff, 1.0)

def test_z_eff_consistency():
    """
    Checks if Z_eff is calculated in a self-consistent way.
    For a pure D-T plasma with He ash, Z_eff should be between 1 and 2.
    """
    params = {
        "n_i_total": 1.5e20 / u.m**3,
        "T_i": 15 * u.keV, # Use a lower temp to avoid runaway
        "tau_p": 3.0 * u.s,
        "fuel": FUEL_DT
    }
    ash_frac, _, z_eff = solve_steady_state_composition(**params)
    
    # Check if the calculated z_eff is consistent with the ash fraction
    # Z_eff = (f_fuel * Z_fuel^2 + f_ash * Z_ash^2) / (f_fuel * Z_fuel + f_ash * Z_ash)
    # For D-T (Z_fuel=1) and He ash (Z_ash=2)
    f_ash = ash_frac
    f_fuel = 1 - f_ash
    if f_fuel < 0: f_fuel = 0 # Handle runaway case
    
    n_e_norm = f_fuel * 1 + f_ash * 2
    z_eff_manual = (f_fuel * 1**2 + f_ash * 2**2) / n_e_norm
    
    assert u.isclose(z_eff, z_eff_manual, rtol=1e-3)

def test_dd_composition():
    """
    Checks that the solver runs for D-D fuel without error.
    """
    params = {
        "n_i_total": 2.0e20 / u.m**3,
        "T_i": 50 * u.keV,
        "tau_p": 4.0 * u.s,
        "fuel": FUEL_DD
    }
    ash_frac, burnup_frac, z_eff = solve_steady_state_composition(**params)
    assert 0 < ash_frac < 1
    assert 0 < burnup_frac < 1
    assert z_eff > 1.0 