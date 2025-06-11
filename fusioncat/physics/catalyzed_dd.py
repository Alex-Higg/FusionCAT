# fusioncat/physics/catalyzed_dd.py
import astropy.units as u
import numpy as np
from ..utils.fuels import FUEL_DD, FUEL_DT, FUEL_DHE3
from .fusion import calculate_reactivity

def solve_catalyzed_dd_composition(
    n_i_total: u.Quantity, T_i: u.Quantity, tau_p: u.Quantity,
    max_iter: int = 100, tol: float = 1e-6
):
    """
    Solves for the steady-state composition of a catalyzed D-D plasma.

    This function iteratively solves the particle balance equations for the
    primary plasma species (D, T, He3) and the resulting ash (p, He4).
    It assumes the tritium and helium-3 produced are burned in situ.

    Parameters
    ----------
    n_i_total : astropy.units.Quantity
        The total ion density to be maintained.
    T_i : astropy.units.Quantity
        The ion temperature.
    tau_p : astropy.units.Quantity
        The global particle confinement time for all species.
    max_iter : int, optional
        The maximum number of iterations for the solver, by default 100.
    tol : float, optional
        The convergence tolerance for the species densities, by default 1e-6.

    Returns
    -------
    dict[str, u.Quantity | float]
        A dictionary containing the converged densities of 'D', 'T', 'He3',
        the total 'ash' density (p + He4), the total 'fusion_power' density,
        and the final 'z_eff'.
    """
    # Get reactivities for all relevant reactions at the given temperature
    # The two D-D branches are assumed to have equal probability.
    sv_dd_total = calculate_reactivity(FUEL_DD, T_i)
    sv_dd_p = sv_dd_total / 2.0  # D(d,p)T branch
    sv_dd_n = sv_dd_total / 2.0  # D(d,n)He3 branch
    sv_dt = calculate_reactivity(FUEL_DT, T_i)
    sv_dhe3 = calculate_reactivity(FUEL_DHE3, T_i)

    # Initial guess: all ions are deuterium
    n_D = n_i_total
    n_T = 0.0 / u.m**3
    n_He3 = 0.0 / u.m**3
    
    for _ in range(max_iter):
        n_D_old, n_T_old, n_He3_old = n_D, n_T, n_He3

        # --- REACTION RATES (reactions / m^3 / s) ---
        # Note: 0.5 factor for reactions between identical particles (D-D)
        rate_dd = 0.5 * n_D**2 * (sv_dd_p + sv_dd_n)
        rate_dt = n_D * n_T * sv_dt
        rate_dhe3 = n_D * n_He3 * sv_dhe3

        # --- PARTICLE BALANCE EQUATIONS (Source = Loss) ---
        # For a species `s`, loss is n_s/tau_p + sum(consumption_rates)
        # Source is sum(production_rates)
        
        # Tritium: Produced by D(d,p)T, consumed by D-T fusion and transport
        # Source_T = 0.5 * n_D^2 * sv_dd_p
        # Loss_T = n_T/tau_p + n_D*n_T*sv_dt
        # n_T_ss = (0.5 * n_D**2 * sv_dd_p) / (1/tau_p + n_D*sv_dt)
        n_T = (0.5 * n_D**2 * sv_dd_p) / (1/tau_p + n_D * sv_dt)
        
        # Helium-3: Produced by D(d,n)He3, consumed by D-He3 and transport
        # n_He3_ss = (0.5 * n_D**2 * sv_dd_n) / (1/tau_p + n_D*sv_dhe3)
        n_He3 = (0.5 * n_D**2 * sv_dd_n) / (1/tau_p + n_D * sv_dhe3)
        
        # Deuterium: Total ions minus other species (T, He3, and ash)
        # Ash (p, He4) density is implied by the balance
        # Ash source = rate_dd_p (p) + rate_dt (He4) + rate_dhe3 (p + He4)
        # The D(d,p)T reaction makes a proton (ash)
        # The D-T reaction makes a He4 (ash)
        # The D-He3 reaction makes a proton and a He4 (two ash particles)
        ash_source_rate = (0.5 * n_D**2 * sv_dd_p) + (n_D * n_T * sv_dt) + 2 * (n_D * n_He3 * sv_dhe3)
        n_ash = ash_source_rate * tau_p

        # The total number of ions is conserved
        n_D_new = n_i_total - n_T - n_He3 - n_ash
        if n_D_new.value < 0:
            n_D = 1e10 / u.m**3
        else:
            n_D = n_D_new
        
        # Check for convergence
        delta_d = abs((n_D - n_D_old) / (n_D_old + 1e-20/u.m**3)).value
        delta_t = abs((n_T - n_T_old) / (n_T_old + 1e-20/u.m**3)).value
        if max(delta_d, delta_t) < tol:
            break

    # --- FINAL CALCULATIONS ---
    # Recalculate rates and power with converged densities
    rate_dd = 0.5 * n_D**2 * (sv_dd_p + sv_dd_n)
    rate_dt = n_D * n_T * sv_dt
    rate_dhe3 = n_D * n_He3 * sv_dhe3
    
    # Power density from each reaction channel (W/m^3)
    p_dd = rate_dd * FUEL_DD.energy_per_reaction
    p_dt = rate_dt * FUEL_DT.energy_per_reaction
    p_dhe3 = rate_dhe3 * FUEL_DHE3.energy_per_reaction
    
    total_power_density = (p_dd + p_dt + p_dhe3).to(u.W / u.m**3)
    
    # Z_eff calculation: sum(n_s * Z_s^2) / n_e
    # Ash products are protons (Z=1) and He4 (Z=2).
    # Source rate of protons = rate_dd_p + rate_dhe3
    # Source rate of He4 = rate_dt + rate_dhe3
    n_ash_p = (rate_dd / 2.0 + rate_dhe3) * tau_p # rate_dd/2 is rate_dd_p
    n_ash_he4 = (rate_dt + rate_dhe3) * tau_p
    n_ash_total = n_ash_p + n_ash_he4
    
    n_e = n_D*1 + n_T*1 + n_He3*2 + n_ash_p*1 + n_ash_he4*2
    z_eff_num = n_D*1**2 + n_T*1**2 + n_He3*2**2 + n_ash_p*1**2 + n_ash_he4*2**2
    z_eff = (z_eff_num / n_e).to_value(u.dimensionless_unscaled)
    
    return {
        "D_density": n_D.to(1/u.m**3),
        "T_density": n_T.to(1/u.m**3),
        "He3_density": n_He3.to(1/u.m**3),
        "ash_density": n_ash_total.to(1/u.m**3),
        "fusion_power_density": total_power_density,
        "z_eff": z_eff,
    } 