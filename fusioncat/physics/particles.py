# fusioncat/physics/particles.py
import astropy.units as u
import numpy as np
from ..utils.fuels import Fuel
from .fusion import calculate_reactivity

def get_ash_charge(fuel: Fuel) -> float:
    # DEPRECATED: This function is no longer used and will be removed.
    # Ash charge is now stored directly in the Fuel object.
    pass

def solve_steady_state_composition(
    n_i_total: u.Quantity, T_i: u.Quantity, tau_p: u.Quantity,
    fuel: Fuel, ratio: float = 0.5, enhancement_factor: float = 1.0,
    max_iter: int = 100, tol: float = 1e-6
):
    """
    Solves for the steady-state plasma composition, including fuel and ash.

    This function iteratively solves the particle balance equations for a
    reacting plasma to find the equilibrium ash concentration and the
    corresponding fuel dilution.

    Parameters
    ----------
    n_i_total : astropy.units.Quantity
        The total ion density to be maintained.
    T_i : astropy.units.Quantity
        The ion temperature.
    tau_p : astropy.units.Quantity
        The global particle confinement time.
    fuel : Fuel
        The fusion fuel being used.
    ratio : float, optional
        The ratio of the first reactant species to the total fuel ion density,
        by default 0.5.
    enhancement_factor : float, optional
        Reactivity enhancement factor for p-B11, by default 1.0.
    max_iter : int, optional
        The maximum number of iterations for the solver, by default 100.
    tol : float, optional
        The convergence tolerance for the ash fraction, by default 1e-6.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:
        - The converged ash fraction (n_ash / n_i_total).
        - The fuel burnup fraction.
        - The effective charge (Z_eff) of the plasma.
    """
    # Get reactivity for the given conditions
    sigma_v = calculate_reactivity(fuel, T_i, enhancement_factor)

    ash_fraction = 0.0
    
    for _ in range(max_iter):
        ash_fraction_old = ash_fraction

        # Calculate densities based on current ash_fraction
        n_ash = n_i_total * ash_fraction
        n_fuel_ions = n_i_total - n_ash
        n1 = n_fuel_ions * ratio
        n2 = n_fuel_ions * (1 - ratio)

        # Fusion reaction rate (reactions per m^3 per s)
        reaction_rate = (n1 * n2 * sigma_v).to(1 / (u.m**3 * u.s))
        if fuel.reactants[0] == fuel.reactants[1]:
            reaction_rate *= 0.5
        
        # Particle balance: Source = Loss
        # Ash source = reaction_rate. Ash loss = n_ash / tau_p.
        # Solve for the steady-state ash density: n_ash_ss
        n_ash_ss = (reaction_rate * tau_p).to(1 / u.m**3)

        # Update ash fraction for convergence check
        ash_fraction = (n_ash_ss / n_i_total).to_value(u.dimensionless_unscaled)
        
        # If ash fraction becomes unphysically large, break the loop
        if ash_fraction > 1.0:
            break
            
        if np.abs(ash_fraction - ash_fraction_old) < tol:
            break
    
    # Recalculate final densities with converged ash fraction
    n_ash = n_i_total * ash_fraction
    fuel_density = n_i_total - n_ash
    # Ensure fuel density is not negative, which is unphysical.
    if fuel_density < 0 * u.m**-3:
        fuel_density = 0 * u.m**-3
        n_ash = n_i_total # All ions are ash
    
    # Recalculate reaction rate for the final burnup calculation
    n1 = fuel_density * ratio
    n2 = fuel_density * (1-ratio)
    reaction_rate = (n1 * n2 * sigma_v).to(1 / (u.m**3 * u.s))
    if fuel.reactants[0] == fuel.reactants[1]:
        reaction_rate *= 0.5
        
    fuel_loss_rate_fusion = 2 * reaction_rate # Two fuel ions consumed per reaction
    fuel_loss_rate_transport = fuel_density / tau_p
    
    total_fuel_loss_rate = fuel_loss_rate_fusion + fuel_loss_rate_transport
    
    # Burnup fraction = Fusion Loss / Total Loss
    if total_fuel_loss_rate == 0 * total_fuel_loss_rate.unit:
        burnup_fraction = 0.0
    else:
        burnup_fraction = (fuel_loss_rate_fusion / total_fuel_loss_rate).to_value(u.dimensionless_unscaled)
    
    # Calculate Z_eff = sum(n_s * Z_s^2) / n_e
    # Correctly handle charges of all fuel reactants and ash
    z1 = 1.0 if "p" in fuel.reactants[0] else 1.0 if "D" in fuel.reactants[0] else 1.0 if "T" in fuel.reactants[0] else 2.0 if "He3" in fuel.reactants[0] else 5.0
    z2 = 1.0 if "p" in fuel.reactants[1] else 1.0 if "D" in fuel.reactants[1] else 1.0 if "T" in fuel.reactants[1] else 2.0 if "He3" in fuel.reactants[1] else 5.0
    
    # This is getting complicated. Let's create a dictionary for charges.
    charge_dict = {'p': 1, 'D': 1, 'T': 1, 'He3': 2, 'B11': 5}
    z1 = charge_dict[fuel.reactants[0]]
    z2 = charge_dict[fuel.reactants[1]]
    
    n1_final = fuel_density * ratio
    n2_final = fuel_density * (1 - ratio)
    
    # Assuming quasi-neutrality, n_e = sum(n_s * Z_s)
    n_e = n1_final * z1 + n2_final * z2 + n_ash * fuel.ash_charge
    
    # Z_eff = sum(n_s * Z_s^2) / n_e
    z_eff_numerator = n1_final * z1**2 + n2_final * z2**2 + n_ash * fuel.ash_charge**2
    z_eff = (z_eff_numerator / n_e).to_value(u.dimensionless_unscaled)

    return ash_fraction, burnup_fraction, z_eff 