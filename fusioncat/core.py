# fusioncat/core.py
import astropy.units as u
import astropy.constants as const
from .utils.exceptions import ConfigurationError
from .utils.results import ZeroDResults
from .physics.fusion import calculate_fusion_power, calculate_reactivity
from .physics.radiation import calculate_bremsstrahlung_power, calculate_synchrotron_power
from .physics.lawson import calculate_triple_product, calculate_ion_electron_exchange
from .physics.particles import solve_steady_state_composition
from .physics.catalyzed_dd import solve_catalyzed_dd_composition
from .utils.fuels import FUEL_DD, FUEL_DT, FUEL_DHE3

class FusionConcept:
    def __init__(self, name: str):
        self.name = name; self.params = {}
    def set_parameters(self, **kwargs): self.params = kwargs
    def run_0d_analysis(self) -> ZeroDResults:
        """
        Performs a 0D, two-temperature power balance analysis.

        This solver takes a set of plasma parameters (temperatures, density,
        confinement times, etc.) and calculates the resulting fusion power,
        power losses, and the required auxiliary heating power needed to
        maintain the plasma in a steady state.

        The power balance is solved independently for the ion and electron
        species, accounting for energy transfer between them.

        Parameters
        ----------
        All parameters are set via the `set_parameters` method. Required keys
        for this solver include:
        - 'ion_temperature' (u.Quantity)
        - 'electron_temperature' (u.Quantity)
        - 'ion_density' (u.Quantity)
        - 'ion_confinement_time' (u.Quantity)
        - 'electron_confinement_time' (u.Quantity)
        - 'particle_confinement_time' (u.Quantity)
        - 'volume' (u.Quantity)
        - 'fuel' (Fuel)
        - 'magnetic_field' (u.Quantity)
        - 'major_radius' (u.Quantity)
        - 'minor_radius' (u.Quantity)
        Optional keys include:
        - 'ratio' (float): Reactant ratio, defaults to 0.5.
        - 'reactivity_enhancement_factor' (float): For p-B11, defaults to 1.0.
        - 'cycle' (str): The fuel cycle to use. Can be 'simple' (default)
          or 'catalyzed_dd'.

        Returns
        -------
        ZeroDResults
            A dataclass containing all calculated output values, such as
            fusion power, loss powers, required heating, and Q value.

        Raises
        ------
        ConfigurationError
            If any of the required parameters are missing.
        """
        required = [
            'ion_temperature', 'electron_temperature', 'ion_density', 
            'ion_confinement_time', 'electron_confinement_time', 'particle_confinement_time',
            'volume', 'fuel', 'magnetic_field', 'major_radius', 'minor_radius'
        ]
        if not all(key in self.params for key in required):
            raise ConfigurationError(f"Missing one or more required parameters for 2T model: {set(required) - set(self.params.keys())}")

        p = self.params
        T_i, T_e = p['ion_temperature'], p['electron_temperature']
        n_i, V = p['ion_density'], p['volume']
        tau_Ei, tau_Ee = p['ion_confinement_time'], p['electron_confinement_time']
        tau_p = p['particle_confinement_time']
        fuel, ratio = p['fuel'], p.get('ratio', 0.5)
        reactivity_enhancement_factor = p.get('reactivity_enhancement_factor', 1.0)
        cycle = p.get('cycle', 'simple')
        
        if cycle == 'catalyzed_dd':
            if fuel.name != 'D-D':
                raise ConfigurationError("Catalyzed cycle can only be used with D-D fuel.")
            
            # --- CATALYZED D-D PARTICLE BALANCE ---
            cat_results = solve_catalyzed_dd_composition(n_i, T_i, tau_p)
            z_eff = cat_results['z_eff']
            ash_fraction = (cat_results['ash_density'] / n_i).to_value(u.dimensionless_unscaled)
            
            # --- FUSION POWER and CHARGED POWER (Catalyzed) ---
            # Power is sum of power from DD, DT, and DHe3 reactions
            n_D = cat_results['D_density']
            n_T = cat_results['T_density']
            n_He3 = cat_results['He3_density']

            sv_dd = calculate_reactivity(FUEL_DD, T_i)
            sv_dt = calculate_reactivity(FUEL_DT, T_i)
            sv_dhe3 = calculate_reactivity(FUEL_DHE3, T_i)
            
            rate_dd = 0.5 * n_D**2 * sv_dd
            rate_dt = n_D * n_T * sv_dt
            rate_dhe3 = n_D * n_He3 * sv_dhe3

            p_fusion_dd = (rate_dd * FUEL_DD.energy_per_reaction * V).to(u.W)
            p_fusion_dt = (rate_dt * FUEL_DT.energy_per_reaction * V).to(u.W)
            p_fusion_dhe3 = (rate_dhe3 * FUEL_DHE3.energy_per_reaction * V).to(u.W)

            p_charged_dd = p_fusion_dd * FUEL_DD.charged_particle_fraction
            p_charged_dt = p_fusion_dt * FUEL_DT.charged_particle_fraction
            p_charged_dhe3 = p_fusion_dhe3 * FUEL_DHE3.charged_particle_fraction
            
            p_fusion = p_fusion_dd + p_fusion_dt + p_fusion_dhe3
            p_charged = p_charged_dd + p_charged_dt + p_charged_dhe3
            
            # This burnup fraction is not well-defined for a catalyzed cycle, placeholder
            burnup_fraction = 0.0

        else: # Simple cycle
            # --- PARTICLE BALANCE ---
            ash_fraction, burnup_fraction, z_eff = solve_steady_state_composition(
                n_i, T_i, tau_p, fuel, ratio, reactivity_enhancement_factor
            )
            
            # Dilute fuel density by ash fraction
            n_fuel = n_i * (1 - ash_fraction)
            
            # --- HEATING and SOURCE TERMS ---
            p_fusion, p_charged = calculate_fusion_power(n_fuel, T_i, V, fuel, ratio, reactivity_enhancement_factor=reactivity_enhancement_factor)

        # --- ELECTRON DENSITY AND LOSSES (COMMON TO BOTH CYCLES) ---
        n_e = n_i * z_eff
        
        # Partition charged particle power
        p_alpha_i = p_charged * fuel.alpha_heating_fractions[0]
        p_alpha_e = p_charged * fuel.alpha_heating_fractions[1]
        # Assume auxiliary heating is 100% to ions unless specified otherwise
        p_aux_i = p.get('aux_heat_ions', 0*u.W)
        p_aux_e = p.get('aux_heat_electrons', 0*u.W)
        
        # --- LOSS and EXCHANGE TERMS ---
        p_ie_exchange = calculate_ion_electron_exchange(n_i, T_i, n_e, T_e, V, fuel)
        p_brems = calculate_bremsstrahlung_power(n_e, T_e, z_eff) * V
        p_synch = calculate_synchrotron_power(n_e, T_e, p['magnetic_field'], p['major_radius'], p['minor_radius'])
        
        ion_thermal_energy = 1.5 * n_i * T_i.to(u.J, equivalencies=u.temperature_energy()) * V
        p_loss_ions_transport = ion_thermal_energy / tau_Ei
        
        electron_thermal_energy = 1.5 * n_e * T_e.to(u.J, equivalencies=u.temperature_energy()) * V
        p_loss_electrons_transport = electron_thermal_energy / tau_Ee

        # --- SOLVE THE POWER BALANCE ---
        # We need to find the auxiliary heating that creates a steady state.
        # For simplicity in this non-solver model, we calculate the required heat.
        
        # Ion power balance: P_in = P_out
        # P_alpha_i + P_aux_i = P_loss_ions_transport + P_ie_exchange
        required_aux_i = (p_loss_ions_transport + p_ie_exchange - p_alpha_i).to(u.W)
        
        # Electron power balance: P_in = P_out
        # P_alpha_e + P_aux_e + P_ie_exchange = P_loss_electrons_transport + P_brems + P_synch
        required_aux_e = (p_loss_electrons_transport + p_brems + p_synch - p_alpha_e - p_ie_exchange).to(u.W)

        total_required_heating = max(0*u.W, required_aux_i) + max(0*u.W, required_aux_e)
        q_plasma = (p_fusion / total_required_heating).value if total_required_heating > 0*u.W else float('inf')

        return ZeroDResults(
            ion_temperature=T_i, electron_temperature=T_e,
            fusion_power=p_fusion, charged_particle_power=p_charged,
            bremsstrahlung_power=p_brems, synchrotron_power=p_synch,
            ion_confinement_loss=p_loss_ions_transport,
            electron_confinement_loss=p_loss_electrons_transport,
            total_loss_power = p_loss_ions_transport + p_loss_electrons_transport + p_brems + p_synch,
            required_heating_power=total_required_heating,
            fusion_gain_q=q_plasma,
            ion_electron_exchange_power=p_ie_exchange,
            triple_product=calculate_triple_product(n_i, T_i, tau_Ei),
            ash_fraction=ash_fraction,
            fuel_burnup_fraction=burnup_fraction
        ) 