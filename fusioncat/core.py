# fusioncat/core.py
import astropy.units as u
import astropy.constants as const
import numpy as np
from .utils.exceptions import ConfigurationError
from .utils.results import ZeroDResults, OneDResults
from .physics.fusion import calculate_fusion_power
from .physics.radiation import calculate_bremsstrahlung_power, calculate_synchrotron_power, get_ion_species
from .physics.lawson import calculate_triple_product, calculate_ion_electron_exchange
from .solvers.one_d import solve_steady_state_1d
from .profiles.shapes import parabolic_profile

class FusionConcept:
    def __init__(self, name: str):
        self.name = name; self.params = {}
    def set_parameters(self, **kwargs): self.params = kwargs
    def run_0d_analysis(self) -> ZeroDResults:
        required = [
            'ion_temperature', 'electron_temperature', 'ion_density', 
            'ion_confinement_time', 'electron_confinement_time',
            'volume', 'fuel', 'magnetic_field', 'major_radius', 'minor_radius'
        ]
        if not all(key in self.params for key in required):
            raise ConfigurationError(f"Missing one or more required parameters for 2T model: {required}")

        p = self.params
        T_i, T_e = p['ion_temperature'], p['electron_temperature']
        n_i, V = p['ion_density'], p['volume']
        tau_Ei, tau_Ee = p['ion_confinement_time'], p['electron_confinement_time']
        fuel, ratio = p['fuel'], p.get('ratio', 0.5)
        
        # Assume quasi-neutrality for electron density
        n_e = n_i 

        # --- HEATING and SOURCE TERMS ---
        p_fusion, p_charged = calculate_fusion_power(n_i, T_i, V, fuel, ratio)
        # Partition charged particle power
        p_alpha_i = p_charged * fuel.alpha_heating_fractions[0]
        p_alpha_e = p_charged * fuel.alpha_heating_fractions[1]
        # Assume auxiliary heating is 100% to ions unless specified otherwise
        p_aux_i = p.get('aux_heat_ions', 0*u.W)
        p_aux_e = p.get('aux_heat_electrons', 0*u.W)
        
        # --- LOSS and EXCHANGE TERMS ---
        p_ie_exchange = calculate_ion_electron_exchange(n_i, T_i, n_e, T_e, V, fuel)
        charge_map = {'D+': 1.0, 'e-': 1.0}  # For D-T, assume D+ ions and electrons
        p_brems = calculate_bremsstrahlung_power(n_e, T_e, charge_map) * V
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
            triple_product=calculate_triple_product(n_i, T_i, tau_Ei)
        )

    def run_1d_analysis(self, num_points=101, T_alpha=1.5, n_alpha=1.0):
        """
        Runs the 1D steady-state power balance analysis.
        """
        # Extract base parameters
        a = self.params['minor_radius']
        T_i_core, T_e_core = self.params['ion_temperature'], self.params['electron_temperature']
        n_i_core = self.params['ion_density']
        chi_i, chi_e = self.params['ion_diffusivity'], self.params['electron_diffusivity']
        fuel = self.params['fuel']

        # Create radial grid
        r_grid_norm = np.linspace(0, 1, num_points)
        r_grid = r_grid_norm * a

        # Generate plasma profiles
        realistic_edge_temp_keV = 0.01  # 10 eV
        T_i_profile = parabolic_profile(r_grid_norm, T_i_core.to_value(u.keV), realistic_edge_temp_keV, T_alpha) * u.keV
        T_e_profile = parabolic_profile(r_grid_norm, T_e_core.to_value(u.keV), realistic_edge_temp_keV, T_alpha) * u.keV
        
        n_i_profile = parabolic_profile(r_grid_norm, n_i_core.to_value(u.m**-3), 1e18, n_alpha) / u.m**3
        
        # --- THIS IS THE CORRECTED SECTION ---
        # The solver calculates n_e internally based on n_i. We do not need to pass it.
        # The call now correctly has 7 arguments.
        (
            p_fusion, p_brems, p_ie, p_trans_i, p_trans_e,
            p_heat_i, p_heat_e
        ) = solve_steady_state_1d(
            r_grid, T_i_profile, T_e_profile, n_i_profile,
            chi_i, chi_e, fuel
        )
        # ------------------------------------
        
        return OneDResults(
            radius_grid=r_grid, T_i_profile=T_i_profile, T_e_profile=T_e_profile,
            n_i_profile=n_i_profile,
            # For completeness, we add n_e_profile to the results object here
            n_e_profile=n_i_profile, 
            fusion_power_profile=p_fusion, bremsstrahlung_power_profile=p_brems,
            ion_heat_flux_profile=p_trans_i, electron_heat_flux_profile=p_trans_e,
            ion_electron_exchange_profile=p_ie, ion_heating_profile=p_heat_i,
            electron_heating_profile=p_heat_e
        ) 