# fusioncat/core.py
import astropy.units as u
import astropy.constants as const
from .utils.exceptions import ConfigurationError
from .utils.results import ZeroDResults
from .physics.fusion import calculate_fusion_power
from .physics.radiation import calculate_bremsstrahlung_power, calculate_synchrotron_power, get_ion_species
from .physics.lawson import calculate_triple_product, calculate_ion_electron_exchange

class FusionConcept:
    def __init__(self, name: str):
        self.name = name; self.params = {}
    def set_parameters(self, **kwargs): self.params = kwargs
    def run_0d_analysis(self) -> ZeroDResults:
        required = ['ion_temperature', 'electron_temperature', 'ion_density', 'ion_confinement_time', 'electron_confinement_time', 'volume', 'fuel', 'magnetic_field', 'major_radius', 'minor_radius']
        if not all(key in self.params for key in required): raise ConfigurationError(f"Missing parameters: {set(required) - set(self.params.keys())}")
        
        p = self.params
        if p['ion_temperature'].value <= 0 or p['electron_temperature'].value <= 0: raise ValueError("Temperatures must be positive.")
        if p['ion_density'].value <= 0: raise ValueError("Density must be positive.")
        if p['minor_radius'].value >= p['major_radius'].value: raise ValueError("Minor radius must be less than major radius.")

        T_i, T_e, n_i = p['ion_temperature'], p['electron_temperature'], p['ion_density']
        tau_Ei, tau_Ee = p['ion_confinement_time'], p['electron_confinement_time']
        V, B, R, a = p['volume'], p['magnetic_field'], p['major_radius'], p['minor_radius']
        fuel, ratio = p['fuel'], p.get('ratio', 0.5)
        react_factor = p.get('reactivity_enhancement_factor', 1.0)
        
        ion_populations = get_ion_species(fuel, ratio, n_i)
        n_e = sum(n * z for n, z in ion_populations)
        
        p_fusion, p_charged = calculate_fusion_power(n_i, T_i, V, fuel, ratio, react_factor)
        p_brems = calculate_bremsstrahlung_power(n_e, T_e, V, ion_populations)
        p_synch = calculate_synchrotron_power(n_e, T_e, B, R, a)
        p_ie_exchange = calculate_ion_electron_exchange(ion_populations, T_i, n_e, T_e, V, fuel)

        ion_thermal_energy = sum(1.5 * n * T_i for n, z in ion_populations) * V
        p_loss_ions_transport = ion_thermal_energy.to(u.J) / tau_Ei
        
        electron_thermal_energy = 1.5 * n_e * T_e.to(u.J, equivalencies=u.temperature_energy()) * V
        p_loss_electrons_transport = electron_thermal_energy / tau_Ee
        
        f_alpha_i, f_alpha_e = fuel.alpha_heating_fractions
        p_alpha_i = p_charged * f_alpha_i; p_alpha_e = p_charged * f_alpha_e
        
        required_aux_i = p_loss_ions_transport + p_ie_exchange - p_alpha_i
        required_aux_e = p_loss_electrons_transport + p_brems + p_synch - p_alpha_e - p_ie_exchange
        
        total_required_heating = (max(0, required_aux_i.to_value(u.W)) + max(0, required_aux_e.to_value(u.W))) * u.W
        q_plasma = (p_fusion / total_required_heating).value if total_required_heating > 0*u.W else float('inf')

        return ZeroDResults(
            ion_temperature=T_i, electron_temperature=T_e, fusion_power=p_fusion,
            charged_particle_power=p_charged, bremsstrahlung_power=p_brems, synchrotron_power=p_synch,
            ion_confinement_loss=p_loss_ions_transport, electron_confinement_loss=p_loss_electrons_transport,
            total_loss_power = p_loss_ions_transport + p_loss_electrons_transport + p_brems + p_synch,
            required_heating_power=total_required_heating, fusion_gain_q=q_plasma,
            ion_electron_exchange_power=p_ie_exchange,
            triple_product=calculate_triple_product(n_i, T_i, tau_Ei))