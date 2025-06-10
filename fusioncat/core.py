# fusioncat/core.py
import astropy.units as u
import astropy.constants as const
from .utils.exceptions import ConfigurationError
from .utils.results import ZeroDResults
from .physics.fusion import calculate_fusion_power
from .physics.radiation import calculate_bremsstrahlung_power, calculate_synchrotron_power, get_ion_species
from .physics.lawson import calculate_triple_product

class FusionConcept:
    def __init__(self, name: str):
        self.name = name; self.params = {}
    def set_parameters(self, **kwargs): self.params = kwargs
    def run_0d_analysis(self) -> ZeroDResults:
        required = ['ion_temperature', 'ion_density', 'confinement_time', 'volume', 'fuel', 'magnetic_field', 'major_radius']
        if not all(key in self.params for key in required):
            raise ConfigurationError(f"Missing one or more required parameters: {required}")

        p = self.params
        T_i, n_i, tau_E = p['ion_temperature'], p['ion_density'], p['confinement_time']
        V, B, R = p['volume'], p['magnetic_field'], p['major_radius']
        fuel, ratio = p['fuel'], p.get('ratio', 0.5)
        T_e, n_e = T_i, n_i

        # Get ion species populations and calculate electron density via quasi-neutrality
        ion_populations = get_ion_species(fuel, ratio, n_i)
        n_e = sum(n * z for n, z in ion_populations)

        # Call physics modules
        p_fusion, p_charged = calculate_fusion_power(n_i, T_i, V, fuel, ratio)
        p_brems = calculate_bremsstrahlung_power(n_e, T_e, V, Z_eff=1)
        p_synch = calculate_synchrotron_power(n_e, T_e, B, R)
        
        # --- CORRECTED THERMAL ENERGY CALCULATION ---
        # The formula is W_th = 1.5 * n * T. If T is in energy units (like keV),
        # we do not need the Boltzmann constant k_B.
        # We also sum the ion and electron thermal energies.
        thermal_energy_density = 1.5 * n_i * T_i + 1.5 * n_e * T_e
        thermal_energy = thermal_energy_density.to(u.J / u.m**3) * V
        p_confinement_loss = (thermal_energy / tau_E).to(u.W)
        
        p_loss_total = p_brems + p_synch + p_confinement_loss
        p_heating_required = max(0 * u.W, p_loss_total - p_charged)
        q_plasma = (p_fusion / p_heating_required).value if p_heating_required > 0 * u.W else float('inf')

        triple_product = calculate_triple_product(n_i, T_i, tau_E)

        return ZeroDResults(
            fusion_power=p_fusion, charged_particle_power=p_charged,
            bremsstrahlung_power=p_brems, synchrotron_power=p_synch,
            confinement_loss_power=p_confinement_loss, total_loss_power=p_loss_total,
            required_heating_power=p_heating_required, fusion_gain_q=q_plasma,
            triple_product=triple_product
        ) 