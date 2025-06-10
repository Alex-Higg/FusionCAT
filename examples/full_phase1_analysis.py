# examples/full_phase1_analysis.py
import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT
from fusioncat.plotting import plot_power_balance

def main():
    reactor = FusionConcept(name="Example D-T Reactor")
    reactor.set_parameters(
        ion_temperature=22 * u.keV, ion_density=1.2e20 * u.m**-3,
        confinement_time=3.8 * u.s, volume=830 * u.m**3,
        magnetic_field=5.3 * u.T, major_radius=6.2*u.m, fuel=FUEL_DT
    )
    results = reactor.run_0d_analysis()
    print(results)
    plot_power_balance(results, reactor.name, save_path="power_balance.png")

if __name__ == "__main__":
    main() 