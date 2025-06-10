"""
Example script demonstrating the sensitivity analysis capabilities of FusionCAT.

This script performs both 1D and 2D parameter scans on a D-T fusion concept,
showing how to explore the parameter space and visualize the results.
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT
from fusioncat.analysis.scanner import run_scan
from fusioncat.plotting import plot_2d_scan, plot_3d_scan

def main():
    """
    This example demonstrates a 2D sensitivity scan to explore the operating
    space of a D-T tokamak, analyzing how fusion gain (Q) varies with
    ion temperature and density.
    """
    # 1. Define the "base case" for our reactor concept.
    # These are the parameters that will remain fixed during the scan.
    base_reactor = FusionConcept(name="ARC-Class Tokamak")
    base_reactor.set_parameters(
        # Scanned parameters will be overridden, so initial value doesn't matter
        ion_temperature=1 * u.keV, 
        ion_density=1e20 * u.m**-3,
        
        # Fixed parameters
        confinement_time=2.5 * u.s,
        volume=200 * u.m**3,
        magnetic_field=9.0 * u.T,
        major_radius=3.3 * u.m,
        fuel=FUEL_DT
    )

    # 2. Define the parameter space to scan over.
    # We will scan ion temperature and ion density.
    scan_parameters = {
        'ion_temperature': np.linspace(10, 35, 30) * u.keV,
        'ion_density': np.linspace(0.5, 3.0, 30) * 1e20 / u.m**3
    }

    # 3. Run the sensitivity scan.
    # We want to see how the fusion gain (Q) changes.
    scan_results = run_scan(
        base_concept=base_reactor,
        scan_parameters=scan_parameters,
        output_metric='fusion_gain_q'
    )

    # 4. Plot the results.
    # A 2D contour plot is excellent for visualizing the operating space.
    plot_2d_scan(
        scan_results, 
        log_scale=True, 
        save_path="Q_contour_plot.png"
    )
    
    # A 3D surface plot gives a different perspective.
    plot_3d_scan(
        scan_results,
        log_scale=True,
        save_path="Q_surface_plot.png"
    )

if __name__ == "__main__":
    main() 