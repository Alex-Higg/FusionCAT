"""
Provides tools for performing 1D and 2D sensitivity scans on FusionConcept objects.
"""
import copy
import itertools
from dataclasses import dataclass

import numpy as np
import astropy.units as u

from ..core import FusionConcept
from ..utils.results import ZeroDResults

@dataclass(frozen=True)
class ScanResults:
    """
    A structured container for the results of a 1D or 2D parameter scan.

    Attributes
    ----------
    scan_parameters : list[str]
        The names of the parameters that were scanned.
    x_grid : np.ndarray
        The grid of values for the first scanned parameter.
    y_grid : np.ndarray | None
        The grid of values for the second scanned parameter (for 2D scans).
    results_grid : np.ndarray
        A grid containing the output metric at each point in the scan.
    output_metric : str
        The name of the attribute from ZeroDResults that was recorded.
    x_unit : astropy.units.Unit
        The unit of the first scanned parameter.
    y_unit : astropy.units.Unit | None
        The unit of the second scanned parameter.
    results_unit : astropy.units.Unit | None
        The unit of the output metric, if it has one.
    """
    scan_parameters: list[str]
    x_grid: np.ndarray
    y_grid: np.ndarray | None
    results_grid: np.ndarray
    output_metric: str
    x_unit: u.Unit
    y_unit: u.Unit | None
    results_unit: u.Unit | None


def run_scan(
    base_concept: FusionConcept,
    scan_parameters: dict[str, np.ndarray],
    output_metric: str = "fusion_gain_q",
) -> ScanResults:
    """
    Performs a 1D or 2D parameter scan on a FusionConcept.

    Args:
        base_concept: A fully configured `FusionConcept` object that will
                      be used as the template for the scan.
        scan_parameters: A dictionary where keys are the parameter names to scan
                         (e.g., 'ion_temperature') and values are NumPy arrays
                         or astropy Quantities of the values to scan over.
                         Limited to 1 or 2 parameters.
        output_metric: The attribute name of the `ZeroDResults` to record
                       (e.g., 'fusion_gain_q', 'fusion_power').

    Returns:
        A `ScanResults` object containing the grids of data.
    """
    if not 1 <= len(scan_parameters) <= 2:
        raise ValueError("This scanner supports 1D or 2D scans only.")

    param_names = list(scan_parameters.keys())
    param_values = list(scan_parameters.values())

    # Create the grid of parameter combinations to iterate over
    param_combinations = list(itertools.product(*param_values))
    
    # Pre-allocate results array
    results_shape = [len(arr) for arr in param_values]
    results_grid = np.zeros(results_shape)
    
    # Store units for later
    x_unit = param_values[0].unit if hasattr(param_values[0], 'unit') else None
    y_unit = param_values[1].unit if len(param_values) > 1 and hasattr(param_values[1], 'unit') else None
    results_unit = None

    print(f"Starting {len(param_names)}D scan of '{output_metric}' over {param_names}...")
    print(f"Total points to calculate: {len(param_combinations)}")

    for i, combo in enumerate(param_combinations):
        # Create a deep copy of the base parameters to avoid modifying the original
        current_params = copy.deepcopy(base_concept.params)

        # Update the parameters for the current point in the scan
        for name, value in zip(param_names, combo):
            current_params[name] = value

        # Run the analysis for this single point
        temp_concept = FusionConcept(name="scan_point")
        temp_concept.set_parameters(**current_params)
        
        try:
            results: ZeroDResults = temp_concept.run_0d_analysis()
            
            # Get the desired output metric from the results object
            output_value = getattr(results, output_metric)
            
            # Store the unit from the first successful run
            if i == 0 and hasattr(output_value, 'unit'):
                results_unit = output_value.unit

            # Get the numerical value (in its original unit)
            value_to_store = output_value.value if hasattr(output_value, 'unit') else output_value

            # Find the correct index in the multi-dimensional results grid
            indices = tuple(np.where(arr == val)[0][0] for arr, val in zip(param_values, combo))
            results_grid[indices] = value_to_store

        except Exception as e:
            # If a point fails (e.g., physically impossible), store NaN
            indices = tuple(np.where(arr == val)[0][0] for arr, val in zip(param_values, combo))
            results_grid[indices] = np.nan
            print(f"Warning: Calculation failed for combo {combo}: {e}")

    # Create meshgrid for plotting
    x_vals = param_values[0].value if hasattr(param_values[0], 'unit') else param_values[0]
    if len(param_names) == 2:
        y_vals = param_values[1].value if hasattr(param_values[1], 'unit') else param_values[1]
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        # Note: The loop calculates results in (y,x) order, so we transpose.
        results_grid = results_grid.T
    else:
        x_grid, y_grid = x_vals, None

    print("Scan complete.")
    return ScanResults(
        scan_parameters=param_names,
        x_grid=x_grid,
        y_grid=y_grid,
        results_grid=results_grid,
        output_metric=output_metric,
        x_unit=x_unit,
        y_unit=y_unit,
        results_unit=results_unit
    ) 