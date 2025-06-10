# fusioncat/plotting.py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.units as u
import numpy as np

from .utils.results import ZeroDResults
from .analysis.scanner import ScanResults

plt.style.use('seaborn-v0_8-talk')

def plot_power_balance(results: ZeroDResults, name: str, save_path: str = None):
    """Generates a bar chart comparing power sources and losses."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sources = {'Self-Heating': results.charged_particle_power.to(u.MW), 'External Heating': results.required_heating_power.to(u.MW)}
    losses = {'Bremsstrahlung': results.bremsstrahlung_power.to(u.MW), 'Synchrotron': results.synchrotron_power.to(u.MW), 'Transport': results.confinement_loss_power.to(u.MW)}
    ax.bar(sources.keys(), sources.values(), label='Power Sources', color='g', width=0.4)
    ax.bar(losses.keys(), losses.values(), label='Power Losses', color='r', width=0.4, alpha=0.7)
    ax.set_ylabel('Power (MW)')
    ax.set_title(f'Power Balance for {name} (Q = {results.fusion_gain_q:.2f})')
    ax.legend()
    fig.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()

def plot_1d_scan(scan_results: ScanResults, save_path: str = None):
    """Generates a line plot for a 1D sensitivity scan."""
    if scan_results.y_grid is not None: raise ValueError("This function is for 1D scans only.")
    fig, ax = plt.subplots(figsize=(10, 6))
    x_label = f"{scan_results.scan_parameters[0].replace('_', ' ').title()} [{scan_results.x_unit}]"
    y_label = f"{scan_results.output_metric.replace('_', ' ').title()}"
    if scan_results.results_unit: y_label += f" [{scan_results.results_unit}]"
    ax.plot(scan_results.x_grid, scan_results.results_grid)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(f"Sensitivity of {y_label} to {x_label}")
    ax.grid(True, linestyle=':'); fig.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()


def _sanitize_grid_for_plotting(grid: np.ndarray):
    """
    Internal helper to prepare data for plotting, especially log-scale plots.
    Replaces NaNs with 0 and infinities with a value slightly larger than the
    max finite value in the grid.
    """
    # Create a mask of only the finite values
    finite_mask = np.isfinite(grid)
    if not np.any(finite_mask):
        # If there's no valid data, return a grid of zeros
        return np.zeros_like(grid), 1, 100

    # Determine plotting limits based *only* on finite data
    max_finite_val = np.max(grid[finite_mask])
    
    # Replace non-finite values
    plot_grid = np.nan_to_num(grid, nan=0.0, posinf=max_finite_val * 1.1, neginf=0.0)
    
    # Determine minimum for log scale, avoiding zero or negative values
    positive_mask = (grid[finite_mask] > 0)
    min_log_val = np.min(grid[finite_mask][positive_mask]) if np.any(positive_mask) else max_finite_val * 1e-3

    return plot_grid, min_log_val, max_finite_val

def plot_2d_scan(scan_results: ScanResults, log_scale: bool = True, save_path: str = None):
    """
    Generates a 2D contour plot for a 2D sensitivity scan.
    """
    if scan_results.y_grid is None: raise ValueError("This function is for 2D scans only.")
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    x_label = f"{scan_results.scan_parameters[0].replace('_', ' ').title()} [{scan_results.x_unit}]"
    y_label = f"{scan_results.scan_parameters[1].replace('_', ' ').title()} [{scan_results.y_unit}]"
    z_label = f"{scan_results.output_metric.replace('_', ' ').title()}"
    if scan_results.results_unit: z_label += f" [{scan_results.results_unit}]"
        
    # Sanitize the data to handle NaNs and Infs before plotting
    Z, vmin, vmax = _sanitize_grid_for_plotting(scan_results.results_grid)
    
    norm = colors.LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    
    contour = ax.contourf(scan_results.x_grid, scan_results.y_grid, Z, levels=30, cmap='viridis', norm=norm)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(z_label)
    
    contour_lines = ax.contour(scan_results.x_grid, scan_results.y_grid, Z, levels=10, colors='white', alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%1.1f')
    
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(f"Contour Plot of {scan_results.output_metric.replace('_', ' ').title()}")
    fig.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()
    
def plot_3d_scan(scan_results: ScanResults, log_scale: bool = True, save_path: str = None):
    """Generates a 3D surface plot for a 2D sensitivity scan."""
    if scan_results.y_grid is None: raise ValueError("This function is for 2D scans only.")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x_label = f"{scan_results.scan_parameters[0].title()} [{scan_results.x_unit}]"
    y_label = f"{scan_results.scan_parameters[1].title()} [{scan_results.y_unit}]"
    z_label = f"{scan_results.output_metric.title()}"
    if scan_results.results_unit: z_label += f" [{scan_results.results_unit}]"
        
    Z, vmin, vmax = _sanitize_grid_for_plotting(scan_results.results_grid)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    surf = ax.plot_surface(scan_results.x_grid, scan_results.y_grid, Z, cmap='viridis', norm=norm, rcount=100, ccount=100)
    cbar = fig.colorbar(surf, shrink=0.6, aspect=10)
    cbar.set_label(z_label)
    
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_zlabel(z_label)
    ax.set_title(f"Surface Plot of {scan_results.output_metric.replace('_', ' ').title()}")
    fig.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()