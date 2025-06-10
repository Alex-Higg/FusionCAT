# fusioncat/plotting.py
import matplotlib.pyplot as plt
import astropy.units as u
from .utils.results import ZeroDResults

def plot_power_balance(results: ZeroDResults, name: str, save_path: str = None):
    """Generates a bar chart comparing power sources and losses."""
    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sources = {'Self-Heating': results.charged_particle_power.to_value(u.MW), 'External Heating': results.required_heating_power.to_value(u.MW)}
    losses = {'Bremsstrahlung': results.bremsstrahlung_power.to_value(u.MW), 'Synchrotron': results.synchrotron_power.to_value(u.MW), 'Transport': results.confinement_loss_power.to_value(u.MW)}

    ax.bar(sources.keys(), sources.values(), label='Power Sources', color='g', width=0.4)
    ax.bar(losses.keys(), losses.values(), label='Power Losses', color='r', width=0.4, alpha=0.7)
    
    ax.set_ylabel('Power (MW)')
    ax.set_title(f'Power Balance for {name} (Q = {results.fusion_gain_q:.2f})')
    ax.legend()
    fig.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show() 