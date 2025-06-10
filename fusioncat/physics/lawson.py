# fusioncat/physics/lawson.py
import astropy.units as u

def calculate_triple_product(n: u.Quantity, T: u.Quantity, tau_E: u.Quantity) -> u.Quantity:
    """Calculates the fusion triple product (n * T * tau_E)."""
    return (n * T * tau_E).to(u.s * u.m**-3 * u.keV) 