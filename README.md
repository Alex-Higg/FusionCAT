# FusionCAT: Fusion Concept Analysis Tool

**FusionCAT** is a Python-based engineering and physics tool for 0D and 1D analysis of fusion energy concepts. It provides a flexible, user-friendly framework for modeling plasma performance, power balance, and particle composition.

[![Pytest](https://github.com/YourUsername/FusionCAT/actions/workflows/pytest.yml/badge.svg)](https://github.com/YourUsername/FusionCAT/actions/workflows/pytest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Mission Statement

The goal of FusionCAT is to provide a transparent, physically-grounded, and easy-to-use tool for researchers, engineers, and students to quickly evaluate and compare different fusion concepts. By building on robust, community-standard libraries like `astropy`, FusionCAT aims to be both a practical design tool and an educational resource.

## Installation

FusionCAT can be installed directly from this repository using `pip`. It is recommended to do this within a virtual environment.

```bash
# Clone the repository
git clone https://github.com/YourUsername/FusionCAT.git
cd FusionCAT

# Install the package in editable mode
pip install -e .
```

## Quick Start: 0D Power Balance

Here is a simple example of how to perform a zero-dimensional, two-temperature (2T) power balance analysis for a D-T tokamak.

```python
import astropy.units as u
from fusioncat.core import FusionConcept
from fusioncat.utils.fuels import FUEL_DT
from fusioncat.utils.results import ZeroDResults

# 1. Initialize the fusion concept
concept = FusionConcept(name="MyFirstTokamak")

# 2. Set the plasma and machine parameters
#    Note the use of astropy.units to ensure physical consistency.
concept.set_parameters(
    ion_temperature=20 * u.keV,
    electron_temperature=18 * u.keV,
    ion_density=1.5e20 / u.m**3,
    ion_confinement_time=3.0 * u.s,
    electron_confinement_time=2.8 * u.s,
    particle_confinement_time=5.0 * u.s, # Key parameter for ash buildup
    volume=520 * u.m**3,
    fuel=FUEL_DT,
    magnetic_field=5.3 * u.T,
    major_radius=3.1 * u.m,
    minor_radius=1.1 * u.m
)

# 3. Run the 0D analysis
results: ZeroDResults = concept.run_0d_analysis()

# 4. Print the key results in a user-friendly format
print(f"--- Results for {concept.name} ---")
print(f"Fusion Power: {results.fusion_power.to(u.MW):.2f}")
print(f"Required Heating Power: {results.required_heating_power.to(u.MW):.2f}")
print(f"Plasma Q: {results.fusion_gain_q:.2f}")
print("-" * 20)
print(f"Ash Fraction: {results.ash_fraction:.3f}")
print(f"Fuel Burn-up Fraction: {results.fuel_burnup_fraction:.3f}")
print("-" * 20)
print("Power Losses:")
print(f"  - Ion Transport: {results.ion_confinement_loss.to(u.MW):.2f}")
print(f"  - Electron Transport: {results.electron_confinement_loss.to(u.MW):.2f}")
print(f"  - Bremsstrahlung: {results.bremsstrahlung_power.to(u.MW):.2f}")
print(f"  - Synchrotron: {results.synchrotron_power.to(u.MW):.2f}")
```

This example demonstrates how to define a concept, run the solver, and access the rich set of output parameters available in the `ZeroDResults` object. 