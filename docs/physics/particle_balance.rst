Steady-State Particle Balance
=============================

To self-consistently model the plasma composition, FusionCAT solves the steady-state particle balance for each species, where particle sources are balanced by losses.

Simple Fuel Cycle
-----------------

For simple, single-fuel cycles (like D-T), the model considers the primary fuel ions and a generic "ash" product (e.g., Helium-4).

The balance equation for the ash density :math:`n_{ash}` is:

.. math::

   S_{ash} = L_{ash} \implies R_{fuse} = \frac{n_{ash}}{\tau_p}

Where:
- :math:`R_{fuse}` is the fusion reaction rate, which acts as the source of ash.
- :math:`\tau_p` is the particle confinement time, which governs the loss of ash via transport.

The solver iteratively finds the ash density :math:`n_{ash}` that satisfies this equation. This ash density then dilutes the fuel, reducing the reaction rate and creating a self-consistent equilibrium.

Catalyzed D-D Cycle
-------------------

For the more complex catalyzed D-D cycle, a multi-species model is used. The solver simultaneously balances the sources and losses for Deuterium (D), Tritium (T), and Helium-3 (³He).

- **Tritium Balance**: T is produced by the D(d,p)T reaction and is lost via both transport and its own D-T fusion burn-up.
- **Helium-3 Balance**: ³He is produced by the D(d,n)³He reaction and is lost via transport and its own D-³He fusion burn-up.

This creates a network of coupled equations that the solver solves iteratively to find the equilibrium densities of D, T, ³He, and the final reaction products. 