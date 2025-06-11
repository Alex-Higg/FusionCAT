The 2T Power Balance
====================

The core of the 0D solver is a two-temperature (2T) power balance model, which solves the power balance equations for the ion and electron species independently.

Ion Power Balance
-----------------

The ion temperature is maintained by a balance between heating and loss terms.

.. math::

   P_{\alpha,i} + P_{aux,i} = P_{ie} + P_{loss,i}

Where:
- :math:`P_{\alpha,i}` is the fraction of fusion alpha particle power that heats the ions.
- :math:`P_{aux,i}` is the external auxiliary heating applied to the ions.
- :math:`P_{ie}` is the power transferred from ions to electrons via Coulomb collisions.
- :math:`P_{loss,i}` is the power lost from the ions due to transport, defined as :math:`E_i / \tau_{E,i}`.

Electron Power Balance
----------------------

Similarly, the electron temperature is maintained by its own power balance.

.. math::

   P_{\alpha,e} + P_{aux,e} + P_{ie} = P_{brem} + P_{synch} + P_{loss,e}

Where:
- :math:`P_{\alpha,e}` is the fraction of alpha power heating electrons.
- :math:`P_{aux,e}` is auxiliary heating for electrons.
- :math:`P_{ie}` is the collisional power transfer from ions.
- :math:`P_{brem}` is the Bremsstrahlung radiation loss.
- :math:`P_{synch}` is the Synchrotron radiation loss.
- :math:`P_{loss,e}` is the electron transport power loss, :math:`E_e / \tau_{E,e}`.

In the current model, the solver calculates the required auxiliary heating (:math:`P_{aux,i} + P_{aux,e}`) needed to achieve a steady state where the temperatures are constant. 