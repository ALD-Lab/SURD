surd documentation
==================

SURD measures the information that past observations of a set of variables
carry about the future of a target variable and splits it into redundant,
unique and synergistic contributions, together with the causality leak from
unobserved variables (Martínez-Sánchez, Arranz & Lozano-Durán, *Nature
Communications* 15, 9296, 2024). The subpackage ``surd.states`` resolves the
decomposition over the states of the variables (Martínez-Sánchez &
Lozano-Durán, *Communications Physics* 9, 15, 2025). The subpackage
``surd.neural`` estimates the specific mutual information with a classifier
instead of the histogram, for short records and many sources.

Install with ``pip install surd``. The README of the repository has the
installation options, a quick start and the list of examples.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/surd
   api/states
   api/neural
