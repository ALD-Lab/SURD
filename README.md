# SURD: Synergistic-Unique-Redundant Decomposition of causality

[![tests](https://github.com/ALD-Lab/SURD/actions/workflows/tests.yml/badge.svg)](https://github.com/ALD-Lab/SURD/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/surd.svg)](https://pypi.org/project/surd/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Nat.%20Commun.-10.1038%2Fs41467--024--53373--4-blue)](https://doi.org/10.1038/s41467-024-53373-4)

SURD is a method for causal inference in time series. It measures the information that past observations of a set of variables carry about the future of a target variable and splits it into redundant, unique and synergistic contributions. The method also quantifies the causality leak, the part of the future of the target that the observed variables cannot explain. SURD needs only pairs of past and future events, so it applies to simulations and experiments alike. It handles several variables, dependencies at different time lags and instantaneous links. The method is described in Martínez-Sánchez, Arranz & Lozano-Durán, *Nature Communications* 15, 9296 (2024).

![SURD diagram](https://github.com/user-attachments/assets/937fa18d-1179-4cfe-ae5a-e262582614a0)

## Installation

From PyPI:
```sh
pip install surd
```

From a clone of this repository:
```sh
git clone https://github.com/ALD-Lab/SURD.git
cd SURD
pip install .
```

With conda, the environment file installs the dependencies and the package:
```sh
conda env create -f environment.yml
conda activate surd
```

SURD depends on NumPy, SciPy, Matplotlib and tqdm and runs on Python 3.9 or newer, on Linux, macOS and Windows. Three optional extras exist:

- `pip install "surd[parallel]"` adds `pymp-pypi`, which runs the loop over target variables of `surd.run_parallel` in parallel (Linux and macOS). Without it the targets run one after another with the same result.
- `pip install "surd[transport]"` adds `mpart` for density estimation with transport maps (example E07). On `osx-arm64` use `pip install mpart`, since the conda package does not exist for that platform. See the [MParT documentation](https://measuretransport.github.io/MParT/).
- `pip install "surd[examples]"` adds JupyterLab to run the notebooks.

The cost of SURD grows with the number of variables and bins, since it estimates joint probability distributions. The examples run on a laptop with 16 GB of RAM.

## Quick start

Two independent bits and a target equal to their XOR one step later. Each bit alone says nothing about the target, so the information is synergistic.

```python
import numpy as np
import surd

rng = np.random.default_rng(0)
q1, q2 = rng.integers(0, 2, size=(2, 200_000))    # two independent bits in time
target = np.roll(q1 ^ q2, 1)                      # their XOR, one step later
Y = np.vstack([target[1:], q1[:-1], q2[:-1]])     # future of the target first, sources after
hist, _ = np.histogramdd(Y.T, bins=2)
I_R, I_S, MI, info_leak = surd.surd(hist)
surd.nice_print(I_R, I_S, MI, info_leak)
```

```
    Redundant (R):
        (1, 2)      : 0.0000
    Unique (U):
        (1,)        : 0.0000
        (2,)        : 0.0000
    Synergystic (S):
        (1, 2)      : 1.0000
    Information Leak:  0.00%
```

## How to use the method

The input is a histogram of the joint distribution of the future of the target and the present of the sources. Axis 0 is the target and axes 1 to N are the sources. The outputs are dictionaries keyed by tuples of source indices, counted from 1.

| Function | What it does |
|---|---|
| `surd.surd(p)` | Decomposition of the histogram `p`. Returns `I_R` (redundant terms, with the unique terms under the one-element keys), `I_S` (synergistic terms), `MI` (mutual information of each subset of sources with the target) and `info_leak`, the causality leak in [0, 1]. |
| `surd.surd_hd(Y, nbins, max_combs)` | Same decomposition from the signals `Y` (target first) for systems with many variables. Synergies are computed up to `max_combs` sources at a time. Returns `I_R`, `I_S` and `MI`. |
| `surd.run(X, nvars, nlag, nbins, axs)` | Runs `surd.surd` with each of the `nvars` signals of `X` as target at lag `nlag`, prints and plots the result. |
| `surd.run_parallel(...)` | Same as `surd.run`, with the targets in parallel when `pymp` is available. |
| `surd.run_multiple_lags(X, nvars, nlag, nbins, max_combs, axs)` | Same as `surd.run` with all lags from 1 to `nlag` as sources. |
| `surd.nice_print(I_R, I_S, MI, info_leak)` | Prints the terms normalized by the mutual information of the full set of sources and the leak in percent. |
| `surd.plot(I_R, I_S, info_leak, axs, nvars, threshold)` | Bar chart of the normalized terms (blue redundant, red unique, orange synergistic) and a separate bar for the leak. |

`surd.it_tools` has the entropies, conditional entropies, mutual information and transfer entropy used by the method. `surd.analytic_eqs` has the benchmark systems of the paper. `surd.transport_map` estimates densities with transport maps and needs `mpart`.

Scripts written for the previous layout, which added `utils/` to `sys.path` and imported `surd`, `it_tools`, `analytic_eqs` and `transport_map` as top-level modules, keep working. The files in `utils/` load the package.

## Examples

The notebooks in [`examples/`](examples) reproduce the cases of the paper. Run them from the `examples` folder. They import the installed package and fall back to `../src` from a plain clone.

| Notebook | What it shows |
|---|---|
| [E01](examples/E01_logic_gates.ipynb) | Logic gates. A duplicated source is redundant, a copied source is unique, an XOR is synergistic. |
| [E02](examples/E02_building_blocks.ipynb) | Building blocks of causal interactions: mediator, confounder, synergistic collider and redundant collider. |
| [E03](examples/E03_energy_cascade.ipynb) | Energy cascade in isotropic turbulence. Causality between the energy transfer at four scales from a direct numerical simulation. |
| [E04](examples/E04_lagged_dependencies.ipynb) | Stochastic systems with dependencies at several time lags, linear and nonlinear. |
| [E05](examples/E05_Moran_effect.ipynb) | Moran effect. Two populations driven by a common environmental forcing that is not observed. |
| [E06](examples/E06_logistic_system.ipynb) | Coupled logistic maps with one-way and two-way coupling. |
| [E07](examples/E07_transport_map.ipynb) | Density estimation with transport maps instead of histograms (needs `mpart`). |
| [E08](examples/E08_eight_species.ipynb) | System with eight interacting species. Decomposition with synergies truncated at a given order. |
| [E09](examples/E09_contemporaneous.ipynb) | Systems with contemporaneous links on top of mediator, confounder and synergistic interactions. |
| [E10](examples/E10_inner_outer.ipynb) | Experimental data from a turbulent boundary layer. Causality between inner-layer and outer-layer velocity motions as a function of the time lag. |

The data for E03 and E10 are in [`data/`](data). E02 writes the signals it generates to `data/` on the first run.

## Citation

If you use SURD in your research or software, please cite the paper:

```bibtex
@article{surd,
author={Mart{\'\i}nez-S{\'a}nchez, {\'A}lvaro and Arranz, Gonzalo and Lozano-Dur{\'a}n, Adri{\'a}n},
title={Decomposing causality into its synergistic, unique, and redundant components},
journal={Nature Communications},
year={2024},
month={Nov},
day={01},
volume={15},
number={1},
pages={9296},
issn={2041-1723},
doi={10.1038/s41467-024-53373-4}
}
```

The file [`CITATION.cff`](CITATION.cff) carries the same reference in machine-readable form.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
