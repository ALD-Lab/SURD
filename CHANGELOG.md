# Changelog

## 1.2.0 (2026-10-30)

- `surd.states`: causality by states (Martínez-Sánchez & Lozano-Durán,
  *Communications Physics* 9, 15, 2025). The code of the SURD-states
  repository is now a subpackage, so one installation serves both papers:
  `states.surd_states`, `states.run`, `states.plot`, `states.plot_states`,
  `states.plot_states_3d` and `states.nice_print`, with the same
  signatures and the same numbers as SURD-states. The subpackage keeps
  its own `states.it_tools` and `states.analytic_eqs`, since they differ
  from the ones of `surd` (no regularization of the histogram, benchmark
  systems `source` and `target`). It loads on first access, so
  `import surd` is unchanged.
- `surd.states.datasets.load_inner_outer` downloads the boundary layer
  record of example E03 (148 MB, two files) on first use into
  `~/.cache/surd` or `$SURD_DATA` and checks its SHA-256. The data stay
  out of the package.
- Examples of SURD-states in `examples/states/` (E01 to E03), as
  notebooks and as marimo files. Only the import and data-loading cells
  changed. Outputs are the original ones.
- New extras: `states` (scikit-image for `plot_states_3d`, scikit-learn
  for `it_tools.hist_knn`) and `docs` (Sphinx). The `examples` extra now
  includes marimo.
- Tests: `surd.states` against reference outputs of the original code
  (logic gates, random histograms with two and three sources, the
  benchmark system of E01), the dataset helper, and a slow test
  (`pytest -m slow`) that reproduces E03 on the full record.
- Sphinx documentation in `docs/` with the API of `surd` and
  `surd.states`.
- `CITATION.cff` lists the Communications Physics paper under
  `references`. README with a section on causality by states and both
  citations.

## 1.1.0 (2026-10-30)

- `surd` is an installable package with a `src/` layout and a
  `pyproject.toml`. `pip install .` works from a clone. The public functions
  keep their names, signatures and results: `surd.surd`, `surd.surd_hd`,
  `surd.run`, `surd.run_parallel`, `surd.run_multiple_lags`, `surd.plot`,
  `surd.plot_nlabels`, `surd.plot_multiple_lags` and `surd.nice_print`.
- `it_tools`, `analytic_eqs` and `transport_map` are submodules
  (`surd.it_tools`, `surd.analytic_eqs`, `surd.transport_map`). The last
  two load on first access, so `import surd` does not need `mpart` and
  does not seed the NumPy random generator.
- Optional extras: `parallel` (pymp-pypi, Linux and macOS), `transport`
  (mpart) and `examples` (jupyterlab).
- `utils/` keeps working for scripts that add it to `sys.path`. Its files
  now load the package.
- The example notebooks import the installed package and fall back to
  `../src` from a plain clone.
- Test suite (`pytest`) with the logic gates of example E01, the benchmark
  systems of E02, the information-theoretic tools, bounds of the terms and
  the causality leak, `surd.run_parallel` with and without pymp, and the
  equivalence between the `utils/` import path and the package.
- Continuous integration on Linux, macOS and Windows with Python 3.9 and
  3.12, and a release workflow that builds and publishes to PyPI.
- `CITATION.cff` with the Nature Communications paper as preferred
  citation.
- Portable `environment.yml` (only the packages the code uses) and a
  `requirements.txt` for pip.
- pymp is optional. Without it, `surd.run_parallel` runs the targets one
  after another and returns the same result. This makes the code run on
  Windows.
- README rewritten: installation, quick start, table of examples and the
  main functions.

## 1.0.1 (2024-09-11)

- Fixes: file names of the turbulent boundary layer data (example E10) and
  README corrections.

## 1.0.0 (2024-09-03)

- Initial release.
