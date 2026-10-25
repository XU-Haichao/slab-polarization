# Polarized Thomson Scattering in a Slab

This repository accompanies our study of the Stokes-vector scattering-order
recursion framework in a plane-parallel slab. We provide an executable notebook,
numerical data, and plotting routines to reproduce the paper figures and
supporting validation tests, including comparisons with independent Monte Carlo
simulations and compPSc calculations.

## Getting started

Clone the complete repository, install the dependencies in a dedicated Python
environment, and run the notebook from top to bottom:

```bash
git clone https://github.com/XU-Haichao/slab-polarization.git
cd slab-polarization
python -m pip install -r requirements.txt
jupyter lab paper_companion.ipynb
```

Open [paper_companion.ipynb](paper_companion.ipynb). Default execution uses
bundled benchmark data and high-resolution caches; it does not launch MCRT,
XSPEC, compPSc, or remote jobs. No private compPSc installation is required.

See the [full usage and data guide](README.txt) for section descriptions,
recalculation controls, conventions, limitations and provenance.

For the observational analysis, open
[observational_analysis.ipynb](observational_analysis.ipynb). It documents the
4U 1624-49 observation IDs, source/background apertures, non-dip GTIs, extraction
settings, ungrouped spectra and adopted low-/high-reflection fit parameters.
Joint I/Q/U plots retain the three IXPE DUs separately. **Run All uses cached
products only: it performs no extraction, model recalculation or refitting.**
This lightweight processing record does not reconstruct raw telemetry or bundle
the original event lists, responses and calibration databases.
Section 9 demonstrates the actual fit/display binning rules, before/after I
grouping, and Q/U three-DU summation with propagated covariance. These display
operations are checked against the cached products and never alter the fit.

## Contents

- Scalar and Stokes scattering-order recursions and three seed-source configurations.
- Escape, survival and mean-scattering statistics with MCRT checks.
- Thin- and thick-limit eigenmodes, surface angular distributions and polarization.
- A supplementary sixth-order thick-limit diagnostic and full-matrix avoided crossings.
- MCRT, recursion + Jones and compPSc energy-spectrum comparisons.
- Parameter-scan diagnostics and low-/high-reflection Western-model fit exports
  for 4U 1624-49, with independent best-fit MCRT comparisons.

The notebook contains 81 cells (37 executable) and 34 displayed figures.
Seventeen paper/appendix PDFs are exported to [figure/](figure/); supporting
figures are displayed without additional exports.

## Numerical data

Keep [data/](data/), [helpers/](helpers/), [plot_style.py](plot_style.py) and
[public_manifest.json](public_manifest.json) alongside the notebook.
The manifest verifies all distributed data files, including the lightweight
[observational exports](data/observations/README.txt).

The [81-case selected-bin table](data/statistics/point_contributions.csv.gz)
contains 146,248 rows (approximately 6.5 MiB compressed), including model
predictions, MCRT means, standard errors, energy edges and chi-square
contributions. It includes compPSc numerical outputs, not private source code.
Read the [data dictionary](data/statistics/PER_BIN_DATA.txt) before using it:
these are the bins selected for the heatmaps, not the complete unfiltered scan.

## Interpretation and reproducibility

- The projected D/P channels are distinct from full-operator eigenbranches.
  Four full-matrix P0 avoided crossings are separately checked; the largest
  gap among them is approximately 1.2%, not a proven bound over all crossings.
- The supplementary k^6 polarization coefficient is not yet fully converged.
  Its sensitivity envelope is not a statistical confidence interval.
- compPSc iteration convergence does not establish full mesh convergence.
- The heatmaps use fixed parameters, absolute normalization and diagonal MCRT
  errors. Their chi-square values are not fractional spectral errors.

[execution_report.json](execution_report.json) records the full notebook run.
[release_audit.json](release_audit.json) records path cleanup and successful
execution from a relocated directory. Historical production-path identifiers
refer to unbundled inputs and are not runtime dependencies.

## Related software and attribution

- [MCRT](https://github.com/XU-Haichao/MCRT)
- [slab-comptonization](https://github.com/XU-Haichao/slab-comptonization)

Requests for access to the modified compPSc source may be directed to the
corresponding author. Raw observational event lists, instrument response files
and raw photon trajectories are not included.

Code organization and documentation for this companion were assisted by
GPT 6.1 sol. Please cite the accompanying paper and underlying methods/software.
Final paper/archive citation information will be added when available.

## Licensing status

The companion code/data licensing has not yet been finalized in this repository;
no blanket license is asserted. The separately released MCRT and
slab-comptonization packages retain their own licenses. This repository does
not relicense third-party software or astronomical data.
