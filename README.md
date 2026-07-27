# configural-psychometrics

R and C++ engines for the **configural psychometrics** framework — identifying the minimally sufficient conditions under which a targeted outcome is expected.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21633129.svg)](https://doi.org/10.5281/zenodo.21633129)

Configural psychometrics asks which specific *combinations* of inputs are jointly sufficient to expect a targeted outcome, rather than how much each input contributes on its own. This repository holds the working engines and post-hoc diagnostics behind that framework. A full plain-language and technical account of the framework — with worked examples and applications — is on the project page:

**https://www.dynamicpsychlab.com/configural-psychometrics**

## What's here

Two search engines and a set of diagnostics computed on their output.

**Engines** build the *antichain* of minimally sufficient sets for a target at a sufficiency threshold θ:

- **exhaustive** (`enumerate_msc`) — a downward-closed ascent over the full subset lattice, pruning by support; provably complete wherever computable.
- **constructive** (`pairmi`) — builds sets upward only among mutually informative elements; sparser and faster, at the cost of completeness.

**Diagnostics** operate on an engine's antichain:

- **necessity** — the proportion of minimally sufficient sets containing each element.
- **dispensability** and the necessity × dispensability quadrants (keystone / specialist gateway / eliminable / inert).
- **coverage** — the share of cases satisfying at least one minimally sufficient set.

## Files

| File | Role |
| --- | --- |
| `enumerate_msc_v4.R` | Exhaustive engine; R entry point `enumerate_msc()`. Compiles its C++ core on first call. |
| `emsc_v4.cpp` | C++ core for the exhaustive engine (packed-bitset lattice ascent). |
| `pairmi_v5.R` | Constructive engine; R entry point `pairmi()`. Compiles its C++ core on first call. |
| `pairmi_v5.cpp` | C++ core for the constructive engine. |
| `antichain_diagnostics.R` | Necessity index (`compute_necessity_index`), necessity rollup (`summarize_necessity`), antichain inventory, co-membership; maps either engine's output to a common cell representation. |
| `dispensability_diagnostics.R` | Dispensability index and N-D quadrants (`compute_dispensability_index`, `classify_nd_quadrants`). |
| `compute_dispensability_v4.R` | Runner applying the dispensability diagnostics to saved analysis bundles. |
| `coverage_cross_cluster.R` | Coverage computation. |
| `run_coverage.R` | Top-level runner for `coverage_cross_cluster.R`. |
| `subfactor_pipeline_v4.R` | Orchestration: runs the exhaustive engine and the diagnostics end to end (`run_subfactor_analysis_v4()`). |
| `cluster_map_v5.R` | Item-to-cluster mapping used to define the vocabularies analyzed. |

## Requirements

- **R** (4.0 or later recommended).
- **A C++ toolchain** — the engines compile their C++ cores at runtime via `Rcpp::sourceCpp()`: Rtools on Windows, the Xcode Command Line Tools on macOS, `build-essential` on Linux.
- **R packages**: `Rcpp` (both engines); `foreign` and `bit` (the subfactor pipeline).

## Usage

Each engine is a single R function that compiles its C++ core on first call and returns the antichain of minimally sufficient sets for a binary (present/absent) matrix of cases × elements.

```r
# Exhaustive engine
source("enumerate_msc_v4.R")
res <- enumerate_msc(data, ...)          # data: cases × elements, coded 0/1

# Constructive engine
source("pairmi_v5.R")
res <- pairmi(data, alpha = 0.05, ...)
```

Diagnostics run on an engine's output. Both engines are reduced to a common "cell" representation first, so the same diagnostic functions apply to either:

```r
source("antichain_diagnostics.R")

cells     <- cells_from_exhaustive(res, benchmarks)   # or cells_from_mi(res) for pairmi
necessity <- compute_necessity_index(cells, element_names, element_labels)
rollup    <- summarize_necessity(necessity)
```

Dispensability and coverage follow the same pattern via `dispensability_diagnostics.R` / `compute_dispensability_v4.R` and `coverage_cross_cluster.R` / `run_coverage.R`. For an end-to-end run that wires the exhaustive engine to the diagnostics, see `run_subfactor_analysis_v4()` in `subfactor_pipeline_v4.R`.

## Relationship to setweaver

These are the current working engines of the framework. The original **setweaver** package was created by Nicolas Leenaerts and Aaron J. Fisher; these engines will be incorporated into **setweaver 2.0**, a forthcoming joint release. Until then, this repository is the canonical reference implementation.

## How to cite

If you use this software, please cite it via the **Cite this repository** button on the repository page (generated from `CITATION.cff`), or the archived release DOI:

> Fisher, A. J. (2026). *Configural Psychometrics: R and C++ engines for identifying minimally sufficient conditions* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21633129

## License

GPL-3.0. See [`LICENSE`](LICENSE).
