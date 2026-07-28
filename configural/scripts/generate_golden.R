#!/usr/bin/env Rscript
# Generate golden parity fixtures for configural stats module.
#
# Produces JSON files in configural/tests/fixtures/golden/:
#   small_4x3.json   — small dataset (4 rows x 3 cols), seed 42
#   medium_50x13.json — medium dataset (50 rows x 13 cols), seed 123
#   large_200x25.json — large dataset (200 rows x 25 cols), seed 456
#
# Each file contains:
#   format_version, seed, dataset name
#   wilson_lower[]  — {p, n, z, value}
#   mi_2x2[]        — {n11, nx1, nx2, n, value}
#   g_test[]        — {n11, n, mi, g, p}
#   chi2_sf_df1[]   — {g, value}
#   qnorm[]         — {p, value}
#
# R is the reference implementation (same formulas as C++ emsc_v4.cpp +
# pairmi_v5.cpp). Rust must match to ~1e-12 relative tolerance (pure math)
# or max(1e-12 * |ref|, 1e-12) absolute tolerance (platform-dependent).
#
# Usage:
#   Rscript configural/scripts/generate_golden.R
#
# Requires: jsonlite package (install.packages("jsonlite"))

library(jsonlite)

# ---- Configuration ----

fixture_dir <- file.path("configural", "tests", "fixtures", "golden")
dir.create(fixture_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Source R reference files ----
# These provide the canonical R implementations that the Rust crate ports.

repo_root <- getwd()
source(file.path(repo_root, "enumerate_msc_v4.R"))
source(file.path(repo_root, "pairmi_v5.R"))
# cluster_map_v5.R, compute_dispensability_v4.R, etc. may also be sourced
# if engine/diagnostics parity fixtures are needed in future.

# ---- Reference function implementations ----
# These match the C++ cores (emsc_v4.cpp, pairmi_v5.cpp) exactly.

# Wilson score lower bound (emsc_v4.cpp:57-62)
wilson_lower <- function(p, n, z) {
  denom  <- 1.0 + z * z / n
  center <- p + z * z / (2.0 * n)
  margin <- z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
  (center - margin) / denom
}

# Guarded single-cell MI contribution (pairmi_v5.cpp:70-74)
safe_term <- function(n_cell, m1, m2, n) {
  if (n_cell > 0 && m1 > 0 && m2 > 0 && n > 0)
    (n_cell / n) * log((n_cell * n) / (m1 * m2))
  else
    0.0
}

# Four-term mutual information in nats (pairmi_v5.cpp:76-85)
mi_2x2 <- function(n11, nx1, nx2, n) {
  n10 <- nx1 - n11
  n01 <- nx2 - n11
  n00 <- n - nx1 - nx2 + n11
  safe_term(n11, nx1,     nx2,     n) +
  safe_term(n10, nx1,     n - nx2, n) +
  safe_term(n01, n - nx1, nx2,     n) +
  safe_term(n00, n - nx1, n - nx2, n)
}

# Chi-squared survival function, 1 df (erfc-based)
chi2_sf_df1 <- function(g) {
  if (g <= 0) return(1.0)
  2 * pnorm(-sqrt(g / 2))
}

# G-test with flipped-count correction (pairmi_v5.cpp:146-150)
g_test <- function(n11, n, mi) {
  jc <- min(n11, n - n11)
  g  <- 2.0 * jc * mi
  p  <- chi2_sf_df1(g)
  list(g = g, p = p)
}

# Standard normal quantile (R's qnorm — reference for AS241 port)
qnorm_ref <- function(p) {
  qnorm(p)
}

# ---- Dataset generation ----

generate_dataset <- function(n_rows, n_cols, seed) {
  set.seed(seed)
  matrix(rbinom(n_rows * n_cols, 1, 0.5), nrow = n_rows, ncol = n_cols)
}

datasets <- list(
  list(name = "small_4x3",   rows = 4,   cols = 3,  seed = 42),
  list(name = "medium_50x13", rows = 50,  cols = 13, seed = 123),
  list(name = "large_200x25", rows = 200, cols = 25, seed = 456)
)

# ---- Build fixtures per dataset ----

for (ds in datasets) {
  mat <- generate_dataset(ds$rows, ds$cols, ds$seed)
  n_total <- ds$rows

  # wilson_lower: sweep p across observed column rates + fixed z values
  col_rates <- colSums(mat) / n_total
  z_values <- c(1.645, 1.96, 2.576)
  wilson_cases <- list()
  for (p in col_rates) {
    for (z in z_values) {
      wilson_cases <- c(wilson_cases, list(list(
        p = p, n = n_total, z = z,
        value = wilson_lower(p, n_total, z)
      )))
    }
  }

  # mi_2x2: all pairs of columns
  mi_cases <- list()
  if (ds$cols >= 2) {
    pairs <- combn(ds$cols, 2, simplify = FALSE)
    for (pair in pairs) {
      j1 <- pair[1]
      j2 <- pair[2]
      n11 <- sum(mat[, j1] & mat[, j2])
      nx1 <- sum(mat[, j1])
      nx2 <- sum(mat[, j2])
      mi_cases <- c(mi_cases, list(list(
        n11 = n11, nx1 = nx1, nx2 = nx2, n = n_total,
        value = mi_2x2(n11, nx1, nx2, n_total)
      )))
    }
  }

  # g_test: from mi_2x2 results
  gtest_cases <- list()
  for (mc in mi_cases) {
    gt <- g_test(mc$n11, mc$n, mc$value)
    gtest_cases <- c(gtest_cases, list(list(
      n11 = mc$n11, n = mc$n, mi = mc$value,
      g = gt$g, p = gt$p
    )))
  }

  # chi2_sf_df1: sweep g values
  g_values <- c(0.0, 0.5, 1.0, 2.0, 3.8414588, 5.0, 10.0, 20.0)
  chi2_cases <- list()
  for (g in g_values) {
    chi2_cases <- c(chi2_cases, list(list(
      g = g, value = chi2_sf_df1(g)
    )))
  }

  # qnorm: sweep p values
  p_values <- c(0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999)
  qnorm_cases <- list()
  for (p in p_values) {
    qnorm_cases <- c(qnorm_cases, list(list(
      p = p, value = qnorm_ref(p)
    )))
  }

  fixture <- list(
    format_version = 1L,
    seed = ds$seed,
    dataset = ds$name,
    wilson_lower = wilson_cases,
    mi_2x2 = mi_cases,
    g_test = gtest_cases,
    chi2_sf_df1 = chi2_cases,
    qnorm = qnorm_cases
  )

  out_path <- file.path(fixture_dir, paste0(ds$name, ".json"))
  writeLines(
    toJSON(fixture, auto_unbox = TRUE, digits = 17, pretty = TRUE),
    out_path
  )
  cat("Wrote", out_path, "\n")
}

cat("\nDone. Fixtures written to", fixture_dir, "\n")
