#!/usr/bin/env Rscript
# Generate golden parity fixtures for configural stats module.
#
# Produces JSON files in configural/tests/golden/:
#   wilson_parity.json  — wilson_lower(p, n, z)
#   mi_parity.json      — mi_2x2(n11, nx1, nx2, n)
#   gtest_parity.json   — g_test(n11, n, mi) -> {g, p}
#   chi2_parity.json    — chi2_sf_df1(g)
#   qnorm_parity.json   — normal_inverse_cdf(p)
#
# Each file is a JSON array of {"args": [...], "expected": ...} objects.
# R is the reference implementation (same formulas as C++ emsc_v4.cpp +
# pairmi_v5.cpp). Rust must match to ~1e-12 relative tolerance.

library(jsonlite)

golden_dir <- file.path("configural", "tests", "golden")
dir.create(golden_dir, showWarnings = FALSE, recursive = TRUE)

# ---- wilson_lower ----
# Exact C++ term order from emsc_v4.cpp:57-62:
#   denom  = 1 + z^2/n
#   center = p + z^2/(2n)
#   margin = z * sqrt(p(1-p)/n + z^2/(4n^2))
#   result = (center - margin) / denom
wilson_lower <- function(p, n, z) {
  denom  <- 1.0 + z * z / n
  center <- p + z * z / (2.0 * n)
  margin <- z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
  (center - margin) / denom
}

wilson_cases <- expand.grid(
  p = c(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
  n = c(10, 50, 100, 500, 1000),
  z = c(1.645, 1.96, 2.576),
  stringsAsFactors = FALSE
)
wilson_data <- lapply(seq_len(nrow(wilson_cases)), function(i) {
  list(args = list(wilson_cases$p[i], wilson_cases$n[i], wilson_cases$z[i]),
       expected = wilson_lower(wilson_cases$p[i], wilson_cases$n[i], wilson_cases$z[i]))
})
writeLines(toJSON(wilson_data, auto_unbox = TRUE, digits = 17, pretty = TRUE),
          file.path(golden_dir, "wilson_parity.json"))

# ---- safe_term + mi_2x2 ----
# Exact C++ from pairmi_v5.cpp:70-85:
#   safe_term(n_cell, m1, m2, n) = (n_cell/n)*ln(n_cell*n/(m1*m2)) iff all > 0
#   mi_2x2(n11, nx1, nx2, n) = t11 + t10 + t01 + t00
safe_term <- function(n_cell, m1, m2, n) {
  if (n_cell > 0 && m1 > 0 && m2 > 0 && n > 0)
    (n_cell / n) * log((n_cell * n) / (m1 * m2))
  else
    0.0
}
mi_2x2 <- function(n11, nx1, nx2, n) {
  n10 <- nx1 - n11
  n01 <- nx2 - n11
  n00 <- n - nx1 - nx2 + n11
  safe_term(n11, nx1,     nx2,     n) +
  safe_term(n10, nx1,     n - nx2, n) +
  safe_term(n01, n - nx1, nx2,     n) +
  safe_term(n00, n - nx1, n - nx2, n)
}

mi_cases <- list(
  list(10, 50, 60, 100),
  list(5, 20, 30, 100),
  list(0, 50, 50, 100),
  list(50, 50, 50, 100),
  list(25, 50, 50, 100),
  list(1, 10, 10, 100),
  list(99, 100, 100, 100),
  list(0, 0, 0, 0),
  list(1e-300, 2e-10, 3e-10, 100),
  list(30, 60, 50, 100),
  list(15, 40, 35, 80),
  list(3, 10, 15, 50),
  list(7, 20, 18, 50),
  list(0, 10, 0, 50),
  list(10, 0, 10, 50),
  list(100, 100, 100, 100),
  list(1, 1, 1, 1),
  list(50, 100, 100, 200),
  list(1, 2, 2, 4),
  list(0, 0, 10, 100)
)
mi_data <- lapply(mi_cases, function(args) {
  list(args = as.list(args),
       expected = do.call(mi_2x2, args))
})
writeLines(toJSON(mi_data, auto_unbox = TRUE, digits = 17, pretty = TRUE),
          file.path(golden_dir, "mi_parity.json"))

# ---- chi2_sf_df1 ----
# chi2_sf_df1(g) = erfc(sqrt(g/2)) = pchisq(g, 1, lower.tail=FALSE)
chi2_cases <- list(
  list(0.0),
  list(0.001),
  list(0.01),
  list(0.1),
  list(0.5),
  list(1.0),
  list(2.0),
  list(3.8414588),
  list(5.0),
  list(6.6348966),
  list(10.0),
  list(15.0),
  list(20.0),
  list(0.0001),
  list(50.0),
  list(100.0),
  list(0.5),
  list(1.5),
  list(2.5),
  list(7.0)
)
chi2_data <- lapply(chi2_cases, function(args) {
  g <- args[[1]]
  list(args = as.list(args),
       expected = pchisq(g, 1, lower.tail = FALSE))
})
writeLines(toJSON(chi2_data, auto_unbox = TRUE, digits = 17, pretty = TRUE),
          file.path(golden_dir, "chi2_parity.json"))

# ---- g_test ----
# g_test(n11, n, mi) -> {g, p}
#   jc = min(n11, n - n11)   [flipped-count correction]
#   g  = 2 * jc * mi
#   p  = chi2_sf_df1(g) = pchisq(g, 1, lower.tail=FALSE)
gtest_cases <- list(
  list(10, 100, 0.05),    # n11 < n/2 -> jc = n11
  list(60, 100, 0.05),    # n11 > n/2 -> jc = n - n11
  list(50, 100, 0.1),     # n11 = n/2  -> jc = n11
  list(5, 100, 0.01),
  list(95, 100, 0.02),
  list(25, 50, 0.08),
  list(40, 50, 0.15),
  list(1, 100, 0.001),
  list(99, 100, 0.005),
  list(0, 100, 0.0),      # mi=0 -> g=0 -> p=1
  list(30, 200, 0.03),
  list(170, 200, 0.04),
  list(10, 50, 0.06),
  list(45, 50, 0.07),
  list(3, 30, 0.02),
  list(27, 30, 0.025),
  list(15, 30, 0.035),
  list(50, 100, 0.0),
  list(75, 100, 0.12),
  list(25, 100, 0.11)
)
gtest_data <- lapply(gtest_cases, function(args) {
  n11 <- args[[1]]; n <- args[[2]]; mi <- args[[3]]
  jc <- min(n11, n - n11)
  g <- 2.0 * jc * mi
  p <- pchisq(g, 1, lower.tail = FALSE)
  list(args = as.list(args), expected = list(g = g, p = p))
})
writeLines(toJSON(gtest_data, auto_unbox = TRUE, digits = 17, pretty = TRUE),
          file.path(golden_dir, "gtest_parity.json"))

# ---- normal_inverse_cdf (qnorm) ----
# R::qnorm(p, 0, 1) — standard normal quantile function (AS241)
qnorm_cases <- list(
  list(0.001),
  list(0.005),
  list(0.01),
  list(0.025),
  list(0.05),
  list(0.1),
  list(0.15),
  list(0.2),
  list(0.25),
  list(0.3),
  list(0.4),
  list(0.5),
  list(0.6),
  list(0.7),
  list(0.75),
  list(0.8),
  list(0.85),
  list(0.9),
  list(0.95),
  list(0.975),
  list(0.99),
  list(0.995),
  list(0.999),
  list(0.0001),
  list(0.9999)
)
qnorm_data <- lapply(qnorm_cases, function(args) {
  p <- args[[1]]
  list(args = as.list(args), expected = qnorm(p, 0, 1))
})
writeLines(toJSON(qnorm_data, auto_unbox = TRUE, digits = 17, pretty = TRUE),
          file.path(golden_dir, "qnorm_parity.json"))

cat("Generated golden fixtures in", golden_dir, "\n")
cat("  wilson:", length(wilson_data), "points\n")
cat("  mi:",     length(mi_data),     "points\n")
cat("  chi2:",   length(chi2_data),   "points\n")
cat("  gtest:",  length(gtest_data),  "points\n")
cat("  qnorm:",  length(qnorm_data),  "points\n")
