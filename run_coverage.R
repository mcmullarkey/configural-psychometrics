##############################################################################
# run_coverage.R  --  runs coverage_cross_cluster.R at TOP LEVEL.
#
# WHY THIS EXISTS: coverage_cross_cluster.R guards its driver with
#   if (sys.nframe() == 0L) { ... }
# so a plain source("coverage_cross_cluster.R") in RStudio only DEFINES its
# functions and runs nothing (under source(), sys.nframe() is > 0). This wrapper
# spawns it in a fresh Rscript process, where sys.nframe() == 0, so the driver
# actually executes. Run it exactly like the other phases:
#
#     source("run_coverage.R")
#
# PREREQ: Phase 1 enumeration is done (full_results.rds bundles exist under
#         results_v4/). This step is data-dependent (needs X and y) but reuses
#         the saved bundles -- no re-enumeration.
# OUTPUT: coverage_cross_cluster.csv  (one row per run x benchmark x theta).
##############################################################################

rscript <- file.path(R.home("bin"),
                     if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
if (!file.exists(rscript)) stop("Could not find Rscript at: ", rscript)
if (!file.exists("coverage_cross_cluster.R"))
  stop("coverage_cross_cluster.R not found in the working directory.")

cat("\n=================== coverage_cross_cluster.R ===================\n")
status <- system2(rscript, shQuote("coverage_cross_cluster.R"))   # streams live
if (!identical(status, 0L))
  stop("FAILED (exit ", status, "): coverage_cross_cluster.R -- inspect above.")

cat("\n>> COVERAGE DONE.\n")
cat(">> Wrote coverage_cross_cluster.csv -- upload that one file.\n")
cat(">> Headline columns: frac_cross_only (of everyone covered, the share\n")
cat(">> reachable ONLY via cross-cluster sets) and frac_cross_only_y1 (same,\n")
cat(">> restricted to benchmark-positive cases -- the sensitivity reading).\n")
