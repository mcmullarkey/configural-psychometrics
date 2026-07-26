##############################################################################
# compute_dispensability_v4.R
#
# Post-hoc dispensability + N-D quadrants for the v4 internalizing reruns.
# Reads the full_results.rds bundle each run already saved (which stores
# $cells, $X, $benchmarks, $items, $item_to_cluster) and computes, per run:
#   dispensability_index.csv   (delta = 1 - orphaned/covered; full-footprint + y1)
#   nd_quadrants.csv           (keystone / specialist_gateway / eliminable / inert)
# using the same functions as the manuscript (dispensability_diagnostics.R).
#
# WHY A SEPARATE STEP (Option B, decoupled): dispensability rides on the saved
# bundles, so it does NOT re-run enumerate_msc. Re-tuning the quadrant knobs
# (near_necessary / near_dispensable) or the coverage definition means re-running
# ONLY this script, not the 16 enumerations.
#
# WHERE OUTPUTS GO: into each run's OWN result folder, UNSUFFIXED, exactly
# parallel to necessity_index.csv. collate_v4_outputs.R's generic per-run loop
# then rolls them up into collated/combined_dispensability_index.csv and
# combined_nd_quadrants.csv automatically -- no separate "Dispensability results"
# folder and no suffix matching. (If a stale "Dispensability results" folder is
# present, delete it so collate doesn't double-count; see the note printed at end.)
#
# USAGE (from the project directory, the same way you run the other scripts):
#   source("compute_dispensability_v4.R")
# Optional overrides before sourcing:
#   RESULTS_ROOT      <- "results_v4"   # folder holding the per-run subfolders
#   NEAR_NECESSARY    <- 0.95           # quadrant cutoffs (match classify_nd_quadrants)
#   NEAR_DISPENSABLE  <- 0.95
#   USE_Y1_QUADRANTS  <- FALSE          # classify on full-footprint delta (default)
#
# Requires: antichain_diagnostics.R and dispensability_diagnostics.R in the wd.
##############################################################################

source("antichain_diagnostics.R")        # compute_necessity_index, %||%
source("dispensability_diagnostics.R")   # compute_dispensability_index, classify_nd_quadrants

if (!exists("RESULTS_ROOT"))     RESULTS_ROOT     <- "results_v4"
if (!exists("NEAR_NECESSARY"))   NEAR_NECESSARY   <- 0.95
if (!exists("NEAR_DISPENSABLE")) NEAR_DISPENSABLE <- 0.95
if (!exists("USE_Y1_QUADRANTS")) USE_Y1_QUADRANTS <- FALSE

# --- Extract the pieces dispensability needs from a v4 bundle ----------------
# (v4 bundles always carry $cells; this is the v4 branch of the standalone's
#  extract_run, kept self-contained so this driver doesn't depend on the
#  v2-vintage run_dispensability_standalone.R.)
.extract_run_v4 <- function(bundle) {
  X          <- bundle$X
  benchmarks <- bundle$benchmarks
  cells      <- bundle$cells
  if (is.null(X) || is.null(benchmarks))
    stop("bundle missing $X and/or $benchmarks (dispensability is data-dependent).")
  if (is.null(cells))
    stop("bundle missing $cells; not a v4 subfactor bundle.")
  symptom_names  <- bundle$items %||% colnames(X)
  symptom_labels <- bundle$sym_labels %||% setNames(symptom_names, symptom_names)
  list(cells = cells, X = X, benchmarks = benchmarks,
       symptom_names = symptom_names, symptom_labels = symptom_labels,
       item_to_cluster = bundle$item_to_cluster)
}

# --- Compute + write dispensability/quadrants for ONE run folder -------------
.process_run <- function(bundle_path) {
  out_dir <- dirname(bundle_path)                 # the run's OWN folder
  b <- readRDS(bundle_path)
  r <- .extract_run_v4(b)
  message(sprintf("  cells: %d | items: %d | N: %d",
                  length(r$cells), length(r$symptom_names), nrow(r$X)))

  # Necessity recomputed from the SAME cells -> guarantees the quadrant join
  # shares keys exactly (matches the manuscript's necessity to rounding).
  necessity_df <- compute_necessity_index(
    cells = r$cells, symptom_names = r$symptom_names,
    symptom_labels = r$symptom_labels)

  dispensability_df <- compute_dispensability_index(
    cells = r$cells, X = r$X, benchmarks = r$benchmarks,
    symptom_names = r$symptom_names, symptom_labels = r$symptom_labels)

  quadrants_df <- classify_nd_quadrants(
    necessity_index = necessity_df, dispensability_index = dispensability_df,
    near_necessary = NEAR_NECESSARY, near_dispensable = NEAR_DISPENSABLE,
    use_y1 = USE_Y1_QUADRANTS)

  # Carry the cluster column, matching necessity_index.csv's layout.
  if (!is.null(r$item_to_cluster)) {
    dispensability_df$cluster <- unname(r$item_to_cluster[dispensability_df$symptom])
    if (nrow(quadrants_df) > 0L)
      quadrants_df$cluster <- unname(r$item_to_cluster[quadrants_df$symptom])
  }

  write.csv(dispensability_df, file.path(out_dir, "dispensability_index.csv"),
            row.names = FALSE)
  write.csv(quadrants_df, file.path(out_dir, "nd_quadrants.csv"),
            row.names = FALSE)
  message(sprintf("  wrote dispensability_index.csv (%d rows), nd_quadrants.csv (%d rows)",
                  nrow(dispensability_df), nrow(quadrants_df)))
  invisible(TRUE)
}

# --- Batch over all v4 bundles ----------------------------------------------
if (sys.nframe() == 0L || isTRUE(get0(".RUN_MAIN"))) {
  bundles <- list.files(RESULTS_ROOT, pattern = "^full_results\\.rds$",
                        recursive = TRUE, full.names = TRUE)
  if (length(bundles) == 0L)
    stop("No full_results.rds bundles under '",
         normalizePath(RESULTS_ROOT, mustWork = FALSE),
         "'. Run the enumeration first, or set RESULTS_ROOT.")

  message("=== compute_dispensability_v4 ===")
  message("Found ", length(bundles), " bundle(s) under '", RESULTS_ROOT, "'.")
  ok <- character(0); failed <- character(0)
  for (bp in bundles) {
    run_folder <- basename(dirname(bp))
    message("\n[", run_folder, "]")
    res <- tryCatch({ .process_run(bp); TRUE },
                    error = function(e) { message("  FAILED: ", conditionMessage(e)); FALSE })
    if (isTRUE(res)) ok <- c(ok, run_folder) else failed <- c(failed, run_folder)
  }

  message("\n=== coverage ===")
  message("dispensability written for ", length(ok), "/", length(bundles), " runs.")
  if (length(failed))
    message("FAILED (", length(failed), "): ", paste(failed, collapse = ", "))
  if (dir.exists("Dispensability results"))
    message("NOTE: a 'Dispensability results' folder exists. Delete it before ",
            "collating so the old suffix-matching path doesn't double-count ",
            "these run-folder files.")
  message("\nNext: Rscript collate_v4_outputs.R  (will pick these up as ",
          "combined_dispensability_index.csv / combined_nd_quadrants.csv)")
}
