##############################################################################
# coverage_cross_cluster.R
#
# Coverage-weighted cross-cluster analysis -- the sturdier version of the
# "confident, severe classification requires cross-cluster contents" claim.
#
# For each (benchmark x theta) cell: of the participants COVERED by >= 1
# minimally sufficient set, what fraction are reachable ONLY via cross-cluster
# sets -- i.e. they satisfy NO single-cluster MSC? A cross-cluster-only case is
# one that a single-cluster analysis would orphan. Reported for the full
# covered footprint and, separately, for benchmark-positive cases (y == 1),
# the sensitivity-relevant reading.
#
# DATA-DEPENDENT (needs X and y), so it runs on the saved full_results.rds
# bundles, NOT on the inventory CSVs. Reuses .satisfaction_matrix / .cell_y
# from the dispensability stack; no re-enumeration.
#
# USAGE: set RESULTS_ROOT below, then  Rscript coverage_cross_cluster.R
# Output: coverage_cross_cluster.csv  (one row per run x benchmark x theta)
##############################################################################

source("antichain_diagnostics.R")        # %||%, .cell_meta_columns
source("dispensability_diagnostics.R")   # .satisfaction_matrix, .cell_y

# ----------------------------- CONFIG --------------------------------------
if (!exists("RESULTS_ROOT")) RESULTS_ROOT <- "results_v4"
if (!exists("OUT_CSV")) OUT_CSV <- "coverage_cross_cluster.csv"
# Whitelist internalizing runs by LEADING NUMBER. Run 08 (Distress-33) is
# DROPPED: after the v5 fold it is IDENTICAL to run 04 (depression + anxious
# arousal = the two Distress clusters, 33 items), so including both would
# double-weight Distress at the person level. Keep 04 (cluster-pair framing).
WHITELIST_NUMS <- c(2, 3, 4, 5, 6, 7, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
# ---------------------------------------------------------------------------

# Minimal bundle adapter (v4 subfactor bundles).
extract_run <- function(bundle) {
  if (is.null(bundle$X) || is.null(bundle$benchmarks))
    stop("bundle missing $X / $benchmarks")
  if (is.null(bundle$cells)) stop("bundle has no $cells")
  if (is.null(bundle$item_to_cluster))
    stop("bundle has no $item_to_cluster; cannot label clusters")
  list(cells = bundle$cells, X = bundle$X, benchmarks = bundle$benchmarks,
       symptom_names = bundle$items %||% colnames(bundle$X),
       item_to_cluster = bundle$item_to_cluster)
}

#' Coverage-weighted cross-cluster metrics for ONE cell. Returns a 1-row df.
cell_coverage <- function(cell, X, benchmarks, symptom_names, item_to_cluster) {
  meta <- .cell_meta_columns(cell, 1L)
  members <- cell$members
  n_msc <- length(members)

  if (n_msc == 0L) {
    return(cbind(meta, data.frame(
      n_msc = 0L, n_cross_msc = 0L, n_covered = 0L, n_cross_only = 0L,
      frac_cross_only = NA_real_, n_covered_y1 = 0L, n_cross_only_y1 = 0L,
      frac_cross_only_y1 = NA_real_, stringsAsFactors = FALSE)))
  }

  # distinct-cluster count per member -> cross-cluster flag
  member_nclus <- vapply(members, function(idx)
    length(unique(item_to_cluster[symptom_names[idx]])), integer(1))
  is_cross <- member_nclus >= 2L

  S <- .satisfaction_matrix(X, members)        # N x n_msc, 0/1
  covered <- rowSums(S) > 0L
  y <- .cell_y(cell, benchmarks, nrow(X))
  pos <- y == 1L

  single_cols <- which(!is_cross)
  sat_single <- if (length(single_cols))
    rowSums(S[, single_cols, drop = FALSE]) > 0L else rep(FALSE, nrow(X))

  cross_only <- covered & !sat_single          # covered, but no single-cluster route

  n_covered      <- sum(covered)
  n_covered_y1   <- sum(covered & pos)
  n_cross_only   <- sum(cross_only)
  n_cross_only_y1<- sum(cross_only & pos)

  cbind(meta, data.frame(
    n_msc = n_msc, n_cross_msc = sum(is_cross),
    n_covered = n_covered, n_cross_only = n_cross_only,
    frac_cross_only = if (n_covered > 0L) n_cross_only / n_covered else NA_real_,
    n_covered_y1 = n_covered_y1, n_cross_only_y1 = n_cross_only_y1,
    frac_cross_only_y1 = if (n_covered_y1 > 0L) n_cross_only_y1 / n_covered_y1 else NA_real_,
    stringsAsFactors = FALSE))
}

process_bundle <- function(bundle_path) {
  run <- sub("_v[0-9]+_thr.*$", "", basename(dirname(bundle_path)))
  r <- extract_run(readRDS(bundle_path))
  Xm <- r$X[, r$symptom_names, drop = FALSE]; storage.mode(Xm) <- "integer"
  rows <- lapply(r$cells, function(cell)
    cell_coverage(cell, Xm, r$benchmarks, r$symptom_names, r$item_to_cluster))
  out <- do.call(rbind, rows)
  cbind(run = run, out, stringsAsFactors = FALSE)
}

if (sys.nframe() == 0L || isTRUE(get0(".RUN_MAIN"))) {
  bundles <- list.files(RESULTS_ROOT, pattern = "^full_results\\.rds$",
                        recursive = TRUE, full.names = TRUE)
  if (length(bundles) == 0L)
    stop("No full_results.rds bundles under '", RESULTS_ROOT, "'.")
  message("Found ", length(bundles), " bundle(s).")
  # Keep only whitelisted internalizing runs (drops the run-04 duplicate, 08).
  .leadnum <- function(p) {
    rn <- sub("_v[0-9]+_thr.*$", "", basename(dirname(p)))
    as.integer(regmatches(rn, regexpr("^[0-9]+", rn)))
  }
  keep    <- vapply(bundles, function(p) isTRUE(.leadnum(p) %in% WHITELIST_NUMS), logical(1))
  dropped <- unique(vapply(bundles[!keep],
               function(p) sub("_v[0-9]+_thr.*$", "", basename(dirname(p))), character(1)))
  bundles <- bundles[keep]
  if (length(dropped)) message("Dropped ", length(dropped),
      " non-whitelisted bundle(s): ", paste(dropped, collapse = ", "))
  message("Processing ", length(bundles), " whitelisted bundle(s).")
  all <- list()
  for (b in bundles) {
    run <- sub("_v[0-9]+_thr.*$", "", basename(dirname(b)))
    res <- tryCatch(process_bundle(b),
                    error = function(e) { warning("FAILED ", run, ": ",
                                                  conditionMessage(e)); NULL })
    if (!is.null(res)) { all[[length(all) + 1L]] <- res; message("  ok: ", run) }
  }
  out <- do.call(rbind, all)
  write.csv(out, OUT_CSV, row.names = FALSE)
  message("\nWrote ", OUT_CSV, " (", nrow(out), " rows). Upload that one file.")
}
