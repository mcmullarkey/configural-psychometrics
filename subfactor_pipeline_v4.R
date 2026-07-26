##############################################################################
# subfactor_pipeline_v4.R
#
# Shared analytic pipeline for symptom-cluster / subfactor-level
# set-theoretic analyses on the Forbes MQRF data. v3 differs from v2/v3 by:
#   (a) the v4 C++ enumerator (enumerate_msc_v4.R + emsc_v4.cpp) as the
#       canonical engine, run in sufficient_detail = "count" mode (the
#       antichain is retained; the full non-minimal S_theta is not
#       materialized -- essential for K=9 on larger vocabularies);
#   (b) the NEW canonical benchmark set (5): K10 elevated/severe, WHODAS-12/8
#       (75th %ile, stringent -- deliberately hard to clear), and CURRENT
#       mental-health treatment (Mental_tx_curr). The lifetime-treatment and
#       MQRF_324 "overwhelmed" benchmarks are DROPPED, so the MQRF_324
#       tautology / target_exclude apparatus is gone entirely;
#   (c) max_cardinality = 9 by default.
# There is no backwards compatibility with the v2 (K=8) outputs by design;
# the whole series is being re-run under these choices.
#
# Each run script defines a `clusters` list and calls
# run_subfactor_analysis_v4(). The pipeline:
#   1. Loads SPSS data, builds binary item indicators (>= dichot_threshold).
#   2. Restricts to complete-case rows on the universal set + benchmark inputs.
#   3. Builds the five benchmarks above.
#   4. Runs enumerate_msc() (v4, count mode) at the configured n_min, alpha,
#      criterion, and max_cardinality. Wilson-only by default.
#   5. Converts enumerate_msc output into the unified-cell format required
#      by antichain_diagnostics.R.
#   6. Runs the four diagnostic functions (inventory, necessity index,
#      necessity summary, co-membership) and writes outputs. The inventory
#      carries a cluster_signature column (used for cross-cluster analysis).
#   7. Saves a full_results.rds bundle.
#
# Table-1 n_sufficient is read from emsc_result$n_sufficient (the per-cell
# count v4 returns in count mode), NOT nrow($sufficient).
#
##############################################################################

source("enumerate_msc_v4.R")        # v4 C++ core; identical() to saveall on shared fields
.emsc4_env$cpp_path <- "emsc_v4.cpp"
source("antichain_diagnostics.R")

suppressPackageStartupMessages({
  library(foreign)
  library(bit)
})

stopifnot("sufficient_detail" %in% names(formals(enumerate_msc)))  # confirm v4 in scope


# =============================================================================
# Default configuration. Override individual fields by passing them to
# run_subfactor_analysis_v4(...).
# =============================================================================

DEFAULT_CONFIG_V4 <- list(
  spss_path = "Reconstructing Psychopathology_20240202_200minimum.sav",

  dichot_threshold = 4L,

  k10_elevated = 25L,
  k10_severe   = 30L,
  whodas_body_items = c("WHODAS_01", "WHODAS_04", "WHODAS_07",
                        "WHODAS_11", "WHODAS_12", "WHODAS_13",
                        "WHODAS_16", "WHODAS_17", "WHODAS_21",
                        "WHODAS_25", "WHODAS_29", "WHODAS_33"),
  whodas_psych_items = c("WHODAS_01", "WHODAS_04",
                         "WHODAS_16", "WHODAS_17",
                         "WHODAS_21", "WHODAS_33",
                         "WHODAS_25", "WHODAS_29"),

  # Fixed ABSOLUTE WHODAS cutoffs (harmonized). Source: step1_whodas_reference.R,
  # whole-sample complete-case reference (N = 14,761 with all 12 body items).
  # These replace the former per-run quantile(total, .75), which drifted across
  # runs (DECISIONS_LOG TODO #1). Values are raw summed-item scores.
  #   whodas12_cut    : WHODAS-12 whole-sample 75th  -> >= 35  (high impairment)
  #   whodas8_cut     : WHODAS-8  whole-sample 75th  -> >= 26  (high impairment)
  #   whodas8_mod_cut : WHODAS-8  whole-sample 60th  -> >= 24  (moderate; permissive.
  #                     60th chosen under strict '>' so the median tie-mass is not
  #                     swept in; >24 on integer scores == >=24 -> the cutoff below.)
  whodas12_cut    = 35L,
  whodas8_cut     = 26L,
  whodas8_mod_cut = 24L,

  thresholds = c(1.00, 0.99, 0.95, 0.90, 0.85, 0.80),
  alpha      = 0.05,
  criterion  = "wilson",      # Wilson-only by default for production work

  # Admissibility floor: sample-relative.
  #   n_min = max(n_min_floor, ceil(n_min_prop * N))
  # The same rule used in 09_v2; auditable, scale-aware, not arbitrary.
  n_min_floor = 10L,
  n_min_prop  = 0.005,

  # Cardinality cap for enumerate_msc. Soft cap: the admissibility floor
  # may halt the ascent before this depth is reached. For analyses we want
  # to run to exhaustion, set this >= V (the vocabulary size).
  max_cardinality = 9L,

  # v4 detail mode. "count" retains the antichain (minimally-sufficient sets)
  # and reports per-cell n_sufficient WITHOUT materializing the full
  # non-minimal S_theta -- required to keep K=9 runs on 30-38 item sets
  # within memory. Table-1 n_sufficient is read from emsc_result$n_sufficient.
  sufficient_detail = "count",

  # Safety valve for runaway lattices. enumerate_msc aborts cleanly if a
  # single depth would produce more than this many admissible configurations.
  # Raised to 5e8: high enough to never truncate a feasible K=9 run up to
  # ~38 items (worst case ~C(38,9) = 1.6e8), low enough to abort a genuine
  # runaway. The real feasibility guard is RAM (see the memory frontier).
  max_admissible_per_depth = 5e8,

  near_necessary_thresholds     = c(0.95, 0.90, 0.80),
  comembership_min_proportion   = 0.50,
  comembership_include_triplets = TRUE
)


# =============================================================================
# Helpers
# =============================================================================

#' Convert enumerate_msc output to mi_records-equivalent list of cells.
#'
#' @param emsc_result return value of enumerate_msc()
#' @param benchmarks named list of benchmark spec used in the run
#' @param criteria character vector of criteria actually computed
#' @param thresholds numeric vector of theta values
#' @return list of records compatible with cells_from_mi()
.emsc_to_records <- function(emsc_result, benchmarks, criteria, thresholds) {
  records <- list()
  for (bn in names(benchmarks)) {
    b <- benchmarks[[bn]]
    for (crit in criteria) {
      for (thr in thresholds) {
        key <- sprintf("%s_p%03d", crit, round(thr * 100))
        theta_key <- paste0(bn, "__theta", format(thr))
        df <- emsc_result$minimally_sufficient[[crit]][[theta_key]]
        if (is.null(df) || nrow(df) == 0L) {
          ac_member_items <- list()
          cprob_vec <- numeric(0)
          wilson_vec <- numeric(0)
          n_set_vec <- integer(0)
          n_set_y1_vec <- integer(0)
        } else {
          # set_idx is a list-column of integer vectors (vocabulary indices).
          # Strip names defensively (the column comes from an I()-wrapped
          # list and can carry names that propagate to downstream code).
          ac_member_items <- lapply(df$set_idx, function(idx) {
            unname(as.integer(idx))
          })
          cprob_vec    <- df$cprob
          wilson_vec   <- df$wilson_lower
          n_set_vec    <- df$n_set
          n_set_y1_vec <- df$n_set_y1
        }
        records[[length(records) + 1L]] <- list(
          alpha           = NA_real_,   # not pairmi alpha; left for schema
          benchmark       = bn,
          bench_label     = b$label,
          criterion       = crit,
          threshold       = thr,
          key             = key,
          ac_member_items = ac_member_items,
          cprob           = cprob_vec,
          wilson_lower    = wilson_vec,
          n_set           = n_set_vec,
          n_set_y1        = n_set_y1_vec
        )
      }
    }
  }
  records
}


# =============================================================================
# Main entry point
# =============================================================================

#' Run the full subfactor / cluster analysis pipeline (v2: enumerate_msc).
#'
#' @param universal_set_name character. Short name used to label this run
#'   (e.g. "distress", "fear", "anxious_arousal"). Used in the output_dir.
#' @param clusters named list. Each element is a character vector of MQRF
#'   item names belonging to that cluster. Cluster names are short labels
#'   like "depmood", "agora_soc", etc. The names of the list become the
#'   cluster labels used in diagnostics. For analyses with no internal
#'   cluster structure (e.g., a flat 13-item set), pass a single-element
#'   list with the items.
#' @param output_dir_base character. Will be appended with the universal
#'   set name and key config knobs.
#' @param config list. Overrides for DEFAULT_CONFIG_V4 (any subset).
#' @return invisibly: the saved results bundle (also written to disk).
run_subfactor_analysis_v4 <- function(universal_set_name,
                                      clusters,
                                      output_dir_base = "results",
                                      config = list()) {

  CONFIG <- modifyList(DEFAULT_CONFIG_V4, config)
  CONFIG$universal_set_name <- universal_set_name

  # Validate criterion
  CONFIG$criterion <- as.character(CONFIG$criterion)
  bad_crit <- setdiff(CONFIG$criterion, c("wilson", "point"))
  if (length(bad_crit) > 0L) {
    stop("Invalid criterion: ", paste(bad_crit, collapse = ", "),
         ". Must be 'wilson' and/or 'point'.")
  }

  CONFIG$output_dir <- file.path(
    output_dir_base,
    sprintf("%s_v4_thr%d_card%d_%s",
            universal_set_name, CONFIG$dichot_threshold,
            CONFIG$max_cardinality,
            paste(CONFIG$criterion, collapse = "+"))
  )
  if (!dir.exists(CONFIG$output_dir)) {
    dir.create(CONFIG$output_dir, recursive = TRUE)
  }
  message("\n=========================================================")
  message("Subfactor analysis (v4 enumerate_msc v4-count): ", universal_set_name)
  message("Output directory: ", CONFIG$output_dir)
  message("=========================================================\n")

  # --- Universal set ---------------------------------------------------------
  items <- unique(unlist(clusters, use.names = FALSE))
  item_to_cluster <- setNames(
    rep(names(clusters), vapply(clusters, length, integer(1))),
    unlist(clusters, use.names = FALSE)
  )
  message("Universal set: ", length(items), " items across ",
          length(clusters), " clusters")
  for (cn in names(clusters)) {
    message(sprintf("  %-30s  %d items", cn, length(clusters[[cn]])))
  }

  # --- Load data -------------------------------------------------------------
  message("\n=== Loading SPSS data ===")
  raw <- read.spss(CONFIG$spss_path,
                   to.data.frame    = TRUE,
                   use.value.labels = FALSE,
                   stringsAsFactors = FALSE)
  message("Loaded: ", nrow(raw), " rows")

  # Strict column-presence checks
  missing_items <- setdiff(items, colnames(raw))
  if (length(missing_items) > 0L) {
    stop("Items not found in data: ", paste(missing_items, collapse = ", "))
  }
  k10_cols <- sprintf("K10_%02d", 1:10)
  required_for_bench <- c(k10_cols, CONFIG$whodas_body_items,
                          "Mental_tx_curr", "Mental_tx_past")
  missing_bench <- setdiff(required_for_bench, colnames(raw))
  if (length(missing_bench) > 0L) {
    stop("Benchmark columns not found in data: ",
         paste(missing_bench, collapse = ", "))
  }

  # Binarize at >= dichot_threshold
  X_raw <- as.matrix(raw[, items, drop = FALSE])
  X_bin <- (X_raw >= CONFIG$dichot_threshold)
  storage.mode(X_bin) <- "integer"
  seen_mat <- !is.na(X_raw)

  # Complete-case mask on the universal set
  complete_items <- rowSums(seen_mat) == length(items)

  # K10 missingness mask: require >= 8 of 10
  k10_n_valid <- rowSums(!is.na(raw[, k10_cols, drop = FALSE]))
  complete_k10 <- k10_n_valid >= 8L

  # Other benchmark columns must be non-missing
  complete_other <- rep(TRUE, nrow(raw))
  for (cc in c(CONFIG$whodas_body_items, "Mental_tx_curr", "Mental_tx_past")) {
    complete_other <- complete_other & !is.na(raw[[cc]])
  }

  complete_case <- complete_items & complete_k10 & complete_other
  n_complete <- sum(complete_case)
  message("Complete-case N: ", format(n_complete, big.mark = ","),
          " of ", format(nrow(raw), big.mark = ","),
          " (", round(100 * n_complete / nrow(raw), 1), "%)")

  keep_idx <- which(complete_case)
  X <- X_bin[keep_idx, , drop = FALSE]
  X[is.na(X)] <- 0L
  raw_cc <- raw[keep_idx, , drop = FALSE]

  # Sample-relative n_min
  N <- nrow(X)
  n_min <- max(CONFIG$n_min_floor, ceiling(CONFIG$n_min_prop * N))
  message("Admissibility: n_min = max(", CONFIG$n_min_floor,
          ", ceil(", CONFIG$n_min_prop, " * ", N, ")) = ", n_min)

  # Item prevalences
  item_prev <- data.frame(
    item       = items,
    cluster    = item_to_cluster[items],
    prevalence = round(colMeans(X), 3),
    n_endorsed = colSums(X),
    stringsAsFactors = FALSE
  )
  write.csv(item_prev,
            file.path(CONFIG$output_dir, "item_prevalences.csv"),
            row.names = FALSE)

  # --- Benchmarks ------------------------------------------------------------
  # K10 scoring detection + offset
  k10_range <- range(unlist(raw_cc[, k10_cols]), na.rm = TRUE)
  k10_offset <- if (k10_range[1] == 0) 10L else 0L
  message(sprintf("K10 items: range [%d, %d], threshold offset = %d",
                  k10_range[1], k10_range[2], k10_offset))

  k10_total <- rowSums(raw_cc[, k10_cols, drop = FALSE], na.rm = TRUE)
  dist_elev   <- as.integer(k10_total >= (CONFIG$k10_elevated - k10_offset))
  dist_severe <- as.integer(k10_total >= (CONFIG$k10_severe   - k10_offset))

  whodas_body <- raw_cc[, CONFIG$whodas_body_items, drop = FALSE]
  whodas12_n <- rowSums(!is.na(whodas_body))
  whodas12_total <- rowSums(whodas_body, na.rm = TRUE)
  whodas12_total[whodas12_n < 10] <- NA_real_
  # Fixed ABSOLUTE cutoff (whole-sample 75th). Complete-case already guarantees
  # all 12 body items present, so whodas12_total is non-missing in practice and
  # the is.na() guard never fires here -- retained defensively.
  impair_whodas12 <- ifelse(is.na(whodas12_total), 0L,
                            as.integer(whodas12_total >= CONFIG$whodas12_cut))

  whodas_psych <- raw_cc[, CONFIG$whodas_psych_items, drop = FALSE]
  whodas8_n <- rowSums(!is.na(whodas_psych))
  whodas8_total <- rowSums(whodas_psych, na.rm = TRUE)
  whodas8_total[whodas8_n < 6] <- NA_real_
  # Two fixed absolute cutoffs on the SAME WHODAS-8 psychosocial score:
  #   high     >= whodas8_cut     (26, whole-sample 75th)
  #   moderate >= whodas8_mod_cut (24, whole-sample 60th; permissive anchor)
  impair_whodas8 <- ifelse(is.na(whodas8_total), 0L,
                           as.integer(whodas8_total >= CONFIG$whodas8_cut))
  impair_whodas8_mod <- ifelse(is.na(whodas8_total), 0L,
                               as.integer(whodas8_total >= CONFIG$whodas8_mod_cut))

  tx_curr_raw <- raw_cc[["Mental_tx_curr"]]
  tx_curr <- ifelse(is.na(tx_curr_raw), 0L, as.integer(tx_curr_raw == 2))
  tx_past_raw <- raw_cc[["Mental_tx_past"]]
  tx_past <- ifelse(is.na(tx_past_raw), 0L, as.integer(tx_past_raw == 2))

  # 7-benchmark harmonized panel. The original five names are UNCHANGED and in
  # their original order (so anything keyed on them is unaffected); this is an
  # additive change appending whodas8_mod (permissive impairment) and tx_past
  # (reintroduced lifetime treatment). "overwhelmed" (MQRF_324) remains out.
  benchmarks <- list(
    k10_elev    = list(label = "K10 elevated distress (>=25)",
                       y = dist_elev),
    k10_severe  = list(label = "K10 severe distress (>=30)",
                       y = dist_severe),
    whodas12    = list(label = sprintf("WHODAS-12 high impairment (>=%d, fixed 75th ref)",
                                       CONFIG$whodas12_cut),
                       y = impair_whodas12),
    whodas8     = list(label = sprintf("WHODAS-8 psychosocial high (>=%d, fixed 75th ref)",
                                       CONFIG$whodas8_cut),
                       y = impair_whodas8),
    whodas8_mod = list(label = sprintf("WHODAS-8 psychosocial moderate (>=%d, fixed 60th ref)",
                                       CONFIG$whodas8_mod_cut),
                       y = impair_whodas8_mod),
    tx_curr     = list(label = "Current mental health treatment",
                       y = tx_curr),
    tx_past     = list(label = "Past mental health treatment",
                       y = tx_past)
  )

  message("\nBenchmark base rates (n=", N, "):")
  for (bn in names(benchmarks)) {
    br <- mean(benchmarks[[bn]]$y)
    message(sprintf("  %-50s %.3f", benchmarks[[bn]]$label, br))
  }

  # --- enumerate_msc --------------------------------------------------------
  message("\n=== Running enumerate_msc ===")
  message(sprintf("  V = %d, N = %d, n_min = %d, alpha = %g, max_card = %d",
                  length(items), N, n_min, CONFIG$alpha,
                  CONFIG$max_cardinality))
  message(sprintf("  criterion = %s, thresholds = %s",
                  paste(CONFIG$criterion, collapse = "+"),
                  paste(CONFIG$thresholds, collapse = ", ")))

  targets <- lapply(benchmarks, function(b) b$y)
  names(targets) <- names(benchmarks)

  # No tautological target in the v4 benchmark set (overwhelm dropped), so no
  # predictor is excluded from any target's pool.
  target_exclude <- NULL

  t0 <- Sys.time()
  emsc_result <- enumerate_msc(
    data                       = X,
    targets                    = targets,
    thetas                     = CONFIG$thresholds,
    n_min                      = n_min,
    alpha                      = CONFIG$alpha,
    criterion                  = CONFIG$criterion,
    max_cardinality            = CONFIG$max_cardinality,
    max_admissible_per_depth   = CONFIG$max_admissible_per_depth,
    verbose                    = TRUE,
    target_exclude             = target_exclude,
    sufficient_detail          = CONFIG$sufficient_detail
  )
  t_emsc <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  message(sprintf("\nenumerate_msc completed in %.2f sec (%.2f min)",
                  t_emsc, t_emsc / 60))

  if (isTRUE(emsc_result$stopped_early)) {
    warning("enumerate_msc stopped early: ", emsc_result$stop_reason)
  }

  # Admissibility / evaluability diagnostics
  admis_df <- data.frame(
    depth    = seq_along(emsc_result$admissibility),
    n_admis  = emsc_result$admissibility,
    stringsAsFactors = FALSE
  )
  write.csv(admis_df,
            file.path(CONFIG$output_dir, "admissibility_per_depth.csv"),
            row.names = FALSE)

  eval_df <- as.data.frame(emsc_result$evaluability)
  eval_df$theta <- rownames(eval_df)
  write.csv(eval_df,
            file.path(CONFIG$output_dir, "evaluability_per_depth_theta.csv"),
            row.names = FALSE)

  message("\nAdmissibility per depth:")
  message("  depth | n_admissible")
  for (d in seq_along(emsc_result$admissibility)) {
    message(sprintf("  %5d | %12s", d,
                    format(emsc_result$admissibility[d], big.mark = ",")))
  }

  # --- Convert to mi_records-equivalent and build table1 -------------------
  message("\n=== Extracting antichains per cell ===")
  mi_records <- .emsc_to_records(emsc_result, benchmarks,
                                 CONFIG$criterion, CONFIG$thresholds)

  # n_sufficient per cell, read from the v4 enumerator's $n_sufficient count
  # (count mode). Fisher (2025) Table 1 reports the number of sufficient sets with
  # the number of MINIMALLY sufficient sets in parentheses; here n_sufficient
  # is the headline count and n_antichain is the parenthetical. The cell key
  # matches enumerate_msc's: paste0(<benchmark short name>, "__theta", theta).
  # If sufficient sets were not retained (older enumerator), n_sufficient is NA
  # so the column degrades gracefully rather than erroring.
  .n_sufficient_for <- function(bn, cr, thr) {
    key <- paste0(bn, "__theta", format(thr))
    # v4 count mode: the per-cell sufficient count lives in $n_sufficient.
    ns <- emsc_result$n_sufficient
    if (!is.null(ns) && !is.null(ns[[cr]]) && !is.null(ns[[cr]][[key]])) {
      return(as.integer(ns[[cr]][[key]]))
    }
    # Fallback (full-detail mode / older engine): count S_theta rows.
    suff <- emsc_result$sufficient
    if (is.null(suff)) return(NA_integer_)
    df <- suff[[cr]][[key]]
    if (is.null(df)) 0L else nrow(df)
  }

  table1_rows <- list()
  for (r in mi_records) {
    table1_rows[[length(table1_rows) + 1L]] <- data.frame(
      benchmark    = r$bench_label,
      criterion    = r$criterion,
      threshold    = r$threshold,
      n_sufficient = .n_sufficient_for(r$benchmark, r$criterion, r$threshold),
      n_antichain  = length(r$ac_member_items),
      stringsAsFactors = FALSE
    )
  }
  table1_df <- do.call(rbind, table1_rows)
  rownames(table1_df) <- NULL
  write.csv(table1_df,
            file.path(CONFIG$output_dir, "table1_all_benchmarks.csv"),
            row.names = FALSE)

  # --- Diagnostics ---------------------------------------------------------
  message("\n=== Computing diagnostics ===")
  cells <- cells_from_mi(mi_records)

  inventory_df <- build_antichain_inventory(
    cells          = cells,
    symptom_names  = items,
    symptom_labels = setNames(items, items),
    include_empty  = TRUE
  )

  # Compute cluster_signature post-hoc from set_idx and item_to_cluster.
  # (v1 carried this on candidates; here we derive it from the antichain
  # member items.)
  cluster_sig_for_key <- function(set_key) {
    if (is.na(set_key) || set_key == "") return(NA_character_)
    idx <- as.integer(strsplit(set_key, ",", fixed = TRUE)[[1]])
    paste(sort(unique(item_to_cluster[items[idx]])), collapse = "+")
  }
  inventory_df$cluster_signature <- vapply(inventory_df$set_key,
                                           cluster_sig_for_key, character(1))

  write.csv(inventory_df,
            file.path(CONFIG$output_dir, "antichain_inventory.csv"),
            row.names = FALSE)
  message("  inventory rows: ", nrow(inventory_df))

  necessity_index_df <- compute_necessity_index(
    cells          = cells,
    symptom_names  = items,
    symptom_labels = setNames(items, items)
  )
  necessity_index_df$cluster <- item_to_cluster[necessity_index_df$symptom]
  write.csv(necessity_index_df,
            file.path(CONFIG$output_dir, "necessity_index.csv"),
            row.names = FALSE)
  message("  necessity_index rows: ", nrow(necessity_index_df))

  necessity_summary_df <- summarize_necessity(
    necessity_index           = necessity_index_df,
    near_necessary_thresholds = CONFIG$near_necessary_thresholds,
    include_empty_antichains  = FALSE
  )
  write.csv(necessity_summary_df,
            file.path(CONFIG$output_dir, "necessity_summary.csv"),
            row.names = FALSE)
  message("  necessity_summary rows: ", nrow(necessity_summary_df))

  comembership_df <- compute_comembership(
    cells           = cells,
    symptom_names   = items,
    symptom_labels  = setNames(items, items),
    min_proportion  = CONFIG$comembership_min_proportion,
    include_triplets = CONFIG$comembership_include_triplets
  )
  if (nrow(comembership_df) > 0L) {
    comembership_df$cluster_signature <- vapply(
      comembership_df$set_key,
      function(sk) {
        idx <- as.integer(strsplit(sk, ",", fixed = TRUE)[[1]])
        paste(sort(unique(item_to_cluster[items[idx]])), collapse = "+")
      },
      character(1)
    )
  }
  write.csv(comembership_df,
            file.path(CONFIG$output_dir, "comembership.csv"),
            row.names = FALSE)
  message("  comembership rows: ", nrow(comembership_df))

  # --- Save bundle ---------------------------------------------------------
  result <- list(
    config             = CONFIG,
    universal_set_name = universal_set_name,
    items              = items,
    clusters           = clusters,
    item_to_cluster    = item_to_cluster,
    keep_idx           = keep_idx,
    X                  = X,
    n_min              = n_min,
    benchmarks         = benchmarks,
    item_prevalences   = item_prev,
    emsc_result        = emsc_result,
    mi_records         = mi_records,
    cells              = cells,
    table1             = table1_df,
    admissibility      = admis_df,
    evaluability       = eval_df,
    inventory          = inventory_df,
    necessity_index    = necessity_index_df,
    necessity_summary  = necessity_summary_df,
    comembership       = comembership_df,
    runtime_seconds    = t_emsc
  )

  saveRDS(result, file.path(CONFIG$output_dir, "full_results.rds"))

  message("\n=========================================================")
  message("Complete. Results saved to: ", CONFIG$output_dir)
  message(sprintf("Total enumerate_msc runtime: %.1f min", t_emsc / 60))
  message("=========================================================")

  invisible(result)
}
