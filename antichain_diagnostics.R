##############################################################################
# antichain_diagnostics.R
#
# Reusable diagnostic functions for antichain results from set-theoretic
# psychopathology analyses. Operates on a "unified cell" representation that
# is agnostic to whether the antichain came from exhaustive search or
# MI-restricted search; both sources can therefore be processed identically.
#
# Functions:
#   build_antichain_inventory()   long-format CSV of every antichain member
#   compute_necessity_index()     proportion of members containing each item
#   summarize_necessity()         high-level rollup matching the original
#                                 necessity_summary() format
#   compute_comembership()        pair- and triplet-level co-occurrence
#                                 within antichain members
#
# Unified cell format. A "cell" is a single (source × alpha × benchmark ×
# criterion × threshold) result. Each cell is a named list:
#   $source        character  ("exhaustive" or "mi_restricted")
#   $alpha         numeric or NA_real_ (MI alpha; NA for exhaustive)
#   $benchmark     character  (the benchmark's `bn` short name, e.g. "k10_elev")
#   $bench_label   character  (human-readable benchmark label)
#   $criterion     character  ("point" or "wilson")
#   $threshold     numeric    (e.g. 0.95)
#   $key           character  (e.g. "point_p095")
#   $members       list of integer vectors. Each vector is one antichain
#                  member, holding 1-based indices into symptom_names.
#                  Length 0 list = empty antichain.
#   $cprob         numeric vector, same length as members
#   $wilson_lower  numeric vector, same length as members
#   $n_set         integer vector, same length as members
#   $n_set_y1      integer vector, same length as members
#
# All outputs are tidy data frames suitable for write.csv().
#
# Author: Aaron Fisher / Claude assistance
##############################################################################


# =============================================================================
# Internal helpers
# =============================================================================

# null-coalesce
`%||%` <- function(a, b) if (is.null(a)) b else a

.set_to_label <- function(idx, symptom_labels, symptom_names, sep = " + ") {
  if (length(idx) == 0L) return("")
  paste(symptom_labels[symptom_names[idx]], collapse = sep)
}

.set_to_key <- function(idx, sep = ",") {
  if (length(idx) == 0L) return("")
  paste(sort(idx), collapse = sep)
}

.cell_meta_columns <- function(cell, n_rows) {
  data.frame(
    source     = rep(cell$source,      n_rows),
    alpha      = rep(cell$alpha,       n_rows),
    benchmark  = rep(cell$benchmark,   n_rows),
    bench_label = rep(cell$bench_label, n_rows),
    criterion  = rep(cell$criterion,   n_rows),
    threshold  = rep(cell$threshold,   n_rows),
    key        = rep(cell$key,         n_rows),
    stringsAsFactors = FALSE
  )
}


# =============================================================================
# 1. Antichain inventory
# =============================================================================

#' Long-format inventory of every antichain member across all cells
#'
#' @param cells list of unified-format cells (see header)
#' @param symptom_names character vector of symptom short names
#' @param symptom_labels named character vector mapping symptom_names ->
#'   human-readable labels
#' @param include_empty logical; if TRUE, cells with empty antichains get
#'   one row with NA-valued member columns. If FALSE, they are omitted.
#' @return data.frame with columns: source, alpha, benchmark, bench_label,
#'   criterion, threshold, key, set_key, set_label, cardinality, cprob,
#'   wilson_lower, n_set, n_set_y1
build_antichain_inventory <- function(cells, symptom_names, symptom_labels,
                                      include_empty = TRUE) {
  rows <- list()

  for (cell in cells) {
    n_members <- length(cell$members)

    if (n_members == 0L) {
      if (!include_empty) next
      meta <- .cell_meta_columns(cell, 1L)
      rows[[length(rows) + 1L]] <- cbind(
        meta,
        data.frame(
          set_key      = NA_character_,
          set_label    = NA_character_,
          cardinality  = NA_integer_,
          cprob        = NA_real_,
          wilson_lower = NA_real_,
          n_set        = NA_integer_,
          n_set_y1     = NA_integer_,
          stringsAsFactors = FALSE
        )
      )
      next
    }

    set_keys   <- vapply(cell$members, .set_to_key,   character(1))
    set_labels <- vapply(cell$members, .set_to_label, character(1),
                         symptom_labels = symptom_labels,
                         symptom_names  = symptom_names)
    cards      <- vapply(cell$members, length, integer(1))

    meta <- .cell_meta_columns(cell, n_members)
    rows[[length(rows) + 1L]] <- cbind(
      meta,
      data.frame(
        set_key      = set_keys,
        set_label    = set_labels,
        cardinality  = cards,
        cprob        = round(cell$cprob, 6),
        wilson_lower = round(cell$wilson_lower, 6),
        n_set        = as.integer(cell$n_set),
        n_set_y1     = as.integer(cell$n_set_y1),
        stringsAsFactors = FALSE
      )
    )
  }

  if (length(rows) == 0L) {
    return(data.frame())
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}


# =============================================================================
# 2. Necessity index
# =============================================================================

#' Proportion of antichain members containing each symptom, for each cell
#'
#' For each cell, computes the proportion of antichain members in which a
#' given symptom appears. proportion = 1.0 means the symptom is *necessary*
#' for the corresponding (benchmark, criterion, threshold) configuration;
#' proportion = 0.0 means the symptom never appears in any minimally
#' sufficient configuration.
#'
#' @param cells list of unified-format cells
#' @param symptom_names character vector of symptom short names
#' @param symptom_labels named character vector mapping symptom_names ->
#'   labels
#' @return data.frame in long format: source, alpha, benchmark, bench_label,
#'   criterion, threshold, key, n_antichain, symptom, label, count, proportion
compute_necessity_index <- function(cells, symptom_names, symptom_labels) {
  K <- length(symptom_names)
  rows <- list()

  for (cell in cells) {
    n_ac <- length(cell$members)

    if (n_ac == 0L) {
      meta <- .cell_meta_columns(cell, K)
      rows[[length(rows) + 1L]] <- cbind(
        meta,
        data.frame(
          n_antichain = rep(0L, K),
          symptom     = symptom_names,
          label       = unname(symptom_labels[symptom_names]),
          count       = rep(0L, K),
          proportion  = rep(NA_real_, K),
          stringsAsFactors = FALSE
        )
      )
      next
    }

    flat <- unlist(cell$members, use.names = FALSE)
    counts <- tabulate(flat, nbins = K)
    props  <- counts / n_ac

    meta <- .cell_meta_columns(cell, K)
    rows[[length(rows) + 1L]] <- cbind(
      meta,
      data.frame(
        n_antichain = rep(n_ac, K),
        symptom     = symptom_names,
        label       = unname(symptom_labels[symptom_names]),
        count       = as.integer(counts),
        proportion  = round(props, 6),
        stringsAsFactors = FALSE
      )
    )
  }

  if (length(rows) == 0L) return(data.frame())
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}


# =============================================================================
# 3. Necessity summary (rollup)
# =============================================================================

#' Per-cell rollup of necessary and near-necessary symptoms
#'
#' Matches the format of the original necessity_summary() utility:
#' for each cell, lists symptoms with proportion = 1.0 (necessary) and
#' symptoms with near_necessary_threshold <= proportion < 1.0 (near-
#' necessary). Multiple thresholds may be passed; one column per threshold.
#'
#' @param necessity_index data.frame from compute_necessity_index()
#' @param near_necessary_thresholds numeric vector. Default c(0.90).
#'   Values must be in (0, 1).
#' @param include_empty_antichains logical. If FALSE, cells with
#'   n_antichain == 0 are omitted (matches the original behavior).
#' @return data.frame with one row per cell.
summarize_necessity <- function(necessity_index,
                                near_necessary_thresholds = 0.90,
                                include_empty_antichains  = FALSE) {
  if (nrow(necessity_index) == 0L) return(data.frame())
  stopifnot(all(near_necessary_thresholds > 0 & near_necessary_thresholds < 1))
  near_necessary_thresholds <- sort(near_necessary_thresholds, decreasing = TRUE)

  cell_id_cols <- c("source", "alpha", "benchmark", "bench_label",
                    "criterion", "threshold", "key")

  # Build a stable per-row cell key using paste; split() preserves order
  # within each group.
  ni <- necessity_index
  # NA-safe key: replace NA in alpha (the only column that can be NA among
  # cell_id_cols) with a sentinel so paste produces matching keys.
  alpha_chr <- ifelse(is.na(ni$alpha), "NA", as.character(ni$alpha))
  cell_key_per_row <- paste(ni$source, alpha_chr, ni$benchmark, ni$bench_label,
                            ni$criterion, ni$threshold, ni$key,
                            sep = "\x1f")  # unit-separator avoids collision

  splits <- split(seq_len(nrow(ni)), cell_key_per_row)

  rows <- vector("list", length(splits))
  for (i in seq_along(splits)) {
    idx <- splits[[i]]
    sub <- ni[idx, , drop = FALSE]

    n_ac <- sub$n_antichain[1]
    if (!include_empty_antichains && (is.na(n_ac) || n_ac == 0L)) {
      rows[[i]] <- NULL
      next
    }

    # Preserve the input row order (which reflects the symptom order from
    # compute_necessity_index) when listing necessary / near-necessary syms.
    necessary_mask <- !is.na(sub$proportion) & sub$proportion == 1.0
    necessary_str  <- if (any(necessary_mask))
      paste(sub$label[necessary_mask], collapse = "; ") else "none"

    near_cols <- list()
    for (t in near_necessary_thresholds) {
      mask <- !is.na(sub$proportion) &
              sub$proportion >= t &
              sub$proportion < 1.0
      if (any(mask)) {
        near_str <- paste(
          sprintf("%s (%.0f%%)",
                  sub$label[mask],
                  100 * sub$proportion[mask]),
          collapse = "; "
        )
      } else {
        near_str <- "none"
      }
      col_name <- sprintf("near_necessary_%02d", round(t * 100))
      near_cols[[col_name]] <- near_str
    }

    cid <- sub[1, cell_id_cols, drop = FALSE]
    row <- cbind(
      cid,
      data.frame(
        n_antichain = n_ac,
        necessary   = necessary_str,
        stringsAsFactors = FALSE
      ),
      as.data.frame(near_cols, stringsAsFactors = FALSE)
    )
    rows[[i]] <- row
  }

  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) return(data.frame())
  out <- do.call(rbind, rows)
  rownames(out) <- NULL

  # Sort output deterministically: source, alpha (NA last), benchmark,
  # criterion, threshold descending
  out <- out[order(out$source,
                   ifelse(is.na(out$alpha), Inf, out$alpha),
                   out$benchmark,
                   out$criterion,
                   -out$threshold), , drop = FALSE]
  rownames(out) <- NULL
  out
}


# =============================================================================
# 4. Co-membership (pairs and triplets)
# =============================================================================

#' Pair- and triplet-level co-occurrence within antichain members
#'
#' For each cell, counts how many antichain members contain each pair or
#' triplet of symptoms. Only pairs/triplets meeting the minimum proportion
#' threshold are returned. This identifies symptom doublets and triplets
#' that "travel together" across configurations.
#'
#' @param cells list of unified-format cells
#' @param symptom_names character vector of symptom short names
#' @param symptom_labels named character vector mapping symptom_names ->
#'   labels
#' @param min_proportion numeric in (0, 1]; only pairs/triplets that appear
#'   in at least this proportion of antichain members are returned.
#'   Default 0.50.
#' @param include_triplets logical; if FALSE, only pairs are returned.
#'   Default TRUE.
#' @return data.frame with columns: source, alpha, benchmark, bench_label,
#'   criterion, threshold, key, n_antichain, set_size, set_key, set_label,
#'   count, proportion. Sorted within cell by descending proportion.
compute_comembership <- function(cells, symptom_names, symptom_labels,
                                 min_proportion   = 0.50,
                                 include_triplets = TRUE) {
  stopifnot(min_proportion > 0 && min_proportion <= 1)
  K <- length(symptom_names)
  rows <- list()

  for (cell in cells) {
    n_ac <- length(cell$members)
    if (n_ac == 0L) next

    # Pairs (require K >= 2)
    if (K >= 2L) {
      pair_counts <- matrix(0L, nrow = K, ncol = K)
      for (m in cell$members) {
        if (length(m) < 2L) next
        pair_counts[m, m] <- pair_counts[m, m] + 1L
      }
      diag(pair_counts) <- 0L

      pair_rows <- list()
      for (i in seq_len(K - 1L)) {
        for (j in (i + 1L):K) {
          cnt <- pair_counts[i, j]
          if (cnt == 0L) next
          prop <- cnt / n_ac
          if (prop < min_proportion) next
          idx <- c(i, j)
          pair_rows[[length(pair_rows) + 1L]] <- data.frame(
            set_size  = 2L,
            set_key   = .set_to_key(idx),
            set_label = .set_to_label(idx, symptom_labels, symptom_names),
            count     = cnt,
            proportion = round(prop, 6),
            stringsAsFactors = FALSE
          )
        }
      }
    } else {
      pair_rows <- list()
    }

    # Triplets
    triplet_rows <- list()
    if (include_triplets) {
      members_with_3 <- cell$members[
        vapply(cell$members, length, integer(1)) >= 3L
      ]
      if (length(members_with_3) > 0L) {
        triplet_counts <- list()  # keyed by sorted "i,j,k"
        for (m in members_with_3) {
          combos <- utils::combn(sort(m), 3L)
          for (col in seq_len(ncol(combos))) {
            ijk <- combos[, col]
            k <- paste(ijk, collapse = ",")
            triplet_counts[[k]] <- (triplet_counts[[k]] %||% 0L) + 1L
          }
        }
        for (k in names(triplet_counts)) {
          cnt <- triplet_counts[[k]]
          prop <- cnt / n_ac
          if (prop < min_proportion) next
          idx <- as.integer(strsplit(k, ",", fixed = TRUE)[[1]])
          triplet_rows[[length(triplet_rows) + 1L]] <- data.frame(
            set_size  = 3L,
            set_key   = k,
            set_label = .set_to_label(idx, symptom_labels, symptom_names),
            count     = cnt,
            proportion = round(prop, 6),
            stringsAsFactors = FALSE
          )
        }
      }
    }

    cell_rows <- c(pair_rows, triplet_rows)
    if (length(cell_rows) == 0L) next
    body <- do.call(rbind, cell_rows)
    body <- body[order(-body$proportion, body$set_size, body$set_key), ,
                 drop = FALSE]
    body$n_antichain <- n_ac

    meta <- .cell_meta_columns(cell, nrow(body))
    rows[[length(rows) + 1L]] <- cbind(
      meta,
      body[, c("n_antichain", "set_size", "set_key", "set_label",
               "count", "proportion")]
    )
  }

  if (length(rows) == 0L) return(data.frame())
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

# =============================================================================
# 5. Cell builders for the two analysis sources
# =============================================================================

#' Build unified cells from an exhaustive run_set_analysis() result
#'
#' @param exhaustive_results named list: one entry per benchmark, each the
#'   return value of run_set_analysis().
#' @param benchmarks named list of benchmark specs (each with $label).
#'   Names must align with names(exhaustive_results).
#' @return list of unified cells (one per benchmark × key)
cells_from_exhaustive <- function(exhaustive_results, benchmarks) {
  cells <- list()
  for (bn in names(exhaustive_results)) {
    res <- exhaustive_results[[bn]]
    bench_label <- benchmarks[[bn]]$label

    for (key in names(res$antichains)) {
      # Parse key like "point_p095" -> criterion="point", threshold=0.95
      m <- regmatches(key, regexec("^([^_]+)_p(\\d{3})$", key))[[1]]
      if (length(m) < 3L) {
        warning("Could not parse antichain key: ", key)
        next
      }
      criterion <- m[2]
      threshold <- as.integer(m[3]) / 100

      ac_idx  <- res$antichains[[key]]
      members <- res$sets[ac_idx]

      cells[[length(cells) + 1L]] <- list(
        source       = "exhaustive",
        alpha        = NA_real_,
        benchmark    = bn,
        bench_label  = bench_label,
        criterion    = criterion,
        threshold    = threshold,
        key          = key,
        members      = members,
        cprob        = res$cp_table$cprob[ac_idx],
        wilson_lower = res$cp_table$wilson_lower_95[ac_idx],
        n_set        = res$cp_table$n_set[ac_idx],
        n_set_y1     = res$cp_table$n_set_y1[ac_idx]
      )
    }
  }
  cells
}


#' Build unified cells from MI-restricted antichain results
#'
#' @param mi_records list of per-cell records, where each record has fields:
#'   alpha, benchmark (short name), bench_label, criterion, threshold, key,
#'   ac_member_items (list of integer index vectors), cprob, wilson_lower,
#'   n_set, n_set_y1. Build this list inside the analysis loop.
#' @return list of unified cells
cells_from_mi <- function(mi_records) {
  cells <- vector("list", length(mi_records))
  for (i in seq_along(mi_records)) {
    r <- mi_records[[i]]
    cells[[i]] <- list(
      source       = "mi_restricted",
      alpha        = r$alpha,
      benchmark    = r$benchmark,
      bench_label  = r$bench_label,
      criterion    = r$criterion,
      threshold    = r$threshold,
      key          = r$key,
      members      = r$ac_member_items,
      cprob        = r$cprob,
      wilson_lower = r$wilson_lower,
      n_set        = r$n_set,
      n_set_y1     = r$n_set_y1
    )
  }
  cells
}


# =============================================================================
# Print usage
# =============================================================================
if (sys.nframe() == 0L) {
  message("antichain_diagnostics.R loaded.")
  message("Functions:")
  message("  build_antichain_inventory()   — long inventory CSV")
  message("  compute_necessity_index()     — per-symptom proportions")
  message("  summarize_necessity()         — necessity rollup")
  message("  compute_comembership()        — pair/triplet co-occurrence")
  message("  cells_from_exhaustive()       — adapter from run_set_analysis()")
  message("  cells_from_mi()               — adapter from MI loop records")
}
