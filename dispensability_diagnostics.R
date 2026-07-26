##############################################################################
# dispensability_diagnostics.R
#
# The dual of the necessity index. Where compute_necessity_index() asks
# "in what proportion of minimally sufficient configurations does symptom s
# appear?", this module asks the inverse, coverage-grounded question:
# "if symptom s were removed from the item pool, who would lose every
# minimally sufficient route to the benchmark?"
#
# The honest complement of necessity is NOT low necessity. A symptom can sit
# in very few MSCs (low necessity) yet still be the sole route to sufficiency
# for a real subpopulation -- rarely necessary, but for those cases
# irreplaceable. Rarity and redundancy are different axes. This module
# measures the second axis directly.
#
# Functions:
#   compute_dispensability_index()  per-symptom orphan / unique-coverage
#                                   counts against the FIXED antichain. Fast.
#   audit_dispensability_loso()     leave-one-symptom-out re-derivation that
#                                   AUDITS the fast index by re-running the
#                                   enumerator on V \ {s}. Heavy; validation.
#   classify_nd_quadrants()         joins necessity x dispensability into the
#                                   keystone / gateway / redundant / eliminable
#                                   2x2. Pure post-processing.
#
# -------------------------------------------------------------------------
# WHY DISPENSABILITY USES THE DATA AND NECESSITY DOES NOT
#
# The necessity index is a property of the antichain ALONE: it counts member
# sets. Dispensability, done honestly, is a property of the antichain PLUS the
# empirical joint distribution of symptoms among observed cases, because
# whether a case is "orphaned" by removing s depends on whether that case ALSO
# satisfies some s-free member -- a fact about the data, not the lattice.
# Hence these functions require X and the benchmark y, not just `cells`.
#
# This is the formal reason the naive complement (1 - necessity) fails: it
# tries to read a data-dependent quantity off a data-free one.
#
# -------------------------------------------------------------------------
# EQUIVALENCE THEOREM (fixed-antichain == leave-one-out)
#
# For a monotone sufficiency criterion with a ROW-COUNT admissibility floor
# (n_min depends on N, not on the vocabulary), deleting symptom s and
# re-deriving the antichain yields EXACTLY the s-free members of the original
# antichain. No substitute routes can emerge:
#
#   * Sufficiency of any set S subset of V\{s} is computed from the same
#     rows and columns whether or not s is in the vocabulary, so
#     Suff(V\{s}) = { S in Suff(V) : s not in S }.
#   * If an s-free set were non-minimal, its minimality witness is also
#     s-free, so minimality status is preserved under deletion.
#   * Removing a COLUMN does not change N, so n_min is untouched.
#
# Consequence: the fast fixed-antichain orphan count is not an approximation,
# it is the leave-one-out counterfactual -- PROVIDED the original enumeration
# was complete up to its max_cardinality / early-stop caps. The LOSO function
# therefore serves as a COMPLETENESS AUDIT (does the enumerator, re-run on the
# smaller lattice, recover the same s-free antichain?), not as a second
# estimator. A mismatch flags a truncated original ascent.
#
# -------------------------------------------------------------------------
# Like every quantity in this framework, dispensability is intrinsically
# "with respect to benchmark b at threshold theta" -- and, additionally, with
# respect to the realized sample's joint symptom distribution. It is reported
# on the same (benchmark x criterion x threshold) grid as the necessity index.
#
# Consumes the same unified "cell" format as antichain_diagnostics.R. Source
# that file first (these functions reuse its .cell_meta_columns / .set_to_key
# helpers).
#
# Author: Aaron Fisher / Claude assistance
##############################################################################

# Reuse the antichain_diagnostics helpers (.cell_meta_columns, .set_to_key,
# .set_to_label, %||%). In the pipeline this file is already sourced; guard in
# case dispensability_diagnostics.R is loaded standalone.
if (!exists(".cell_meta_columns", mode = "function")) {
  source("antichain_diagnostics.R")
}


# =============================================================================
# Internal helpers
# =============================================================================

#' Resolve and validate the benchmark y vector for a cell.
#'
#' @param cell unified cell (must carry $benchmark short name)
#' @param benchmarks named list keyed by benchmark short name; each element
#'   must contain a $y integer 0/1 vector aligned to the rows of X.
#' @param n_rows expected length of y (nrow(X))
#' @return integer 0/1 vector
.cell_y <- function(cell, benchmarks, n_rows) {
  bn <- cell$benchmark
  if (is.null(benchmarks[[bn]]) || is.null(benchmarks[[bn]]$y)) {
    stop("benchmarks[['", bn, "']]$y not found; cannot compute coverage for ",
         "this cell. Pass the same `benchmarks` list used in the run.")
  }
  y <- as.integer(benchmarks[[bn]]$y)
  if (length(y) != n_rows) {
    stop("benchmark '", bn, "' y has length ", length(y),
         " but X has ", n_rows, " rows; they must be the analytic sample.")
  }
  y
}

#' Build the participant x member satisfaction matrix for one cell.
#'
#' S[p, k] = 1 iff participant p endorses every item in antichain member k
#' (i.e., p "satisfies" minimally sufficient set k). Members are integer
#' index vectors into the columns of X (which are ordered as symptom_names).
#'
#' @param X integer 0/1 matrix (participants x items), columns = symptom_names
#' @param members list of integer index vectors (one per antichain member)
#' @return integer matrix N x length(members) (0/1)
.satisfaction_matrix <- function(X, members) {
  n_obs <- nrow(X)
  r <- length(members)
  S <- matrix(0L, nrow = n_obs, ncol = r)
  for (k in seq_len(r)) {
    m <- members[[k]]
    if (length(m) == 1L) {
      S[, k] <- X[, m]
    } else {
      S[, k] <- as.integer(rowSums(X[, m, drop = FALSE]) == length(m))
    }
  }
  S
}


# =============================================================================
# 1. Dispensability index (fast; fixed antichain)
# =============================================================================

#' Per-symptom dispensability against the fixed antichain
#'
#' For each cell and each symptom s, computes the "orphan" count: the number
#' of participants who currently satisfy at least one minimally sufficient set
#' but whose satisfied sets ALL contain s -- i.e., who would have zero
#' sufficient routes if s were deleted from the item pool. The dispensability
#' index is the complement of the orphan share:
#'
#'   dispensability(s) = 1 - n_orphan(s) / n_covered
#'
#'   * dispensability = 1  : removing s orphans no one. Fully replaceable
#'     (or never in any MSC, i.e. inert). Eliminable for THIS cell.
#'   * dispensability = 0  : every covered case relies on s. Maximally
#'     essential. A strictly necessary symptom (necessity = 1) always lands
#'     here.
#'   * intermediate         : s uniquely covers some cases but not others.
#'
#' Note the relationship to necessity is NOT 1 - necessity. A low-necessity
#' symptom that is the sole route for a small subgroup has LOW dispensability
#' (the "specialist gateway"); a high-necessity symptom whose cases are all
#' also covered by other members has HIGH dispensability. The two indices are
#' orthogonal; see classify_nd_quadrants().
#'
#' Two denominators are reported. The primary index uses the full coverage
#' footprint (anyone satisfying >= 1 MSC). The `_y1` variant restricts to
#' benchmark-positive cases (y == 1) -- the sensitivity-relevant reading,
#' since orphaning a benchmark-positive case turns a true positive into a
#' false negative under the antichain's coverage.
#'
#' @param cells list of unified-format cells (see antichain_diagnostics.R)
#' @param X integer 0/1 matrix, participants x items. Column names must cover
#'   symptom_names; columns are reordered to symptom_names internally so that
#'   the integer member indices align.
#' @param benchmarks named list keyed by benchmark short name; each element
#'   has $y (0/1 vector aligned to rows of X). The same object passed to the
#'   enumerator run.
#' @param symptom_names character vector of symptom short names (the column
#'   ordering that member indices refer to)
#' @param symptom_labels named character vector mapping symptom_names -> labels
#' @return data.frame in long format: source, alpha, benchmark, bench_label,
#'   criterion, threshold, key, n_antichain, symptom, label, n_msc_with,
#'   n_covered, n_orphan, dispensability, n_covered_y1, n_orphan_y1,
#'   dispensability_y1. One row per (cell x symptom). Empty antichains yield
#'   NA-valued rows, matching compute_necessity_index().
compute_dispensability_index <- function(cells, X, benchmarks,
                                         symptom_names, symptom_labels) {
  K <- length(symptom_names)

  # Align X columns to symptom_names so integer member indices index correctly.
  missing_cols <- setdiff(symptom_names, colnames(X))
  if (length(missing_cols) > 0L) {
    stop("X is missing columns for symptoms: ",
         paste(missing_cols, collapse = ", "))
  }
  X <- X[, symptom_names, drop = FALSE]
  storage.mode(X) <- "integer"
  n_obs <- nrow(X)

  rows <- list()

  for (cell in cells) {
    n_ac <- length(cell$members)

    # --- Empty antichain: emit NA rows, parallel to necessity index --------
    if (n_ac == 0L) {
      meta <- .cell_meta_columns(cell, K)
      rows[[length(rows) + 1L]] <- cbind(
        meta,
        data.frame(
          n_antichain      = rep(0L, K),
          symptom          = symptom_names,
          label            = unname(symptom_labels[symptom_names]),
          n_msc_with       = rep(0L, K),
          n_covered        = rep(0L, K),
          n_orphan         = rep(0L, K),
          dispensability   = rep(NA_real_, K),
          n_covered_y1     = rep(0L, K),
          n_orphan_y1      = rep(0L, K),
          dispensability_y1 = rep(NA_real_, K),
          stringsAsFactors = FALSE
        )
      )
      next
    }

    y <- .cell_y(cell, benchmarks, n_obs)

    # Satisfaction matrix and coverage footprint
    S <- .satisfaction_matrix(X, cell$members)
    covered     <- rowSums(S) > 0L           # >= 1 satisfied MSC
    covered_y1  <- covered & (y == 1L)
    n_covered    <- sum(covered)
    n_covered_y1 <- sum(covered_y1)

    # Membership matrix B[s, k] = 1 iff symptom s is in member k
    B <- matrix(0L, nrow = K, ncol = n_ac)
    for (k in seq_len(n_ac)) B[cell$members[[k]], k] <- 1L
    n_msc_with <- rowSums(B)                  # == necessity count

    n_orphan    <- integer(K)
    n_orphan_y1 <- integer(K)

    for (s in seq_len(K)) {
      without_cols <- which(B[s, ] == 0L)     # members NOT containing s
      if (length(without_cols) == 0L) {
        # Every member contains s (s is necessary here): all covered cases
        # are orphaned by removing s.
        cov_without <- rep(FALSE, n_obs)
      } else {
        cov_without <- rowSums(S[, without_cols, drop = FALSE]) > 0L
      }
      orphan <- covered & !cov_without        # covered, but only via s-members
      n_orphan[s]    <- sum(orphan)
      n_orphan_y1[s] <- sum(orphan & (y == 1L))
    }

    disp    <- if (n_covered    > 0L) 1 - n_orphan    / n_covered    else NA_real_
    disp_y1 <- if (n_covered_y1 > 0L) 1 - n_orphan_y1 / n_covered_y1 else NA_real_

    meta <- .cell_meta_columns(cell, K)
    rows[[length(rows) + 1L]] <- cbind(
      meta,
      data.frame(
        n_antichain      = rep(n_ac, K),
        symptom          = symptom_names,
        label            = unname(symptom_labels[symptom_names]),
        n_msc_with       = as.integer(n_msc_with),
        n_covered        = rep(as.integer(n_covered), K),
        n_orphan         = as.integer(n_orphan),
        dispensability   = round(disp, 6),
        n_covered_y1     = rep(as.integer(n_covered_y1), K),
        n_orphan_y1      = as.integer(n_orphan_y1),
        dispensability_y1 = round(disp_y1, 6),
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
# 2. Leave-one-symptom-out audit (heavy; validation)
# =============================================================================

#' Audit the fast dispensability index by re-deriving each antichain on V\{s}
#'
#' For each audited (cell x symptom s), re-runs the enumerator on X with the
#' column for s deleted, then checks whether the resulting antichain equals the
#' s-free members of the STORED antichain (set equality of member item-sets).
#' Under the equivalence theorem in the file header these must match; a
#' mismatch indicates the original enumeration was truncated (cardinality cap
#' or early stop), so the fast index for that cell is provisional.
#'
#' This is expensive: one enumerator call per (cell x symptom). Scope it with
#' `benchmarks_audit`, `criteria`, `thresholds`, and `symptoms`. By default it
#' audits, for each cell, only the symptoms that actually appear in >= 1 member
#' of that cell (the only symptoms whose removal can change coverage).
#'
#' @param cells list of unified-format cells
#' @param X integer 0/1 matrix, participants x items (columns >= symptom_names)
#' @param benchmarks named list keyed by benchmark short name, each with $y
#' @param symptom_names character vector (column ordering for member indices)
#' @param enumerate_fn the enumerator function (default: enumerate_msc; must be
#'   in scope -- source enumerate_msc_saveall.R)
#' @param n_min,alpha,max_cardinality,max_admissible_per_depth run parameters;
#'   MUST match those used to produce `cells`. Read them from the saved
#'   config / full_results bundle.
#' @param target_exclude named list (target short name -> item names to forbid
#'   in that target's predictor pool), as passed to the original run; NULL if
#'   none. The audited symptom is dropped from any exclude list automatically.
#' @param benchmarks_audit optional character vector of benchmark short names
#'   to audit (default: all present in cells)
#' @param criteria optional character vector subset of c("wilson","point")
#' @param thresholds optional numeric vector of thetas to audit (default: all)
#' @param symptoms optional character vector of symptom short names to audit
#'   (default: per-cell, the symptoms appearing in >= 1 member)
#' @param verbose logical
#' @return data.frame: source, alpha, benchmark, bench_label, criterion,
#'   threshold, key, symptom, label, n_msc_total, n_msc_with_s,
#'   n_msc_without_s_fixed, n_msc_loso, members_match, note. members_match ==
#'   FALSE is the audit's red flag.
audit_dispensability_loso <- function(cells, X, benchmarks, symptom_names,
                                      symptom_labels = NULL,
                                      enumerate_fn = NULL,
                                      n_min,
                                      alpha = 0.05,
                                      max_cardinality = 8L,
                                      max_admissible_per_depth = 1e7,
                                      target_exclude = NULL,
                                      benchmarks_audit = NULL,
                                      criteria = NULL,
                                      thresholds = NULL,
                                      symptoms = NULL,
                                      verbose = TRUE) {

  if (is.null(enumerate_fn)) {
    if (!exists("enumerate_msc", mode = "function")) {
      stop("enumerate_msc not found. source('enumerate_msc_saveall.R') first, ",
           "or pass enumerate_fn explicitly.")
    }
    enumerate_fn <- get("enumerate_msc", mode = "function")
  }

  missing_cols <- setdiff(symptom_names, colnames(X))
  if (length(missing_cols) > 0L) {
    stop("X is missing columns for symptoms: ",
         paste(missing_cols, collapse = ", "))
  }
  X <- X[, symptom_names, drop = FALSE]
  storage.mode(X) <- "integer"

  if (is.null(symptom_labels)) {
    symptom_labels <- setNames(symptom_names, symptom_names)
  }

  # Canonical set-of-sets key for order-insensitive antichain comparison.
  # Accepts members given as item-NAME vectors; each member -> sorted names.
  namesets_from_names <- function(name_members) {
    if (length(name_members) == 0L) return(character(0))
    sort(vapply(name_members, function(nm) {
      paste(sort(nm), collapse = "|")
    }, character(1)))
  }

  out_rows <- list()

  for (cell in cells) {
    if (length(cell$members) == 0L) next
    bn  <- cell$benchmark
    cr  <- cell$criterion
    thr <- cell$threshold

    if (!is.null(benchmarks_audit) && !(bn %in% benchmarks_audit)) next
    if (!is.null(criteria)         && !(cr %in% criteria))         next
    if (!is.null(thresholds)       && !any(abs(thresholds - thr) < 1e-9)) next

    y <- .cell_y(cell, benchmarks, nrow(X))

    member_items <- lapply(cell$members, function(idx) symptom_names[idx])

    # Candidate symptoms: those in >= 1 member, optionally intersected with the
    # caller's `symptoms` restriction.
    in_any <- sort(unique(unlist(member_items, use.names = FALSE)))
    cand <- if (is.null(symptoms)) in_any else intersect(symptoms, in_any)
    if (length(cand) == 0L) next

    if (verbose) {
      message(sprintf("LOSO audit | %s / %s / theta=%s | %d candidate symptom(s)",
                      bn, cr, format(thr), length(cand)))
    }

    for (s_name in cand) {
      s_col <- match(s_name, colnames(X))
      X_red <- X[, -s_col, drop = FALSE]
      vocab_red <- colnames(X_red)

      # Stored s-free members (the theorem's predicted LOSO antichain)
      keep_fixed <- !vapply(member_items, function(it) s_name %in% it, logical(1))
      fixed_without_namesets <- namesets_from_names(member_items[keep_fixed])

      # Scope target_exclude to this target, dropping s if present (s is gone
      # from the reduced vocabulary, so referencing it would error).
      te <- NULL
      if (!is.null(target_exclude) && !is.null(target_exclude[[bn]])) {
        ex <- setdiff(target_exclude[[bn]], s_name)
        if (length(ex) > 0L) te <- setNames(list(ex), bn)
      }

      note <- ""
      loso_namesets <- character(0)
      res <- tryCatch(
        enumerate_fn(
          data                     = X_red,
          targets                  = setNames(list(y), bn),
          thetas                   = thr,
          n_min                    = n_min,
          alpha                    = alpha,
          criterion                = cr,
          max_cardinality          = max_cardinality,
          max_admissible_per_depth = max_admissible_per_depth,
          verbose                  = FALSE,
          target_exclude           = te
        ),
        error = function(e) { note <<- paste0("enumerator error: ", conditionMessage(e)); NULL }
      )

      if (!is.null(res)) {
        theta_key <- paste0(bn, "__theta", format(thr))
        df <- res$minimally_sufficient[[cr]][[theta_key]]
        if (!is.null(df) && nrow(df) > 0L) {
          loso_members <- lapply(df$set_idx, function(ix) vocab_red[ix])
          loso_namesets <- namesets_from_names(loso_members)
        }
        if (isTRUE(res$stopped_early)) {
          note <- paste0(note, if (nzchar(note)) "; " else "",
                         "loso enumerator stopped_early: ",
                         res$stop_reason %||% "")
        }
      }

      members_match <- setequal(loso_namesets, fixed_without_namesets)

      out_rows[[length(out_rows) + 1L]] <- cbind(
        .cell_meta_columns(cell, 1L),
        data.frame(
          symptom               = s_name,
          label                 = unname(symptom_labels[s_name]),
          n_msc_total           = length(cell$members),
          n_msc_with_s          = sum(!keep_fixed),
          n_msc_without_s_fixed = sum(keep_fixed),
          n_msc_loso            = length(loso_namesets),
          members_match         = members_match,
          note                  = note,
          stringsAsFactors      = FALSE
        )
      )
    }
  }

  if (length(out_rows) == 0L) return(data.frame())
  out <- do.call(rbind, out_rows)
  rownames(out) <- NULL
  out
}

# =============================================================================
# 3. Necessity x Dispensability quadrant classification
# =============================================================================

#' Join necessity and dispensability into the keystone/gateway/redundant/
#' eliminable 2x2.
#'
#' The two indices are orthogonal axes:
#'
#'                          | irreplaceable (low delta) | replaceable (high delta)
#'   ----------------------+---------------------------+-------------------------
#'   necessary  (high nu)  |        keystone           |   redundant_ubiquitous
#'   peripheral (low  nu)  |    specialist_gateway     |       eliminable
#'
#' "Unnecessary" lives only in the bottom-right (eliminable). The bottom-left
#' (specialist_gateway: low necessity but low dispensability) is exactly the
#' cell that necessity alone would wrongly discard -- a symptom that is the
#' sole route for a subgroup. The top-right (high necessity yet replaceable)
#' is rare and usually signals a collinear/redundant cluster.
#'
#' @param necessity_index data.frame from compute_necessity_index()
#' @param dispensability_index data.frame from compute_dispensability_index()
#' @param near_necessary numeric cutoff for "high necessity" (nu >=). Default
#'   0.95.
#' @param near_dispensable numeric cutoff for "high dispensability"
#'   (delta >=). Default 0.95.
#' @param use_y1 logical; if TRUE, classify on dispensability_y1 (the
#'   benchmark-positive / sensitivity reading) instead of the full-footprint
#'   dispensability. Default FALSE.
#' @return data.frame: source, alpha, benchmark, bench_label, criterion,
#'   threshold, key, symptom, label, necessity, dispensability, quadrant.
classify_nd_quadrants <- function(necessity_index, dispensability_index,
                                  near_necessary   = 0.95,
                                  near_dispensable = 0.95,
                                  use_y1 = FALSE) {
  if (nrow(necessity_index) == 0L || nrow(dispensability_index) == 0L) {
    return(data.frame())
  }

  join_cols <- c("benchmark", "bench_label", "criterion", "threshold",
                 "key", "symptom")
  disp_col <- if (use_y1) "dispensability_y1" else "dispensability"

  ni <- necessity_index[, c("source", "alpha", join_cols, "label",
                            "proportion")]
  names(ni)[names(ni) == "proportion"] <- "necessity"

  di <- dispensability_index[, c(join_cols, disp_col)]
  names(di)[names(di) == disp_col] <- "dispensability"

  m <- merge(ni, di, by = join_cols, all.x = TRUE, sort = FALSE)

  nu    <- m$necessity
  delta <- m$dispensability

  quadrant <- rep(NA_character_, nrow(m))
  hi_nu    <- !is.na(nu)    & nu    >= near_necessary
  hi_delta <- !is.na(delta) & delta >= near_dispensable

  quadrant[ hi_nu & !hi_delta] <- "keystone"
  quadrant[ hi_nu &  hi_delta] <- "redundant_ubiquitous"
  quadrant[!hi_nu & !hi_delta] <- "specialist_gateway"
  quadrant[!hi_nu &  hi_delta] <- "eliminable"
  # Symptoms in no MSC (nu == 0, delta == 1 by construction) fall into
  # "eliminable"; flag the fully inert ones distinctly for clarity.
  quadrant[!is.na(nu) & nu == 0] <- "inert"
  quadrant[is.na(nu) | is.na(delta)] <- NA_character_

  out <- data.frame(
    source       = m$source,
    alpha        = m$alpha,
    benchmark    = m$benchmark,
    bench_label  = m$bench_label,
    criterion    = m$criterion,
    threshold    = m$threshold,
    key          = m$key,
    symptom      = m$symptom,
    label        = m$label,
    necessity    = round(nu, 6),
    dispensability = round(delta, 6),
    quadrant     = quadrant,
    stringsAsFactors = FALSE
  )
  out <- out[order(out$benchmark, out$criterion, -out$threshold,
                   out$symptom), , drop = FALSE]
  rownames(out) <- NULL
  out
}


# =============================================================================
# Print usage
# =============================================================================
if (sys.nframe() == 0L) {
  message("dispensability_diagnostics.R loaded.")
  message("Functions:")
  message("  compute_dispensability_index()  - per-symptom orphan / coverage (fast)")
  message("  audit_dispensability_loso()     - leave-one-out re-derivation (audit)")
  message("  classify_nd_quadrants()         - necessity x dispensability 2x2")
}
