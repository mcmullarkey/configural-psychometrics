##############################################################################
# enumerate_msc_v4.R
#
# Drop-in replacement for enumerate_msc() in enumerate_msc_saveall.R,
# with the ascent, sufficiency evaluation, and antichain Reduce moved
# into a C++ core (emsc_v4.cpp). Public signature, validation, verbose
# messages, and the returned object structure are unchanged; on identical
# inputs the result is identical() to the R reference implementation.
#
# Differences from the reference (documented, additive only):
#   - Requires Rcpp and a working C++ toolchain (compiled once per session).
#   - V is capped at 64 vocabulary elements (single-word vocab mask).
#     The reference's soft cap was 256; raise by moving to 2-word masks
#     if a >64-item universal set is ever needed.
#   - Row bitsets are held only for the current/previous depth, so memory
#     no longer scales with N x (total admissible configurations).
##############################################################################

.emsc4_env <- new.env(parent = baseenv())

.emsc4_compile <- function(verbose = FALSE) {
  if (!is.null(.emsc4_env$compiled)) return(invisible(TRUE))
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("enumerate_msc_v4 requires the 'Rcpp' package.")
  }
  cpp_path <- .emsc4_env$cpp_path
  if (is.null(cpp_path)) cpp_path <- "emsc_v4.cpp"
  if (!file.exists(cpp_path)) {
    stop("Cannot find emsc_v4.cpp; set .emsc4_env$cpp_path to its location ",
         "before calling enumerate_msc().")
  }
  Rcpp::sourceCpp(cpp_path, env = .emsc4_env)
  .emsc4_env$compiled <- TRUE
  invisible(TRUE)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

enumerate_msc <- function(data,
                          targets,
                          thetas,
                          n_min,
                          alpha = 0.05,
                          criterion = c("wilson", "point"),
                          max_cardinality = 8L,
                          max_admissible_per_depth = 1e7,
                          verbose = TRUE,
                          target_exclude = NULL,
                          sufficient_detail = c("full", "count")) {

  sufficient_detail <- match.arg(sufficient_detail)

  .emsc4_compile()

  # ---- Validate inputs (verbatim from the reference implementation) ----
  if (!is.data.frame(data)) data <- as.data.frame(data)
  X <- as.matrix(data)
  if (anyNA(X)) {
    stop("enumerate_msc requires complete cases (no NAs). Subset before calling.")
  }
  rng <- range(X)
  if (rng[1L] < 0 || rng[2L] > 1) {
    stop("enumerate_msc requires 0/1 indicator columns.")
  }
  N <- nrow(X)
  V <- ncol(X)
  vocab <- colnames(data)
  if (is.null(vocab)) vocab <- paste0("V", seq_len(V))

  if (V > 64L) {
    stop("enumerate_msc_v4 supports up to 64 vocabulary elements ",
         "(single-word vocab mask). Got ", V, ".")
  }

  if (!is.list(targets) && !is.data.frame(targets)) {
    stop("'targets' must be a named list or data.frame of 0/1 columns.")
  }
  if (is.data.frame(targets)) targets <- as.list(targets)
  target_names <- names(targets)
  if (is.null(target_names) || any(nchar(target_names) == 0L)) {
    stop("'targets' must be named.")
  }
  for (tn in target_names) {
    if (length(targets[[tn]]) != N) {
      stop("target '", tn, "' has length ", length(targets[[tn]]),
           "; expected ", N, ".")
    }
    if (anyNA(targets[[tn]])) {
      stop("target '", tn, "' contains NAs.")
    }
    rt <- range(targets[[tn]])
    if (rt[1L] < 0 || rt[2L] > 1) {
      stop("target '", tn, "' has values outside [0,1].")
    }
  }

  if (length(thetas) == 0L || any(thetas <= 0 | thetas > 1)) {
    stop("'thetas' must be non-empty and in (0, 1].")
  }
  thetas <- sort(unique(as.numeric(thetas)))

  criterion <- as.character(criterion)
  criterion <- unique(criterion)
  valid_crit <- c("wilson", "point")
  bad_crit <- setdiff(criterion, valid_crit)
  if (length(bad_crit) > 0L) {
    stop("Unknown criterion(s): ", paste(bad_crit, collapse = ", "),
         ". Must be one or both of 'wilson', 'point'.")
  }
  if (length(criterion) == 0L) {
    stop("'criterion' must specify at least one of 'wilson', 'point'.")
  }

  if (!is.numeric(n_min) || length(n_min) != 1L || n_min < 1) {
    stop("'n_min' must be a single integer >= 1.")
  }
  n_min <- as.integer(n_min)
  if (!is.numeric(max_cardinality) || length(max_cardinality) != 1L ||
      max_cardinality < 1) {
    stop("'max_cardinality' must be a single integer >= 1.")
  }
  max_cardinality <- as.integer(max_cardinality)

  if (verbose) {
    base::message("enumerate_msc: V = ", V, ", N = ", N,
                  ", n_min = ", n_min, ", alpha = ", alpha,
                  ", max_cardinality = ", max_cardinality,
                  ", criterion = ", paste(criterion, collapse = "+"),
                  ", thetas = ", paste(format(thetas), collapse = ", "),
                  ", targets = ", paste(target_names, collapse = ", "))
  }

  # ---- Per-target predictor exclusions (verbatim validation) ----
  target_exclude_idx <- vector("list", length(target_names))
  names(target_exclude_idx) <- target_names
  if (!is.null(target_exclude)) {
    if (!is.list(target_exclude) || is.null(names(target_exclude))) {
      stop("'target_exclude' must be a named list mapping a target name to a ",
           "character vector of vocabulary elements to forbid in its configs.")
    }
    bad_tn <- setdiff(names(target_exclude), target_names)
    if (length(bad_tn) > 0L) {
      stop("'target_exclude' names not among targets: ",
           paste(bad_tn, collapse = ", "))
    }
    for (tn in names(target_exclude)) {
      ex <- target_exclude[[tn]]
      if (length(ex) == 0L) next
      mi <- match(ex, vocab)
      if (anyNA(mi)) {
        stop("target_exclude[['", tn, "']] contains items not in the ",
             "vocabulary: ", paste(ex[is.na(mi)], collapse = ", "))
      }
      target_exclude_idx[[tn]] <- as.integer(mi)
      if (verbose) {
        base::message("  target '", tn, "': excluding from its own predictor ",
                      "pool: ", paste(ex, collapse = ", "))
      }
    }
  }

  storage.mode(X) <- "integer"
  targets_int <- lapply(targets, function(t) as.integer(t))

  # ---- C++ core, phase 1: ascent (row bitsets freed on return) ----
  core <- .emsc4_env$emsc4_ascend(
    X, targets_int, thetas, n_min, alpha,
    max_cardinality, max_admissible_per_depth,
    target_exclude_idx, verbose
  )

  stop_reason <- NA_character_
  if (isTRUE(core$stopped_early)) {
    stop_reason <- paste0("max_admissible_per_depth (",
                          format(max_admissible_per_depth, big.mark = ","),
                          ") exceeded at depth ", core$stop_depth)
    warning(stop_reason, "; returning partial results.")
  }

  admissibility_count <- as.integer(core$admissibility)
  evaluability_count <- core$evaluability
  dimnames(evaluability_count) <- list(theta = format(thetas),
                                       depth = seq_len(max_cardinality))

  # ---- Assemble per-cell data frames exactly as the reference does ----
  .empty_msc_df <- function() {
    data.frame(
      target       = character(0),
      theta        = numeric(0),
      criterion    = character(0),
      cardinality  = integer(0),
      set          = character(0),
      set_idx      = I(list()),
      n_set        = integer(0),
      n_set_y1     = integer(0),
      cprob        = numeric(0),
      wilson_lower = numeric(0),
      minimal      = logical(0),
      stringsAsFactors = FALSE
    )
  }

  msc_results  <- vector("list", length(criterion))
  suff_results <- vector("list", length(criterion))
  names(msc_results)  <- criterion
  names(suff_results) <- criterion
  for (cr in criterion) { msc_results[[cr]] <- list(); suff_results[[cr]] <- list() }

  n_sufficient_counts <- list()
  detail_flag <- if (sufficient_detail == "full") 0L else 1L

  n_thetas <- length(thetas)
  for (k in seq_along(target_names)) {
    tn <- target_names[k]
    for (ti in seq_len(n_thetas)) {
      theta <- thetas[ti]
      key <- paste0(tn, "__theta", format(theta))

      for (cr in criterion) {
        cell <- .emsc4_env$emsc4_cell(core$store, k - 1L, theta,
                                      if (cr == "wilson") 0L else 1L,
                                      detail_flag)
        n_sufficient_counts[[cr]][[key]] <- cell$n_sufficient
        n_suff <- length(cell$cardinality)
        if (n_suff == 0L) {
          msc_results[[cr]][[key]]  <- .empty_msc_df()
          suff_results[[cr]][[key]] <- .empty_msc_df()
          next
        }
        set_idx <- cell$set_idx
        card    <- as.integer(cell$cardinality)
        # Reference quirk, replicated for identical(): depth-1 vocab_idx
        # entries inherit names from which() on a named support vector, so
        # the set_idx list-column is named iff any cardinality-1 row exists
        # (vocab name on those entries, "" elsewhere).
        if (any(card == 1L)) {
          nm <- character(length(set_idx))
          is1 <- card == 1L
          nm[is1] <- vocab[vapply(set_idx[is1], `[`, integer(1), 1L)]
          names(set_idx) <- nm
        }
        set_names_chr <- vapply(set_idx, function(ix) {
          paste(vocab[ix], collapse = "_")
        }, character(1), USE.NAMES = FALSE)
        full_df <- data.frame(
          target       = tn,
          theta        = theta,
          criterion    = cr,
          cardinality  = as.integer(cell$cardinality),
          set          = set_names_chr,
          set_idx      = I(set_idx),
          n_set        = as.integer(cell$n_set),
          n_set_y1     = as.integer(cell$n_set_y1),
          cprob        = cell$cprob,
          wilson_lower = cell$wilson_lower,
          minimal      = as.logical(cell$minimal),
          stringsAsFactors = FALSE
        )
        suff_results[[cr]][[key]] <- full_df
        msc_results[[cr]][[key]]  <- full_df[full_df$minimal, , drop = FALSE]
      }

      if (verbose) {
        sizes <- vapply(criterion, function(cr) {
          nr <- nrow(msc_results[[cr]][[key]])
          if (is.null(nr)) 0L else nr
        }, integer(1))
        sizes_str <- paste(paste0(criterion, "=", sizes), collapse = ", ")
        base::message("  target = ", tn, ", theta = ", format(theta),
                      ": MSC sizes [", sizes_str, "]")
      }
    }
  }

  list(
    minimally_sufficient = msc_results,
    sufficient           = suff_results,
    admissibility        = admissibility_count,
    evaluability         = evaluability_count,
    config = list(
      N                = N,
      V                = V,
      vocab            = vocab,
      target_names     = target_names,
      thetas           = thetas,
      n_min            = n_min,
      alpha            = alpha,
      criterion        = criterion,
      max_cardinality  = max_cardinality,
      max_admissible_per_depth = max_admissible_per_depth
    ),
    stopped_early = isTRUE(core$stopped_early),
    stop_reason   = stop_reason,
    # Additive field, not in the reference: exact |S_theta| per cell, valid
    # in both detail modes. Under sufficient_detail = "count", $sufficient
    # data frames contain only the antichain rows (minimal == TRUE), so use
    # this field -- not nrow($sufficient) -- for the Table-1 headline count.
    n_sufficient  = n_sufficient_counts
  )
}
