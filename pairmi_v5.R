#' ============================================================================
#' pairmi v5 -- CANONICAL. Supersedes v2, v3, and v4.
#'
#' v5 = v2's SEMANTICS on v4's packed-bitset ENGINE. It is the recommended
#' pairmi for all future work. Rationale (validated 2026-07; see CHANGELOG):
#'
#'   * v2 (original): correct semantics -- a canonical set is retained iff ANY
#'     of its decompositions is mutually-informative-significant -- but slow
#'     (R-level per-candidate loop).
#'   * v3/v4: fast (C++ core) but changed the dedup rule to "test only the
#'     FIRST decomposition reached." That makes retained membership depend on
#'     input COLUMN ORDER (permute the columns -> different sets) and drops
#'     genuinely sufficient conjunctions. An order-dependent result is a
#'     reproducibility hazard; v4 is therefore NOT recommended.
#'   * v5: evaluates ALL decompositions and keeps the first SIGNIFICANT one
#'     (v2's rule) on v4's core. Byte-for-byte identical to v2 on both study
#'     datasets (pipeline pool AND raw $sets); order-invariant (v4 is not);
#'     ~8x faster than v2 on the densest cell (0.27s vs 2.15s), matching v4.
#'
#' Drop-in: identical public signature and outputs to v2 on complete-case 0/1
#' data. Names every set alphabetically at all depths (v2 convention).
#' Scope: up to 64 variables. Requires Rcpp + the pairmi_v5.cpp core.
#' ============================================================================

.pm5_env <- new.env(parent = baseenv())

.pm5_compile <- function() {
  if (isTRUE(.pm5_env$compiled)) return(invisible(TRUE))
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("pairmi v4 requires the 'Rcpp' package.")
  }
  src <- file.path(.pm5_env$src_dir %||% getwd(), "pairmi_v5.cpp")
  if (!file.exists(src)) stop("pairmi_v5.cpp not found at: ", src)
  Rcpp::sourceCpp(src, env = .pm5_env)
  .pm5_env$compiled <- TRUE
  invisible(TRUE)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

pairmi <- function(data, alpha = 0.05, MI.threshold = NULL,
                   n_elements = 5, sep = "_", verbose = TRUE,
                   expand = TRUE) {

  .pm5_compile()

  if (!is.data.frame(data)) data <- as.data.frame(data)
  stopifnot(n_elements >= 2)

  X_full <- as.matrix(data)
  if (anyNA(X_full)) {
    stop("pairmi() requires complete cases (no NAs). ",
         "Subset to complete.cases(data) before calling.")
  }
  rng <- range(X_full)
  if (rng[1L] < 0 || rng[2L] > 1) {
    stop("pairmi() requires 0/1 columns. Found values outside [0,1].")
  }

  orig_variables <- base::colnames(data)

  # Guard: v2/v3 track set membership by splitting name strings on `sep`.
  # If any variable name itself contains `sep`, that bookkeeping corrupts
  # silently: the overlap filter cannot recognize the variable inside a
  # set name, so sets get re-extended by their own members, producing
  # phantom duplicate-member "sets" at inflated cardinality whose
  # near-tautological MI (A vs A-and-B) makes them almost always
  # significant. v4 is index-based and immune, but that means byte-for-byte
  # agreement with legacy output is impossible on such names -- so refuse
  # explicitly rather than diverge silently.
  bad_nm <- orig_variables[base::grepl(sep, orig_variables, fixed = TRUE)]
  if (base::length(bad_nm) > 0L) {
    stop("Variable name(s) contain the separator '", sep, "': ",
         base::paste(utils::head(bad_nm, 5L), collapse = ", "),
         if (base::length(bad_nm) > 5L) ", ..." else "",
         ". pairmi v2/v3 silently produce corrupted results (phantom ",
         "duplicate-member sets) in this situation. Choose a `sep` that ",
         "occurs in no variable name, or rename the variables.")
  }

  n_rows <- base::nrow(X_full)
  Xi <- X_full
  storage.mode(Xi) <- "integer"

  st <- .pm5_env$pm5_create(Xi)

  base::message("Pairing Data")
  pb <- utils::txtProgressBar(min = 0, max = (n_elements - 1L), initial = 0)

  n_retained_by_depth <- integer(n_elements)

  for (depth_level in 2:n_elements) {

    depth_t0 <- proc.time()[["elapsed"]]

    if (depth_level > 2L &&
        n_retained_by_depth[depth_level - 1L] == 0L) {
      if (verbose) {
        base::message("  depth ", depth_level,
                      ": no retained sets from depth ", depth_level - 1L,
                      "; stopping.")
      }
      base::message(base::paste(
        "stopped at max number of elements:", (depth_level - 1L)))
      break
    }

    res <- .pm5_env$pm5_depth(st, as.integer(depth_level),
                              as.numeric(alpha),
                              !base::is.null(MI.threshold),
                              if (base::is.null(MI.threshold)) 0.0
                              else as.numeric(MI.threshold))
    n_raw  <- res[1L]; n_uniq <- res[2L]; n_kept <- res[3L]
    n_retained_by_depth[depth_level] <- n_kept

    if (n_uniq == 0L) {
      if (verbose) {
        base::message("  depth ", depth_level,
                      ": 0 candidates after overlap filtering; stopping.")
      }
      base::message(base::paste(
        "stopped at max number of elements:", (depth_level - 1L)))
      break
    }

    if (verbose) {
      base::message("  depth ", depth_level, ": evaluating ",
                    format(n_uniq, big.mark = ","), " unique candidates",
                    if (n_raw != n_uniq)
                      paste0(" (", format(n_raw - n_uniq, big.mark = ","),
                             " duplicates pre-collapsed from ",
                             format(n_raw, big.mark = ","), ")")
                    else "")
    }

    if (n_kept == 0L) {
      if (verbose) {
        depth_t1 <- proc.time()[["elapsed"]]
        base::message("  depth ", depth_level, ": retained 0 / ",
                      format(n_uniq, big.mark = ","),
                      " candidates in ",
                      sprintf("%.1f", depth_t1 - depth_t0), "s")
      }
      base::message(base::paste(
        "stopped at max number of elements:", (depth_level - 1L)))
      break
    }

    if (verbose) {
      depth_t1 <- proc.time()[["elapsed"]]
      base::message("  depth ", depth_level, ": retained ",
                    format(n_kept, big.mark = ","), " / ",
                    format(n_uniq, big.mark = ","),
                    " unique candidates in ",
                    sprintf("%.1f", depth_t1 - depth_t0), "s")
    }

    utils::setTxtProgressBar(pb, (depth_level - 1L))
  }

  base::close(pb)

  # ---- Extract results and rebuild v3's exact output structure ----
  rs <- .pm5_env$pm5_results(st)
  n_sets <- base::length(rs$depth)

  if (n_sets > 0L) {
    # Set names. v2 names EVERY set alphabetically via sort(unique(.)) at all
    # depths (locale-identical). v5 follows v2 here (v4 used column-order names
    # at depth 2; v5 does not), so the raw $sets$set matches v2 byte-for-byte.
    set_names <- base::vapply(rs$members, function(m) {
      base::paste(base::sort(base::unique(orig_variables[m])), collapse = sep)
    }, character(1))
    is_d2 <- rs$depth == 2L

    # relative.mi in v3 carries names at depth >= 3 (from prev_lookup[x2]);
    # replicate so the sets data.frame is attribute-identical.
    relmi <- rs$relative_mi
    nm_vec <- base::character(n_sets)
    if (base::any(!is_d2)) {
      # name = the x2 (parent) set's canonical name
      nm_vec[!is_d2] <- set_names[rs$parent[!is_d2]]
    }
    if (base::any(!is_d2)) base::names(relmi) <- nm_vec

    # Indicator columns, unpacked once at output time (as in v3, this is
    # the only place the dense representation appears). For very large
    # retained-set counts this N x nsets dense matrix dominates memory;
    # expand = FALSE skips it and returns expanded.data = NULL.
    if (expand) {
      set_cols <- .pm5_env$pm5_indicators(st)
      base::colnames(set_cols) <- set_names
      expanded <- base::cbind(data, base::as.data.frame(set_cols))
    } else {
      expanded <- NULL
    }
  } else {
    set_names <- character(0)
    relmi <- numeric(0)
    expanded <- data
  }

  sets_df <- base::data.frame(
    n_elements  = rs$depth,
    set         = set_names,
    mi          = rs$mi,
    relative.mi = relmi,
    p           = rs$p,
    stringsAsFactors = FALSE
  )

  results_list <- base::list()
  results_list$expanded.data      <- expanded
  results_list$original.variables <- orig_variables
  results_list$sets               <- sets_df

  results_list
}
