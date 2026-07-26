// pairmi_v5.cpp  (v2 semantics on the v4 packed-bitset core)
//
// C++ core for pairmi v4. Replaces the R-level per-candidate loop of v3
// with an index-based packed-bitset engine:
//
//   * Indicators packed into uint64 words; conjunction support computed by
//     fused AND + popcount (cost = ceil(N/64) word ops per candidate).
//   * Retained sets referenced by integer index. No name lookups anywhere
//     in the hot path (v3's retained_bits[[name]] was an O(store) linear
//     scan per candidate -- the cost that decoupled wall time from
//     candidate count at high depth).
//   * The accumulated store lives in C++ vectors behind an XPtr, so R's
//     GC never traverses it.
//   * Candidate generation replicates v3's order exactly:
//       depth 2:  combn order (i < j by column position)
//       depth k:  expand.grid(orig, prev) order = outer loop over
//                 previous-depth retained sets in retention order,
//                 inner loop over original variables in column order;
//                 overlap-filtered; canonical-set dedup keeps the FIRST
//                 occurrence (inserted into the seen-set whether or not
//                 the candidate is subsequently retained -- matching
//                 v3's duplicated() on the full candidate vector).
//   * MI (nats), G with flipped-count correction, and p via R::pchisq
//     use the same expressions, operation order, and C library routines
//     as v3, so results are bit-for-bit identical on complete data.
//
// Scope: V <= 64 original variables (vocab masks are a single uint64).
// Extendable to multi-word masks if a larger vocabulary is ever needed.

#include <Rcpp.h>
#include <cstdint>
#include <vector>
#include <unordered_set>
using namespace Rcpp;

struct PMState {
  int N = 0;          // rows
  int V = 0;          // original variables
  int W = 0;          // uint64 words per row-indicator

  // Original variables: packed row bits + popcounts
  std::vector<uint64_t> orig_bits;   // V * W words
  std::vector<int>      orig_pop;

  // Retained-set store (append-only, retention order)
  std::vector<uint64_t> set_bits;    // nsets * W words (row indicators)
  std::vector<uint64_t> set_vmask;   // vocab bitmask (V <= 64)
  std::vector<int>      set_pop;     // popcount of row indicator
  std::vector<int>      set_depth;   // cardinality
  std::vector<int>      set_var;     // x1: original variable index (0-based)
  std::vector<int>      set_parent;  // x2: index into store (depth>=3), or
                                     //     original var index for depth 2
  std::vector<double>   set_mi;
  std::vector<double>   set_relmi;
  std::vector<double>   set_p;

  // [start, end) index range of the previous depth's retained sets
  int prev_start = 0, prev_end = 0;

  size_t nsets() const { return set_pop.size(); }
};

static inline int popcnt_and(const uint64_t* a, const uint64_t* b, int W) {
  int c = 0;
  for (int w = 0; w < W; ++w) c += __builtin_popcountll(a[w] & b[w]);
  return c;
}

// Exact replica of v3's safe_term(): zero unless all counts positive.
static inline double safe_term(double n_cell, double m1, double m2, double n) {
  if (n_cell > 0 && m1 > 0 && m2 > 0 && n > 0)
    return (n_cell / n) * std::log((n_cell * n) / (m1 * m2));
  return 0.0;
}

// MI in nats: four terms summed in v3's order: t11 + t10 + t01 + t00.
static inline double mi_2x2(double n11, double nx1, double nx2, double n) {
  double n10 = nx1 - n11;
  double n01 = nx2 - n11;
  double n00 = n - nx1 - nx2 + n11;
  return safe_term(n11, nx1,     nx2,     n)
       + safe_term(n10, nx1,     n - nx2, n)
       + safe_term(n01, n - nx1, nx2,     n)
       + safe_term(n00, n - nx1, n - nx2, n);
}

// [[Rcpp::export]]
SEXP pm5_create(IntegerMatrix X) {
  PMState* st = new PMState();
  st->N = X.nrow();
  st->V = X.ncol();
  if (st->V > 64)
    stop("pairmi v4 core currently supports up to 64 variables (got %d).",
         st->V);
  st->W = (st->N + 63) / 64;

  st->orig_bits.assign((size_t)st->V * st->W, 0ULL);
  st->orig_pop.assign(st->V, 0);
  for (int j = 0; j < st->V; ++j) {
    uint64_t* col = st->orig_bits.data() + (size_t)j * st->W;
    int pc = 0;
    for (int i = 0; i < st->N; ++i) {
      if (X(i, j) == 1) { col[i >> 6] |= (1ULL << (i & 63)); ++pc; }
    }
    st->orig_pop[j] = pc;
  }
  XPtr<PMState> p(st, true);
  return p;
}

// Run one depth of the ascent. Returns integer vector:
//   c(n_candidates_raw, n_candidates_unique, n_retained)
// where "raw" counts candidates surviving the overlap filter (matching
// v3's n_candidates_raw) and "unique" counts first-occurrence canonical
// sets (matching v3's n_candidates). At depth 2 raw == unique.
// [[Rcpp::export]]
IntegerVector pm5_depth(SEXP state, int depth, double alpha,
                        bool use_mi_threshold, double mi_threshold) {
  XPtr<PMState> st(state);
  const int V = st->V, W = st->W;
  const double n = (double) st->N;          // n_complete (complete cases)
  const double half_n = 0.5 * n;
  const uint64_t* OB = st->orig_bits.data();

  long long raw = 0, uniq = 0, kept = 0;
  const size_t store_before = st->nsets();

  std::vector<uint64_t> scratch(W);

  auto evaluate_and_keep =
    [&](const uint64_t* a_bits, int a_pop, int a_var_or_parent, int x1_var,
        double parent_mi, int d, uint64_t new_vmask) {
      // a_bits = x2 row indicator (prev set, or 2nd variable at depth 2)
      // x1_var = original-variable index of x1
      const uint64_t* b_bits = OB + (size_t)x1_var * W;
      int n11i = popcnt_and(a_bits, b_bits, W);

      double n11 = (double) n11i;
      double nx1 = (double) st->orig_pop[x1_var];  // x1 marginal
      double nx2 = (double) a_pop;                 // x2 marginal
      double mi  = mi_2x2(n11, nx1, nx2, n);

      bool keep;
      double pval;
      // G with flipped-count correction (v3): joint_correct then p.
      double jc = (n11 > half_n) ? (n - n11) : n11;
      double g  = 2.0 * jc * mi;
      pval = R::pchisq(g, 1.0, /*lower_tail=*/0, /*log_p=*/0);
      keep = use_mi_threshold ? (mi > mi_threshold) : (pval < alpha);

      if (keep) {
        // Materialize the conjunction row-indicator into the store.
        size_t off = st->set_bits.size();
        st->set_bits.resize(off + W);
        uint64_t* out = st->set_bits.data() + off;
        for (int w = 0; w < W; ++w) out[w] = a_bits[w] & b_bits[w];

        st->set_vmask.push_back(new_vmask);
        st->set_pop.push_back(n11i);
        st->set_depth.push_back(d);
        st->set_var.push_back(x1_var);
        st->set_parent.push_back(a_var_or_parent);
        st->set_mi.push_back(mi);
        st->set_relmi.push_back(d == 2 ? 0.0 : (mi - parent_mi));
        st->set_p.push_back(pval);
        ++kept;
      }
      return keep;
    };

  if (depth == 2) {
    // combn(orig_variables, 2) order: (0,1),(0,2),...,(0,V-1),(1,2),...
    // v3 assigns x1 = first, x2 = second.
    for (int i = 0; i < V; ++i) {
      const uint64_t* a = OB + (size_t)i * W;
      for (int j = i + 1; j < V; ++j) {
        ++raw; ++uniq;
        // NOTE v3 depth-2 orientation: x1 = var i, x2 = var j.
        // evaluate_and_keep takes x2's bits as a_bits and x1 as x1_var,
        // so pass a = bits of j? No: x1 = i, x2 = j =>
        //   a_bits (x2) = bits of j, x1_var = i.
        const uint64_t* bj = OB + (size_t)j * W;
        uint64_t vm = (1ULL << i) | (1ULL << j);
        evaluate_and_keep(bj, st->orig_pop[j], /*parent=*/j, /*x1=*/i,
                          0.0, 2, vm);
      }
    }
  } else {
    if (st->prev_end <= st->prev_start)
      return IntegerVector::create(0, 0, 0);

    // v2 semantics: significance is tested on EVERY overlap-filtered
    // decomposition; a canonical set is retained iff ANY decomposition is
    // significant, and it is recorded from the FIRST SIGNIFICANT one in
    // enumeration order (outer prev, inner orig). This is order-invariant in
    // membership (unlike v4's first-reached rule) and reproduces v2's
    // duplicated()-after-significance-filter exactly.
    //   seen_all  : distinct canonical sets encountered (for the uniq count)
    //   seen_kept : canonical sets already materialized (first significant)
    std::unordered_set<uint64_t> seen_all, seen_kept;
    seen_all.reserve((size_t)(st->prev_end - st->prev_start) * V);
    seen_kept.reserve((size_t)(st->prev_end - st->prev_start) * V);

    for (int p = st->prev_start; p < st->prev_end; ++p) {
      const uint64_t  pv    = st->set_vmask[p];
      const int       ppop  = st->set_pop[p];
      const double    pmi   = st->set_mi[p];
      // Copy the parent's row bits into stable scratch: evaluate_and_keep
      // may resize set_bits (reallocating the buffer), which would leave a
      // raw pointer into set_bits dangling for the rest of this inner loop.
      std::copy(st->set_bits.begin() + (size_t)p * W,
                st->set_bits.begin() + (size_t)(p + 1) * W,
                scratch.begin());
      const uint64_t* pbits = scratch.data();

      for (int j = 0; j < V; ++j) {
        if (pv & (1ULL << j)) continue;         // overlap filter
        ++raw;
        uint64_t vm = pv | (1ULL << j);
        if (seen_all.insert(vm).second) ++uniq; // count distinct canonical sets
        if (seen_kept.count(vm)) continue;      // already have first significant
        bool k = evaluate_and_keep(pbits, ppop, /*parent=*/p, /*x1=*/j, pmi,
                                   depth, vm);
        if (k) seen_kept.insert(vm);            // lock in first significant
      }
      if ((p - st->prev_start) % 4096 == 0) Rcpp::checkUserInterrupt();
    }
  }

  st->prev_start = (int) store_before;
  st->prev_end   = (int) st->nsets();

  return IntegerVector::create((int)raw, (int)uniq, (int)kept);
}

// Extract results in retention order.
// [[Rcpp::export]]
List pm5_results(SEXP state) {
  XPtr<PMState> st(state);
  const size_t S = st->nsets();

  IntegerVector depth(S), var(S), parent(S), pop(S);
  NumericVector mi(S), relmi(S), pval(S);
  List members(S);

  for (size_t s = 0; s < S; ++s) {
    depth[s]  = st->set_depth[s];
    var[s]    = st->set_var[s] + 1;                     // 1-based
    parent[s] = st->set_parent[s] + 1;                  // 1-based
    pop[s]    = st->set_pop[s];
    mi[s]     = st->set_mi[s];
    relmi[s]  = st->set_relmi[s];
    pval[s]   = st->set_p[s];

    // Reconstruct member indices from the vocab mask (ascending order).
    uint64_t vm = st->set_vmask[s];
    IntegerVector m(__builtin_popcountll(vm));
    int k = 0;
    for (int j = 0; j < st->V; ++j)
      if (vm & (1ULL << j)) m[k++] = j + 1;             // 1-based
    members[s] = m;
  }

  return List::create(_["depth"] = depth, _["var"] = var,
                      _["parent"] = parent, _["pop"] = pop,
                      _["mi"] = mi, _["relative_mi"] = relmi,
                      _["p"] = pval, _["members"] = members);
}

// Unpack retained-set row indicators to an N x nsets integer matrix.
// [[Rcpp::export]]
IntegerMatrix pm5_indicators(SEXP state) {
  XPtr<PMState> st(state);
  const size_t S = st->nsets();
  IntegerMatrix out(st->N, (int)S);
  for (size_t s = 0; s < S; ++s) {
    const uint64_t* bits = st->set_bits.data() + s * st->W;
    for (int i = 0; i < st->N; ++i)
      out(i, (int)s) = (int)((bits[i >> 6] >> (i & 63)) & 1ULL);
  }
  return out;
}

// Store size diagnostics: c(nsets, bytes_row_bits)
// [[Rcpp::export]]
NumericVector pm5_store_info(SEXP state) {
  XPtr<PMState> st(state);
  return NumericVector::create(
    (double) st->nsets(),
    (double) st->set_bits.size() * 8.0);
}
