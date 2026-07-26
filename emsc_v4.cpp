// emsc_v4.cpp
//
// C++ core for enumerate_msc v4: Apriori ascent over the admissible
// sub-lattice with packed-bitset row indicators, streaming per-target
// y1 counts, and in-core antichain Reduce.
//
// Exact-reproduction contract with enumerate_msc_saveall.R:
//   - identical extension order (ordered extension over the depth-1
//     admissible indices, ascending)
//   - identical Wilson arithmetic (same expression order, R::qnorm z)
//   - identical sufficiency gates for both criteria, including the
//     point criterion's DBL_EPSILON tolerance
//   - identical row order in each cell's sufficient set (depth-major,
//     within-depth storage order), which the R reference achieves via a
//     stable sort on cardinality of an already-sorted accumulation
//   - identical greedy antichain Reduce semantics
//
// Memory design: row bitsets exist only during the ascent (current and
// previous depth), and are freed before cell evaluation. Each finalized
// configuration keeps a uint64 vocabulary mask, its support, and its
// per-target y1 counts (~40-60 bytes/config regardless of N), held in a
// C++ store behind an XPtr so the R GC never traverses it. Cells are
// evaluated one at a time on demand, so peak memory is the store plus
// a single cell's output.
//
// Scope: V <= 64 vocabulary elements (single-word vocab mask).

#include <Rcpp.h>
#include <cstdint>
#include <vector>
#include <string>
using namespace Rcpp;

static inline int popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(x);
#else
  int c = 0; while (x) { x &= x - 1; ++c; } return c;
#endif
}

static void pack_column(const int* col, int n, uint64_t* out, int nwords) {
  for (int w = 0; w < nwords; ++w) out[w] = 0ULL;
  for (int i = 0; i < n; ++i) {
    if (col[i]) out[i >> 6] |= (1ULL << (i & 63));
  }
}

static inline int and_popcount(const uint64_t* a, const uint64_t* b,
                               int nwords) {
  int s = 0;
  for (int w = 0; w < nwords; ++w) s += popcount64(a[w] & b[w]);
  return s;
}

// Wilson score lower bound, replicating .wilson_lower_bound() term order.
static inline double wilson_lower(double p, double nn, double z) {
  double denom  = 1.0 + z * z / nn;
  double center = p + z * z / (2.0 * nn);
  double margin = z * std::sqrt(p * (1.0 - p) / nn + z * z / (4.0 * nn * nn));
  return (center - margin) / denom;
}

struct EmscStore {
  int n_targets = 0;
  double z = 0.0, z2 = 0.0;
  std::vector<uint64_t> cfg_mask;
  std::vector<int>      cfg_supp;
  std::vector<int>      cfg_y1;      // n_targets per config, contiguous
  std::vector<int>      depth_end;   // exclusive offsets, depth-major
  std::vector<uint64_t> excl_mask;   // per target
};

// [[Rcpp::export]]
List emsc4_ascend(IntegerMatrix X,
                  List targets,
                  NumericVector thetas,
                  int n_min,
                  double alpha,
                  int max_cardinality,
                  double max_admissible_per_depth,
                  List target_exclude_idx,
                  bool verbose) {

  const int N = X.nrow();
  const int V = X.ncol();
  const int nwords = (N + 63) / 64;
  const int n_targets = targets.size();
  const int n_thetas = thetas.size();

  XPtr<EmscStore> store(new EmscStore(), true);
  store->n_targets = n_targets;
  store->z  = R::qnorm(1.0 - alpha / 2.0, 0.0, 1.0, 1, 0);
  store->z2 = store->z * store->z;
  const double z2 = store->z2;

  std::vector<uint64_t> ind_bits((size_t)V * nwords);
  for (int j = 0; j < V; ++j) {
    std::vector<int> col(N);
    for (int i = 0; i < N; ++i) col[i] = (X(i, j) == 1);
    pack_column(col.data(), N, &ind_bits[(size_t)j * nwords], nwords);
  }
  std::vector<uint64_t> tgt_bits((size_t)n_targets * nwords);
  for (int k = 0; k < n_targets; ++k) {
    IntegerVector tv = targets[k];
    std::vector<int> col(N);
    for (int i = 0; i < N; ++i) col[i] = (tv[i] == 1);
    pack_column(col.data(), N, &tgt_bits[(size_t)k * nwords], nwords);
  }

  store->excl_mask.assign(n_targets, 0ULL);
  for (int k = 0; k < n_targets; ++k) {
    RObject o = target_exclude_idx[k];
    if (!o.isNULL()) {
      IntegerVector idx(o);
      uint64_t m = 0ULL;
      for (int q = 0; q < idx.size(); ++q) m |= (1ULL << (idx[q] - 1));
      store->excl_mask[k] = m;
    }
  }

  std::vector<uint64_t>& cfg_mask = store->cfg_mask;
  std::vector<int>&      cfg_supp = store->cfg_supp;
  std::vector<int>&      cfg_y1   = store->cfg_y1;
  std::vector<int>&      depth_end = store->depth_end;
  cfg_mask.reserve(1 << 20);
  cfg_supp.reserve(1 << 20);
  cfg_y1.reserve((size_t)(1 << 20) * n_targets);

  IntegerVector admissibility(max_cardinality);
  IntegerMatrix evaluability(n_thetas, max_cardinality);

  bool stopped_early = false;
  int stop_depth = NA_INTEGER;

  // ---- Depth 1 ----
  std::vector<int> d1_admis;
  for (int j = 0; j < V; ++j) {
    int s = 0;
    const uint64_t* b = &ind_bits[(size_t)j * nwords];
    for (int w = 0; w < nwords; ++w) s += popcount64(b[w]);
    if (s >= n_min) d1_admis.push_back(j);
  }
  admissibility[0] = (int)d1_admis.size();
  for (int q = 0; q < (int)d1_admis.size(); ++q) {
    int j = d1_admis[q];
    const uint64_t* b = &ind_bits[(size_t)j * nwords];
    int s = 0;
    for (int w = 0; w < nwords; ++w) s += popcount64(b[w]);
    cfg_mask.push_back(1ULL << j);
    cfg_supp.push_back(s);
    for (int k = 0; k < n_targets; ++k) {
      cfg_y1.push_back(and_popcount(b, &tgt_bits[(size_t)k * nwords], nwords));
    }
  }
  depth_end.push_back((int)cfg_mask.size());
  for (int ti = 0; ti < n_thetas; ++ti) {
    int c = 0;
    for (int q = 0; q < (int)d1_admis.size(); ++q) {
      double n = (double)cfg_supp[q];
      if (n / (n + z2) >= thetas[ti]) ++c;
    }
    evaluability(ti, 0) = c;
  }
  if (verbose) Rcpp::Rcout << "  depth 1: " << d1_admis.size() << " / " << V
                           << " indicators admissible (#(c) >= " << n_min
                           << ")\n";

  // ---- Depths 2..max_cardinality ----
  std::vector<uint64_t> prev_bits((size_t)d1_admis.size() * nwords);
  for (int q = 0; q < (int)d1_admis.size(); ++q) {
    const uint64_t* b = &ind_bits[(size_t)d1_admis[q] * nwords];
    std::copy(b, b + nwords, &prev_bits[(size_t)q * nwords]);
  }

  if (!d1_admis.empty() && max_cardinality >= 2) {
    for (int d = 2; d <= max_cardinality; ++d) {
      int lo = (d == 2) ? 0 : depth_end[d - 3];
      int hi = depth_end[d - 2];
      int n_prev = hi - lo;
      if (n_prev == 0) break;

      // Row bits of the final depth are never extended, so don't store them.
      const bool keep_bits = (d < max_cardinality);
      std::vector<uint64_t> cur_bits;
      if (keep_bits) cur_bits.reserve((size_t)n_prev * nwords);
      long long n_cands = 0;
      int n_admis = 0;
      std::vector<uint64_t> scratch(nwords);

      for (int i = 0; i < n_prev; ++i) {
        int gid = lo + i;
        uint64_t mask_i = cfg_mask[gid];
        int max_j = 63 - __builtin_clzll(mask_i);
        const uint64_t* bits_i = &prev_bits[(size_t)i * nwords];

        for (int qi = 0; qi < (int)d1_admis.size(); ++qi) {
          int j = d1_admis[qi];
          if (j <= max_j) continue;
          ++n_cands;
          const uint64_t* bj = &ind_bits[(size_t)j * nwords];
          int supp = 0;
          for (int w = 0; w < nwords; ++w) {
            scratch[w] = bits_i[w] & bj[w];
            supp += popcount64(scratch[w]);
          }
          if (supp < n_min) continue;
          ++n_admis;
          cfg_mask.push_back(mask_i | (1ULL << j));
          cfg_supp.push_back(supp);
          for (int k = 0; k < n_targets; ++k) {
            cfg_y1.push_back(and_popcount(scratch.data(),
                                          &tgt_bits[(size_t)k * nwords],
                                          nwords));
          }
          if (keep_bits) {
            size_t off = cur_bits.size();
            cur_bits.resize(off + nwords);
            std::copy(scratch.begin(), scratch.end(), cur_bits.begin() + off);
          }
        }
        if ((double)n_admis > max_admissible_per_depth) {
          stopped_early = true;
          stop_depth = d;
          break;
        }
        if ((i & 1023) == 0) Rcpp::checkUserInterrupt();
      }

      if (stopped_early) {
        cfg_mask.resize(depth_end.back());
        cfg_supp.resize(depth_end.back());
        cfg_y1.resize((size_t)depth_end.back() * n_targets);
        break;
      }

      admissibility[d - 1] = n_admis;
      if (n_admis == 0) {
        if (verbose) Rcpp::Rcout << "  depth " << d << ": " << n_cands
                                 << " candidates -> 0 admissible; ascent terminates.\n";
        depth_end.push_back((int)cfg_mask.size());
        break;
      }
      depth_end.push_back((int)cfg_mask.size());

      int base = depth_end[d - 2];
      for (int ti = 0; ti < n_thetas; ++ti) {
        int c = 0;
        for (int g = base; g < depth_end[d - 1]; ++g) {
          double n = (double)cfg_supp[g];
          if (n / (n + z2) >= thetas[ti]) ++c;
        }
        evaluability(ti, d - 1) = c;
      }

      if (verbose) Rcpp::Rcout << "  depth " << d << ": " << n_cands
                               << " candidates -> " << n_admis
                               << " admissible\n";
      prev_bits.swap(cur_bits);
    }
  }
  // Row bitsets are no longer needed: y1 counts were streamed per target.
  // (prev_bits/ind_bits/tgt_bits free on scope exit.)

  return List::create(
    _["store"]         = store,
    _["admissibility"] = admissibility,
    _["evaluability"]  = evaluability,
    _["stopped_early"] = stopped_early,
    _["stop_depth"]    = stop_depth,
    _["n_configs"]     = (int)cfg_mask.size()
  );
}

// Evaluate one (target, theta, criterion) cell from the store.
// crit: 0 = wilson, 1 = point. detail: 0 = full rows, 1 = counts only
// (still returns full rows for antichain members plus n_sufficient).
// [[Rcpp::export]]
List emsc4_cell(SEXP store_xp, int target_k, double theta, int crit,
                int detail) {
  XPtr<EmscStore> store(store_xp);
  const int n_targets = store->n_targets;
  const double z = store->z, z2 = store->z2;
  const int n_cfg = (int)store->cfg_mask.size();
  const uint64_t excl = store->excl_mask[target_k];

  std::vector<int> suff_gid;
  std::vector<double> suff_wlb;
  std::vector<char> suff_min;
  std::vector<uint64_t> acc_masks;
  std::vector<int> acc_cards;
  long long n_suff_total = 0;

  for (int g = 0; g < n_cfg; ++g) {
    uint64_t m = store->cfg_mask[g];
    if (excl && (m & excl)) continue;
    double nn = (double)store->cfg_supp[g];
    int y1 = store->cfg_y1[(size_t)g * n_targets + target_k];
    double w;
    if (crit == 0) {
      if (!(nn / (nn + z2) >= theta)) continue;
      double p = (double)y1 / nn;
      w = wilson_lower(p, nn, z);
      if (!(w >= theta)) continue;
    } else {
      if (!(((double)y1 +
             std::numeric_limits<double>::epsilon() * nn) >=
            theta * nn)) continue;
      w = wilson_lower((double)y1 / nn, nn, z);
    }
    ++n_suff_total;

    // Greedy antichain membership (stream is cardinality-ascending).
    int card = popcount64(m);
    bool is_min = true;
    for (size_t a = 0; a < acc_masks.size(); ++a) {
      if (acc_cards[a] > card) continue;
      if ((acc_masks[a] & m) == acc_masks[a]) { is_min = false; break; }
    }
    if (is_min) {
      acc_masks.push_back(m);
      acc_cards.push_back(card);
    }
    if (detail == 0 || is_min) {
      suff_gid.push_back(g);
      suff_wlb.push_back(w);
      suff_min.push_back(is_min ? 1 : 0);
    }
  }

  const int n_out = (int)suff_gid.size();
  IntegerVector o_card(n_out), o_nset(n_out), o_y1(n_out);
  NumericVector o_cprob(n_out), o_wlb(n_out);
  LogicalVector o_min(n_out);
  List o_idx(n_out);
  for (int s = 0; s < n_out; ++s) {
    int g = suff_gid[s];
    uint64_t m = store->cfg_mask[g];
    int card = popcount64(m);
    o_card[s] = card;
    o_nset[s] = store->cfg_supp[g];
    int y1 = store->cfg_y1[(size_t)g * n_targets + target_k];
    o_y1[s] = y1;
    o_cprob[s] = (double)y1 / (double)store->cfg_supp[g];
    o_wlb[s] = suff_wlb[s];
    o_min[s] = (suff_min[s] != 0);
    IntegerVector iv(card);
    int pos = 0;
    uint64_t mm = m;
    while (mm) {
      int b = __builtin_ctzll(mm);
      iv[pos++] = b + 1;
      mm &= mm - 1;
    }
    o_idx[s] = iv;
  }

  return List::create(
    _["cardinality"]  = o_card,
    _["set_idx"]      = o_idx,
    _["n_set"]        = o_nset,
    _["n_set_y1"]     = o_y1,
    _["cprob"]        = o_cprob,
    _["wilson_lower"] = o_wlb,
    _["minimal"]      = o_min,
    _["n_sufficient"] = (double)n_suff_total
  );
}

// [[Rcpp::export]]
double emsc4_store_bytes(SEXP store_xp) {
  XPtr<EmscStore> store(store_xp);
  return (double)(store->cfg_mask.size() * 8 +
                  store->cfg_supp.size() * 4 +
                  store->cfg_y1.size() * 4);
}
