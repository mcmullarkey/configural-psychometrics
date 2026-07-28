//! Constructive ascent engine for configural psychometrics.
//!
//! Ports `pairmi_v5.cpp`: packed variable indicators, depth-2 combn,
//! depth≥3 expand.grid, canonical-set dedup, first-significant retention,
//! flipped-count G correction.
//!
//! ## What this module provides
//!
//! - [`PairMiEngine`] — constructive ascent engine with append-only store
//! - [`EvaluationMode`] — significance test mode (alpha or MI threshold)
//! - [`PairMiResultRow`] — one retained set's statistics
//!
//! ## What this module does NOT provide
//!
//! - File I/O or serialization
//! - R/C++ bindings
//! - Antichain reduction or post-processing
//!
//! ## Orientation
//!
//! [`BinaryMatrix`] stores observation indicators column-major (each column
//! = one observation, packed into `ceil(V/64)` words). The engine transposes
//! to variable indicators: each variable `j` has its indicator packed across
//! `N` observations into `ceil(N/64)` words. This matches `pairmi_v5.cpp`
//! where `orig_bits[j * W .. (j+1) * W]` is variable `j`'s row indicator.
//!
//! ## Candidate generation order
//!
//! - **Depth 2**: combn order `(i, j)` for `i < j`. `x1 = i` (first),
//!   `x2 = j` (second). `var = i`, `parent = j` (0-based internally,
//!   1-based in [`PairMiResultRow`]).
//! - **Depth ≥3**: outer loop over previous-depth retained sets (retention
//!   order), inner loop over original variables (column order). Overlap
//!   filter skips variables already in the parent's vocab mask. Canonical-set
//!   dedup via `HashSet<u64>` keeps the **first significant** decomposition.

use crate::bitset::{and_popcount, extract_members};
use crate::matrix::BinaryMatrix;
use crate::stats::{chi2_sf_df1, mi_2x2};
use crate::ConfiguralError;
use std::collections::HashSet;

/// Significance test mode for retaining sets.
///
/// - [`Alpha`](EvaluationMode::Alpha): retain if p-value < alpha (G-test).
/// - [`MiThreshold`](EvaluationMode::MiThreshold): retain if MI > threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationMode {
    /// Retain if p-value < alpha (G-test significance).
    Alpha(f64),
    /// Retain if mutual information > threshold (nats).
    MiThreshold(f64),
}

/// One retained set's statistics, in retention order.
///
/// All indices are 1-based to match the R/C++ reference output.
/// `members` is ascending.
#[derive(Debug, Clone, PartialEq)]
pub struct PairMiResultRow {
    /// Cardinality of the set (2, 3, 4, ...).
    pub depth: u32,
    /// `x1`: the original variable added at this depth (1-based).
    pub var: u32,
    /// `parent`: for depth 2, the second variable (1-based); for depth ≥3,
    /// the 1-based index into the store of the parent set.
    pub parent: u32,
    /// Popcount of the conjunction row indicator.
    pub pop: u32,
    /// Mutual information in nats.
    pub mi: f64,
    /// Relative MI: `mi - parent_mi` (0.0 at depth 2).
    pub relmi: f64,
    /// G-test p-value.
    pub p: f64,
    /// Member variable indices (1-based, ascending).
    pub members: Vec<u32>,
}

/// Depth-level statistics returned by [`PairMiEngine::depth`].
///
/// - `raw` — candidates surviving the overlap filter.
/// - `unique` — distinct canonical sets (first-occurrence dedup).
/// - `kept` — sets actually retained (first-significant).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStats {
    /// Candidates surviving the overlap filter.
    pub raw: u64,
    /// Distinct canonical sets encountered.
    pub unique: u64,
    /// Sets retained (first-significant).
    pub kept: u64,
}

/// Constructive ascent engine.
///
/// Ports `pairmi_v5.cpp`. Stores original variable indicators as packed
/// `u64` words and retained sets in an append-only store. Vocabulary is
/// limited to V ≤ 64 (single-word vocab masks).
///
/// # Construction
///
/// [`PairMiEngine::create`] takes a [`BinaryMatrix`] (V = `n_rows`,
/// N = `n_cols`) and transposes to variable indicators. V > 64 is
/// refused (defensive — [`BinaryMatrix`] already enforces this).
///
/// # Usage
///
/// ```
/// use configural::{BinaryMatrix, EvaluationMode, PairMiEngine};
///
/// let data = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]; // 2 vars × 3 obs
/// let matrix = BinaryMatrix::new(&data, 2, 3).unwrap();
/// let mut engine = PairMiEngine::create(&matrix).unwrap();
/// let stats = engine.depth(2, EvaluationMode::Alpha(0.05));
/// let rows = engine.results();
/// ```
pub struct PairMiEngine {
    // Original variables: packed indicators + popcounts.
    /// `V * W` words. Variable `j` occupies words `[j*W, (j+1)*W)`.
    orig_bits: Vec<u64>,
    /// Popcount of each variable's indicator.
    orig_pop: Vec<u32>,

    // Retained-set store (append-only, retention order).
    /// `nsets * W` words. Set `s` occupies words `[s*W, (s+1)*W)`.
    set_bits: Vec<u64>,
    /// Vocab bitmask (V ≤ 64 → single word).
    set_vmask: Vec<u64>,
    /// Popcount of the conjunction row indicator.
    set_pop: Vec<u32>,
    /// Cardinality (2, 3, 4, ...).
    set_depth: Vec<u32>,
    /// `x1`: original variable index (0-based internally).
    set_var: Vec<u32>,
    /// `parent`: depth-2 = second var index; depth≥3 = store index (0-based).
    set_parent: Vec<u32>,
    /// Mutual information (nats).
    set_mi: Vec<f64>,
    /// Relative MI: `mi - parent_mi` (0.0 at depth 2).
    set_relmi: Vec<f64>,
    /// G-test p-value.
    set_p: Vec<f64>,

    // Previous depth's retained-set index range [prev_start, prev_end).
    prev_start: usize,
    prev_end: usize,

    // Dimensions.
    /// Number of observations (N).
    n: usize,
    /// Number of variables (V).
    v: usize,
    /// Words per variable indicator: `ceil(N / 64)`.
    w: usize,
}

impl PairMiEngine {
    /// Create an engine from a validated [`BinaryMatrix`].
    ///
    /// Transposes the matrix from observation indicators (column-major)
    /// to variable indicators (packed across observations), matching
    /// `pairmi_v5.cpp:pm5_create`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfiguralError::VocabularyTooLarge`] if V > 64.
    /// (Defensive — [`BinaryMatrix`] already enforces V ≤ 64.)
    pub fn create(matrix: &BinaryMatrix) -> Result<Self, ConfiguralError> {
        let v = matrix.n_rows(); // V = variables
        let n = matrix.n_cols(); // N = observations

        // V > 64 refused (defensive — BinaryMatrix already enforces this).
        if v > 64 {
            return Err(ConfiguralError::VocabularyTooLarge { got: v, max: 64 });
        }

        let w = n.div_ceil(64); // words per variable indicator

        // Transpose: BinaryMatrix stores observation indicators (column-major,
        // each column = 1 observation). pairmi needs variable indicators packed
        // across observations. For V ≤ 64, each observation is 1 word, so
        // bit j of observation word c = variable j in observation c.
        let mut orig_bits = vec![0u64; v * w];
        let mut orig_pop = vec![0u32; v];

        for j in 0..v {
            let mut pc = 0u32;
            for i in 0..n {
                if matrix.get(j, i) {
                    orig_bits[j * w + i / 64] |= 1u64 << (i % 64);
                    pc += 1;
                }
            }
            orig_pop[j] = pc;
        }

        Ok(Self {
            orig_bits,
            orig_pop,
            set_bits: Vec::new(),
            set_vmask: Vec::new(),
            set_pop: Vec::new(),
            set_depth: Vec::new(),
            set_var: Vec::new(),
            set_parent: Vec::new(),
            set_mi: Vec::new(),
            set_relmi: Vec::new(),
            set_p: Vec::new(),
            prev_start: 0,
            prev_end: 0,
            n,
            v,
            w,
        })
    }

    /// Run one depth of the ascent.
    ///
    /// - **Depth 2**: combn order `(i, j)` for `i < j`. `x1 = i`, `x2 = j`.
    /// - **Depth ≥3**: outer loop over previous-depth retained sets,
    ///   inner loop over original variables. Overlap-filtered, canonical-set
    ///   dedup, first-significant retention.
    ///
    /// Returns [`DepthStats`] with raw, unique, and kept counts.
    /// If depth ≥3 is called with no previous-depth sets, returns zeros.
    pub fn depth(&mut self, depth: u32, mode: EvaluationMode) -> DepthStats {
        let v = self.v;
        let w = self.w;
        let n_f = self.n as f64;

        let mut raw = 0u64;
        let mut unique = 0u64;
        let mut kept = 0u64;

        let store_before = self.set_bits.len() / w;

        // Take immutable borrows of original-variable data for the duration
        // of the loops. These don't conflict with mutations to set_* fields
        // (disjoint field borrows).
        let orig_bits = &self.orig_bits;
        let orig_pop = &self.orig_pop;

        if depth == 2 {
            // combn order: (i, j) for i < j. x1 = i (first), x2 = j (second).
            for i in 0..v {
                for j in (i + 1)..v {
                    raw += 1;
                    unique += 1;

                    let n11 = and_popcount(
                        &orig_bits[i * w..(i + 1) * w],
                        &orig_bits[j * w..(j + 1) * w],
                    );
                    let mi = mi_2x2(n11 as f64, orig_pop[i] as f64, orig_pop[j] as f64, n_f);
                    let p = g_test_p(n11 as f64, n_f, mi);

                    if should_keep(mode, mi, p) {
                        // Materialize conjunction: a_bits & b_bits.
                        let off = self.set_bits.len();
                        self.set_bits.resize(off + w, 0);
                        for word in 0..w {
                            self.set_bits[off + word] =
                                orig_bits[i * w + word] & orig_bits[j * w + word];
                        }
                        self.set_vmask.push((1u64 << i) | (1u64 << j));
                        self.set_pop.push(n11);
                        self.set_depth.push(2);
                        self.set_var.push(i as u32);
                        self.set_parent.push(j as u32);
                        self.set_mi.push(mi);
                        self.set_relmi.push(0.0);
                        self.set_p.push(p);
                        kept += 1;
                    }
                }
            }
        } else {
            // Depth ≥3: expand.grid (outer prev, inner orig).
            if self.prev_end <= self.prev_start {
                return DepthStats {
                    raw: 0,
                    unique: 0,
                    kept: 0,
                };
            }

            let mut seen_all: HashSet<u64> = HashSet::new();
            let mut seen_kept: HashSet<u64> = HashSet::new();
            let mut scratch = vec![0u64; w];

            for p in self.prev_start..self.prev_end {
                let pv = self.set_vmask[p];
                let ppop = self.set_pop[p];
                let pmi = self.set_mi[p];

                // Copy parent bits to scratch. This prevents a dangling
                // pointer after set_bits realloc (matching pairmi_v5.cpp:211).
                // In Rust the borrow checker would catch a dangling slice,
                // but the copy is still needed because set_bits is mutated
                // inside the inner loop (push/resize invalidate any slice
                // borrowed from it).
                scratch[..w].copy_from_slice(&self.set_bits[p * w..(p + 1) * w]);

                for j in 0..v {
                    if pv & (1u64 << j) != 0 {
                        continue; // overlap filter
                    }
                    raw += 1;
                    let vm = pv | (1u64 << j);
                    if seen_all.insert(vm) {
                        unique += 1;
                    }
                    if seen_kept.contains(&vm) {
                        continue; // already have first significant
                    }

                    let n11 = and_popcount(&scratch, &orig_bits[j * w..(j + 1) * w]);
                    let mi = mi_2x2(n11 as f64, orig_pop[j] as f64, ppop as f64, n_f);
                    let p = g_test_p(n11 as f64, n_f, mi);

                    if should_keep(mode, mi, p) {
                        // Materialize conjunction: scratch & orig_bits[j].
                        let off = self.set_bits.len();
                        self.set_bits.resize(off + w, 0);
                        for word in 0..w {
                            self.set_bits[off + word] = scratch[word] & orig_bits[j * w + word];
                        }
                        self.set_vmask.push(vm);
                        self.set_pop.push(n11);
                        self.set_depth.push(depth);
                        self.set_var.push(j as u32);
                        self.set_parent.push(p as u32);
                        self.set_mi.push(mi);
                        self.set_relmi.push(mi - pmi);
                        self.set_p.push(p);
                        seen_kept.insert(vm);
                        kept += 1;
                    }
                }
            }
        }

        self.prev_start = store_before;
        self.prev_end = self.set_bits.len() / w;

        DepthStats { raw, unique, kept }
    }

    /// Extract all retained sets as rows, in retention order.
    ///
    /// `var` and `parent` are 1-based. `members` is 1-based ascending.
    #[must_use]
    pub fn results(&self) -> Vec<PairMiResultRow> {
        let nsets = self.set_pop.len();

        (0..nsets)
            .map(|s| {
                let vm = self.set_vmask[s];
                let members: Vec<u32> = extract_members(vm).map(|m| m + 1).collect();
                PairMiResultRow {
                    depth: self.set_depth[s],
                    var: self.set_var[s] + 1, // 0-based → 1-based
                    parent: self.set_parent[s] + 1,
                    pop: self.set_pop[s],
                    mi: self.set_mi[s],
                    relmi: self.set_relmi[s],
                    p: self.set_p[s],
                    members,
                }
            })
            .collect()
    }

    /// Unpack retained-set row indicators to observation vectors.
    ///
    /// Returns `nsets` vectors, each of length `N` (observations).
    /// Element `i` is `1` if observation `i` is in the conjunction, `0` otherwise.
    #[must_use]
    pub fn indicators(&self) -> Vec<Vec<u8>> {
        let nsets = self.set_pop.len();
        let n = self.n;
        let w = self.w;

        (0..nsets)
            .map(|s| {
                let off = s * w;
                (0..n)
                    .map(|i| ((self.set_bits[off + i / 64] >> (i % 64)) & 1) as u8)
                    .collect()
            })
            .collect()
    }

    /// Number of retained sets in the store.
    #[must_use]
    pub fn n_sets(&self) -> usize {
        self.set_pop.len()
    }
}

/// Compute the G-test p-value with flipped-count correction.
///
/// `jc = min(n11, n - n11)`, `g = 2 * jc * mi`, `p = chi2_sf_df1(g)`.
///
/// If `g ≤ 0` (e.g. `n11 = 0` or `mi ≤ 0`), returns `1.0` — matching
/// R's `pchisq(negative, 1, lower.tail=FALSE)` which returns 1.0.
/// This avoids the panic in [`chi2_sf_df1`] for negative `g`.
#[inline]
fn g_test_p(n11: f64, n: f64, mi: f64) -> f64 {
    let jc = n11.min(n - n11);
    let g = 2.0 * jc * mi;
    if g <= 0.0 {
        1.0
    } else {
        chi2_sf_df1(g)
    }
}

/// Decide whether to retain a set based on the evaluation mode.
#[inline]
fn should_keep(mode: EvaluationMode, mi: f64, p: f64) -> bool {
    match mode {
        EvaluationMode::Alpha(a) => p < a,
        EvaluationMode::MiThreshold(t) => mi > t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 vars × 10 obs, all identical: [1,1,1,1,1,0,0,0,0,0].
    fn identical_matrix() -> BinaryMatrix {
        let row: [f64; 10] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let data: Vec<f64> = [row, row, row].into_iter().flatten().collect();
        BinaryMatrix::new(&data, 3, 10).unwrap()
    }

    #[test]
    fn create_transposes_to_variable_indicators() {
        // 2 vars × 3 obs. Var 0 = [1,0,1], Var 1 = [0,1,0].
        // BinaryMatrix: n_rows=2, n_cols=3, row-major: [1,0,1, 0,1,0].
        let m = BinaryMatrix::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 2, 3).unwrap();
        let engine = PairMiEngine::create(&m).unwrap();

        // W = ceil(3/64) = 1. Var 0 indicator = 0b101 = 5. Var 1 = 0b010 = 2.
        assert_eq!(engine.w, 1);
        assert_eq!(engine.orig_bits, vec![0b101, 0b010]);
        assert_eq!(engine.orig_pop, vec![2, 1]);
    }

    #[test]
    fn create_multi_word_indicator() {
        // 1 var × 100 obs (all 1). W = ceil(100/64) = 2.
        let data = vec![1.0; 100];
        let m = BinaryMatrix::new(&data, 1, 100).unwrap();
        let engine = PairMiEngine::create(&m).unwrap();

        assert_eq!(engine.w, 2);
        assert_eq!(engine.orig_bits.len(), 2); // 1 var × 2 words
        assert_eq!(engine.orig_bits[0], u64::MAX); // bits 0-63
        assert_eq!(engine.orig_bits[1], (1u64 << 36) - 1); // bits 64-99 (36 bits)
        assert_eq!(engine.orig_pop, vec![100]);
    }

    #[test]
    fn create_v_exceeds_64_defensive() {
        // BinaryMatrix already refuses V > 64, so this path is unreachable
        // from a valid BinaryMatrix. The defensive check exists to satisfy
        // the AC and document the invariant.
        // (Cannot construct a BinaryMatrix with V=65 to test directly.)
    }

    #[test]
    fn depth_2_combn_order() {
        let m = identical_matrix();
        let mut engine = PairMiEngine::create(&m).unwrap();
        let stats = engine.depth(2, EvaluationMode::Alpha(0.05));

        // C(3,2) = 3 pairs.
        assert_eq!(stats.raw, 3);
        assert_eq!(stats.unique, 3); // raw == unique at depth 2
        assert_eq!(stats.kept, 3);
    }

    #[test]
    fn depth_2_mi_equals_ln2() {
        let m = identical_matrix();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        let ln2 = 2.0f64.ln();
        for row in engine.results() {
            assert!(
                (row.mi - ln2).abs() < 1e-10,
                "MI should be ln(2), got {}",
                row.mi
            );
        }
    }

    #[test]
    fn depth_3_first_significant_retention() {
        let m = identical_matrix();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        let stats = engine.depth(3, EvaluationMode::Alpha(0.05));
        // Triple {0,1,2} encountered 3 times, deduped to 1.
        assert_eq!(stats.raw, 3);
        assert_eq!(stats.unique, 1);
        assert_eq!(stats.kept, 1);
    }

    #[test]
    fn depth_3_relmi_is_delta() {
        let m = identical_matrix();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));
        engine.depth(3, EvaluationMode::Alpha(0.05));

        let rows = engine.results();
        let depth3 = rows.iter().find(|r| r.depth == 3).unwrap();
        // parent_mi = ln(2), mi = ln(2) → relmi = 0.
        assert!(depth3.relmi.abs() < 1e-10, "relmi = mi - parent_mi = 0");
    }

    #[test]
    fn results_members_ascending_1_based() {
        let m = identical_matrix();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));
        engine.depth(3, EvaluationMode::Alpha(0.05));

        for row in engine.results() {
            assert!(
                row.members.windows(2).all(|w| w[0] < w[1]),
                "members must be ascending: {:?}",
                row.members
            );
            assert!(
                row.members.iter().all(|&m| m >= 1),
                "1-based: {:?}",
                row.members
            );
        }
    }

    #[test]
    fn indicators_match_conjunction() {
        let m = identical_matrix();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        let expected = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        for ind in engine.indicators() {
            assert_eq!(ind, expected);
        }
    }

    #[test]
    fn mi_threshold_mode() {
        let m = identical_matrix();
        let ln2 = 2.0f64.ln();

        // MI = ln(2) ≈ 0.693 > 0.5 → retain.
        let mut engine = PairMiEngine::create(&m).unwrap();
        let stats = engine.depth(2, EvaluationMode::MiThreshold(0.5));
        assert_eq!(stats.kept, 3);

        // MI = ln(2) ≈ 0.693 < 1.0 → reject.
        let mut engine2 = PairMiEngine::create(&m).unwrap();
        let stats2 = engine2.depth(2, EvaluationMode::MiThreshold(1.0));
        assert_eq!(stats2.kept, 0);

        // Threshold exactly at ln(2) → mi > t is false (strict >).
        let mut engine3 = PairMiEngine::create(&m).unwrap();
        let stats3 = engine3.depth(2, EvaluationMode::MiThreshold(ln2));
        assert_eq!(stats3.kept, 0, "strict > : mi == threshold → not kept");
    }

    #[test]
    fn no_significant_pairs_n11_zero() {
        // Anti-correlated: n11 = 0 → g = 0 → p = 1.0.
        let data = [
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
        ];
        let m = BinaryMatrix::new(&data, 2, 10).unwrap();
        let mut engine = PairMiEngine::create(&m).unwrap();
        let stats = engine.depth(2, EvaluationMode::Alpha(0.05));
        assert_eq!(stats.kept, 0);
    }

    #[test]
    fn depth_3_without_depth_2() {
        let m = identical_matrix();
        let mut engine = PairMiEngine::create(&m).unwrap();
        let stats = engine.depth(3, EvaluationMode::Alpha(0.05));
        assert_eq!(stats.raw, 0);
        assert_eq!(stats.unique, 0);
        assert_eq!(stats.kept, 0);
    }

    #[test]
    fn g_test_p_negative_g_returns_one() {
        // n11 = 0 → jc = 0 → g = 0 → p = 1.0.
        assert_eq!(g_test_p(0.0, 10.0, 0.693), 1.0);
        // n11 = 5, n = 10, mi = -1.0 → jc = 5, g = -10 → p = 1.0.
        assert_eq!(g_test_p(5.0, 10.0, -1.0), 1.0);
    }

    #[test]
    fn n_sets_tracks_store() {
        let m = identical_matrix();
        let mut engine = PairMiEngine::create(&m).unwrap();
        assert_eq!(engine.n_sets(), 0);
        engine.depth(2, EvaluationMode::Alpha(0.05));
        assert_eq!(engine.n_sets(), 3);
        engine.depth(3, EvaluationMode::Alpha(0.05));
        assert_eq!(engine.n_sets(), 4);
    }
}
