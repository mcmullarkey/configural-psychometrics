//! Exhaustive cell evaluation + greedy antichain reduction.
//!
//! Ports `emsc_v4.cpp` cell evaluation: Wilson/point sufficiency gates,
//! per-target exclusion masks, greedy antichain reduction via cardinality-
//! ascending subset test, and `n_sufficient` counting.
//!
//! ## What this module provides
//!
//! - [`Criterion`] — sufficiency gate mode (Wilson lower bound or point estimate)
//! - [`Detail`] — output verbosity (all sufficient rows or minimal antichain only)
//! - [`CellRow`] — one evaluated configuration row
//! - [`CellOutput`] — cell evaluation result with rows and sufficient count
//! - [`evaluate_cell`] — evaluate one (target, theta, criterion) cell from the store
//!
//! ## What this module does NOT provide
//!
//! - Lattice ascent or configuration enumeration (see [`crate::exhaustive`])
//! - Statistical primitives (see [`crate::stats`])
//! - File I/O or serialization
//!
//! ## Algorithm
//!
//! The configuration store (`EmscStore`) is depth-major, so configs stream
//! in cardinality-ascending order. For each config:
//!
//! 1. **Exclusion check**: skip if the config's vocabulary mask shares any
//!    bit with the target's exclusion mask.
//! 2. **Sufficiency gate**: depending on `Criterion`:
//!    - `Wilson`: pre-filter `n/(n+z²) >= θ`, then `wilson_lower(p, n, z) >= θ`
//!    - `Point`: `(y1 + ε·n) >= θ·n` (EPSILON tolerance for floating-point)
//! 3. **Antichain membership**: a config is minimal if no previously accepted
//!    config is a subset of it (`acc_mask & m == acc_mask`). Accepted configs
//!    have cardinality ≤ current (stream is ascending), so only smaller-or-equal
//!    cardinality entries need checking.
//! 4. **Output**: `Detail::Full` emits all sufficient configs; `Detail::Count`
//!    emits only minimal (antichain) configs.

use crate::bitset::{extract_members, popcount};
use crate::exhaustive::EmscStore;
use crate::stats::wilson_lower;

/// Sufficiency gate mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Wilson score lower bound gate: `wilson_lower(p, n, z) >= θ`.
    Wilson,
    /// Point estimate gate: `(y1 + ε·n) >= θ·n` with EPSILON tolerance.
    Point,
}

/// Output verbosity mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detail {
    /// Emit all sufficient configurations (minimal and non-minimal).
    Full,
    /// Emit only minimal (antichain) configurations.
    Count,
}

/// One evaluated configuration row.
#[derive(Debug, Clone)]
pub struct CellRow {
    /// Cardinality (number of observations in the config).
    pub cardinality: u32,
    /// 0-based observation indices, ascending.
    pub set_idx: Vec<u32>,
    /// Support count (number of variables co-occurring).
    pub n_set: u32,
    /// Count of co-occurrences with the target.
    pub n_set_y1: u32,
    /// Conditional probability `y1 / n_set`.
    pub cprob: f64,
    /// Wilson score lower bound (computed for all, gated only for Wilson).
    pub wilson_lower: f64,
    /// Whether this config is minimal (antichain member).
    pub minimal: bool,
}

/// Cell evaluation output.
#[derive(Debug, Clone)]
pub struct CellOutput {
    /// Output rows (all sufficient for `Full`, minimal only for `Count`).
    pub rows: Vec<CellRow>,
    /// Total count of sufficient configurations (not just minimal).
    pub n_sufficient: i64,
}

/// Evaluate one (target, theta, criterion) cell from the store.
///
/// Iterates over all configurations in the store (cardinality-ascending),
/// applies the exclusion mask, sufficiency gate, and greedy antichain
/// reduction. Returns rows and the total sufficient count.
///
/// # Arguments
///
/// - `store` — the configuration store from exhaustive ascent
/// - `target_k` — target index (0-based)
/// - `theta` — sufficiency threshold
/// - `criterion` — sufficiency gate mode (Wilson or Point)
/// - `detail` — output verbosity (Full or Count)
///
/// # Exclusion mask default
///
/// If `target_k >= n_targets`, exclusion mask defaults to 0 (no exclusion).
#[must_use]
pub fn evaluate_cell(
    store: &EmscStore,
    target_k: usize,
    theta: f64,
    criterion: Criterion,
    detail: Detail,
) -> CellOutput {
    let n_cfg = store.cfg_mask.len();
    let excl = store.excl_mask.get(target_k).copied().unwrap_or(0);
    let z = store.z;
    let z2 = store.z2;
    let n_targets = store.n_targets as usize;

    let mut rows: Vec<CellRow> = Vec::new();
    let mut acc_masks: Vec<u64> = Vec::new();
    let mut acc_cards: Vec<u32> = Vec::new();
    let mut n_sufficient: i64 = 0;

    for g in 0..n_cfg {
        let m = store.cfg_mask[g];
        // Check exclusion mask
        if excl != 0 && (m & excl) != 0 {
            continue;
        }

        let nn = store.cfg_supp[g] as f64;
        let y1 = store.cfg_y1[g * n_targets + target_k];
        let p = y1 as f64 / nn;

        // Sufficiency gate
        let w: f64;
        match criterion {
            Criterion::Wilson => {
                // Pre-filter: n/(n+z²) >= theta
                if nn / (nn + z2) < theta {
                    continue;
                }
                w = wilson_lower(p, nn, z);
                if w < theta {
                    continue;
                }
            }
            Criterion::Point => {
                // Point gate with EPSILON tolerance
                let eps = f64::EPSILON;
                if (y1 as f64 + eps * nn) < theta * nn {
                    continue;
                }
                w = wilson_lower(p, nn, z); // computed but NOT gated
            }
        }

        n_sufficient += 1;

        // Greedy antichain membership (stream is cardinality-ascending)
        let card = popcount(m);
        let mut is_min = true;
        for a in 0..acc_masks.len() {
            if acc_cards[a] > card {
                continue;
            }
            if (acc_masks[a] & m) == acc_masks[a] {
                is_min = false;
                break;
            }
        }
        if is_min {
            acc_masks.push(m);
            acc_cards.push(card);
        }

        // Add to output based on detail mode
        if detail == Detail::Full || is_min {
            let members: Vec<u32> = extract_members(m).collect();
            rows.push(CellRow {
                cardinality: card,
                set_idx: members,
                n_set: store.cfg_supp[g],
                n_set_y1: y1,
                cprob: p,
                wilson_lower: w,
                minimal: is_min,
            });
        }
    }

    CellOutput { rows, n_sufficient }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exhaustive::ascend;
    use crate::matrix::BinaryMatrix;

    /// 4 vars × 4 obs (row-major input).
    /// Var 0: [1, 1, 0, 0]  (obs 0,1 have var 0)
    /// Var 1: [1, 0, 1, 0]  (obs 0,2 have var 1)
    /// Var 2: [0, 1, 1, 0]  (obs 1,2 have var 2)
    /// Var 3: [0, 0, 0, 1]  (obs 3 has var 3)
    ///
    /// Observation indicators (packed, V=4 → 1 word each):
    ///   Obs 0: [1,1,0,0] → 0b0011 = 3
    ///   Obs 1: [1,0,1,0] → 0b0101 = 5
    ///   Obs 2: [0,1,1,0] → 0b0110 = 6
    ///   Obs 3: [0,0,0,1] → 0b1000 = 8
    fn test_matrix_4x4() -> BinaryMatrix {
        let data = [
            1.0, 1.0, 0.0, 0.0, // var 0
            1.0, 0.0, 1.0, 0.0, // var 1
            0.0, 1.0, 1.0, 0.0, // var 2
            0.0, 0.0, 0.0, 1.0, // var 3
        ];
        BinaryMatrix::new(&data, 4, 4).unwrap()
    }

    /// Build a store from the 4×4 test matrix with 1 target.
    /// Target = [1,1,0,0] → packed = 0b0011 = 3.
    /// With n_min=1, max_card=3: depth-1 = 4 configs, depth-2 = 3 configs.
    fn build_store(excludes: &[Option<Vec<u32>>]) -> crate::exhaustive::EmscStore {
        let m = test_matrix_4x4();
        let targets = vec![vec![1, 1, 0, 0]];
        let thetas = vec![0.3];
        ascend(&m, &targets, &thetas, 1, 0.05, 3, 1000.0, excludes).store
    }

    // ---- Wilson gate: pre-filter passes but wilson_lower fails ----

    #[test]
    fn wilson_gate_prefilter_passes_but_wilson_fails() {
        // Store: depth-1 configs have supp=2 (obs 0,1,2) or supp=1 (obs 3).
        // z = z_score(0.05) ≈ 1.96, z² ≈ 3.8416
        // Pre-filter: n/(n+z²) >= theta
        //   supp=2: 2/(2+3.84) ≈ 0.342 >= 0.3 → passes pre-filter
        //   supp=1: 1/(1+3.84) ≈ 0.206 < 0.3 → fails pre-filter
        //
        // For obs 0 (supp=2, y1=2, p=1.0):
        //   wilson_lower(1.0, 2.0, 1.96) = (1 + z²/(2n) - z²/(2n)) / (1 + z²/n)
        //                                = 1 / (1 + 3.8416/2) = 1/2.9208 ≈ 0.342
        //   With theta=0.3: 0.342 >= 0.3 → passes Wilson gate
        //
        // For obs 1 (supp=2, y1=1, p=0.5):
        //   wilson_lower(0.5, 2.0, 1.96):
        //     denom = 1 + 3.8416/2 = 2.9208
        //     center = 0.5 + 3.8416/4 = 1.4604
        //     margin = 1.96 * sqrt(0.25/2 + 3.8416/16) = 1.96 * sqrt(0.125 + 0.2401)
        //            = 1.96 * sqrt(0.3651) = 1.96 * 0.6042 = 1.1843
        //     result = (1.4604 - 1.1843) / 2.9208 = 0.2761 / 2.9208 ≈ 0.0945
        //   With theta=0.3: 0.0945 < 0.3 → FAILS Wilson gate
        //
        // So obs 1 passes pre-filter (supp=2 → 0.342 >= 0.3) but fails wilson_lower.
        // This is the exact "pre-filter passes but wilson_lower fails" case.
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.3, Criterion::Wilson, Detail::Full);

        // obs 0 (mask=1): supp=2, y1=2, p=1.0 → wilson ≈ 0.342 >= 0.3 → sufficient
        // obs 1 (mask=2): supp=2, y1=1, p=0.5 → wilson ≈ 0.0945 < 0.3 → excluded
        // obs 2 (mask=4): supp=2, y1=1, p=0.5 → wilson ≈ 0.0945 < 0.3 → excluded
        // obs 3 (mask=8): supp=1 → pre-filter fails → excluded
        // depth-2 configs: supp=1 → pre-filter fails → excluded
        assert_eq!(out.n_sufficient, 1, "only obs 0 should pass Wilson gate");

        // Verify the one sufficient row is obs 0
        assert_eq!(out.rows.len(), 1);
        assert_eq!(out.rows[0].set_idx, vec![0]);
        assert_eq!(out.rows[0].n_set, 2);
        assert_eq!(out.rows[0].n_set_y1, 2);
        assert!((out.rows[0].cprob - 1.0).abs() < 1e-15);
        assert!(out.rows[0].minimal);
    }

    // ---- Wilson gate: verify wilson_lower value in output ----

    #[test]
    fn wilson_gate_wilson_lower_value_correct() {
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.3, Criterion::Wilson, Detail::Full);
        // Only obs 0 passes: p=1.0, n=2, z≈1.96
        let expected_w = wilson_lower(1.0, 2.0, store.z);
        assert!(
            (out.rows[0].wilson_lower - expected_w).abs() < 1e-15,
            "got {}, expected {}",
            out.rows[0].wilson_lower,
            expected_w
        );
    }

    // ---- Point gate with EPSILON ----

    #[test]
    fn point_gate_epsilon_boundary() {
        // Point gate: (y1 + eps*nn) >= theta*nn
        // With theta=1.0 and y1=nn (p=1.0): (nn + eps*nn) >= 1.0*nn → true (eps > 0)
        // With theta=1.0 and y1=nn-1 (p<1.0): (nn-1 + eps*nn) >= nn → eps*nn >= 1
        //   For nn=2: eps*2 ≈ 4.4e-16 < 1 → false → excluded
        //
        // obs 0: supp=2, y1=2, p=1.0 → (2 + eps*2) >= 1.0*2 → true → sufficient
        // obs 1: supp=2, y1=1, p=0.5 → (1 + eps*2) >= 1.0*2 → 1+4.4e-16 >= 2 → false
        // obs 2: supp=2, y1=1, p=0.5 → same as obs 1 → false
        // obs 3: supp=1, y1=0, p=0.0 → (0 + eps*1) >= 1.0*1 → false
        // depth-2: supp=1, y1 varies → (y1 + eps*1) >= 1.0*1 → only if y1=1
        //   config {0,1}: supp=1, y1=1 → (1 + eps) >= 1 → true → sufficient
        //   config {0,2}: supp=1, y1=1 → true → sufficient
        //   config {1,2}: supp=1, y1=0 → false
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 1.0, Criterion::Point, Detail::Full);

        // obs 0 (p=1.0) + depth-2 configs {0,1} and {0,2} (y1=1, supp=1, p=1.0)
        // = 3 sufficient
        assert_eq!(
            out.n_sufficient, 3,
            "obs 0 + 2 depth-2 configs with y1=supp"
        );

        // Verify all rows have p=1.0 (the only way to pass point gate with theta=1.0)
        for row in &out.rows {
            assert!(
                (row.cprob - 1.0).abs() < 1e-15,
                "point gate with theta=1.0 should only pass p=1.0, got p={}",
                row.cprob
            );
        }
    }

    #[test]
    fn point_gate_computes_wilson_but_does_not_gate() {
        // Point criterion computes wilson_lower but does NOT gate on it.
        // With theta=0.0: everything passes (y1 + eps*nn >= 0 always true for non-negative y1)
        // Even configs with very low wilson_lower should be included.
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        // All 7 configs should pass (theta=0.0 → always sufficient)
        assert_eq!(out.n_sufficient, 7);

        // wilson_lower should be computed (not zero/NaN) even though not gated
        for row in &out.rows {
            assert!(
                row.wilson_lower.is_finite(),
                "wilson_lower should be finite"
            );
            assert!(
                row.wilson_lower >= 0.0,
                "wilson_lower should be non-negative"
            );
        }
    }

    // ---- Greedy antichain: nested superset → minimal=false ----

    #[test]
    fn greedy_antichain_superset_not_minimal() {
        // With Point criterion and theta=0.0, all 7 configs are sufficient.
        // Depth-1 configs (card=1): obs 0 (mask=1), obs 1 (mask=2), obs 2 (mask=4), obs 3 (mask=8)
        // Depth-2 configs (card=2): {0,1} (mask=3), {0,2} (mask=5), {1,2} (mask=6)
        //
        // Antichain processing (cardinality-ascending):
        //   mask=1 (card=1): no accepted → minimal=true, accept
        //   mask=2 (card=1): no subset of 2 in {1} → minimal=true, accept
        //   mask=4 (card=1): no subset of 4 in {1,2} → minimal=true, accept
        //   mask=8 (card=1): no subset of 8 in {1,2,4} → minimal=true, accept
        //   mask=3 (card=2): 1&3==1 → 1 is subset of 3 → minimal=false
        //   mask=5 (card=2): 1&5==1 → 1 is subset of 5 → minimal=false
        //   mask=6 (card=2): 2&6==2 → 2 is subset of 6 → minimal=false
        //
        // So all depth-2 configs should have minimal=false.
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        // 4 minimal (depth-1) + 3 non-minimal (depth-2)
        let minimal_count = out.rows.iter().filter(|r| r.minimal).count();
        let non_minimal_count = out.rows.iter().filter(|r| !r.minimal).count();
        assert_eq!(minimal_count, 4, "4 depth-1 configs should be minimal");
        assert_eq!(
            non_minimal_count, 3,
            "3 depth-2 configs should be non-minimal"
        );

        // Verify depth-2 configs specifically have minimal=false
        for row in &out.rows {
            if row.cardinality == 2 {
                assert!(
                    !row.minimal,
                    "depth-2 config {:?} should be non-minimal (superset of depth-1)",
                    row.set_idx
                );
            }
        }
    }

    #[test]
    fn greedy_antichain_subset_test_correct() {
        // Verify the subset test: (acc_mask & m) == acc_mask
        // mask=3 ({0,1}) is superset of mask=1 ({0}) and mask=2 ({1})
        // mask=5 ({0,2}) is superset of mask=1 ({0}) and mask=4 ({2})
        // mask=6 ({1,2}) is superset of mask=2 ({1}) and mask=4 ({2})
        //
        // But mask=3 is NOT a superset of mask=4 ({2}): 4&3 = 0 != 4
        // So the subset test must check ALL accepted, not just the first.
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        // Find the row for mask=3 ({0,1})
        let row_01 = out.rows.iter().find(|r| r.set_idx == vec![0, 1]).unwrap();
        assert!(!row_01.minimal, "{{0,1}} should be non-minimal");

        // Find the row for mask=6 ({1,2})
        let row_12 = out.rows.iter().find(|r| r.set_idx == vec![1, 2]).unwrap();
        assert!(!row_12.minimal, "{{1,2}} should be non-minimal");
    }

    // ---- n_sufficient counts ALL sufficient (not just minimal) ----

    #[test]
    fn n_sufficient_counts_all_not_just_minimal() {
        // With Point criterion and theta=0.0: all 7 configs sufficient.
        // 4 minimal (depth-1) + 3 non-minimal (depth-2) = 7 total.
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        assert_eq!(
            out.n_sufficient, 7,
            "n_sufficient should count ALL sufficient"
        );

        // In Full mode, rows also = 7 (all emitted)
        assert_eq!(out.rows.len(), 7);
    }

    #[test]
    fn n_sufficient_independent_of_detail_mode() {
        // n_sufficient should be the same regardless of Detail mode.
        let store = build_store(&[None]);

        let out_full = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);
        let out_count = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Count);

        assert_eq!(
            out_full.n_sufficient, out_count.n_sufficient,
            "n_sufficient should be identical regardless of detail mode"
        );
        // Full emits all 7, Count emits only 4 minimal
        assert_eq!(out_full.rows.len(), 7);
        assert_eq!(
            out_count.rows.len(),
            4,
            "Count mode should emit only minimal"
        );
    }

    // ---- Detail::Full vs Detail::Count ----

    #[test]
    fn detail_full_emits_all_sufficient() {
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        // All 7 sufficient configs emitted
        assert_eq!(out.rows.len(), 7);

        // Should include both minimal and non-minimal
        assert!(
            out.rows.iter().any(|r| r.minimal),
            "should have minimal rows"
        );
        assert!(
            out.rows.iter().any(|r| !r.minimal),
            "should have non-minimal rows"
        );
    }

    #[test]
    fn detail_count_emits_only_minimal() {
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Count);

        // Only 4 minimal configs emitted
        assert_eq!(out.rows.len(), 4);

        // All emitted rows should be minimal
        for row in &out.rows {
            assert!(
                row.minimal,
                "Count mode should only emit minimal rows, got {:?}",
                row.set_idx
            );
        }
    }

    #[test]
    fn detail_count_n_sufficient_still_counts_all() {
        // Even in Count mode, n_sufficient counts ALL sufficient (7), not just minimal (4).
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Count);

        assert_eq!(out.rows.len(), 4, "Count mode emits 4 minimal");
        assert_eq!(out.n_sufficient, 7, "but n_sufficient counts all 7");
    }

    // ---- Exclusion mask: config sharing bits with excl → excluded ----

    #[test]
    fn exclusion_mask_excludes_sharing_bits() {
        // Exclude variable 0 (bit 0): excl_mask = 1
        // Configs with bit 0 set: mask=1 (obs 0), mask=3 ({0,1}), mask=5 ({0,2})
        // Configs without bit 0: mask=2 (obs 1), mask=4 (obs 2), mask=8 (obs 3), mask=6 ({1,2})
        let store = build_store(&[Some(vec![0])]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        // 7 total configs - 3 with bit 0 = 4 remaining
        assert_eq!(
            out.n_sufficient, 4,
            "3 configs with bit 0 should be excluded"
        );

        // Verify no emitted row contains observation index 0
        for row in &out.rows {
            assert!(
                !row.set_idx.contains(&0),
                "obs 0 should be excluded, but found in {:?}",
                row.set_idx
            );
        }
    }

    #[test]
    fn exclusion_mask_multiple_bits() {
        // Exclude variables 0 and 2 (bits 0 and 2): excl_mask = 0b101 = 5
        // Configs with bit 0 OR bit 2 set:
        //   mask=1 (bit 0), mask=4 (bit 2), mask=3 (bits 0,1), mask=5 (bits 0,2), mask=6 (bits 1,2)
        // Configs without bits 0 or 2: mask=2 (obs 1), mask=8 (obs 3)
        let store = build_store(&[Some(vec![0, 2])]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        // 7 total - 5 excluded = 2 remaining
        assert_eq!(
            out.n_sufficient, 2,
            "5 configs with bits 0 or 2 should be excluded"
        );

        // Verify only obs 1 and obs 3 remain
        let mut all_indices: Vec<u32> = out
            .rows
            .iter()
            .flat_map(|r| r.set_idx.iter().copied())
            .collect();
        all_indices.sort();
        all_indices.dedup();
        assert_eq!(all_indices, vec![1, 3], "only obs 1 and 3 should remain");
    }

    #[test]
    fn exclusion_mask_none_excludes_nothing() {
        // No exclusion → all configs pass exclusion check
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        assert_eq!(
            out.n_sufficient, 7,
            "no exclusion → all 7 configs sufficient"
        );
    }

    // ---- set_idx 0-based ascending ----

    #[test]
    fn set_idx_zero_based_ascending() {
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        for row in &out.rows {
            // Verify ascending
            for w in row.set_idx.windows(2) {
                assert!(
                    w[0] < w[1],
                    "set_idx should be ascending, got {:?}",
                    row.set_idx
                );
            }
            // Verify 0-based (all indices < 4 = n_cols)
            for &idx in &row.set_idx {
                assert!(idx < 4, "set_idx should be 0-based, got {}", idx);
            }
        }
    }

    // ---- CellRow fields correctness ----

    #[test]
    fn cell_row_fields_correct() {
        let store = build_store(&[None]);
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        // Find obs 0 (mask=1, card=1)
        let row = out.rows.iter().find(|r| r.set_idx == vec![0]).unwrap();
        assert_eq!(row.cardinality, 1);
        assert_eq!(row.n_set, 2, "obs 0 has supp=2");
        assert_eq!(
            row.n_set_y1, 2,
            "obs 0 y1=2 (target=[1,1,0,0], obs 0=[1,1,0,0])"
        );
        assert!((row.cprob - 1.0).abs() < 1e-15, "p = 2/2 = 1.0");
        assert!(row.minimal, "depth-1 config should be minimal");
    }

    // ---- Wilson vs Point gate difference ----

    #[test]
    fn wilson_gate_stricter_than_point() {
        // With theta=0.3:
        // Wilson gate: only obs 0 passes (wilson_lower(1.0, 2, 1.96) ≈ 0.342 >= 0.3)
        // Point gate: obs 0 (p=1.0 >= 0.3), obs 1 (p=0.5 >= 0.3), obs 2 (p=0.5 >= 0.3)
        //   depth-2: {0,1} (p=1.0), {0,2} (p=1.0), {1,2} (p=0.0 < 0.3 → excluded)
        //   obs 3 (p=0.0 < 0.3 → excluded)
        //   Point sufficient: obs 0, obs 1, obs 2, {0,1}, {0,2} = 5
        let store = build_store(&[None]);

        let out_wilson = evaluate_cell(&store, 0, 0.3, Criterion::Wilson, Detail::Full);
        let out_point = evaluate_cell(&store, 0, 0.3, Criterion::Point, Detail::Full);

        assert!(
            out_wilson.n_sufficient < out_point.n_sufficient,
            "Wilson gate should be stricter: Wilson={} vs Point={}",
            out_wilson.n_sufficient,
            out_point.n_sufficient
        );
        assert_eq!(out_wilson.n_sufficient, 1, "Wilson: only obs 0");
        assert_eq!(
            out_point.n_sufficient, 5,
            "Point: obs 0,1,2 + {{0,1}},{{0,2}}"
        );
    }

    // ---- Empty store edge case ----

    #[test]
    fn empty_store_returns_empty() {
        let store = crate::exhaustive::EmscStore {
            cfg_mask: vec![],
            cfg_supp: vec![],
            cfg_y1: vec![],
            depth_end: vec![],
            excl_mask: vec![0],
            n_targets: 1,
            z: 1.96,
            z2: 3.8416,
        };
        let out = evaluate_cell(&store, 0, 0.3, Criterion::Wilson, Detail::Full);
        assert_eq!(out.n_sufficient, 0);
        assert!(out.rows.is_empty());
    }

    // ---- Equal-cardinality antichain: disjoint configs both minimal ----

    #[test]
    fn equal_cardinality_disjoint_both_minimal() {
        // Two card=2 configs that are NOT subsets of each other (disjoint).
        // mask=3 ({0,1}) and mask=12 ({2,3}) — both card=2, no shared bits.
        // Neither is a subset of the other → both should be minimal.
        let store = crate::exhaustive::EmscStore {
            cfg_mask: vec![0b0011, 0b1100],
            cfg_supp: vec![2, 2],
            cfg_y1: vec![2, 2],
            depth_end: vec![2],
            excl_mask: vec![0],
            n_targets: 1,
            z: 1.96,
            z2: 3.8416,
        };
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        assert_eq!(out.n_sufficient, 2, "both configs should be sufficient");
        assert_eq!(out.rows.len(), 2, "both should be emitted in Full mode");

        // Both should be minimal — neither is a subset of the other
        for row in &out.rows {
            assert!(
                row.minimal,
                "disjoint equal-cardinality config {:?} should be minimal",
                row.set_idx
            );
        }
    }

    // ---- Equal-cardinality antichain: duplicate → later non-minimal ----

    #[test]
    fn equal_cardinality_duplicate_later_non_minimal() {
        // Two identical card=2 configs (same mask=3, {0,1}).
        // The second is a subset of the first (acc_mask & m == acc_mask when equal).
        // First should be minimal=true, second should be minimal=false.
        let store = crate::exhaustive::EmscStore {
            cfg_mask: vec![0b0011, 0b0011],
            cfg_supp: vec![2, 2],
            cfg_y1: vec![2, 2],
            depth_end: vec![2],
            excl_mask: vec![0],
            n_targets: 1,
            z: 1.96,
            z2: 3.8416,
        };
        let out = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);

        assert_eq!(out.n_sufficient, 2, "both configs should be sufficient");
        assert_eq!(out.rows.len(), 2, "both should be emitted in Full mode");

        // First config should be minimal, second should NOT
        assert!(
            out.rows[0].minimal,
            "first duplicate config should be minimal"
        );
        assert!(
            !out.rows[1].minimal,
            "second duplicate config should be non-minimal (subset of first)"
        );
    }

    // ---- Multi-target y1 indexing: same config, different target_k ----

    #[test]
    fn multi_target_y1_indexing_differs_by_target() {
        // Build a store with n_targets=2, distinct targets.
        // target 0 = [1,1,0,0] (obs 0,1) → packed = 0b0011 = 3
        // target 1 = [0,0,1,1] (obs 2,3) → packed = 0b1100 = 12
        //
        // obs 0 (mask=1, packed=3, supp=2):
        //   y1 for target 0 = AND(3, 3) = 3 → popcount = 2
        //   y1 for target 1 = AND(3, 12) = 0 → popcount = 0
        //
        // This proves the config-major indexing cfg_y1[g * n_targets + target_k]
        // is correct: the same config g yields different y1 for different target_k.
        let m = test_matrix_4x4();
        let targets = vec![
            vec![1, 1, 0, 0], // target 0: obs 0,1
            vec![0, 0, 1, 1], // target 1: obs 2,3
        ];
        let thetas = vec![0.3];
        let excludes = vec![None, None];
        let store = ascend(&m, &targets, &thetas, 1, 0.05, 3, 1000.0, &excludes).store;

        assert_eq!(store.n_targets, 2, "store should have 2 targets");

        // Evaluate with target_k=0 and target_k=1
        let out0 = evaluate_cell(&store, 0, 0.0, Criterion::Point, Detail::Full);
        let out1 = evaluate_cell(&store, 1, 0.0, Criterion::Point, Detail::Full);

        // Find obs 0 (set_idx=[0]) in both outputs
        let row0 = out0
            .rows
            .iter()
            .find(|r| r.set_idx == vec![0])
            .expect("obs 0 should be in target 0 output");
        let row1 = out1
            .rows
            .iter()
            .find(|r| r.set_idx == vec![0])
            .expect("obs 0 should be in target 1 output");

        // n_set should be the same (support is target-independent)
        assert_eq!(row0.n_set, row1.n_set, "n_set should be target-independent");

        // n_set_y1 should DIFFER — this is the key assertion
        assert_eq!(
            row0.n_set_y1, 2,
            "obs 0 y1 for target 0 (obs 0,1) should be 2"
        );
        assert_eq!(
            row1.n_set_y1, 0,
            "obs 0 y1 for target 1 (obs 2,3) should be 0"
        );
        assert_ne!(
            row0.n_set_y1, row1.n_set_y1,
            "n_set_y1 must differ between targets — proves config-major indexing"
        );
    }
}
