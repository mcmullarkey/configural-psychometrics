//! Unified Cell representation + adapters from both engines.
//!
//! ## What this module provides
//!
//! - [`CellSource`] — which engine produced this cell
//! - [`CellMeta`] — per-cell metadata (benchmark, criterion, threshold, key)
//! - [`Cell`] — unified row collection with members, stats, and metadata
//! - [`from_exhaustive`] — adapter from exhaustive engine [`CellOutput`]
//! - [`from_pairmi`] — adapter from constructive engine [`PairMiEngine`]
//!
//! ## What this module does NOT provide
//!
//! - Cell evaluation logic (see [`crate::cell_eval`])
//! - Engine ascent logic (see [`crate::exhaustive`], [`crate::pairmi`])
//! - Statistical primitives (see [`crate::stats`])
//!
//! ## Design
//!
//! Both engines produce antichains of configurations but with different
//! representations. The exhaustive engine emits [`CellRow`]s with 0-based
//! `set_idx` and pre-computed stats. The constructive engine emits
//! [`PairMiResultRow`]s with 1-based `members` and only MI/G-test stats.
//!
//! [`from_exhaustive`] filters to `minimal == true` rows and copies stats
//! directly. [`from_pairmi`] converts 1-based members to 0-based and
//! recomputes `cprob`/`wilson_lower`/`n_set`/`n_set_y1` from the engine's
//! observation indicators and the target vector.

use crate::cell_eval::{CellOutput, Criterion};
use crate::pairmi::PairMiEngine;
use crate::stats::{wilson_lower, z_score};

/// Which engine produced this cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellSource {
    /// Exhaustive lattice ascent engine.
    Exhaustive,
    /// MI-restricted constructive ascent engine.
    MiRestricted,
}

/// Per-cell metadata.
#[derive(Debug, Clone)]
pub struct CellMeta {
    /// Which engine produced this cell.
    pub source: CellSource,
    /// Significance level (if alpha-mode was used).
    pub alpha: Option<f64>,
    /// Benchmark identifier.
    pub benchmark: String,
    /// Human-readable benchmark label.
    pub bench_label: String,
    /// Sufficiency gate mode (Wilson or Point).
    pub criterion: Criterion,
    /// Sufficiency threshold.
    pub threshold: f64,
    /// Unique cell key (e.g. "target0_theta0.3_wilson").
    pub key: String,
}

/// Unified cell representation — one antichain of configurations.
///
/// All vectors are parallel: element `i` corresponds to the `i`-th
/// configuration in the antichain. `members[i]` is 0-based, strictly
/// ascending.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Per-cell metadata.
    pub meta: CellMeta,
    /// 0-based observation indices, each inner vec strictly ascending.
    pub members: Vec<Vec<u32>>,
    /// Conditional probability `y1 / n_set` for each config.
    pub cprob: Vec<f64>,
    /// Wilson score lower bound for each config.
    pub wilson_lower: Vec<f64>,
    /// Support count for each config.
    pub n_set: Vec<u32>,
    /// Count of co-occurrences with the target for each config.
    pub n_set_y1: Vec<u32>,
}

/// Adapter from exhaustive engine output ([`CellOutput`] from `evaluate_cell`).
///
/// Filters to `minimal == true` rows only, preserving 0-based `set_idx`
/// and copying pre-computed stats directly.
#[must_use]
pub fn from_exhaustive(output: &CellOutput, meta: CellMeta) -> Cell {
    let mut members = Vec::new();
    let mut cprob = Vec::new();
    let mut wl = Vec::new();
    let mut n_set = Vec::new();
    let mut n_set_y1 = Vec::new();

    for row in &output.rows {
        if row.minimal {
            members.push(row.set_idx.clone());
            cprob.push(row.cprob);
            wl.push(row.wilson_lower);
            n_set.push(row.n_set);
            n_set_y1.push(row.n_set_y1);
        }
    }

    Cell {
        meta,
        members,
        cprob,
        wilson_lower: wl,
        n_set,
        n_set_y1,
    }
}

/// Adapter from constructive engine ([`PairMiEngine`]).
///
/// Converts 1-based members to 0-based, computes `cprob`/`wilson_lower`/
/// `n_set`/`n_set_y1` from the engine's observation indicators and the
/// target vector.
///
/// # Arguments
///
/// - `engine` — the constructive engine (after running `depth()`)
/// - `target` — target indicator vector (length N, 0/1 per observation)
/// - `alpha` — significance level for Wilson z-score
/// - `meta` — per-cell metadata
#[must_use]
pub fn from_pairmi(engine: &PairMiEngine, target: &[u8], alpha: f64, meta: CellMeta) -> Cell {
    let results = engine.results();
    let indicators = engine.indicators(); // nsets × N (set-major)
    let z = z_score(alpha);
    let n_obs = target.len();

    let mut members = Vec::new();
    let mut cprob = Vec::new();
    let mut wl = Vec::new();
    let mut n_set = Vec::new();
    let mut n_set_y1 = Vec::new();

    for (k, row) in results.iter().enumerate() {
        // Convert 1-based members to 0-based.
        let members_0based: Vec<u32> = row.members.iter().map(|&m| m - 1).collect();
        members.push(members_0based);

        // n_set = pop (support count from PairMi).
        let ns = row.pop;
        n_set.push(ns);

        // n_set_y1 = count of observations where indicators[k][i] == 1
        // AND target[i] == 1.
        let mut y1 = 0u32;
        if k < indicators.len() {
            for i in 0..n_obs {
                if i < indicators[k].len() && indicators[k][i] == 1 && target[i] == 1 {
                    y1 += 1;
                }
            }
        }
        n_set_y1.push(y1);

        // cprob = y1 / n_set.
        let cp = if ns > 0 { y1 as f64 / ns as f64 } else { 0.0 };
        cprob.push(cp);

        // Wilson score lower bound.
        let w = if ns > 0 {
            wilson_lower(cp, ns as f64, z)
        } else {
            0.0
        };
        wl.push(w);
    }

    Cell {
        meta,
        members,
        cprob,
        wilson_lower: wl,
        n_set,
        n_set_y1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell_eval::{evaluate_cell, Criterion, Detail};
    use crate::exhaustive::ascend;
    use crate::matrix::BinaryMatrix;
    use crate::pairmi::{EvaluationMode, PairMiEngine};

    // ---- Test fixtures ----

    /// 4 vars × 4 obs (row-major input).
    /// Var 0: [1, 1, 0, 0]  (obs 0,1 have var 0)
    /// Var 1: [1, 0, 1, 0]  (obs 0,2 have var 1)
    /// Var 2: [0, 1, 1, 0]  (obs 1,2 have var 2)
    /// Var 3: [0, 0, 0, 1]  (obs 3 has var 3)
    fn test_matrix_4x4() -> BinaryMatrix {
        let data = [
            1.0, 1.0, 0.0, 0.0, // var 0
            1.0, 0.0, 1.0, 0.0, // var 1
            0.0, 1.0, 1.0, 0.0, // var 2
            0.0, 0.0, 0.0, 1.0, // var 3
        ];
        BinaryMatrix::new(&data, 4, 4).unwrap()
    }

    fn dummy_meta(source: CellSource) -> CellMeta {
        CellMeta {
            source,
            alpha: Some(0.05),
            benchmark: "test".to_string(),
            bench_label: "Test".to_string(),
            criterion: Criterion::Wilson,
            threshold: 0.3,
            key: "test_key".to_string(),
        }
    }

    // ---- from_exhaustive: filter minimal==true, preserve 0-based set_idx, copy stats ----

    #[test]
    fn from_exhaustive_filters_minimal_only() {
        let m = test_matrix_4x4();
        let targets = vec![vec![1, 1, 0, 0]];
        let thetas = vec![0.3];
        let result = ascend(&m, &targets, &thetas, 1, 0.05, 3, 1000.0, &[None]);
        let store = &result.store;

        // Use Point criterion with theta=0.0 so all 7 configs are sufficient.
        // In Full mode: 4 minimal + 3 non-minimal.
        let out = evaluate_cell(store, 0, 0.0, Criterion::Point, Detail::Full);
        assert_eq!(out.rows.len(), 7, "Full mode emits all 7");
        let minimal_count = out.rows.iter().filter(|r| r.minimal).count();
        assert_eq!(minimal_count, 4, "4 minimal rows");

        let cell = from_exhaustive(&out, dummy_meta(CellSource::Exhaustive));

        // Should only have the 4 minimal rows.
        assert_eq!(
            cell.members.len(),
            4,
            "from_exhaustive should filter to minimal"
        );
        assert_eq!(cell.cprob.len(), 4);
        assert_eq!(cell.wilson_lower.len(), 4);
        assert_eq!(cell.n_set.len(), 4);
        assert_eq!(cell.n_set_y1.len(), 4);
    }

    #[test]
    fn from_exhaustive_preserves_zero_based_set_idx() {
        let m = test_matrix_4x4();
        let targets = vec![vec![1, 1, 0, 0]];
        let thetas = vec![0.3];
        let result = ascend(&m, &targets, &thetas, 1, 0.05, 3, 1000.0, &[None]);
        let out = evaluate_cell(&result.store, 0, 0.0, Criterion::Point, Detail::Full);
        let cell = from_exhaustive(&out, dummy_meta(CellSource::Exhaustive));

        // All indices should be 0-based (< 4 = n_cols).
        for members in &cell.members {
            for &idx in members {
                assert!(idx < 4, "set_idx should be 0-based, got {idx}");
            }
            // Strictly ascending.
            for w in members.windows(2) {
                assert!(w[0] < w[1], "members should be ascending: {members:?}");
            }
        }
    }

    #[test]
    fn from_exhaustive_copies_stats_correctly() {
        let m = test_matrix_4x4();
        let targets = vec![vec![1, 1, 0, 0]];
        let thetas = vec![0.3];
        let result = ascend(&m, &targets, &thetas, 1, 0.05, 3, 1000.0, &[None]);
        let out = evaluate_cell(&result.store, 0, 0.0, Criterion::Point, Detail::Full);

        // Get the minimal rows directly from CellOutput for comparison.
        let minimal_rows: Vec<_> = out.rows.iter().filter(|r| r.minimal).collect();

        let cell = from_exhaustive(&out, dummy_meta(CellSource::Exhaustive));

        for (i, row) in minimal_rows.iter().enumerate() {
            assert_eq!(cell.members[i], row.set_idx, "members should match");
            assert!(
                (cell.cprob[i] - row.cprob).abs() < 1e-15,
                "cprob should match"
            );
            assert!(
                (cell.wilson_lower[i] - row.wilson_lower).abs() < 1e-15,
                "wilson_lower should match"
            );
            assert_eq!(cell.n_set[i], row.n_set, "n_set should match");
            assert_eq!(cell.n_set_y1[i], row.n_set_y1, "n_set_y1 should match");
        }
    }

    // ---- from_pairmi: convert 1-based→0-based, compute n_set_y1 from indicators+target ----

    /// 3 vars × 10 obs, all identical: [1,1,1,1,1,0,0,0,0,0].
    fn identical_matrix_3x10() -> BinaryMatrix {
        let row: [f64; 10] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let data: Vec<f64> = [row, row, row].into_iter().flatten().collect();
        BinaryMatrix::new(&data, 3, 10).unwrap()
    }

    #[test]
    fn from_pairmi_converts_1_based_to_0_based() {
        let m = identical_matrix_3x10();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        // Target: first 5 obs are 1.
        let target = vec![1u8, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        let cell = from_pairmi(&engine, &target, 0.05, dummy_meta(CellSource::MiRestricted));

        // 3 pairs retained: {0,1}, {0,2}, {1,2} (1-based: {1,2}, {1,3}, {2,3}).
        assert_eq!(cell.members.len(), 3, "3 pairs retained");

        // All members should be 0-based (0, 1, 2 — not 1, 2, 3).
        for members in &cell.members {
            for &idx in members {
                assert!(idx < 3, "members should be 0-based, got {idx}");
            }
        }

        // Verify specific sets: {0,1}, {0,2}, {1,2} in some order.
        let mut sorted_members: Vec<_> = cell.members.clone();
        sorted_members.sort();
        assert_eq!(sorted_members, vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
    }

    #[test]
    fn from_pairmi_computes_n_set_y1_from_indicators_and_target() {
        let m = identical_matrix_3x10();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        // Target: first 5 obs are 1 (same as the conjunction → y1 = 5 for all).
        let target = vec![1u8, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        let cell = from_pairmi(&engine, &target, 0.05, dummy_meta(CellSource::MiRestricted));

        // All 3 pairs have pop=5 (obs 0-4 in conjunction).
        // Target also has 1s at obs 0-4 → y1 = 5 for all.
        for i in 0..3 {
            assert_eq!(cell.n_set[i], 5, "n_set should be 5 (pop of conjunction)");
            assert_eq!(
                cell.n_set_y1[i], 5,
                "n_set_y1 should be 5 (all conjunction obs have target=1)"
            );
            assert!(
                (cell.cprob[i] - 1.0).abs() < 1e-15,
                "cprob should be 1.0 (5/5)"
            );
        }
    }

    #[test]
    fn from_pairmi_n_set_y1_with_partial_target_overlap() {
        // Use a target that only partially overlaps the conjunction.
        // Conjunction = obs 0-4 (all 1s). Target = obs 0-2 only.
        let m = identical_matrix_3x10();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        let target = vec![1u8, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let cell = from_pairmi(&engine, &target, 0.05, dummy_meta(CellSource::MiRestricted));

        // n_set = 5 (conjunction has 5 obs), y1 = 3 (only obs 0-2 have target=1).
        for i in 0..3 {
            assert_eq!(cell.n_set[i], 5, "n_set = 5");
            assert_eq!(cell.n_set_y1[i], 3, "n_set_y1 = 3 (partial overlap)");
            assert!(
                (cell.cprob[i] - 3.0 / 5.0).abs() < 1e-15,
                "cprob should be 3/5 = 0.6"
            );
        }
    }

    #[test]
    fn from_pairmi_n_set_y1_zero_target() {
        // Target all zeros → y1 = 0 for all.
        let m = identical_matrix_3x10();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        let target = vec![0u8; 10];
        let cell = from_pairmi(&engine, &target, 0.05, dummy_meta(CellSource::MiRestricted));

        for i in 0..3 {
            assert_eq!(cell.n_set_y1[i], 0, "y1 should be 0 with all-zero target");
            assert_eq!(cell.cprob[i], 0.0, "cprob should be 0.0");
        }
    }

    // ---- Empty antichain → empty vectors, no panic ----

    #[test]
    fn from_exhaustive_empty_output_no_panic() {
        let empty_output = CellOutput {
            rows: vec![],
            n_sufficient: 0,
        };
        let cell = from_exhaustive(&empty_output, dummy_meta(CellSource::Exhaustive));

        assert!(cell.members.is_empty());
        assert!(cell.cprob.is_empty());
        assert!(cell.wilson_lower.is_empty());
        assert!(cell.n_set.is_empty());
        assert!(cell.n_set_y1.is_empty());
    }

    #[test]
    fn from_pairmi_empty_engine_no_panic() {
        // Engine with no retained sets (no depth() called).
        let m = identical_matrix_3x10();
        let engine = PairMiEngine::create(&m).unwrap();
        let target = vec![1u8; 10];
        let cell = from_pairmi(&engine, &target, 0.05, dummy_meta(CellSource::MiRestricted));

        assert!(cell.members.is_empty());
        assert!(cell.cprob.is_empty());
        assert!(cell.wilson_lower.is_empty());
        assert!(cell.n_set.is_empty());
        assert!(cell.n_set_y1.is_empty());
    }

    #[test]
    fn from_pairmi_no_significant_sets_no_panic() {
        // Anti-correlated matrix → no significant pairs → empty results.
        let data = [
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
        ];
        let m = BinaryMatrix::new(&data, 2, 10).unwrap();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        let target = vec![1u8; 10];
        let cell = from_pairmi(&engine, &target, 0.05, dummy_meta(CellSource::MiRestricted));

        assert!(cell.members.is_empty(), "no significant pairs → empty cell");
    }

    // ---- Cross-engine parity: same members + same n_set/n_set_y1 → identical cprob/wilson_lower ----

    #[test]
    fn cross_engine_parity_identical_stats() {
        // Build a Cell via from_exhaustive with known n_set/n_set_y1.
        // Then build a Cell via from_pairmi with the same n_set/n_set_y1.
        // Verify cprob and wilson_lower match.

        let m = identical_matrix_3x10();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));

        let target = vec![1u8, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        let alpha = 0.05;
        let z = z_score(alpha);

        let cell_pairmi = from_pairmi(
            &engine,
            &target,
            alpha,
            dummy_meta(CellSource::MiRestricted),
        );

        // Manually compute expected cprob/wilson_lower from n_set/n_set_y1.
        for i in 0..cell_pairmi.members.len() {
            let ns = cell_pairmi.n_set[i] as f64;
            let y1 = cell_pairmi.n_set_y1[i] as f64;
            let expected_cp = y1 / ns;
            let expected_wl = wilson_lower(expected_cp, ns, z);

            assert!(
                (cell_pairmi.cprob[i] - expected_cp).abs() < 1e-15,
                "cprob mismatch: got {}, expected {}",
                cell_pairmi.cprob[i],
                expected_cp
            );
            assert!(
                (cell_pairmi.wilson_lower[i] - expected_wl).abs() < 1e-15,
                "wilson_lower mismatch: got {}, expected {}",
                cell_pairmi.wilson_lower[i],
                expected_wl
            );
        }

        // Now build a synthetic CellOutput with the same n_set/n_set_y1
        // and verify from_exhaustive produces the same cprob/wilson_lower.
        let mut synthetic_rows = Vec::new();
        for i in 0..cell_pairmi.members.len() {
            let ns = cell_pairmi.n_set[i];
            let y1 = cell_pairmi.n_set_y1[i];
            let cp = y1 as f64 / ns as f64;
            synthetic_rows.push(crate::cell_eval::CellRow {
                cardinality: cell_pairmi.members[i].len() as u32,
                set_idx: cell_pairmi.members[i].clone(),
                n_set: ns,
                n_set_y1: y1,
                cprob: cp,
                wilson_lower: wilson_lower(cp, ns as f64, z),
                minimal: true,
            });
        }
        let synthetic_output = CellOutput {
            rows: synthetic_rows,
            n_sufficient: cell_pairmi.members.len() as i64,
        };
        let cell_exhaustive =
            from_exhaustive(&synthetic_output, dummy_meta(CellSource::Exhaustive));

        // Cross-engine parity: same n_set/n_set_y1 → identical cprob/wilson_lower.
        for i in 0..cell_pairmi.members.len() {
            assert_eq!(
                cell_pairmi.n_set[i], cell_exhaustive.n_set[i],
                "n_set should match"
            );
            assert_eq!(
                cell_pairmi.n_set_y1[i], cell_exhaustive.n_set_y1[i],
                "n_set_y1 should match"
            );
            assert!(
                (cell_pairmi.cprob[i] - cell_exhaustive.cprob[i]).abs() < 1e-15,
                "cprob parity: pairmi={} vs exhaustive={}",
                cell_pairmi.cprob[i],
                cell_exhaustive.cprob[i]
            );
            assert!(
                (cell_pairmi.wilson_lower[i] - cell_exhaustive.wilson_lower[i]).abs() < 1e-15,
                "wilson_lower parity: pairmi={} vs exhaustive={}",
                cell_pairmi.wilson_lower[i],
                cell_exhaustive.wilson_lower[i]
            );
        }
    }

    // ---- Vector length consistency ----

    #[test]
    fn from_exhaustive_vector_length_consistency() {
        let m = test_matrix_4x4();
        let targets = vec![vec![1, 1, 0, 0]];
        let thetas = vec![0.3];
        let result = ascend(&m, &targets, &thetas, 1, 0.05, 3, 1000.0, &[None]);
        let out = evaluate_cell(&result.store, 0, 0.0, Criterion::Point, Detail::Full);
        let cell = from_exhaustive(&out, dummy_meta(CellSource::Exhaustive));

        let n = cell.members.len();
        assert_eq!(cell.cprob.len(), n, "cprob length should match members");
        assert_eq!(
            cell.wilson_lower.len(),
            n,
            "wilson_lower length should match members"
        );
        assert_eq!(cell.n_set.len(), n, "n_set length should match members");
        assert_eq!(
            cell.n_set_y1.len(),
            n,
            "n_set_y1 length should match members"
        );
    }

    #[test]
    fn from_pairmi_vector_length_consistency() {
        let m = identical_matrix_3x10();
        let mut engine = PairMiEngine::create(&m).unwrap();
        engine.depth(2, EvaluationMode::Alpha(0.05));
        engine.depth(3, EvaluationMode::Alpha(0.05));

        let target = vec![1u8, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        let cell = from_pairmi(&engine, &target, 0.05, dummy_meta(CellSource::MiRestricted));

        let n = cell.members.len();
        assert!(n > 0, "should have retained sets");
        assert_eq!(cell.cprob.len(), n, "cprob length should match members");
        assert_eq!(
            cell.wilson_lower.len(),
            n,
            "wilson_lower length should match members"
        );
        assert_eq!(cell.n_set.len(), n, "n_set length should match members");
        assert_eq!(
            cell.n_set_y1.len(),
            n,
            "n_set_y1 length should match members"
        );
    }

    // ---- CellSource enum ----

    #[test]
    fn cell_source_enum_variants() {
        assert_eq!(CellSource::Exhaustive, CellSource::Exhaustive);
        assert_eq!(CellSource::MiRestricted, CellSource::MiRestricted);
        assert_ne!(CellSource::Exhaustive, CellSource::MiRestricted);
    }

    // ---- Meta propagation ----

    #[test]
    fn from_exhaustive_meta_propagated() {
        let empty_output = CellOutput {
            rows: vec![],
            n_sufficient: 0,
        };
        let meta = CellMeta {
            source: CellSource::Exhaustive,
            alpha: Some(0.01),
            benchmark: "bench42".to_string(),
            bench_label: "Benchmark 42".to_string(),
            criterion: Criterion::Point,
            threshold: 0.5,
            key: "target0_theta0.5_point".to_string(),
        };
        let cell = from_exhaustive(&empty_output, meta);

        assert_eq!(cell.meta.source, CellSource::Exhaustive);
        assert_eq!(cell.meta.alpha, Some(0.01));
        assert_eq!(cell.meta.benchmark, "bench42");
        assert_eq!(cell.meta.bench_label, "Benchmark 42");
        assert_eq!(cell.meta.criterion, Criterion::Point);
        assert!((cell.meta.threshold - 0.5).abs() < 1e-15);
        assert_eq!(cell.meta.key, "target0_theta0.5_point");
    }

    #[test]
    fn from_pairmi_meta_propagated() {
        let m = identical_matrix_3x10();
        let engine = PairMiEngine::create(&m).unwrap();
        let target = vec![1u8; 10];
        let meta = CellMeta {
            source: CellSource::MiRestricted,
            alpha: Some(0.05),
            benchmark: "bench99".to_string(),
            bench_label: "Benchmark 99".to_string(),
            criterion: Criterion::Wilson,
            threshold: 0.3,
            key: "target1_theta0.3_wilson".to_string(),
        };
        let cell = from_pairmi(&engine, &target, 0.05, meta);

        assert_eq!(cell.meta.source, CellSource::MiRestricted);
        assert_eq!(cell.meta.alpha, Some(0.05));
        assert_eq!(cell.meta.benchmark, "bench99");
        assert_eq!(cell.meta.bench_label, "Benchmark 99");
        assert_eq!(cell.meta.criterion, Criterion::Wilson);
        assert!((cell.meta.threshold - 0.3).abs() < 1e-15);
        assert_eq!(cell.meta.key, "target1_theta0.3_wilson");
    }
}
