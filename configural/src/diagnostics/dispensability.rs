//! Dispensability index: δ(s) = 1 − n_orphan(s) / n_covered.
//!
//! Ports `compute_dispensability_index` from `dispensability_diagnostics.R`.
//!
//! ## What this module provides
//!
//! - [`DispensabilityIndexRow`] — per-element orphan counts, δ, and y1-restricted variant
//! - [`dispensability_index`] — compute dispensability for one cell
//!
//! ## What this module does NOT provide
//!
//! - Necessity index (see [`crate::diagnostics::necessity`])
//! - Coverage aggregation (see [`crate::coverage`])
//! - Leave-one-symptom-out re-derivation (audit function, not ported)
//! - File I/O or serialization
//!
//! ## Orientation
//!
//! Dispensability measures how replaceable an element is within the fixed
//! antichain. An element is **orphaned** when a covered case is covered
//! *only* by members containing that element — removing the element would
//! leave that case uncovered.
//!
//! - δ = 0.0 → element in ALL members; every covered case depends on it
//!   (not dispensable)
//! - δ = 1.0 → element in NO members; removing it changes nothing
//!   (fully dispensable)
//! - δ = None → no covered cases (n_covered == 0)
//!
//! The y1-restricted variant (`delta_y1`) applies the same formula to
//! target-1 cases only.

use crate::cell::Cell;
use crate::matrix::BinaryMatrix;
use crate::satisfaction::satisfaction_matrix;

/// One row of the dispensability index: per-element orphan counts and δ.
#[derive(Debug, Clone)]
pub struct DispensabilityIndexRow {
    /// 0-based element index.
    pub element_idx: u32,
    /// Human-readable element name.
    pub element_name: String,
    /// Number of antichain members containing this element.
    pub n_msc_with: usize,
    /// Total covered cases (same for all elements in one cell).
    pub n_covered: usize,
    /// Covered cases orphaned by removing this element.
    pub n_orphan: usize,
    /// δ(s) = 1 − n_orphan / n_covered. None when n_covered == 0.
    pub delta: Option<f64>,
    /// Covered cases with target == 1 (same for all elements in one cell).
    pub n_covered_y1: usize,
    /// Orphaned cases with target == 1.
    pub n_orphan_y1: usize,
    /// y1-restricted δ. None when n_covered_y1 == 0.
    pub delta_y1: Option<f64>,
}

/// Compute dispensability index for one cell.
///
/// δ(s) = 1 − n_orphan(s) / n_covered
///
/// - Element in ALL members → δ = 0.0 (not dispensable)
/// - Element in NO members → δ = 1.0 (fully dispensable)
/// - n_covered == 0 → δ = None
///
/// # Arguments
///
/// - `cell` — the cell (antichain of configurations)
/// - `matrix` — the binary data matrix
/// - `target` — binary target vector (length = `matrix.n_rows()`)
/// - `element_names` — names for each element index (length = number of
///   possible element indices)
///
/// # Panics
///
/// Panics if `target.len() != matrix.n_rows()`.
pub fn dispensability_index(
    cell: &Cell,
    matrix: &BinaryMatrix,
    target: &[u8],
    element_names: &[&str],
) -> Vec<DispensabilityIndexRow> {
    assert_eq!(
        target.len(),
        matrix.n_rows(),
        "target length must match matrix rows"
    );

    let n_rows = matrix.n_rows();
    let n_elements = element_names.len();
    let members = &cell.members;

    // Build satisfaction matrix S[p][k].
    let s = satisfaction_matrix(matrix, members);

    // covered[p] = any_k(S[p][k]).
    let covered: Vec<bool> = s.iter().map(|row| row.iter().any(|&x| x)).collect();
    let n_covered = covered.iter().filter(|&&c| c).count();

    // y1-restricted coverage.
    let n_covered_y1 = covered
        .iter()
        .zip(target.iter())
        .filter(|(&c, &t)| c && t == 1)
        .count();

    let mut rows = Vec::with_capacity(n_elements);

    for (idx, name) in element_names.iter().enumerate() {
        let idx_u32 = idx as u32;

        // Count how many members contain this element.
        let n_msc_with = members.iter().filter(|m| m.contains(&idx_u32)).count();

        // Find members NOT containing this element (without_cols).
        let without_cols: Vec<usize> = members
            .iter()
            .enumerate()
            .filter(|(_, m)| !m.contains(&idx_u32))
            .map(|(k, _)| k)
            .collect();

        // cov_without[p] = any_k(S[p][k] for k in without_cols).
        // orphan[p] = covered[p] && !cov_without[p].
        let mut n_orphan = 0;
        let mut n_orphan_y1 = 0;
        for p in 0..n_rows {
            if covered[p] {
                let cov_without = without_cols.iter().any(|&k| s[p][k]);
                if !cov_without {
                    n_orphan += 1;
                    if target[p] == 1 {
                        n_orphan_y1 += 1;
                    }
                }
            }
        }

        let delta = if n_covered > 0 {
            Some(1.0 - n_orphan as f64 / n_covered as f64)
        } else {
            None
        };
        let delta_y1 = if n_covered_y1 > 0 {
            Some(1.0 - n_orphan_y1 as f64 / n_covered_y1 as f64)
        } else {
            None
        };

        rows.push(DispensabilityIndexRow {
            element_idx: idx_u32,
            element_name: name.to_string(),
            n_msc_with,
            n_covered,
            n_orphan,
            delta,
            n_covered_y1,
            n_orphan_y1,
            delta_y1,
        });
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{Cell, CellMeta, CellSource};
    use crate::cell_eval::Criterion;
    use crate::matrix::BinaryMatrix;

    /// Build a minimal Cell with only members populated.
    fn cell_with_members(members: Vec<Vec<u32>>) -> Cell {
        let n = members.len();
        Cell {
            meta: CellMeta {
                source: CellSource::Exhaustive,
                alpha: Some(0.05),
                benchmark: "test".to_string(),
                bench_label: "test".to_string(),
                criterion: Criterion::Wilson,
                threshold: 0.3,
                key: "test".to_string(),
            },
            members,
            cprob: vec![0.0; n],
            wilson_lower: vec![0.0; n],
            n_set: vec![0; n],
            n_set_y1: vec![0; n],
        }
    }

    /// 3 vars × 4 obs:
    /// ```text
    ///         obs0  obs1  obs2  obs3
    /// var0:     1     1     0     0
    /// var1:     1     0     1     0
    /// var2:     0     1     1     1
    /// ```
    fn matrix_3x4() -> BinaryMatrix {
        let data = [
            1.0, 1.0, 0.0, 0.0, // var0
            1.0, 0.0, 1.0, 0.0, // var1
            0.0, 1.0, 1.0, 1.0, // var2
        ];
        BinaryMatrix::new(&data, 3, 4).unwrap()
    }

    /// 3 vars × 5 obs (obs4 added so element 4 can be in NO members):
    /// ```text
    ///         obs0  obs1  obs2  obs3  obs4
    /// var0:     1     1     0     0     1
    /// var1:     1     0     1     0     0
    /// var2:     0     1     1     1     0
    /// ```
    fn matrix_3x5() -> BinaryMatrix {
        let data = [
            1.0, 1.0, 0.0, 0.0, 1.0, // var0
            1.0, 0.0, 1.0, 0.0, 0.0, // var1
            0.0, 1.0, 1.0, 1.0, 0.0, // var2
        ];
        BinaryMatrix::new(&data, 3, 5).unwrap()
    }

    // ---- Element in ALL members → δ = 0.0 ----

    #[test]
    fn element_in_all_members_delta_zero() {
        // Members: {0,1}, {0,2} — element 0 is in ALL members.
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1], vec![0, 2]]);
        let target = vec![1u8, 0, 1];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        // Element 0 is in both members.
        assert_eq!(rows[0].n_msc_with, 2, "element 0 in both members");
        assert_eq!(
            rows[0].n_orphan, rows[0].n_covered,
            "all covered cases orphaned"
        );
        assert_eq!(
            rows[0].delta,
            Some(0.0),
            "δ=0.0 when element in ALL members"
        );
    }

    // ---- Element in NO members → δ = 1.0 ----

    #[test]
    fn element_in_no_members_delta_one() {
        // Members: {0,1}, {2,3} — element 4 is in NO members.
        let m = matrix_3x5();
        let cell = cell_with_members(vec![vec![0, 1], vec![2, 3]]);
        let target = vec![1u8, 0, 1];
        let names = ["obs0", "obs1", "obs2", "obs3", "obs4"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        // Element 4 is in no members.
        assert_eq!(rows[4].n_msc_with, 0, "element 4 in no members");
        assert_eq!(rows[4].n_orphan, 0, "no orphans for absent element");
        assert_eq!(rows[4].delta, Some(1.0), "δ=1.0 when element in NO members");
    }

    // ---- Empty antichain → δ = None, all counts zero ----

    #[test]
    fn empty_antichain_delta_none() {
        let m = matrix_3x4();
        let cell = cell_with_members(vec![]);
        let target = vec![1u8, 0, 1];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        assert_eq!(rows.len(), 4);
        for row in &rows {
            assert_eq!(row.n_msc_with, 0);
            assert_eq!(row.n_covered, 0);
            assert_eq!(row.n_orphan, 0);
            assert!(
                row.delta.is_none(),
                "delta must be None for empty antichain"
            );
            assert_eq!(row.n_covered_y1, 0);
            assert_eq!(row.n_orphan_y1, 0);
            assert!(
                row.delta_y1.is_none(),
                "delta_y1 must be None for empty antichain"
            );
        }
    }

    // ---- n_covered == 0 → δ = None ----

    #[test]
    fn no_coverage_delta_none() {
        // Member {obs3} — no variable has obs3=1 except var2.
        // Actually var2 has obs3=1, so this covers var2.
        // Use a member that no variable satisfies: {obs1, obs2} requires
        // both obs1=1 and obs2=1 in the same row. var0: obs1=1,obs2=0 → no.
        // var1: obs1=0,obs2=1 → no. var2: obs1=1,obs2=1 → yes.
        // So {obs1,obs2} covers var2. Let me use {obs0, obs3} instead.
        // var0: obs0=1,obs3=0 → no. var1: obs0=1,obs3=0 → no.
        // var2: obs0=0,obs3=1 → no. → n_covered = 0.
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 3]]);
        let target = vec![1u8, 0, 1];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        for row in &rows {
            assert_eq!(row.n_covered, 0, "no variable covered by {{obs0,obs3}}");
            assert!(
                row.delta.is_none(),
                "delta must be None when n_covered == 0"
            );
        }
    }

    // ---- n_covered_y1 == 0 → delta_y1 = None ----

    #[test]
    fn no_y1_coverage_delta_y1_none() {
        // Use target all zeros → n_covered_y1 == 0.
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1], vec![2, 3]]);
        let target = vec![0u8, 0, 0];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        // n_covered > 0 (var0 and var2 covered), so delta is Some.
        assert!(rows[0].n_covered > 0, "some variables covered");
        assert!(rows[0].delta.is_some(), "delta defined when n_covered > 0");
        // But n_covered_y1 == 0 → delta_y1 == None.
        assert_eq!(rows[0].n_covered_y1, 0, "no y1 covered");
        assert!(
            rows[0].delta_y1.is_none(),
            "delta_y1 must be None when n_covered_y1 == 0"
        );
    }

    // ---- Partial membership → 0.0 < δ < 1.0 ----

    #[test]
    fn partial_membership_delta_between_zero_and_one() {
        // Members: {0,1}, {2,3} — each element in exactly 1 of 2 members.
        // var0 covered by member0 only, var2 covered by member1 only.
        // Element 0: without_cols = [1], cov_without[var0] = S[var0][1] = false
        //   → orphan[var0] = true. cov_without[var2] = S[var2][1] = true
        //   → orphan[var2] = false. n_orphan = 1, n_covered = 2, δ = 0.5.
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1], vec![2, 3]]);
        let target = vec![1u8, 0, 1];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        // Element 0: in member0 only.
        assert_eq!(rows[0].n_msc_with, 1);
        assert_eq!(rows[0].n_covered, 2);
        assert_eq!(rows[0].n_orphan, 1, "var0 orphaned by removing obs0");
        assert_eq!(rows[0].delta, Some(0.5));

        // Element 2: in member1 only.
        assert_eq!(rows[2].n_msc_with, 1);
        assert_eq!(rows[2].n_orphan, 1, "var2 orphaned by removing obs2");
        assert_eq!(rows[2].delta, Some(0.5));

        // All partial deltas strictly between 0 and 1.
        for row in &rows {
            if let Some(d) = row.delta {
                assert!(d > 0.0 && d < 1.0, "partial delta in (0,1), got {d}");
            }
        }
    }

    // ---- y1-restricted variant ----

    #[test]
    fn y1_restricted_delta() {
        // Members: {0,1}, {2,3}. target = [1, 0, 1].
        // covered = [true, false, true] (var0, var2).
        // n_covered = 2, n_covered_y1 = 2 (both var0 and var2 have target=1).
        // Element 0: orphan = {var0} → n_orphan=1, n_orphan_y1=1 (var0 target=1).
        //   delta = 1 - 1/2 = 0.5, delta_y1 = 1 - 1/2 = 0.5.
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1], vec![2, 3]]);
        let target = vec![1u8, 0, 1];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        assert_eq!(rows[0].n_covered_y1, 2);
        assert_eq!(rows[0].n_orphan_y1, 1, "var0 is y1 and orphaned");
        assert_eq!(rows[0].delta_y1, Some(0.5));
    }

    #[test]
    fn y1_restricted_differs_from_overall_when_target_skewed() {
        // target = [1, 0, 0] — only var0 is y1.
        // covered = [true, false, true] (var0, var2).
        // n_covered = 2, n_covered_y1 = 1 (only var0).
        // Element 0: orphan = {var0} → n_orphan=1, n_orphan_y1=1.
        //   delta = 1 - 1/2 = 0.5, delta_y1 = 1 - 1/1 = 0.0.
        // Element 2: orphan = {var2} → n_orphan=1, n_orphan_y1=0 (var2 not y1).
        //   delta = 1 - 1/2 = 0.5, delta_y1 = 1 - 0/1 = 1.0.
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1], vec![2, 3]]);
        let target = vec![1u8, 0, 0];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        // Element 0: orphans var0 (y1), so delta_y1 = 0.0 (not dispensable for y1).
        assert_eq!(rows[0].n_orphan_y1, 1);
        assert_eq!(rows[0].delta_y1, Some(0.0));

        // Element 2: orphans var2 (not y1), so delta_y1 = 1.0 (dispensable for y1).
        assert_eq!(rows[2].n_orphan_y1, 0);
        assert_eq!(rows[2].delta_y1, Some(1.0));
    }

    // ---- Debug + Clone on DispensabilityIndexRow ----

    #[test]
    fn row_is_debug_clone() {
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1]]);
        let target = vec![1u8, 0, 1];
        let names = ["obs0", "obs1", "obs2", "obs3"];
        let rows = dispensability_index(&cell, &m, &target, &names);

        let cloned = rows[0].clone();
        assert_eq!(rows[0].element_idx, cloned.element_idx);
        let _ = format!("{:?}", rows[0]); // Debug
    }

    // ---- Target length mismatch panics ----

    #[test]
    #[should_panic(expected = "target length must match matrix rows")]
    fn target_length_mismatch_panics() {
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1]]);
        let target = vec![1u8, 0]; // length 2, not 3
        let names = ["obs0", "obs1", "obs2", "obs3"];
        let _ = dispensability_index(&cell, &m, &target, &names);
    }

    // ---- Single member covering one variable ----

    #[test]
    fn single_member_single_coverage() {
        // Member {0,1} covers var0 only (var0 has obs0=1, obs1=1).
        // var1: obs0=1, obs1=0 → no. var2: obs0=0, obs1=1 → no.
        // n_covered = 1.
        // Element 0: in the only member → without_cols = [] → all covered orphaned.
        //   n_orphan = 1, delta = 1 - 1/1 = 0.0.
        // Element 2: NOT in the only member → without_cols = [0] → cov_without = covered.
        //   n_orphan = 0, delta = 1 - 0/1 = 1.0.
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1]]);
        let target = vec![1u8, 0, 0];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        assert_eq!(rows[0].n_covered, 1);
        assert_eq!(rows[0].n_msc_with, 1);
        assert_eq!(rows[0].n_orphan, 1);
        assert_eq!(rows[0].delta, Some(0.0), "element in sole member → δ=0.0");

        assert_eq!(rows[2].n_msc_with, 0);
        assert_eq!(rows[2].n_orphan, 0);
        assert_eq!(
            rows[2].delta,
            Some(1.0),
            "element not in sole member → δ=1.0"
        );
    }

    // ---- R parity: exact orphan counts matching R reference logic ----

    #[test]
    fn r_parity_orphan_counts() {
        // Replicate the R logic exactly:
        // S <- satisfaction_matrix(X, members)
        // covered <- rowSums(S) > 0
        // B[s,k] = 1 iff s in member k
        // without_cols = which(B[s,] == 0)
        // cov_without = rowSums(S[, without_cols]) > 0
        // orphan = covered & !cov_without
        //
        // Matrix 3x4, members {0,1}, {2,3}:
        // S[var0] = [true, false]  (var0 satisfies {0,1} but not {2,3})
        // S[var1] = [false, false] (var1 satisfies neither)
        // S[var2] = [false, true]  (var2 satisfies {2,3} but not {0,1})
        // covered = [true, false, true]
        //
        // Element 0 (in member 0 only):
        //   without_cols = [1]
        //   cov_without = [S[var0][1], S[var1][1], S[var2][1]] = [false, false, true]
        //   orphan = [true & !false, false & ..., true & !true] = [true, false, false]
        //   n_orphan = 1
        //
        // Element 1 (in member 0 only): same as element 0 → n_orphan = 1
        //
        // Element 2 (in member 1 only):
        //   without_cols = [0]
        //   cov_without = [S[var0][0], S[var1][0], S[var2][0]] = [true, false, false]
        //   orphan = [true & !true, false, true & !false] = [false, false, true]
        //   n_orphan = 1
        //
        // Element 3 (in member 1 only): same as element 2 → n_orphan = 1
        let m = matrix_3x4();
        let cell = cell_with_members(vec![vec![0, 1], vec![2, 3]]);
        let target = vec![1u8, 1, 1];
        let names = ["obs0", "obs1", "obs2", "obs3"];

        let rows = dispensability_index(&cell, &m, &target, &names);

        assert_eq!(rows[0].n_orphan, 1, "element 0 orphans var0");
        assert_eq!(rows[1].n_orphan, 1, "element 1 orphans var0");
        assert_eq!(rows[2].n_orphan, 1, "element 2 orphans var2");
        assert_eq!(rows[3].n_orphan, 1, "element 3 orphans var2");

        // All deltas = 1 - 1/2 = 0.5
        for row in &rows {
            assert_eq!(row.delta, Some(0.5));
        }

        // y1: all target=1, so n_covered_y1 = 2, n_orphan_y1 = 1 for all.
        for row in &rows {
            assert_eq!(row.n_covered_y1, 2);
            assert_eq!(row.n_orphan_y1, 1);
            assert_eq!(row.delta_y1, Some(0.5));
        }
    }
}
