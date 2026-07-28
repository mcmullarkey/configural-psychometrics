//! Coverage: share of cases satisfying at least one antichain member.
//!
//! Ports `coverage` from `coverage_cross_cluster.R`.
//!
//! ## What this module provides
//!
//! - [`CoverageResult`] — coverage counts and shares (overall + y1-stratified)
//! - [`coverage`] — compute coverage from a [`BinaryMatrix`], antichain
//!   members, and a binary target vector
//!
//! ## What this module does NOT provide
//!
//! - Satisfaction matrix construction (see [`crate::satisfaction`])
//! - Antichain construction (see [`crate::cell_eval`], [`crate::exhaustive`])
//! - File I/O or serialization
//!
//! ## Orientation
//!
//! Coverage is the share of rows (variables) that satisfy at least one
//! antichain member. A row satisfies a member iff every column in the member
//! is set in that row (see [`crate::satisfaction::satisfaction_matrix`]).
//! The target vector stratifies coverage: `covered_share_y1` is the share of
//! target-1 rows that are covered, or `None` when there are no target-1 rows
//! (avoids division by zero).

use crate::matrix::BinaryMatrix;
use crate::satisfaction::satisfaction_matrix;

/// Coverage result: which rows are covered + counts + shares.
#[derive(Debug, Clone)]
pub struct CoverageResult {
    /// Per-row coverage flag: `covered[p] = true` iff row p satisfies ≥1 member.
    pub covered: Vec<bool>,
    /// Number of covered rows.
    pub n_covered: usize,
    /// Number of covered rows with `target == 1`.
    pub n_covered_y1: usize,
    /// `n_covered / n_rows` (0.0 when n_rows == 0).
    pub covered_share: f64,
    /// `n_covered_y1 / n_y1`, or `None` when `n_y1 == 0` (no div-by-zero).
    pub covered_share_y1: Option<f64>,
}

/// Compute coverage from matrix, antichain members, and target vector.
///
/// # Panics
///
/// Panics if `target.len() != matrix.n_rows()`.
pub fn coverage(matrix: &BinaryMatrix, members: &[Vec<u32>], target: &[u8]) -> CoverageResult {
    assert_eq!(
        target.len(),
        matrix.n_rows(),
        "target length must match matrix rows"
    );

    let s = satisfaction_matrix(matrix, members);
    let n_rows = matrix.n_rows();

    let covered: Vec<bool> = s.iter().map(|row| row.iter().any(|&x| x)).collect();

    let n_covered = covered.iter().filter(|&&c| c).count();
    let n_y1 = target.iter().filter(|&&t| t == 1).count();
    let n_covered_y1 = covered
        .iter()
        .zip(target.iter())
        .filter(|(&c, &t)| c && t == 1)
        .count();

    let covered_share = if n_rows > 0 {
        n_covered as f64 / n_rows as f64
    } else {
        0.0
    };
    let covered_share_y1 = if n_y1 > 0 {
        Some(n_covered_y1 as f64 / n_y1 as f64)
    } else {
        None
    };

    CoverageResult {
        covered,
        n_covered,
        n_covered_y1,
        covered_share,
        covered_share_y1,
    }
}

#[cfg(test)]
mod tests {
    use crate::coverage::coverage;
    use crate::matrix::BinaryMatrix;

    /// 5×3 matrix (5 variables, 3 observations):
    ///
    /// ```text
    ///         obs0  obs1  obs2
    /// var0:    1     1     0
    /// var1:    1     0     1
    /// var2:    0     1     1
    /// var3:    1     1     1
    /// var4:    0     0     0
    /// ```
    fn sample_matrix() -> BinaryMatrix {
        let data = [
            1.0, 1.0, 0.0, // var0
            1.0, 0.0, 1.0, // var1
            0.0, 1.0, 1.0, // var2
            1.0, 1.0, 1.0, // var3
            0.0, 0.0, 0.0, // var4
        ];
        BinaryMatrix::new(&data, 5, 3).expect("valid 5x3 matrix")
    }

    #[test]
    fn coverage_counts_and_shares() {
        // members: {0,1} covers var0,var3; {1,2} covers var2,var3
        // union covered: var0, var2, var3 → n_covered = 3
        // target = [1,0,1,1,0] → n_y1 = 3 (var0,var2,var3)
        // covered & y1: var0,var2,var3 all covered & target=1 → n_covered_y1 = 3
        // covered_share = 3/5 = 0.6
        // covered_share_y1 = 3/3 = 1.0
        let m = sample_matrix();
        let members = vec![vec![0, 1], vec![1, 2]];
        let target = [1u8, 0, 1, 1, 0];

        let result = coverage(&m, &members, &target);

        assert_eq!(result.n_covered, 3, "n_covered");
        assert_eq!(result.n_covered_y1, 3, "n_covered_y1");
        assert!(
            (result.covered_share - 0.6).abs() < f64::EPSILON,
            "covered_share"
        );
        assert_eq!(result.covered_share_y1, Some(1.0), "covered_share_y1");

        // Per-row covered flags
        assert!(result.covered[0], "var0 covered");
        assert!(!result.covered[1], "var1 not covered");
        assert!(result.covered[2], "var2 covered");
        assert!(result.covered[3], "var3 covered");
        assert!(!result.covered[4], "var4 not covered");
    }

    #[test]
    fn coverage_y1_zero_returns_none() {
        // target all zeros → n_y1 == 0 → covered_share_y1 == None
        let m = sample_matrix();
        let members = vec![vec![0, 1]];
        let target = [0u8, 0, 0, 0, 0];

        let result = coverage(&m, &members, &target);

        assert_eq!(result.n_covered_y1, 0, "no y1 covered");
        assert_eq!(result.covered_share_y1, None, "None when n_y1 == 0");
        // covered_share still well-defined
        // member {0,1} covers var0, var3 → 2/5
        assert!(
            (result.covered_share - 0.4).abs() < f64::EPSILON,
            "covered_share"
        );
    }

    #[test]
    fn coverage_empty_members() {
        // No members → nothing covered.
        let m = sample_matrix();
        let members: Vec<Vec<u32>> = vec![];
        let target = [1u8, 0, 1, 1, 0];

        let result = coverage(&m, &members, &target);

        assert_eq!(result.n_covered, 0, "n_covered == 0 with no members");
        assert!(
            (result.covered_share - 0.0).abs() < f64::EPSILON,
            "covered_share == 0.0"
        );
        assert_eq!(result.n_covered_y1, 0, "n_covered_y1 == 0");
        // n_y1 = 3 > 0 so covered_share_y1 is Some(0.0), not None
        assert_eq!(
            result.covered_share_y1,
            Some(0.0),
            "covered_share_y1 Some(0.0)"
        );
    }

    #[test]
    fn coverage_result_is_debug_clone() {
        // CoverageResult must be Debug + Clone for downstream ACs.
        let m = sample_matrix();
        let members = vec![vec![0]];
        let target = [1u8, 0, 0, 0, 0];
        let result = coverage(&m, &members, &target);
        let cloned = result.clone();
        assert_eq!(
            result.n_covered, cloned.n_covered,
            "clone preserves n_covered"
        );
        let _ = format!("{result:?}"); // Debug
    }

    #[test]
    #[should_panic(expected = "target length must match matrix rows")]
    fn coverage_target_length_mismatch_panics() {
        let m = sample_matrix();
        let members = vec![vec![0]];
        let target = [1u8, 0]; // length 2, not 5
        let _ = coverage(&m, &members, &target);
    }
}
