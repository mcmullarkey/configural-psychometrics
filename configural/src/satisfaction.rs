//! Satisfaction matrix primitive: S[p][k] = true iff all elements in member k present in row p.
//!
//! Ports `satisfaction_matrix` from `coverage_cross_cluster.R` /
//! `dispensability_diagnostics.R`.
//!
//! ## What this module provides
//!
//! - [`satisfaction_matrix`] — build the boolean satisfaction matrix from a
//!   [`BinaryMatrix`] and a slice of antichain members.
//!
//! ## What this module does NOT provide
//!
//! - Coverage aggregation (see [`crate::coverage`])
//! - Antichain construction (see [`crate::cell_eval`], [`crate::exhaustive`])
//! - File I/O or serialization
//!
//! ## Orientation
//!
//! [`BinaryMatrix`] stores variables as rows (`n_rows = V`) and observations
//! as columns (`n_cols = N`). A member is a set of column indices
//! (observations). `S[p][k]` is `true` iff every column in `members[k]` is
//! set in row `p` — i.e. variable `p` co-occurs across all observations in
//! the member. An empty member is vacuously satisfied (all-true column).

use crate::matrix::BinaryMatrix;

/// Build satisfaction matrix. S[p][k] = true iff all elements in members[k] are present in row p.
/// Empty member → vacuously true (all-true column).
pub fn satisfaction_matrix(matrix: &BinaryMatrix, members: &[Vec<u32>]) -> Vec<Vec<bool>> {
    let n_rows = matrix.n_rows();
    let mut s = vec![vec![false; members.len()]; n_rows];

    for (k, member) in members.iter().enumerate() {
        for (p, row) in s.iter_mut().enumerate() {
            // All elements in member must be present in row p
            row[k] = member.iter().all(|&col| matrix.get(p, col as usize));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use crate::matrix::BinaryMatrix;
    use crate::satisfaction::satisfaction_matrix;

    /// Build a 5×3 matrix (5 variables, 3 observations):
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
    fn elementwise_satisfaction_two_members() {
        // member0 = {obs0, obs1} → S[p][0] = true iff var p has 1 in both obs0 and obs1
        //   var0: 1&1 = 1, var1: 1&0 = 0, var2: 0&1 = 0, var3: 1&1 = 1, var4: 0&0 = 0
        // member1 = {obs1, obs2} → S[p][1] = true iff var p has 1 in both obs1 and obs2
        //   var0: 1&0 = 0, var1: 0&1 = 0, var2: 1&1 = 1, var3: 1&1 = 1, var4: 0&0 = 0
        let m = sample_matrix();
        let members = vec![vec![0, 1], vec![1, 2]];
        let s = satisfaction_matrix(&m, &members);

        assert_eq!(s.len(), 5, "5 rows");
        assert_eq!(s[0].len(), 2, "2 members");

        // member0 = {0,1}
        assert!(s[0][0], "var0 satisfies {{0,1}}");
        assert!(!s[1][0], "var1 does not satisfy {{0,1}}");
        assert!(!s[2][0], "var2 does not satisfy {{0,1}}");
        assert!(s[3][0], "var3 satisfies {{0,1}}");
        assert!(!s[4][0], "var4 does not satisfy {{0,1}}");

        // member1 = {1,2}
        assert!(!s[0][1], "var0 does not satisfy {{1,2}}");
        assert!(!s[1][1], "var1 does not satisfy {{1,2}}");
        assert!(s[2][1], "var2 satisfies {{1,2}}");
        assert!(s[3][1], "var3 satisfies {{1,2}}");
        assert!(!s[4][1], "var4 does not satisfy {{1,2}}");
    }

    #[test]
    fn single_element_member_parity() {
        // S[p][k] == X[p, member[0]] when member has exactly one element.
        let m = sample_matrix();
        let members = vec![vec![0], vec![1], vec![2]];
        let s = satisfaction_matrix(&m, &members);

        for (p, row) in s.iter().enumerate().take(5) {
            for (k, member) in members.iter().enumerate() {
                let col = member[0] as usize;
                assert_eq!(
                    row[k],
                    m.get(p, col),
                    "S[{p}][{k}] should equal X[{p}, {col}] for single-element member"
                );
            }
        }
    }

    #[test]
    fn empty_member_vacuous_truth() {
        // Empty member → vacuously true for every row (all-true column).
        let m = sample_matrix();
        let members = vec![vec![]];
        let s = satisfaction_matrix(&m, &members);

        assert_eq!(s.len(), 5);
        for (p, row) in s.iter().enumerate() {
            assert!(row[0], "empty member vacuously satisfied by var{p}");
        }
    }

    #[test]
    fn empty_members_list() {
        // No members → S has n_rows rows, each empty.
        let m = sample_matrix();
        let members: Vec<Vec<u32>> = vec![];
        let s = satisfaction_matrix(&m, &members);

        assert_eq!(s.len(), 5, "5 rows even with no members");
        for row in &s {
            assert!(row.is_empty(), "each row empty when no members");
        }
    }
}
