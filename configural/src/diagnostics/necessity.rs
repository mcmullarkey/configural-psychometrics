//! Necessity diagnostics: per-element proportion across antichain members.

use crate::cell::Cell;

#[derive(Debug, Clone)]
pub struct NecessityIndexRow {
    pub element_idx: u32,
    pub element_name: String,
    pub n_antichain: usize,
    pub count: usize,
    pub proportion: Option<f64>, // None = empty antichain, Some(0.0) = absent
}

/// Compute necessity index for one cell.
/// ν(s) = |{S ∈ frontier : s ∈ S}| / |frontier|
/// Proportion rounded to 6 decimals. None if antichain empty.
pub fn necessity_index(cell: &Cell, element_names: &[&str]) -> Vec<NecessityIndexRow> {
    let n_ac = cell.members.len();
    let n_elements = element_names.len();
    let mut rows = Vec::with_capacity(n_elements);

    for (idx, name) in element_names.iter().enumerate() {
        let count = cell
            .members
            .iter()
            .filter(|member| member.contains(&(idx as u32)))
            .count();
        let proportion = if n_ac > 0 {
            let raw = count as f64 / n_ac as f64;
            // Round to 6 decimals
            Some((raw * 1e6).round() / 1e6)
        } else {
            None
        };
        rows.push(NecessityIndexRow {
            element_idx: idx as u32,
            element_name: name.to_string(),
            n_antichain: n_ac,
            count,
            proportion,
        });
    }
    rows
}

#[derive(Debug, Clone)]
pub struct NecessitySummaryRow {
    pub n_antichain: usize,
    pub necessary: Vec<String>, // elements where proportion == Some(1.0)
    pub near_necessary: Vec<(String, f64)>, // elements where threshold <= proportion < 1.0
}

/// Summarize necessity: classify elements as necessary (ν==1.0) or near-necessary (threshold <= ν < 1.0).
/// Thresholds must satisfy 0 < t < 1.
pub fn necessity_summary(index: &[NecessityIndexRow], threshold: f64) -> NecessitySummaryRow {
    assert!(
        threshold > 0.0 && threshold < 1.0,
        "threshold must be in (0, 1)"
    );

    let n_ac = index.first().map(|r| r.n_antichain).unwrap_or(0);
    let mut necessary = Vec::new();
    let mut near_necessary = Vec::new();

    for row in index {
        match row.proportion {
            None => {} // undefined, skip
            Some(1.0) => {
                necessary.push(row.element_name.clone());
            }
            Some(p) if p >= threshold => {
                near_necessary.push((row.element_name.clone(), p));
            }
            _ => {}
        }
    }

    NecessitySummaryRow {
        n_antichain: n_ac,
        necessary,
        near_necessary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{Cell, CellMeta, CellSource};
    use crate::cell_eval::Criterion;

    /// Build a minimal Cell with only members populated (necessity_index
    /// only reads `cell.members`).
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

    // ---- necessity_index ----

    #[test]
    fn empty_antichain_proportion_none() {
        let cell = cell_with_members(vec![]);
        let names = ["a", "b", "c"];
        let rows = necessity_index(&cell, &names);

        assert_eq!(rows.len(), 3);
        for row in &rows {
            assert_eq!(row.n_antichain, 0);
            assert_eq!(row.count, 0);
            assert!(
                row.proportion.is_none(),
                "proportion must be None for empty antichain"
            );
        }
    }

    #[test]
    fn single_member_proportion_one_or_zero() {
        // One member: {0, 2} → element 0 present, 1 absent, 2 present
        let cell = cell_with_members(vec![vec![0, 2]]);
        let names = ["a", "b", "c"];
        let rows = necessity_index(&cell, &names);

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].count, 1);
        assert_eq!(rows[0].proportion, Some(1.0));
        assert_eq!(rows[1].count, 0);
        assert_eq!(rows[1].proportion, Some(0.0));
        assert_eq!(rows[2].count, 1);
        assert_eq!(rows[2].proportion, Some(1.0));
    }

    #[test]
    fn multi_member_correct_counts_and_proportions() {
        // 3 members: {0,1}, {0,2}, {1,2}
        // element 0: in 2/3, element 1: in 2/3, element 2: in 2/3
        let cell = cell_with_members(vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
        let names = ["a", "b", "c"];
        let rows = necessity_index(&cell, &names);

        assert_eq!(rows.len(), 3);
        for row in &rows {
            assert_eq!(row.n_antichain, 3);
            assert_eq!(row.count, 2);
        }
        // 2/3 = 0.666667 rounded to 6 decimals
        assert_eq!(rows[0].proportion, Some(0.666667));
        assert_eq!(rows[1].proportion, Some(0.666667));
        assert_eq!(rows[2].proportion, Some(0.666667));
    }

    #[test]
    fn rounding_to_six_decimals() {
        // 1/3 = 0.333333... → rounds to 0.333333
        let cell = cell_with_members(vec![vec![0]]);
        let names = ["a", "b", "c"];
        let rows = necessity_index(&cell, &names);

        // element 0: 1/1 = 1.0
        assert_eq!(rows[0].proportion, Some(1.0));
        // element 1: 0/1 = 0.0
        assert_eq!(rows[1].proportion, Some(0.0));

        // Now test 1/3 with 3 members, only 1 containing element 0
        let cell2 = cell_with_members(vec![vec![0, 1], vec![1, 2], vec![1, 2]]);
        let rows2 = necessity_index(&cell2, &names);
        // element 0: 1/3 = 0.333333...
        assert_eq!(rows2[0].proportion, Some(0.333333));
        // element 1: 3/3 = 1.0
        assert_eq!(rows2[1].proportion, Some(1.0));
        // element 2: 2/3 = 0.666667
        assert_eq!(rows2[2].proportion, Some(0.666667));
    }

    // ---- necessity_summary ----

    #[test]
    fn summary_necessary_and_near_necessary() {
        // 4 members: {0,1}, {0,1}, {0,2}, {2,3}
        // element 0: 3/4 = 0.75 (near-necessary at threshold 0.5)
        // element 1: 2/4 = 0.5  (near-necessary at threshold 0.5)
        // element 2: 2/4 = 0.5  (near-necessary at threshold 0.5)
        // element 3: 1/4 = 0.25 (below threshold 0.5)
        let cell = cell_with_members(vec![vec![0, 1], vec![0, 1], vec![0, 2], vec![2, 3]]);
        let names = ["a", "b", "c", "d"];
        let index = necessity_index(&cell, &names);
        let summary = necessity_summary(&index, 0.5);

        assert_eq!(summary.n_antichain, 4);
        assert!(summary.necessary.is_empty(), "no element at ν==1.0");

        let near_names: Vec<&str> = summary
            .near_necessary
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(near_names.contains(&"a"));
        assert!(near_names.contains(&"b"));
        assert!(near_names.contains(&"c"));
        assert!(!near_names.contains(&"d"), "element d below threshold");

        // Check proportions in near_necessary
        for (name, p) in &summary.near_necessary {
            if name == "a" {
                assert_eq!(*p, 0.75);
            }
        }
    }

    #[test]
    fn summary_necessary_not_in_near_necessary() {
        // 2 members: {0,1}, {0,1} → element 0: 2/2 = 1.0 (necessary)
        // element 1: 2/2 = 1.0 (necessary)
        // element 2: 0/2 = 0.0 (neither)
        let cell = cell_with_members(vec![vec![0, 1], vec![0, 1]]);
        let names = ["a", "b", "c"];
        let index = necessity_index(&cell, &names);
        let summary = necessity_summary(&index, 0.5);

        assert_eq!(summary.n_antichain, 2);
        assert_eq!(summary.necessary.len(), 2);
        assert!(summary.necessary.contains(&"a".to_string()));
        assert!(summary.necessary.contains(&"b".to_string()));

        // ν==1.0 must NOT appear in near_necessary
        assert!(
            summary.near_necessary.is_empty(),
            "ν==1.0 elements must not be in near_necessary"
        );
    }

    #[test]
    #[should_panic(expected = "threshold must be in (0, 1)")]
    fn summary_threshold_zero_panics() {
        let cell = cell_with_members(vec![vec![0]]);
        let names = ["a"];
        let index = necessity_index(&cell, &names);
        necessity_summary(&index, 0.0);
    }

    #[test]
    #[should_panic(expected = "threshold must be in (0, 1)")]
    fn summary_threshold_one_panics() {
        let cell = cell_with_members(vec![vec![0]]);
        let names = ["a"];
        let index = necessity_index(&cell, &names);
        necessity_summary(&index, 1.0);
    }

    #[test]
    fn summary_empty_antichain() {
        let cell = cell_with_members(vec![]);
        let names = ["a", "b"];
        let index = necessity_index(&cell, &names);
        let summary = necessity_summary(&index, 0.5);

        assert_eq!(summary.n_antichain, 0);
        assert!(summary.necessary.is_empty());
        assert!(summary.near_necessary.is_empty());
    }
}
