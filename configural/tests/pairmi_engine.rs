//! Integration tests for the constructive ascent engine.
//!
//! This file lives in `tests/` so it compiles as a separate crate that
//! imports `configural` exactly as a downstream consumer would. If a
//! `pub use` re-export is accidentally removed or renamed, this test
//! fails to compile — catching breakage that inline unit tests miss.

use configural::{BinaryMatrix, EvaluationMode, PairMiEngine};

/// Build a 3-variable × 10-observation matrix where every variable is
/// identical: `[1,1,1,1,1,0,0,0,0,0]`.
///
/// BinaryMatrix orientation: `n_rows = 3` (V = 3 variables),
/// `n_cols = 10` (N = 10 observations). Row-major input: row `r` is
/// variable `r`'s indicator across the 10 observations.
fn identical_column_matrix() -> BinaryMatrix {
    let row: [f64; 10] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let data: Vec<f64> = [row, row, row].into_iter().flatten().collect();
    BinaryMatrix::new(&data, 3, 10).expect("valid 3x10 matrix")
}

#[test]
fn depth_2_identical_columns() {
    let matrix = identical_column_matrix();
    let mut engine = PairMiEngine::create(&matrix).expect("engine created");

    // Depth 2: combn order (0,1),(0,2),(1,2) — 3 pairs, all identical.
    // n11 = 5, pop_i = pop_j = 5, N = 10 → MI = ln(2), p < 0.05.
    let stats = engine.depth(2, EvaluationMode::Alpha(0.05));
    assert_eq!(stats.raw, 3, "depth-2 raw candidates (combn pairs)");
    assert_eq!(stats.unique, 3, "depth-2 unique (raw == unique at depth 2)");
    assert_eq!(stats.kept, 3, "depth-2 retained (all significant)");
}

#[test]
fn depth_3_canonical_dedup() {
    let matrix = identical_column_matrix();
    let mut engine = PairMiEngine::create(&matrix).expect("engine created");
    engine.depth(2, EvaluationMode::Alpha(0.05));

    // Depth 3: triple {0,1,2} encountered 3 times (from each depth-2 parent),
    // but canonical-set dedup keeps only the first significant decomposition.
    let stats = engine.depth(3, EvaluationMode::Alpha(0.05));
    assert_eq!(
        stats.raw, 3,
        "depth-3 raw candidates (3 parent × 1 new var)"
    );
    assert_eq!(stats.unique, 1, "depth-3 unique canonical set {{0,1,2}}");
    assert_eq!(stats.kept, 1, "depth-3 retained (first significant)");
}

#[test]
fn results_mi_and_p_values() {
    let matrix = identical_column_matrix();
    let mut engine = PairMiEngine::create(&matrix).expect("engine created");
    engine.depth(2, EvaluationMode::Alpha(0.05));
    engine.depth(3, EvaluationMode::Alpha(0.05));

    let rows = engine.results();
    assert_eq!(rows.len(), 4, "3 depth-2 + 1 depth-3 = 4 total");

    let ln2 = 2.0f64.ln();

    // Depth-2 rows: MI = ln(2), p < 0.05, relmi = 0.
    for row in &rows[..3] {
        assert_eq!(row.depth, 2, "depth-2 row");
        assert!(
            (row.mi - ln2).abs() < 1e-10,
            "depth-2 MI should be ln(2) ≈ 0.693, got {}",
            row.mi
        );
        assert!(row.p < 0.05, "depth-2 p < 0.05, got {}", row.p);
        assert_eq!(row.relmi, 0.0, "depth-2 relmi = 0");
        assert_eq!(row.members.len(), 2, "depth-2 has 2 members");
    }

    // Depth-3 row: MI = ln(2), p < 0.05, relmi = mi - parent_mi = 0.
    let row3 = &rows[3];
    assert_eq!(row3.depth, 3, "depth-3 row");
    assert!(
        (row3.mi - ln2).abs() < 1e-10,
        "depth-3 MI should be ln(2), got {}",
        row3.mi
    );
    assert!(row3.p < 0.05, "depth-3 p < 0.05, got {}", row3.p);
    assert_eq!(row3.relmi, 0.0, "depth-3 relmi = mi - parent_mi = 0");
    assert_eq!(
        row3.members,
        vec![1, 2, 3],
        "depth-3 members 1-based ascending"
    );
}

#[test]
fn results_var_parent_1_based() {
    let matrix = identical_column_matrix();
    let mut engine = PairMiEngine::create(&matrix).expect("engine created");
    engine.depth(2, EvaluationMode::Alpha(0.05));

    let rows = engine.results();

    // combn order: (0,1), (0,2), (1,2) — 0-based.
    // 1-based: var = i+1, parent = j+1.
    // x1 = first (i), x2 = second (j).
    assert_eq!(rows[0].var, 1, "pair (0,1): var = 1 (0-based i=0 +1)");
    assert_eq!(rows[0].parent, 2, "pair (0,1): parent = 2 (0-based j=1 +1)");
    assert_eq!(rows[0].members, vec![1, 2]);

    assert_eq!(rows[1].var, 1, "pair (0,2): var = 1");
    assert_eq!(rows[1].parent, 3, "pair (0,2): parent = 3");
    assert_eq!(rows[1].members, vec![1, 3]);

    assert_eq!(rows[2].var, 2, "pair (1,2): var = 2");
    assert_eq!(rows[2].parent, 3, "pair (1,2): parent = 3");
    assert_eq!(rows[2].members, vec![2, 3]);
}

#[test]
fn indicators_unpack_conjunction() {
    let matrix = identical_column_matrix();
    let mut engine = PairMiEngine::create(&matrix).expect("engine created");
    engine.depth(2, EvaluationMode::Alpha(0.05));
    engine.depth(3, EvaluationMode::Alpha(0.05));

    let inds = engine.indicators();
    assert_eq!(inds.len(), 4, "4 retained sets");

    // Every conjunction of identical variables = the variable itself:
    // [1,1,1,1,1,0,0,0,0,0].
    let expected = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    for (i, ind) in inds.iter().enumerate() {
        assert_eq!(ind, &expected, "indicator {} should be conjunction", i);
    }
}

#[test]
fn mi_threshold_mode() {
    let matrix = identical_column_matrix();

    // ln(2) ≈ 0.693 > 0.5 → all 3 pairs retained.
    let mut engine = PairMiEngine::create(&matrix).expect("engine");
    let stats = engine.depth(2, EvaluationMode::MiThreshold(0.5));
    assert_eq!(stats.kept, 3, "MI > 0.5 retains all 3");

    // ln(2) ≈ 0.693 < 1.0 → none retained.
    let mut engine2 = PairMiEngine::create(&matrix).expect("engine");
    let stats2 = engine2.depth(2, EvaluationMode::MiThreshold(1.0));
    assert_eq!(stats2.kept, 0, "MI < 1.0 retains none");
}

#[test]
fn no_significant_pairs() {
    // Anti-correlated: var 0 = [1,0,1,0,...], var 1 = [0,1,0,1,...].
    // n11 = 0 → jc = 0 → g = 0 → p = 1.0 → not significant.
    let data = [
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // var 0
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // var 1
    ];
    let matrix = BinaryMatrix::new(&data, 2, 10).expect("valid 2x10");
    let mut engine = PairMiEngine::create(&matrix).expect("engine");
    let stats = engine.depth(2, EvaluationMode::Alpha(0.05));
    assert_eq!(stats.kept, 0, "n11=0 → g=0 → p=1.0 → not significant");
}

#[test]
fn depth_3_without_depth_2_returns_empty() {
    let matrix = identical_column_matrix();
    let mut engine = PairMiEngine::create(&matrix).expect("engine");
    // No depth-2 run → prev range empty → depth 3 returns (0, 0, 0).
    let stats = engine.depth(3, EvaluationMode::Alpha(0.05));
    assert_eq!(stats.raw, 0);
    assert_eq!(stats.unique, 0);
    assert_eq!(stats.kept, 0);
}

#[test]
fn depth_2_pop_counts() {
    let matrix = identical_column_matrix();
    let mut engine = PairMiEngine::create(&matrix).expect("engine");
    engine.depth(2, EvaluationMode::Alpha(0.05));

    let rows = engine.results();
    for row in &rows {
        assert_eq!(row.pop, 5, "conjunction of identical vars has popcount 5");
    }
}
