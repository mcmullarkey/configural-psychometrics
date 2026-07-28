//! Exhaustive lattice ascent via Apriori ordered extension.
//!
//! Ports `emsc_v4.cpp`: depth-1 admissibility screening, Apriori ordered
//! extension (j > highest_set_bit), downward-closure pruning (supp < n_min),
//! per-target y1 streaming, depth_end offsets, evaluability gating, and
//! early stop on max_admissible.
//!
//! ## What this module provides
//!
//! - [`EmscStore`] — append-only store of admissible configurations
//! - [`AscentResult`] — ascent output with admissibility/evaluability counts
//! - [`ascend`] — run the exhaustive lattice ascent
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
//! = one observation, packed into `ceil(V/64)` words). The ascent engine
//! packs each observation's indicator and enumerates subsets of observations
//! via Apriori ordered extension. A configuration's support = popcount of
//! the AND of all observation indicators in the set (number of variables
//! co-occurring across all selected observations).

use crate::bitset::{and_popcount, max_element, pack_column, popcount};
use crate::matrix::BinaryMatrix;
use crate::stats::z_score;
use rayon::prelude::*;

/// Store of all admissible configurations found during ascent.
pub struct EmscStore {
    /// Vocabulary masks (one `u64` per config), depth-major.
    pub cfg_mask: Vec<u64>,
    /// Support counts (popcount of conjunction), one per config.
    pub cfg_supp: Vec<u32>,
    /// Config-major y1 counts: `cfg_y1[g * n_targets + k]`.
    pub cfg_y1: Vec<u32>,
    /// Exclusive depth-end offsets, monotonic. `depth_end[d-1]` = end of depth d.
    pub depth_end: Vec<u32>,
    /// Per-target exclusion masks.
    pub excl_mask: Vec<u64>,
    /// Number of targets.
    pub n_targets: u32,
    /// Z-score from alpha.
    pub z: f64,
    /// Z-score squared (used in evaluability gate).
    pub z2: f64,
}

/// Result of the exhaustive ascent.
pub struct AscentResult {
    /// The configuration store.
    pub store: EmscStore,
    /// Admissible config count per depth (0-indexed: index 0 = depth 1).
    pub admissibility: Vec<u32>,
    /// Evaluability counts: `[theta][depth]`.
    pub evaluability: Vec<Vec<u32>>,
    /// Whether the ascent stopped early due to max_admissible.
    pub stopped_early: bool,
    /// Depth at which early stop occurred (if any).
    pub stop_depth: Option<u32>,
}

/// Per-parent extension results collected from parallel tasks.
///
/// Each parent config `i` independently extends with admissible observations.
/// Results are merged in parent order after the parallel loop to preserve
/// depth-major storage.
struct ParentExtension {
    /// Vocabulary masks for admissible configs from this parent.
    masks: Vec<u64>,
    /// Support counts for admissible configs from this parent.
    supps: Vec<u32>,
    /// Config-major y1 counts: `y1s[g * n_targets + k]`.
    y1s: Vec<u32>,
    /// Conjunction bitsets for the next depth (empty if `!keep_bits`).
    bits: Vec<u64>,
}

/// Run the exhaustive ascent.
///
/// Enumerates all admissible configurations via Apriori ordered extension.
/// Depth 1 screens individual observations for support >= n_min. Each
/// subsequent depth extends previous-depth configs with observations whose
/// index exceeds the highest set bit in the config's vocabulary mask
/// (ordered extension). Downward-closure pruning skips configs with
/// support < n_min. Evaluability is gated by `n / (n + z²) >= θ`.
///
/// # Arguments
///
/// - `matrix` — binary matrix (V = n_rows variables, N = n_cols observations)
/// - `targets` — target indicators, each a `Vec<u8>` of length V
/// - `thetas` — evaluability thresholds
/// - `n_min` — minimum support for admissibility
/// - `alpha` — significance level (for z-score)
/// - `max_cardinality` — maximum depth to explore
/// - `max_admissible_per_depth` — early stop threshold
/// - `excludes` — per-target exclusion masks (variable indices)
#[allow(clippy::too_many_arguments)]
pub fn ascend(
    matrix: &BinaryMatrix,
    targets: &[Vec<u8>],
    thetas: &[f64],
    n_min: u32,
    alpha: f64,
    max_cardinality: u32,
    max_admissible_per_depth: f64,
    excludes: &[Option<Vec<u32>>],
) -> AscentResult {
    let n_rows = matrix.n_rows();
    let n_cols = matrix.n_cols();
    let nwords = (n_rows + 63) / 64;
    let n_targets = targets.len() as u32;
    let n_thetas = thetas.len();
    let z = z_score(alpha);
    let z2 = z * z;

    // Step 1: Pack columns (observations) into bitsets.
    // ind_bits[j * nwords + w] = word w of observation j's indicator.
    let ind_bits: Vec<u64> = {
        let mut bits = vec![0u64; n_cols * nwords];
        for j in 0..n_cols {
            let col: Vec<bool> = (0..n_rows).map(|i| matrix.get(i, j)).collect();
            let packed = pack_column(&col);
            for w in 0..packed.len() {
                bits[j * nwords + w] = packed[w];
            }
        }
        bits
    };

    // Step 2: Pack targets into bitsets (same layout as observations).
    let tgt_bits: Vec<u64> = {
        let mut bits = vec![0u64; n_targets as usize * nwords];
        for k in 0..n_targets as usize {
            let col: Vec<bool> = targets[k].iter().map(|&v| v == 1).collect();
            let packed = pack_column(&col);
            for w in 0..packed.len() {
                bits[k * nwords + w] = packed[w];
            }
        }
        bits
    };

    // Step 3: Build per-target exclusion masks.
    let excl_mask: Vec<u64> = excludes
        .iter()
        .map(|opt| match opt {
            Some(idx) => idx.iter().fold(0u64, |m, &i| m | (1u64 << i)),
            None => 0u64,
        })
        .collect();

    // Step 4: Initialize store and counters.
    let mut cfg_mask: Vec<u64> = Vec::new();
    let mut cfg_supp: Vec<u32> = Vec::new();
    let mut cfg_y1: Vec<u32> = Vec::new();
    let mut depth_end: Vec<u32> = Vec::new();
    let mut admissibility: Vec<u32> = vec![0; max_cardinality as usize];
    let mut evaluability: Vec<Vec<u32>> = vec![vec![0; max_cardinality as usize]; n_thetas];
    let mut stopped_early = false;
    let mut stop_depth: Option<u32> = None;

    // Step 5: Depth 1 — screen observations for admissibility.
    let mut d1_admis: Vec<usize> = Vec::new();
    for j in 0..n_cols {
        let mut supp = 0u32;
        for w in 0..nwords {
            supp += popcount(ind_bits[j * nwords + w]);
        }
        if supp >= n_min {
            d1_admis.push(j);
        }
    }
    admissibility[0] = d1_admis.len() as u32;

    // Store depth-1 admissible configs.
    for &j in &d1_admis {
        let mut supp = 0u32;
        for w in 0..nwords {
            supp += popcount(ind_bits[j * nwords + w]);
        }
        cfg_mask.push(1u64 << j);
        cfg_supp.push(supp);
        // Stream y1 for each target.
        for k in 0..n_targets as usize {
            let y1 = and_popcount(
                &ind_bits[j * nwords..(j + 1) * nwords],
                &tgt_bits[k * nwords..(k + 1) * nwords],
            );
            cfg_y1.push(y1);
        }
    }
    depth_end.push(cfg_mask.len() as u32);

    // Evaluability for depth 1.
    for (ti, &theta) in thetas.iter().enumerate() {
        let mut count = 0u32;
        for g in 0..cfg_mask.len() {
            let n = cfg_supp[g] as f64;
            if n / (n + z2) >= theta {
                count += 1;
            }
        }
        evaluability[ti][0] = count;
    }

    // Early stop check for depth 1: if depth-1 admissible count exceeds
    // max_admissible_per_depth, stop immediately without exploring deeper.
    if (d1_admis.len() as f64) > max_admissible_per_depth {
        stopped_early = true;
        stop_depth = Some(1);
        return AscentResult {
            store: EmscStore {
                cfg_mask,
                cfg_supp,
                cfg_y1,
                depth_end,
                excl_mask,
                n_targets,
                z,
                z2,
            },
            admissibility,
            evaluability,
            stopped_early,
            stop_depth,
        };
    }

    // Step 6: Depths 2..=max_cardinality.
    // prev_bits holds the conjunction bitsets of the previous depth's configs.
    let mut prev_bits: Vec<u64> = Vec::new();
    for &j in &d1_admis {
        for w in 0..nwords {
            prev_bits.push(ind_bits[j * nwords + w]);
        }
    }

    for d in 2..=max_cardinality {
        let lo = if d == 2 {
            0
        } else {
            depth_end[(d - 3) as usize] as usize
        };
        let hi = depth_end[(d - 2) as usize] as usize;
        let n_prev = hi - lo;
        if n_prev == 0 {
            break;
        }

        // Pre-reserve store vectors for this depth's worst case.
        let worst_case = n_prev * d1_admis.len();
        cfg_mask.reserve(worst_case);
        cfg_supp.reserve(worst_case);
        cfg_y1.reserve(worst_case * n_targets as usize);

        let keep_bits = d < max_cardinality;
        let mut cur_bits: Vec<u64> = Vec::new();
        let mut n_admis: u64 = 0;

        // Parallelize the outer loop over parent configs.
        // Each parent config i is independent — they extend different parents.
        // Per-thread scratch buffer created inside the closure.
        // Early stop check deferred to after the parallel loop.
        let cfg_mask_ref = &cfg_mask;
        let prev_bits_ref = &prev_bits;
        let ind_bits_ref = &ind_bits;
        let tgt_bits_ref = &tgt_bits;
        let d1_admis_ref = &d1_admis;

        let results: Vec<ParentExtension> = (0..n_prev)
            .into_par_iter()
            .map(|i| {
                let gid = lo + i;
                let mask_i = cfg_mask_ref[gid];
                let max_j = max_element(mask_i).unwrap(); // highest set bit

                // Per-thread scratch — no sharing across parent configs.
                let mut scratch = vec![0u64; nwords];
                let mut masks = Vec::new();
                let mut supps = Vec::new();
                let mut y1s = Vec::new();
                let mut bits: Vec<u64> = Vec::new();

                for &j in d1_admis_ref {
                    if j as u32 <= max_j {
                        continue;
                    }

                    // AND parent bits with observation j bits.
                    let mut supp = 0u32;
                    for w in 0..nwords {
                        scratch[w] = prev_bits_ref[i * nwords + w] & ind_bits_ref[j * nwords + w];
                        supp += popcount(scratch[w]);
                    }

                    if supp < n_min {
                        continue;
                    }

                    masks.push(mask_i | (1u64 << j));
                    supps.push(supp);
                    for k in 0..n_targets as usize {
                        let y1 =
                            and_popcount(&scratch, &tgt_bits_ref[k * nwords..(k + 1) * nwords]);
                        y1s.push(y1);
                    }

                    if keep_bits {
                        for w in 0..nwords {
                            bits.push(scratch[w]);
                        }
                    }
                }

                ParentExtension {
                    masks,
                    supps,
                    y1s,
                    bits,
                }
            })
            .collect();

        // Merge results in order (preserves depth-major storage).
        for ext in &results {
            cfg_mask.extend_from_slice(&ext.masks);
            cfg_supp.extend_from_slice(&ext.supps);
            cfg_y1.extend_from_slice(&ext.y1s);
            cur_bits.extend_from_slice(&ext.bits);
            n_admis += ext.masks.len() as u64;
        }

        // Early stop check — done after the parallel loop, not inside it.
        if (n_admis as f64) > max_admissible_per_depth {
            stopped_early = true;
            stop_depth = Some(d);
            // Truncate to last completed depth — no partial configs.
            let last = *depth_end.last().unwrap() as usize;
            cfg_mask.truncate(last);
            cfg_supp.truncate(last);
            cfg_y1.truncate(last * n_targets as usize);
            break;
        }

        if stopped_early {
            break;
        }

        admissibility[(d - 1) as usize] = n_admis as u32;
        if n_admis == 0 {
            depth_end.push(cfg_mask.len() as u32);
            break;
        }
        depth_end.push(cfg_mask.len() as u32);

        // Evaluability for this depth.
        let base = depth_end[(d - 2) as usize] as usize;
        for (ti, &theta) in thetas.iter().enumerate() {
            let mut count = 0u32;
            for g in base..cfg_mask.len() {
                let n = cfg_supp[g] as f64;
                if n / (n + z2) >= theta {
                    count += 1;
                }
            }
            evaluability[ti][(d - 1) as usize] = count;
        }

        // Release prev_bits memory — swap cur_bits in, old prev_bits dropped.
        prev_bits = cur_bits;
    }

    AscentResult {
        store: EmscStore {
            cfg_mask,
            cfg_supp,
            cfg_y1,
            depth_end,
            excl_mask,
            n_targets,
            z,
            z2,
        },
        admissibility,
        evaluability,
        stopped_early,
        stop_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// 2 vars × 5 obs: all 4 non-trivial observations share both variables.
    /// Var 0: [1, 1, 1, 1, 0]
    /// Var 1: [1, 1, 1, 1, 0]
    ///
    /// Observation indicators (packed, V=2 → 1 word each):
    ///   Obs 0: [1,1] → 0b11 = 3
    ///   Obs 1: [1,1] → 0b11 = 3
    ///   Obs 2: [1,1] → 0b11 = 3
    ///   Obs 3: [1,1] → 0b11 = 3
    ///   Obs 4: [0,0] → 0b00 = 0
    ///
    /// With n_min=1: d1=4 (obs 0-3), d2=6 (all C(4,2) pairs admissible).
    fn test_matrix_2x5() -> BinaryMatrix {
        let data = [
            1.0, 1.0, 1.0, 1.0, 0.0, // var 0
            1.0, 1.0, 1.0, 1.0, 0.0, // var 1
        ];
        BinaryMatrix::new(&data, 2, 5).unwrap()
    }

    /// Standard test parameters: 1 target [1,1,0,0], theta=0.3, n_min=1.
    fn run_ascend(n_min: u32, max_adm: f64) -> AscentResult {
        let m = test_matrix_4x4();
        let targets = vec![vec![1, 1, 0, 0]]; // target: vars 0 and 1
        let thetas = vec![0.3];
        let excludes = vec![None];
        ascend(&m, &targets, &thetas, n_min, 0.05, 3, max_adm, &excludes)
    }

    // ---- Depth-1 admissibility screening ----

    #[test]
    fn depth1_all_admissible_with_n_min_1() {
        let result = run_ascend(1, 1000.0);
        // All 4 observations have popcount >= 1.
        assert_eq!(result.admissibility[0], 4);
        assert_eq!(result.store.depth_end[0], 4);
    }

    #[test]
    fn depth1_prunes_low_support_with_n_min_2() {
        let result = run_ascend(2, 1000.0);
        // Obs 3 has popcount=1 < 2 → pruned. Only 3 admissible.
        assert_eq!(result.admissibility[0], 3);
    }

    // ---- Ordered extension (j > highest_set_bit) ----

    #[test]
    fn depth2_ordered_extension_count() {
        let result = run_ascend(1, 1000.0);
        // Pairs (i,j) with j > max_bit(i):
        //   (0,1): AND(3,5)=1, supp=1 ✓
        //   (0,2): AND(3,6)=2, supp=1 ✓
        //   (0,3): AND(3,8)=0, supp=0 ✗
        //   (1,2): AND(5,6)=4, supp=1 ✓
        //   (1,3): AND(5,8)=0, supp=0 ✗
        //   (2,3): AND(6,8)=0, supp=0 ✗
        // 3 admissible.
        assert_eq!(result.admissibility[1], 3);
        assert_eq!(result.store.depth_end[1], 7); // 4 + 3
    }

    #[test]
    fn depth2_cfg_mask_values() {
        let result = run_ascend(1, 1000.0);
        let masks = &result.store.cfg_mask;
        // Depth 1: [1, 2, 4, 8]
        assert_eq!(masks[0], 1);
        assert_eq!(masks[1], 2);
        assert_eq!(masks[2], 4);
        assert_eq!(masks[3], 8);
        // Depth 2: [3, 5, 6] = {0,1}, {0,2}, {1,2}
        assert_eq!(masks[4], 3);
        assert_eq!(masks[5], 5);
        assert_eq!(masks[6], 6);
    }

    // ---- Downward-closure pruning ----

    #[test]
    fn downward_closure_skips_zero_support() {
        let result = run_ascend(1, 1000.0);
        // Configs (0,3), (1,3), (2,3) all have supp=0 → skipped.
        // Only 3 depth-2 configs, not 6.
        assert_eq!(result.admissibility[1], 3);
    }

    // ---- Depth 3 exhaustion ----

    #[test]
    fn depth3_no_admissible_triples() {
        let result = run_ascend(1, 1000.0);
        // All triples have AND=0 (obs 3 shares no vars with others).
        assert_eq!(result.admissibility[2], 0);
    }

    // ---- y1 streaming (config-major) ----

    #[test]
    fn y1_streaming_depth1() {
        let result = run_ascend(1, 1000.0);
        let store = &result.store;
        // Target = [1,1,0,0] → packed = 0b0011 = 3
        // cfg_y1 is config-major: cfg_y1[g * n_targets + k]
        // Obs 0: AND(3, 3) = 2
        assert_eq!(store.cfg_y1[0], 2);
        // Obs 1: AND(5, 3) = 1
        assert_eq!(store.cfg_y1[1], 1);
        // Obs 2: AND(6, 3) = 1
        assert_eq!(store.cfg_y1[2], 1);
        // Obs 3: AND(8, 3) = 0
        assert_eq!(store.cfg_y1[3], 0);
    }

    #[test]
    fn y1_streaming_depth2() {
        let result = run_ascend(1, 1000.0);
        let store = &result.store;
        // Depth-2 configs start at index 4.
        // Config {0,1}: AND(3,5)=1, y1 = AND(1, 3) = 1
        assert_eq!(store.cfg_y1[4], 1);
        // Config {0,2}: AND(3,6)=2 (0b0010), y1 = AND(2, 3) = 2 → popcount=1
        assert_eq!(store.cfg_y1[5], 1);
        // Config {1,2}: AND(5,6)=4, y1 = AND(4, 3) = 0
        assert_eq!(store.cfg_y1[6], 0);
    }

    // ---- Evaluability gate (n / (n + z²) >= θ) ----

    #[test]
    fn evaluability_uses_z_squared() {
        let result = run_ascend(1, 1000.0);
        // z = z_score(0.05) ≈ 1.96, z² ≈ 3.8416
        // Depth 1: supp=2 → 2/(2+3.84) ≈ 0.342 >= 0.3 → evaluable (3 configs)
        //          supp=1 → 1/(1+3.84) ≈ 0.206 < 0.3 → not evaluable (1 config)
        assert_eq!(result.evaluability[0][0], 3);
        // Depth 2: all supp=1 → 0.206 < 0.3 → 0 evaluable
        assert_eq!(result.evaluability[0][1], 0);
    }

    // ---- Early stop ----

    #[test]
    fn early_stop_truncates_to_last_completed_depth() {
        // Use 2×5 matrix: d1=4, d2=6 (all C(4,2) pairs admissible).
        // max_admissible=5 → depth 1: 4 ≤ 5 (no stop), depth 2: 6 > 5 → stop.
        let m = test_matrix_2x5();
        let targets = vec![vec![1, 1]]; // target: vars 0,1
        let thetas = vec![0.3];
        let excludes = vec![None];
        let result = ascend(&m, &targets, &thetas, 1, 0.05, 3, 5.0, &excludes);
        assert!(result.stopped_early);
        assert_eq!(result.stop_depth, Some(2));
        // Truncated to depth 1 (4 configs).
        assert_eq!(result.store.cfg_mask.len(), 4);
        assert_eq!(result.store.cfg_supp.len(), 4);
        assert_eq!(result.store.cfg_y1.len(), 4);
    }

    #[test]
    fn no_early_stop_with_high_threshold() {
        let result = run_ascend(1, 1000.0);
        assert!(!result.stopped_early);
        assert_eq!(result.stop_depth, None);
    }

    #[test]
    fn early_stop_at_depth1_with_zero_max_admissible() {
        // max_admissible=0.0 → depth-1 has 4 admissible > 0.0 → stop at depth 1.
        let result = run_ascend(1, 0.0);
        assert!(result.stopped_early);
        assert_eq!(result.stop_depth, Some(1));
        // Depth-1 configs are stored (4 admissible observations).
        assert_eq!(result.store.cfg_mask.len(), 4);
        assert_eq!(result.store.depth_end.len(), 1);
        assert_eq!(result.store.depth_end[0], 4);
        // No depth-2 configs explored.
        assert_eq!(result.admissibility[1], 0);
    }

    // ---- depth_end offsets ----

    #[test]
    fn depth_end_monotonic() {
        let result = run_ascend(1, 1000.0);
        let de = &result.store.depth_end;
        // depth_end = [4, 7, 7]
        assert_eq!(de[0], 4);
        assert_eq!(de[1], 7);
        assert_eq!(de[2], 7);
        // Monotonic non-decreasing.
        for w in de.windows(2) {
            assert!(w[0] <= w[1], "depth_end not monotonic: {:?}", de);
        }
    }

    // ---- Support counts ----

    #[test]
    fn cfg_supp_matches_popcount() {
        let result = run_ascend(1, 1000.0);
        let supp = &result.store.cfg_supp;
        // Depth 1: obs 0,1,2 have popcount=2; obs 3 has popcount=1.
        assert_eq!(supp[0], 2);
        assert_eq!(supp[1], 2);
        assert_eq!(supp[2], 2);
        assert_eq!(supp[3], 1);
        // Depth 2: all conjunctions have popcount=1.
        assert_eq!(supp[4], 1);
        assert_eq!(supp[5], 1);
        assert_eq!(supp[6], 1);
    }

    // ---- EmscStore fields ----

    #[test]
    fn store_has_z_and_z2() {
        let result = run_ascend(1, 1000.0);
        let z = crate::stats::z_score(0.05);
        assert!((result.store.z - z).abs() < 1e-15);
        assert!((result.store.z2 - z * z).abs() < 1e-15);
    }

    #[test]
    fn store_n_targets() {
        let result = run_ascend(1, 1000.0);
        assert_eq!(result.store.n_targets, 1);
    }

    // ---- Multiple targets ----

    #[test]
    fn multiple_targets_y1_config_major() {
        let m = test_matrix_4x4();
        let targets = vec![
            vec![1, 1, 0, 0], // target 0: vars 0,1
            vec![0, 0, 1, 1], // target 1: vars 2,3
        ];
        let thetas = vec![0.3];
        let excludes = vec![None, None];
        let result = ascend(&m, &targets, &thetas, 1, 0.05, 2, 1000.0, &excludes);

        let store = &result.store;
        assert_eq!(store.n_targets, 2);
        // Config 0 (obs 0, packed=3):
        //   y1[0*2+0] = AND(3, 0b0011) = 2
        //   y1[0*2+1] = AND(3, 0b1100) = 0
        assert_eq!(store.cfg_y1[0], 2);
        assert_eq!(store.cfg_y1[1], 0);
        // Config 3 (obs 3, packed=8):
        //   y1[3*2+0] = AND(8, 0b0011) = 0
        //   y1[3*2+1] = AND(8, 0b1100) = 0  (obs 3 = [0,0,0,1], target 1 = [0,0,1,1])
        //   Wait: obs 3 packed = 0b1000 = 8, target 1 packed = 0b1100 = 12
        //   AND(8, 12) = 8 → popcount = 1
        assert_eq!(store.cfg_y1[6], 0); // target 0
        assert_eq!(store.cfg_y1[7], 1); // target 1

        // Depth-2 configs (indices 4-6, y1 at cfg_y1 indices 8-13):
        // Config 4 {0,1}: AND(3,5)=1, y1[8]=AND(1,3)=1→popcount=1, y1[9]=AND(1,12)=0→popcount=0
        assert_eq!(store.cfg_y1[8], 1);
        assert_eq!(store.cfg_y1[9], 0);
        // Config 5 {0,2}: AND(3,6)=2, y1[10]=AND(2,3)=2→popcount=1, y1[11]=AND(2,12)=0→popcount=0
        assert_eq!(store.cfg_y1[10], 1);
        assert_eq!(store.cfg_y1[11], 0);
        // Config 6 {1,2}: AND(5,6)=4, y1[12]=AND(4,3)=0→popcount=0, y1[13]=AND(4,12)=4→popcount=1
        assert_eq!(store.cfg_y1[12], 0);
        assert_eq!(store.cfg_y1[13], 1);
    }

    // ---- Exclusion masks ----

    #[test]
    fn excl_mask_stored_correctly() {
        let m = test_matrix_4x4();
        let targets = vec![vec![1, 1, 0, 0]];
        let thetas = vec![0.3];
        let excludes = vec![Some(vec![0, 2])]; // exclude vars 0 and 2
        let result = ascend(&m, &targets, &thetas, 1, 0.05, 1, 1000.0, &excludes);
        // excl_mask[0] = (1<<0) | (1<<2) = 0b0101 = 5
        assert_eq!(result.store.excl_mask[0], 5);
    }

    #[test]
    fn excl_mask_none_is_zero() {
        let m = test_matrix_4x4();
        let targets = vec![vec![1, 1, 0, 0]];
        let thetas = vec![0.3];
        let excludes = vec![None];
        let result = ascend(&m, &targets, &thetas, 1, 0.05, 1, 1000.0, &excludes);
        assert_eq!(result.store.excl_mask[0], 0);
    }

    // ---- u64 counters (admissibility stored as u32) ----

    #[test]
    fn admissibility_counts_correct() {
        let result = run_ascend(1, 1000.0);
        assert_eq!(result.admissibility.len(), 3);
        assert_eq!(result.admissibility[0], 4);
        assert_eq!(result.admissibility[1], 3);
        assert_eq!(result.admissibility[2], 0);
    }
}
