//! Golden-fixture parity tests: Rust stats functions vs R-generated reference
//! fixtures stored in `tests/fixtures/golden/`.
//!
//! Fixture format (one JSON file per dataset):
//! ```json
//! {
//!   "format_version": 1,
//!   "seed": 42,
//!   "dataset": "small_4x3",
//!   "wilson_lower": [
//!     {"p": 0.5, "n": 100, "z": 1.96, "value": 0.40238}
//!   ],
//!   "mi_2x2": [
//!     {"n11": 10, "nx1": 50, "nx2": 60, "n": 100, "value": 0.42281}
//!   ],
//!   "g_test": [
//!     {"n11": 10, "n": 100, "mi": 0.05, "g": 1.0, "p": 0.31731}
//!   ],
//!   "chi2_sf_df1": [
//!     {"g": 3.841, "value": 0.05}
//!   ],
//!   "qnorm": [
//!     {"p": 0.975, "value": 1.95996}
//!   ]
//! }
//! ```
//!
//! Tolerances:
//! - Pure math (wilson_lower, mi_2x2, g_test.g): |rust - ref| / |ref| <= 1e-12
//! - Platform-dependent (chi2_sf_df1, qnorm, g_test.p): |rust - ref| <= max(1e-12 * |ref|, 1e-12)
//! - Integers: exact equality
//!
//! If fixtures don't exist, tests skip gracefully (R not installed → no fixtures).

use std::fs;
use std::path::Path;

use configural::stats::{chi2_sf_df1, g_test, mi_2x2, normal_inverse_cdf, wilson_lower};
use serde_json::Value;

/// Load all golden fixture JSON files from `tests/fixtures/golden/`.
///
/// Returns `None` (skip) if the directory doesn't exist or contains no `.json`
/// files. This allows the test suite to run without R-installed fixtures.
fn load_fixtures() -> Option<Vec<Value>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden");
    if !dir.exists() {
        eprintln!("skipped: golden fixtures not found at {dir:?}");
        return None;
    }
    let mut fixtures = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|e| e == "json") {
                let content = fs::read_to_string(entry.path()).unwrap_or_else(|e| {
                    panic!("failed to read fixture {:?}: {e}", entry.path())
                });
                let parsed: Value = serde_json::from_str(&content).unwrap_or_else(|e| {
                    panic!("failed to parse JSON in {:?}: {e}", entry.path())
                });
                fixtures.push(parsed);
            }
        }
    }
    if fixtures.is_empty() {
        eprintln!("skipped: no golden fixtures found");
        return None;
    }
    Some(fixtures)
}

/// Extract an f64 from a JSON value (handles both int and float).
fn as_f64(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("expected number, got {v}"))
}

/// Relative tolerance check for pure-math functions:
/// `|rust - ref| / |ref| <= 1e-12`, or `|rust - ref| <= 1e-300` if ref ≈ 0.
fn assert_rel_tol(rust: f64, reference: f64, label: &str) {
    let abs_diff = (rust - reference).abs();
    if reference.abs() < 1e-300 {
        assert!(
            abs_diff <= 1e-300,
            "{label}: rust={rust}, ref≈0, abs_diff={abs_diff}"
        );
    } else {
        let rel_err = abs_diff / reference.abs();
        assert!(
            rel_err <= 1e-12,
            "{label}: rust={rust}, ref={reference}, rel_err={rel_err} (tol 1e-12)"
        );
    }
}

/// Platform-dependent tolerance check (erfc/qnorm):
/// `|rust - ref| <= max(1e-12 * |ref|, 1e-12)`.
fn assert_platform_tol(rust: f64, reference: f64, label: &str) {
    let abs_diff = (rust - reference).abs();
    let tol = (1e-12 * reference.abs()).max(1e-12);
    assert!(
        abs_diff <= tol,
        "{label}: rust={rust}, ref={reference}, abs_diff={abs_diff} (tol {tol})"
    );
}

// ---- structural validation ----

#[test]
fn golden_fixtures_exist_and_match() {
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return, // skip gracefully
    };
    assert!(!fixtures.is_empty(), "at least one fixture should exist");
    for fixture in &fixtures {
        assert!(
            fixture.get("format_version").is_some(),
            "fixture missing format_version"
        );
        assert!(fixture.get("seed").is_some(), "fixture missing seed");
    }
}

// ---- wilson_lower parity (pure math) ----

#[test]
fn wilson_lower_parity() {
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };
    for fixture in &fixtures {
        if let Some(cases) = fixture.get("wilson_lower").and_then(|v| v.as_array()) {
            for case in cases {
                let p = as_f64(&case["p"]);
                let n = as_f64(&case["n"]);
                let z = as_f64(&case["z"]);
                let expected = as_f64(&case["value"]);
                let actual = wilson_lower(p, n, z);
                assert_rel_tol(actual, expected, &format!("wilson_lower({p}, {n}, {z})"));
            }
        }
    }
}

// ---- mi_2x2 parity (pure math) ----

#[test]
fn mi_2x2_parity() {
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };
    for fixture in &fixtures {
        if let Some(cases) = fixture.get("mi_2x2").and_then(|v| v.as_array()) {
            for case in cases {
                let n11 = as_f64(&case["n11"]);
                let nx1 = as_f64(&case["nx1"]);
                let nx2 = as_f64(&case["nx2"]);
                let n = as_f64(&case["n"]);
                let expected = as_f64(&case["value"]);
                let actual = mi_2x2(n11, nx1, nx2, n);
                assert_rel_tol(
                    actual,
                    expected,
                    &format!("mi_2x2({n11}, {nx1}, {nx2}, {n})"),
                );
            }
        }
    }
}

// ---- g_test parity (g = pure math, p = platform-dependent) ----

#[test]
fn g_test_parity() {
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };
    for fixture in &fixtures {
        if let Some(cases) = fixture.get("g_test").and_then(|v| v.as_array()) {
            for case in cases {
                let n11 = as_f64(&case["n11"]);
                let n = as_f64(&case["n"]);
                let mi = as_f64(&case["mi"]);
                let expected_g = as_f64(&case["g"]);
                let expected_p = as_f64(&case["p"]);
                let result = g_test(n11, n, mi);
                assert_rel_tol(result.g, expected_g, &format!("g_test({n11}, {n}, {mi}).g"));
                assert_platform_tol(result.p, expected_p, &format!("g_test({n11}, {n}, {mi}).p"));
            }
        }
    }
}

// ---- chi2_sf_df1 parity (platform-dependent: erfc) ----

#[test]
fn chi2_sf_df1_parity() {
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };
    for fixture in &fixtures {
        if let Some(cases) = fixture.get("chi2_sf_df1").and_then(|v| v.as_array()) {
            for case in cases {
                let g = as_f64(&case["g"]);
                let expected = as_f64(&case["value"]);
                let actual = chi2_sf_df1(g);
                assert_platform_tol(actual, expected, &format!("chi2_sf_df1({g})"));
            }
        }
    }
}

// ---- qnorm parity (platform-dependent: AS241) ----

#[test]
fn qnorm_parity() {
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };
    for fixture in &fixtures {
        if let Some(cases) = fixture.get("qnorm").and_then(|v| v.as_array()) {
            for case in cases {
                let p = as_f64(&case["p"]);
                let expected = as_f64(&case["value"]);
                let actual = normal_inverse_cdf(p);
                assert_platform_tol(actual, expected, &format!("qnorm({p})"));
            }
        }
    }
}
