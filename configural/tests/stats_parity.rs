//! Parity tests: Rust stats functions vs R-generated golden fixtures.
//!
//! Each test loads a JSON file from `tests/golden/` and asserts that the
//! Rust implementation matches the R reference to within 1e-12 relative
//! tolerance (or absolute tolerance for values near zero).

use configural::stats::{
    chi2_sf_df1, g_test, mi_2x2, normal_inverse_cdf, wilson_lower, GTestResult,
};
use serde_json::Value;
use std::fs;

/// Load a golden fixture JSON file.
fn load_golden(name: &str) -> Vec<Value> {
    let path = format!("tests/golden/{name}");
    let content =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    serde_json::from_str::<Vec<Value>>(&content)
        .unwrap_or_else(|e| panic!("failed to parse JSON in {path}: {e}"))
}

/// Extract an f64 from a JSON value (handles both int and float).
fn as_f64(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("expected number, got {v}"))
}

/// Relative tolerance check: |rust - ref| / |ref| <= tol, or |rust - ref| <= abs_tol if ref ≈ 0.
fn assert_parity(rust: f64, reference: f64, label: &str) {
    let abs_diff = (rust - reference).abs();
    if reference.abs() < 1e-300 {
        assert!(
            abs_diff < 1e-15,
            "{label}: rust={rust}, ref≈0, abs_diff={abs_diff}"
        );
    } else {
        let rel_diff = abs_diff / reference.abs();
        assert!(
            rel_diff < 1e-12,
            "{label}: rust={rust}, ref={reference}, rel_diff={rel_diff} (tol 1e-12)"
        );
    }
}

// ---- wilson_lower parity ----

#[test]
fn wilson_parity() {
    let cases = load_golden("wilson_parity.json");
    assert!(!cases.is_empty(), "wilson golden fixture is empty");
    for case in &cases {
        let args = case["args"].as_array().unwrap();
        let p = as_f64(&args[0]);
        let n = as_f64(&args[1]);
        let z = as_f64(&args[2]);
        let expected = as_f64(&case["expected"]);
        let result = wilson_lower(p, n, z);
        assert_parity(result, expected, &format!("wilson_lower({p}, {n}, {z})"));
    }
}

// ---- mi_2x2 parity ----

#[test]
fn mi_parity() {
    let cases = load_golden("mi_parity.json");
    assert!(!cases.is_empty(), "mi golden fixture is empty");
    for case in &cases {
        let args = case["args"].as_array().unwrap();
        let n11 = as_f64(&args[0]);
        let nx1 = as_f64(&args[1]);
        let nx2 = as_f64(&args[2]);
        let n = as_f64(&args[3]);
        let expected = as_f64(&case["expected"]);
        let result = mi_2x2(n11, nx1, nx2, n);
        assert_parity(
            result,
            expected,
            &format!("mi_2x2({n11}, {nx1}, {nx2}, {n})"),
        );
    }
}

// ---- chi2_sf_df1 parity ----

#[test]
fn chi2_parity() {
    let cases = load_golden("chi2_parity.json");
    assert!(!cases.is_empty(), "chi2 golden fixture is empty");
    for case in &cases {
        let args = case["args"].as_array().unwrap();
        let g = as_f64(&args[0]);
        let expected = as_f64(&case["expected"]);
        let result = chi2_sf_df1(g);
        assert_parity(result, expected, &format!("chi2_sf_df1({g})"));
    }
}

// ---- g_test parity ----

#[test]
fn gtest_parity() {
    let cases = load_golden("gtest_parity.json");
    assert!(!cases.is_empty(), "gtest golden fixture is empty");
    for case in &cases {
        let args = case["args"].as_array().unwrap();
        let n11 = as_f64(&args[0]);
        let n = as_f64(&args[1]);
        let mi = as_f64(&args[2]);
        let expected_g = as_f64(&case["expected"]["g"]);
        let expected_p = as_f64(&case["expected"]["p"]);
        let result: GTestResult = g_test(n11, n, mi);
        assert_parity(result.g, expected_g, &format!("g_test({n11}, {n}, {mi}).g"));
        assert_parity(result.p, expected_p, &format!("g_test({n11}, {n}, {mi}).p"));
    }
}

// ---- normal_inverse_cdf parity ----

#[test]
fn qnorm_parity() {
    let cases = load_golden("qnorm_parity.json");
    assert!(!cases.is_empty(), "qnorm golden fixture is empty");
    for case in &cases {
        let args = case["args"].as_array().unwrap();
        let p = as_f64(&args[0]);
        let expected = as_f64(&case["expected"]);
        let result = normal_inverse_cdf(p);
        assert_parity(result, expected, &format!("normal_inverse_cdf({p})"));
    }
}
