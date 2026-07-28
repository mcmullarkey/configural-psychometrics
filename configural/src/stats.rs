//! Statistical primitives for sufficiency testing (Wilson) and significance
//! testing (MI/G-test/chi2/qnorm).
//!
//! Ported from `emsc_v4.cpp` (Wilson, qnorm) and `pairmi_v5.cpp` (safe_term,
//! mi_2x2, G-test). Numerical parity ~1e-12 vs R reference.
//!
//! ## What this module provides
//!
//! - [`wilson_lower`] — Wilson score lower bound (sufficiency testing)
//! - [`safe_term`] — guarded single-cell MI contribution
//! - [`mi_2x2`] — four-term mutual information in nats
//! - [`g_test`] — G-squared test with flipped-count correction
//! - [`chi2_sf_df1`] — chi-squared survival function (1 df) via erfc
//! - [`normal_inverse_cdf`] — standard normal quantile (AS241)
//! - [`z_score`] — two-tailed z from alpha
//!
//! ## What this module does NOT provide
//!
//! - Data loading or matrix operations (see [`crate::matrix`])
//! - Bitset operations (see [`crate::matrix::BinaryMatrix`])
//! - Engine ascent logic or search

/// Result of a G-test: the G statistic and its p-value are inseparable.
///
/// Both fields are always finite and non-negative for valid inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GTestResult {
    /// The G-squared statistic: `2 * jc * mi`.
    pub g: f64,
    /// The p-value from chi-squared with 1 df: `erfc(sqrt(g/2))`.
    pub p: f64,
}

/// Wilson score lower bound.
///
/// Exact C++ term order from `emsc_v4.cpp:57-62`:
/// ```text
/// denom  = 1 + z²/n
/// center = p + z²/(2n)
/// margin = z * sqrt(p(1-p)/n + z²/(4n²))
/// result = (center - margin) / denom
/// ```
///
/// # Panics
///
/// Panics if `n <= 0.0` (division by zero).
///
/// # Examples
///
/// ```
/// use configural::stats::wilson_lower;
/// // 60% observed rate, n=100, z=1.96 → ~0.497
/// let lb = wilson_lower(0.6, 100.0, 1.96);
/// assert!(lb > 0.49 && lb < 0.51);
/// ```
#[must_use]
pub fn wilson_lower(p: f64, n: f64, z: f64) -> f64 {
    assert!(n > 0.0, "wilson_lower: n must be > 0, got {n}");
    let denom = 1.0 + z * z / n;
    let center = p + z * z / (2.0 * n);
    let margin = z * libm::sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n));
    (center - margin) / denom
}

/// Guarded single-cell MI contribution.
///
/// Returns `(n_cell / n) * ln(n_cell * n / (m1 * m2))` iff **all four**
/// arguments are strictly positive. Otherwise returns `0.0`.
///
/// The four-argument guard (not two-argument) matches `pairmi_v5.cpp:70-74`
/// and prevents `NaN` when any marginal count is zero.
///
/// # Examples
///
/// ```
/// use configural::stats::safe_term;
/// // All positive → normal computation (result sign depends on cell ratio)
/// assert!(safe_term(50.0, 50.0, 50.0, 100.0) > 0.0);
/// // Any zero → 0.0 (not NaN)
/// assert_eq!(safe_term(0.0, 50.0, 60.0, 100.0), 0.0);
/// assert_eq!(safe_term(0.0, 0.0, 0.0, 0.0), 0.0);
/// ```
#[must_use]
pub fn safe_term(n_cell: f64, m1: f64, m2: f64, n: f64) -> f64 {
    if n_cell > 0.0 && m1 > 0.0 && m2 > 0.0 && n > 0.0 {
        (n_cell / n) * libm::log((n_cell * n) / (m1 * m2))
    } else {
        0.0
    }
}

/// Four-term mutual information (nats) for a 2×2 table.
///
/// Computes `t11 + t10 + t01 + t00` in that exact order, matching
/// `pairmi_v5.cpp:76-85`. Each term uses [`safe_term`] with the
/// appropriate marginals.
///
/// # Arguments
///
/// - `n11` — count of (1,1) co-occurrence
/// - `nx1` — marginal count of x=1
/// - `nx2` — marginal count of x=2
/// - `n` — total count
///
/// # Examples
///
/// ```
/// use configural::stats::mi_2x2;
/// // All-zero table → 0.0 (safe_term guard)
/// assert_eq!(mi_2x2(0.0, 0.0, 0.0, 0.0), 0.0);
/// ```
#[must_use]
pub fn mi_2x2(n11: f64, nx1: f64, nx2: f64, n: f64) -> f64 {
    let n10 = nx1 - n11;
    let n01 = nx2 - n11;
    let n00 = n - nx1 - nx2 + n11;
    safe_term(n11, nx1, nx2, n)
        + safe_term(n10, nx1, n - nx2, n)
        + safe_term(n01, n - nx1, nx2, n)
        + safe_term(n00, n - nx1, n - nx2, n)
}

/// Chi-squared survival function with 1 degree of freedom.
///
/// Computed as `erfc(sqrt(g/2))` via the `libm` crate. This is the
/// upper-tail p-value of a chi-squared(1) distribution.
///
/// # Panics
///
/// Panics if `g < 0.0` (negative chi-squared is impossible).
///
/// # Examples
///
/// ```
/// use configural::stats::chi2_sf_df1;
/// assert_eq!(chi2_sf_df1(0.0), 1.0);
/// // g=3.841 → p≈0.05
/// let p = chi2_sf_df1(3.8414588);
/// assert!((p - 0.05).abs() < 0.001);
/// ```
#[must_use]
pub fn chi2_sf_df1(g: f64) -> f64 {
    assert!(g >= 0.0, "chi2_sf_df1: g must be >= 0, got {g}");
    if g == 0.0 {
        return 1.0;
    }
    libm::erfc(libm::sqrt(g / 2.0))
}

/// G-test (G-squared) with flipped-count correction.
///
/// Computes `jc = min(n11, n - n11)`, `g = 2 * jc * mi`, then
/// `p = chi2_sf_df1(g)`. The flipped-count correction matches
/// `pairmi_v5.cpp:146-150`: when `n11 > n/2`, the joint count is
/// replaced by its complement `n - n11`.
///
/// # Panics
///
/// Panics if `n < 0.0` (negative total is impossible).
///
/// # Examples
///
/// ```
/// use configural::stats::{g_test, GTestResult};
/// let r = g_test(10.0, 100.0, 0.05);
/// assert!(r.g > 0.0);
/// assert!(r.p > 0.0 && r.p < 1.0);
/// ```
#[must_use]
pub fn g_test(n11: f64, n: f64, mi: f64) -> GTestResult {
    assert!(n >= 0.0, "g_test: n must be >= 0, got {n}");
    let jc = if n11 > n / 2.0 { n - n11 } else { n11 };
    let g = 2.0 * jc * mi;
    let p = chi2_sf_df1(g);
    GTestResult { g, p }
}

/// Standard normal inverse CDF (quantile function) via AS241.
///
/// Hand-rolled implementation of Algorithm AS241 (Wichura 1988),
/// matching R's `qnorm` source (`src/nmath/qnorm.c`). Achieves full
/// `f64` precision (~1e-16 relative).
///
/// # Panics
///
/// Panics if `p <= 0.0` or `p >= 1.0` (inverse CDF is undefined at
/// the boundaries).
///
/// # Examples
///
/// ```
/// use configural::stats::normal_inverse_cdf;
/// // qnorm(0.975) ≈ 1.96
/// let z = normal_inverse_cdf(0.975);
/// assert!((z - 1.959963984540054).abs() < 1e-12);
/// ```
#[must_use]
#[allow(clippy::excessive_precision)] // AS241 coefficients from R source
pub fn normal_inverse_cdf(p: f64) -> f64 {
    assert!(
        p > 0.0 && p < 1.0,
        "normal_inverse_cdf: p must be in (0, 1), got {p}"
    );

    // AS241 constants (Wichura 1988, as used in R's qnorm).
    const SPLIT1: f64 = 0.425;
    const SPLIT2: f64 = 5.0;
    const SPLIT3: f64 = 27.0;
    const CONST1: f64 = 0.180625;
    const CONST2: f64 = 1.6;

    // Coefficients for |q| <= SPLIT1 (p close to 0.5).
    // Stored highest-degree-first to match R's Horner evaluation order.
    const A: [f64; 8] = [
        2509.0809287301226727,
        33430.575583588128105,
        67265.770927008700853,
        45921.953931549871457,
        13731.693765509461125,
        1971.5909503065514427,
        133.14166789178437745,
        3.387132872796366608,
    ];
    const B: [f64; 8] = [
        5226.495278852854561,
        28729.085735721942674,
        39307.89580009271061,
        21213.794301586595867,
        5394.1960214247511077,
        687.1870074920579083,
        42.313330701600911252,
        1.0,
    ];

    // Coefficients for r <= SPLIT2 (moderate tail, after r -= 1.6).
    const C: [f64; 8] = [
        7.7454501427834140764e-4,
        0.0227238449892691845833,
        0.24178072517745061177,
        1.27045825245236838258,
        3.64784832476320460504,
        5.7694972214606914055,
        4.6303378461565452959,
        1.42343711074968357734,
    ];
    const D: [f64; 8] = [
        1.05075007164441684324e-9,
        5.475938084995344946e-4,
        0.0151986665636164571966,
        0.14810397642748007459,
        0.68976733498510000455,
        1.6763848301838038494,
        2.05319162663775882187,
        1.0,
    ];

    // Coefficients for SPLIT2 < r <= SPLIT3 (far tail, after r -= 5.0).
    const E: [f64; 8] = [
        2.01033439929228813265e-7,
        2.71155556874348757815e-5,
        0.0012426609473880784386,
        0.026532189526576123093,
        0.29656057182850489123,
        1.7848265399172913358,
        5.4637849111641143699,
        6.6579046435011037772,
    ];
    const F: [f64; 8] = [
        2.04426310338993978564e-15,
        1.4215117583164458887e-7,
        1.8463183175100546818e-5,
        7.868691311456132591e-4,
        0.0148753612908506148525,
        0.13692988092273580531,
        0.59983220655588793769,
        1.0,
    ];

    let q = p - 0.5;

    if q.abs() <= SPLIT1 {
        // Region 1: p close to 0.5
        let r = CONST1 - q * q;
        let num = horner8(&A, r);
        let den = horner8(&B, r);
        q * num / den
    } else {
        // Tail regions: compute r = sqrt(-log(min(p, 1-p)))
        let tail = if q < 0.0 { p } else { 1.0 - p };
        let lp = libm::log(tail);
        let r = libm::sqrt(-lp);
        let val = if r <= SPLIT2 {
            // Region 2: moderate tail (r <= 5)
            let r = r - CONST2;
            let num = horner8(&C, r);
            let den = horner8(&D, r);
            num / den
        } else if r <= SPLIT3 {
            // Region 3: far tail (5 < r <= 27)
            let r = r - SPLIT2;
            let num = horner8(&E, r);
            let den = horner8(&F, r);
            num / den
        } else {
            // Region 4: extreme tail (r > 27) — asymptotic formula.
            // Only reached for p < exp(-27²) ≈ 2.5e-317. Uses the
            // zeroth-order asymptotic: val ≈ r * sqrt(2).
            // Higher-order terms (R's xs_1..xs_4) omitted; not needed
            // for parity at any practical p value.
            r * libm::sqrt(2.0)
        };
        if q < 0.0 {
            -val
        } else {
            val
        }
    }
}

/// Horner's method for an 8-coefficient polynomial.
///
/// Coefficients are stored highest-degree-first: `coeffs[0]` is the
/// leading term, `coeffs[7]` is the constant. This matches R's qnorm
/// source evaluation order.
#[inline]
fn horner8(coeffs: &[f64; 8], x: f64) -> f64 {
    let mut acc = coeffs[0];
    acc = acc * x + coeffs[1];
    acc = acc * x + coeffs[2];
    acc = acc * x + coeffs[3];
    acc = acc * x + coeffs[4];
    acc = acc * x + coeffs[5];
    acc = acc * x + coeffs[6];
    acc = acc * x + coeffs[7];
    acc
}

/// Two-tailed z-score from significance level alpha.
///
/// Computes `normal_inverse_cdf(1 - alpha/2)`. For the standard
/// `alpha = 0.05`, this returns `z ≈ 1.96`.
///
/// # Panics
///
/// Panics if `alpha <= 0.0` or `alpha >= 1.0`.
///
/// # Examples
///
/// ```
/// use configural::stats::z_score;
/// let z = z_score(0.05);
/// assert!((z - 1.959963984540054).abs() < 1e-12);
/// ```
#[must_use]
pub fn z_score(alpha: f64) -> f64 {
    assert!(
        alpha > 0.0 && alpha < 1.0,
        "z_score: alpha must be in (0, 1), got {alpha}"
    );
    normal_inverse_cdf(1.0 - alpha / 2.0)
}

#[cfg(test)]
#[allow(clippy::excessive_precision)] // R reference values have full f64 precision
mod tests {
    use super::*;

    // ---- wilson_lower ----

    #[test]
    fn wilson_basic() {
        // p=0.6, n=100, z=1.96 → ~0.497
        let lb = wilson_lower(0.6, 100.0, 1.96);
        assert!(lb > 0.49 && lb < 0.51, "got {lb}");
    }

    #[test]
    fn wilson_p_zero() {
        // p=0 is valid (not a panic case); only n=0 panics
        let lb = wilson_lower(0.0, 100.0, 1.96);
        assert_eq!(lb, 0.0);
    }

    #[test]
    fn wilson_p_one() {
        let lb = wilson_lower(1.0, 100.0, 1.96);
        // center = 1 + z²/(2n), margin = z*sqrt(0 + z²/(4n²)) = z²/(2n)
        // result = (1 + z²/(2n) - z²/(2n)) / (1 + z²/n) = 1/(1+z²/n)
        let expected = 1.0 / (1.0 + 1.96 * 1.96 / 100.0);
        assert!(
            (lb - expected).abs() < 1e-15,
            "got {lb}, expected {expected}"
        );
    }

    #[test]
    #[should_panic(expected = "n must be > 0")]
    fn wilson_n_zero_panics() {
        let _ = wilson_lower(0.6, 0.0, 1.96);
    }

    // ---- safe_term ----

    #[test]
    fn safe_term_all_positive() {
        let v = safe_term(10.0, 50.0, 60.0, 100.0);
        // (10/100)*ln(10*100/(50*60)) = 0.1*ln(1000/3000) = 0.1*ln(1/3)
        let expected = 0.1 * libm::log(1.0 / 3.0);
        assert!((v - expected).abs() < 1e-15, "got {v}, expected {expected}");
    }

    #[test]
    fn safe_term_cell_zero() {
        assert_eq!(safe_term(0.0, 50.0, 60.0, 100.0), 0.0);
    }

    #[test]
    fn safe_term_m1_zero() {
        assert_eq!(safe_term(10.0, 0.0, 60.0, 100.0), 0.0);
    }

    #[test]
    fn safe_term_m2_zero() {
        assert_eq!(safe_term(10.0, 50.0, 0.0, 100.0), 0.0);
    }

    #[test]
    fn safe_term_n_zero() {
        assert_eq!(safe_term(10.0, 50.0, 60.0, 0.0), 0.0);
    }

    #[test]
    fn safe_term_all_zero() {
        assert_eq!(safe_term(0.0, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn safe_term_subnormal() {
        // Subnormal values pass the > 0 guard
        let v = safe_term(1e-300, 2e-10, 3e-10, 100.0);
        assert!(v.is_finite(), "got non-finite: {v}");
    }

    // ---- mi_2x2 ----

    #[test]
    fn mi_all_zero() {
        assert_eq!(mi_2x2(0.0, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn mi_basic() {
        let v = mi_2x2(10.0, 50.0, 60.0, 100.0);
        assert!(v.is_finite(), "got non-finite: {v}");
        // MI should be non-negative for these counts
        assert!(v >= 0.0, "MI should be >= 0, got {v}");
    }

    #[test]
    fn mi_subnormal_finite() {
        let v = mi_2x2(1e-300, 2e-10, 3e-10, 100.0);
        assert!(v.is_finite(), "got non-finite: {v}");
    }

    #[test]
    fn mi_independence_zero() {
        // If n11 = nx1*nx2/n (independence), MI should be ~0
        // nx1=50, nx2=50, n=100 → expected n11 = 25
        let v = mi_2x2(25.0, 50.0, 50.0, 100.0);
        assert!(
            v.abs() < 1e-10,
            "MI under independence should be ~0, got {v}"
        );
    }

    // ---- chi2_sf_df1 ----

    #[test]
    fn chi2_g_zero() {
        assert_eq!(chi2_sf_df1(0.0), 1.0);
    }

    #[test]
    fn chi2_basic() {
        // g=3.841 → p≈0.05
        let p = chi2_sf_df1(3.8414588);
        assert!((p - 0.05).abs() < 0.001, "got {p}");
    }

    #[test]
    #[should_panic(expected = "g must be >= 0")]
    fn chi2_negative_panics() {
        let _ = chi2_sf_df1(-1.0);
    }

    #[test]
    fn chi2_monotone_decreasing() {
        // p should decrease as g increases
        let p1 = chi2_sf_df1(1.0);
        let p2 = chi2_sf_df1(5.0);
        let p3 = chi2_sf_df1(10.0);
        assert!(p1 > p2 && p2 > p3, "expected p1>p2>p3, got {p1},{p2},{p3}");
    }

    // ---- g_test ----

    #[test]
    fn g_test_n11_less_than_half() {
        // n11=10, n=100 → n11 < n/2 → jc = n11 = 10
        let r = g_test(10.0, 100.0, 0.05);
        assert!((r.g - 2.0 * 10.0 * 0.05).abs() < 1e-15, "got g={}", r.g);
    }

    #[test]
    fn g_test_n11_greater_than_half() {
        // n11=60, n=100 → n11 > n/2 → jc = n - n11 = 40
        let r = g_test(60.0, 100.0, 0.05);
        assert!((r.g - 2.0 * 40.0 * 0.05).abs() < 1e-15, "got g={}", r.g);
    }

    #[test]
    fn g_test_n11_equal_half() {
        // n11=50, n=100 → n11 = n/2 → jc = n11 = 50 (not n-n11)
        let r = g_test(50.0, 100.0, 0.1);
        assert!((r.g - 2.0 * 50.0 * 0.1).abs() < 1e-15, "got g={}", r.g);
    }

    #[test]
    fn g_test_mi_zero() {
        let r = g_test(10.0, 100.0, 0.0);
        assert_eq!(r.g, 0.0);
        assert_eq!(r.p, 1.0);
    }

    #[test]
    fn g_test_result_fields() {
        let r = g_test(10.0, 100.0, 0.05);
        assert!(r.g > 0.0);
        assert!(r.p > 0.0 && r.p < 1.0);
    }

    // ---- normal_inverse_cdf ----

    #[test]
    fn qnorm_median() {
        let z = normal_inverse_cdf(0.5);
        assert!(z.abs() < 1e-15, "qnorm(0.5) should be 0, got {z}");
    }

    #[test]
    fn qnorm_975() {
        let z = normal_inverse_cdf(0.975);
        assert!((z - 1.959963984540054).abs() < 1e-12, "got {z}");
    }

    #[test]
    fn qnorm_025() {
        let z = normal_inverse_cdf(0.025);
        assert!((z + 1.959963984540054).abs() < 1e-12, "got {z}");
    }

    #[test]
    fn qnorm_001() {
        let z = normal_inverse_cdf(0.001);
        assert!((z + 3.0902323061678132).abs() < 1e-12, "got {z}");
    }

    #[test]
    fn qnorm_999() {
        let z = normal_inverse_cdf(0.999);
        assert!((z - 3.0902323061678132).abs() < 1e-12, "got {z}");
    }

    #[test]
    fn qnorm_symmetry() {
        // qnorm(p) = -qnorm(1-p)
        for &p in &[0.1, 0.25, 0.4, 0.6, 0.75, 0.9] {
            let z1 = normal_inverse_cdf(p);
            let z2 = normal_inverse_cdf(1.0 - p);
            assert!(
                (z1 + z2).abs() < 1e-12,
                "symmetry failed at p={p}: {z1} vs {z2}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "p must be in (0, 1)")]
    fn qnorm_p_zero_panics() {
        let _ = normal_inverse_cdf(0.0);
    }

    #[test]
    #[should_panic(expected = "p must be in (0, 1)")]
    fn qnorm_p_one_panics() {
        let _ = normal_inverse_cdf(1.0);
    }

    #[test]
    #[should_panic(expected = "p must be in (0, 1)")]
    fn qnorm_p_negative_panics() {
        let _ = normal_inverse_cdf(-0.1);
    }

    #[test]
    #[should_panic(expected = "p must be in (0, 1)")]
    fn qnorm_p_above_one_panics() {
        let _ = normal_inverse_cdf(1.5);
    }

    // ---- z_score ----

    #[test]
    fn z_score_05() {
        let z = z_score(0.05);
        assert!((z - 1.959963984540054).abs() < 1e-12, "got {z}");
    }

    #[test]
    fn z_score_01() {
        let z = z_score(0.01);
        assert!((z - 2.5758293035489008).abs() < 1e-12, "got {z}");
    }

    #[test]
    #[should_panic(expected = "alpha must be in (0, 1)")]
    fn z_score_alpha_zero_panics() {
        let _ = z_score(0.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be in (0, 1)")]
    fn z_score_alpha_one_panics() {
        let _ = z_score(1.0);
    }

    // ---- GTestResult ----

    #[test]
    fn gtest_result_equality() {
        let a = GTestResult { g: 1.0, p: 0.5 };
        let b = GTestResult { g: 1.0, p: 0.5 };
        let c = GTestResult { g: 2.0, p: 0.5 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
