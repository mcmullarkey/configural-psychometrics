---
ac: 3
depends_on: "AC-1"
risk: medium
status: in-progress
---

## AC spec: Statistical math module

### Executable Spec
- **predicate:**
  1. `wilson_lower(p, n, z)` matches C++ term order: `denom = 1 + z²/n`, `center = p + z²/(2n)`, `margin = z * sqrt(p(1-p)/n + z²/(4n²))`, result = `(center - margin) / denom`. Parity: `|rust - R_ref| / R_ref ≤ 1e-12`
  2. `safe_term(n_cell, m1, m2, n)` returns `(n_cell/n) * ln(n_cell*n/(m1*m2))` iff ALL four > 0; else 0.0. Four-arg guard (not two-arg).
  3. `mi_2x2(n11, nx1, nx2, n)` sums: `safe_term(n11,...) + safe_term(n10,...) + safe_term(n01,...) + safe_term(n00,...)` in that order.
  4. `g_test(n11, n, mi)` → `GTestResult{ g, p }` where `jc = min(n11, n-n11)`, `g = 2*jc*mi`, `p = chi2_sf_df1(g)`. Both branches tested (n11 < n/2 and n11 > n/2).
  5. `chi2_sf_df1(g)` = `erfc(sqrt(g/2))` via `libm` crate. g=0 → 1.0; g<0 → panic.
  6. `normal_inverse_cdf(p)` (AS241, hand-rolled ~80 lines). Parity: `|rust - R_qnorm| ≤ 1e-12` at p ∈ {0.001, 0.025, 0.5, 0.975, 0.999}. p ≤ 0 or p ≥ 1 → panic.
  7. `z_score(alpha)` = `normal_inverse_cdf(1 - alpha/2)`. alpha ∉ (0,1) → panic.
- **probe:** `cargo test stats -- --nocapture`
- **negative:**
  - `wilson_lower(0.6, 0.0, 1.96)` panics (n=0)
  - `mi_2x2(0.0, 0.0, 0.0, 0.0)` returns 0.0 (not NaN — safe_term four-arg guard)
  - `mi_2x2(1e-300, 2e-10, 3e-10, 100.0)` returns finite (subnormal passes guard)
  - `g_test` with n11 > n/2: `jc = n - n11` (not n11 — flipped-count correction)
  - Wilson: `4n²` in sqrt denominator (not `2n` — term-order pinning)
  - qnorm: must test ≥2 distinct input points (not just α=0.05)
- **verification:** code
- **fixture status:** NEW — `tests/golden/wilson_parity.json`, `tests/golden/mi_parity.json`, `tests/golden/gtest_parity.json`, `tests/golden/chi2_parity.json`, `tests/golden/qnorm_parity.json` (R-generated, ~20-50 points each)
- **rubric anchor:** §1 (GTestResult struct makes g,p inseparable; panic on impossible states), §2 (all 6 fns pure), §3 (stats.rs leaf module), §5 (one formula per function)

### Design Intent
- §1: Domain guards enforce invariants: n > 0, p ∈ [0,1], alpha ∈ (0,1). Panic-at-boundary prevents poisoned intermediates. `GTestResult{g, p}` struct.
- §2: 6 pure functions: f64 in → f64 out (or struct). No I/O, no randomness, no global state.
- §3: `stats.rs` leaf module. No deps on bitset or engine modules. Engines import stats, never reverse.
- §4: Header: "Statistical primitives for sufficiency testing (Wilson) and significance testing (MI/G-test/chi2/qnorm). Ported from emsc_v4.cpp + pairmi_v5.cpp. Numerical parity ~1e-12 vs R reference. NOT: data loading, bitset operations, engine ascent logic."
- §5: One function per formula. `wilson_lower` = Wilson. `mi_2x2` = four-term sum. `g_test` = flip + multiply + chi2. `chi2_sf_df1` = erfc. `normal_inverse_cdf` = AS241. `z_score` = thin wrapper.

### Technical Context
- Files: `src/stats.rs` (NEW — 6 functions + GTestResult struct), `tests/stats_parity.rs` (NEW), `tests/golden/*.json` (NEW), `Cargo.toml` (MODIFY — add `libm = "0.2"`, `approx = "0.5"` dev-dep), `scripts/generate_golden.R` (NEW)
- Module name: `stats.rs` (not `math.rs` — more precise scope)
- erfc: `libm` crate (std::f64::erfc does NOT exist in Rust std)
- qnorm: hand-rolled AS241 (~80 lines, public domain algorithm). Fallback: `statrs` crate if parity fails.
- C++ refs: emsc_v4.cpp:57-62 (Wilson), emsc_v4.cpp:93 (z=qnorm), pairmi_v5.cpp:70-85 (safe_term, mi_2x2), pairmi_v5.cpp:146-150 (G-test)

### Progress
- [x] implement — complete (2026-07-27)

### Decision Log
- Used R's qnorm source coefficients (not original AS241 paper) — R has improved coefficients for A[5], A[7], C[1], and all D coefficients. Original AS241 paper coefficients give wrong results (~3.37 instead of ~1.96 for qnorm(0.975)).
- Polynomial evaluation order: highest-degree-first (matching R source). Original implementation used lowest-degree-first (standard Horner), which produced wrong results because the coefficient arrays were stored in the wrong order.
- Added region 3 (5 < r <= 27) with 8-coefficient polynomials from R source. Original AS241 only had 6-coefficient polynomials for this region.
- Region 4 (r > 27): used zeroth-order asymptotic `r * sqrt(2)`. R's full asymptotic expansion (xs_1..xs_4) omitted — only relevant for p < 2.5e-317, not in golden fixtures.
- Added `serde_json` dev-dependency for parsing golden fixture JSON files (in addition to `approx`).

### Surprises & Discoveries
- AS241 coefficients from the original Wichura (1988) paper differ from R's qnorm source. R uses improved coefficients (A[5]: 67265.77 vs 67313.48, A[7]: 2509.0809287301 vs 2509.0809288301). Using the paper's coefficients produced qnorm(0.975) = 3.366 instead of 1.960 — a 70% error. Resolution: extracted exact coefficients from R's `src/nmath/qnorm.c` on GitHub.
- R's qnorm evaluates polynomials highest-degree-first (e.g., `r * A[7] + A[6]`), not lowest-degree-first. Standard Horner's method (starting from constant term) with the same coefficients produces a completely different polynomial. Resolution: reversed the Horner iteration to start from the highest-degree coefficient.
- R's qnorm has 3 tail regions (r ≤ 5, 5 < r ≤ 27, r > 27), not 2 as in the original AS241. The middle region uses 8-coefficient polynomials (E/F), not 6-coefficient as in the paper. Resolution: added the third region with correct coefficients from R source.

### Idempotence & Recovery
- Safe retry: `cargo test stats` re-runs parity tests
- Rollback: `git checkout -- src/stats.rs Cargo.toml`
