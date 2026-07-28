//! N-D quadrant classification: join necessity × dispensability.
//!
//! Provides:
//! - [`Quadrant`] enum: the four N-D quadrants.
//! - [`NdQuadrantConfig`]: cutoffs + y1 selector.
//! - [`classify`]: map (ν, δ) → quadrant with override priority.
//!
//! Does NOT provide:
//! - Computation of necessity (ν) or dispensability (δ) themselves —
//!   those arrive from [`crate::diagnostics::necessity`] and
//!   [`crate::diagnostics::dispensability`].
//! - Batch classification over an index — callers loop and select the
//!   appropriate δ / δ_y1 field based on `config.use_y1`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quadrant {
    Keystone,
    SpecialistGateway,
    Eliminable,
    Inert,
}

#[derive(Debug, Clone)]
pub struct NdQuadrantConfig {
    pub necessity_cutoff: f64,      // default 0.95
    pub dispensability_cutoff: f64, // default 0.05 (indispensable threshold: δ≤0.05 = indispensable)
    pub use_y1: bool,
}

impl NdQuadrantConfig {
    pub fn new(necessity_cutoff: f64, dispensability_cutoff: f64, use_y1: bool) -> Self {
        assert!(
            (0.0..=1.0).contains(&necessity_cutoff),
            "necessity_cutoff must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&dispensability_cutoff),
            "dispensability_cutoff must be in [0, 1]"
        );
        Self {
            necessity_cutoff,
            dispensability_cutoff,
            use_y1,
        }
    }
}

impl Default for NdQuadrantConfig {
    fn default() -> Self {
        Self {
            necessity_cutoff: 0.95,
            dispensability_cutoff: 0.05,
            use_y1: false,
        }
    }
}

/// Classify an element into a quadrant based on necessity (ν) and dispensability (δ).
///
/// Override order: None→Inert→Keystone→SpecialistGateway→Eliminable
///
/// - `None`/`NaN` inputs → `None`.
/// - `ν == 0.0` → `Inert` (overrides everything, including Eliminable).
/// - `ν >= necessity_cutoff` AND `δ <= dispensability_cutoff` → `Keystone`.
/// - `ν < necessity_cutoff` AND `δ <= dispensability_cutoff` → `SpecialistGateway`.
/// - `δ > dispensability_cutoff` → `Eliminable` (catches redundant_ubiquitous too).
pub fn classify(
    nu: Option<f64>,
    delta: Option<f64>,
    config: &NdQuadrantConfig,
) -> Option<Quadrant> {
    // 1. None/NaN → None
    let nu = nu?;
    let delta = delta?;
    if nu.is_nan() || delta.is_nan() {
        return None;
    }

    // 2. ν == 0.0 → Inert (overrides everything)
    if nu == 0.0 {
        return Some(Quadrant::Inert);
    }

    // 3. ν >= cutoff AND δ <= cutoff → Keystone
    if nu >= config.necessity_cutoff && delta <= config.dispensability_cutoff {
        return Some(Quadrant::Keystone);
    }

    // 4. ν < cutoff AND δ <= cutoff → SpecialistGateway
    if nu < config.necessity_cutoff && delta <= config.dispensability_cutoff {
        return Some(Quadrant::SpecialistGateway);
    }

    // 5. δ > cutoff → Eliminable (catches redundant_ubiquitous too)
    Some(Quadrant::Eliminable)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NdQuadrantConfig {
        NdQuadrantConfig::default()
    }

    // ---- None / NaN propagation ----

    #[test]
    fn nu_none_returns_none() {
        assert_eq!(classify(None, Some(0.05), &cfg()), None);
    }

    #[test]
    fn delta_none_returns_none() {
        assert_eq!(classify(Some(0.95), None, &cfg()), None);
    }

    #[test]
    fn both_none_returns_none() {
        assert_eq!(classify(None, None, &cfg()), None);
    }

    #[test]
    fn nu_nan_returns_none() {
        assert_eq!(classify(Some(f64::NAN), Some(0.05), &cfg()), None);
    }

    #[test]
    fn delta_nan_returns_none() {
        assert_eq!(classify(Some(0.95), Some(f64::NAN), &cfg()), None);
    }

    // ---- ν == 0.0 → Inert (overrides Eliminable) ----

    #[test]
    fn nu_zero_with_dispensable_delta_is_inert() {
        // δ > cutoff would normally be Eliminable, but ν=0.0 overrides.
        assert_eq!(
            classify(Some(0.0), Some(0.99), &cfg()),
            Some(Quadrant::Inert)
        );
    }

    #[test]
    fn nu_zero_with_indispensable_delta_is_inert() {
        // δ <= cutoff would normally be SpecialistGateway, but ν=0.0 overrides.
        assert_eq!(
            classify(Some(0.0), Some(0.05), &cfg()),
            Some(Quadrant::Inert)
        );
    }

    // ---- Boundary-exact quadrant mapping ----

    #[test]
    fn keystone_boundary_exact() {
        // ν=0.95 (>= cutoff) AND δ=0.05 (<= cutoff) → Keystone
        assert_eq!(
            classify(Some(0.95), Some(0.05), &cfg()),
            Some(Quadrant::Keystone)
        );
    }

    #[test]
    fn specialist_gateway_just_below_cutoff() {
        // ν=0.949 (< cutoff) AND δ=0.05 (<= cutoff) → SpecialistGateway
        assert_eq!(
            classify(Some(0.949), Some(0.05), &cfg()),
            Some(Quadrant::SpecialistGateway)
        );
    }

    #[test]
    fn eliminable_just_above_dispensability_cutoff() {
        // ν=0.95 (>= cutoff) AND δ=0.051 (> cutoff) → Eliminable
        assert_eq!(
            classify(Some(0.95), Some(0.051), &cfg()),
            Some(Quadrant::Eliminable)
        );
    }

    #[test]
    fn redundant_ubiquitous_is_eliminable() {
        // ν=0.99 (>= cutoff) AND δ=0.99 (> cutoff) → Eliminable
        assert_eq!(
            classify(Some(0.99), Some(0.99), &cfg()),
            Some(Quadrant::Eliminable)
        );
    }

    #[test]
    fn inert_low_nu_high_delta() {
        // ν < cutoff AND δ > cutoff → falls through to Eliminable per spec
        // (step 5 catches all δ > cutoff). Verify the catch-all.
        assert_eq!(
            classify(Some(0.5), Some(0.5), &cfg()),
            Some(Quadrant::Eliminable)
        );
    }

    // ---- use_y1 config ----

    #[test]
    fn use_y1_with_none_delta_returns_none() {
        // When use_y1=true, the caller selects δ_y1; if that is None → None.
        let cfg_y1 = NdQuadrantConfig::new(0.95, 0.05, true);
        assert_eq!(classify(Some(0.95), None, &cfg_y1), None);
    }

    #[test]
    fn use_y1_true_still_classifies_when_delta_present() {
        let cfg_y1 = NdQuadrantConfig::new(0.95, 0.05, true);
        assert_eq!(
            classify(Some(0.95), Some(0.05), &cfg_y1),
            Some(Quadrant::Keystone)
        );
    }

    // ---- Cutoff validation ----

    #[test]
    #[should_panic(expected = "necessity_cutoff must be in [0, 1]")]
    fn necessity_cutoff_above_one_panics() {
        let _ = NdQuadrantConfig::new(1.5, 0.05, false);
    }

    #[test]
    #[should_panic(expected = "necessity_cutoff must be in [0, 1]")]
    fn necessity_cutoff_below_zero_panics() {
        let _ = NdQuadrantConfig::new(-0.1, 0.05, false);
    }

    #[test]
    #[should_panic(expected = "dispensability_cutoff must be in [0, 1]")]
    fn dispensability_cutoff_above_one_panics() {
        let _ = NdQuadrantConfig::new(0.95, 1.5, false);
    }

    #[test]
    #[should_panic(expected = "dispensability_cutoff must be in [0, 1]")]
    fn dispensability_cutoff_below_zero_panics() {
        let _ = NdQuadrantConfig::new(0.95, -0.1, false);
    }

    #[test]
    fn cutoffs_at_boundaries_are_valid() {
        // 0.0 and 1.0 are inclusive endpoints — must not panic.
        let c0 = NdQuadrantConfig::new(0.0, 0.0, false);
        assert_eq!(c0.necessity_cutoff, 0.0);
        assert_eq!(c0.dispensability_cutoff, 0.0);
        let c1 = NdQuadrantConfig::new(1.0, 1.0, false);
        assert_eq!(c1.necessity_cutoff, 1.0);
        assert_eq!(c1.dispensability_cutoff, 1.0);
    }

    // ---- Default config ----

    #[test]
    fn default_config_values() {
        let d = NdQuadrantConfig::default();
        assert_eq!(d.necessity_cutoff, 0.95);
        assert_eq!(d.dispensability_cutoff, 0.05);
        assert!(!d.use_y1);
    }
}
