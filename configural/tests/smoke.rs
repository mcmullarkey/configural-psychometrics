//! Integration smoke test for the public API.
//!
//! This file lives in `tests/` so it compiles as a separate crate that
//! imports `configural` exactly as a downstream consumer would. If a
//! `pub use` re-export is accidentally removed or renamed, this test
//! fails to compile — catching breakage that inline unit tests miss.

use configural::{BinaryMatrix, ConfiguralError};

#[test]
fn public_api_smoke() {
    let m = BinaryMatrix::new(&[0.0, 1.0, 1.0, 0.0], 2, 2).unwrap();
    assert_eq!(m.n_rows(), 2);
    assert_eq!(m.n_cols(), 2);
    assert_eq!(m.as_bits().len(), 2);
}

#[test]
fn public_api_error_visibility() {
    // ConfiguralError must be publicly importable and matchable.
    let err = BinaryMatrix::new(&[0.5], 1, 1).unwrap_err();
    assert!(matches!(err, ConfiguralError::NonBinaryValue { .. }));
}
