//! Error type for configural matrix validation.
//!
//! See [`crate`] docs for the crate-level overview.

use std::fmt;

/// Exhaustive error enum for [`BinaryMatrix`](crate::BinaryMatrix) construction
/// failures.
///
/// Each variant names a distinct validation failure so callers can match
/// exhaustively. The flat `&[f64]` + `(n_rows, n_cols)` input shape makes
/// jaggedness unrepresentable — `DimensionMismatch` is the only structural
/// error.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfiguralError {
    /// A value in the input is not exactly `0.0` or `1.0`.
    ///
    /// `row` and `col` are 0-based matrix coordinates (row-major flat index
    /// decomposed as `row = index / n_cols`, `col = index % n_cols`).
    /// `value` is the offending `f64`.
    ///
    /// Note: `-0.0` passes validation because `-0.0 == 0.0` per IEEE 754.
    NonBinaryValue {
        /// 0-based row index of the offending value.
        row: usize,
        /// 0-based column index of the offending value.
        col: usize,
        /// The value that failed the 0/1 check.
        value: f64,
    },

    /// The vocabulary (number of rows / original variables) exceeds 64.
    ///
    /// Each column is packed into `ceil(n_rows / 64)` `u64` words; the
    /// `V <= 64` limit keeps each column in a single word, matching the
    /// C++ `pairmi` engine.
    VocabularyTooLarge {
        /// The vocabulary size that was provided.
        got: usize,
        /// The maximum allowed vocabulary size (64).
        max: usize,
    },

    /// The matrix has zero rows or zero columns.
    EmptyMatrix,

    /// `data.len()` does not equal `n_rows * n_cols`.
    DimensionMismatch {
        /// The expected length (`n_rows * n_cols`).
        expected: usize,
        /// The actual `data.len()`.
        got: usize,
    },
}

impl fmt::Display for ConfiguralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfiguralError::NonBinaryValue { row, col, value } => {
                write!(
                    f,
                    "non-binary value {value} at row {row}, col {col} (expected 0.0 or 1.0)"
                )
            }
            ConfiguralError::VocabularyTooLarge { got, max } => {
                write!(f, "vocabulary size {got} exceeds maximum {max}")
            }
            ConfiguralError::EmptyMatrix => {
                write!(f, "matrix must have at least one row and one column")
            }
            ConfiguralError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "data length {got} does not match n_rows * n_cols = {expected}"
                )
            }
        }
    }
}

impl std::error::Error for ConfiguralError {}
