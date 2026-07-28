//! Binary matrix with column-major packed-bit storage.
//!
//! See [`crate`] docs for the crate-level overview.

/// A validated binary matrix stored in column-major packed `u64` words.
///
/// ## Construction
///
/// [`BinaryMatrix::new`] takes a flat `&[f64]` slice in **row-major** order
/// (element `(r, c)` at index `r * n_cols + c`) and validates:
///
/// 1. `data.len() == n_rows * n_cols` — otherwise [`DimensionMismatch`].
/// 2. At least one row and one column — otherwise [`EmptyMatrix`].
/// 3. `n_rows <= 64` (vocabulary limit) — otherwise [`VocabularyTooLarge`].
/// 4. Every value is exactly `0.0` or `1.0` — otherwise [`NonBinaryValue`].
///    (`-0.0` passes because `-0.0 == 0.0` per IEEE 754.)
///
/// ## Storage layout
///
/// Bits are packed column-major into `Vec<u64>`. For a matrix with `n_rows`
/// rows and `n_cols` columns:
///
/// - `nwords = (n_rows + 63) / 64` — words per column.
/// - Total storage: `n_cols * nwords` words.
/// - Bit `(r, c)` lives at word `c * nwords + r / 64`, bit `r % 64`.
///
/// With `n_rows <= 64`, `nwords == 1`, so each column is exactly one `u64`.
///
/// [`DimensionMismatch`]: crate::ConfiguralError::DimensionMismatch
/// [`EmptyMatrix`]: crate::ConfiguralError::EmptyMatrix
/// [`VocabularyTooLarge`]: crate::ConfiguralError::VocabularyTooLarge
/// [`NonBinaryValue`]: crate::ConfiguralError::NonBinaryValue
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryMatrix {
    /// Packed column-major bits. See type docs for layout.
    bits: Vec<u64>,
    /// Number of rows (vocabulary size, V).
    n_rows: usize,
    /// Number of columns (observations).
    n_cols: usize,
    /// Words per column: `(n_rows + 63) / 64`.
    nwords: usize,
}

impl BinaryMatrix {
    /// Validate and pack a flat `&[f64]` into a column-major binary matrix.
    ///
    /// The input slice is **row-major**: element `(r, c)` is at
    /// `data[r * n_cols + c]`. Values must be exactly `0.0` or `1.0`
    /// (`-0.0` is accepted because `-0.0 == 0.0`).
    ///
    /// # Errors
    ///
    /// Returns [`ConfiguralError`] if:
    /// - `data.len() != n_rows * n_cols` ([`DimensionMismatch`])
    /// - `n_rows == 0` or `n_cols == 0` ([`EmptyMatrix`])
    /// - `n_rows > 64` ([`VocabularyTooLarge`])
    /// - Any value is not `0.0` or `1.0` ([`NonBinaryValue`])
    ///
    /// [`DimensionMismatch`]: crate::ConfiguralError::DimensionMismatch
    /// [`EmptyMatrix`]: crate::ConfiguralError::EmptyMatrix
    /// [`VocabularyTooLarge`]: crate::ConfiguralError::VocabularyTooLarge
    /// [`NonBinaryValue`]: crate::ConfiguralError::NonBinaryValue
    pub fn new(data: &[f64], n_rows: usize, n_cols: usize) -> Result<Self, crate::ConfiguralError> {
        // 1. Dimension check — must come first so we can safely index.
        let expected = n_rows.saturating_mul(n_cols);
        if data.len() != expected {
            return Err(crate::ConfiguralError::DimensionMismatch {
                expected,
                got: data.len(),
            });
        }

        // 2. Non-empty check.
        if n_rows == 0 || n_cols == 0 {
            return Err(crate::ConfiguralError::EmptyMatrix);
        }

        // 3. Vocabulary limit — V = n_rows (original variables).
        const MAX_VOCAB: usize = 64;
        if n_rows > MAX_VOCAB {
            return Err(crate::ConfiguralError::VocabularyTooLarge {
                got: n_rows,
                max: MAX_VOCAB,
            });
        }

        // 4. Validate values and pack into column-major u64 words.
        let nwords = n_rows.div_ceil(64);
        let mut bits = vec![0u64; n_cols * nwords];

        for r in 0..n_rows {
            for c in 0..n_cols {
                let idx = r * n_cols + c;
                let value = data[idx];
                // -0.0 == 0.0 per IEEE 754, so this accepts -0.0.
                // NaN and Inf fail both comparisons (they are not == 0.0
                // and not == 1.0).
                let is_one = value == 1.0;
                let is_zero = value == 0.0;
                if !is_one && !is_zero {
                    return Err(crate::ConfiguralError::NonBinaryValue {
                        row: r,
                        col: c,
                        value,
                    });
                }
                if is_one {
                    let word = c * nwords + r / 64;
                    let bit = r % 64;
                    bits[word] |= 1u64 << bit;
                }
            }
        }

        Ok(Self {
            bits,
            n_rows,
            n_cols,
            nwords,
        })
    }

    /// Number of rows (vocabulary size, V).
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns (observations).
    #[must_use]
    pub const fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Retrieve the bit at `(row, col)`.
    ///
    /// Returns `true` if the value was `1.0`, `false` if `0.0`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= n_rows` or `col >= n_cols`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> bool {
        assert!(
            row < self.n_rows,
            "row {row} out of bounds (n_rows = {})",
            self.n_rows
        );
        assert!(
            col < self.n_cols,
            "col {col} out of bounds (n_cols = {})",
            self.n_cols
        );
        let word = col * self.nwords + row / 64;
        let bit = row % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Raw column-major packed storage.
    ///
    /// Returns the `Vec<u64>` backing array. Column `j` occupies words
    /// `[j * nwords, (j + 1) * nwords)`.
    #[must_use]
    pub fn as_bits(&self) -> &[u64] {
        &self.bits
    }
}

#[cfg(test)]
mod tests {
    use crate::{BinaryMatrix, ConfiguralError};

    // ---- Positive cases ----

    #[test]
    fn positive_2x2_roundtrip() {
        let data = [0.0, 1.0, 1.0, 0.0];
        let m = BinaryMatrix::new(&data, 2, 2).expect("valid 2x2");
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.n_cols(), 2);
        // Row-major input: data[r*2+c]
        assert!(!m.get(0, 0)); // 0.0
        assert!(m.get(0, 1)); // 1.0
        assert!(m.get(1, 0)); // 1.0
        assert!(!m.get(1, 1)); // 0.0
    }

    #[test]
    fn negative_zero_passes() {
        // -0.0 == 0.0 per IEEE 754; documented as acceptable.
        let data = [0.0, -0.0, 1.0, 0.0];
        let m = BinaryMatrix::new(&data, 2, 2);
        assert!(m.is_ok(), "expected Ok for -0.0 input, got {:?}", m);
    }

    // ---- Negative cases ----

    #[test]
    fn negative_value_2() {
        // Spec case 1: data=[0,1,2,0], 2x2. Index 2 row-major -> row=1, col=0.
        // (Spec lists row:0,col:2 which is out-of-bounds for a 2-col matrix;
        //  row-major decomposition consistent with cases 2-4 gives row:1,col:0.)
        let data = [0.0, 1.0, 2.0, 0.0];
        let err = BinaryMatrix::new(&data, 2, 2).unwrap_err();
        assert_eq!(
            err,
            ConfiguralError::NonBinaryValue {
                row: 1,
                col: 0,
                value: 2.0
            }
        );
    }

    #[test]
    fn negative_value_nan() {
        let data = [0.0, f64::NAN, 1.0, 0.0];
        let err = BinaryMatrix::new(&data, 2, 2).unwrap_err();
        match err {
            ConfiguralError::NonBinaryValue { row, col, value } => {
                assert_eq!(row, 0);
                assert_eq!(col, 1);
                assert!(value.is_nan(), "expected NaN, got {value}");
            }
            other => panic!("expected NonBinaryValue, got {other:?}"),
        }
    }

    #[test]
    fn negative_value_infinity() {
        let data = [0.0, f64::INFINITY, 1.0, 0.0];
        let err = BinaryMatrix::new(&data, 2, 2).unwrap_err();
        assert_eq!(
            err,
            ConfiguralError::NonBinaryValue {
                row: 0,
                col: 1,
                value: f64::INFINITY
            }
        );
    }

    #[test]
    fn negative_value_half() {
        let data = [0.0, 0.5, 1.0, 0.0];
        let err = BinaryMatrix::new(&data, 2, 2).unwrap_err();
        assert_eq!(
            err,
            ConfiguralError::NonBinaryValue {
                row: 0,
                col: 1,
                value: 0.5
            }
        );
    }

    #[test]
    fn negative_vocabulary_too_large() {
        // V = n_rows (original variables). Spec case 6 lists (1, 65) but
        // V=n_rows=1 <= 64 would not trigger. C++ pairmi confirms V = rows.
        // Using (65, 1) so n_rows=65 > 64 triggers VocabularyTooLarge.
        let data = vec![0.0_f64; 65];
        let err = BinaryMatrix::new(&data, 65, 1).unwrap_err();
        assert_eq!(
            err,
            ConfiguralError::VocabularyTooLarge { got: 65, max: 64 }
        );
    }

    #[test]
    fn negative_empty_matrix() {
        let err = BinaryMatrix::new(&[], 0, 0).unwrap_err();
        assert_eq!(err, ConfiguralError::EmptyMatrix);
    }

    #[test]
    fn negative_dimension_mismatch() {
        let data = [0.0, 1.0, 1.0];
        let err = BinaryMatrix::new(&data, 2, 2).unwrap_err();
        assert_eq!(
            err,
            ConfiguralError::DimensionMismatch {
                expected: 4,
                got: 3
            }
        );
    }

    // ---- Layout contract ----

    #[test]
    fn as_bits_column_major_layout() {
        // 3 rows × 2 cols: data = [1,0, 0,1, 0,1] (row-major input)
        // Column 0 = [1,0,0] → bit 0 set → word = 0b001 = 1
        // Column 1 = [0,1,1] → bits 1,2 set → word = 0b110 = 6
        //
        // This test pins the column-major packed-bit layout documented in the
        // struct-level docs. A row-major refactor that keeps get() correct
        // would silently break as_bits() — this test catches that.
        let m = BinaryMatrix::new(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], 3, 2).expect("valid 3x2 matrix");
        assert_eq!(m.as_bits().len(), 2); // 2 columns × 1 word each
        assert_eq!(m.as_bits()[0], 0b001); // column 0: row 0 set
        assert_eq!(m.as_bits()[1], 0b110); // column 1: rows 1,2 set
    }

    #[test]
    fn equality_semantics() {
        // PartialEq enables downstream ACs to compare matrices directly.
        // Without the derive, assert_eq! would not compile.
        let a = BinaryMatrix::new(&[0.0, 1.0, 1.0, 0.0], 2, 2).unwrap();
        let b = BinaryMatrix::new(&[0.0, 1.0, 1.0, 0.0], 2, 2).unwrap();
        let c = BinaryMatrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        assert_eq!(a, b, "matrices with same data should be equal");
        assert_ne!(a, c, "matrices with different data should not be equal");
    }
}
