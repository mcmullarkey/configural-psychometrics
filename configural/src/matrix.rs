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
}
