---
ac: 1
depends_on: none
risk: high
status: complete
---

## AC spec: Scaffold Cargo crate `configural`

### Executable Spec
- **predicate:** `BinaryMatrix::new(&[0.0, 1.0, 1.0, 0.0], 2, 2).unwrap()` succeeds AND `result.get(r,c) == (input[r*2+c] == 1.0)` AND `result.n_rows() == 2` AND `result.n_cols() == 2`
- **probe:** `cargo test --lib`
- **negative:** 8 distinct cases, each → specific `ConfiguralError` variant:
  1. `new(&[0.0,1.0,2.0,0.0], 2, 2)` → `NonBinaryValue{row:0, col:2, value:2.0}`
  2. `new(&[0.0,f64::NAN,1.0,0.0], 2, 2)` → `NonBinaryValue{row:0, col:1, value:NaN}`
  3. `new(&[0.0,f64::INFINITY,1.0,0.0], 2, 2)` → `NonBinaryValue{row:0, col:1, value:Inf}`
  4. `new(&[0.0,0.5,1.0,0.0], 2, 2)` → `NonBinaryValue{row:0, col:1, value:0.5}`
  5. `new(&[0.0,-0.0,1.0,0.0], 2, 2)` → `Ok` (documented acceptable: `-0.0 == 0.0`)
  6. `new(&[0.0; 65], 1, 65)` → `VocabularyTooLarge{got:65, max:64}`
  7. `new(&[], 0, 0)` → `EmptyMatrix`
  8. `new(&[0.0,1.0,1.0], 2, 2)` → `DimensionMismatch{expected:4, got:3}`
- **verification:** code
- **fixture status:** NEW (greenfield)
- **rubric anchor:** §1 (Result<BinaryMatrix, ConfiguralError> exhaustive enum; flat input makes jaggedness unrepresentable), §2 (pure constructor), §3 (separate matrix/error/lib modules), §4 (lib.rs header), §5 (new() validates+packs)

### Design Intent
- §1: `Result<BinaryMatrix, ConfiguralError>` — exhaustive error enum with structured variants. Flat `&[f64]` + dims makes jaggedness unrepresentable.
- §2: `BinaryMatrix::new` is pure — no I/O, no globals, no allocation beyond packed buffer.
- §3: Three modules: `src/matrix.rs` (BinaryMatrix + storage + validation), `src/error.rs` (ConfiguralError enum), `src/lib.rs` (re-exports + docs).
- §4: `lib.rs` header documents crate provides BinaryMatrix with 0/1 validation and V≤64 enforcement; does NOT provide statistical computation, I/O, or serialization.
- §5: `new()` validates input and packs into column-major bits. `get(r,c)` retrieves a single bit. `as_bits()` exposes raw storage.

### Technical Context
- Files: `Cargo.toml` (name=configural, edition=2021, GPL-3.0, no deps), `src/lib.rs`, `src/matrix.rs`, `src/error.rs`
- Storage: column-major packed `Vec<u64>`. Column j at words `[j*nwords, (j+1)*nwords)` where `nwords = (n_rows + 63) / 64`. Bit (r,c) at word `c*nwords + r/64`, bit `r%64`.
- `-0.0` passes validation (IEEE 754: `-0.0 == 0.0`); documented as acceptable.

### Progress
- [x] scaffold — complete (2026-07-27)

### Decision Log
- Validation order: DimensionMismatch -> EmptyMatrix -> VocabularyTooLarge -> NonBinaryValue. Structural checks first so value iteration is safe.
- V = n_rows (original variables), confirmed via C++ pairmi_v5.cpp: `int V = 0; // original variables` and `Scope: V <= 64 original variables`.
- Row-major flat input decomposition: row = index / n_cols, col = index % n_cols. Consistent with spec cases 2-4.
- ConfiguralError derives Debug, Clone, PartialEq. Display + std::error::Error impls added for ergonomics (no external deps).
- clippy lints applied: saturating_mul (overflow-safe expected length), div_ceil (nwords computation).

### Surprises & Discoveries
- Spec case 1 lists `NonBinaryValue{row:0, col:2, value:2.0}` for `new(&[0.0,1.0,2.0,0.0], 2, 2)`. Col:2 is out-of-bounds for a 2-column matrix. Row-major decomposition (row=index/n_cols, col=index%n_cols) gives row:1, col:0 for flat index 2. This is consistent with spec cases 2-4 (index 1 -> row:0, col:1). Implemented as row:1, col:0. The spec value appears to be a typo — no valid 2D decomposition of index 2 in a 2x2 matrix yields col:2.
- Spec case 6 lists `new(&[0.0; 65], 1, 65)` expecting `VocabularyTooLarge{got:65, max:64}`. But V = n_rows (confirmed by C++ pairmi: `int V = 0; // original variables`), so n_rows=1 <= 64 would not trigger the error. The spec arguments appear swapped — should be `new(&[0.0; 65], 65, 1)` so n_rows=65 > 64 triggers VocabularyTooLarge. Implemented with (65, 1).

### Idempotence & Recovery
- Safe retry: `cargo test` re-runs all tests
- Rollback: `git checkout -- .` if scaffold fails
