---
ac: 2
depends_on: "AC-1"
risk: low
status: in-progress
---

## AC spec: Bit-parallel bitset core

### Executable Spec
- **predicate:**
  - `popcount(0xFFFF_FFFF_FFFF_FFFF) == 64` AND `popcount(0) == 0`
  - `pack_column(&[]) == vec![]` AND `pack_column(&[true]) == vec![1]` AND `pack_column(&[true; 64]) == vec![0xFFFF_FFFF_FFFF_FFFF]` AND `pack_column(&[true; 65]) == vec![0xFFFF_FFFF_FFFF_FFFF, 1]`
  - `and_popcount(&[0b1010_u64], &[0b1100_u64]) == 1` AND `and_popcount(&[0], &[0xFFFF_FFFF_FFFF_FFFF]) == 0`
  - `extract_members(0b1010_u64).collect::<Vec<_>>() == vec![1u32, 3]` AND `extract_members(0u64).collect::<Vec<_>>() == vec![]`
  - `max_element(0b1010_u64) == Some(3u32)` AND `max_element(0u64) == None`
- **probe:** `cargo test bitset_core -- --nocapture`
- **negative:**
  - All-zero column packs to all-zero words; `extract_members(0)` yields empty (not panic)
  - `popcount` must use `u64::count_ones` — `(x as u32).count_ones()` truncates high bits
  - `pack_column` must OR bits into zeroed words — short-circuiting without clearing leaves stale bits
  - `max_element(0)` must return `None` — `63u32.wrapping_sub(64)` underflows
- **verification:** code
- **fixture status:** C++ refs: emsc_v4.cpp:34-40 (popcount64), emsc_v4.cpp:42-47 (pack_column), emsc_v4.cpp:49-54 (and_popcount), emsc_v4.cpp:194 (max_element/clz), emsc_v4.cpp:350-354 (extract_members/ctz), pairmi_v5.cpp:63-67 (popcnt_and)
- **rubric anchor:** §1 (u64 masks, &[u64] slices, u32 counts), §2 (pure transforms), §3 (bitset owns only packing/popcount/iteration), §5 (one function per primitive)

### Design Intent
- §1: `u64` vocab masks (V≤64), `&[u64]` word slices, `u32` counts. `max_element -> Option<u32>` makes "no set bit" unrepresentable. `extract_members -> impl Iterator<Item=u32>` yields 0-based indices.
- §2: All five pure. `pack_column` returns `Vec<u64>` (allocates) rather than mutating caller buffer. `and_popcount` takes `&[u64]` immutably.
- §3: Bitset module owns ONLY: word packing, AND+popcount, single-word member iteration, highest-bit. Does NOT own: Wilson, MI, antichain reduce, config storage.
- §4: Module header: bit-parallel primitives for packed column indicators and vocabulary masks; does NOT perform inference, sufficiency gating, or antichain reduction.
- §5: One function per primitive. `popcount` wraps `u64::count_ones`. `pack_column` zeroes then ORs. `and_popcount` fuses AND+popcount per word. `extract_members` uses `trailing_zeros()` + `mask &= mask - 1`. `max_element` uses `63 - leading_zeros` guarded by `Option`.

### Technical Context
- Files: `src/bitset.rs` (new), `src/lib.rs` (mod declaration), `tests/bitset_core.rs` (test matrix)
- No external dependencies — all Rust stable intrinsics
- Rust `u64::count_ones`, `trailing_zeros`, `leading_zeros` are safe for 0 (return 64) but `63 - 64` underflows → `max_element` returns `Option`
- C++ `extract_members` yields 1-based (R convention); Rust core yields 0-based; +1 offset belongs to FFI layer

### Progress
- [x] implement — complete (2026-07-27)

### Decision Log
- Used `n.div_ceil(64)` instead of `(n + 63) / 64` per clippy `manual_div_ceil` lint (Rust 1.94+).
- `extract_members` uses `std::iter::from_fn` with `FnMut` closure capturing `mask` by move — allows mutation of `m &= m - 1` inside the closure.

### Surprises & Discoveries
- Initial test `pack_column_mixed_pattern_bit_positions` had a hand-written 64-element vector that was actually 63 elements long (miscounted). Replaced with programmatic `vec![false; 64]` + index assignment to eliminate human counting errors.

### Idempotence & Recovery
- Safe retry: `cargo test bitset_core` re-runs
- Rollback: `git checkout -- src/bitset.rs`
