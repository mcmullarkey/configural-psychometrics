//! Bit-parallel primitives for packed column indicators and vocabulary masks.
//!
//! This module owns ONLY: word packing, fused AND+popcount, single-word
//! member iteration, and highest-bit extraction. It does NOT perform
//! inference, sufficiency gating, or antichain reduction.
//!
//! ## Vocabulary
//!
//! - `u64` — vocabulary masks (V <= 64 variables per column).
//! - `&[u64]` — word slices for multi-word columns.
//! - `u32` — bit indices and popcount results.
//!
//! ## C++ reference
//!
//! - `popcount64`: `emsc_v4.cpp:34-40`
//! - `pack_column`: `emsc_v4.cpp:42-47`
//! - `and_popcount`: `emsc_v4.cpp:49-54`
//! - `max_element` (clz): `emsc_v4.cpp:194`
//! - `extract_members` (ctz): `emsc_v4.cpp:350-354`

/// Count the number of set bits in a `u64` word.
///
/// Wraps the hardware-accelerated `u64::count_ones` intrinsic. The full
/// 64-bit width is counted — `(x as u32).count_ones()` would truncate the
/// high 32 bits and silently undercount.
///
/// # Examples
///
/// ```
/// use configural::bitset::popcount;
/// assert_eq!(popcount(0xFFFF_FFFF_FFFF_FFFF), 64);
/// assert_eq!(popcount(0), 0);
/// ```
pub fn popcount(x: u64) -> u32 {
    x.count_ones()
}

/// Pack a column of boolean indicators into `u64` words.
///
/// Produces `ceil(n / 64)` words. Bit `i` of the input lands at word
/// `i >> 6` (i.e. `i / 64`), bit position `i & 63` (i.e. `i % 64`).
/// Words are zeroed before OR-ing in set bits, so all-`false` columns
/// produce all-zero words (no stale bits).
///
/// # Examples
///
/// ```
/// use configural::bitset::pack_column;
/// assert_eq!(pack_column(&[]), Vec::<u64>::new());
/// assert_eq!(pack_column(&[true]), vec![1]);
/// assert_eq!(pack_column(&[true; 64]), vec![0xFFFF_FFFF_FFFF_FFFF]);
/// ```
pub fn pack_column(col: &[bool]) -> Vec<u64> {
    let n = col.len();
    if n == 0 {
        return Vec::new();
    }
    let nwords = n.div_ceil(64);
    let mut words = vec![0u64; nwords];
    for (i, &bit) in col.iter().enumerate() {
        if bit {
            words[i >> 6] |= 1u64 << (i & 63);
        }
    }
    words
}

/// Fused bitwise AND + popcount across word slices.
///
/// Sums `popcount(a[w] & b[w])` for each word index `w`. Iterates up to
/// `min(a.len(), b.len())` via `zip` — mismatched lengths silently
/// truncate (standard zip behavior).
///
/// # Examples
///
/// ```
/// use configural::bitset::and_popcount;
/// assert_eq!(and_popcount(&[0b1010_u64], &[0b1100_u64]), 1);
/// assert_eq!(and_popcount(&[0], &[0xFFFF_FFFF_FFFF_FFFF]), 0);
/// ```
pub fn and_popcount(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&aw, &bw)| (aw & bw).count_ones())
        .sum()
}

/// Yield 0-based indices of set bits in a `u64` mask, in ascending order.
///
/// Uses `trailing_zeros()` + `mask &= mask - 1` (Brian Kernighan's method).
/// For `mask == 0`, yields nothing (does not panic).
///
/// # Examples
///
/// ```
/// use configural::bitset::extract_members;
/// assert_eq!(extract_members(0b1010_u64).collect::<Vec<_>>(), vec![1u32, 3]);
/// assert_eq!(extract_members(0u64).collect::<Vec<_>>(), Vec::<u32>::new());
/// ```
pub fn extract_members(mask: u64) -> impl Iterator<Item = u32> {
    let mut m = mask;
    std::iter::from_fn(move || {
        if m == 0 {
            None
        } else {
            let bit = m.trailing_zeros();
            m &= m - 1;
            Some(bit)
        }
    })
}

/// Return the index of the highest set bit in a `u64` mask.
///
/// Returns `None` for `mask == 0`. Uses `63 - leading_zeros` guarded by
/// `Option` — the naive `63u32.wrapping_sub(64)` for zero would underflow
/// to a garbage value.
///
/// # Examples
///
/// ```
/// use configural::bitset::max_element;
/// assert_eq!(max_element(0b1010_u64), Some(3u32));
/// assert_eq!(max_element(0u64), None);
/// ```
pub fn max_element(mask: u64) -> Option<u32> {
    if mask == 0 {
        None
    } else {
        Some(63 - mask.leading_zeros())
    }
}
