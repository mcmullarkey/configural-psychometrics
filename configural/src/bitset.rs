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
/// Dispatches to a SIMD-optimized path (NEON on aarch64, AVX2 on x86_64)
/// when the slice has ≥ 4 words, falling back to scalar otherwise.
///
/// # Examples
///
/// ```
/// use configural::bitset::and_popcount;
/// assert_eq!(and_popcount(&[0b1010_u64], &[0b1100_u64]), 1);
/// assert_eq!(and_popcount(&[0], &[0xFFFF_FFFF_FFFF_FFFF]), 0);
/// ```
pub fn and_popcount(a: &[u64], b: &[u64]) -> u32 {
    let n = a.len().min(b.len());

    // SIMD path for 4+ words; scalar for small inputs.
    if n >= 4 {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on aarch64 — always available.
            return unsafe { and_popcount_neon(a, b) };
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return unsafe { and_popcount_avx2(a, b) };
            }
        }
    }

    and_popcount_scalar(a, b)
}

/// Scalar fallback: fused AND + popcount via iterator zip.
fn and_popcount_scalar(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&aw, &bw)| (aw & bw).count_ones())
        .sum()
}

/// NEON-optimized AND + popcount (aarch64).
///
/// Processes 4 u64 words per iteration using two 128-bit NEON registers
/// for the AND, then scalar `count_ones` for popcount (no SIMD popcount
/// in NEON for u64 lanes).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_popcount_neon(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = 0u32;
    for i in 0..chunks {
        let off = i * 4;
        // Load 2 u64s from each slice, AND them.
        let va0 = vld1q_u64(a.as_ptr().add(off));
        let vb0 = vld1q_u64(b.as_ptr().add(off));
        let vand0 = vandq_u64(va0, vb0);

        // Load next 2 u64s.
        let va1 = vld1q_u64(a.as_ptr().add(off + 2));
        let vb1 = vld1q_u64(b.as_ptr().add(off + 2));
        let vand1 = vandq_u64(va1, vb1);

        // Extract and popcount each lane.
        sum += vgetq_lane_u64(vand0, 0).count_ones();
        sum += vgetq_lane_u64(vand0, 1).count_ones();
        sum += vgetq_lane_u64(vand1, 0).count_ones();
        sum += vgetq_lane_u64(vand1, 1).count_ones();
    }

    // Scalar tail.
    for i in 0..remainder {
        let off = chunks * 4 + i;
        sum += (a[off] & b[off]).count_ones();
    }

    sum
}

/// AVX2-optimized AND + popcount (x86_64).
///
/// Processes 4 u64 words per iteration using a single 256-bit AVX2
/// register for the AND, then scalar `count_ones` for popcount.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_popcount_avx2(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = 0u32;
    for i in 0..chunks {
        let off = i * 4;
        let va = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(off) as *const __m256i);
        let vand = _mm256_and_si256(va, vb);

        sum += (_mm256_extract_epi64(vand, 0) as u64).count_ones();
        sum += (_mm256_extract_epi64(vand, 1) as u64).count_ones();
        sum += (_mm256_extract_epi64(vand, 2) as u64).count_ones();
        sum += (_mm256_extract_epi64(vand, 3) as u64).count_ones();
    }

    // Scalar tail.
    for i in 0..remainder {
        let off = chunks * 4 + i;
        sum += (a[off] & b[off]).count_ones();
    }

    sum
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn and_popcount_single_word() {
        assert_eq!(and_popcount(&[0b1010_u64], &[0b1100_u64]), 1);
        assert_eq!(and_popcount(&[0], &[0xFFFF_FFFF_FFFF_FFFF]), 0);
        assert_eq!(
            and_popcount(&[0xFFFF_FFFF_FFFF_FFFF], &[0xFFFF_FFFF_FFFF_FFFF]),
            64
        );
    }

    #[test]
    fn and_popcount_mismatched_lengths() {
        // zip truncates to shorter slice.
        assert_eq!(and_popcount(&[1, 2, 3], &[1]), 1);
        assert_eq!(and_popcount(&[1], &[1, 2, 3]), 1);
    }

    #[test]
    fn and_popcount_empty() {
        assert_eq!(and_popcount(&[], &[]), 0);
    }

    #[test]
    fn and_popcount_simd_matches_scalar_4_words() {
        // 4-word slice — exercises the SIMD path (n >= 4).
        let a = [
            0xFF00FF00FF00FF00,
            0x0F0F0F0F0F0F0F0F,
            0xAAAAAAAAAAAAAAAA,
            0x123456789ABCDEF0,
        ];
        let b = [
            0x00FF00FF00FF00FF,
            0xF0F0F0F0F0F0F0F0,
            0x5555555555555555,
            0x0FEDCBA987654321,
        ];
        let simd = and_popcount(&a, &b);
        let scalar = and_popcount_scalar(&a, &b);
        assert_eq!(simd, scalar, "SIMD and scalar must agree for 4-word slices");
    }

    #[test]
    fn and_popcount_simd_matches_scalar_8_words() {
        // 8-word slice — two SIMD iterations + no remainder.
        let a = [
            0xFF00FF00FF00FF00,
            0x0F0F0F0F0F0F0F0F,
            0xAAAAAAAAAAAAAAAA,
            0x123456789ABCDEF0,
            0xFFFFFFFFFFFFFFFF,
            0x0000000000000000,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
        ];
        let b = [
            0x00FF00FF00FF00FF,
            0xF0F0F0F0F0F0F0F0,
            0x5555555555555555,
            0x0FEDCBA987654321,
            0x0000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
        ];
        let simd = and_popcount(&a, &b);
        let scalar = and_popcount_scalar(&a, &b);
        assert_eq!(simd, scalar, "SIMD and scalar must agree for 8-word slices");
    }

    #[test]
    fn and_popcount_simd_matches_scalar_6_words() {
        // 6-word slice — one SIMD iteration + 2-word remainder.
        let a = [
            0xFF00FF00FF00FF00,
            0x0F0F0F0F0F0F0F0F,
            0xAAAAAAAAAAAAAAAA,
            0x123456789ABCDEF0,
            0xFFFFFFFFFFFFFFFF,
            0x0000000000000000,
        ];
        let b = [
            0x00FF00FF00FF00FF,
            0xF0F0F0F0F0F0F0F0,
            0x5555555555555555,
            0x0FEDCBA987654321,
            0x0000000000000000,
            0xFFFFFFFFFFFFFFFF,
        ];
        let simd = and_popcount(&a, &b);
        let scalar = and_popcount_scalar(&a, &b);
        assert_eq!(simd, scalar, "SIMD and scalar must agree for 6-word slices");
    }

    #[test]
    fn and_popcount_simd_all_ones() {
        let n = 8;
        let a = vec![u64::MAX; n];
        let b = vec![u64::MAX; n];
        assert_eq!(and_popcount(&a, &b), (n * 64) as u32);
    }

    #[test]
    fn and_popcount_simd_all_zeros() {
        let n = 8;
        let a = vec![0u64; n];
        let b = vec![u64::MAX; n];
        assert_eq!(and_popcount(&a, &b), 0);
    }
}
