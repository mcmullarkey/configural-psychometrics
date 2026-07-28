//! Integration tests for the bit-parallel bitset core.
//!
//! Exercises all five pure primitives against the predicates from AC-2,
//! including edge cases (empty input, single bit, full word, multi-word)
//! and negative cases (all-zero, underflow guards, high-bit truncation).

use configural::bitset::{and_popcount, extract_members, max_element, pack_column, popcount};

// ---------------------------------------------------------------------------
// popcount
// ---------------------------------------------------------------------------

#[test]
fn popcount_full_word_is_64() {
    assert_eq!(popcount(0xFFFF_FFFF_FFFF_FFFF), 64);
}

#[test]
fn popcount_zero_is_0() {
    assert_eq!(popcount(0), 0);
}

#[test]
fn popcount_single_bit_is_1() {
    for i in 0..64u32 {
        assert_eq!(popcount(1u64 << i), 1, "bit {} should give popcount 1", i);
    }
}

#[test]
fn popcount_mixed_pattern() {
    // 0b1010 = bits 1 and 3 set
    assert_eq!(popcount(0b1010_u64), 2);
}

#[test]
fn popcount_uses_full_64_bits() {
    // Negative: (x as u32).count_ones() would truncate high bits.
    // Bit 63 alone must count as 1, not 0.
    assert_eq!(popcount(1u64 << 63), 1);
    // High half all-set, low half clear: 32 bits.
    assert_eq!(popcount(0xFFFF_FFFF_0000_0000), 32);
}

// ---------------------------------------------------------------------------
// pack_column
// ---------------------------------------------------------------------------

#[test]
fn pack_column_empty_is_empty_vec() {
    assert_eq!(pack_column(&[]), Vec::<u64>::new());
}

#[test]
fn pack_column_single_true_is_one() {
    assert_eq!(pack_column(&[true]), vec![1u64]);
}

#[test]
fn pack_column_64_trues_is_full_word() {
    assert_eq!(pack_column(&[true; 64]), vec![0xFFFF_FFFF_FFFF_FFFF]);
}

#[test]
fn pack_column_65_trues_is_full_word_plus_one() {
    assert_eq!(pack_column(&[true; 65]), vec![0xFFFF_FFFF_FFFF_FFFF, 1u64]);
}

#[test]
fn pack_column_all_false_is_zero_words() {
    // Negative: all-zero column must pack to zeroed words, not stale bits.
    assert_eq!(pack_column(&[false; 64]), vec![0u64]);
    assert_eq!(pack_column(&[false; 65]), vec![0u64, 0u64]);
}

#[test]
fn pack_column_mixed_pattern_bit_positions() {
    // Bits 0, 2, 63 set → word 0 = (1<<0)|(1<<2)|(1<<63)
    let mut col = vec![false; 64];
    col[0] = true;
    col[2] = true;
    col[63] = true;
    assert_eq!(col.len(), 64);
    let packed = pack_column(&col);
    assert_eq!(packed, vec![(1u64 << 0) | (1u64 << 2) | (1u64 << 63)]);
}

#[test]
fn pack_column_bit_at_index_64_lands_in_word_1_bit_0() {
    // 65-element column: only element 64 is true.
    let mut col = vec![false; 65];
    col[64] = true;
    assert_eq!(pack_column(&col), vec![0u64, 1u64]);
}

// ---------------------------------------------------------------------------
// and_popcount
// ---------------------------------------------------------------------------

#[test]
fn and_popcount_overlapping_single_bit() {
    // 0b1010 & 0b1100 = 0b1000 → popcount 1
    assert_eq!(and_popcount(&[0b1010_u64], &[0b1100_u64]), 1);
}

#[test]
fn and_popcount_zero_masked_by_full() {
    assert_eq!(and_popcount(&[0u64], &[0xFFFF_FFFF_FFFF_FFFF]), 0);
}

#[test]
fn and_popcount_full_and_full_is_64() {
    assert_eq!(
        and_popcount(&[0xFFFF_FFFF_FFFF_FFFF], &[0xFFFF_FFFF_FFFF_FFFF]),
        64
    );
}

#[test]
fn and_popcount_multi_word() {
    let a = [0b1010_u64, 0b1100_u64];
    let b = [0b1100_u64, 0b1010_u64];
    // word 0: 0b1010 & 0b1100 = 0b1000 → 1
    // word 1: 0b1100 & 0b1010 = 0b1000 → 1
    assert_eq!(and_popcount(&a, &b), 2);
}

#[test]
fn and_popcount_empty_slices_is_zero() {
    assert_eq!(and_popcount(&[], &[]), 0);
}

// ---------------------------------------------------------------------------
// extract_members
// ---------------------------------------------------------------------------

#[test]
fn extract_members_1010_yields_1_then_3() {
    assert_eq!(
        extract_members(0b1010_u64).collect::<Vec<_>>(),
        vec![1u32, 3]
    );
}

#[test]
fn extract_members_zero_is_empty() {
    // Negative: mask=0 must yield empty iterator, not panic.
    assert_eq!(extract_members(0u64).collect::<Vec<_>>(), Vec::<u32>::new());
}

#[test]
fn extract_members_single_bit_at_63() {
    assert_eq!(extract_members(1u64 << 63).collect::<Vec<_>>(), vec![63u32]);
}

#[test]
fn extract_members_full_word_yields_0_to_63() {
    let members: Vec<u32> = extract_members(0xFFFF_FFFF_FFFF_FFFF).collect();
    assert_eq!(members, (0..64u32).collect::<Vec<_>>());
}

#[test]
fn extract_members_ascending_order() {
    // Bits 5, 0, 10 set — must come out ascending: 0, 5, 10
    let mask = (1u64 << 0) | (1u64 << 5) | (1u64 << 10);
    assert_eq!(extract_members(mask).collect::<Vec<_>>(), vec![0u32, 5, 10]);
}

// ---------------------------------------------------------------------------
// max_element
// ---------------------------------------------------------------------------

#[test]
fn max_element_1010_is_some_3() {
    assert_eq!(max_element(0b1010_u64), Some(3u32));
}

#[test]
fn max_element_zero_is_none() {
    // Negative: max_element(0) must return None, not underflow.
    assert_eq!(max_element(0u64), None);
}

#[test]
fn max_element_single_bit_at_63() {
    assert_eq!(max_element(1u64 << 63), Some(63u32));
}

#[test]
fn max_element_full_word_is_63() {
    assert_eq!(max_element(0xFFFF_FFFF_FFFF_FFFF), Some(63u32));
}

#[test]
fn max_element_bit_0_only() {
    assert_eq!(max_element(1u64), Some(0u32));
}
