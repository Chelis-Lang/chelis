//! [05-OP-80]: `chelis_abi::mapped::sha256_hex`, the written-out SHA-256 both
//! lanes share, agrees with the `sha2` crate on every message length through
//! several blocks, so each padding boundary (55, 56, 63, 64 bytes, and their
//! multiples) is crossed.

use sha2::{Digest, Sha256};

fn reference(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn every_length_through_four_blocks_matches_sha2() {
    for length in 0..=260usize {
        let message: Vec<u8> = (0..length)
            .map(|index| (index.wrapping_mul(131).wrapping_add(7) % 256) as u8)
            .collect();
        assert_eq!(
            chelis_abi::mapped::sha256_hex(&message),
            reference(&message),
            "length {length}"
        );
    }
}
