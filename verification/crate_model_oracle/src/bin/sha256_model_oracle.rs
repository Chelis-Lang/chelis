//! Native oracle for Sha256ModelChecks.lean: `sha2` 0.10.9 SHA-256 of
//! concatenated parts, as computed by `chelis_types::native::sha256_concat`.

use sha2::{Digest, Sha256};

fn sha256_concat(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn main() {
    const PREFIX: &[u8] = b"chelis-library-proof-v1";
    // Canonical text of the library in context::tests::library_proof_id_digest_is_stable.
    let canonical: &[u8] = b"(def {} answer (lit {type: (t-prim {} i64)} 42))\n\n(def {} twice (fn {} (params {} x) (app {} (var {} add) (var {} x) (var {} x))))\n";
    let length = (canonical.len() as u64).to_le_bytes();
    let base = sha256_concat(&[PREFIX, &[0], &length, canonical]);
    let chained = sha256_concat(&[PREFIX, &[1], &base, &length, canonical]);
    // Pinned in context::tests::library_proof_id_digest_is_stable.
    assert_eq!(base, [52, 101, 74, 50, 108, 52, 77, 3, 169, 119, 248, 140, 139, 97, 67, 3, 185, 163,
        206, 120, 15, 71, 41, 67, 244, 140, 225, 171, 217, 9, 191, 69]);
    assert_eq!(chained, [252, 90, 11, 182, 68, 160, 198, 126, 141, 162, 151, 224, 198, 176, 40, 53,
        137, 29, 61, 167, 221, 250, 173, 248, 27, 148, 72, 50, 51, 166, 16, 144]);
    let long: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
    let cases: Vec<Vec<&[u8]>> = vec![
        vec![],
        vec![b""],
        vec![b"abc"],
        vec![b"a", b"", b"bc"],
        vec![b"abcdbcdecdefdefgefghfghighijhijk", b"ijkljklmklmnlmnomnopnopq"],
        vec![&long[..1], &long[1..55], &long[55..64], &long[64..]],
        vec![&long[..55]],
        vec![&long[..56]],
        vec![&long[..64]],
    ];
    for (i, parts) in cases.iter().enumerate() {
        println!("sha256-{i}={:?}", sha256_concat(parts));
    }
    println!("library-proof-base={base:?}");
    println!("library-proof-chained={chained:?}");
}
