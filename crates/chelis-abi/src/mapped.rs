//! The mapped-range reads every lane shares: `mmap_read` ([05-OP-60]),
//! `mmap_tensor` ([05-OP-80]), and `mmap_text` and `mmap_sha256`
//! ([05-OP-81]).
//!
//! The evaluator and the compiled runtime both select a range, check a
//! `bool` payload, decode text, and compute a digest through these
//! functions, so the lanes cannot disagree about which ranges are valid or
//! how a failure reads. Each `Err` is the complete failure message.

use std::ops::Range;

use chelis_vocab::RuntimeDType;

use crate::failure::{NumericTrapKind, NumericTrapLine};
use crate::metadata::ElementCount;

fn trap(kind: NumericTrapKind, op: &str, prim: &str, context: &str) -> String {
    let line = NumericTrapLine {
        kind,
        op,
        dtype: prim,
    };
    format!("{context}\n{line}")
}

/// The byte range `offset .. offset + length` of a mapping `mapped_len`
/// bytes long. A negative offset or length, or a range that ends past the
/// mapping, traps `Domain` in `op`; an end outside i64 traps `Overflow`.
/// An offset at the end of the mapping is valid only for zero length.
pub fn mapped_range(
    op: &str,
    offset: i64,
    length: i64,
    mapped_len: usize,
) -> Result<Range<usize>, String> {
    checked_range(op, offset, None, length, mapped_len)
}

/// Every failure names the offset, the element count when there is one, the
/// byte length, and the mapping length ([05-OP-80], [05-OP-81]).
fn checked_range(
    op: &str,
    offset: i64,
    count: Option<i64>,
    length: i64,
    mapped_len: usize,
) -> Result<Range<usize>, String> {
    let count_part = count
        .map(|count| format!(", count {count}"))
        .unwrap_or_default();
    let fail = |kind, reason: &str| {
        let context = format!(
            "{op} offset {offset}{count_part}, byte length {length}, mapping length \
             {mapped_len}: {reason}"
        );
        Err(trap(kind, op, "i64", &context))
    };
    if offset < 0 {
        return fail(NumericTrapKind::Domain, "the offset is negative");
    }
    if length < 0 {
        return fail(NumericTrapKind::Domain, "the byte length is negative");
    }
    let Some(end) = offset.checked_add(length) else {
        return fail(NumericTrapKind::Overflow, "the range end overflows i64");
    };
    match usize::try_from(end) {
        Ok(end) if end <= mapped_len => Ok(offset as usize..end),
        _ => fail(NumericTrapKind::Domain, "the range ends past the mapping"),
    }
}

/// The payload range of `mmap_tensor(mapped, offset, count, T)` at the
/// runtime dtype of `T` ([05-OP-80]). The byte length comes from the
/// checked element-count metadata, the one owner of element widths. A
/// negative count traps `Domain`; a byte length outside i64 traps
/// `Overflow`; the byte range then follows [`mapped_range`].
pub fn mapped_tensor_range(
    offset: i64,
    count: i64,
    dtype: RuntimeDType,
    mapped_len: usize,
) -> Result<Range<usize>, String> {
    const OP: &str = "mmap_tensor";
    let fail = |kind, reason: &str| {
        let context =
            format!("{OP} offset {offset}, count {count}, mapping length {mapped_len}: {reason}");
        Err(trap(kind, OP, "i64", &context))
    };
    if count < 0 {
        return fail(NumericTrapKind::Domain, "the count is negative");
    }
    let Ok(length) =
        ElementCount::from_extents(&[count]).and_then(|elements| elements.bytes(dtype))
    else {
        return fail(NumericTrapKind::Overflow, "the byte length overflows i64");
    };
    checked_range(OP, offset, Some(count), length.get(), mapped_len)
}

/// A `bool` payload holds only the bytes 0 and 1 ([05-OP-80]); any other
/// byte traps `Domain` at `bool`, naming the element and the byte.
pub fn check_bool_payload(bytes: &[u8]) -> Result<(), String> {
    match bytes.iter().position(|byte| *byte > 1) {
        None => Ok(()),
        Some(index) => {
            let context = format!(
                "mmap_tensor bool element {index} has byte {}, expected 0 or 1",
                bytes[index]
            );
            Err(trap(
                NumericTrapKind::Domain,
                "mmap_tensor",
                "bool",
                &context,
            ))
        }
    }
}

/// `mmap_text`'s exact UTF-8 decoding of `bytes`, which begin at mapping
/// offset `start` ([05-OP-81]). Invalid UTF-8 fails with the mapping offset
/// of the first byte that does not begin or continue a valid sequence.
pub fn mapped_text(start: usize, bytes: &[u8]) -> Result<String, String> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(text.to_string()),
        Err(error) => Err(format!(
            "mmap_text: invalid UTF-8 at byte {}",
            start + error.valid_up_to()
        )),
    }
}

/// `mmap_sha256`'s digest of `bytes` ([05-OP-81]): 64 lowercase hexadecimal
/// characters, two per digest byte in digest order, high nibble first.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in sha256(bytes) {
        text.push(char::from(HEX[usize::from(byte >> 4)]));
        text.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    text
}

/// SHA-256 (FIPS 180-4 section 6.2). It lives here, written out, because
/// this crate depends on nothing but `std` and the vocabulary: the evaluator
/// and the compiled runtime share this one definition through it.
fn sha256(message: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    // Padding: a 1 bit, zeros to 56 mod 64 bytes, then the bit length as a
    // big-endian u64.
    let bit_length = (message.len() as u64).wrapping_mul(8);
    let mut tail = message[message.len() - message.len() % 64..].to_vec();
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&bit_length.to_be_bytes());
    let whole = &message[..message.len() - message.len() % 64];
    for block in whole
        .as_chunks::<64>()
        .0
        .iter()
        .chain(tail.as_chunks::<64>().0)
    {
        let mut schedule = [0u32; 64];
        for (word, bytes) in schedule.iter_mut().zip(block.as_chunks::<4>().0) {
            *word = u32::from_be_bytes(*bytes);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for (constant, word) in K.iter().zip(schedule) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(*constant)
                .wrapping_add(word);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (word, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *word = word.wrapping_add(value);
        }
    }
    let mut digest = [0u8; 32];
    for (bytes, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use chelis_vocab::RuntimeDType;

    use super::{check_bool_payload, mapped_range, mapped_tensor_range, mapped_text, sha256_hex};

    #[test]
    fn ranges_inside_the_mapping_are_selected_exactly() {
        assert_eq!(mapped_range("mmap_read", 2, 3, 5), Ok(2..5));
        assert_eq!(mapped_range("mmap_read", 5, 0, 5), Ok(5..5));
        assert_eq!(mapped_tensor_range(1, 2, RuntimeDType::F32, 9), Ok(1..9));
        assert_eq!(mapped_tensor_range(0, 3, RuntimeDType::Bf16, 6), Ok(0..6));
    }

    #[test]
    fn ranges_outside_the_mapping_trap_with_every_value() {
        assert_eq!(
            mapped_range("mmap_read", 3, 3, 5),
            Err(
                "mmap_read offset 3, byte length 3, mapping length 5: the range ends past the \
                 mapping\nnumeric trap: domain in mmap_read at i64"
                    .to_string()
            )
        );
        let at_end = mapped_range("mmap_read", 6, 0, 5).unwrap_err();
        assert!(
            at_end.ends_with("numeric trap: domain in mmap_read at i64"),
            "{at_end}"
        );
        assert_eq!(
            mapped_range("mmap_text", -1, 0, 5),
            Err(
                "mmap_text offset -1, byte length 0, mapping length 5: the offset is \
                 negative\nnumeric trap: domain in mmap_text at i64"
                    .to_string()
            )
        );
        let overflow = mapped_range("mmap_sha256", i64::MAX, 1, 5).unwrap_err();
        assert!(
            overflow.ends_with("numeric trap: overflow in mmap_sha256 at i64"),
            "{overflow}"
        );
        assert_eq!(
            mapped_tensor_range(0, -1, RuntimeDType::F32, 5),
            Err(
                "mmap_tensor offset 0, count -1, mapping length 5: the count is \
                 negative\nnumeric trap: domain in mmap_tensor at i64"
                    .to_string()
            )
        );
        assert_eq!(
            mapped_tensor_range(0, i64::MAX / 2, RuntimeDType::F32, 5),
            Err(format!(
                "mmap_tensor offset 0, count {}, mapping length 5: the byte length overflows \
                 i64\nnumeric trap: overflow in mmap_tensor at i64",
                i64::MAX / 2
            ))
        );
        assert_eq!(
            mapped_tensor_range(4, 2, RuntimeDType::I32, 8),
            Err(
                "mmap_tensor offset 4, count 2, byte length 8, mapping length 8: the range \
                 ends past the mapping\nnumeric trap: domain in mmap_tensor at i64"
                    .to_string()
            )
        );
    }

    #[test]
    fn bool_payloads_admit_only_zero_and_one() {
        assert_eq!(check_bool_payload(&[0, 1, 1, 0]), Ok(()));
        assert_eq!(
            check_bool_payload(&[0, 1, 2]),
            Err("mmap_tensor bool element 2 has byte 2, expected 0 or 1\n\
                 numeric trap: domain in mmap_tensor at bool"
                .to_string())
        );
    }

    #[test]
    fn text_decodes_exactly_and_names_the_first_invalid_byte() {
        assert_eq!(
            mapped_text(0, "a\u{e9}\r\n".as_bytes()),
            Ok("a\u{e9}\r\n".to_string())
        );
        assert_eq!(
            mapped_text(10, &[b'a', 0xc3, b'b']),
            Err("mmap_text: invalid UTF-8 at byte 11".to_string())
        );
    }

    #[test]
    fn digests_are_lowercase_hex_in_digest_order() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // FIPS 180-4 example vectors: a two-block message, and one million
        // `a`s, which crosses every padding boundary on the way.
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_hex(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
}
