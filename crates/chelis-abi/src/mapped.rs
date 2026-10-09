//! The mapped-range reads every lane shares: `mmap_read` ([05-OP-60]),
//! `mmap_tensor` ([05-OP-79]), and `mmap_text` and `mmap_sha256`
//! ([05-OP-80]).
//!
//! The evaluator and the compiled runtime both select a range, check a
//! `bool` payload, decode text, and compute a digest through these
//! functions, so the lanes cannot disagree about which ranges are valid or
//! how a failure reads. Each `Err` is the complete failure message.

use std::ops::Range;

use sha2::{Digest, Sha256};

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
/// byte length, and the mapping length ([05-OP-79], [05-OP-80]).
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
/// runtime dtype of `T` ([05-OP-79]). The byte length comes from the
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
        let reason = format!("the byte length at {} overflows i64", dtype.name());
        return fail(NumericTrapKind::Overflow, &reason);
    };
    checked_range(OP, offset, Some(count), length.get(), mapped_len)
}

/// A `bool` payload holds only the bytes 0 and 1 ([05-OP-79]); any other
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
/// offset `start` ([05-OP-80]). Invalid UTF-8 fails with the mapping offset
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

/// `mmap_sha256`'s digest of `bytes` ([05-OP-80]): 64 lowercase hexadecimal
/// characters, two per digest byte in digest order, high nibble first.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push(char::from(HEX[usize::from(byte >> 4)]));
        text.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    text
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
                "mmap_tensor offset 0, count {}, mapping length 5: the byte length at f32 \
                 overflows i64\nnumeric trap: overflow in mmap_tensor at i64",
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
    }
}
