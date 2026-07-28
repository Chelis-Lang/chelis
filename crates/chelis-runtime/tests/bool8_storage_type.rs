//! `Bool8` is the storage element type for boolean tensors.
//!
//! These tests pin the properties that make it usable where `bool` is not:
//! totality of the read, distinctness from `i8`, and a transparent one-byte
//! layout. They do not exercise tensor access, because `Bool8` has no
//! `TensorElement` impl yet — see the type's documentation for why that has to
//! land with the storage-width flip rather than before it.

use chelis_runtime::Bool8;

#[test]
fn layout_is_one_transparent_byte() {
    assert_eq!(size_of::<Bool8>(), 1);
    assert_eq!(align_of::<Bool8>(), 1);
    assert_eq!(size_of::<Bool8>(), size_of::<u8>());
}

#[test]
fn canonical_values_round_trip() {
    assert_eq!(Bool8::new(true).to_byte(), 1);
    assert_eq!(Bool8::new(false).to_byte(), 0);
    assert!(Bool8::TRUE.get());
    assert!(!Bool8::FALSE.get());

    for value in [true, false] {
        assert_eq!(Bool8::new(value).get(), value);
        assert_eq!(bool::from(Bool8::from(value)), value);
    }
}

/// The property that lets this type exist where `bool` cannot: every one of
/// the 256 byte values is an inhabitant and reads without undefined behaviour.
///
/// A `bool` would make 254 of these UB on read, which is precisely why
/// `TensorElement` cannot be implemented for it.
#[test]
fn every_byte_value_reads_without_undefined_behaviour() {
    let mut truthy = 0usize;
    for byte in 0..=u8::MAX {
        let value = Bool8::from_byte(byte);
        if value.get() {
            truthy += 1;
        }
        // Round-trips as raw storage: a non-canonical byte is preserved, not
        // normalised, so an inspector sees what is actually in the buffer.
        assert_eq!(value.to_byte(), byte);
    }
    assert_eq!(truthy, 255, "only 0x00 is false");
}

/// Non-canonical bytes are wrong answers, not undefined behaviour.
#[test]
fn non_canonical_bytes_are_truthy_not_undefined() {
    assert!(Bool8::from_byte(2).get());
    assert!(Bool8::from_byte(0xFF).get());
    assert_ne!(Bool8::from_byte(2), Bool8::TRUE, "2 is not canonical true");
}

/// Distinctness from `i8` is the point of the newtype. Both are one byte, and
/// the type system must still refuse to interchange them — mirroring
/// `Repr::Bool8` vs `Repr::TwosComplement8` at the vocabulary level.
///
/// This is a compile-time property; the test documents it and checks the
/// widths really do collide, which is what makes the distinction load-bearing.
#[test]
fn shares_a_width_with_i8_and_is_still_a_distinct_type() {
    assert_eq!(size_of::<Bool8>(), size_of::<i8>());

    // Deliberately not compiled:
    //   let _: Bool8 = 1i8;              // mismatched types
    //   let _: *mut i8 = bool8_ptr;      // mismatched types
    // Equal widths would make both of those silently fine on a raw `u8`.
}
