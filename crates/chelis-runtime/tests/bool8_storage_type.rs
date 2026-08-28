//! `Bool8` is the storage element type for boolean tensors.
//!
//! These tests pin the properties that make it usable where `bool` is not:
//! distinctness from `i8`, a transparent one-byte layout, and rejection of
//! non-canonical storage before a value enters typed tensor access.

use chelis_runtime::{Bool8, RuntimeDType, TensorElement};

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

    assert_eq!(Bool8::from_u8(0), Some(Bool8::FALSE));
    assert_eq!(Bool8::from_u8(1), Some(Bool8::TRUE));
}

#[test]
fn non_canonical_bytes_are_rejected() {
    for byte in 2..=u8::MAX {
        assert_eq!(
            Bool8::from_u8(byte),
            None,
            "byte {byte} must not enter canonical boolean storage"
        );
    }
}

#[test]
fn tensor_element_registration_is_boolean_not_int8() {
    assert_eq!(<Bool8 as TensorElement>::DTYPE, RuntimeDType::Bool);
    assert_ne!(
        <Bool8 as TensorElement>::DTYPE,
        <i8 as TensorElement>::DTYPE
    );
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
