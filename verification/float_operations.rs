pub fn arithmetic64(left: u64, right: u64) -> (u64, u64, u64, u64, u64, u64) {
    let left = f64::from_bits(left);
    let right = f64::from_bits(right);
    (
        (left + right).to_bits(),
        (left - right).to_bits(),
        (left * right).to_bits(),
        (left / right).to_bits(),
        (left % right).to_bits(),
        (-left).to_bits(),
    )
}

pub fn compare64(left: u64, right: u64) -> (bool, bool, bool, bool, bool, bool) {
    let left = f64::from_bits(left);
    let right = f64::from_bits(right);
    (left == right, left != right, left < right, left <= right, left > right, left >= right)
}

pub fn self_equal64(bits: u64) -> bool {
    let value = f64::from_bits(bits);
    value == value
}

pub fn casts64(bits: u64) -> (i64, u64, u32) {
    let value = f64::from_bits(bits);
    (value as i64, value as u64, (value as f32).to_bits())
}

pub fn integer_casts(signed: i128, unsigned: u128) -> (u64, u32) {
    ((signed as f64).to_bits(), (unsigned as f32).to_bits())
}

pub fn decimal_literals() -> (u32, u64, u32, u64, u64) {
    (
        0.1f32.to_bits(),
        0.1f64.to_bits(),
        1.401298464324817e-45f32.to_bits(),
        4.9406564584124654e-324f64.to_bits(),
        1.7976931348623157e308f64.to_bits(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ieee_primitives_keep_rounding_zero_and_unordered_semantics() {
        let result = arithmetic64(6.0f64.to_bits(), 2.0f64.to_bits());
        assert_eq!(result, (8.0f64.to_bits(), 4.0f64.to_bits(), 12.0f64.to_bits(), 3.0f64.to_bits(), 0, (-6.0f64).to_bits()));
        assert_eq!(arithmetic64(1.0f64.to_bits(), (2.0f64).powi(-53).to_bits()).0, 1.0f64.to_bits());
        assert_eq!(arithmetic64(1.0f64.to_bits(), 0).3, f64::INFINITY.to_bits());
        assert_eq!(compare64(0, 0x8000_0000_0000_0000), (true, false, false, true, false, true));
        assert_eq!(compare64(0x7ff0_0000_0000_0001, 0), (false, true, false, false, false, false));
        assert!(!self_equal64(0x7ff0_0000_0000_0001));
    }

    #[test]
    fn casts_saturate_and_width_narrowing_rounds_once() {
        assert_eq!(casts64(0x7ff0_0000_0000_0000), (i64::MAX, u64::MAX, 0x7f80_0000));
        assert_eq!(casts64(0xfff0_0000_0000_0000), (i64::MIN, 0, 0xff80_0000));
        let nan = casts64(0x7ff0_0000_0000_0001);
        assert_eq!((nan.0, nan.1), (0, 0));
        assert!(f32::from_bits(nan.2).is_nan());
        assert_eq!(casts64(0x3ff0_0000_1000_0000), (1, 1, 0x3f80_0000));
        assert_eq!(casts64(0x3ff0_0000_1000_0001), (1, 1, 0x3f80_0001));
        assert_eq!(integer_casts(-1, 16_777_217), (0xbff0_0000_0000_0000, 0x4b80_0000));
    }

    #[test]
    fn decimal_literals_preserve_normal_subnormal_and_boundary_bits() {
        assert_eq!(decimal_literals(), (0x3dcc_cccd, 0x3fb9_9999_9999_999a, 1, 1, 0x7fef_ffff_ffff_ffff));
    }
}
