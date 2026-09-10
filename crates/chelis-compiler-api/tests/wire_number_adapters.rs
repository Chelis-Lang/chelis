//! Fixed-domain number adapters in spec/10 §§3.3/3.5 and [04-FIT-18].

use chelis_compiler_api::schema::numbers::{
    NonnegativeCount, SourceFloat, SourceInteger, UnitInterval,
};
use chelis_types::{ScalarValue, scalar_from_f64, scalar_from_i64, types::Prim};

#[test]
fn float_adapters_preserve_their_exact_binary64_domain() {
    for value in [
        -0.0,
        0.0,
        f64::MIN_POSITIVE,
        0.5,
        1.0,
        f64::from_bits(0x3fb9_52b7_4674_9bc0),
    ] {
        let admitted = UnitInterval::new(value).unwrap();
        let json = serde_json::to_string(&admitted).unwrap();
        let decoded: UnitInterval = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.get().to_bits(), value.to_bits());
        assert_eq!(admitted.scalar().prim(), Prim::F64);
    }
    for value in [
        f64::MIN,
        -0.0,
        f64::from_bits(1),
        f64::MAX,
        f64::from_bits(0xb1c2_48bb_49e9_53c6),
    ] {
        let admitted = SourceFloat::new(value).unwrap();
        let decoded: SourceFloat =
            serde_json::from_str(&serde_json::to_string(&admitted).unwrap()).unwrap();
        assert_eq!(decoded.get().to_bits(), value.to_bits());
    }

    // Every finite exponent, both signs, and mantissa boundaries/interiors.
    // JSON must preserve the very bits that the binary codec preserves.
    for exponent in 0_u64..0x7ff {
        for mantissa in [
            0,
            1,
            0x0005_5555_5555_5555,
            0x000a_aaaa_aaaa_aaaa,
            0x000f_ffff_ffff_ffff,
        ] {
            for sign in [0, 1_u64 << 63] {
                let value = f64::from_bits(sign | (exponent << 52) | mantissa);
                let source = SourceFloat::new(value).unwrap();
                let json = serde_json::to_string(&source).unwrap();
                let decoded: SourceFloat = serde_json::from_str(&json).unwrap();
                assert_eq!(decoded.get().to_bits(), value.to_bits(), "source {json}");
                let binary: SourceFloat =
                    bincode::deserialize(&bincode::serialize(&source).unwrap()).unwrap();
                assert_eq!(binary.get().to_bits(), value.to_bits());
                if (0.0..=1.0).contains(&value) {
                    let score = UnitInterval::new(value).unwrap();
                    let json = serde_json::to_string(&score).unwrap();
                    let decoded: UnitInterval = serde_json::from_str(&json).unwrap();
                    assert_eq!(decoded.get().to_bits(), value.to_bits(), "score {json}");
                } else {
                    assert!(UnitInterval::new(value).is_err());
                    assert!(serde_json::from_str::<UnitInterval>(&json).is_err());
                }
            }
        }
    }
}

#[test]
fn float_adapters_reject_invalid_domains_and_wrong_carrier_dtypes() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(UnitInterval::new(value).is_err());
        assert!(SourceFloat::new(value).is_err());
    }
    for value in [-f64::from_bits(1), 1.0 + f64::EPSILON] {
        assert!(UnitInterval::new(value).is_err());
        assert!(SourceFloat::new(value).is_ok());
    }
    let narrower = scalar_from_f64("fixture", Prim::F32, 0.5).unwrap();
    assert!(UnitInterval::try_from(narrower).is_err());
    assert!(SourceFloat::try_from(narrower).is_err());
    for json in [
        "null",
        "true",
        "\"0.5\"",
        "{\"dtype\":\"f64\",\"bits\":\"0000000000000000\"}",
        "1e999",
    ] {
        assert!(serde_json::from_str::<SourceFloat>(json).is_err());
        assert!(serde_json::from_str::<UnitInterval>(json).is_err());
    }
}

#[test]
fn integer_adapters_never_pass_through_binary64() {
    for value in [i64::MIN, -9007199254740993, 0, 9007199254740993, i64::MAX] {
        let admitted = SourceInteger::new(value);
        let encoded = serde_json::to_string(&admitted).unwrap();
        assert_eq!(encoded, value.to_string());
        let decoded: SourceInteger = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.get(), value);
        assert_eq!(admitted.scalar().prim(), Prim::Int64);
        if value >= 0 {
            let count = NonnegativeCount::new(value).unwrap();
            let decoded: NonnegativeCount =
                serde_json::from_str(&serde_json::to_string(&count).unwrap()).unwrap();
            assert_eq!(decoded.get(), value);
        } else {
            assert!(NonnegativeCount::new(value).is_err());
        }
    }
}

#[test]
fn integer_adapters_reject_alternate_numbers_overflow_and_wrong_dtype() {
    for json in [
        "1.0",
        "1e0",
        "\"1\"",
        "true",
        "null",
        "9223372036854775808",
        "-9223372036854775809",
    ] {
        assert!(serde_json::from_str::<SourceInteger>(json).is_err());
        assert!(serde_json::from_str::<NonnegativeCount>(json).is_err());
    }
    assert!(serde_json::from_str::<NonnegativeCount>("-1").is_err());
    for wrong in [
        scalar_from_i64("fixture", Prim::Int32, 1).unwrap(),
        scalar_from_f64("fixture", Prim::F64, 1.0).unwrap(),
    ] {
        assert!(SourceInteger::try_from(wrong).is_err());
        assert!(NonnegativeCount::try_from(wrong).is_err());
    }
    let boolean: ScalarValue = serde_json::from_str("{\"dtype\":\"bool\",\"value\":true}").unwrap();
    assert!(SourceInteger::try_from(boolean).is_err());
    assert!(NonnegativeCount::try_from(boolean).is_err());
    if usize::BITS == 64 {
        assert!(NonnegativeCount::try_from(usize::MAX).is_err());
    }
}

#[test]
fn positional_cache_uses_the_same_domain_validation() {
    let bits = SourceFloat::new(-0.0).unwrap();
    let bytes = bincode::serialize(&bits).unwrap();
    let decoded: SourceFloat = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded.get().to_bits(), (-0.0_f64).to_bits());
    assert!(bincode::deserialize::<SourceFloat>(&bincode::serialize(&f64::NAN).unwrap()).is_err());
    assert!(bincode::deserialize::<UnitInterval>(&bincode::serialize(&2.0_f64).unwrap()).is_err());
    assert!(
        bincode::deserialize::<NonnegativeCount>(&bincode::serialize(&-1_i64).unwrap()).is_err()
    );
}
