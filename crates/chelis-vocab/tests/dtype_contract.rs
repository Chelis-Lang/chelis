//! The stored/arithmetic identity table in [04-NUM-8], including its negative cases.

use chelis_vocab::{ArithmeticRepr, DTypeContract, Repr, RuntimeDType};

#[test]
fn every_dtype_has_its_exact_stored_and_arithmetic_representation() {
    use ArithmeticRepr as A;
    use Repr as R;
    use RuntimeDType as D;
    let rows = [
        (D::F32, R::Ieee754Binary32, Some(A::Ieee754Binary32), 4),
        (D::F64, R::Ieee754Binary64, Some(A::Ieee754Binary64), 8),
        (
            D::I32,
            R::TwosComplement32,
            Some(A::ExactTwosComplement32),
            4,
        ),
        (D::Bool, R::Bool8, None, 1),
        (
            D::I64,
            R::TwosComplement64,
            Some(A::ExactTwosComplement64),
            8,
        ),
        (D::Bf16, R::Bfloat16, Some(A::Ieee754Binary32), 2),
        (D::F16, R::Ieee754Binary16, Some(A::Ieee754Binary32), 2),
        (D::I8, R::TwosComplement8, Some(A::ExactTwosComplement8), 1),
        (
            D::I16,
            R::TwosComplement16,
            Some(A::ExactTwosComplement16),
            2,
        ),
    ];
    assert_eq!(rows.map(|row| row.0), RuntimeDType::ALL);
    for (dtype, repr, arithmetic, width) in rows {
        let contract: DTypeContract = dtype.contract();
        assert_eq!(contract.dtype(), dtype);
        assert_eq!(contract.repr(), repr);
        assert_eq!(contract.arithmetic(), arithmetic);
        assert_eq!(contract.byte_width(), width);
        assert_eq!(dtype.repr(), contract.repr());
        assert_eq!(dtype.byte_width(), contract.byte_width());
        assert_eq!(contract.byte_width(), contract.repr().byte_width());
        for other in RuntimeDType::ALL {
            if other != dtype {
                assert_ne!(contract, other.contract());
                assert_ne!(contract.repr(), other.contract().repr());
            }
        }
    }
}

#[test]
fn equal_width_is_not_representation_identity() {
    for (lhs, rhs) in [
        (RuntimeDType::F32, RuntimeDType::I32),
        (RuntimeDType::Bool, RuntimeDType::I8),
        (RuntimeDType::F16, RuntimeDType::Bf16),
    ] {
        assert_eq!(lhs.contract().byte_width(), rhs.contract().byte_width());
        assert_ne!(lhs.contract().repr(), rhs.contract().repr());
    }
}

#[test]
fn arithmetic_does_not_follow_storage_width_or_grant_bool_arithmetic() {
    for dtype in [RuntimeDType::F16, RuntimeDType::Bf16] {
        assert_eq!(
            dtype.contract().arithmetic(),
            Some(ArithmeticRepr::Ieee754Binary32)
        );
        assert_ne!(
            dtype.contract().arithmetic(),
            Some(ArithmeticRepr::Ieee754Binary64)
        );
        assert_eq!(dtype.byte_width(), 2);
    }
    assert_eq!(RuntimeDType::Bool.contract().arithmetic(), None);
    for dtype in RuntimeDType::ALL {
        if dtype != RuntimeDType::Bool {
            assert!(dtype.contract().arithmetic().is_some());
        }
    }
}

#[test]
fn arithmetic_vocabulary_is_exact_and_fully_registered() {
    use ArithmeticRepr as A;
    let expected = [
        (A::Ieee754Binary32, Repr::Ieee754Binary32),
        (A::Ieee754Binary64, Repr::Ieee754Binary64),
        (A::ExactTwosComplement8, Repr::TwosComplement8),
        (A::ExactTwosComplement16, Repr::TwosComplement16),
        (A::ExactTwosComplement32, Repr::TwosComplement32),
        (A::ExactTwosComplement64, Repr::TwosComplement64),
    ];
    assert_eq!(ArithmeticRepr::ALL, expected.map(|row| row.0));
    for (arithmetic, repr) in expected {
        assert_eq!(arithmetic.repr(), repr);
        assert!(
            RuntimeDType::ALL
                .into_iter()
                .any(|dtype| { dtype.contract().arithmetic() == Some(arithmetic) })
        );
    }
}

#[test]
fn contract_is_available_in_constant_contexts() {
    const CONTRACT: DTypeContract = RuntimeDType::F16.contract();
    const DTYPE: RuntimeDType = CONTRACT.dtype();
    const REPR: Repr = CONTRACT.repr();
    const WIDTH: usize = CONTRACT.byte_width();
    const ARITHMETIC: Option<ArithmeticRepr> = CONTRACT.arithmetic();
    assert_eq!(DTYPE, RuntimeDType::F16);
    assert_eq!(REPR, Repr::Ieee754Binary16);
    assert_eq!(WIDTH, 2);
    assert_eq!(ARITHMETIC, Some(ArithmeticRepr::Ieee754Binary32));
}
