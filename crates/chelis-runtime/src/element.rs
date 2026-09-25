//! C1's closed storage owner. No pointer access lives in this module.
use super::{Bool8, KeyWord, TensorElement};
use chelis_vocab::{ArithmeticRepr, Repr, RuntimeDType};

pub use half::{bf16 as Bf16Bits, f16 as F16Bits};

mod private {
    pub trait Sealed {}
}

// Public only to appear in TensorElement's bound. The private supertrait
// prevents even a sibling runtime module from extending the storage universe.
pub trait ElementStorage: private::Sealed + Copy {
    const STORAGE_DTYPE: RuntimeDType;
    const STORED_REPR: Repr;
    type ArithmeticStorage;
}

impl<T: ElementStorage> TensorElement for T {
    const DTYPE: RuntimeDType = T::STORAGE_DTYPE;
    const REPR: Repr = T::STORED_REPR;
    type Arithmetic = T::ArithmeticStorage;
}

impl private::Sealed for f64 {}
impl ElementStorage for f64 {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::F64;
    const STORED_REPR: Repr = Repr::Ieee754Binary64;
    type ArithmeticStorage = f64;
}

impl private::Sealed for f32 {}
impl ElementStorage for f32 {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::F32;
    const STORED_REPR: Repr = Repr::Ieee754Binary32;
    type ArithmeticStorage = f32;
}

impl private::Sealed for F16Bits {}
impl ElementStorage for F16Bits {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::F16;
    const STORED_REPR: Repr = Repr::Ieee754Binary16;
    type ArithmeticStorage = f32;
}

impl private::Sealed for Bf16Bits {}
impl ElementStorage for Bf16Bits {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::Bf16;
    const STORED_REPR: Repr = Repr::Bfloat16;
    type ArithmeticStorage = f32;
}

impl private::Sealed for i64 {}
impl ElementStorage for i64 {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::I64;
    const STORED_REPR: Repr = Repr::TwosComplement64;
    type ArithmeticStorage = i64;
}

impl private::Sealed for i32 {}
impl ElementStorage for i32 {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::I32;
    const STORED_REPR: Repr = Repr::TwosComplement32;
    type ArithmeticStorage = i32;
}

impl private::Sealed for i16 {}
impl ElementStorage for i16 {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::I16;
    const STORED_REPR: Repr = Repr::TwosComplement16;
    type ArithmeticStorage = i16;
}

impl private::Sealed for i8 {}
impl ElementStorage for i8 {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::I8;
    const STORED_REPR: Repr = Repr::TwosComplement8;
    type ArithmeticStorage = i8;
}

impl private::Sealed for Bool8 {}
impl ElementStorage for Bool8 {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::Bool;
    const STORED_REPR: Repr = Repr::Bool8;
    type ArithmeticStorage = ();
}

impl private::Sealed for KeyWord {}
impl ElementStorage for KeyWord {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::Key;
    const STORED_REPR: Repr = Repr::Word64;
    type ArithmeticStorage = ();
}

trait ArithmeticIdentity {
    const REPR: Option<ArithmeticRepr>;
}

impl ArithmeticIdentity for f64 {
    const REPR: Option<ArithmeticRepr> = Some(ArithmeticRepr::Ieee754Binary64);
}
impl ArithmeticIdentity for f32 {
    const REPR: Option<ArithmeticRepr> = Some(ArithmeticRepr::Ieee754Binary32);
}
impl ArithmeticIdentity for i64 {
    const REPR: Option<ArithmeticRepr> = Some(ArithmeticRepr::ExactTwosComplement64);
}
impl ArithmeticIdentity for i32 {
    const REPR: Option<ArithmeticRepr> = Some(ArithmeticRepr::ExactTwosComplement32);
}
impl ArithmeticIdentity for i16 {
    const REPR: Option<ArithmeticRepr> = Some(ArithmeticRepr::ExactTwosComplement16);
}
impl ArithmeticIdentity for i8 {
    const REPR: Option<ArithmeticRepr> = Some(ArithmeticRepr::ExactTwosComplement8);
}
impl ArithmeticIdentity for () {
    const REPR: Option<ArithmeticRepr> = None;
}

const fn same_arithmetic(left: Option<ArithmeticRepr>, right: Option<ArithmeticRepr>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left as usize == right as usize,
        (None, None) => true,
        _ => false,
    }
}

const fn assert_registration<T: TensorElement>(dtype: RuntimeDType)
where
    T::Arithmetic: ArithmeticIdentity,
{
    assert!(
        T::DTYPE.id() == dtype.id(),
        "element registration must match dtype"
    );
    assert!(
        T::REPR as usize == dtype.contract().repr() as usize,
        "stored representation must match dtype contract"
    );
    assert!(
        size_of::<T>() == dtype.contract().byte_width(),
        "storage size must match dtype contract"
    );
    assert!(
        same_arithmetic(
            <T::Arithmetic as ArithmeticIdentity>::REPR,
            dtype.contract().arithmetic()
        ),
        "arithmetic representation must match dtype contract"
    );
}

// This match, not RuntimeDType::ALL alone, makes a newly introduced dtype a
// compile error before it can lack a checked storage registration.
const fn assert_registered_dtype(dtype: RuntimeDType) {
    match dtype {
        RuntimeDType::F64 => assert_registration::<f64>(dtype),
        RuntimeDType::F32 => assert_registration::<f32>(dtype),
        RuntimeDType::F16 => assert_registration::<F16Bits>(dtype),
        RuntimeDType::Bf16 => assert_registration::<Bf16Bits>(dtype),
        RuntimeDType::I64 => assert_registration::<i64>(dtype),
        RuntimeDType::I32 => assert_registration::<i32>(dtype),
        RuntimeDType::I16 => assert_registration::<i16>(dtype),
        RuntimeDType::I8 => assert_registration::<i8>(dtype),
        RuntimeDType::Bool => assert_registration::<Bool8>(dtype),
        RuntimeDType::Key => assert_registration::<KeyWord>(dtype),
    }
}

const _: () = {
    let mut index = 0;
    while index < RuntimeDType::ALL.len() {
        assert_registered_dtype(RuntimeDType::ALL[index]);
        index += 1;
    }
};
