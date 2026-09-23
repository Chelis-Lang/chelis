#[inline(never)]
pub fn synth_const(wide: f64) -> f64 {
    wide
}

// The production grad_dag_result call passes this exact F64 literal to synth_const.
pub fn adjoint_seed() -> f64 {
    synth_const(1.0)
}

pub fn negative_zero32() -> f32 {
    -0.0
}

pub fn negative_zero64() -> f64 {
    -0.0
}

// These are runtime bit conversions, not decimal NaN literals: the payload must survive.
pub fn nan_payload32(bits: u32) -> f32 {
    f32::from_bits(bits)
}

pub fn nan_payload64(bits: u64) -> f64 {
    f64::from_bits(bits)
}

pub fn signed_and_unsigned() -> (i64, u64) {
    (-1, u64::MAX)
}

pub struct SpecAtomRef(&'static str);

impl SpecAtomRef {
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

pub struct Authority {
    pub atom: SpecAtomRef,
    hint: &'static str,
}

impl Authority {
    pub const fn hint(self) -> &'static str {
        self.hint
    }
}

pub fn static_atom() -> &'static str {
    SpecAtomRef("atom").as_str()
}

pub fn static_hint() -> &'static str {
    Authority {
        atom: SpecAtomRef("atom"),
        hint: "hint",
    }
    .hint()
}

// Reuse one static input across both constructors and return both shared aliases.
// Dropping Authority.atom must not end the loan used by either returned reference.
pub fn static_shared_argument(value: &'static str) -> (&'static str, &'static str) {
    let atom = SpecAtomRef(value).as_str();
    let hint = Authority {
        atom: SpecAtomRef(value),
        hint: value,
    }
    .hint();
    (atom, hint)
}

#[test]
fn exact_constants_and_static_returns() {
    assert_eq!(adjoint_seed().to_bits(), 0x3ff0_0000_0000_0000);
    assert_eq!(negative_zero32().to_bits(), 0x8000_0000);
    assert_eq!(negative_zero64().to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(nan_payload32(0x7fc0_0042).to_bits(), 0x7fc0_0042);
    assert_eq!(nan_payload64(0x7ff8_0000_0000_0042).to_bits(), 0x7ff8_0000_0000_0042);
    assert_eq!(signed_and_unsigned(), (-1, u64::MAX));
    assert_eq!(static_atom(), "atom");
    assert_eq!(static_hint(), "hint");
    assert_eq!(static_shared_argument("input"), ("input", "input"));
}
