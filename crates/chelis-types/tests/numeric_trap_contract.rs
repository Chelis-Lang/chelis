//! chelis#729 Phase 2: the numeric-trap diagnostic grammar is a public,
//! frozen cross-lane contract rather than duplicated string literals.

use chelis_types::{
    NUMERIC_TRAP_DIV_ZERO_KIND, NUMERIC_TRAP_DOMAIN_KIND, NUMERIC_TRAP_DTYPE_SEPARATOR,
    NUMERIC_TRAP_OPERATION_SEPARATOR, NUMERIC_TRAP_OVERFLOW_KIND, NUMERIC_TRAP_PREFIX, NumericTrap,
    types::Prim,
};

const ALL_PRIMS: [Prim; 11] = [
    Prim::F32,
    Prim::F64,
    Prim::F16,
    Prim::Bf16,
    Prim::F8e4m3,
    Prim::Int8,
    Prim::Int16,
    Prim::Int32,
    Prim::Int64,
    Prim::Bool,
    Prim::String,
];

#[test]
fn public_fragments_freeze_the_exact_numeric_trap_grammar() {
    assert_eq!(NUMERIC_TRAP_PREFIX, "numeric trap: ");
    assert_eq!(NUMERIC_TRAP_OVERFLOW_KIND, "overflow");
    assert_eq!(NUMERIC_TRAP_DOMAIN_KIND, "domain");
    assert_eq!(NUMERIC_TRAP_DIV_ZERO_KIND, "division by zero");
    assert_eq!(NUMERIC_TRAP_OPERATION_SEPARATOR, " in ");
    assert_eq!(NUMERIC_TRAP_DTYPE_SEPARATOR, " at ");
}

#[test]
fn every_kind_and_dtype_render_through_the_frozen_grammar() {
    for prim in ALL_PRIMS {
        let cases = [
            (
                NumericTrap::Overflow { op: "add", prim },
                NUMERIC_TRAP_OVERFLOW_KIND,
                "add",
            ),
            (
                NumericTrap::Domain { op: "cast", prim },
                NUMERIC_TRAP_DOMAIN_KIND,
                "cast",
            ),
            (
                NumericTrap::DivZero {
                    op: "trunc_div",
                    prim,
                },
                NUMERIC_TRAP_DIV_ZERO_KIND,
                "trunc_div",
            ),
        ];
        for (trap, kind, op) in cases {
            assert_eq!(
                trap.to_string(),
                format!(
                    "{NUMERIC_TRAP_PREFIX}{kind}{NUMERIC_TRAP_OPERATION_SEPARATOR}{op}\
                     {NUMERIC_TRAP_DTYPE_SEPARATOR}{}",
                    prim.name()
                )
            );
        }
    }
}

#[test]
fn operation_slot_is_the_canonical_raising_primitive() {
    assert_eq!(
        NumericTrap::Overflow {
            op: "sum",
            prim: Prim::Int8,
        }
        .to_string(),
        "numeric trap: overflow in sum at i8"
    );
}
