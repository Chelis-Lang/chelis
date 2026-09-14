//! chelis#1606 / chelis#1527: `f8e4m3` rejected by the type checker in
//! EVERY named type position, not only the tensor element slot and the
//! `cast` target that `spec/04-type-system.md` §1.1.1 already covered.
//!
//! Before this repair, `Prim::parse_name("f8e4m3")` succeeded (it is a real
//! `Prim` variant, per §1.1.1's design note), so the resolver's `t-prim` arm
//! returned `Type::Prim(F8e4m3)` before ever consulting whether the parsed
//! primitive is admissible. Only the tensor precision slot and the cast
//! target ran their own explicit `Prim::F8e4m3` check downstream of the
//! resolver, so every other position scored a clean 1.0. See
//! `f8e4m3_rejection.rs` for the pre-existing tensor/cast lock this suite
//! must not disturb.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::{check_ir_program, check_typed_program};

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn assert_rejected(src: &str, position: &str) {
    for name in ["f8e4m3", "f8e5m2"] {
        let source = src.replace("f8e4m3", name);
        let desugared = surf_to_deep(&source);
        let text = chelis_deep::printer::print_canonical_flat(&desugared);
        let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp canonical Deep");
        for (carrier, exprs) in [("desugared", desugared), ("stamped", stamped)] {
            for (entry, result) in [
                ("ir", check_ir_program(&exprs)),
                ("typed", check_typed_program(&exprs)),
            ] {
                let report = result.err().unwrap_or_else(|| {
                    panic!("{name} in {position} must reject at {carrier}/{entry}: {source}")
                });
                assert!(
                    report.errors.iter().any(|error| {
                        error.message.contains(name)
                            && error.message.contains("spec/04-type-system.md §1.1.1")
                    }),
                    "{carrier}/{entry}/{position}: {:?}",
                    report.errors
                );
            }
        }
    }
}

#[test]
fn handwritten_deep_cannot_bypass_the_type_resolver() {
    for name in ["f8e4m3", "f8e5m2"] {
        let source = format!(
            "(defsig {{}} f (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))\n\
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        );
        let exprs = chelis_deep::parse_and_stamp_file(&source).expect("stamp Deep fixture");
        for result in [check_ir_program(&exprs), check_typed_program(&exprs)] {
            let report = result.expect_err("reserved primitive must reject");
            assert!(
                report.errors.iter().any(|error| {
                    error.message.contains(name)
                        && error.message.contains("spec/04-type-system.md §1.1.1")
                }),
                "{:?}",
                report.errors
            );
        }
    }
}

#[test]
fn handwritten_reserved_type_variable_cannot_bypass_the_resolver() {
    for name in ["f8e4m3", "f8e5m2"] {
        let source = format!(
            "(defsig {{}} f (t-fn {{}} (t-var {{}} {name}) (t-var {{}} {name})))\n\
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        );
        let exprs = chelis_deep::parse_and_stamp_file(&source).expect("stamp Deep fixture");
        for result in [check_ir_program(&exprs), check_typed_program(&exprs)] {
            let report = result.expect_err("reserved type variable must reject");
            assert!(
                report.errors.iter().any(|error| {
                    error.message.contains(name)
                        && error.message.contains("spec/04-type-system.md §1.1.1")
                }),
                "{:?}",
                report.errors
            );
        }
    }
}

#[test]
fn aliases_fields_and_value_annotations_reject_fp8() {
    assert_rejected("type ReservedAlias = f8e4m3", "a type alias");
    assert_rejected("type Holder = | Holder { value: f8e4m3 }", "an ADT field");
    assert_rejected("value: f8e4m3 = 1.0", "a value annotation");
}

fn assert_accepted(source: &str) {
    let desugared = surf_to_deep(source);
    let text = chelis_deep::printer::print_canonical_flat(&desugared);
    let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp canonical Deep");
    for exprs in [&desugared, &stamped] {
        for result in [check_ir_program(exprs), check_typed_program(exprs)] {
            assert!(result.is_ok(), "{source}: {result:?}");
        }
    }
}

#[test]
fn scalar_parameter_and_return_rejected() {
    assert_rejected(
        "def f(x: f8e4m3) -> f8e4m3 = x",
        "a scalar parameter and return",
    );
}

#[test]
fn standalone_sig_rejected() {
    assert_rejected("sig f: f8e4m3 -> f8e4m3\ndef f(x) = x", "a standalone sig");
}

#[test]
fn explicit_binder_scalar_rejected() {
    assert_rejected(
        "def f[f8e4m3](x: f8e4m3) -> f8e4m3 = x",
        "an explicit binder (scalar)",
    );
}

#[test]
fn explicit_binder_tensor_rejected() {
    assert_rejected(
        "def f[f8e4m3](x: tensor[3, f8e4m3]) -> tensor[3, f8e4m3] = x",
        "an explicit binder (tensor)",
    );
}

#[test]
fn reference_type_rejected() {
    assert_rejected("def f(x: &f8e4m3) -> int32 = 0i32", "a reference type");
}

#[test]
fn tuple_element_rejected() {
    assert_rejected(
        "def f(x: (f8e4m3, int32)) -> int32 = 0i32",
        "a tuple element",
    );
}

#[test]
fn arrow_parameter_rejected() {
    assert_rejected(
        "def f(g: (f8e4m3) -> int32) -> int32 = 0i32",
        "an arrow parameter",
    );
}

#[test]
fn list_element_rejected() {
    assert_rejected("def f(x: List[f8e4m3]) -> int32 = 0i32", "a List element");
}

/// DISPOSITION LOCK. The pre-existing locks (`f8e4m3_rejection.rs`) must
/// keep rejecting after this repair: the tensor slot and the cast target.
#[test]
fn tensor_slot_and_cast_target_still_rejected() {
    assert_rejected(
        "def f(x: tensor[3, f8e4m3]) -> tensor[3, f8e4m3] = x",
        "a tensor element slot",
    );
    assert_rejected("def main() -> f32 = cast(1.0, f8e4m3)", "a cast target");
}

/// DISPOSITION LOCK. Every currently-active float dtype must keep checking
/// clean; the admissibility check added for `f8e4m3` must not widen.
#[test]
fn every_active_float_is_still_accepted() {
    for name in ["f32", "f64", "f16", "bf16"] {
        let src = format!("def f(x: {name}) -> {name} = x");
        assert_accepted(&src);
    }
}

/// DISPOSITION LOCK. An ordinary lowercase name still quantifies and still
/// checks clean; the reserved-name routing must not widen to catch it.
#[test]
fn an_ordinary_type_variable_still_checks_clean() {
    assert_accepted("def f(x: a) -> a = x");
    assert_accepted("type ScalarAlias = f32\nvalue: ScalarAlias = 1.0");
    assert_accepted("type Holder = | Holder { value: f32 }");
}
