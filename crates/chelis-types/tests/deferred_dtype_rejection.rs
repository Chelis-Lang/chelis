//! chelis#944: the reserved-but-deferred dtype names of
//! `spec/04-type-system.md` §1.1.1 rejected with a §1.1.1-citing
//! diagnostic.
//!
//! Mirrors the f8e4m3 §1.1.1 rejection contract from
//! `f8e4m3_rejection.rs` for the names PR #896 reserved: `f8e5m2`,
//! `int4`/`uint4`, `complex64`/`complex128`, `decimal128`/`decimal256`
//! (the unsigned `uint*` family has its own sibling suite in
//! `unsigned_dtype_rejection.rs`). `Prim::parse_name` returns `None`
//! for every one of these spellings, so without a dedicated rejection
//! path they fall to the generic unknown-name diagnostics with no
//! §1.1.1 citation - which is what the spec's diagnostic contract
//! forbids.

use chelis_types::check_ir_program;
use chelis_types::types::Prim;

const DEFERRED_NAMES: &[&str] = &[
    "f8e5m2",
    "int4",
    "uint4",
    "complex64",
    "complex128",
    "decimal128",
    "decimal256",
];

fn parse_core(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("valid core fixture")
}

fn assert_deferred_rejection(src: &str, dtype_name: &str) {
    let deep = parse_core(src);
    let report = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("a deferred dtype must reject");
    assert!(
        report.errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
        ) && error.message.contains(dtype_name)),
        "{dtype_name}: {:?}",
        report.errors
    );
}

/// §1.1.1: none of the reserved names resolves to a `Prim` variant.
/// Reservation is a spelling claim, not an active dtype.
#[test]
fn parse_name_rejects_every_deferred_name() {
    for name in DEFERRED_NAMES {
        assert_eq!(
            Prim::parse_name(name),
            None,
            "spec §1.1.1: `{name}` is reserved but deferred; parse_name must \
             not resolve it"
        );
    }
}

/// Cast-target surface: `cast(1, <name>)` is rejected with the §1.1.1
/// diagnostic for every reserved name.
#[test]
fn cast_scalar_to_deferred_name_rejected_with_spec_1_1_1_diagnostic() {
    for name in DEFERRED_NAMES {
        assert_deferred_rejection(
            &format!(
                r#"(defsig {{}} main (t-fn {{}} (t-prim {{}} i32)))

(def {{}}
  main
  (fn {{}}
    (params {{}})
    (cast {{span: "surf:20..35"}}
      (lit {{span: "surf:25..26", type: (t-prim {{}} i32)}} 1)
      (t-prim {{}} {name}))))
"#
            ),
            name,
        );
    }
}

/// Tensor-element surface: `tensor[3, <name>]` in a value-position
/// annotation is rejected with the §1.1.1 diagnostic for every reserved
/// name.
#[test]
fn tensor_element_deferred_name_rejected_with_spec_1_1_1_diagnostic() {
    for name in DEFERRED_NAMES {
        assert_deferred_rejection(
            &format!(
                r#"(defsig {{}} stash (t-fn {{}} (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} {name}))))

(def {{}}
  stash
  (fn {{}}
    (params {{}})
    (app {{span: "surf:35..55"}}
      (var {{span: "surf:35..44"}} to_tensor)
      (app {{span: "surf:45..54"}}
        (var {{}} Cons)
        (lit {{span: "surf:46..47", type: (t-prim {{}} i32)}} 1)
        (app {{}}
          (var {{}} Cons)
          (lit {{span: "surf:49..50", type: (t-prim {{}} i32)}} 2)
          (app {{}}
            (var {{}} Cons)
            (lit {{span: "surf:52..53", type: (t-prim {{}} i32)}} 3)
            (var {{}} Nil)))))))
"#
            ),
            name,
        );
    }
}

/// Sig surface: a reserved name in a sig's tensor precision slot must
/// reach the §1.1.1 rejection path, not be admitted as a type variable
/// (the trap the unsigned family's desugar exclusion already guards against).
#[test]
fn sig_precision_deferred_name_rejected_not_quantified() {
    for name in DEFERRED_NAMES {
        let src = format!(
            r#"(defsig {{}}
  f
  (d)
  (t-fn {{}}
    (t-tensor {{}} (d-var {{}} d) (t-prim {{}} {name}))
    (t-tensor {{}} (d-var {{}} d) (t-prim {{}} {name}))))

(def {{}} f (fn {{}} (params {{}} x) (var {{span: "surf:60..61"}} x)))
"#
        );
        assert_deferred_rejection(&src, name);
    }
}

/// Negative-parity twin: an ordinary lowercase name explicitly listed in a
/// sig remains a type variable, so the desugar exclusion is exactly the
/// reserved list and nothing wider.
#[test]
fn sig_precision_explicit_ordinary_tvar_still_quantifies() {
    let src = include_str!(
        "fixtures/core/tests/deferred_dtype_rejection/sig_precision_explicit_ordinary_tvar_still_quantifies.0.deep"
    );
    let deep = parse_core(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_ok(),
        "an ordinary lowercase precision name must still quantify; got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// Negative-parity twin: active dtypes sharing a prefix with reserved
/// names (`i32` vs `int4`, `f16` vs `f8e5m2`) must not trip the
/// deferred rejection path.
#[test]
fn active_dtypes_do_not_match_deferred_family() {
    for (src, what) in [
        (
            include_str!(
                "fixtures/core/tests/deferred_dtype_rejection/active_dtypes_do_not_match_deferred_family.0.deep"
            ),
            "i32",
        ),
        (
            include_str!(
                "fixtures/core/tests/deferred_dtype_rejection/active_dtypes_do_not_match_deferred_family.1.deep"
            ),
            "f16",
        ),
    ] {
        let deep = parse_core(src);
        let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
        assert!(
            res.is_ok(),
            "`{what}` must NOT match the deferred family; got: {:?}",
            res.err().map(|e| e.errors)
        );
    }
}
