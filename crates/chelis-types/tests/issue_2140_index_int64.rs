//! chelis#2140: direct `index` applications require the exact i64 index type.

use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut messages = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>();
    messages.sort();
    messages
}

fn whole_program(source: &str) -> [(&'static str, Result<(), Vec<CheckError>>); 2] {
    let desugared: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    [
        (
            "typed",
            check_typed_program(&desugared, chelis_types::ProgramOrigin::Unlinked)
                .map(|_| ())
                .map_err(|report| report.errors),
        ),
        (
            "IR",
            check_ir_program(&desugared, chelis_types::ProgramOrigin::Unlinked)
                .map(|_| ())
                .map_err(|report| report.errors),
        ),
    ]
}

fn assert_accepted(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        if let Err(errors) = result {
            panic!(
                "{label} [{entry}] over-rejected:\n{}\nsource:\n{source}",
                rendered(&errors).join("\n")
            );
        }
    }
}

fn assert_index_type_rejected(label: &str, source: &str, index_type: &str) {
    let expected = format!("index expects i64 index, got {index_type}");
    for (entry, result) in whole_program(source) {
        let errors = match result {
            Err(errors) => errors,
            Ok(()) => panic!("{label} [{entry}] unexpectedly accepted:\n{source}"),
        };
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::TypeMismatch)
                    && error.message.contains(&expected)
            }),
            "{label} [{entry}] expected operation-owned diagnostic {expected:?}:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

#[test]
fn direct_index_accepts_exact_i64_at_both_checker_ingresses() {
    assert_accepted(
        "exact i64 index",
        include_str!(
            "fixtures/core/tests/issue_2140_index_int64/direct_index_accepts_exact_i64_at_both_checker_ingresses.0.deep"
        ),
    );
}

#[test]
fn direct_index_rejects_every_other_integer_width_at_both_checker_ingresses() {
    // Mutation-equivalent negative control: restoring Prim::is_integer() in
    // the direct route makes every one of these witnesses incorrectly pass.
    for index_type in ["i8", "i16", "i32"] {
        let source = format!(
            r#"(defsig {{}} out (t-prim {{}} i64))

(def {{}}
  out
  (app {{span: "surf:11..30"}}
    (var {{span: "surf:11..16"}} index)
    (app {{span: "surf:17..23"}}
      (var {{}} Cons)
      (lit {{span: "surf:18..22", type: (t-prim {{}} i64)}} 1)
      (var {{}} Nil))
    (lit {{span: "surf:25..29", type: (t-prim {{}} {index_type})}} 0)))
"#
        );
        assert_index_type_rejected(index_type, &source, index_type);
    }
}

#[test]
fn direct_index_rejects_active_non_integer_primitives_at_both_checker_ingresses() {
    // Reserved f8e4m3 is rejected earlier by its UnsupportedTensorPrecision
    // owner and therefore must not be forced through this operation-owned guard.
    for index_type in ["f16", "bf16", "f32", "f64", "bool", "string"] {
        let source = format!(
            r#"(defsig {{}}
  pick
  (t-fn {{}} (t-adt {{}} List (t-prim {{}} i64)) (t-prim {{}} {index_type}) (t-prim {{}} i64)))

(def {{}}
  pick
  (fn {{}}
    (params {{}} (xs {{type: (t-var {{}} _)}}) (i {{type: (t-var {{}} _)}}))
    (app {{span: "surf:41..53"}}
      (var {{span: "surf:41..46"}} index)
      (var {{span: "surf:47..49"}} xs)
      (var {{span: "surf:51..52"}} i))))
"#
        );
        assert_index_type_rejected(index_type, &source, index_type);
    }
}
