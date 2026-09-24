//! [04-DTYPE-2], [04-INF-6], [05-OP-36] and spec/05 §1.2:
//! a dtype binder denotes a scalar dtype, never an implicit tensor shape.
use chelis_types::{check_ir_program, check_typed_program, errors::CheckError};

const COMPARISONS: [&str; 7] = ["cmplt", "eq", "neq", "lt", "gt", "lte", "gte"];
const ARITHMETIC: [&str; 8] = [
    "add",
    "sub",
    "mul",
    "div",
    "floor_div",
    "trunc_div",
    "max_elem",
    "min_elem",
];

fn diagnostics(source: &str) -> Vec<CheckError> {
    let deep = if source.trim_start().starts_with('[') {
        serde_json::from_str(source).expect("lossless expanded AST fixture")
    } else {
        chelis_deep::parse_and_stamp_file(source).expect("Deep fixture")
    };
    deep_diagnostics(&deep)
}

fn deep_diagnostics(deep: &[chelis_deep::Expr]) -> Vec<CheckError> {
    let ir = check_ir_program(
        deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .err()
    .map_or(vec![], |r| r.errors);
    let typed = check_typed_program(
        deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .err()
    .map_or(vec![], |r| r.errors);
    let render = |errors: &[CheckError]| {
        errors
            .iter()
            .map(|e| format!("{:?}: {} {:?}", e.kind, e.message, e.suggestions))
            .collect::<Vec<_>>()
    };
    assert_eq!(render(&ir), render(&typed), "checker ingress disagreement");
    ir
}

fn accepts(source: &str) {
    let errors = diagnostics(source);
    assert!(errors.is_empty(), "{source}\n{errors:?}");
}

fn rejects(source: &str, comparison: bool) {
    let errors = diagnostics(source);
    assert_mixed(&errors, comparison, source);
}

fn assert_mixed(errors: &[CheckError], comparison: bool, source: &str) {
    let authority = if comparison {
        "[05-OP-36]"
    } else {
        "section 1.2"
    };
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("scalar beside a tensor")
                && e.message.contains(authority)
                && e.suggestions
                    .iter()
                    .any(|s| s.contains("scalar_to_tensor") && s.contains("insert"))),
        "missing mixed-surface diagnostic: {source}\n{errors:?}"
    );
    assert!(
        !errors
            .iter()
            .any(|e| format!("{:?}", e.kind).contains("OccursCheck")),
        "{errors:?}"
    );
}

#[test]
fn handwritten_deep_uses_the_same_declared_binder_at_both_ingresses() {
    for op in COMPARISONS.into_iter().chain(ARITHMETIC) {
        let comparison = COMPARISONS.contains(&op);
        let result = if comparison {
            "(t-prim {} bool)"
        } else {
            "(t-var {} p)"
        };
        for (lhs, rhs) in [
            ("(var {} xs)", "(cast {} (lit {} 1) (t-var {} p))"),
            ("(cast {} (lit {} 1) (t-var {} p))", "(var {} xs)"),
        ] {
            let source = format!(
                "(defsig {{dtype_bounds: {{p: int}}}} f (p) (t-fn {{}} (t-tensor {{}} (d-lit {{}} 3) (t-var {{}} p)) (t-tensor {{}} (d-lit {{}} 3) {result}))) (def {{}} f (fn {{}} (params {{}} xs) (app {{}} (var {{}} {op}) {lhs} {rhs})))"
            );
            let deep = chelis_deep::parser::parse_str(&source).expect("Deep parse");
            assert_mixed(&deep_diagnostics(&deep), comparison, &source);
        }
    }
}

#[test]
fn every_bounded_comparison_rejects_mixed_surfaces_in_both_orders() {
    for bound in ["Float", "Int", "Numeric"] {
        for op in COMPARISONS {
            for (lhs, rhs) in [
                (
                    "(var {} xs)",
                    "(cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 1) (t-var {} p))",
                ),
                (
                    "(cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 1) (t-var {} p))",
                    "(var {} xs)",
                ),
                ("(var {} xs)", "(var {} c)"),
                ("(var {} c)", "(var {} xs)"),
            ] {
                rejects(
                    &include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_8.deep")
                        .replace("@bound@", &bound.to_ascii_lowercase())
                        .replace("@op@", op)
                        .replace("@lhs@", lhs)
                        .replace("@rhs@", rhs),
                    true,
                );
            }
            accepts(
                &include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_9.deep")
                    .replace("@bound@", &bound.to_ascii_lowercase())
                    .replace("@op@", op),
            );
            accepts(
                &include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_10.deep")
                    .replace("@bound@", &bound.to_ascii_lowercase())
                    .replace("@op@", op),
            );
        }
    }
}

#[test]
fn arithmetic_has_the_same_surface_boundary() {
    for op in ARITHMETIC {
        let bound = if ["floor_div", "trunc_div"].contains(&op) {
            "Int"
        } else {
            "Float"
        };
        for (lhs, rhs) in [
            (
                "(var {} xs)",
                "(cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 1) (t-var {} p))",
            ),
            (
                "(cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 1) (t-var {} p))",
                "(var {} xs)",
            ),
            ("(var {} xs)", "(var {} c)"),
            ("(var {} c)", "(var {} xs)"),
        ] {
            rejects(
                &include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_19.deep")
                    .replace("@bound@", &bound.to_ascii_lowercase())
                    .replace("@op@", op)
                    .replace("@lhs@", lhs)
                    .replace("@rhs@", rhs),
                false,
            );
        }
        accepts(
            &include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_20.deep")
                .replace("@bound@", &bound.to_ascii_lowercase())
                .replace("@op@", op),
        );
        accepts(
            &include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_21.deep")
                .replace("@bound@", &bound.to_ascii_lowercase())
                .replace("@op@", op),
        );
    }
}

#[test]
fn nesting_and_aliases_do_not_erase_the_scalar_surface() {
    for body in [
        r#"(let {span: "surf:36..67"}
      (bind {}
        c
        (cast {span: "surf:42..54"}
          (lit {span: "surf:47..50", surf_literal_style: "unsuffixed", type: (t-var {} p)} 0.1)
          (t-var {} p)))
      (app {span: "surf:56..65"}
        (var {span: "surf:56..58"} gt)
        (var {span: "surf:59..61"} xs)
        (var {span: "surf:63..64"} c)))"#,
        r#"(let {span: "surf:36..70"}
      (bind {}
        c
        (cast {span: "surf:45..57",
                surf_binding_type: "explicit",
                type: (t-var {} p)
              }
          (lit {span: "surf:50..53", surf_literal_style: "unsuffixed", type: (t-var {} p)} 0.1)
          (t-var {} p)))
      (app {span: "surf:59..68"}
        (var {span: "surf:59..61"} gt)
        (var {span: "surf:62..64"} xs)
        (var {span: "surf:66..67"} c)))"#,
        r#"(let {span: "surf:36..94"}
      (bind {}
        compare
        (fn {span: "surf:48..69"}
          (params {} (c {type: (t-var {} p)}))
          (app {span: "surf:60..69"}
            (var {span: "surf:60..62"} gt)
            (var {span: "surf:63..65"} xs)
            (var {span: "surf:67..68"} c))))
      (app {span: "surf:71..92"}
        (var {span: "surf:71..78"} compare)
        (cast {span: "surf:79..91"}
          (lit {span: "surf:84..87", surf_literal_style: "unsuffixed", type: (t-var {} p)} 0.1)
          (t-var {} p))))"#,
        r#"(app {span: "surf:36..71"}
      (var {span: "surf:36..41"} where)
      (app {span: "surf:42..62"}
        (var {span: "surf:42..44"} gt)
        (var {span: "surf:45..47"} xs)
        (cast {span: "surf:49..61"}
          (lit {span: "surf:54..57", surf_literal_style: "unsuffixed", type: (t-var {} p)} 0.1)
          (t-var {} p)))
      (var {span: "surf:64..66"} xs)
      (var {span: "surf:68..70"} xs))"#,
        r#"(app {span: "surf:36..92"}
      (var {span: "surf:36..38"} gt)
      (var {span: "surf:39..41"} xs)
      (app {span: "surf:43..91"}
        (var {span: "surf:43..59"} tensor_to_scalar)
        (app {span: "surf:60..90"}
          (var {span: "surf:60..76"} scalar_to_tensor)
          (cast {span: "surf:77..89"}
            (lit {span: "surf:82..85", surf_literal_style: "unsuffixed", type: (t-var {} p)} 0.1)
            (t-var {} p)))))"#,
    ] {
        rejects(
            &include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_27.deep")
                .replace("@body@", body),
            true,
        );
    }
}

#[test]
fn explicit_shape_construction_preserves_generic_dtype() {
    for size in [0, 1, 3] {
        for op in ["gt", "mul"] {
            for extent in [
                format!("(lit {{type: (t-prim {{}} i64)}} {size})"),
                "(app {} (var {} shape) (var {} xs) (lit {type: (t-prim {} i32)} 0))".into(),
            ] {
                accepts(
                    &include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_30.deep")
                        .replace("@size@", &size.to_string())
                        .replace("@op@", op)
                        .replace("@extent@", &extent),
                );
            }
        }
    }
    accepts(include_str!(
        "fixtures/core/tests/issue_1621_scalar_surface/explicit_shape_construction_preserves_generic_dtype_1.deep"
    ));
    accepts(include_str!(
        "fixtures/core/tests/issue_1621_scalar_surface/explicit_shape_construction_preserves_generic_dtype_2.deep"
    ));
    accepts(include_str!(
        "fixtures/core/tests/issue_1621_scalar_surface/explicit_shape_construction_preserves_generic_dtype_3.deep"
    ));
    rejects(
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/explicit_shape_construction_preserves_generic_dtype_4.deep"
        ),
        true,
    );
}

#[test]
fn declaration_binders_stay_independent_and_rigid() {
    accepts(include_str!(
        "fixtures/core/tests/issue_1621_scalar_surface/declaration_binders_stay_independent_and_rigid_1.deep"
    ));
    assert!(!diagnostics(include_str!("fixtures/core/tests/issue_1621_scalar_surface/declaration_binders_stay_independent_and_rigid_2.deep")).is_empty());
    assert!(!diagnostics(include_str!("fixtures/core/tests/issue_1621_scalar_surface/declaration_binders_stay_independent_and_rigid_3.deep")).is_empty());
    accepts(include_str!(
        "fixtures/core/tests/issue_1621_scalar_surface/declaration_binders_stay_independent_and_rigid_4.deep"
    ));
}

#[test]
fn cached_library_bounds_and_lexical_aliases_preserve_the_boundary() {
    let library = chelis_deep::parse_and_stamp_file(include_str!("fixtures/core/tests/issue_1621_scalar_surface/cached_library_bounds_and_lexical_aliases_preserve_the_boundary_1.deep")).expect("Deep fixture");
    let context = chelis_types::build_type_env_from_library(
        &library,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .unwrap();
    let bytes = bincode::serialize(&context).unwrap();
    let context: chelis_types::TypeEnv = bincode::deserialize(&bytes).unwrap();
    for (operand, mixed) in [("(var {} xs)", true), ("(var {} c)", false)] {
        let source = include_str!("fixtures/core/tests/issue_1621_scalar_surface/template_33.deep")
            .replace("@operand@", operand);
        let deep = chelis_deep::parse_and_stamp_file(&source).expect("Deep fixture");
        let errors = chelis_types::check_ir_with_context(
            &context,
            &deep,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )
        .err()
        .map_or(vec![], |r| r.errors);
        if mixed {
            assert_mixed(&errors, true, &source);
        } else {
            assert!(errors.is_empty(), "{errors:?}");
        }
    }
    rejects(
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/cached_library_bounds_and_lexical_aliases_preserve_the_boundary_2.deep"
        ),
        true,
    );
    accepts(include_str!(
        "fixtures/core/tests/issue_1621_scalar_surface/cached_library_bounds_and_lexical_aliases_preserve_the_boundary_3.deep"
    ));
    accepts(include_str!(
        "fixtures/core/tests/issue_1621_scalar_surface/cached_library_bounds_and_lexical_aliases_preserve_the_boundary_4.deep"
    ));
}

/// #1621 does not enable the previously rejected generic tensor annotation
/// surface. Its dimension/rank identity obligations require a separate repair.
#[test]
fn nested_generic_tensor_annotations_keep_their_existing_rejection_boundary() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/nested_generic_tensor_annotations_keep_their_existing_rejection_boundary_1.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/nested_generic_tensor_annotations_keep_their_existing_rejection_boundary_2.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/nested_generic_tensor_annotations_keep_their_existing_rejection_boundary_3.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/nested_generic_tensor_annotations_keep_their_existing_rejection_boundary_4.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/nested_generic_tensor_annotations_keep_their_existing_rejection_boundary_5.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/nested_generic_tensor_annotations_keep_their_existing_rejection_boundary_6.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1621_scalar_surface/nested_generic_tensor_annotations_keep_their_existing_rejection_boundary_7.json"
        ),
    ] {
        assert!(
            !diagnostics(source).is_empty(),
            "must not newly admit {source}"
        );
    }
    accepts(include_str!(
        "fixtures/core/tests/issue_1621_scalar_surface/nested_generic_tensor_annotations_keep_their_existing_rejection_boundary_8.json"
    ));
}
