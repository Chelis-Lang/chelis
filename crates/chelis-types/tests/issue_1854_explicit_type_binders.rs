//! chelis#1854 acceptance at both checker ingresses.

use chelis_deep::{Expr, parser::parse_str as parse_deep, printer::print_canonical_flat};
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{check_ir_program, check_typed_program};

const UNKNOWN_DTYPES: [(&str, Option<&str>); 5] = [
    ("float32", Some("f32")),
    ("fp32", Some("f32")),
    ("int33", Some("i32")),
    ("double", Some("f64")),
    ("half", Some("f16")),
];
const RESERVED_DTYPES: [&str; 2] = ["f8e4m3", "f8e5m2"];

fn surf(source: &str) -> Vec<Expr> {
    desugar_program(&parse_surf(source).expect("Surf fixture must parse"))
}

#[derive(Debug, PartialEq, Eq)]
struct Diagnostic {
    kind: String,
    message: String,
    suggestions: Vec<String>,
    severity_bits: u64,
    expected: Option<String>,
    got: Option<String>,
    span_offset: Option<usize>,
    span_id: Option<String>,
}

fn diagnostics(
    result: Result<chelis_types::CheckedProgram, chelis_types::InferResult>,
) -> Vec<Diagnostic> {
    result.err().map_or_else(Vec::new, |report| {
        report
            .errors
            .into_iter()
            .map(|error| Diagnostic {
                kind: error.kind.diagnostic_name().to_string(),
                message: error.message,
                suggestions: error.suggestions,
                severity_bits: error.severity.to_bits(),
                expected: error.expected,
                got: error.got,
                span_offset: error.span_offset,
                span_id: error.span_id,
            })
            .collect()
    })
}

fn assert_both_ingresses_reject_once(program: &[Expr], name: &str, nearest: Option<&str>) {
    let ir = diagnostics(check_ir_program(program));
    let typed = diagnostics(check_typed_program(program));
    assert_eq!(
        typed, ir,
        "checker ingresses must report the same ordered diagnostics for `{name}`"
    );
    assert_eq!(
        ir.len(),
        1,
        "`{name}` must have one owning diagnostic: {ir:?}"
    );
    assert!(
        ir[0].message.contains(&format!("`{name}`")),
        "the diagnostic must name the unknown spelling: {ir:?}"
    );
    if let Some(nearest) = nearest {
        assert!(
            ir[0]
                .suggestions
                .iter()
                .any(|suggestion| suggestion.contains(&format!("`{nearest}`"))),
            "the diagnostic for `{name}` must suggest nearest dtype `{nearest}`: {ir:?}"
        );
    }
}

fn assert_both_ingresses_reject_each_site(program: &[Expr], name: &str, expected: usize) {
    let ir = diagnostics(check_ir_program(program));
    let typed = diagnostics(check_typed_program(program));
    assert_eq!(
        typed, ir,
        "checker ingresses must report the same ordered diagnostics for `{name}`"
    );
    assert_eq!(
        ir.len(),
        expected,
        "`{name}` must report at every authored site: {ir:?}"
    );
    assert!(
        ir.iter()
            .all(|diagnostic| diagnostic.message.contains(&format!("`{name}`"))),
        "every diagnostic must name the reserved spelling: {ir:?}"
    );
}

#[test]
fn unknown_surf_scalar_names_reject_once_at_both_checker_ingresses() {
    for (name, nearest) in UNKNOWN_DTYPES {
        assert_both_ingresses_reject_once(
            &surf(&format!("def ident(x: {name}) -> {name} = x")),
            name,
            nearest,
        );
    }
    for name in RESERVED_DTYPES {
        assert_both_ingresses_reject_each_site(
            &surf(&format!("def ident(x: {name}) -> {name} = x")),
            name,
            2,
        );
    }
}

#[test]
fn distinct_unknown_surf_scalar_names_keep_distinct_diagnostics() {
    let program = surf("def convert(x: float32) -> fp32 = x");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(
        typed, ir,
        "checker ingresses must preserve diagnostic order"
    );
    assert_eq!(
        ir.len(),
        2,
        "distinct invalid names need distinct owners: {ir:?}"
    );
    for name in ["float32", "fp32"] {
        assert_eq!(
            ir.iter()
                .filter(|diagnostic| diagnostic.message.contains(&format!("`{name}`")))
                .count(),
            1,
            "`{name}` must have exactly one diagnostic: {ir:?}"
        );
    }
}

fn assert_unknown_spelling_order(program: &[Expr], expected: &[&str]) {
    let ir = diagnostics(check_ir_program(program));
    let typed = diagnostics(check_typed_program(program));
    assert_eq!(
        typed, ir,
        "checker ingresses must preserve declaration/name ownership and order"
    );
    let actual = ir
        .iter()
        .filter_map(|diagnostic| {
            expected
                .iter()
                .copied()
                .find(|name| diagnostic.message.contains(&format!("`{name}`")))
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "wrong unknown-name owners: {ir:?}");
    assert_eq!(
        ir.len(),
        expected.len(),
        "no extra diagnostic may preempt the ownership contract: {ir:?}"
    );
}

fn assert_unknown_spelling_order_amid_other_errors(program: &[Expr], expected: &[&str]) {
    let ir = diagnostics(check_ir_program(program));
    let typed = diagnostics(check_typed_program(program));
    assert_eq!(
        typed, ir,
        "checker ingresses must preserve module-scoped ownership and order"
    );
    let actual = ir
        .iter()
        .filter(|diagnostic| diagnostic.message.contains("unknown primitive type"))
        .filter_map(|diagnostic| {
            expected
                .iter()
                .copied()
                .find(|name| diagnostic.message.contains(&format!("`{name}`")))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "unknown-name owners must remain isolated across modules: {ir:?}"
    );
}

#[test]
fn standalone_signature_and_matching_inline_annotation_share_one_unknown_owner() {
    for source in [
        "sig ident: float32 -> float32\n\
         def ident(x: float32) -> float32 = x",
        "def ident(x: float32) -> float32 = x\n\
         sig ident: float32 -> float32",
        "sig ident[a]: float32 -> float32\n\
         def ident(x: float32) -> float32 = x",
        "def ident(x: float32) -> float32 = x\n\
         sig ident[a]: float32 -> float32",
    ] {
        assert_unknown_spelling_order(&surf(source), &["float32"]);
    }

    for source in [
        "(defsig {} ident
           (t-fn {} (t-prim {} float32) (t-prim {} float32)))
         (def {} ident
           (fn {} (params {} (x {type: (t-prim {} float32)}))
             (var {} x)))",
        "(def {} ident
           (fn {} (params {} (x {type: (t-prim {} float32)}))
             (var {} x)))
         (defsig {} ident
           (t-fn {} (t-prim {} float32) (t-prim {} float32)))",
    ] {
        let program = parse_deep(source).expect("Deep fixture must parse");
        assert_unknown_spelling_order(&program, &["float32"]);
    }
}

#[test]
fn explicit_binders_do_not_collapse_reserved_primitive_diagnostics() {
    for name in RESERVED_DTYPES {
        assert_both_ingresses_reject_each_site(
            &surf(&format!(
                "sig ident[a]: {name} -> {name}\n\
                 def ident(x: {name}) -> {name} = x"
            )),
            name,
            3,
        );
    }
}

#[test]
fn unknown_primitive_owners_preserve_spelling_declaration_order_and_module_scope() {
    assert_unknown_spelling_order(
        &surf(
            "sig convert: float32 -> float32\n\
             def convert(x: fp32) -> fp32 = x",
        ),
        &["float32", "fp32"],
    );
    for (source, expected) in [
        (
            "sig first: fp32 -> fp32\n\
             def first(x: fp32) -> fp32 = x\n\
             sig second: float32 -> float32\n\
             def second(x: float32) -> float32 = x",
            ["fp32", "float32"],
        ),
        (
            "sig second: float32 -> float32\n\
             def second(x: float32) -> float32 = x\n\
             sig first: fp32 -> fp32\n\
             def first(x: fp32) -> fp32 = x",
            ["float32", "fp32"],
        ),
    ] {
        assert_unknown_spelling_order(&surf(source), &expected);
    }
    assert_unknown_spelling_order(
        &surf(
            "sig first: float32 -> float32\n\
             def first(x: float32) -> float32 = x\n\
             sig second: float32 -> float32\n\
             def second(x: float32) -> float32 = x",
        ),
        &["float32", "float32"],
    );

    let modules = parse_deep(
        "(module {} Left
           (defsig {} ident
             (t-fn {} (t-prim {} float32) (t-prim {} float32)))
           (def {} ident
             (fn {} (params {} (x {type: (t-prim {} float32)}))
               (var {} x))))
         (module {} Right
           (defsig {} ident
             (t-fn {} (t-prim {} float32) (t-prim {} float32)))
           (def {} ident
             (fn {} (params {} (x {type: (t-prim {} float32)}))
               (var {} x))))",
    )
    .expect("module fixture must parse");
    assert_unknown_spelling_order_amid_other_errors(&modules, &["float32", "float32"]);
}

#[test]
fn unknown_surf_tensor_precision_names_reject_once_at_both_checker_ingresses() {
    for (name, nearest) in UNKNOWN_DTYPES
        .into_iter()
        .chain(RESERVED_DTYPES.map(|name| (name, None)))
    {
        assert_both_ingresses_reject_once(
            &surf(&format!(
                "sig ident: tensor[2, {name}] -> f32\ndef ident(x) = 0.0f32"
            )),
            name,
            nearest,
        );
    }
}

#[test]
fn undeclared_surf_dimension_and_rank_variables_reject_at_both_ingresses() {
    for (source, name, needle) in [
        (
            "sig shaped: tensor[n, f32] -> f32\ndef shaped(x) = 0.0f32",
            "n",
            "undeclared dimension variable",
        ),
        (
            "sig shaped: tensor[..r, f32] -> f32\ndef shaped(x) = 0.0f32",
            "r",
            "undeclared rank variable",
        ),
    ] {
        let program = surf(source);
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "Surf ingress parity for undeclared `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong Surf undeclared-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn explicit_surf_binders_and_active_primitives_check_clean() {
    for source in [
        "def ident[a](x: a) -> a = x",
        "sig ident[a]: a -> a\ndef ident(x) = x",
        "def constant[a]() -> i32 = 1i32",
        "sig constant[a]: i32 -> i32\ndef constant(x) = x",
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = x",
        "sig ident[n, p]: tensor[n, p] -> tensor[n, p]\ndef ident(x) = x",
        "sig ident[n, p]: tensor[n, p] -> tensor[n, p]\n\
         def ident(x: tensor[n, p]) -> tensor[n, p] = x",
        "sig ident[r]: tensor[..r, f32] -> tensor[..r, f32]\ndef ident(x) = x",
        "def ident(x: f32) -> f32 = x",
    ] {
        let program = surf(source);
        check_ir_program(&program).unwrap_or_else(|report| {
            panic!("IR ingress must accept `{source}`: {:?}", report.errors)
        });
        check_typed_program(&program).unwrap_or_else(|report| {
            panic!("typed ingress must accept `{source}`: {:?}", report.errors)
        });
    }
}

#[test]
fn declaration_body_annotations_preserve_the_existing_role_sensitive_scope() {
    for source in [
        "def ident[p](x: p) -> p = { y: p = x\n y }",
        "def ident[p](x: p) -> p = { apply = fn(y: p) -> y\n apply(x) }",
        "def ident[p](x: p) -> p = (x: p)",
        "def ident[p](xs: List[p]) -> List[p] = { ys: List[p] = xs\n ys }",
        "sig ident[p]: p -> p\ndef ident(x) = { y: p = x\n y }",
    ] {
        let program = surf(source);
        check_ir_program(&program).unwrap_or_else(|report| {
            panic!(
                "IR ingress must accept body binder use in `{source}`: {:?}",
                report.errors
            )
        });
        check_typed_program(&program).unwrap_or_else(|report| {
            panic!(
                "typed ingress must accept body binder use in `{source}`: {:?}",
                report.errors
            )
        });
    }

    for source in [
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = \
         { y: tensor[n, p] = x\n y }",
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = \
         { apply = fn(y: tensor[n, p]) -> y\n apply(x) }",
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = \
         (x: tensor[n, p])",
    ] {
        let program = surf(source);
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "checker ingress parity for `{source}`");
        assert!(
            ir.iter()
                .any(|diagnostic| { diagnostic.message.contains("tensor element precision `p`") }),
            "chelis#1904's existing tensor-annotation boundary must remain rejected: \
             {source}\n{ir:?}"
        );
    }
}

#[test]
fn body_only_surf_binders_check_at_both_ingresses() {
    for source in [
        "def maker[p]() = fn (x: p) -> x",
        "def maker[n]() = fn (x: tensor[n, f32]) -> x",
        "def maker[r]() = fn (x: tensor[..r, f32]) -> x",
    ] {
        let program = surf(source);
        check_ir_program(&program).unwrap_or_else(|report| {
            panic!(
                "normalized IR ingress must accept body-only binder in `{source}`: {:?}",
                report.errors
            )
        });
        check_typed_program(&program).unwrap_or_else(|report| {
            panic!(
                "typed ingress must accept body-only binder in `{source}`: {:?}",
                report.errors
            )
        });
    }
}

fn assert_non_function_binder_narrowing_rejects(source: &str, declaration: &str, binders: &[&str]) {
    let surf_program = surf(source);
    let canonical_source = print_canonical_flat(&surf_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared non-function Deep must parse");
    for (carrier, program) in [
        ("Surf-desugared Deep", surf_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        let ir = diagnostics(check_ir_program(program));
        let typed = diagnostics(check_typed_program(program));
        assert_eq!(
            typed, ir,
            "typed-Deep and normalized-IR ingresses must agree for `{declaration}` via {carrier}"
        );
        assert!(
            !ir.is_empty(),
            "`{declaration}` must reject concrete narrowing of its declared scheme via {carrier}"
        );
        for binder in binders {
            assert!(
                ir.iter().any(|diagnostic| {
                    diagnostic.message.contains(declaration)
                        && diagnostic.message.contains(&format!("`{binder}`"))
                        && diagnostic.message.contains("[04-INF-6]")
                }),
                "`{declaration}` must report concrete narrowing of binder `{binder}` via \
                 {carrier}: {ir:?}\n{canonical_source}"
            );
        }
    }
}

#[test]
fn non_function_declaration_binders_reject_concrete_narrowing_at_both_ingresses() {
    for (source, declaration, binders) in [
        (
            "sig type_value[p]: p\n\
             type_value = 1.0f32",
            "type_value",
            &["p"][..],
        ),
        (
            "sig dimension_value[n]: tensor[n, f32]\n\
             dimension_value = to_tensor([1.0f32, 2.0f32])",
            "dimension_value",
            &["n"][..],
        ),
        (
            "sig rank_value[r]: tensor[..r, f32]\n\
             rank_value = to_tensor([1.0f32])",
            "rank_value",
            &["r"][..],
        ),
        (
            "sig matrix_value[rows, cols]: tensor[rows, cols, f32]\n\
             matrix_value = to_tensor([\n\
               [1.0f32, 2.0f32],\n\
               [3.0f32, 4.0f32]\n\
             ])",
            "matrix_value",
            &["rows", "cols"][..],
        ),
    ] {
        assert_non_function_binder_narrowing_rejects(source, declaration, binders);
    }
}

#[test]
fn unconstrained_polymorphic_non_function_value_remains_valid_at_both_ingresses() {
    let surf_program = surf("sig empty[p]: List[p]\nempty = []");
    let canonical_source = print_canonical_flat(&surf_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared polymorphic value must parse");
    for (carrier, program) in [
        ("Surf-desugared Deep", surf_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        check_ir_program(program).unwrap_or_else(|report| {
            panic!(
                "normalized-IR ingress must preserve an unconstrained polymorphic value through \
                 {carrier}: {:?}",
                report.errors
            )
        });
        check_typed_program(program).unwrap_or_else(|report| {
            panic!(
                "typed-Deep ingress must preserve an unconstrained polymorphic value through \
                 {carrier}: {:?}",
                report.errors
            )
        });
    }
}

#[test]
fn body_only_type_binder_is_generalized_and_remains_rigid() {
    let polymorphic = surf(
        "def maker[p]() = fn (x: p) -> x\n\
         int_result = (maker())(1i32)\n\
         float_result = (maker())(1.0f32)",
    );
    check_ir_program(&polymorphic)
        .expect("normalized IR ingress must generalize the body-only type binder");
    check_typed_program(&polymorphic)
        .expect("typed ingress must generalize the body-only type binder");

    let narrowed = surf("def maker[p]() = fn (x: p) -> add(x, 1i32)");
    let ir = diagnostics(check_ir_program(&narrowed));
    let typed = diagnostics(check_typed_program(&narrowed));
    assert_eq!(typed, ir, "body-only binder rigidity ingress parity");
    assert!(
        ir.iter().any(|diagnostic| {
            diagnostic.message.contains("declared type parameter `p`")
                && diagnostic.message.contains("[04-INF-6]")
        }),
        "a body-only authored binder must remain rigid: {ir:?}"
    );
}

fn assert_body_only_shape_binder_pin_rejects(
    source: &str,
    declaration: &str,
    binder: &str,
    role: &str,
) {
    let surf_program = surf(source);
    let canonical_source = print_canonical_flat(&surf_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared canonical Deep must parse");

    for (carrier, program) in [
        ("Surf-desugared Deep", surf_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        let ir = diagnostics(check_ir_program(program));
        let typed = diagnostics(check_typed_program(program));
        assert_eq!(
            typed, ir,
            "{carrier} checker ingresses must agree for body-only {role} binder `{binder}`"
        );
        assert!(
            ir.iter().any(|diagnostic| {
                diagnostic.message.contains(declaration)
                    && diagnostic.message.contains(&format!("`{binder}`"))
                    && diagnostic.message.contains("[04-INF-6]")
            }),
            "{carrier} must reject concrete narrowing of body-only {role} binder `{binder}`: \
             {ir:?}\n{canonical_source}"
        );
    }
}

#[test]
fn body_only_dimension_and_rank_binders_cannot_be_pinned_to_concrete_shapes() {
    for (source, binder, role) in [
        (
            "def narrowed_dimension[n]() = {\n\
               value: tensor[n, f32] = to_tensor([1.0f32, 2.0f32])\n\
               value\n\
             }",
            "n",
            "dimension",
        ),
        (
            "def narrowed_rank[r]() = {\n\
               value: tensor[..r, f32] = to_tensor([1.0f32, 2.0f32])\n\
               value\n\
             }",
            "r",
            "rank",
        ),
    ] {
        let declaration = if role == "dimension" {
            "narrowed_dimension"
        } else {
            "narrowed_rank"
        };
        assert_body_only_shape_binder_pin_rejects(source, declaration, binder, role);
    }
}

#[test]
fn recursive_body_only_dimension_and_rank_binders_remain_rigid() {
    for (source, declaration, binder, role) in [
        (
            "def recursive_dimension[n](stop: bool) -> i32 = {\n\
               value: tensor[n, f32] = to_tensor([1.0f32, 2.0f32])\n\
               if stop then 0i32 else recursive_dimension(true)\n\
             }",
            "recursive_dimension",
            "n",
            "dimension",
        ),
        (
            "def recursive_rank[r](stop: bool) -> i32 = {\n\
               value: tensor[..r, f32] = to_tensor([1.0f32, 2.0f32])\n\
               if stop then 0i32 else recursive_rank(true)\n\
             }",
            "recursive_rank",
            "r",
            "rank",
        ),
    ] {
        assert_body_only_shape_binder_pin_rejects(source, declaration, binder, role);
    }
}

fn assert_body_only_dimension_cannot_collapse_into_return_only(source: &str, declaration: &str) {
    let surf_program = surf(source);
    let canonical_source = print_canonical_flat(&surf_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared canonical Deep must parse");

    for (carrier, program) in [
        ("Surf-desugared Deep", surf_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        let ir = diagnostics(check_ir_program(program));
        let typed = diagnostics(check_typed_program(program));
        assert_eq!(
            typed, ir,
            "{carrier} checker ingresses must agree for `{declaration}`"
        );
        assert!(
            ir.iter().any(|diagnostic| {
                diagnostic.message.contains(declaration)
                    && diagnostic.message.contains("`n`")
                    && diagnostic.message.contains("`m`")
                    && diagnostic.message.contains("[04-INF-6]")
            }),
            "{carrier} must reject a rigid body-only `m` collapsed into return-only `n`: \
             {ir:?}\n{canonical_source}"
        );
    }
}

#[test]
fn body_only_dimension_cannot_collapse_into_a_distinct_return_only_binder() {
    for (source, declaration) in [
        (
            "def mixed_return_only[n, m](values: List[f32]) -> tensor[n, f32] = {\n\
               result: tensor[m, f32] = to_tensor(values)\n\
               result\n\
             }",
            "mixed_return_only",
        ),
        (
            "def recursive_mixed_return_only[n, m](\n\
               values: List[f32],\n\
               stop: bool\n\
             ) -> tensor[n, f32] = {\n\
               result: tensor[m, f32] = to_tensor(values)\n\
               if stop then result else recursive_mixed_return_only(values, true)\n\
             }",
            "recursive_mixed_return_only",
        ),
    ] {
        assert_body_only_dimension_cannot_collapse_into_return_only(source, declaration);
    }
}

#[test]
fn distinct_return_only_dimensions_may_share_one_body_inferred_output() {
    let source = "def shared_return_only[n, m](values: List[f32]) \
                  -> (tensor[n, f32], tensor[m, f32]) = {\n\
                    result = to_tensor(values)\n\
                    (result, result)\n\
                  }";
    let surf_program = surf(source);
    let canonical_source = print_canonical_flat(&surf_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared canonical Deep must parse");

    for (carrier, program) in [
        ("Surf-desugared Deep", surf_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        check_ir_program(program).unwrap_or_else(|report| {
            panic!(
                "normalized IR ingress must preserve return-only output inference through \
                 {carrier}: {:?}",
                report.errors
            )
        });
        check_typed_program(program).unwrap_or_else(|report| {
            panic!(
                "typed ingress must preserve return-only output inference through {carrier}: {:?}",
                report.errors
            )
        });
    }
}

#[test]
fn body_only_surf_annotations_cannot_introduce_undeclared_binders() {
    for (source, name, needle) in [
        (
            "def maker() = fn (x: p) -> x",
            "p",
            "unknown primitive type",
        ),
        (
            "def maker() = fn (x: tensor[n, f32]) -> x",
            "n",
            "undeclared dimension variable",
        ),
        (
            "def maker() = fn (x: tensor[..r, f32]) -> x",
            "r",
            "undeclared rank variable",
        ),
    ] {
        let program = surf(source);
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "undeclared Surf body-binder parity for `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong undeclared Surf body-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn canonical_deep_body_only_binders_fill_a_wildcard_result_at_both_ingresses() {
    for (binder, annotation) in [
        ("p", "(t-var {} p)"),
        ("n", "(t-tensor {} (d-var {} n) (t-prim {} f32))"),
        ("r", "(t-tensor {} (d-rank {} r) (t-prim {} f32))"),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} maker ({binder}) (t-fn {{}} (t-var {{}} _)))
             (def {{}}
               maker
               (fn {{}}
                 (params {{}})
                 (fn {{}}
                   (params {{}} (x {{type: {annotation}}}))
                   (var {{}} x))))"
        ))
        .expect("canonical Deep fixture must parse");
        check_ir_program(&program).unwrap_or_else(|report| {
            panic!(
                "normalized IR ingress must let body-only `{binder}` fill the wildcard: {:?}",
                report.errors
            )
        });
        check_typed_program(&program).unwrap_or_else(|report| {
            panic!(
                "typed ingress must let body-only `{binder}` fill the wildcard: {:?}",
                report.errors
            )
        });
    }
}

#[test]
fn body_annotations_cannot_introduce_undeclared_binders_at_either_ingress() {
    for (name, annotation, needle) in [
        ("p", "(t-var {} p)", "undeclared type variable"),
        (
            "n",
            "(t-tensor {} (d-var {} n) (t-prim {} f32))",
            "undeclared dimension variable",
        ),
        (
            "r",
            "(t-tensor {} (d-rank {} r) (t-prim {} f32))",
            "undeclared rank variable",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} maker (t-fn {{}} (t-var {{}} _)))
             (def {{}}
               maker
               (fn {{}}
                 (params {{}})
                 (fn {{}}
                   (params {{}} (x {{type: {annotation}}}))
                   (var {{}} x))))"
        ))
        .expect("canonical Deep fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(
            typed, ir,
            "undeclared body-binder ingress parity for `{name}`"
        );
        assert_eq!(ir.len(), 1, "`{name}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong undeclared body-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn explicit_deep_t_var_nodes_remain_valid_binders() {
    let program = parse_deep(
        "(defsig {} ident (float32)
           (t-fn {} (t-var {} float32) (t-var {} float32)))
         (def {} ident
           (fn {} (params {} (x {type: (t-var {} float32)})) (var {} x)))",
    )
    .expect("Deep fixture must parse");
    check_ir_program(&program).expect("an authored Deep `t-var` is explicit");
    check_typed_program(&program).expect("typed ingress must agree");
}

#[test]
fn forbidden_dtype_names_reject_once_at_the_deep_binder_list() {
    for (name, ty, body) in [
        (
            "f32",
            "(t-fn {} (t-tensor {} (d-var {} f32) (t-prim {} i32)) (t-prim {} i32))",
            "(fn {} (params {} x) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "f8e4m3",
            "(t-fn {} (t-tensor {} (d-var {} f8e4m3) (t-prim {} i32)) (t-prim {} i32))",
            "(fn {} (params {} x) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "i32",
            "(t-fn {} (t-tensor {} (d-rank {} i32) (t-prim {} f32)) (t-prim {} i32))",
            "(fn {} (params {} x) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "f8e5m2",
            "(t-fn {} (t-tensor {} (d-rank {} f8e5m2) (t-prim {} f32)) (t-prim {} i32))",
            "(fn {} (params {} x) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "bool",
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "u8",
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "int32",
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "complex64",
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} ident ({name})
               {ty})
             (def {{}} ident {body})"
        ))
        .expect("Deep fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "forbidden-binder rejection parity for `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must report once: {ir:?}");
        assert!(
            ir[0].message.contains("cannot be a `defsig` binder")
                && ir[0].message.contains(&format!("`{name}`")),
            "wrong forbidden-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn explicit_deep_unknown_primitives_reject_like_desugared_surf() {
    for (name, nearest) in UNKNOWN_DTYPES {
        let program = parse_deep(&format!(
            "(defsig {{}} ident
               (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))
             (def {{}} ident
               (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("Deep fixture must parse");
        assert_both_ingresses_reject_once(&program, name, nearest);
    }
    for name in RESERVED_DTYPES {
        let program = parse_deep(&format!(
            "(defsig {{}} ident
               (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))
             (def {{}} ident
               (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("Deep fixture must parse");
        assert_both_ingresses_reject_each_site(&program, name, 2);
    }
}

#[test]
fn distinct_unknown_deep_scalar_names_keep_distinct_diagnostics() {
    let program = parse_deep(
        "(defsig {} convert
           (t-fn {} (t-prim {} float32) (t-prim {} fp32)))
         (def {} convert (fn {} (params {} x) (var {} x)))",
    )
    .expect("Deep fixture must parse");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(
        typed, ir,
        "checker ingresses must preserve diagnostic order"
    );
    assert_eq!(
        ir.len(),
        2,
        "distinct invalid names need distinct owners: {ir:?}"
    );
    for name in ["float32", "fp32"] {
        assert_eq!(
            ir.iter()
                .filter(|diagnostic| diagnostic.message.contains(&format!("`{name}`")))
                .count(),
            1,
            "`{name}` must have exactly one diagnostic: {ir:?}"
        );
    }
}

#[test]
fn arbitrary_non_dtype_names_remain_legal_deep_binders() {
    for (ty, body) in [
        (
            "(t-fn {} (t-tensor {} (d-var {} float32) (t-prim {} f32)) \
                       (t-tensor {} (d-var {} float32) (t-prim {} f32)))",
            "(fn {} (params {} x) (var {} x))",
        ),
        (
            "(t-fn {} (t-tensor {} (d-rank {} float32) (t-prim {} f32)) \
                       (t-tensor {} (d-rank {} float32) (t-prim {} f32)))",
            "(fn {} (params {} x) (var {} x))",
        ),
        (
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} ident (float32) {ty})
             (def {{}} ident {body})"
        ))
        .expect("Deep fixture must parse");
        check_ir_program(&program).expect("arbitrary Deep binder must remain legal");
        check_typed_program(&program).expect("typed ingress must agree");
    }
}

#[test]
fn undeclared_deep_type_dimension_and_rank_variables_reject() {
    for (name, ty, needle) in [
        (
            "a",
            "(t-fn {} (t-var {} a) (t-prim {} f32))",
            "undeclared type variable",
        ),
        (
            "n",
            "(t-fn {} (t-tensor {} (d-var {} n) (t-prim {} f32)) (t-prim {} f32))",
            "undeclared dimension variable",
        ),
        (
            "r",
            "(t-fn {} (t-tensor {} (d-rank {} r) (t-prim {} f32)) (t-prim {} f32))",
            "undeclared rank variable",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {ty})
             (def {{}} f (fn {{}} (params {{}} x) (lit {{type: (t-prim {{}} f32)}} 0.0)))"
        ))
        .expect("Deep fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "ingress parity for `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must report once: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong undeclared-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn empty_deep_binder_list_is_rejected_at_both_checker_ingresses() {
    let program = parse_deep(
        "(defsig {} f () (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 0.0)))",
    )
    .expect("structural fixture parses before semantic validation");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(typed, ir, "empty-list rejection must have ingress parity");
    assert_eq!(ir.len(), 1, "empty binder list must have one owner: {ir:?}");
    assert!(
        ir[0].message.contains("empty list must be omitted"),
        "wrong empty-list diagnostic: {ir:?}"
    );
}

#[test]
fn malformed_and_duplicate_deep_binder_lists_reject_at_both_ingresses() {
    for (binders, needle) in [
        ("(a a)", "duplicate `defsig` binder `a`"),
        (
            "(a 1)",
            "expected a nonempty structural list of distinct symbol names",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {binders} (t-fn {{}} (t-var {{}} a) (t-var {{}} a)))
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("structural fixture parses before checker validation");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "malformed-list ingress parity for `{binders}`");
        assert_eq!(ir.len(), 1, "`{binders}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle),
            "wrong malformed-list diagnostic for `{binders}`: {ir:?}"
        );
    }
}

#[test]
fn declared_deep_type_dimension_and_rank_variables_check_clean() {
    for (binders, ty) in [
        ("(a)", "(t-fn {} (t-var {} a) (t-var {} a))"),
        (
            "(n)",
            "(t-fn {} (t-tensor {} (d-var {} n) (t-prim {} f32)) \
                      (t-tensor {} (d-var {} n) (t-prim {} f32)))",
        ),
        (
            "(r)",
            "(t-fn {} (t-tensor {} (d-rank {} r) (t-prim {} f32)) \
                      (t-tensor {} (d-rank {} r) (t-prim {} f32)))",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {binders} {ty})
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("Deep fixture must parse");
        check_ir_program(&program).expect("declared Deep binder must check");
        check_typed_program(&program).expect("typed ingress must agree");
    }
}

#[test]
fn declared_but_unused_deep_binder_is_a_vacuous_quantifier() {
    let program = parse_deep(
        "(defsig {} constant (a) (t-fn {} (t-prim {} i32)))
         (def {} constant
           (fn {} (params {}) (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("Deep fixture must parse");
    check_ir_program(&program).expect("IR ingress must accept an unused unbounded binder");
    check_typed_program(&program).expect("typed ingress must accept the same binder policy");
}

#[test]
fn declared_but_unused_bounded_deep_binder_rejects_at_both_ingresses() {
    let program = parse_deep(
        "(defsig {dtype_bounds: {a: numeric}} constant (a)
           (t-fn {} (t-prim {} i32)))
         (def {} constant
           (fn {} (params {}) (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("Deep fixture must parse");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(
        typed, ir,
        "bounded-unused rejection must have ingress parity"
    );
    assert_eq!(
        ir.len(),
        1,
        "bounded-unused binder must report once: {ir:?}"
    );
    assert!(
        ir[0].message.contains("does not occur") && ir[0].message.contains("`a`"),
        "wrong bounded-unused diagnostic: {ir:?}"
    );
}
