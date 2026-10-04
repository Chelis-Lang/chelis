//! Authored generic contracts must entail every operation's dtype requirements.
use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{
    check_ir_program, check_typed_program,
    errors::{CheckError, CheckErrorKind},
};

fn check(source: &str, accepted: bool) {
    let program = desugar_program(&parse_str(source).expect("valid source"))
        .expect("Surf fixture must desugar");
    let result = check_typed_program(&program);
    if accepted {
        assert!(result.is_ok(), "{source}\n{result:?}");
    } else {
        let Err(report) = result else {
            panic!("the authored contract is insufficient: {source}");
        };
        assert!(
            report
                .errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::PrecisionMismatch)),
            "{source}\n{report:?}"
        );
    }
}

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut messages: Vec<_> = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect();
    messages.sort();
    messages
}

fn both_ingress_diagnostics(source: &str) -> Vec<CheckError> {
    let parsed = parse_str(source).expect("valid source");
    let desugared = desugar_program(&parsed).expect("Surf fixture must desugar");
    let expanded: Vec<Expr> = expand_program(&desugared, &ExpansionOptions::default())
        .expect("macro expansion")
        .into_exprs();
    let typed = check_typed_program(&desugared)
        .err()
        .map_or_else(Vec::new, |report| report.errors);
    let ir = check_ir_program(&expanded)
        .err()
        .map_or_else(Vec::new, |report| report.errors);
    assert_eq!(
        rendered(&typed),
        rendered(&ir),
        "the stamped and normalized checker ingresses disagree:\n{source}\n\
         typed={typed:?}\nir={ir:?}"
    );
    typed
}

fn check_both_ingresses(source: &str, accepted: bool) {
    let diagnostics = both_ingress_diagnostics(source);
    if accepted {
        assert!(
            diagnostics.is_empty(),
            "both ingresses must accept:\n{source}\n{diagnostics:?}"
        );
    } else {
        assert!(
            diagnostics
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::PrecisionMismatch)),
            "both ingresses must reject with PrecisionMismatch:\n{source}\n{diagnostics:?}"
        );
    }
}

#[test]
fn a_deferred_cast_preserves_its_consumers_family_rejection() {
    for dtype in ["i32", "f32"] {
        check(
            &format!("out = 2.0f32 |> recip |> fn (v) -> cast(floor(v), {dtype}) |> recip\n"),
            dtype == "f32",
        );
    }
}

#[test]
fn each_family_operation_requires_a_sufficient_authored_bound() {
    for (operation, expression, bound) in [
        ("mean", "mean(x, 0i32)", "Float"),
        ("softmax", "softmax(x, 0i32)", "Float"),
        ("div", "div(x, x)", "Float"),
        ("trunc_div", "trunc_div(x, x)", "Int"),
        ("exp", "exp(x)", "Float"),
        ("log", "log(x)", "Float"),
        ("sin", "sin(x)", "Float"),
        ("cos", "cos(x)", "Float"),
        ("layer_norm", "layer_norm(x, x, x, 0.01f32)", "Float"),
        ("tan", "tan(x)", "Float"),
        ("atan", "atan(x)", "Float"),
        ("sqrt", "sqrt(x)", "Float"),
        ("relu", "relu(x)", "Float"),
        ("sigmoid", "sigmoid(x)", "Float"),
        ("tanh", "tanh(x)", "Float"),
        ("silu", "silu(x)", "Float"),
        ("gelu", "gelu(x)", "Float"),
        ("recip", "recip(x)", "Float"),
    ] {
        let result = if operation == "mean" {
            "tensor[p]"
        } else {
            "tensor[3, p]"
        };
        for binder in [
            "p".to_string(),
            "p: Numeric".to_string(),
            format!("p: {bound}"),
        ] {
            check(
                &format!("def g[{binder}](x: tensor[3, p]) -> {result} = {expression}\n"),
                binder == format!("p: {bound}"),
            );
        }
    }
}

#[test]
fn standalone_signatures_express_the_same_contract() {
    check(
        "sig g[p]: tensor[3, p] -> tensor[p]\ndef g(x) = mean(x, 0i32)\n",
        false,
    );
    check(
        "sig g[p: Float]: tensor[3, p] -> tensor[p]\ndef g(x) = mean(x, 0i32)\n",
        true,
    );
}

#[test]
fn a_wrapper_cannot_silently_narrow_its_authored_contract() {
    let helper = "def g[p: Float](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n";
    check(
        &format!("{helper}def mid[q](x: tensor[3, q]) -> tensor[q] = g(x)\n"),
        false,
    );
    check(
        &format!("{helper}def mid[q: Numeric](x: tensor[3, q]) -> tensor[q] = g(x)\n"),
        false,
    );
    check(
        &format!("{helper}def mid[q: Float](x: tensor[3, q]) -> tensor[q] = g(x)\n"),
        true,
    );
}

#[test]
fn bounded_function_values_keep_their_restriction() {
    let helper = "def g[p: Float](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n";
    for dtype in ["i32", "f32"] {
        check(
            &format!(
                "{helper}def apply_it(f: tensor[3, {dtype}] -> tensor[{dtype}], x: tensor[3, {dtype}]) -> tensor[{dtype}] = f(x)\ndef use_it(x: tensor[3, {dtype}]) -> tensor[{dtype}] = apply_it(g, x)\n"
            ),
            dtype == "f32",
        );
    }
}

#[test]
fn local_lambda_and_aggregate_projection_check_the_declared_precision() {
    for bound in ["p", "p: Float"] {
        check(
            &format!(
                "def g[{bound}](x: tensor[3, p]) -> tensor[p] = {{\n h = fn (t) -> mean(t, 0i32)\n h(x)\n}}\n"
            ),
            bound == "p: Float",
        );
    }
    for dtype in ["i32", "f32"] {
        check(
            &format!(
                "type Box =\n | Box {{ t: tensor[3, {dtype}] }}\ndef g[p: Float](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\ndef pick(b: Box) -> tensor[{dtype}] = g(b.t)\n"
            ),
            dtype == "f32",
        );
    }
}

#[test]
fn inferred_local_operands_still_bind_before_the_declaration_boundary() {
    check(
        "def f(x: tensor[3, f32]) -> tensor[f32] = {\n h = fn (t) -> mean(t, 0i32)\n h(x)\n}\n",
        true,
    );
    check(
        "def f(x: tensor[3, i32]) -> tensor[i32] = {\n h = fn (t) -> mean(t, 0i32)\n h(x)\n}\n",
        false,
    );
}

#[test]
fn primitive_scalar_contracts_require_bounds_too() {
    check("def g[p](x: p) -> p = sin(x)\n", false);
    check("def g[p: Numeric](x: p) -> p = sin(x)\n", false);
    check("def g[p: Float](x: p) -> p = sin(x)\n", true);
}

#[test]
fn numeric_and_integer_operation_contracts_cover_each_spec_family() {
    // [05-OP-46], [05-OP-40], [05-OP-36], [05-OP-30]/[05-OP-12..16].
    for (operations, args, result) in [
        ("abs floor ceil round", "x", "tensor[3, p]"),
        ("max_elem min_elem", "x, x", "tensor[3, p]"),
        ("cmplt lt gt gte lte", "x, x", "tensor[3, bool]"),
        ("max_reduce min_reduce prod_reduce", "x, 0i32", "tensor[p]"),
        ("argmax_reduce argmin_reduce", "x, 0i32", "tensor[i64]"),
    ] {
        for operation in operations.split_whitespace() {
            for binder in ["p", "p: Numeric"] {
                for alias in [false, true] {
                    let prefix = if alias {
                        format!("op = {operation}\n")
                    } else {
                        String::new()
                    };
                    let callee = if alias { "op" } else { operation };
                    check(
                        &format!(
                            "{prefix}def g[{binder}](x: tensor[3, p]) -> {result} = {callee}({args})\n"
                        ),
                        binder == "p: Numeric",
                    );
                }
            }
        }
    }
    // [05-OP-30] with spec/04 §5.7.1: `sum` returns sum_result(p, default(p)),
    // which is one type across `Float` but not across `Numeric`, whose i8
    // member sums to i32 (#3009). Only the direct call is covered: an aliased
    // reduction's result is not checked by its arm at all.
    check(
        "def g[p: Float](x: tensor[3, p]) -> tensor[p] = sum(x, 0i32)\n",
        true,
    );
    for binder in ["p", "p: Numeric", "p: Int"] {
        let source = format!("def g[{binder}](x: tensor[3, p]) -> tensor[p] = sum(x, 0i32)\n");
        let program = desugar_program(&parse_str(&source).expect("valid source"))
            .expect("Surf fixture must desugar");
        assert!(check_typed_program(&program).is_err(), "{source}");
    }
    // [05-OP-64], [05-OP-47]. Scalar controls avoid claiming a repair of
    // the separate pre-existing bounded-tensor integer validator limitation.
    for operation in ["mod", "bitand", "bitor", "bitxor", "shl", "shr"] {
        for binder in ["p", "p: Numeric", "p: Int"] {
            for alias in [false, true] {
                let prefix = if alias {
                    format!("op = {operation}\n")
                } else {
                    String::new()
                };
                let callee = if alias { "op" } else { operation };
                check(
                    &format!("{prefix}def g[{binder}](x: p) -> p = {callee}(x, x)\n"),
                    binder == "p: Int",
                );
            }
        }
        check(&format!("def g(x) = {operation}(x, x)\n"), false);
    }
    check(
        "def less[p](x: p, y: p) -> bool = lt(x, y)\nout = less(true, false)\n",
        false,
    );
    check(
        "def less[p: Numeric](x: p, y: p) -> bool = lt(x, y)\nout = less(1i32, 2i32)\n",
        true,
    );
}

#[test]
fn window_reduction_contracts_cover_each_spec_family_on_both_ingresses() {
    // [05-OP-39]: mean is Float; sum/max/min are Numeric. Direct calls use
    // the specialized shape path, while aliases use ordinary checked
    // function-value application. Both must consume the same scheme contract.
    for (operation, binders, invalid_dtype, valid_dtype) in [
        (
            "reduce_window_mean",
            &[("p", false), ("p: Numeric", false), ("p: Float", true)][..],
            "i32",
            "f32",
        ),
        (
            "reduce_window_sum",
            &[("p", false), ("p: Numeric", true)][..],
            "bool",
            "i32",
        ),
        (
            "reduce_window_max",
            &[("p", false), ("p: Numeric", true)][..],
            "bool",
            "i32",
        ),
        (
            "reduce_window_min",
            &[("p", false), ("p: Numeric", true)][..],
            "bool",
            "i32",
        ),
    ] {
        for &(binder, accepted) in binders {
            for alias in [false, true] {
                let prefix = if alias {
                    format!("window_op = {operation}\n")
                } else {
                    String::new()
                };
                let callee = if alias { "window_op" } else { operation };
                check_both_ingresses(
                    &format!(
                        "{prefix}def g[{binder}](x: tensor[3, p]) -> tensor[2, p] = \
                         {callee}(x, [2i64], [1i64])\n"
                    ),
                    accepted,
                );
            }
        }
        for (dtype, accepted) in [(invalid_dtype, false), (valid_dtype, true)] {
            for alias in [false, true] {
                let prefix = if alias {
                    format!("window_op = {operation}\n")
                } else {
                    String::new()
                };
                let callee = if alias { "window_op" } else { operation };
                check_both_ingresses(
                    &format!(
                        "{prefix}def g(x: tensor[3, {dtype}]) -> tensor[2, {dtype}] = \
                         {callee}(x, [2i64], [1i64])\n"
                    ),
                    accepted,
                );
            }
        }
    }
}

#[test]
fn window_shape_diagnostics_precede_family_admission_on_both_ingresses() {
    for (source, expected, got) in [
        (
            "def g(x: tensor[3, bool]) -> tensor[2, bool] = \
             reduce_window_sum(x, [0i64], [1i64])\n",
            "window extent >= 1",
            "0",
        ),
        (
            "def g(x: tensor[3, bool]) -> tensor[2, bool] = \
             reduce_window_sum(x, [2i64], [0i64])\n",
            "stride >= 1",
            "0",
        ),
        (
            "def g(x: tensor[3, bool]) -> tensor[2, bool] = \
             reduce_window_sum(x, [1i64, 1i64], [1i64, 1i64])\n",
            "window arity at most 1",
            "window arity 2",
        ),
    ] {
        let diagnostics = both_ingress_diagnostics(source);
        assert!(
            diagnostics.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::DimensionMismatch)
                    && error.expected.as_deref() == Some(expected)
                    && error.got.as_deref() == Some(got)
                    && error.span_offset == source.find("reduce_window_sum(")
            }),
            "the specialized window diagnostic must remain authoritative:\n\
             {source}\n{diagnostics:?}"
        );
        assert!(
            diagnostics
                .iter()
                .all(|error| !matches!(error.kind, CheckErrorKind::PrecisionMismatch)),
            "family admission must not hide a prior window error:\n{source}\n{diagnostics:?}"
        );
    }
}

#[test]
fn deferred_window_shape_diagnostics_precede_late_family_rejection_on_both_ingresses() {
    let source = |dtype: &str, windows: &str| {
        format!(
            "def use(x: tensor[3, {dtype}]) -> tensor[2, {dtype}] = {{\n\
             windowed = fn (value) -> \
             reduce_window_sum(value, {windows}, {windows})\n\
             windowed(x)\n\
             }}\n"
        )
    };

    for dtype in ["bool", "i32"] {
        let invalid_rank = source(dtype, "[1i64, 1i64]");
        let diagnostics = both_ingress_diagnostics(&invalid_rank);
        assert!(
            diagnostics.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::DimensionMismatch)
                    && error.expected.as_deref() == Some("window arity at most 1")
                    && error.got.as_deref() == Some("window arity 2")
                    && error.span_offset == invalid_rank.find("reduce_window_sum(")
            }),
            "the deferred window diagnostic must remain authoritative:\n\
             {invalid_rank}\n{diagnostics:?}"
        );
        assert!(
            diagnostics
                .iter()
                .all(|error| !matches!(error.kind, CheckErrorKind::PrecisionMismatch)),
            "late family admission must not hide a deferred window error:\n\
             {invalid_rank}\n{diagnostics:?}"
        );
    }

    let valid_rank = source("bool", "[1i64]");
    let diagnostics = both_ingress_diagnostics(&valid_rank);
    assert!(
        diagnostics
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::PrecisionMismatch)),
        "a shape-valid bool operand must still fail Numeric admission:\n\
         {valid_rank}\n{diagnostics:?}"
    );
}

#[test]
fn a_lexically_shadowed_window_builtin_keeps_its_function_contract() {
    check_both_ingresses(
        "def apply(\
           reduce_window_sum: &tensor[3, bool] -> List[i64] -> List[i64] -> tensor[2, bool], \
           x: tensor[3, bool]) -> tensor[2, bool] = \
           reduce_window_sum(x, [2i64], [1i64])\n",
        true,
    );
}

// [04-INF-9], proposed in #2074: missing annotations are inference holes,
// not permission to publish a newly inferred constrained generic contract.
#[test]
fn omitted_signatures_do_not_publish_inferred_dtype_contracts() {
    let factory = "def source[p: Float]() -> p = cast(0.0f32, p)\n";
    check(&format!("{factory}def make() -> f32 = source()\n"), true);
    check(
        &format!("{factory}def make[q: Float]() -> q = source()\n"),
        true,
    );
    check(&format!("{factory}def make() = source()\n"), false);
    check(&format!("{factory}def make() = (source(), 1i32)\n"), false);
    check("def source() = sin\ndef make() = source()\n", true);
    check(
        "type Box[a] =\n | Box { value: a }\ndef source() = Box { value: sin }\ndef make() = source()\n",
        true,
    );
    check("def g() -> tensor[0, f32] = sin(to_tensor([]))\n", true);
    check("def g() = sin(to_tensor([]))\n", false);
    for (expression, concrete) in [
        ("sin(x)", "f32"),
        ("add(x, x)", "i32"),
        ("trunc_div(x, x)", "i32"),
    ] {
        check(
            &format!("def g(x: {concrete}) -> {concrete} = {expression}\n"),
            true,
        );
        check(&format!("def g(x) = {expression}\n"), false);
    }
}

#[test]
fn later_callers_do_not_supply_an_omitted_generic_contract() {
    check("def g(x: f32) -> f32 = sin(x)\nout = g(1.0f32)\n", true);
    check("def g(x) = sin(x)\nout = g(1.0f32)\n", false);
}

#[test]
fn unannotated_wrappers_do_not_acquire_their_callees_bounds() {
    let helper = "def g[p: Float](x: p) -> p = sin(x)\n";
    check(
        &format!("{helper}def wrap[q: Float](x: q) -> q = g(x)\n"),
        true,
    );
    check(&format!("{helper}def wrap(x) = g(x)\n"), false);
}

#[test]
fn escaping_lambdas_need_sufficient_authored_admission() {
    for body in [
        "fn (x) -> sin(x)",
        "{\n h = fn (x) -> sin(x)\n h\n}",
        "(fn (x) -> sin(x), 1i32)",
    ] {
        check(
            &format!("def make() = {}\n", body.replace("fn (x)", "fn (x: f32)")),
            true,
        );
        check(&format!("def make() = {body}\n"), false);
    }
}

#[test]
fn inferred_family_lambdas_bind_monomorphically_within_the_declaration() {
    check(
        "def use() -> f32 = {\n h = fn (x) -> sin(x)\n h(1.0f32)\n}\n",
        true,
    );
    check(
        "def use() -> i32 = {\n h = fn (x) -> sin(x)\n h(1i32)\n}\n",
        false,
    );
    check(
        "def use() -> f64 = {\n h = fn (x) -> sin(x)\n _ = h(1.0f32)\n h(1.0f64)\n}\n",
        false,
    );
}

#[test]
fn contract_transport_and_unconstrained_inference_remain_polymorphic() {
    check("def make() = sin\n", true);
    check(
        "def apply[p: Float](x: p) -> p = {\n h = fn (t) -> sin(t)\n h(x)\n}\n",
        true,
    );
    check(
        "def g[p: Float](x: p) -> p = sin(x)\ndef use() -> f32 = {\n h = fn (t) -> g(t)\n h(1.0f32)\n}\n",
        true,
    );
    check(
        "def identity(x) = x\na = identity(1i32)\nb = identity(true)\n",
        true,
    );
    check(
        "def use() -> f64 = {\n op = sin\n _ = op(1.0f32)\n op(1.0f64)\n}\n",
        true,
    );
    check("def use() -> i32 = {\n op = sin\n op(1i32)\n}\n", false);
}

#[test]
fn arithmetic_and_matmul_require_their_operand_families() {
    check(
        "def g[p](x: tensor[3, p]) -> tensor[3, p] = add(x, x)\n",
        false,
    );
    for bound in ["Float", "Int", "Numeric"] {
        check(
            &format!("def g[p: {bound}](x: tensor[3, p]) -> tensor[3, p] = add(x, x)\n"),
            true,
        );
    }
    check(
        "def g[p](x: tensor[2, 2, p]) -> tensor[2, 2, p] = matmul(x, x)\n",
        false,
    );
    check(
        "def g[p: Float](x: tensor[2, 2, p]) -> tensor[2, 2, p] = matmul(x, x)\n",
        true,
    );
}

#[test]
fn a_primitive_function_value_carries_its_admission_requirement() {
    for dtype in ["i32", "f32"] {
        check(
            &format!(
                "def apply_it(f: &tensor[3, {dtype}] -> i32 -> tensor[{dtype}], x: tensor[3, {dtype}]) -> tensor[{dtype}] = f(x, 0i32)\ndef g(x: tensor[3, {dtype}]) -> tensor[{dtype}] = apply_it(mean, x)\n"
            ),
            dtype == "f32",
        );
        check(
            &format!(
                "def g(x: tensor[3, {dtype}]) -> tensor[{dtype}] = {{\n op = mean\n op(x, 0i32)\n}}\n"
            ),
            dtype == "f32",
        );
    }
}

#[test]
fn empty_literal_precision_cannot_evade_the_operation_contract() {
    let program = desugar_program(
        &parse_str("def f() -> tensor[i32] = mean(to_tensor([]), 0i32)\n").unwrap(),
    )
    .expect("Surf fixture must desugar");
    let Err(report) = check_typed_program(&program) else {
        panic!("an empty reduction operand must not escape admission checking");
    };
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.message.contains("unresolved `mean` shape obligation"))
    );
    check(
        "def f() -> tensor[3, i32] = softmax(to_tensor([]), 0i32)\n",
        false,
    );
}

#[test]
fn every_family_primitive_alias_preserves_its_requirement() {
    for (operation, args, output) in [
        ("mean", "x, 0i32", "tensor[p]"),
        ("softmax", "x, 0i32", "tensor[3, p]"),
        ("div", "x, x", "tensor[3, p]"),
        ("exp", "x", "tensor[3, p]"),
        ("log", "x", "tensor[3, p]"),
        ("sin", "x", "tensor[3, p]"),
        ("cos", "x", "tensor[3, p]"),
        ("layer_norm", "x, x, x, 0.01f32", "tensor[3, p]"),
        ("tan", "x", "tensor[3, p]"),
        ("atan", "x", "tensor[3, p]"),
        ("sqrt", "x", "tensor[3, p]"),
        ("relu", "x", "tensor[3, p]"),
        ("sigmoid", "x", "tensor[3, p]"),
        ("tanh", "x", "tensor[3, p]"),
        ("silu", "x", "tensor[3, p]"),
        ("gelu", "x", "tensor[3, p]"),
        ("recip", "x", "tensor[3, p]"),
    ] {
        for binder in ["p", "p: Float"] {
            check(
                &format!(
                    "def g[{binder}](x: tensor[3, p]) -> {output} = {{\n op = {operation}\n op({args})\n}}\n"
                ),
                binder == "p: Float",
            );
        }
    }
    for binder in ["p", "p: Int"] {
        check(
            &format!(
                "def g[{binder}](x: tensor[3, p]) -> tensor[3, p] = {{\n op = trunc_div\n op(x, x)\n}}\n"
            ),
            binder == "p: Int",
        );
    }
}

#[test]
fn a_failed_family_application_does_not_also_claim_its_operand_was_never_bound() {
    let source = "def f() -> i32 = {\n a = fn(t) -> mean(t, 0i32)\n b = fn(u) -> matmul(u, u)\n _ = a(to_tensor([1i32, 2i32, 4i32]))\n 1i32\n}\n";
    let program = desugar_program(&parse_str(source).unwrap()).expect("Surf fixture must desugar");
    let Err(report) = check_typed_program(&program) else {
        panic!("invalid family");
    };
    assert!(
        report
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::PrecisionMismatch))
    );
    assert!(
        !report
            .errors
            .iter()
            .any(|error| error.message.contains("unresolved `mean`"))
    );
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.message.contains("unresolved `matmul`"))
    );
}

#[test]
fn inferred_family_requirements_intersect_without_narrowing_authored_contracts() {
    let helper = "def id_dt[p](x: tensor[3, p]) -> tensor[3, p] = x\n";
    for (dtype, accepts) in [("f32", true), ("i32", false)] {
        check(
            &format!(
                "{helper}def f(x: tensor[3, {dtype}]) -> tensor[{dtype}] = {{\n h = fn(t) -> mean(add(id_dt(t), id_dt(t)), 0i32)\n h(x)\n}}\n"
            ),
            accepts,
        );
    }
}
