//! Authored generic contracts must entail every operation's dtype requirements.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{check_typed_program, errors::CheckErrorKind};

fn check(source: &str, accepted: bool) {
    let program = desugar_program(&parse_str(source).expect("valid source"));
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

#[test]
fn a_deferred_cast_preserves_its_consumers_family_rejection() {
    for dtype in ["int32", "f32"] {
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
        "sig g: tensor[3, p] -> tensor[p]\ndef g(x) = mean(x, 0i32)\n",
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
    for dtype in ["int32", "f32"] {
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
    for dtype in ["int32", "f32"] {
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
        "def f(x: tensor[3, int32]) -> tensor[int32] = {\n h = fn (t) -> mean(t, 0i32)\n h(x)\n}\n",
        false,
    );
}

#[test]
fn primitive_scalar_contracts_require_bounds_too() {
    check("def g[p](x: p) -> p = sin(x)\n", false);
    check("def g[p: Numeric](x: p) -> p = sin(x)\n", false);
    check("def g[p: Float](x: p) -> p = sin(x)\n", true);
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
        ("add(x, x)", "int32"),
        ("trunc_div(x, x)", "int32"),
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
        "def use() -> int32 = {\n h = fn (x) -> sin(x)\n h(1i32)\n}\n",
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
    check("def use() -> int32 = {\n op = sin\n op(1i32)\n}\n", false);
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
    for dtype in ["int32", "f32"] {
        check(
            &format!(
                "def apply_it(f: &tensor[3, {dtype}] -> int32 -> tensor[{dtype}], x: tensor[3, {dtype}]) -> tensor[{dtype}] = f(x, 0i32)\ndef g(x: tensor[3, {dtype}]) -> tensor[{dtype}] = apply_it(mean, x)\n"
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
        &parse_str("def f() -> tensor[int32] = mean(to_tensor([]), 0i32)\n").unwrap(),
    );
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
        "def f() -> tensor[3, int32] = softmax(to_tensor([]), 0i32)\n",
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
    let source = "def f() -> int32 = {\n a = fn(t) -> mean(t, 0i32)\n b = fn(u) -> matmul(u, u)\n _ = a(to_tensor([1i32, 2i32, 4i32]))\n 1i32\n}\n";
    let program = desugar_program(&parse_str(source).unwrap());
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
    for (dtype, accepts) in [("f32", true), ("int32", false)] {
        check(
            &format!(
                "{helper}def f(x: tensor[3, {dtype}]) -> tensor[{dtype}] = {{\n h = fn(t) -> mean(add(id_dt(t), id_dt(t)), 0i32)\n h(x)\n}}\n"
            ),
            accepts,
        );
    }
}
