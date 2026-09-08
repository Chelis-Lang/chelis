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
    let surf = chelis_surf::parser::parse_str(source).expect("Surf parse");
    let deep = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&surf),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("expand")
    .into_exprs();
    deep_diagnostics(&deep)
}

fn deep_diagnostics(deep: &[chelis_deep::Expr]) -> Vec<CheckError> {
    let ir = check_ir_program(deep).err().map_or(vec![], |r| r.errors);
    let typed = check_typed_program(deep).err().map_or(vec![], |r| r.errors);
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
                "(defsig {{dtype_bounds: {{p: int}}}} f (t-fn {{}} (t-tensor {{}} (d-lit {{}} 3) (t-var {{}} p)) (t-tensor {{}} (d-lit {{}} 3) {result}))) (def {{}} f (fn {{}} (params {{}} xs) (app {{}} (var {{}} {op}) {lhs} {rhs})))"
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
                ("xs", "cast(1, p)"),
                ("cast(1, p)", "xs"),
                ("xs", "c"),
                ("c", "xs"),
            ] {
                rejects(
                    &format!(
                        "def f[p: {bound}](xs: tensor[3, p], c: p) -> tensor[3, bool] = {op}({lhs}, {rhs})\n"
                    ),
                    true,
                );
            }
            accepts(&format!(
                "def f[p: {bound}](x: p, y: p) -> bool = {op}(x, y)\n"
            ));
            accepts(&format!(
                "def f[p: {bound}](x: tensor[3, p], y: tensor[3, p]) -> tensor[3, bool] = {op}(x, y)\n"
            ));
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
            ("xs", "cast(1, p)"),
            ("cast(1, p)", "xs"),
            ("xs", "c"),
            ("c", "xs"),
        ] {
            rejects(
                &format!(
                    "def f[p: {bound}](xs: tensor[3, p], c: p) -> tensor[3, p] = {op}({lhs}, {rhs})\n"
                ),
                false,
            );
        }
        accepts(&format!(
            "def f[p: {bound}](x: p, y: p) -> p = {op}(x, y)\n"
        ));
        accepts(&format!(
            "def f[p: {bound}](x: tensor[3, p], y: tensor[3, p]) -> tensor[3, p] = {op}(x, y)\n"
        ));
    }
}

#[test]
fn nesting_and_aliases_do_not_erase_the_scalar_surface() {
    for body in [
        "{ c = cast(0.1, p)\n gt(xs, c) }",
        "{ c: p = cast(0.1, p)\n gt(xs, c) }",
        "{ compare = fn(c: p) -> gt(xs, c)\n compare(cast(0.1, p)) }",
        "where(gt(xs, cast(0.1, p)), xs, xs)",
        "gt(xs, tensor_to_scalar(scalar_to_tensor(cast(0.1, p))))",
    ] {
        rejects(
            &format!("def f[p: Float](xs: tensor[3, p]) = {body}\n"),
            true,
        );
    }
}

#[test]
fn explicit_shape_construction_preserves_generic_dtype() {
    for size in [0, 1, 3] {
        for op in ["gt", "mul"] {
            for extent in [format!("{size}i64"), "shape(xs, 0i32)".into()] {
                accepts(&format!(
                    "def f[p: Float](xs: tensor[{size}, p]) = {op}(xs, insert(scalar_to_tensor(cast(0.1, p)), 0i32, {extent}))\n"
                ));
            }
        }
    }
    accepts(
        "def f[p: Float](xs: tensor[2, 3, p]) = gt(xs, insert(insert(scalar_to_tensor(cast(0.1, p)), 0i32, 3i64), 0i32, 2i64))\n",
    );
    accepts("def f[p: Float](x: p) -> p = tensor_to_scalar(scalar_to_tensor(x))\n");
    accepts("def f[p: Float](x: p) -> p = { y: p = x\n y }\n");
    rejects("def f[p: Float](x: p) = gt(scalar_to_tensor(x), x)\n", true);
}

#[test]
fn declaration_binders_stay_independent_and_rigid() {
    accepts(
        "def f[p: Float](x: p) -> p = mul(x, cast(0.1, p))\ndef g[p: Int](x: p) -> p = add(x, cast(1, p))\na = f(3.0f64)\nb = g(9007199254740993i64)\n",
    );
    assert!(!diagnostics("def f[p: Float](x: p) -> p = cast(1, p)\na = f(1i64)\n").is_empty());
    assert!(!diagnostics("def f[p: Float](x: p) -> p = add(x, 1.0f32)\n").is_empty());
    accepts("def identity[p](x: p) -> p = x\nx = identity(to_tensor([1.0f32]))\n");
}

#[test]
fn cached_library_bounds_and_lexical_aliases_preserve_the_boundary() {
    let library = chelis_surf::desugar::desugar_program(
        &chelis_surf::parser::parse_str("def id[p: Float](x: p) -> p = x\n").unwrap(),
    );
    let context = chelis_types::build_type_env_from_library(&library).unwrap();
    let bytes = bincode::serialize(&context).unwrap();
    let context: chelis_types::TypeEnv = bincode::deserialize(&bytes).unwrap();
    for (operand, mixed) in [("xs", true), ("c", false)] {
        let source = format!("def f[p: Float](xs: tensor[3, p], c: p) = gt({operand}, id(c))\n");
        let deep = chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(&source).unwrap(),
        );
        let errors = chelis_types::check_ir_with_context(&context, &deep)
            .err()
            .map_or(vec![], |r| r.errors);
        if mixed {
            assert_mixed(&errors, true, &source);
        } else {
            assert!(errors.is_empty(), "{errors:?}");
        }
    }
    rejects(
        "def f[p: Float](xs: &tensor[3, p], c: &p) = gt(xs, c)\n",
        true,
    );
    accepts("def f[p: Float](x: &p, y: &p) -> bool = gt(x, y)\n");
    accepts(
        "def f[p: Float](xs: tensor[3, p], c: p) -> p = { gt = fn(a: tensor[3, p], b: p) -> b\n gt(xs, c) }\n",
    );
}
