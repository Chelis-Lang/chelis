//! A macro call supplies exactly one positional argument per parameter and
//! no named argument, and a macro definition names each parameter once
//! (spec/02-surf-syntax.md [02-MACRO-1], [02-MACRO-2], [02-MACRO-3]), at both
//! the Surf `macro` and the internal Deep `defmacro` ingress.

use chelis_deep::DeepTag;
use chelis_deep::ast::{Atom, Expr, ExprCarrier, strip_metadata};
use chelis_deep::printer::print_canonical;
use chelis_macros::{ExpandedProgram, ExpansionError, ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;

fn desugar(source: &str) -> Vec<Expr> {
    desugar_program(&parse_str(source).expect("Surf parses")).expect("Surf desugars")
}

fn deep(source: &str) -> Vec<Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep fixture parses")
}

fn expand(program: &[Expr]) -> Result<ExpandedProgram, ExpansionError> {
    expand_program(program, &ExpansionOptions::default())
}

fn assert_arity_mismatch(
    result: Result<ExpandedProgram, ExpansionError>,
    macro_name: &str,
    parameters: usize,
    arguments: usize,
) {
    let error = match result {
        Ok(expanded) => panic!(
            "`{macro_name}` called with {arguments} argument(s) for {parameters} \
             parameter(s) must not expand; got:\n{}",
            print_canonical(expanded.exprs())
        ),
        Err(error) => error,
    };
    assert!(
        matches!(
            &error,
            ExpansionError::ArityMismatch { name, expected, found }
                if name == macro_name && *expected == parameters && *found == arguments
        ),
        "expected an arity mismatch for `{macro_name}` ({parameters} vs {arguments}); \
         got {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!("macro `{macro_name}`"))
            && message.contains(&format!("expects {parameters} argument(s)"))
            && message.contains(&format!("supplies {arguments}")),
        "the diagnostic must name the macro and both counts: {message}"
    );
}

fn assert_duplicate_parameter(
    result: Result<ExpandedProgram, ExpansionError>,
    macro_name: &str,
    repeated: &str,
) {
    let error = match result {
        Ok(expanded) => panic!(
            "`{macro_name}` repeats parameter `{repeated}` and must not expand; got:\n{}",
            print_canonical(expanded.exprs())
        ),
        Err(error) => error,
    };
    assert!(
        matches!(
            &error,
            ExpansionError::DuplicateParameter { name, parameter }
                if name == macro_name && parameter == repeated
        ),
        "expected a repeated-parameter rejection for `{macro_name}`; got {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!("macro `{macro_name}`"))
            && message.contains(&format!("parameter `{repeated}`")),
        "the diagnostic must name the macro and the repeated parameter: {message}"
    );
}

fn assert_named_argument(
    result: Result<ExpandedProgram, ExpansionError>,
    macro_name: &str,
    named: &str,
) {
    let error = match result {
        Ok(expanded) => panic!(
            "`{macro_name}` called with `{named}=` must not expand; got:\n{}",
            print_canonical(expanded.exprs())
        ),
        Err(error) => error,
    };
    assert!(
        matches!(
            &error,
            ExpansionError::NamedArgument { name, argument }
                if name == macro_name && *argument == named
        ),
        "expected a named-argument rejection for `{macro_name}`; got {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!("macro `{macro_name}`"))
            && message.contains(&format!("named argument `{named}=`")),
        "the diagnostic must name the macro and the named argument: {message}"
    );
}

/// The `accumulator` annotation of every call to `callee` in the expansion,
/// as canonical type syntax, or `none`.
fn accumulators_on_calls_to(exprs: &[Expr], callee: &str) -> Vec<String> {
    fn walk(expr: &Expr, callee: &str, out: &mut Vec<String>) {
        let ExprCarrier::DecodedNode(tag, metadata, children) = expr.carrier() else {
            return;
        };
        if tag == DeepTag::App
            && matches!(
                children.first().map(Expr::carrier),
                Some(ExprCarrier::DecodedNode(
                    DeepTag::Var,
                    _,
                    [Expr::Atom(Atom::Name(name), _)]
                )) if name == callee
            )
        {
            out.push(metadata.accumulator().map_or_else(
                || "none".to_string(),
                |accumulator| {
                    print_canonical(std::slice::from_ref(accumulator.expression()))
                        .trim_end()
                        .to_string()
                },
            ));
        }
        for child in children {
            walk(child, callee, out);
        }
    }
    let mut out = Vec::new();
    for expr in exprs {
        walk(expr, callee, &mut out);
    }
    out
}

/// Every top-level form of the expansion without metadata, so the `source`
/// provenance that records a call's arguments cannot satisfy a claim about
/// the substituted structure.
fn expanded_forms(program: &[Expr]) -> Vec<String> {
    expand(program)
        .expect("an exact-arity call expands")
        .exprs()
        .iter()
        .map(|form| {
            print_canonical(std::slice::from_ref(&strip_metadata(form)))
                .trim_end()
                .to_string()
        })
        .collect()
}

/// Without the rule, `y` stayed unsubstituted and read `f`'s own parameter.
#[test]
fn short_call_is_rejected_instead_of_capturing_a_caller_binding() {
    assert_arity_mismatch(
        expand(&desugar(
            "macro second(x, y) = y\ndef f(y: i32) -> i32 = second(1i32)\n",
        )),
        "second",
        2,
        1,
    );
}

/// Without the rule, the surplus argument vanished before name, effect, and
/// linearity checking could see it.
#[test]
fn long_call_is_rejected_instead_of_dropping_the_surplus_argument() {
    for surplus in ["never_declared", "print(\"boom\")", "k"] {
        let source = format!("macro keep(x) = x\ndef f(k: key) -> key = keep(k, {surplus})\n");
        assert_arity_mismatch(expand(&desugar(&source)), "keep", 1, 2);
    }
}

#[test]
fn zero_parameter_macro_takes_no_argument() {
    assert_arity_mismatch(
        expand(&desugar("macro seven() = 7i32\nout = seven(1i32)\n")),
        "seven",
        0,
        1,
    );
}

#[test]
fn standard_prelude_macro_calls_have_the_same_rule() {
    for (call, name, parameters, arguments) in [
        ("residual(x)", "residual", 2, 1),
        ("residual(x, g, x)", "residual", 2, 3),
        ("linear_layer(x, x)", "linear_layer", 3, 2),
        ("linear_layer(x, x, x, x)", "linear_layer", 3, 4),
        ("cross_entropy(x)", "cross_entropy", 2, 1),
    ] {
        let source = format!("def g(x: f32) -> f32 = x\ndef f(x: f32) -> f32 = {call}\n");
        assert_arity_mismatch(expand(&desugar(&source)), name, parameters, arguments);
    }
}

/// The count comes from the definition that resolution selects: a user macro
/// that overrides a prelude macro brings its own parameter list.
#[test]
fn a_user_override_of_a_prelude_macro_has_the_override_arity() {
    assert_eq!(
        expanded_forms(&desugar("macro residual(x) = x\nout = residual(1i32)\n")),
        ["(def {} out (lit {} 1))"]
    );
    assert_arity_mismatch(
        expand(&desugar(
            "macro residual(x) = x\nout = residual(1i32, 2i32)\n",
        )),
        "residual",
        1,
        2,
    );
}

#[test]
fn internal_deep_defmacro_calls_have_the_same_rule() {
    let definitions = "(defmacro {} second (params {} x y) (var {} y))\n";
    assert_arity_mismatch(
        expand(&deep(&format!(
            "{definitions}(def {{}} out (app {{}} (var {{}} second) (lit {{}} 1)))"
        ))),
        "second",
        2,
        1,
    );
    assert_arity_mismatch(
        expand(&deep(&format!(
            "{definitions}(def {{}} out (app {{}} (var {{}} second) (lit {{}} 1) \
             (lit {{}} 2) (var {{}} never_declared)))"
        ))),
        "second",
        2,
        3,
    );
    assert_eq!(
        expanded_forms(&deep(&format!(
            "{definitions}(def {{}} out (app {{}} (var {{}} second) (lit {{}} 1) (lit {{}} 2)))"
        ))),
        ["(def {} out (lit {} 2))"]
    );
}

/// A call that an expansion produces, from a macro body or inside an
/// argument, is expanded by the same step and meets the same rule.
#[test]
fn calls_produced_by_an_expansion_have_the_same_rule() {
    assert_arity_mismatch(
        expand(&desugar(
            "macro inner(x, y) = y\nmacro outer(a) = inner(a)\ndef f(y: i32) -> i32 = outer(1i32)\n",
        )),
        "inner",
        2,
        1,
    );
    assert_arity_mismatch(
        expand(&desugar(
            "macro keep(x) = x\nout = keep(keep(1i32, 2i32))\n",
        )),
        "keep",
        1,
        2,
    );
}

#[test]
fn module_scoped_macro_calls_have_the_same_rule() {
    assert_arity_mismatch(
        expand(&desugar(
            "module Demo.Main\nmacro keep(x) = x\nout = keep(7i32, never_declared)\n",
        )),
        "keep",
        1,
        2,
    );
}

/// Without the rule, the later binding of the name won and the first
/// argument was dropped. The definition is rejected even when nothing calls it.
#[test]
fn a_repeated_parameter_name_is_rejected_at_the_definition() {
    assert_duplicate_parameter(
        expand(&desugar("macro dup(x, x) = x\nout = dup(1i32, 2i32)\n")),
        "dup",
        "x",
    );
    assert_duplicate_parameter(
        expand(&desugar("macro dup(a, b, a) = b\nout = 1i32\n")),
        "dup",
        "a",
    );
    assert_duplicate_parameter(
        expand(&deep(
            "(defmacro {} dup (params {} x y x) (var {} y))\n(def {} out (lit {} 1))",
        )),
        "dup",
        "x",
    );
}

#[test]
fn exact_arity_calls_substitute_every_argument() {
    assert_eq!(
        expanded_forms(&desugar(
            "macro second(x, y) = y\nout = second(1i32, 9i32)\n"
        )),
        ["(def {} out (lit {} 9))"]
    );
    assert_eq!(
        expanded_forms(&desugar("macro seven() = 7i32\nout = seven()\n")),
        ["(def {} out (lit {} 7))"]
    );
    assert_eq!(
        expanded_forms(&desugar(
            "macro inner(x, y) = y\nmacro outer(a) = inner(a, a)\nout = outer(3i32)\n"
        )),
        ["(def {} out (lit {} 3))"]
    );
    let prelude = expanded_forms(&desugar(
        "def g(x: f32) -> f32 = x\ndef f(x: f32) -> f32 = residual(x, g)\n",
    ));
    assert!(
        prelude
            .iter()
            .any(|form| form
                .contains("(app {} (var {} add) (var {} x) (app {} (var {} g) (var {} x)))")),
        "the prelude call must substitute both arguments: {prelude:#?}"
    );
}

/// Without the rule, the call's `accumulator=` was dropped with the replaced
/// invocation node, so `total(x, accumulator=f64)` summed in `f32`.
#[test]
fn a_named_argument_on_a_macro_call_is_rejected() {
    for (source, name) in [
        (
            "macro total(x) = sum(x, 0i32)\n\
             def f(x: tensor[3, f32]) -> f32 = total(x, accumulator=f64)\n",
            "total",
        ),
        (
            "macro keep(x) = x\nout = keep(7i32, accumulator=f64)\n",
            "keep",
        ),
        (
            "macro keep(x) = x\nout = 7i32 |> keep(accumulator=f64)\n",
            "keep",
        ),
        (
            "out = residual(1i32, fn (a: i32) -> a, accumulator=f64)\n",
            "residual",
        ),
        (
            "macro keep(x) = x\nmacro outer(a) = keep(a, accumulator=f64)\nout = outer(7i32)\n",
            "keep",
        ),
    ] {
        assert_named_argument(expand(&desugar(source)), name, "accumulator");
    }
    assert_named_argument(
        expand(&deep(
            "(defmacro {} keep (params {} x) (var {} x))\n\
             (def {} out (app {accumulator: (t-prim {} f64)} (var {} keep) (lit {} 7)))",
        )),
        "keep",
        "accumulator",
    );
}

/// A named argument that the body writes on a built-in call belongs to that
/// call, so expansion keeps it on the expanded call.
#[test]
fn a_named_argument_inside_a_macro_body_survives_expansion() {
    for (source, callee) in [
        (
            "macro total(x) = sum(x, 0i32, accumulator=f64)\n\
             def f(x: tensor[3, f32]) -> f32 = total(x)\n",
            "sum",
        ),
        (
            "macro mm(a, b) = matmul(a, b, accumulator=f64)\n\
             def f(a: tensor[1, 3, f32], b: tensor[3, 1, f32]) -> tensor[1, 1, f32] = mm(a, b)\n",
            "matmul",
        ),
    ] {
        let expanded = expand(&desugar(source)).expect("the body's named argument expands");
        assert_eq!(
            accumulators_on_calls_to(expanded.exprs(), callee),
            ["(t-prim {} f64)"],
            "the expanded `{callee}` call must keep its accumulator"
        );
    }
}
