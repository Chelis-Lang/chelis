//! Every parameter of a macro definition occurs in its body as a reference
//! that substitution replaces, so expansion never discards an argument
//! (spec/02-surf-syntax.md [02-MACRO-4]). The rule holds at the Surf `macro`
//! and the internal Deep `defmacro` ingress, whether or not the macro is called.

use chelis_deep::ast::{Expr, strip_metadata};
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

fn assert_unused_parameter(
    result: Result<ExpandedProgram, ExpansionError>,
    macro_name: &str,
    unused: &str,
) {
    let error = match result {
        Ok(expanded) => panic!(
            "`{macro_name}` never references `{unused}` and must not expand; got:\n{}",
            print_canonical(expanded.exprs())
        ),
        Err(error) => error,
    };
    assert!(
        matches!(
            &error,
            ExpansionError::UnusedParameter { name, parameter }
                if name == macro_name && parameter == unused
        ),
        "expected an unused-parameter rejection for `{macro_name}`; got {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!("macro `{macro_name}`"))
            && message.contains(&format!("parameter `{unused}`"))
            && message.contains("[02-MACRO-4]"),
        "the diagnostic must name the macro, the unused parameter, and the rule: {message}"
    );
}

/// Every top-level form of the expansion without metadata, so the `source`
/// provenance that records a call's arguments cannot satisfy a claim about
/// the substituted structure.
fn expanded_forms(program: &[Expr]) -> Vec<String> {
    expand(program)
        .expect("a definition that references every parameter expands")
        .exprs()
        .iter()
        .map(|form| {
            print_canonical(std::slice::from_ref(&strip_metadata(form)))
                .trim_end()
                .to_string()
        })
        .collect()
}

/// Without the rule, the argument for `b` vanished with the replaced call:
/// an undefined name, a type error, and an effect that never ran all checked
/// clean, and a wrong-arity macro call inside it was never expanded.
#[test]
fn a_call_cannot_hide_an_argument_in_an_unused_parameter() {
    for discarded in [
        "never_declared",
        "add(1i32, true)",
        "print(\"boom\")",
        "debug(2i32)",
        "keep(1i32, 2i32)",
    ] {
        let source =
            format!("macro keep(x) = x\nmacro first(a, b) = a\nout = first(1i32, {discarded})\n");
        assert_unused_parameter(expand(&desugar(&source)), "first", "b");
    }
}

/// A second consuming use of an affine key could hide in the unused position.
#[test]
fn a_reused_key_cannot_hide_in_an_unused_parameter() {
    assert_unused_parameter(
        expand(&desugar(
            "macro first(a, b) = a\ndef f(k: key) -> key = first(k, k)\n",
        )),
        "first",
        "b",
    );
}

#[test]
fn the_definition_is_rejected_whether_or_not_it_is_called() {
    assert_unused_parameter(
        expand(&desugar("macro first(a, b) = a\nout = 1i32\n")),
        "first",
        "b",
    );
    assert_unused_parameter(
        expand(&desugar("macro ignore(x) = 0i32\nout = 1i32\n")),
        "ignore",
        "x",
    );
    assert_unused_parameter(
        expand(&desugar(
            "module Demo.Main\nmacro first(a, b) = a\nout = 1i32\n",
        )),
        "first",
        "b",
    );
}

#[test]
fn internal_deep_defmacro_definitions_have_the_same_rule() {
    assert_unused_parameter(
        expand(&deep(
            "(defmacro {} first (params {} a b) (var {} a))\n(def {} out (lit {} 1))",
        )),
        "first",
        "b",
    );
}

/// A reference that a body binder of the same name shadows is not replaced
/// by substitution, so it does not use the parameter.
#[test]
fn a_shadowed_reference_does_not_use_the_parameter() {
    assert_unused_parameter(
        expand(&desugar("macro lam(x) = fn (x: i32) -> x\nout = 1i32\n")),
        "lam",
        "x",
    );
    assert_unused_parameter(
        expand(&desugar(
            "macro rebind(x) = {\n  x = 1i32\n  x\n}\nout = 1i32\n",
        )),
        "rebind",
        "x",
    );
    assert_unused_parameter(
        expand(&deep(
            "(defmacro {} arm_bound (params {} x) \
               (match {} (lit {} 1) (arm {} (pat-var {} x) () (var {} x))))\n\
             (def {} out (lit {} 1))",
        )),
        "arm_bound",
        "x",
    );
}

/// The first unused parameter in declaration order is reported, and a
/// repeated parameter is reported before an unused one ([02-MACRO-1]).
#[test]
fn the_first_unused_parameter_is_named() {
    assert_unused_parameter(
        expand(&desugar("macro last(a, b, c) = c\nout = 1i32\n")),
        "last",
        "a",
    );
    assert!(
        matches!(
            expand(&desugar("macro dup(a, b, a) = b\nout = 1i32\n")),
            Err(ExpansionError::DuplicateParameter { .. })
        ),
        "a repeated parameter is rejected by [02-MACRO-1] first"
    );
}

/// With every parameter referenced, every argument occurs in the expansion,
/// so a wrong-arity call inside an argument is expanded and rejected.
#[test]
fn a_wrong_arity_call_in_an_argument_is_expanded_and_rejected() {
    let error = expand(&desugar(
        "macro keep(x) = x\nmacro both(a, b) = add(a, b)\nout = both(1i32, keep(1i32, 2i32))\n",
    ))
    .expect_err("the argument is expanded, so its wrong-arity call is rejected");
    assert!(
        matches!(
            &error,
            ExpansionError::ArityMismatch { name, expected: 1, found: 2 } if name == "keep"
        ),
        "expected the [02-MACRO-2] arity diagnostic for `keep`; got {error:?}"
    );
}

#[test]
fn definitions_that_reference_every_parameter_expand() {
    assert_eq!(
        expanded_forms(&desugar(
            "macro both(a, b) = add(a, b)\nout = both(1i32, 2i32)\n"
        )),
        ["(def {} out (app {} (var {} add) (lit {} 1) (lit {} 2)))"]
    );
    assert_eq!(
        expanded_forms(&desugar(
            "macro double(x) = add(x, x)\nout = double(3i32)\n"
        )),
        ["(def {} out (app {} (var {} add) (lit {} 3) (lit {} 3)))"]
    );
    assert_eq!(
        expanded_forms(&desugar("macro seven() = 7i32\nout = seven()\n")),
        ["(def {} out (lit {} 7))"]
    );
    assert_eq!(
        expanded_forms(&desugar(
            "macro apply(f, x) = f(x)\ndef g(x: i32) -> i32 = x\nout = apply(g, 4i32)\n"
        ))
        .last()
        .map(String::as_str),
        Some("(def {} out (app {} (var {} g) (lit {} 4)))")
    );
    let lifted = expanded_forms(&deep(
        "(defmacro {} lift (params {} y) \
           (fn {} (params {} x) (app {} (var {} add) (var {} x) (var {} y))))\n\
         (def {} out (app {} (var {} lift) (lit {} 5)))",
    ));
    assert!(
        lifted[0].contains("(lit {} 5)"),
        "a reference inside a body `fn` whose binder differs uses the parameter: {lifted:#?}"
    );
}

#[test]
fn standard_prelude_macros_reference_every_parameter() {
    let prelude = expanded_forms(&desugar(
        "def g(x: f32) -> f32 = x\ndef f(x: f32) -> f32 = residual(x, g)\n",
    ));
    assert!(
        prelude
            .iter()
            .any(|form| form
                .contains("(app {} (var {} add) (var {} x) (app {} (var {} g) (var {} x)))")),
        "the prelude call substitutes both arguments: {prelude:#?}"
    );
}
