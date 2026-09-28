//! spec/06 sections 2.1-2.2: all differentiable parameters contribute, using
//! their final inferred types. A type variable is not a discrete parameter.

use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::check_typed_program;

fn check(source: &str, accepted: bool) {
    let parsed = parse_str(source).expect("fixture parses");
    let deep = desugar_program(&parsed).expect("fixture desugars");
    let result = check_typed_program(&deep);
    assert_eq!(result.is_ok(), accepted, "{source}\n{result:?}");
}

#[test]
fn inferred_float_parameter_contributes_to_the_gradient_tuple() {
    check(
        "def main() -> (f32, f32) = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, 1.0f32)",
        true,
    );
    check(
        "def main() -> f32 = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, 1.0f32)",
        false,
    );
}

#[test]
fn inferred_discrete_parameter_is_excluded() {
    check(
        "def main() -> f32 = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, true)",
        true,
    );
    check(
        "def main() -> (f32, f32) = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, true)",
        false,
    );
}

#[test]
fn selected_inferred_parameter_requires_a_float_leaf() {
    check(
        "def main() -> f32 = grad(fn (z: f32, w) -> mul(z, z), wrt=w)(3.0f32, 1.0f32)",
        true,
    );
    check(
        "def main() -> f32 = grad(fn (z: f32, w) -> mul(z, z), wrt=w)(3.0f32, true)",
        false,
    );
}

#[test]
fn inferred_nested_parameter_keeps_discrete_cotangent_positions() {
    check(
        "def main() -> (f32, (f32, unit)) = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, (1.0f32, true))",
        true,
    );
    check(
        "def main() -> (f32, (unit, unit)) = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, (1.0f32, true))",
        false,
    );
}

#[test]
fn let_bound_gradient_waits_for_its_application() {
    check(
        "def main() -> (f32, f32) = {\n g = grad(fn (z: f32, w) -> mul(z, z))\n g(3.0f32, 1.0f32)\n}",
        true,
    );
    check(
        "def main() -> f32 = {\n g = grad(fn (z: f32, w) -> mul(z, z))\n g(3.0f32, 1.0f32)\n}",
        false,
    );
}

#[test]
fn an_unresolved_parameter_cannot_publish_a_gradient_shape() {
    check("def f() = grad(fn (z: f32, w) -> mul(z, z))", false);
    check("def f() = grad(fn (z: f32, w: bool) -> mul(z, z))", true);
}

#[test]
fn deferred_gradient_stays_monomorphic_after_its_first_application() {
    check(
        "def main() -> (f32, f32) = {\n g = grad(fn (z: f32, w) -> mul(z, z))\n first = g(3.0f32, 1.0f32)\n g(4.0f32, 2.0f32)\n}",
        true,
    );
    check(
        "def main() -> f32 = {\n g = grad(fn (z: f32, w) -> mul(z, z))\n first = g(3.0f32, 1.0f32)\n g(4.0f32, true)\n}",
        false,
    );
}

#[test]
fn unselected_parameter_does_not_delay_the_gradient_result() {
    check("def f() = grad(fn (z: f32, w) -> mul(z, z), wrt=z)", true);
    check("def f() = grad(fn (z: f32, w) -> mul(z, z), wrt=w)", false);
}

#[test]
fn inferred_tensor_parameter_retains_its_shape_and_precision() {
    check(
        "def main() -> (f32, tensor[2, f64]) = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, to_tensor([1.0f64, 2.0f64]))",
        true,
    );
    check(
        "def main() -> (f32, tensor[2, f32]) = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, to_tensor([1.0f64, 2.0f64]))",
        false,
    );
}

#[test]
fn a_declaration_result_annotation_is_not_a_gradient_parameter_binding_site() {
    for source in [
        "def make_grad() -> (f32 -> f32) = grad(fn (w) -> 1.0f32)",
        "def make_grad() -> (f32 -> f32) = grad(fn (w) -> 1.0f32, wrt=w)",
        "def make_grad() -> (f32 -> f32) = {\n g = grad(fn (w) -> 1.0f32)\n g\n}",
        "def make_grad() -> (tensor[2, f32] -> tensor[2, f32]) = grad(fn (w) -> 1.0f32)",
    ] {
        check(source, false);
    }
    for source in [
        "def make_grad() -> (f32 -> f32) = grad(fn (w: f32) -> 1.0f32)",
        "def make_grad() -> (f32 -> f32) = {\n g = grad(fn (w) -> 1.0f32)\n zero = g(1.0f32)\n g\n}",
        "def make_grad() -> (tensor[2, f32] -> tensor[2, f32]) = {\n g = grad(fn (w) -> 1.0f32)\n zero = g(to_tensor([1.0f32, 2.0f32]))\n g\n}",
    ] {
        check(source, true);
    }
}

#[test]
fn a_local_result_ascription_waits_for_a_gradient_parameter_binding_site() {
    check(
        "def make_grad() = {\n g: (f32 -> f32) = grad(fn (w) -> 1.0f32)\n g\n}",
        false,
    );
    check(
        "def make_grad() = {\n g: (f32 -> f32) = grad(fn (w) -> 1.0f32)\n zero = g(1.0f32)\n g\n}",
        true,
    );
    check(
        "def make_grad() = {\n g: (f32 -> f32) = grad(fn (w) -> 1.0f32)\n zero = g(true)\n g\n}",
        false,
    );
    check(
        "def make_grad() = {\n g = grad(fn (w) -> 1.0f32)\n h: (f32 -> f32) = g\n h\n}",
        false,
    );
    check(
        "def make_grad() = {\n g = grad(fn (w) -> 1.0f32)\n h: (f32 -> f32) = g\n zero = h(1.0f32)\n h\n}",
        true,
    );
    check(
        "def make_grad() = {\n g = grad(fn (w) -> 1.0f32)\n unrelated: f32 = 2.0f32\n zero = g(1.0f32)\n g\n}",
        true,
    );
}

#[test]
fn deep_expression_ascription_is_not_a_gradient_parameter_binding_site() {
    for (parameter, accepted) in [("w", false), ("(w {type: (t-prim {} f32)})", true)] {
        let source = format!(
            "(def {{}} make_grad (fn {{}} (params {{}})
                (grad {{type: (t-fn {{}} (t-prim {{}} f32) (t-prim {{}} f32))}}
                    (fn {{}} (params {{}} {parameter})
                        (lit {{type: (t-prim {{}} f32)}} 1.0)))))"
        );
        let deep = chelis_deep::parse_and_stamp_file(&source).expect("Deep fixture stamps");
        let result = check_typed_program(&deep);
        assert_eq!(result.is_ok(), accepted, "{source}\n{result:?}");
    }
}

/// [04-INF-1]: resolving the enclosing callable is not an application of the
/// gradient it returns. The result constraint waits for that inner binding.
#[test]
fn result_ascriptions_do_not_bind_late_revealed_gradient_parameters() {
    for (formals, target, argument) in [
        ("f", "f", "fn (w) -> 1.0f32"),
        ("p", "p.0", "(fn (w) -> 1.0f32, true)"),
    ] {
        let prefix = format!(
            "def main() = {{\n make = fn ({formals}) -> {{\n d: (f32 -> f32) = grad({target})\n d\n }}\n"
        );
        check(&format!("{prefix} make({argument})\n}}"), false);
        check(&format!("{prefix} (make({argument}))(2.0f32)\n}}"), true);
        check(&format!("{prefix} (make({argument}))(true)\n}}"), false);
        let annotated = argument.replace("(w)", "(w: f32)");
        check(&format!("{prefix} make({annotated})\n}}"), true);
    }
}

/// The constraint may precede the Grad producer: a projected value is still
/// an output of a suspended semantic rule, even before its type is known.
#[test]
fn result_ascriptions_follow_semantic_producers_through_projection() {
    let prefix = "def main() = {\n take = fn (p) -> {\n d: (f32 -> f32) = p.0\n d\n }\n";
    check(
        &format!("{prefix} take((grad(fn (w) -> 1.0f32), true))\n}}"),
        false,
    );
    check(
        &format!("{prefix} (take((grad(fn (w) -> 1.0f32), true)))(2.0f32)\n}}"),
        true,
    );
    check(
        &format!("{prefix} (take((grad(fn (w) -> 1.0f32), true)))(true)\n}}"),
        false,
    );
    check(
        &format!("{prefix} take((grad(fn (w: f32) -> 1.0f32), true))\n}}"),
        true,
    );
    // Once projection resolves to an ordinary unconstrained value, its
    // ascription can constrain that value; no Grad obligation remains.
    check(
        &format!("{prefix} take((fn (w) -> 1.0f32, true))\n}}"),
        true,
    );
}

#[test]
fn a_declaration_result_ascription_waits_for_late_revealed_grad() {
    let body = "{\n make = fn (f) -> grad(f)\n make(fn (w) -> 1.0f32)\n}";
    check(&format!("def main() -> (f32 -> f32) = {body}"), false);
    check(
        &format!(
            "def main() -> (f32 -> f32) = {}",
            body.replace("(w)", "(w: f32)")
        ),
        true,
    );
}

#[test]
fn deep_result_constraints_follow_an_initially_unknown_callable() {
    for (parameter, apply, accepted) in [
        ("w", false, false),
        ("w", true, true),
        ("(w {type: (t-prim {} f32)})", false, true),
    ] {
        let value = format!(
            "(app {{}}
                (fn {{}} (params {{}} f)
                    (grad {{type: (t-fn {{}} (t-prim {{}} f32) (t-prim {{}} f32))}}
                        (var {{}} f)))
                (fn {{}} (params {{}} {parameter})
                    (lit {{type: (t-prim {{}} f32)}} 1.0)))"
        );
        let value = if apply {
            format!("(app {{}} {value} (lit {{type: (t-prim {{}} f32)}} 2.0))")
        } else {
            value
        };
        let source = format!("(def {{}} main (fn {{}} (params {{}}) {value}))");
        let deep = chelis_deep::parse_and_stamp_file(&source).expect("Deep fixture stamps");
        let result = check_typed_program(&deep);
        assert_eq!(result.is_ok(), accepted, "{source}\n{result:?}");
    }
}

#[test]
fn result_constraints_keep_their_boundary_in_recursive_groups() {
    let sibling = "def sibling(n: i32) -> (f32 -> f32) = driver(n)";
    for (parameter, application, accepted) in [
        ("w", "", false),
        ("w: f32", "", true),
        ("w", " zero = d(2.0f32)\n", true),
        ("w", " zero = d(true)\n", false),
    ] {
        let driver = format!(
            "def driver(n: i32) -> (f32 -> f32) = {{\n next = fn (ignored: unit) -> sibling(n)\n make = fn (f) -> {{\n value: (f32 -> f32) = grad(f)\n value\n }}\n d = make(fn ({parameter}) -> 1.0f32)\n{application} d\n}}"
        );
        for source in [
            format!("{driver}\n{sibling}"),
            format!("{sibling}\n{driver}"),
        ] {
            check(&source, accepted);
        }
    }
}

#[test]
fn result_ascriptions_follow_record_field_derivations() {
    let prefix = "type Holder[a] =\n | Holder { value: a }\ndef main() = {\n take = fn (p) -> {\n d: (f32 -> f32) = p.value\n d\n }\n";
    check(
        &format!("{prefix} take(Holder {{ value: grad(fn (w) -> 1.0f32) }})\n}}"),
        false,
    );
    check(
        &format!("{prefix} (take(Holder {{ value: grad(fn (w) -> 1.0f32) }}))(2.0f32)\n}}"),
        true,
    );
    check(
        &format!("{prefix} take(Holder {{ value: grad(fn (w: f32) -> 1.0f32) }})\n}}"),
        true,
    );
    check(
        &format!("{prefix} (take(Holder {{ value: grad(fn (w) -> 1.0f32) }}))(true)\n}}"),
        false,
    );
}

#[test]
fn branch_results_cannot_supply_a_gradient_parameter_binding() {
    for parameter in ["w", "w: f32"] {
        let unknown = format!("grad(fn ({parameter}) -> 1.0f32)");
        let known = "grad(fn (w: f32) -> 1.0f32)";
        for (left, right) in [(&unknown[..], known), (known, &unknown[..])] {
            for join in [
                format!("if flag then {left} else {right}"),
                format!("match flag with {{\n | true => {left}\n | false => {right}\n }}"),
            ] {
                check(&format!("def main(flag: bool) = {join}"), parameter != "w");
                check(
                    &format!("def main(flag: bool) = {{\n g = {join}\n g(2.0f32)\n}}"),
                    true,
                );
                check(
                    &format!("def main(flag: bool) = {{\n g = {join}\n g(true)\n}}"),
                    false,
                );
            }
        }
    }
}

#[test]
fn joined_gradients_keep_applications_through_aggregate_and_nested_joins() {
    for (left, right, application) in [
        (
            "(grad(fn (w) -> 1.0f32), true)",
            "(grad(fn (w: f32) -> 1.0f32), false)",
            "(g.0)(2.0f32)",
        ),
        (
            "Holder { value: grad(fn (w) -> 1.0f32) }",
            "Holder { value: grad(fn (w: f32) -> 1.0f32) }",
            "(g.value)(2.0f32)",
        ),
        (
            "grad(fn (w, v: f32) -> 1.0f32)",
            "grad(fn (w: f32, v) -> 1.0f32)",
            "g(2.0f32, 3.0f32)",
        ),
    ] {
        let prefix = "type Holder[a] =\n | Holder { value: a }\ndef main(flag: bool) = {\n";
        let join = format!("if flag then {left} else {right}");
        for join in [join.clone(), format!("if flag then ({join}) else ({join})")] {
            check(&format!("{prefix} g = {join}\n g\n}}"), false);
            check(&format!("{prefix} g = {join}\n {application}\n}}"), true);
            check(
                &format!(
                    "{prefix} g = {join}\n {}\n}}",
                    application.replace("2.0f32", "true")
                ),
                false,
            );
        }
    }
}

#[test]
fn joined_late_callable_gradients_keep_the_binding_boundary() {
    for parameter in ["w", "w: f32"] {
        let prefix = format!(
            "def main(flag: bool) = {{\n make = fn (f) -> if flag then grad(f) else grad(fn (w: f32) -> 1.0f32)\n g = make(fn ({parameter}) -> 1.0f32)\n"
        );
        check(&format!("{prefix} g\n}}"), parameter != "w");
        check(&format!("{prefix} g(2.0f32)\n}}"), true);
        check(&format!("{prefix} g(true)\n}}"), false);
    }
}

#[test]
fn recursive_return_joins_cannot_supply_a_gradient_parameter_binding() {
    for (parameter, application, accepted) in [
        ("w", "", false),
        ("w: f32", "", true),
        ("w", " zero = g(2.0f32)\n", true),
        ("w", " zero = g(true)\n", false),
    ] {
        let body = format!(
            "{{\n g = if flag then grad(fn ({parameter}) -> 1.0f32) else sibling(flag)\n{application} g\n}}"
        );
        let driver = format!("def driver(flag: bool) -> (f32 -> f32) = {body}");
        let sibling = "def sibling(flag: bool) -> (f32 -> f32) = driver(flag)";
        check(&format!("{driver}\n{sibling}"), accepted);
        check(&format!("{sibling}\n{driver}"), accepted);
    }
}
