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

#[test]
fn result_constraint_origins_survive_helpers_inferred_before_grad() {
    for helper in [
        "fn (p) -> {\n d: (f32 -> f32) = p\n d\n}",
        "fn (p) -> {\n alias = p\n d: (f32 -> f32) = alias\n d\n}",
        "fn (p) -> if flag then p else grad(fn (w: f32) -> 1.0f32)",
        "fn (p) -> match flag with {\n | true => p\n | false => grad(fn (w: f32) -> 1.0f32)\n}",
    ] {
        for (parameter, tail, accepted) in [
            ("w", "g", false),
            ("w: f32", "g", true),
            ("w", "g(2.0f32)", true),
            ("w", "g(true)", false),
        ] {
            check(
                &format!(
                    "def main(flag: bool) = {{\n take = {helper}\n g = take(grad(fn ({parameter}) -> 1.0f32))\n {tail}\n}}"
                ),
                accepted,
            );
        }
    }
}

#[test]
fn selector_origins_survive_generalization_before_grad() {
    for body in [
        "if flag then f else h",
        "match flag with {\n | true => f\n | false => h\n}",
    ] {
        for (parameter, tail, accepted) in [
            ("w", "g", false),
            ("w: f32", "g", true),
            ("w", "g(2.0f32)", true),
            ("w", "g(true)", false),
        ] {
            check(
                &format!(
                    "def main(flag: bool) = {{\n choose = fn (f, h) -> {body}\n g = choose(grad(fn ({parameter}) -> 1.0f32), grad(fn (w: f32) -> 1.0f32))\n {tail}\n}}"
                ),
                accepted,
            );
        }
        check(
            &format!(
                "def main(flag: bool) = {{\n choose = fn (f, h) -> {body}\n number = choose(1.0f32, 2.0f32)\n truth = choose(true, false)\n (number, truth)\n}}"
            ),
            true,
        );
        check(
            &format!(
                "def main(flag: bool) = {{\n choose = fn (f, h) -> {body}\n choose(1.0f32, true)\n}}"
            ),
            false,
        );
    }
}

#[test]
fn list_element_equality_is_a_result_constraint() {
    for parameter in ["w", "w: f32"] {
        let unknown = format!("grad(fn ({parameter}) -> 1.0f32)");
        let known = "grad(fn (w: f32) -> 1.0f32)";
        for (left, right) in [(&unknown[..], known), (known, &unknown[..])] {
            check(&format!("def main() = [{left}, {right}]"), parameter != "w");
            check(
                &format!("def main() = {{\n g = {unknown}\n zero = g(2.0f32)\n [g, {known}]\n}}"),
                true,
            );
            check(
                &format!("def main() = {{\n g = {unknown}\n zero = g(true)\n [g, {known}]\n}}"),
                false,
            );
        }
    }
}

#[test]
fn published_helpers_preserve_result_constraint_origins() {
    for helper in [
        "def forward_grad(p) -> (f32 -> f32) = p",
        "def forward_grad(p) = {\n d: (f32 -> f32) = p\n d\n}",
    ] {
        check(
            &format!("{helper}\ndef main() = forward_grad(grad(fn (w) -> 1.0f32))"),
            false,
        );
        check(
            &format!("{helper}\ndef main() = (forward_grad(grad(fn (w) -> 1.0f32)))(2.0f32)"),
            true,
        );
        check(
            &format!("{helper}\ndef main() = forward_grad(grad(fn (w: f32) -> 1.0f32))"),
            true,
        );
    }
}

#[test]
fn recursive_helpers_preserve_result_constraint_origins() {
    for flag in ["flag", "flag: bool"] {
        for body in [
            "if flag then {\n d: (f32 -> f32) = p\n d\n} else forward_grad(p, true)",
            "if flag then p else forward_grad(grad(fn (w: f32) -> 1.0f32), true)",
        ] {
            let helper = format!("def forward_grad(p, {flag}) = {body}");
            check(
                &format!("{helper}\ndef main() = forward_grad(grad(fn (w) -> 1.0f32), true)"),
                false,
            );
            check(
                &format!(
                    "{helper}\ndef main() = (forward_grad(grad(fn (w) -> 1.0f32), true))(2.0f32)"
                ),
                true,
            );
            check(
                &format!("{helper}\ndef main() = forward_grad(grad(fn (w: f32) -> 1.0f32), true)"),
                true,
            );
        }
    }
}

#[test]
fn recursive_helper_input_applications_remain_binding_sites() {
    let apply = "def apply_grad(p, flag: bool) = if flag then p(2.0f32) else sibling_grad(p, true)";
    let sibling = "def sibling_grad(p, flag: bool) = apply_grad(p, flag)";
    for definitions in [format!("{apply}\n{sibling}"), format!("{sibling}\n{apply}")] {
        check(
            &format!("{definitions}\ndef main() = sibling_grad(grad(fn (w) -> 1.0f32), true)"),
            true,
        );
        check(
            &format!(
                "{definitions}\ndef main() = sibling_grad(grad(fn (w: bool) -> 1.0f32), true)"
            ),
            false,
        );
    }
}

#[test]
fn result_origins_cross_recursive_sibling_chains() {
    let first = "def first_grad(p, flag: bool) = second_grad(p, flag)";
    let second = "def second_grad(p, flag: bool) = third_grad(p, flag)";
    let third = "def third_grad(p, flag: bool) = if flag then {\n d: (f32 -> f32) = p\n d\n} else first_grad(p, true)";
    for definitions in [
        format!("{first}\n{second}\n{third}"),
        format!("{third}\n{second}\n{first}"),
    ] {
        for name in ["first_grad", "second_grad", "third_grad"] {
            check(
                &format!("{definitions}\ndef main() = {name}(grad(fn (w) -> 1.0f32), true)"),
                false,
            );
            check(
                &format!(
                    "{definitions}\ndef main() = ({name}(grad(fn (w) -> 1.0f32), true))(2.0f32)"
                ),
                true,
            );
        }
    }
}

#[test]
fn result_origins_follow_recursive_result_projections() {
    let first = "def first_grad(p, flag: bool) = second_grad(p, flag).0";
    let second =
        "def second_grad(p, flag: bool) = if flag then (p, true) else (first_grad(p, true), true)";
    for definitions in [format!("{first}\n{second}"), format!("{second}\n{first}")] {
        check(
            &format!("{definitions}\ndef main() = first_grad(grad(fn (w) -> 1.0f32), true)"),
            false,
        );
        check(
            &format!(
                "{definitions}\ndef main() = (first_grad(grad(fn (w) -> 1.0f32), true))(2.0f32)"
            ),
            true,
        );
        check(
            &format!(
                "{definitions}\ndef main() = (first_grad(grad(fn (w: f32) -> 1.0f32), true))(true)"
            ),
            false,
        );
    }
}

#[test]
fn record_updates_preserve_result_constraint_origins() {
    for (parameter, tail, accepted) in [
        ("w", "updated", false),
        ("w: f32", "updated", true),
        ("w", "updated.value(2.0f32)", true),
        ("w", "updated.value(true)", false),
    ] {
        check(
            &format!(
                "type Holder[a] = | Holder {{ value: a }}\ndef main() = {{\n base = Holder {{ value: grad(fn (w: f32) -> 1.0f32) }}\n updated = base with {{ value: grad(fn ({parameter}) -> 1.0f32) }}\n {tail}\n}}"
            ),
            accepted,
        );
    }
}

#[test]
fn checked_library_round_trip_preserves_result_constraint_origins() {
    use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_with_context};
    for library in [
        "def forward_grad(p) -> (f32 -> f32) = p",
        "def forward_grad(p) = index(append([p], grad(fn (w: f32) -> 1.0f32)), 0i64)",
        "def forward_grad(p) = index(concat([p], [grad(fn (w: f32) -> 1.0f32)]), 0i64)",
        "def forward_grad(p) = index(dict_values(dict_insert(dict_of([(\"a\", grad(fn (w: f32) -> 1.0f32))]), \"b\", p)), 0i64)",
        "def forward_grad(p) = index(dict_values(dict_merge(dict_of([(\"a\", p)]), dict_of([(\"b\", grad(fn (w: f32) -> 1.0f32))]))), 0i64)",
        "def forward_grad(p) = fold(fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), p, [1i64])",
        "def forward_grad(p) = index(scan(fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), p, [1i64]), 0i64)",
        "def forward_grad(p, flag: bool) = sibling(p, flag)\ndef sibling(p, flag: bool) = if flag then p else forward_grad(grad(fn (w: f32) -> 1.0f32), true)",
        "def forward_grad(p, flag: bool) = if flag then {\n d: (f32 -> f32) = p\n d\n} else forward_grad(p, true)",
    ] {
        let parsed = parse_str(library).unwrap();
        let library = desugar_program(&parsed).unwrap();
        let context = build_type_env_from_library(&library).unwrap();
        let restored: TypeEnv =
            bincode::deserialize(&bincode::serialize(&context).unwrap()).unwrap();
        let arity = match &context.scheme("forward_grad").unwrap().body {
            chelis_types::types::Type::Fn(params, _) => params.len(),
            _ => unreachable!(),
        };
        let flag = if arity == 2 { ", true" } else { "" };
        for (parameter, application, accepted) in [
            ("w", "", false),
            ("w: f32", "", true),
            ("w", "(2.0f32)", true),
            ("w", "(true)", false),
        ] {
            let source = format!(
                "def main() = (forward_grad(grad(fn ({parameter}) -> 1.0f32){flag})){application}"
            );
            let program = desugar_program(&parse_str(&source).unwrap()).unwrap();
            for context in [&context, &restored] {
                let checked = check_ir_with_context(context, &program);
                assert_eq!(checked.is_ok(), accepted, "{source}\n{checked:?}");
            }
        }
    }
}

#[test]
fn collection_operation_equalities_preserve_result_origin() {
    for expression in [
        "concat([g], [known])",
        "concat([known], [g])",
        "append([g], known)",
        "append([known], g)",
    ] {
        for (parameter, application, accepted) in [
            ("w: f32", "", true),
            ("w", "zero = g(2.0f32)\n", true),
            ("w", "zero = g(true)\n", false),
            ("w", "", false),
        ] {
            check(
                &format!(
                    "def main() = {{\n g = grad(fn ({parameter}) -> 1.0f32)\n known = grad(fn (w: f32) -> 1.0f32)\n {application}{expression}\n}}"
                ),
                accepted,
            );
        }
    }
}

#[test]
fn collection_operation_origins_survive_helper_publication() {
    for (operation, arguments) in [
        ("concat", "[p], [grad(fn (w: f32) -> 1.0f32)]"),
        ("append", "[p], grad(fn (w: f32) -> 1.0f32)"),
    ] {
        for callee in [
            format!("{operation}({arguments})"),
            format!("{{\n join = {operation}\n join({arguments})\n}}"),
        ] {
            for (parameter, application, accepted) in [
                ("w: f32", "", true),
                ("w", "(2.0f32)", true),
                ("w", "(true)", false),
                ("w", "", false),
            ] {
                check(
                    &format!(
                        "def pick(p) = index({callee}, 0i64)\ndef main() = (pick(grad(fn ({parameter}) -> 1.0f32))){application}"
                    ),
                    accepted,
                );
            }
        }
    }
}

#[test]
fn mutual_publication_preserves_required_branch_equalities() {
    let first = "def first(p, flag: bool) = second(p, flag)";
    let second = "def second(p, flag: bool) = if flag then p else first(1.0f32, true)";
    for definitions in [format!("{first}\n{second}"), format!("{second}\n{first}")] {
        check(
            &format!("{definitions}\ndef main() -> f32 = first(2.0f32, false)"),
            true,
        );
        check(
            &format!("{definitions}\ndef main() -> bool = first(true, false)"),
            false,
        );
        check(
            &format!("{definitions}\ndef main() -> bool = first(2.0f32, false)"),
            false,
        );
    }
}

#[test]
fn mutual_publication_keeps_grad_result_equality_without_binding_its_input() {
    for (prefix, wrap, project) in [
        ("", "p", "second(p, flag)"),
        ("", "[p]", "index(second(p, flag), 0i64)"),
        (
            "type Holder[a] = | Holder { value: a }\n",
            "Holder { value: p }",
            "second(p, flag).value",
        ),
    ] {
        let first = format!("def first(p, flag: bool) = {project}");
        let other = wrap.replace("p", "first(grad(fn (w: f32) -> 1.0f32), true)");
        let second = format!("def second(p, flag: bool) = if flag then {wrap} else {other}");
        for definitions in [format!("{first}\n{second}"), format!("{second}\n{first}")] {
            for (parameter, application, accepted) in [
                ("w: f32", "", true),
                ("w", "(2.0f32)", true),
                ("w", "", false),
                ("w", "(true)", false),
            ] {
                check(
                    &format!(
                        "{prefix}{definitions}\ndef main() = (first(grad(fn ({parameter}) -> 1.0f32), false)){application}"
                    ),
                    accepted,
                );
            }
        }
    }
}

#[test]
fn checked_library_round_trip_preserves_recursive_signature_compatibility() {
    use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_with_context};
    let library = "def first(p, flag: bool) = second(p, flag)\ndef second(p, flag: bool) = if flag then p else first(1.0f32, true)";
    let library = desugar_program(&parse_str(library).unwrap()).unwrap();
    let context = build_type_env_from_library(&library).unwrap();
    let restored: TypeEnv = bincode::deserialize(&bincode::serialize(&context).unwrap()).unwrap();
    for (ty, value, accepted) in [
        ("f32", "2.0f32", true),
        ("bool", "true", false),
        ("bool", "2.0f32", false),
    ] {
        let source = format!("def main() -> {ty} = first({value}, false)");
        let program = desugar_program(&parse_str(&source).unwrap()).unwrap();
        for context in [&context, &restored] {
            let checked = check_ir_with_context(context, &program);
            assert_eq!(checked.is_ok(), accepted, "{source}\n{checked:?}");
        }
    }
}

#[test]
fn collection_value_holes_retain_their_required_element_equation() {
    for body in [
        "append([1.0f32], p)",
        "{\n op = append\n op([1.0f32], p)\n}",
    ] {
        check(
            &format!("def wrap(p) = {body}\ndef main() = wrap(2.0f32)"),
            true,
        );
        check(
            &format!("def wrap(p) = {body}\ndef main() = wrap(true)"),
            false,
        );
    }
}

#[test]
fn collection_value_holes_cannot_inherit_a_known_gradient_element() {
    for body in [
        "append([grad(fn (w: f32) -> 1.0f32)], p)",
        "{\n op = append\n op([grad(fn (w: f32) -> 1.0f32)], p)\n}",
    ] {
        for (parameter, application, accepted) in [
            ("w: f32", "", true),
            ("w", "(2.0f32)", true),
            ("w", "(true)", false),
            ("w", "", false),
        ] {
            check(
                &format!(
                    "def wrap(p) = {body}\ndef main() = (index(wrap(grad(fn ({parameter}) -> 1.0f32)), 0i64)){application}"
                ),
                accepted,
            );
        }
    }
}

// [05-OP-56] requires equal value types; spec/06 §2.2 makes that
// equality a result constraint, including before a helper is instantiated.
#[test]
fn dictionary_operation_equalities_preserve_result_origin() {
    for expression in [
        "dict_insert(dict_of([(\"a\", known)]), \"b\", g)",
        "dict_insert(dict_of([(\"a\", g)]), \"b\", known)",
        "dict_merge(dict_of([(\"a\", g)]), dict_of([(\"b\", known)]))",
        "dict_merge(dict_of([(\"a\", known)]), dict_of([(\"b\", g)]))",
    ] {
        for (parameter, application, accepted) in [
            ("w: f32", "", true),
            ("w", "zero = g(2.0f32)\n", true),
            ("w", "zero = g(true)\n", false),
            ("w", "", false),
        ] {
            check(
                &format!(
                    "def main() = {{\n g = grad(fn ({parameter}) -> 1.0f32)\n known = grad(fn (w: f32) -> 1.0f32)\n {application}{expression}\n}}"
                ),
                accepted,
            );
        }
    }
}

#[test]
fn dictionary_operation_origins_survive_helper_publication() {
    for expression in [
        "dict_insert(dict_of([(\"a\", grad(fn (w: f32) -> 1.0f32))]), \"b\", p)",
        "dict_merge(dict_of([(\"a\", p)]), dict_of([(\"b\", grad(fn (w: f32) -> 1.0f32))]))",
    ] {
        for (parameter, application, accepted) in [
            ("w: f32", "", true),
            ("w", "(2.0f32)", true),
            ("w", "(true)", false),
            ("w", "", false),
        ] {
            check(
                &format!(
                    "def pick(p) = index(dict_values({expression}), 0i64)\ndef main() = (pick(grad(fn ({parameter}) -> 1.0f32))){application}"
                ),
                accepted,
            );
        }
    }
}

#[test]
fn dictionary_operation_origins_retain_required_value_equality() {
    for expression in [
        "dict_insert(dict_of([(\"a\", 1.0f32)]), \"b\", p)",
        "dict_merge(dict_of([(\"a\", p)]), dict_of([(\"b\", 1.0f32)]))",
    ] {
        for (value, accepted) in [("2.0f32", true), ("true", false)] {
            check(
                &format!("def main() = {{\n p = {value}\n {expression}\n}}"),
                accepted,
            );
            check(
                &format!("def pick(p) = {expression}\ndef main() = pick({value})"),
                accepted,
            );
        }
    }
}

#[test]
fn callback_accumulator_results_preserve_result_origin() {
    for operation in ["fold", "scan"] {
        for (parameter, application, accepted) in [
            ("w: f32", "", true),
            ("w", "zero = g(2.0f32)\n", true),
            ("w", "zero = g(true)\n", false),
            ("w", "", false),
        ] {
            check(
                &format!(
                    "def main() = {{\n g = grad(fn ({parameter}) -> 1.0f32)\n known = grad(fn (w: f32) -> 1.0f32)\n {application}{operation}(fn (acc, item: i64) -> known, g, [1i64])\n}}"
                ),
                accepted,
            );
        }
    }
}

#[test]
fn callback_accumulator_results_keep_applications_and_required_equality() {
    for operation in ["fold", "scan"] {
        let projection = if operation == "scan" {
            "index(r, 0i64)"
        } else {
            "r"
        };
        for (parameter, application, accepted) in [
            ("w: f32", "", true),
            ("w", "(2.0f32)", true),
            ("w", "(true)", false),
            ("w", "", false),
        ] {
            check(
                &format!(
                    "def pick(p) = {{\n r = {operation}(fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), p, [1i64])\n {projection}\n}}\ndef main() = (pick(grad(fn ({parameter}) -> 1.0f32))){application}"
                ),
                accepted,
            );
        }
        for (value, accepted) in [("2.0f32", true), ("true", false)] {
            check(
                &format!(
                    "def main() = {operation}(fn (acc, item: i64) -> 1.0f32, {value}, [1i64])"
                ),
                accepted,
            );
        }
        // Calling the accumulator itself remains a real parameter binding.
        check(
            &format!(
                "def main() = {operation}(fn (acc, item: i64) -> {{\n zero = acc(2.0f32)\n acc\n}}, grad(fn (w) -> 1.0f32, wrt=w), [1i64])"
            ),
            true,
        );
        check(
            &format!(
                "def main() = {operation}(fn (acc, item: i64) -> {{\n zero = acc(true)\n acc\n}}, grad(fn (w) -> 1.0f32, wrt=w), [1i64])"
            ),
            false,
        );
    }
}

fn check_accumulator_alias_result(operation: &str, result_type: &str) {
    let call = "op(fn (acc: f32, item: i64) -> acc, 1.0f32, [1i64])";
    check(
        &format!("def main() -> {result_type} = {{\n op = {operation}\n {call}\n}}"),
        true,
    );
    check(
        &format!("def main() -> bool = {{\n op = {operation}\n {call}\n}}"),
        false,
    );
}

#[test]
fn fold_alias_retains_its_required_result_type() {
    check_accumulator_alias_result("fold", "f32");
}

#[test]
fn scan_alias_retains_its_required_result_type() {
    check_accumulator_alias_result("scan", "List[f32]");
}

#[test]
fn accumulator_contract_survives_callable_transport_and_cache() {
    use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_with_context};
    for (operation, output) in [("fold", "f32"), ("scan", "List[f32]")] {
        let library = format!(
            "def return_op() = {operation}\ndef pair_op() = ({operation}, true)\ndef identity(f) = f\ndef recursive_op(flag: bool) = if flag then {operation} else recursive_op(true)"
        );
        let program = desugar_program(&parse_str(&library).unwrap()).unwrap();
        let context = build_type_env_from_library(&program).unwrap();
        let restored: TypeEnv =
            bincode::deserialize(&bincode::serialize(&context).unwrap()).unwrap();
        for callee in [
            "return_op()".to_string(),
            "pair_op().0".to_string(),
            "recursive_op(true)".to_string(),
            format!("identity({operation})"),
        ] {
            for (result_type, accepted) in [(output, true), ("bool", false)] {
                let source = format!(
                    "def main() -> {result_type} = {{\n op = {callee}\n op(fn (acc: f32, item: i64) -> acc, 1.0f32, [1i64])\n}}"
                );
                let program = desugar_program(&parse_str(&source).unwrap()).unwrap();
                for context in [&context, &restored] {
                    let checked = check_ir_with_context(context, &program);
                    assert_eq!(checked.is_ok(), accepted, "{source}\n{checked:?}");
                }
            }
        }
    }
}

#[test]
fn accumulator_aliases_retain_callback_and_element_requirements() {
    for operation in ["fold", "scan"] {
        for (callback, initial, elements, accepted) in [
            ("fn (acc: f32, item: i64) -> acc", "1.0f32", "[1i64]", true),
            (
                "fn (acc: f32, item: i64) -> true",
                "1.0f32",
                "[1i64]",
                false,
            ),
            ("fn (acc: f32, item: i64) -> acc", "true", "[1i64]", false),
            ("fn (acc: f32, item: i64) -> acc", "1.0f32", "[true]", false),
            ("fn (acc: f32, item: i64) -> acc", "1.0f32", "1i64", false),
            ("fn (acc: f32) -> acc", "1.0f32", "[1i64]", false),
        ] {
            check(
                &format!(
                    "def main() = {{\n op = {operation}\n op({callback}, {initial}, {elements})\n}}"
                ),
                accepted,
            );
        }
    }
}

#[test]
fn accumulator_aliases_carry_grad_result_origin_before_helper_instantiation() {
    use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_with_context};
    for operation in ["fold", "scan"] {
        let project = if operation == "scan" {
            "index(r, 0i64)"
        } else {
            "r"
        };
        let library = format!(
            "def pick(p) = {{\n op = {operation}\n r = op(fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), p, [1i64])\n {project}\n}}"
        );
        let program = desugar_program(&parse_str(&library).unwrap()).unwrap();
        let context = build_type_env_from_library(&program).unwrap();
        let restored: TypeEnv =
            bincode::deserialize(&bincode::serialize(&context).unwrap()).unwrap();
        for (parameter, application, accepted) in [
            ("w: f32", "", true),
            ("w", "(2.0f32)", true),
            ("w", "", false),
            ("w", "(true)", false),
        ] {
            let source =
                format!("def main() = (pick(grad(fn ({parameter}) -> 1.0f32))){application}");
            check(&format!("{library}\n{source}"), accepted);
            let program = desugar_program(&parse_str(&source).unwrap()).unwrap();
            for context in [&context, &restored] {
                let checked = check_ir_with_context(context, &program);
                assert_eq!(checked.is_ok(), accepted, "{source}\n{checked:?}");
            }
        }
    }
}

fn check_transported_fold_local_result(selection: &str) {
    let source = format!(
        "type Holder[a] =\n  | Holder {{ value: a }}\ndef main() = {{\n  op = {selection}\n  r = op(fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), grad(fn (w) -> 1.0f32), [1i64])\n  g = r\n  g\n}}"
    );
    check(&source, false);
    check(
        &source.replace("grad(fn (w) ->", "grad(fn (w: f32) ->"),
        true,
    );
    check(&source.replace("  g\n}", "  g(2.0f32)\n}"), true);
    check(&source.replace("  g\n}", "  g(true)\n}"), false);
}

#[test]
fn fold_list_escape_keeps_local_result_provenance() {
    check_transported_fold_local_result("index([fold, fold], 0i64)");
}

#[test]
fn fold_record_escape_keeps_local_result_provenance() {
    check_transported_fold_local_result("(Holder { value: fold }).value");
}

#[test]
fn fold_branch_escape_keeps_local_result_provenance() {
    check_transported_fold_local_result("if true then fold else fold");
}

#[test]
fn callable_result_components_survive_transport_alias_depth_and_cache() {
    use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_with_context};
    let library = "type Holder[a] = | Holder { value: a }\ndef select(p) = if true then p else grad(fn (w: f32) -> 1.0f32)\ndef identity(f) = f";
    let program = desugar_program(&parse_str(library).unwrap()).unwrap();
    let context = build_type_env_from_library(&program).unwrap();
    let restored: TypeEnv = bincode::deserialize(&bincode::serialize(&context).unwrap()).unwrap();
    for (selection, arguments, projection) in [
        (
            "index([fold, fold], 0i64)",
            "fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), INPUT, [1i64]",
            "r",
        ),
        (
            "(Holder { value: fold }).value",
            "fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), INPUT, [1i64]",
            "r",
        ),
        (
            "if true then fold else fold",
            "fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), INPUT, [1i64]",
            "r",
        ),
        (
            "index([scan, scan], 0i64)",
            "fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), INPUT, [1i64]",
            "index(r, 0i64)",
        ),
        (
            "(Holder { value: scan }).value",
            "fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), INPUT, [1i64]",
            "index(r, 0i64)",
        ),
        (
            "if true then scan else scan",
            "fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), INPUT, [1i64]",
            "index(r, 0i64)",
        ),
        ("index([select, select], 0i64)", "INPUT", "r"),
        ("(Holder { value: select }).value", "INPUT", "r"),
        ("if true then select else select", "INPUT", "r"),
        ("identity(index([select, select], 0i64))", "INPUT", "r"),
    ] {
        for depth in [1, 3] {
            let mut aliases = format!("  g0 = {projection}\n");
            for index in 1..depth {
                aliases.push_str(&format!("  g{index} = g{}\n", index - 1));
            }
            for (parameter, application, accepted) in [
                ("w", "", false),
                ("w: f32", "", true),
                ("w", "(2.0f32)", true),
                ("w", "(true)", false),
            ] {
                let arguments =
                    arguments.replace("INPUT", &format!("grad(fn ({parameter}) -> 1.0f32)"));
                let source = format!(
                    "def main() = {{\n  op = {selection}\n  r = op({arguments})\n{aliases}  g{}{application}\n}}",
                    depth - 1
                );
                check(&format!("{library}\n{source}"), accepted);
                let program = desugar_program(&parse_str(&source).unwrap()).unwrap();
                for context in [&context, &restored] {
                    let checked = check_ir_with_context(context, &program);
                    assert_eq!(checked.is_ok(), accepted, "{source}\n{checked:?}");
                }
            }
        }
    }
}

#[test]
fn restricted_result_origins_publish_a_coherent_scheme() {
    use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_with_context};
    for source in [
        "def choose[p: Numeric](x: p, flag: bool) -> p = if flag then x else x\n",
        "sig choose[n, p: Numeric]: p -> tensor[n, p]\ndef choose(x) = to_tensor(skip([x], 1i64))\n",
        "def choose[p: Numeric](a: p, b: p) -> p = if lt(a, b) then a else b\n",
        "def choose[p: Int](current: p, stop: p, out: List[p]) -> List[p] = if gte(current, stop) then out else choose(add(current, cast(1, p)), stop, append(out, current))\n",
    ] {
        let library = desugar_program(&parse_str(source).unwrap()).unwrap();
        let live = build_type_env_from_library(&library).unwrap();
        let scheme = live.scheme("choose").unwrap();
        for (variable, _) in &scheme.tvar_restrictions {
            assert!(
                scheme.tvars.contains(variable),
                "{source}: restriction must quantify a published variable: {scheme:?}"
            );
            assert!(
                chelis_types::env::free_tvars(&scheme.body).contains(variable),
                "{source}: restriction must occur in the published body: {scheme:?}"
            );
        }
    }
    let library = desugar_program(
        &parse_str("def choose[p: Numeric](x: p, flag: bool) -> p = if flag then x else x\n")
            .unwrap(),
    )
    .unwrap();
    let live = build_type_env_from_library(&library).unwrap();
    let restored: TypeEnv = bincode::deserialize(&bincode::serialize(&live).unwrap()).unwrap();
    for context in [&live, &restored] {
        for (source, accepted) in [
            ("out = choose(1.0f32, true)", true),
            ("out = choose(1i64, false)", true),
            ("out = choose(true, true)", false),
            ("out = choose(\"bad\", false)", false),
        ] {
            let deep = desugar_program(&parse_str(source).unwrap()).unwrap();
            let result = check_ir_with_context(context, &deep);
            assert_eq!(result.is_ok(), accepted, "{source}: {result:?}");
        }
    }
}

#[test]
fn concrete_empty_list_result_evidence_reaches_shape_operations() {
    check(
        "values: List[f64] = []\nempty = to_tensor(values)\nreshaped = reshape(empty, [2147483648i64, 0i64])",
        true,
    );
    check(
        "values: List[f64] = []\nempty = to_tensor(values)\nreshaped: tensor[2147483648, 0, f32] = reshape(empty, [2147483648i64, 0i64])",
        false,
    );
    check("def bad() = fn (x) -> reshape(x, [2i64])", false);
}

#[test]
fn origin_dtype_restrictions_survive_representative_changes_and_serialization() {
    use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_with_context};
    let source =
        "sig boxed[n, p: Float]: p -> tensor[n, p]\ndef boxed(x) = to_tensor(skip([x], 1i64))\n";
    let library = desugar_program(&parse_str(source).unwrap()).unwrap();
    let live = build_type_env_from_library(&library).unwrap();
    let scheme = live.scheme("boxed").unwrap();
    let origin = scheme
        .result_origin
        .as_ref()
        .expect("fixture has raw result origins");
    assert!(!origin.tvar_restrictions.is_empty());
    let mut wire = serde_json::to_value(scheme).unwrap();
    wire["result_origin"]
        .as_object_mut()
        .unwrap()
        .remove("tvar_restrictions");
    assert!(serde_json::from_value::<chelis_types::types::Scheme>(wire).is_err());
    let restored: TypeEnv = bincode::deserialize(&bincode::serialize(&live).unwrap()).unwrap();
    for context in [&live, &restored] {
        for (source, accepted) in [
            ("out = boxed(1.0f32)", true),
            ("out = boxed(1.0f64)", true),
            ("out = boxed(1i64)", false),
            ("out = boxed(true)", false),
        ] {
            let deep = desugar_program(&parse_str(source).unwrap()).unwrap();
            let result = check_ir_with_context(context, &deep);
            assert_eq!(result.is_ok(), accepted, "{source}: {result:?}");
        }
    }
}
