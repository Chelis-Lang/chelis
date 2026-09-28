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
