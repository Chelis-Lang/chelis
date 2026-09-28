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
