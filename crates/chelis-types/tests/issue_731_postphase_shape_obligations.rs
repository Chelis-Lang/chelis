//! Post-Phase checker-totality contracts for deferred shape checks (#780)
//! and rigid declaration dimensions (#847).

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn diagnostics(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("{:?}: {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn unannotated_lambda_replays_matmul_shape_check_after_first_use() {
    let errors = diagnostics(
        r#"
def driver(good: tensor[4, 4, f32]) -> tensor[9, 9, f32] = {
  f = fn (a) -> matmul(a, good)
  f(good)
}
"#,
    );
    assert!(
        errors.iter().any(|error| {
            error
                .message
                .contains("def 'driver' body doesn't match declared signature")
                && error.message.contains("tensor[4, 4, f32]")
                && error.message.contains("tensor[9, 9, f32]")
        }),
        "the ordinary matmul rule must replay to a resolved 4x4 result, not merely give up; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn deferred_shape_lambda_is_monomorphic_after_first_use() {
    let errors = diagnostics(
        r#"
def driver(a: tensor[2, 3, f32], b: tensor[3, 4, f32], c: tensor[5, 6, f32], d: tensor[6, 7, f32]) -> tensor[2, 4, f32] = {
  f = fn (x, y) -> matmul(x, y)
  first = f(a, b)
  second = f(c, d)
  first
}
"#,
    );
    assert!(
        errors.iter().any(|error| {
            error.kind.diagnostic_name() == "DimensionMismatch"
                && error.expected.as_deref() == Some("2")
                && error.got.as_deref() == Some("5")
                && error.message.contains("`f` argument 1, axis 0")
        }),
        "the second application must conflict with the first concrete instantiation; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn unconstrained_identity_lambda_remains_polymorphic() {
    let errors = diagnostics(
        r#"
def keep_poly(x: i32) -> string = {
  identity = fn (value) -> value
  one = identity(x)
  identity("ok")
}
"#,
    );
    assert!(
        errors.is_empty(),
        "an unconstrained identity has no deferred semantic obligation: {}",
        summary(&errors)
    );
}

#[test]
fn unresolved_shape_obligation_rejects_at_declaration_boundary() {
    let errors = diagnostics(
        r#"
def unresolved(good: tensor[4, 4, f32]) = fn (a) -> matmul(a, good)
"#,
    );
    assert!(
        errors.iter().any(|error| {
            error.message
                == "unresolved `matmul` shape obligation at declaration boundary: add an outer-constructor parameter annotation or apply the lambda before the declaration boundary"
        }),
        "an unbound shape obligation must identify the exact decision boundary; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn synthesized_defsig_hole_does_not_disable_shape_obligation() {
    let errors = diagnostics(
        r#"
def helper(a) -> tensor[9, 8, f32] = matmul(a, a)
"#,
    );
    assert!(
        errors.iter().any(|error| error
            .message
            .starts_with("unresolved `matmul` shape obligation at declaration boundary")),
        "a return annotation's synthesized `(t-var _ )` parameter slot is still an unknown outer constructor; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn authored_bare_type_hole_does_not_disable_shape_obligation() {
    let errors = diagnostics(
        r#"
sig helper[t]: t -> tensor[9, 8, f32]
def helper(a) = matmul(a, a)
"#,
    );
    assert!(
        errors.iter().any(|error| error
            .message
            .starts_with("unresolved `matmul` shape obligation at declaration boundary")),
        "an authored bare type variable still leaves the operand constructor unknown; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn derived_projection_keeps_unannotated_parameter_ownership() {
    let errors = diagnostics(
        r#"
def driver(good: tensor[4, 4, f32]) -> tensor[9, 9, f32] = {
  f = fn (pair) -> matmul(pair.0, good)
  f((good, good))
}
"#,
    );
    assert!(
        errors.iter().any(|error| {
            error
                .message
                .contains("def 'driver' body doesn't match declared signature")
                && error.message.contains("tensor[4, 4, f32]")
                && error.message.contains("tensor[9, 9, f32]")
        }),
        "a projection-derived type variable must retain its lambda-parameter ownership; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn top_level_shape_helper_must_resolve_at_its_own_boundary() {
    let errors = diagnostics(
        r#"
def helper(a) = matmul(a, a)
def d1(x: tensor[4, 4, f32]) -> tensor[4, 4, f32] = helper(x)
def d2(y: tensor[7, 7, f32]) -> tensor[7, 7, f32] = helper(y)
"#,
    );
    assert!(
        errors.iter().any(|error| error
            .message
            .starts_with("unresolved `matmul` shape obligation at declaration boundary")),
        "later declarations are not binding sites for an earlier top-level declaration; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn unresolved_non_lambda_shape_value_keeps_the_existing_wildcard_contract() {
    let errors = diagnostics(
        r#"
def manual_matmul(
  a: tensor[8, 16, f32],
  b: tensor[16, 4, f32]
) -> tensor[8, 4, f32] = {
  ae = insert(a, 2, 4i64)
  be = insert(b, 0, 8i64)
  sum(mul(ae, be), 1)
}
"#,
    );
    assert!(
        errors.is_empty(),
        "[04-INF-1] defers unannotated lambda parameters, not an ordinary let-bound wildcard: {}",
        summary(&errors)
    );
}

#[test]
fn unannotated_scatter_elements_replays_dtype_checks_after_first_use() {
    let errors = diagnostics(
        r#"
def driver(
  data: tensor[2, 3, f32],
  indices: tensor[2, 2, f32],
  updates: tensor[2, 2, f32]
) -> tensor[2, 3, f32] = {
  scatter_at = fn (x, i, u) -> scatter_elements(x, i, u, 1)
  scatter_at(data, indices, updates)
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("i32") || error.message.contains("i64")),
        "deferred scatter_elements must reject floating-point indices: {}",
        summary(&errors)
    );
}

#[test]
fn consistent_deferred_scatter_elements_is_accepted() {
    let errors = diagnostics(
        r#"
def driver(
  data: tensor[2, 3, f32],
  indices: tensor[2, 2, i32],
  updates: tensor[2, 2, f32]
) -> tensor[2, 3, f32] = {
  scatter_at = fn (x, i, u) -> scatter_elements(x, i, u, 1)
  scatter_at(data, indices, updates)
}
"#,
    );
    assert!(
        errors.is_empty(),
        "valid deferred scatter_elements rejected: {}",
        summary(&errors)
    );
}

#[test]
fn generic_jacobian_wrapper_keeps_n_and_m_independent() {
    let errors = diagnostics(
        r#"
def jacobian_row[n, m](
  model: &tensor[n, f32] -> &tensor[m, f32] -> tensor[m, f32],
  theta: tensor[n, f32],
  x_data: tensor[m, f32],
  output_seed: tensor[m, f32]
) -> tensor[n, f32] = {
  target = fn (
    theta_local: tensor[n, f32],
    x_local: tensor[m, f32],
    seed_local: tensor[m, f32]
  ) -> {
    prediction = model(theta_local, x_local)
    tensor_to_scalar(sum(mul(prediction, seed_local), cast(0, i32)))
  }
  grad(target, wrt=theta_local)(theta, x_data, output_seed)
}
"#,
    );
    assert!(
        errors.is_empty(),
        "independently declared rigid dimensions must not collapse: {}",
        summary(&errors)
    );
}

#[test]
fn a_body_that_really_requires_equal_rigid_dims_still_rejects() {
    let errors = diagnostics(
        r#"
def wrong[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = y
"#,
    );
    assert!(
        errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::DimensionMismatch
                | chelis_types::errors::CheckErrorKind::TypeMismatch
        )),
        "a genuine n-versus-m contract violation must stay rejected"
    );
}
