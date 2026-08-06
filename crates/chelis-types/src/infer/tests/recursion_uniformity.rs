//! Unit tests for the spec/04 §3.1.1 uniform-recursive-instantiation rule
//! ([04-INF-2]/[04-INF-3]): recursive binding groups must type every
//! in-group call at the caller's own instantiation. Positive/negative
//! parity per the `add-bounded-monomorphization` change (chelis#1158).

use super::*;

fn surf_errors(src: &str) -> Vec<CheckError> {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    infer_program(&exprs).errors
}

fn assert_no_polymorphic_recursion_error(errors: &[CheckError]) {
    assert!(
        !errors
            .iter()
            .any(|error| error.message.contains("polymorphic recursion")),
        "uniform recursion must not trip [04-INF-3]: {errors:?}"
    );
}

fn polymorphic_recursion_errors(errors: &[CheckError]) -> Vec<&CheckError> {
    errors
        .iter()
        .filter(|error| error.message.contains("polymorphic recursion"))
        .collect()
}

#[test]
fn direct_uniform_recursion_is_accepted() {
    let errors = surf_errors(
        "\
type Box[a] =
  | Empty
  | Full { value: a }
def depth[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 0 else depth(box, n - 1) + 1
def concrete() -> int32 = depth(Full { value: cast(7, int32) }, 3)
",
    );
    assert!(
        errors.is_empty(),
        "uniform twin must check clean: {errors:?}"
    );
}

#[test]
fn direct_polymorphic_recursion_is_rejected_with_the_atom() {
    let errors = surf_errors(
        "\
type Box[a] =
  | Empty
  | Full { value: a }
def f[a](x: a, n: int32) -> int32 =
  if n <= 0 then 0 else f(Full { value: x }, n - 1) + 1
def main() -> int32 = f(1, 3)
",
    );
    let poly = polymorphic_recursion_errors(&errors);
    assert_eq!(
        poly.len(),
        1,
        "exactly one polymorphic-recursion error expected: {errors:?}"
    );
    let error = poly[0];
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
    assert!(error.message.contains("`f`"), "must name the function");
    assert!(
        error.message.contains("Box[a]"),
        "must name the differing recursive instantiation: {}",
        error.message
    );
    assert!(
        error
            .message
            .contains("the caller's own instantiation `[a]`"),
        "must name the caller instantiation: {}",
        error.message
    );
    assert!(
        error.message.contains("[04-INF-3]"),
        "must cite the deciding atom: {}",
        error.message
    );
    assert!(error.span_offset.is_some(), "must carry a source span");
}

#[test]
fn mutual_uniform_recursion_is_accepted() {
    let errors = surf_errors(
        "\
type Box[a] =
  | Empty
  | Full { value: a }
def ping[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 0 else pong(box, n - 1) + 1
def pong[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 100 else ping(box, n - 1) + 1
def concrete() -> int32 = ping(Full { value: cast(1.0, f32) }, 4)
",
    );
    assert!(
        errors.is_empty(),
        "uniform twin must check clean: {errors:?}"
    );
}

#[test]
fn mutual_polymorphic_recursion_growing_across_the_cycle_is_rejected() {
    // The ping -> pong edge grows the instantiation (`Box[a]`); the
    // pong -> ping edge is uniform on its own. The cycle is unbounded, and
    // the growing edge is the one named.
    let errors = surf_errors(
        "\
type Box[a] =
  | Empty
  | Full { value: a }
def ping[a](x: a, n: int32) -> int32 =
  if n <= 0 then 0 else pong(Full { value: x }, n - 1) + 1
def pong[b](y: b, n: int32) -> int32 =
  if n <= 0 then 100 else ping(y, n - 1) + 1
def main() -> int32 = ping(1, 3)
",
    );
    let poly = polymorphic_recursion_errors(&errors);
    assert_eq!(
        poly.len(),
        1,
        "exactly the growing edge must be rejected: {errors:?}"
    );
    let error = poly[0];
    assert!(
        error.message.contains("`ping`") && error.message.contains("`pong`"),
        "must name caller and callee: {}",
        error.message
    );
    assert!(
        error.message.contains("[04-INF-3]"),
        "must cite the deciding atom: {}",
        error.message
    );
}

#[test]
fn polymorphic_recursion_behind_a_let_alias_is_rejected() {
    // A let-bound alias of a group member must not re-generalize the
    // member's in-group instantiation; using the alias at a grown
    // instantiation is the same [04-INF-3] violation.
    let errors = surf_errors(
        "\
type Box[a] =
  | Empty
  | Full { value: a }
def f[a](x: a, n: int32) -> int32 = {
  g = f
  if n <= 0 then 0 else g(Full { value: x }, n - 1) + 1
}
def main() -> int32 = f(1, 3)
",
    );
    let poly = polymorphic_recursion_errors(&errors);
    assert!(
        !poly.is_empty(),
        "an aliased grown recursive call must reject: {errors:?}"
    );
}

#[test]
fn uniform_recursion_behind_a_let_alias_is_accepted() {
    let errors = surf_errors(
        "\
type Box[a] =
  | Empty
  | Full { value: a }
def f[a](box: Box[a], n: int32) -> int32 = {
  g = f
  if n <= 0 then 0 else g(box, n - 1) + 1
}
def main() -> int32 = f(Full { value: cast(1, int32) }, 3)
",
    );
    assert_no_polymorphic_recursion_error(&errors);
}

#[test]
fn out_of_group_calls_at_fresh_instantiations_are_unrestricted() {
    // `main` is not a member of `depth`'s recursive group, so its call may
    // instantiate freely — twice at different types.
    let errors = surf_errors(
        "\
type Box[a] =
  | Empty
  | Full { value: a }
def depth[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 0 else depth(box, n - 1) + 1
def concrete() -> int32 =
  depth(Full { value: cast(7, int32) }, 2) + depth(Full { value: true }, 3)
",
    );
    assert!(
        errors.is_empty(),
        "out-of-group instantiations are unrestricted: {errors:?}"
    );
}
