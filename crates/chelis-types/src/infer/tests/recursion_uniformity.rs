//! Unit tests for the spec/04 §3.1.1 uniform-recursive-instantiation rule
//! ([04-INF-2]/[04-INF-3]): recursive binding groups must type every
//! in-group call at the caller's own instantiation. Positive/negative
//! parity per the `add-bounded-monomorphization` change (chelis#1158).

use super::*;

fn surf_errors(src: &str) -> Vec<CheckError> {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    infer_program(&exprs).errors
}

fn surf_ir_errors(src: &str) -> Vec<CheckError> {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    match check_ir_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
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
def depth[a](box: Box[a], n: i32) -> i32 =
  if n <= 0 then 0 else depth(box, n - 1) + 1
def concrete() -> i32 = depth(Full { value: cast(7, i32) }, 3)
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
def f[a](x: a, n: i32) -> i32 =
  if n <= 0 then 0 else f(Full { value: x }, n - 1) + 1
def main() -> i32 = f(1, 3)
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
    let source = "\
type Box[a] =
  | Empty
  | Full { value: a }
def ping[a](box: Box[a], n: i32) -> i32 =
  if n <= 0 then 0 else pong(box, n - 1) + 1
def pong[a](box: Box[a], n: i32) -> i32 =
  if n <= 0 then 100 else ping(box, n - 1) + 1
def concrete() -> i32 = ping(Full { value: cast(1.0, f32) }, 4)
";
    let errors = surf_errors(source);
    assert!(
        errors.is_empty(),
        "uniform twin must check clean: {errors:?}"
    );
    let ir_errors = surf_ir_errors(source);
    assert!(
        ir_errors.is_empty(),
        "IR recursive driver must generalize authored members after SCC exit: {ir_errors:?}"
    );
}

#[test]
fn inferred_recursive_members_generalize_after_both_driver_scc_boundaries() {
    let source = "(def {} left
            (fn {} (params {} x stop)
              (if {} (var {} stop) (var {} x)
                (app {} (var {} right) (var {} x)
                  (lit {type: (t-prim {} bool)} true)))))
         (def {} right
            (fn {} (params {} x stop)
              (if {} (var {} stop) (var {} x)
                (app {} (var {} left) (var {} x)
                  (lit {type: (t-prim {} bool)} true)))))
         (def {} int_use
            (app {} (var {} left) (lit {type: (t-prim {} i32)} 1)
              (lit {type: (t-prim {} bool)} false)))
         (def {} bool_use
            (app {} (var {} left) (lit {type: (t-prim {} bool)} true)
              (lit {type: (t-prim {} bool)} false)))";
    let exprs = chelis_deep::parser::parse_str(source).expect("Deep fixture parses");
    let inferred = infer_program(&exprs);
    assert!(
        inferred.errors.is_empty(),
        "primary driver must generalize inferred SCC members: {:?}",
        inferred.errors
    );
    check_ir_program(&exprs).expect("IR driver must generalize inferred SCC members");
}

#[test]
fn recursive_group_errors_clear_scope_before_the_next_check_in_both_drivers() {
    let broken = "\
def left(x: i32) -> i32 = right(x)
def right(x: i32) -> i32 = left(x) + missing(x)
";
    assert!(!surf_errors(broken).is_empty());
    assert_eq!(super::super::recursion::group_state_counts(), (0, 0));
    assert!(!surf_ir_errors(broken).is_empty());
    assert_eq!(super::super::recursion::group_state_counts(), (0, 0));

    let clean = "def clean(x: i32) -> i32 = x";
    assert!(surf_errors(clean).is_empty());
    assert!(surf_ir_errors(clean).is_empty());
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
def ping[a](x: a, n: i32) -> i32 =
  if n <= 0 then 0 else pong(Full { value: x }, n - 1) + 1
def pong[b](y: b, n: i32) -> i32 =
  if n <= 0 then 100 else ping(y, n - 1) + 1
def main() -> i32 = ping(1, 3)
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
def f[a](x: a, n: i32) -> i32 = {
  g = f
  if n <= 0 then 0 else g(Full { value: x }, n - 1) + 1
}
def main() -> i32 = f(1, 3)
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
def f[a](box: Box[a], n: i32) -> i32 = {
  g = f
  if n <= 0 then 0 else g(box, n - 1) + 1
}
def main() -> i32 = f(Full { value: cast(1, i32) }, 3)
",
    );
    assert_no_polymorphic_recursion_error(&errors);
}

#[test]
fn monomorphic_caller_in_group_names_the_missing_type_parameters() {
    // Red-team NIT (2026-08-06): a monomorphic group member calling an
    // authored generic at a concrete instantiation is rejected by the
    // strict authored-binder rule, and the diagnostic must not advise
    // reusing type parameters the caller does not have.
    let errors = surf_errors(
        "\
type Box[a] =
  | Empty
  | Full { value: a }
def mono(n: i32) -> i32 =
  if n <= 0 then 0 else gen(Full { value: true }, n)
def gen[a](box: Box[a], n: i32) -> i32 =
  if n <= 0 then 1 else mono(n - 1)
def main() -> i32 = mono(3)
",
    );
    let poly = polymorphic_recursion_errors(&errors);
    assert_eq!(poly.len(), 1, "exactly one rejection expected: {errors:?}");
    let error = poly[0];
    assert!(
        error.message.contains("`mono`") && error.message.contains("`gen`"),
        "must name caller and callee: {}",
        error.message
    );
    assert!(
        error.message.contains("declares no type parameters"),
        "must name the actual constraint instead of an empty instantiation: {}",
        error.message
    );
    assert!(
        !error
            .suggestions
            .iter()
            .any(|s| s.contains("reuse the caller's own type parameters")),
        "must not advise reusing parameters the caller does not have: {:?}",
        error.suggestions
    );
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
def depth[a](box: Box[a], n: i32) -> i32 =
  if n <= 0 then 0 else depth(box, n - 1) + 1
def concrete() -> i32 =
  depth(Full { value: cast(7, i32) }, 2) + depth(Full { value: true }, 3)
",
    );
    assert!(
        errors.is_empty(),
        "out-of-group instantiations are unrestricted: {errors:?}"
    );
}
