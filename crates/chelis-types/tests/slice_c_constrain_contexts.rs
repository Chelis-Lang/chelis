//! Slice C c4: the `Constrain` contexts of `spec/design/runtime_extents.md`
//! C3, over `spec/04-type-system.md` §4.7.2.
//!
//! `Constrain` applies when the context supplies an independently fixed
//! tensor rank or shape equation, and it selects the unique candidate
//! satisfying that equation. Each context below gets all three outcomes: the
//! same-rank replacement candidate selected, the rank-increasing insertion
//! candidate selected, and a shape no candidate satisfies rejected.
//!
//! Every row uses one producer, `expand(a, 0, 3i64)` on a `tensor[2, f32]`,
//! whose two legal results are `tensor[3, f32]` and `tensor[3, 2, f32]`.
//! Asserting the stamped producer type, rather than only that the program
//! checks, is what distinguishes a real selection from the freeze default
//! happening to agree with the declared shape.
//!
//! These rows are a disposition lock, not a regression test. All six pass on
//! `a5137d2c3` as well, because a concrete expected type already reached the
//! settlement transaction through `bind_tvar`. What was missing was the name
//! and the single executor, not the behaviour; these rows fix the behaviour
//! in place so a later change to the executor cannot move a context silently.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

const REPLACEMENT: &str = "(t-tensor {} (d-lit {} 3) (t-prim {} f32))";
const INSERTION: &str = "(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))";

fn check(source: &str) -> Result<String, Vec<CheckError>> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
        Ok(checked) => Ok(print_canonical(checked.annotated_exprs())),
        Err(report) => Err(report.errors),
    }
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Assert that `context` selected the same-rank replacement candidate.
fn selects_replacement(context: &str, source: &str) {
    let rendered = check(source)
        .unwrap_or_else(|errors| panic!("{context}: expected acceptance:\n{}", summary(&errors)));
    assert!(
        rendered.contains(REPLACEMENT),
        "{context}: the replacement candidate must be selected:\n{rendered}"
    );
    assert!(
        !rendered.contains(INSERTION),
        "{context}: the insertion candidate must not appear:\n{rendered}"
    );
}

/// Assert that `context` selected the rank-increasing insertion candidate.
/// This is the outcome the freeze default never produces for this producer,
/// so it is the one that proves the context supplied evidence.
fn selects_insertion(context: &str, source: &str) {
    let rendered = check(source)
        .unwrap_or_else(|errors| panic!("{context}: expected acceptance:\n{}", summary(&errors)));
    assert!(
        rendered.contains(INSERTION),
        "{context}: the insertion candidate must be selected:\n{rendered}"
    );
    // The declared or annotated type in each insertion row also renders as
    // INSERTION, so the presence check alone would pass even if the producer
    // kept the freeze default. No insertion row mentions `tensor[3, f32]`
    // anywhere, so its absence is what proves the producer moved.
    assert!(
        !rendered.contains(REPLACEMENT),
        "{context}: the producer must not keep the rank-1 freeze default \
         beside a rank-2 context:\n{rendered}"
    );
}

/// Assert that `context` rejected a shape no candidate satisfies.
fn rejects_contradiction(context: &str, source: &str) {
    let Err(errors) = check(source) else {
        panic!("{context}: a shape no candidate satisfies must be rejected");
    };
    assert!(
        errors.iter().any(|error| {
            let kind = format!("{:?}", error.kind);
            kind.contains("Dimension") || kind.contains("TypeMismatch")
        }),
        "{context}: the rejection must name the shape disagreement:\n{}",
        summary(&errors)
    );
}

#[test]
fn declared_result_constrains() {
    selects_replacement(
        "declared result",
        "def f(a: tensor[2, f32]) -> tensor[3, f32] = expand(a, 0, 3i64)\n",
    );
    selects_insertion(
        "declared result",
        "def f(a: tensor[2, f32]) -> tensor[3, 2, f32] = expand(a, 0, 3i64)\n",
    );
    rejects_contradiction(
        "declared result",
        "def f(a: tensor[2, f32]) -> tensor[9, 9, f32] = expand(a, 0, 3i64)\n",
    );
}

#[test]
fn ascription_constrains() {
    selects_replacement(
        "ascription",
        "def f(a: tensor[2, f32]) = {\n  e: tensor[3, f32] = expand(a, 0, 3i64)\n  e\n}\n",
    );
    selects_insertion(
        "ascription",
        "def f(a: tensor[2, f32]) = {\n  e: tensor[3, 2, f32] = expand(a, 0, 3i64)\n  e\n}\n",
    );
    rejects_contradiction(
        "ascription",
        "def f(a: tensor[2, f32]) = {\n  e: tensor[9, 9, f32] = expand(a, 0, 3i64)\n  e\n}\n",
    );
}

#[test]
fn instantiated_user_function_parameter_constrains() {
    selects_replacement(
        "user parameter",
        "def want(x: tensor[3, f32]) -> tensor[3, f32] = x\n\
         def f(a: tensor[2, f32]) = want(expand(a, 0, 3i64))\n",
    );
    selects_insertion(
        "user parameter",
        "def want(x: tensor[3, 2, f32]) -> tensor[3, 2, f32] = x\n\
         def f(a: tensor[2, f32]) = want(expand(a, 0, 3i64))\n",
    );
    rejects_contradiction(
        "user parameter",
        "def want(x: tensor[9, 9, f32]) -> tensor[9, 9, f32] = x\n\
         def f(a: tensor[2, f32]) = want(expand(a, 0, 3i64))\n",
    );
}

#[test]
fn instantiated_generic_field_constrains() {
    selects_replacement(
        "generic field",
        "type Box[a] = | Box { v: a }\n\
         def f(a: tensor[2, f32]) -> Box[tensor[3, f32]] = Box { v: expand(a, 0, 3i64) }\n",
    );
    selects_insertion(
        "generic field",
        "type Box[a] = | Box { v: a }\n\
         def f(a: tensor[2, f32]) -> Box[tensor[3, 2, f32]] = Box { v: expand(a, 0, 3i64) }\n",
    );
    rejects_contradiction(
        "generic field",
        "type Box[a] = | Box { v: a }\n\
         def f(a: tensor[2, f32]) -> Box[tensor[9, 9, f32]] = Box { v: expand(a, 0, 3i64) }\n",
    );
}

#[test]
fn branch_join_with_resolved_evidence_constrains() {
    selects_replacement(
        "branch join",
        "def f(flag: bool, a: tensor[2, f32], b: tensor[3, f32]) = \
         if flag then expand(a, 0, 3i64) else b\n",
    );
    selects_insertion(
        "branch join",
        "def f(flag: bool, a: tensor[2, f32], b: tensor[3, 2, f32]) = \
         if flag then expand(a, 0, 3i64) else b\n",
    );
    rejects_contradiction(
        "branch join",
        "def f(flag: bool, a: tensor[2, f32], b: tensor[9, 9, f32]) = \
         if flag then expand(a, 0, 3i64) else b\n",
    );
}

#[test]
fn builtin_relation_with_a_resolved_operand_constrains() {
    selects_replacement(
        "builtin relation",
        "def f(a: tensor[2, f32], b: tensor[3, f32]) = add(expand(a, 0, 3i64), b)\n",
    );
    selects_insertion(
        "builtin relation",
        "def f(a: tensor[2, f32], b: tensor[3, 2, f32]) = add(expand(a, 0, 3i64), b)\n",
    );
    rejects_contradiction(
        "builtin relation",
        "def f(a: tensor[2, f32], b: tensor[9, 9, f32]) = add(expand(a, 0, 3i64), b)\n",
    );
}
