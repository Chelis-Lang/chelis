//! Where the pipe fold runs, and what it is allowed to touch.
//!
//! `spec/02-surf-syntax.md` section 0.1 says `x |> f(y)` MEANS `f(x, y)`.
//! `chelis_deep::pipe::fold_pipe` states that once, over the checker's input,
//! and every pass downstream of the checker lost its own pipe arm as a
//! result (chelis#1923, chelis#1791). Two properties have to hold for that
//! placement to be safe, and neither is visible from the rows the fold
//! repairs, so they are locked here.
//!
//! 1. The fold produces a NEW tree. The desugarer's output is the input to
//!    more than the checker: `chelis deep` prints it, the resugaring laws in
//!    section 0.1 are stated over it, and a caller may check one program and
//!    then do something else with the same expressions. A fold that mutated
//!    its caller's program in place would silently change all of that.
//!
//! 2. EVERY public checker entry folds. A checked program's annotated
//!    expressions are the checker's output and what the lowerer, linearity,
//!    the effect pass and the caches all read; a pipe surviving into one is
//!    now a fail-closed error in each of those passes. An entry added later
//!    that does not fold would turn a working program into that error, so
//!    the entry set itself is enumerated rather than sampled.

use chelis_deep::{DeepTag, Expr};

use chelis_types::TypeEnv;
use chelis_types::infer::{
    CheckedProgram, SignatureInferenceMetadata, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program, infer_ir_program,
    infer_program,
};

/// A program whose body is a pipe chain with both stage shapes: a bare-name
/// stage (`to_tensor`) and a call stage the desugarer wraps in a synthesized
/// unary lambda (`sum(..)`).
const PIPED: &str = include_str!("fixtures/core/tests/issue_1923_pipe_fold_placement/PIPED_1.json");

fn core_program(source: &str) -> Vec<chelis_deep::Expr> {
    serde_json::from_str(source).expect("lossless core fixture")
}

/// Does this expression, or anything under it, carry a `pipe` node?
fn contains_pipe(expr: &Expr) -> bool {
    match expr {
        Expr::List(list, _) => {
            list.tag() == Some(DeepTag::Pipe) || list.elements.iter().any(contains_pipe)
        }
        Expr::Node(node, _) => {
            node.tag() == DeepTag::Pipe || node.children_slice().iter().any(contains_pipe)
        }
        Expr::MetaExpr(meta, _) => contains_pipe(&meta.expr),
        Expr::BareList(elems, _) => elems.iter().any(contains_pipe),
        Expr::UnknownForm(data) => data.children.iter().any(contains_pipe),
        Expr::Map(map, _) => map
            .find_expression(|v| if contains_pipe(v) { Some(()) } else { None })
            .is_some(),
        Expr::Atom(_, _) => false,
    }
}

fn program_contains_pipe(exprs: &[Expr]) -> bool {
    exprs.iter().any(contains_pipe)
}

/// Property 1: checking a piped program leaves the caller's program alone.
///
/// The assertion is on the SAME slice that was handed to the checker, after
/// the call returns. `chelis deep` printing the pipe and `chelis surf`
/// reprinting `|>` are the user-visible face of this, locked in
/// `crates/chelis-cli/tests/issue_1923_pipe_fold_surface.rs`.
///
/// EVIDENTIARY STATUS: disposition lock. `fold_pipe` was written to return a
/// new tree, so this has never failed; it is here because the in-place
/// variant is the cheaper implementation and nothing else would catch it.
#[test]
fn the_fold_never_mutates_the_callers_program() {
    let deep = core_program(PIPED);
    assert!(
        program_contains_pipe(&deep),
        "the fixture must actually carry a pipe before the check"
    );
    let checked = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("the piped program checks");
    assert!(
        program_contains_pipe(&deep),
        "the caller's desugared program still carries its pipe after checking"
    );
    assert!(
        !program_contains_pipe(checked.annotated_exprs()),
        "and the checker's output carries the application instead"
    );
}

/// Property 2b: each of those entries folds.
///
/// The six that return a `CheckedProgram` are asserted on their annotated
/// output directly. The three that do not still have to accept the program
/// without error, which is the observable they own: an unfolded pipe reaching
/// inference is a `MalformedForm` diagnostic now, so a missed fold cannot
/// pass silently there either.
///
/// EVIDENTIARY STATUS: regression test. Before this change the two
/// library-compile entries annotated their input directly rather than through
/// a checked-program call and did NOT fold, so their annotated output kept
/// the pipe.
#[test]
fn every_public_checker_entry_folds_its_input() {
    let deep = core_program(PIPED);
    let empty = TypeEnv::empty();

    let assert_folded = |label: &str, checked: &CheckedProgram| {
        assert!(
            !program_contains_pipe(checked.annotated_exprs()),
            "{label} left a pipe in the checked program"
        );
    };

    assert_folded(
        "check_ir_program",
        &check_ir_program(
            &deep,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("check_ir_program"),
    );
    assert_folded(
        "check_ir_with_context",
        &check_ir_with_context(
            &empty,
            &deep,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("check_ir_with_context"),
    );
    assert_folded(
        "check_ir_with_signature_context",
        &check_ir_with_signature_context(
            &empty,
            &SignatureInferenceMetadata::default(),
            &deep,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("check_ir_with_signature_context"),
    );
    assert_folded(
        "check_typed_program",
        &check_typed_program(
            &deep,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("check_typed_program"),
    );
    let (_, library) = build_compiled_library_context(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("build_compiled_library_context");
    assert_folded("build_compiled_library_context", &library);
    let (_, layered) = build_compiled_library_context_with_base(
        &empty,
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("build_compiled_library_context_with_base");
    assert_folded("build_compiled_library_context_with_base", &layered);

    build_type_env_from_library(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("build_type_env_from_library accepts the pipe");
    assert!(
        infer_program(
            &deep,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None)
        )
        .errors
        .is_empty(),
        "infer_program accepts the pipe"
    );
    assert!(
        infer_ir_program(
            &deep,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None)
        )
        .errors
        .is_empty(),
        "infer_ir_program accepts the pipe"
    );
}
