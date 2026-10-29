//! Every public checker entry accepts Surf pipes already normalized to applications.
//! Normalization belongs to desugaring; checking leaves its input unchanged.

use chelis_checker_core::infer::SignatureInferenceMetadata;
use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::TypeEnv;
use chelis_types::infer::CheckedProgram;
use chelis_types::infer::build_compiled_library_context;
use chelis_types::infer::build_compiled_library_context_with_base;
use chelis_types::infer::build_type_env_from_library;
use chelis_types::infer::check_ir_program;
use chelis_types::infer::check_ir_with_context;
use chelis_types::infer::check_ir_with_signature_context;
use chelis_types::infer::check_typed_program;
use chelis_types::infer::infer_ir_program;
use chelis_types::infer::infer_program;

/// A program whose body is a pipe chain with both stage shapes: a bare-name
/// stage (`to_tensor`) and a call stage the desugarer wraps in a synthesized
/// first-argument application (`sum(..)`).
const PIPED: &str = "def f() -> tensor[f32] = \
                     [1.0f32, 2.0f32] |> to_tensor |> sum(cast(0, i32))\n";

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source, &chelis_syntax_host::NativeStack).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

/// Does this expression, or anything under it, carry a `pipe` node?
fn contains_pipe(expr: &Expr) -> bool {
    match expr {
        Expr::Node(node, _) => {
            node.tag().as_str() == "pipe" || node.children_slice().iter().any(contains_pipe)
        }
        Expr::MetaExpr(meta, _) => contains_pipe(&meta.expr),
        Expr::BareList(elems, _) => elems.iter().any(contains_pipe),
        Expr::UnknownForm(data) => data.head == "pipe" || data.children.iter().any(contains_pipe),
        Expr::Map(map, _) => map
            .find_expression(|v| if contains_pipe(v) { Some(()) } else { None })
            .is_some(),
        Expr::Atom(_, _) => false,
    }
}

fn program_contains_pipe(exprs: &[Expr]) -> bool {
    exprs.iter().any(contains_pipe)
}

/// Checking leaves the already-normalized caller tree unchanged.
#[test]
fn the_fold_never_mutates_the_callers_program() {
    let deep = surf_to_deep(PIPED);
    assert!(
        !program_contains_pipe(&deep),
        "desugaring must erase pipes before every checker entry"
    );
    let before = chelis_deep::printer::print_canonical(&deep);
    let checked = check_ir_program(&deep).expect("the piped program checks");
    assert_eq!(chelis_deep::printer::print_canonical(&deep), before);
    assert!(
        !program_contains_pipe(checked.annotated_exprs()),
        "and the checker's output carries the application instead"
    );
}

/// Every public checker entry accepts the same normalized pipe program.
#[test]
fn every_public_checker_entry_folds_its_input() {
    let deep = surf_to_deep(PIPED);
    let empty = TypeEnv::empty();

    let assert_folded = |label: &str, checked: &CheckedProgram| {
        assert!(
            !program_contains_pipe(checked.annotated_exprs()),
            "{label} left a pipe in the checked program"
        );
    };

    assert_folded(
        "check_ir_program",
        &check_ir_program(&deep).expect("check_ir_program"),
    );
    assert_folded(
        "check_ir_with_context",
        &check_ir_with_context(&empty, &deep).expect("check_ir_with_context"),
    );
    assert_folded(
        "check_ir_with_signature_context",
        &check_ir_with_signature_context(&empty, &SignatureInferenceMetadata::default(), &deep)
            .expect("check_ir_with_signature_context"),
    );
    assert_folded(
        "check_typed_program",
        &check_typed_program(&deep).expect("check_typed_program"),
    );
    let (_, library) =
        build_compiled_library_context(&deep).expect("build_compiled_library_context");
    assert_folded("build_compiled_library_context", &library);
    let (_, layered) = build_compiled_library_context_with_base(&empty, &deep)
        .expect("build_compiled_library_context_with_base");
    assert_folded("build_compiled_library_context_with_base", &layered);

    build_type_env_from_library(&deep).expect("build_type_env_from_library accepts the pipe");
    assert!(
        infer_program(&deep).errors.is_empty(),
        "infer_program accepts the pipe"
    );
    assert!(
        infer_ir_program(&deep).errors.is_empty(),
        "infer_ir_program accepts the pipe"
    );
}
