//! Every public checker entry accepts Surf pipes already normalized to applications.
//! Normalization belongs to desugaring; checking leaves its input unchanged.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::TypeEnv;
use chelis_types::infer::{
    CheckedProgram, SignatureInferenceMetadata, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program, infer_ir_program,
    infer_program,
};

/// A program whose body is a pipe chain with both stage shapes: a bare-name
/// stage (`to_tensor`) and a call stage the desugarer wraps in a synthesized
/// first-argument application (`sum(..)`).
const PIPED: &str = "def f() -> tensor[f32] = \
                     [1.0f32, 2.0f32] |> to_tensor |> sum(cast(0, i32))\n";

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
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

/// Enumerate public checker entries so the behavioral controls below stay complete.
#[test]
fn the_public_checker_entries_are_the_ones_this_file_covers() {
    let source =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/infer/program.rs"))
            .expect("program.rs is readable");

    let mut found: Vec<String> = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("pub fn ") else {
            continue;
        };
        let name = rest.split('(').next().unwrap_or_default().to_string();
        // A checker entry takes a Deep program. Read the signature up to its
        // closing paren; the entries are all short.
        let signature: String = lines[index..lines.len().min(index + 8)].join(" ");
        let signature = signature.split(')').next().unwrap_or_default().to_string();
        if signature.contains("&[deep::Expr]") {
            found.push(name);
        }
    }
    found.sort();

    let expected = vec![
        "build_compiled_library_context".to_string(),
        "build_compiled_library_context_with_base".to_string(),
        "build_type_env_from_library".to_string(),
        "check_ir_program".to_string(),
        "check_ir_with_context".to_string(),
        "check_ir_with_signature_context".to_string(),
        "check_typed_program".to_string(),
        "infer_ir_program".to_string(),
        "infer_program".to_string(),
    ];
    assert_eq!(
        found, expected,
        "a public checker entry was added or removed; add its normalized pipe control"
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
