//! chelis#1277 Slice B2h: the shared kernel decision, `host_def_kernel`,
//! answers "is this def a kernel" before any lowering runs, and its answer
//! is the one the C host program acts on.
//!
//! Each `None` below names a class the byte-identity corpus found the kernel
//! lowering rejecting AFTER a kernel decision (the C lane then fell through
//! to host code, chelis#1515; the eval lane errored). The decision now names
//! the class itself, so both lanes choose host code for the same reason and
//! no lowering error is caught to decide a lane.
//!
//! EVIDENTIARY STATUS: `a_pure_tensor_def_is_a_kernel`,
//! `an_io_effect_row_is_host` and `a_match_on_a_constructor_literal_stays_a_kernel`
//! are disposition locks (they hold on the tree without the walker too); the
//! four others are regression tests, watched returning `Err` (a failed kernel
//! lowering) on the tree without the walker and `Ok(false)` with it.

use chelis_ir::host::host_def_kernel;

fn checked_surf(src: &str) -> chelis_types::CheckedProgram {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expansion")
    .into_exprs();
    let checked = chelis_types::check_ir_program(&exprs).unwrap_or_else(|report| {
        panic!(
            "type check failed: {:?}",
            report
                .errors
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>()
        )
    });
    let checked = chelis_effects::check_program(&checked).expect("effects");
    chelis_types::check_linearity(&checked).expect("linearity")
}

fn decision(src: &str, name: &str) -> Result<bool, String> {
    let program = checked_surf(src);
    host_def_kernel(&program, name, None)
        .map(|kernel| kernel.is_some())
        .map_err(|diagnostic| diagnostic.to_string())
}

#[test]
fn a_pure_tensor_def_is_a_kernel() {
    let src = "def f(x: tensor[n, f32]) -> tensor[n, f32] = mul(x, x)\n";
    assert_eq!(decision(src, "f"), Ok(true));
}

#[test]
fn an_io_effect_row_is_host() {
    let src = "def f(x: tensor[n, f32]) -> tensor[n, f32] = {\n  _ = print(x)\n  x\n}\n";
    assert_eq!(decision(src, "f"), Ok(false));
}

/// chelis#520 D1: only a compile-time-known constructor scrutinee lowers. The
/// match sits inside a `let` block (the issue_1200 shape): a `match` at the
/// body ROOT is already on the keep-in-host list, so only the nested form
/// reaches the walker.
#[test]
fn a_match_on_a_runtime_scrutinee_is_host_before_lowering() {
    let src = "type Flag =\n  | On\n  | Off\n\
               def run(c: Flag, x: tensor[2, f32]) -> tensor[2, f32] = {\n  \
               y = add(x, x)\n  match c with {\n    | On => y\n    | Off => x\n  }\n}\n";
    assert_eq!(decision(src, "run"), Ok(false));
}

/// The positive twin: a constructor-literal scrutinee inside the same block
/// stays a kernel, so the class does not swallow the static matches the
/// lowering does select.
#[test]
fn a_match_on_a_constructor_literal_stays_a_kernel() {
    let src = "type Flag =\n  | On\n  | Off\n\
               def run(x: tensor[2, f32]) -> tensor[2, f32] = {\n  \
               y = add(x, x)\n  match On with {\n    | On => y\n    | Off => x\n  }\n}\n";
    assert_eq!(decision(src, "run"), Ok(true));
}

/// A host-only builtin reached through a `let` inside the body (the
/// issue_1222 shape the corpus found).
#[test]
fn a_host_only_builtin_inside_the_body_is_host_before_lowering() {
    let src = "def f(x: tensor[2, f32]) -> tensor[2, f32] = {\n  \
               z = [7i64]\n  m = len(z)\n  if (m > 0i64) then x else x\n}\n";
    assert_eq!(decision(src, "f"), Ok(false));
}
