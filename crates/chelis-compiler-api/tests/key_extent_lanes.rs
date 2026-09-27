//! chelis#2413: a key tensor's extent is not key material ([04-LIN-9],
//! spec/10 §3.2). A function whose shape parameter `n` is shared between a
//! key tensor and a data tensor reads the key tensor's extent at its call's
//! extent check, and `shape` and `numel` read it in source; none of those is
//! a use, so each program here runs in the DAG evaluator and in compiled C,
//! the key is consumed exactly once, and an extent that disagrees traps with
//! the typed extent error in both lanes.
//!
//! Every expected value is [05-RNG-2]'s, computed by
//! `briefs/keys-switch-10-probes/ks10_ref.py`, an independent transcription
//! of spec/05 (`split_keys(key_from_seed(1), 3)` rows `derive(derive(1, 2),
//! j)`, their unit values at element 0, and `fold_in(row, 1)`).
mod ownership_support;

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;

/// `split_keys(key_from_seed(1), 3)`.
const ROWS: &str = "tensor(shape=[3], data=[key(f2e0ed7d61bc7ab1), key(1c3be871ed9d079c), \
                    key(b2fe04bac90dd534)])";
/// `uniform_like(row, 0.0f32, 0, 1)` for each row of [`ROWS`].
const UNIFORM_ROWS: &str = "tensor(shape=[3], data=[0.120874755, 0.03584103, 0.31235975])";
/// `fold_in(row, 1)` for each row of [`ROWS`].
const FOLDED_ROWS: &str = "tensor(shape=[3], data=[key(d67784c80a3096d4), \
                           key(de79c03bbf4252e7), key(fa911785aed3b207)])";

const DATA3: &str = "to_tensor([1.0f32, 1.0f32, 1.0f32])";
const KEYS3: &str = "split_keys(key_from_seed(1i64), 3i64)";

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

/// `name = display` for every root of `main` in the DAG evaluator.
fn eval_lines(source: &str) -> Result<Vec<String>, String> {
    let result = eval_selected(request(source), &["main".into()]).map_err(|error| {
        error
            .errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    Ok(result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}",
                root.name.as_deref().expect("a named root"),
                root.display.as_deref().expect("an in-process display")
            )
        })
        .collect())
}

/// Each program prints `expected` in the evaluator and in compiled C, with
/// the ownership ledger balanced.
fn both_lanes(rows: &[(&str, String, Vec<String>)]) {
    let mut failures = Vec::new();
    for (label, source, expected) in rows {
        match eval_lines(source) {
            Ok(lines) if &lines == expected => {}
            other => failures.push(format!("eval {label}: {other:?}\n{source}")),
        }
        let generated = ownership_support::emit(source, label);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        let printed: Vec<String> = stdout.lines().map(str::to_string).collect();
        if &printed != expected {
            failures.push(format!("C {label}: {printed:?}\n{source}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn main_is(value: &str) -> Vec<String> {
    vec![format!("main = {value}")]
}

/// The reviewer's probes (round 2a, `sem/x1`, `x2`, `w2`, `w4`, `w6`, `w7`,
/// `w8`): `f[n]` shares `n` between a key tensor and a data tensor, returns
/// the keys, draws from them per row, or derives from them.
///
/// Evidentiary status: REGRESSION TEST. At `096daea8c` the evaluator failed
/// every row with "key `ks` of `f` reaches input 0 of node 2 of `f`", C
/// failed `chelis build` with "ownership lowering invariant failed" (C ran
/// `x1` alone: a lane split), and `x2`'s pass-through result was a second
/// key producer ("node 4 of `f` produces a key").
#[test]
fn a_key_tensor_sharing_its_extent_with_data_runs_in_eval_and_c() {
    let shared = "def f[n](ks: tensor[n, key], xs: tensor[n, f32])";
    let draw = |body: &str| format!("vmap(fn (j: key, v: tensor[f32]) -> {body})(ks, xs)");
    let main = |ret: &str| format!("def main() -> {ret} = f({KEYS3}, {DATA3})\n");
    let rows = vec![
        (
            "x1",
            format!(
                "{shared} -> (tensor[n, key], tensor[n, f32]) = (ks, xs)\ndef main() = f({KEYS3}, {DATA3})\n"
            ),
            vec![
                format!("main.0 = {ROWS}"),
                "main.1 = tensor(shape=[3], data=[1.0, 1.0, 1.0])".to_string(),
            ],
        ),
        (
            "x2",
            format!(
                "{shared} -> tensor[n, key] = {{\n  _ = drop(xs)\n  ks\n}}\n{}",
                main("tensor[3, key]")
            ),
            main_is(ROWS),
        ),
        (
            "w2",
            format!(
                "{shared} -> tensor[n, f32] = {}\n{}",
                draw("dropout(j, v, 0.5f32)"),
                main("tensor[3, f32]")
            ),
            main_is("tensor(shape=[3], data=[0.0, 0.0, 0.0])"),
        ),
        (
            "w4",
            format!(
                "{shared} -> tensor[n, f32] = {}\n{}",
                draw("uniform_like(j, v, 0.0f32, 1.0f32)"),
                main("tensor[3, f32]")
            ),
            main_is(UNIFORM_ROWS),
        ),
        (
            "w6",
            format!(
                "{shared} -> tensor[n, f32] = vmap(fn (v: tensor[f32], j: key) -> uniform_like(j, v, 0.0f32, 1.0f32))(xs, ks)\n{}",
                main("tensor[3, f32]")
            ),
            main_is(UNIFORM_ROWS),
        ),
        (
            "w7",
            "def f[n](ks: tensor[n, key], xs: tensor[n, 4, f32]) -> tensor[n, 4, f32] = vmap(fn (j: key, v: tensor[4, f32]) -> dropout(j, v, 0.5f32))(ks, xs)\n\
                 def main() -> tensor[2, 4, f32] = f(split_keys(key_from_seed(1i64), 2i64), to_tensor([[1.0f32, 1.0f32, 1.0f32, 1.0f32], [1.0f32, 1.0f32, 1.0f32, 1.0f32]]))\n"
                .to_string(),
            main_is("tensor(shape=[2, 4], data=[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])"),
        ),
        (
            "w8",
            format!(
                "{shared} -> tensor[n, key] = {}\n{}",
                draw("{\n  _ = drop(v)\n  fold_in(j, 1i64)\n}"),
                main("tensor[3, key]")
            ),
            main_is(FOLDED_ROWS),
        ),
    ];
    both_lanes(&rows);
}

/// `shape` and `numel` read a key tensor's extent in source and leave the
/// key live for its one use: a draw, or the result.
///
/// Evidentiary status: REGRESSION TEST. At `096daea8c` both were refused by
/// the checker ("`shape` does not admit a key-carrying operand at argument
/// 0").
#[test]
fn shape_and_numel_read_a_key_tensor_and_the_key_is_used_once_in_eval_and_c() {
    let rows = vec![
        (
            "shape_then_draw",
            format!(
                "def f(ks: tensor[3, key], xs: tensor[3, f32]) -> (i64, tensor[3, f32]) = {{\n  n = shape(ks, 0i32)\n  ys = vmap(fn (j: key, v: tensor[f32]) -> uniform_like(j, v, 0.0f32, 1.0f32))(ks, xs)\n  (n, ys)\n}}\ndef main() = f({KEYS3}, {DATA3})\n"
            ),
            vec!["main.0 = 3".to_string(), format!("main.1 = {UNIFORM_ROWS}")],
        ),
        (
            "numel_then_return",
            format!(
                "def f(ks: tensor[3, key]) -> (i64, tensor[3, key]) = {{\n  n = numel(ks)\n  (n, ks)\n}}\ndef main() = f({KEYS3})\n"
            ),
            vec!["main.0 = 3".to_string(), format!("main.1 = {ROWS}")],
        ),
    ];
    both_lanes(&rows);
}

/// Negative parity: an extent that disagrees traps with the typed extent
/// error in both lanes. The key tensor's count is a runtime value, 3, and the
/// data has 2 rows; the trap and its context line agree across the lanes.
///
/// Evidentiary status: REGRESSION TEST. At `096daea8c` the evaluator failed
/// both rows on the key rules and `chelis build` on the ownership invariant,
/// before any extent was compared.
#[test]
fn a_key_tensor_whose_extent_disagrees_traps_in_eval_and_c() {
    let main = "def main() = {\n  xs = to_tensor([1.0f32, 1.0f32])\n  c = add(shape(xs, 0i32), 1i64)\n  f(split_keys(key_from_seed(1i64), c), xs)\n}\n";
    let mut failures = Vec::new();
    for (label, f) in [
        (
            "x2",
            "def f[n](ks: tensor[n, key], xs: tensor[n, f32]) -> tensor[n, key] = {\n  _ = drop(xs)\n  ks\n}\n",
        ),
        (
            "w2",
            "def f[n](ks: tensor[n, key], xs: tensor[n, f32]) -> tensor[n, f32] = vmap(fn (j: key, v: tensor[f32]) -> dropout(j, v, 0.5f32))(ks, xs)\n",
        ),
    ] {
        let source = format!("{f}{main}");
        let evaluated = match eval_lines(&source) {
            Ok(lines) => {
                failures.push(format!("eval {label}: returned {lines:?}"));
                continue;
            }
            Err(message) => message,
        };
        let generated = ownership_support::emit(&source, label);
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        let trap = |text: &str| {
            text.lines()
                .filter(|line| {
                    line.contains("numeric trap: domain in") || line.contains("claimed =")
                })
                .map(str::trim)
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        let (eval_trap, c_trap) = (trap(&evaluated), trap(&stderr));
        if eval_trap.len() != 2 || eval_trap != c_trap {
            failures.push(format!(
                "{label}: eval {evaluated:?}, C {stderr:?}\n{source}"
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
