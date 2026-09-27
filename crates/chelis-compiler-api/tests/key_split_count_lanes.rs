//! chelis#2413, [05-OP-71]: `split_keys(k, n)`'s result extent follows
//! spec/04 §4.7.2's rule for `expand` and `insert`. A literal count gives a
//! literal extent, so a mismatch the checker can see is a check error; any
//! other count gives a fresh runtime extent, which the split checks, before
//! any key exists, against the extent it meets, so a disagreeing runtime
//! count traps `Domain` in the DAG evaluator and in compiled C alike.
//!
//! Every expected value is [05-RNG-2]'s, computed by
//! `briefs/keys-switch-10-probes/ks10_ref.py`, an independent transcription
//! of spec/05: the rows of `split_keys(key_from_seed(1), 3)` and the unit
//! values of their first elements.
mod ownership_support;

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;

/// `uniform_like(row, 0.0f32, 0, 1)` for each row of
/// `split_keys(key_from_seed(1), 3)`.
const UNIFORM_ROWS: &str = "main = tensor(shape=[3], data=[0.120874755, 0.03584103, 0.31235975])";

/// The draw every row of these programs makes.
const DRAW: &str = "vmap(fn (j: key, v: tensor[f32]) -> uniform_like(j, v, 0.0f32, 1.0f32))";
const DATA3: &str = "to_tensor([0.0f32, 0.0f32, 0.0f32])";

/// `name = display` for every root of `main` in the DAG evaluator, or the
/// evaluator's diagnostics.
fn eval_lines(source: &str) -> Result<Vec<String>, String> {
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &["main".into()],
    )
    .map_err(|error| {
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

/// A runtime count sizes its keys for the draw that meets them: the block
/// form (`s3`), a count read from the data's shape (`v2`) and a count
/// passed in (`v3`, either argument order) run, and eval equals C equals the
/// reference.
///
/// Evidentiary status: REGRESSION TEST. At `096daea8c` `s3` evaluated
/// nothing, `v2` evaluated and failed `chelis build` ("random operation
/// requires a key batch matching its data's leading axes"), and `v3` failed
/// both lanes with that error.
#[test]
fn a_split_keys_count_sizes_its_keys_in_eval_and_c() {
    let rows = [
        (
            "s3",
            "def main() = {\n  ks = split_keys(key_from_seed(1i64), 3i64)\n  vmap(fn (j: key) -> uniform_like(j, to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32))(ks)\n}\n".to_string(),
            "main = tensor(shape=[3, 2], data=[0.120874755, 0.632571, 0.03584103, 0.0585877, 0.31235975, 0.53761417])",
        ),
        (
            "v2",
            format!(
                "def f[n](k: key, xs: tensor[n, f32]) -> tensor[n, f32] = {DRAW}(split_keys(k, shape(xs, 0i32)), xs)\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), {DATA3})\n"
            ),
            UNIFORM_ROWS,
        ),
        (
            "v3",
            format!(
                "def f[n](k: key, c: i64, xs: tensor[n, f32]) -> tensor[n, f32] = {DRAW}(split_keys(k, c), xs)\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), 3i64, {DATA3})\n"
            ),
            UNIFORM_ROWS,
        ),
        (
            "v3_data_first",
            format!(
                "def f[n](k: key, c: i64, xs: tensor[n, f32]) -> tensor[n, f32] = vmap(fn (v: tensor[f32], j: key) -> uniform_like(j, v, 0.0f32, 1.0f32))(xs, split_keys(k, c))\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), 3i64, {DATA3})\n"
            ),
            UNIFORM_ROWS,
        ),
    ];
    let mut failures = Vec::new();
    for (label, source, expected) in &rows {
        match eval_lines(source) {
            Ok(lines) if lines == [*expected] => {}
            other => failures.push(format!("eval {label}: {other:?}\n{source}")),
        }
        let generated = ownership_support::emit(source, label);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        if stdout.lines().collect::<Vec<_>>() != [*expected] {
            failures.push(format!("C {label}: {stdout:?}\n{source}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Negative parity: a runtime count that disagrees with the extent it meets
/// traps with the typed `Domain` error, the same two lines in both lanes,
/// whether the keys come first or second.
///
/// Evidentiary status: REGRESSION TEST. At `096daea8c` both rows failed the
/// evaluator's key rules and `chelis build`'s ownership invariant ("random
/// operation requires a key batch matching its data's leading axes") before
/// any count was compared.
#[test]
fn a_runtime_count_that_disagrees_traps_domain_in_eval_and_c() {
    let trap = "numeric trap: domain in split_keys at i64";
    let rows = [
        (
            "keys_first",
            format!(
                "def f[n](k: key, c: i64, xs: tensor[n, f32]) -> tensor[n, f32] = {DRAW}(split_keys(k, c), xs)\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), 2i64, {DATA3})\n"
            ),
            "extent `3`: claimed = 3, split_keys axis 0 = 2",
        ),
        (
            "data_first",
            format!(
                "def f[n](k: key, c: i64, xs: tensor[n, f32]) -> tensor[n, f32] = vmap(fn (v: tensor[f32], j: key) -> uniform_like(j, v, 0.0f32, 1.0f32))(xs, split_keys(k, c))\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), 4i64, {DATA3})\n"
            ),
            "extent `3`: claimed = 3, split_keys axis 0 = 4",
        ),
    ];
    traps_in_both_lanes(trap, &rows);
}

/// Each row traps with `trap` after its `claimed =` context line, the same
/// two lines in the DAG evaluator and in compiled C.
fn traps_in_both_lanes(trap: &str, rows: &[(&str, String, &str)]) {
    let mut failures = Vec::new();
    for (label, source, context) in rows {
        let expected = [context.to_string(), trap.to_string()];
        let lines = |text: &str| {
            text.lines()
                .map(str::trim)
                .filter(|line| line.contains("claimed =") || line.contains("numeric trap"))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        match eval_lines(source) {
            Err(message) if lines(&message) == expected => {}
            other => failures.push(format!("eval {label}: {other:?}\n{source}")),
        }
        let generated = ownership_support::emit(source, label);
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        if lines(&stderr) != expected {
            failures.push(format!("C {label}: {stderr:?}\n{source}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// [04-LIN-9], spec/10 §3.2: a key tensor's extent is not key material. A
/// split bound under a claim on its count axis (`ks: tensor[n, key]`) is the
/// extent every later node of that extent observes, and a function result
/// claim may rest on the split's count axis as on an expansion's size. The
/// keys are returned (`a4`) or drawn from (`a5`, `v6`), and eval equals C
/// equals the reference, with a balanced ownership ledger.
///
/// Evidentiary status: REGRESSION TEST. At `96f5a4f9d` all three rows
/// evaluated and failed `chelis build`: `a4` with "result claim at node 6
/// requires an earlier witness and a supported producing axis", `a5` and
/// `v6` with "node 9 takes the `split_keys` key at node 5 as a dependency".
#[test]
fn a_split_under_an_extent_claim_runs_in_eval_and_c() {
    let bound = "  ks: tensor[n, key] = split_keys(k, shape(xs, 0i32))\n";
    let rows = [
        (
            "a4",
            format!(
                "def f[n](k: key, xs: tensor[n, f32]) -> tensor[n, key] = {{\n{bound}  ks\n}}\ndef main() -> tensor[3, key] = f(key_from_seed(1i64), {DATA3})\n"
            ),
            "main = tensor(shape=[3], data=[key(f2e0ed7d61bc7ab1), key(1c3be871ed9d079c), key(b2fe04bac90dd534)])",
        ),
        (
            "a5",
            format!(
                "def f[n](k: key, xs: tensor[n, f32]) -> tensor[n, f32] = {{\n{bound}  {DRAW}(ks, xs)\n}}\ndef main() -> tensor[2, f32] = f(key_from_seed(1i64), to_tensor([0.0f32, 0.0f32]))\n"
            ),
            "main = tensor(shape=[2], data=[0.120874755, 0.03584103])",
        ),
        (
            "v6",
            format!(
                "def f[n](k: key, xs: tensor[n, f32]) -> tensor[n, f32] = {{\n{bound}  {DRAW}(ks, xs)\n}}\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), {DATA3})\n"
            ),
            UNIFORM_ROWS,
        ),
    ];
    let mut failures = Vec::new();
    for (label, source, expected) in &rows {
        match eval_lines(source) {
            Ok(lines) if lines == [*expected] => {}
            other => failures.push(format!("eval {label}: {other:?}\n{source}")),
        }
        let generated = ownership_support::emit(source, label);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        if stdout.lines().collect::<Vec<_>>() != [*expected] {
            failures.push(format!("C {label}: {stdout:?}\n{source}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Negative parity: a runtime count that disagrees with the claim on the
/// split's count axis, from a binding's ascription (drawn from or returned)
/// or from the function's result, traps the typed `Domain` error in both
/// lanes before any key exists.
///
/// Evidentiary status: REGRESSION TEST. At `96f5a4f9d` `chelis build`
/// refused all three rows before any count was compared:
/// `ascribed_and_drawn` with "node 9 takes the `split_keys` key at node 5 as
/// a dependency" (and the evaluator with the same key rule), the other two
/// with "result claim at node ... requires an earlier witness and a supported
/// producing axis".
#[test]
fn a_split_count_that_disagrees_with_its_extent_claim_traps_domain_in_eval_and_c() {
    let rows = [
        (
            "ascribed_and_drawn",
            format!(
                "def f[n](k: key, c: i64, xs: tensor[n, f32]) -> tensor[n, f32] = {{\n  ks: tensor[n, key] = split_keys(k, c)\n  {DRAW}(ks, xs)\n}}\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), 2i64, {DATA3})\n"
            ),
            "extent `3`: claimed = 3, split_keys axis 0 = 2",
        ),
        (
            "claimed_and_returned",
            format!(
                "def f[n](k: key, c: i64, xs: tensor[n, f32]) -> tensor[n, key] = {{\n  ks: tensor[n, key] = split_keys(k, c)\n  ks\n}}\ndef main() -> tensor[3, key] = f(key_from_seed(1i64), 4i64, {DATA3})\n"
            ),
            "extent `3`: claimed = 3, split_keys axis 0 = 4",
        ),
        (
            "result_claim",
            format!(
                "def f[n](k: key, c: i64, xs: tensor[n, f32]) -> tensor[n, key] = split_keys(k, c)\ndef main() -> tensor[3, key] = f(key_from_seed(1i64), 4i64, {DATA3})\n"
            ),
            "extent `3`: claimed = 3, split_keys axis 0 = 4",
        ),
    ];
    traps_in_both_lanes("numeric trap: domain in split_keys at i64", &rows);
}

/// Observing a split's extent is not a use, so the keys are still consumed
/// exactly once: a second draw from `ks` is [04-LIN-9]'s reuse error.
///
/// Evidentiary status: LOCK (refused the same way at `96f5a4f9d`).
#[test]
fn a_split_under_an_extent_claim_is_still_consumed_once() {
    let source = format!(
        "def f[n](k: key, xs: tensor[n, f32]) -> tensor[n, f32] = {{\n  ks: tensor[n, key] = split_keys(k, shape(xs, 0i32))\n  a = {DRAW}(ks, xs)\n  b = {DRAW}(ks, xs)\n  add(a, b)\n}}\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), {DATA3})\n"
    );
    let error = eval_lines(&source).expect_err("a second draw from `ks` is a reuse");
    assert!(
        error.contains("key-carrying variable `ks` was already consumed")
            && error.contains("[04-LIN-9]"),
        "{error}"
    );
}

/// A literal count is a literal extent, so a mismatch the checker can see is
/// a check error, as it is for `expand` ([04-INF-6] for a rigid binder), and
/// a static negative count is a type error.
///
/// Evidentiary status: REGRESSION TEST. At `096daea8c` `sk2` and `sk6`
/// checked and then failed on key or ownership errors, `sk7` checked and
/// failed, and the negative count checked.
#[test]
fn a_literal_count_is_its_static_extent() {
    let rows = [
        (
            "sk2",
            format!(
                "def main() -> tensor[3, f32] = {DRAW}(split_keys(key_from_seed(1i64), 2i64), {DATA3})\n"
            ),
            "dimension mismatch: Lit(2) vs Lit(3)",
        ),
        (
            "sk6",
            format!(
                "def f[n](k: key, xs: tensor[n, f32]) -> tensor[n, f32] = {DRAW}(split_keys(k, 2i64), xs)\ndef main() -> tensor[3, f32] = f(key_from_seed(1i64), {DATA3})\n"
            ),
            "polymorphic dim parameter `n` forced to concrete Lit(2)",
        ),
        (
            "sk7",
            format!(
                "def f[n](k: key, xs: tensor[n, f32]) -> tensor[n, f32] = {DRAW}(split_keys(k, 2i64), xs)\ndef main() -> tensor[2, f32] = f(key_from_seed(1i64), to_tensor([0.0f32, 0.0f32]))\n"
            ),
            "polymorphic dim parameter `n` forced to concrete Lit(2)",
        ),
        (
            "ascribed",
            "def main() = {\n  ks: tensor[3, key] = split_keys(key_from_seed(1i64), 2i64)\n  ks\n}\n"
                .to_string(),
            "dimension mismatch: Lit(2) vs Lit(3)",
        ),
        (
            "negative",
            "def main() -> tensor[2, key] = split_keys(key_from_seed(1i64), -2i64)\n".to_string(),
            "split_keys requires a non-negative count, got -2 ([05-OP-71])",
        ),
    ];
    let mut failures = Vec::new();
    for (label, source, fragment) in &rows {
        match eval_lines(source) {
            Err(message) if message.contains(fragment) => {}
            other => failures.push(format!("{label}: {other:?}\n{source}")),
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
