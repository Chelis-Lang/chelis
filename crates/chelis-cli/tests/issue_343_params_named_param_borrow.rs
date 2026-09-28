//! Issue Chelis-Lang/chelis#343: the linearity (borrow) checker appeared
//! to special-case the identifier `params`.
//!
//! An owned tensor parameter named `params`, `&`-borrowed and then moved,
//! failed `chelis check` / `chelis reef build` with
//! `borrowed arguments must be tensor or tensor-carrying values`, while the
//! byte-identical body with the parameter renamed to `p` checked clean.
//!
//! Root cause (NOT a name special-case): a typed parameter whose name
//! collides with a Deep *tag* (`params`, `let`, `var`, ...) cannot be
//! desugared to the ordinary `(name {type: T})` list form — `(params {type:
//! T})` is indistinguishable from a `params` tag list — so the surf
//! desugarer emits the caret-metadata wrapper `^{:type T} name` (a
//! `MetaExpr`) instead (`typed_param_needs_meta_wrapper`). Type inference's
//! `extract_params` already read that form, but the linearity checker's
//! `param_name_and_type` / `param_names` did not, so the parameter was
//! declared into the linear scope with NO type. A borrow of it
//! (`&params`) then failed `expr_is_owned_or_borrow_linear` and tripped the
//! spurious InvalidBorrow. The fix teaches the two linearity helpers the
//! `MetaExpr` param form, mirroring `extract_params`.
//!
//! Negative test parity: the fix must NOT blanket-accept any borrow of a
//! `params`-named binding. A `&` borrow of a genuinely non-tensor
//! `params: i32` still rejects (from the inference layer), identically to
//! a non-tensor `p`.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn error_kinds(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect()
}

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

/// The borrow-then-move shape from the issue, with the parameter named
/// `params`. Pre-fix: InvalidBorrow. Post-fix: clean.
const PARAMS_BODY: &str = "module Repro.Borrow\n\
     def take_ref[n](x: &tensor[n, f32]) -> tensor[n, f32] = copy(x)\n\
     def repro[n](params: tensor[n, f32]) -> tensor[n, f32] = {\n\
       d = take_ref(&params)\n\
       sub(params, d)\n\
     }\n";

/// Byte-identical body with `params` renamed to `p` — the issue's control.
const P_BODY: &str = "module Repro.Borrow\n\
     def take_ref[n](x: &tensor[n, f32]) -> tensor[n, f32] = copy(x)\n\
     def repro[n](p: tensor[n, f32]) -> tensor[n, f32] = {\n\
       d = take_ref(&p)\n\
       sub(p, d)\n\
     }\n";

#[test]
fn params_named_owned_tensor_borrow_then_move_checks_clean() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("params.ch");
    write_file(&path, PARAMS_BODY);
    let json = run_check(&path);
    assert!(
        error_kinds(&json).is_empty(),
        "an owned tensor param named `params`, &-borrowed then moved, must \
         check clean (not special-cased); got {:?}",
        error_messages(&json),
    );
}

#[test]
fn params_and_p_named_bodies_check_identically() {
    // The whole point of the issue: the verdict must not depend on the
    // parameter's identifier. Both bodies are byte-identical modulo the
    // name; both must be clean.
    let dir = tempdir().expect("tempdir");
    let pp = dir.path().join("params.ch");
    let qp = dir.path().join("p.ch");
    write_file(&pp, PARAMS_BODY);
    write_file(&qp, P_BODY);
    let params_errs = error_kinds(&run_check(&pp));
    let p_errs = error_kinds(&run_check(&qp));
    assert_eq!(
        params_errs, p_errs,
        "the `params`-named and `p`-named bodies must check identically; \
         params: {params_errs:?}, p: {p_errs:?}"
    );
    assert!(
        p_errs.is_empty(),
        "control (`p`) must check clean; got {p_errs:?}"
    );
}

#[test]
fn other_tag_colliding_param_names_check_clean() {
    // The fix is structural, not a `params`-only patch: a typed param named
    // for ANY Deep tag that is not also a reserved Surf keyword (`let`,
    // `var`) takes the same caret-metadata desugar form and must now check
    // clean through the same borrow-then-move shape. (`fn` / `match` are
    // reserved Surf KEYWORDS and are rejected at the parser, a separate and
    // correct gate — they are deliberately not exercised here.)
    for name in ["let", "var"] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("m.ch");
        let body = format!(
            "module Repro.Borrow\n\
             def take_ref[n](x: &tensor[n, f32]) -> tensor[n, f32] = copy(x)\n\
             def repro[n]({name}: tensor[n, f32]) -> tensor[n, f32] = {{\n\
               d = take_ref(&{name})\n\
               sub({name}, d)\n\
             }}\n"
        );
        write_file(&path, &body);
        let json = run_check(&path);
        assert!(
            error_kinds(&json).is_empty(),
            "a tag-colliding owned-tensor param named `{name}` must check \
             clean; got {:?}",
            error_messages(&json),
        );
    }
}

#[test]
fn params_named_non_tensor_borrow_still_rejects() {
    // Negative parity: the fix must not blanket-accept any borrow of a
    // `params`-named binding. A `&` borrow of a genuinely non-tensor
    // `params: i32` still rejects — the error surfaces from the
    // inference-layer borrow arm (TypeMismatch: borrow requires tensor or
    // tensor-carrying input), exactly as it does for a non-tensor `p`.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg.ch");
    write_file(
        &path,
        "module Repro.Borrow\n\
         def take_ref(x: &tensor[1, f32]) -> tensor[1, f32] = copy(x)\n\
         def repro(params: i32) -> i32 = {\n\
           d = take_ref(&params)\n\
           params\n\
         }\n",
    );
    let json = run_check(&path);
    let msgs = error_messages(&json);
    assert!(
        msgs.iter()
            .any(|m| m.contains("borrow requires tensor or tensor-carrying input")),
        "a borrow of a non-tensor `params: i32` must still reject; got {msgs:?}",
    );
}
