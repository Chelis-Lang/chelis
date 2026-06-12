//! Issue chelis#329 — reported as: `chelis check` emits a spurious
//! package-wide `InvalidBorrow ("borrowed arguments must be tensor or
//! tensor-carrying values (offset 0)")` when a unit contains two or
//! more structurally-distinct `grad`'d functions that each explicitly
//! `&`-borrow their differentiation target.
//!
//! Reproduction verdict (2026-06-12, this PR's investigation): the
//! filed reproducer does NOT fail on any binary the reporter could
//! have run — the official 0.7.23 release binary (byte-identical to
//! the installed toolchain), a v0.7.23-tag build, a build at the
//! Jun-6 dev head (`7af698e`, which still reports "0.7.23"), the
//! official 0.7.24 binary, and current main all check it clean, on
//! both the layered (cached-stdlib) and monolithic paths, with cold
//! and warm caches, same- and cross-module, across repeated runs and
//! an 8-variant form sweep. There is no red baseline to bisect, so
//! the "fixed as collateral of PR #363" hypothesis from the issue
//! thread is not supportable either: pre-#363 binaries already pass.
//!
//! What IS real — and reproducible on main before this change — is
//! the issue's *signature*: any single genuine `InvalidBorrow` in a
//! multi-file reef unit reports the constant `(offset 0)` (linearity
//! diagnostics read the structural span, which
//! `chelis_surf::desugar` zeroes; the real location lives in the
//! `span: "surf:a..b"` metadata) and lands in EVERY file's report
//! (unit-level errors broadcast to each per-file entry). A transient
//! genuine borrow error in a scratch unit therefore looks exactly
//! like a spurious, package-wide, unlocalizable failure — the
//! experience chelis#329 describes.
//!
//! This file pins both halves:
//!
//! - the full trigger matrix from the issue stays green (cross-module
//!   3-file reproducer, same-module form, single / mixed /
//!   identical-bodies / owned-param controls, explicit-`&` vs
//!   auto-borrow check parity AND numeric gradient agreement on the
//!   eval lane);
//! - a genuine invalid borrow still fails with `InvalidBorrow`, and
//!   after this change its message carries the real source site
//!   (`(at surf:a..b)`) instead of the unlocalizable `(offset 0)`.

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

fn reef_toml() -> String {
    format!(
        "[package]\n\
         name = \"repro\"\n\
         version = \"0.1.0\"\n\
         compiler = \"={ver}\"\n\
         module_prefix = \"Repro\"\n",
        ver = chelis_compiler_api::COMPILER_VERSION,
    )
}

/// The issue's borrow-parameter verb: separate `sig` carrying the
/// explicit `&tensor` parameter, dim- and precision-polymorphic.
const LIB: &str = "module Repro.Lib\n\
     export (nb)\n\
     sig nb: &tensor[a, p] -> tensor[a, p]\n\
     def nb(x) = relu(x)\n";

/// grad #1 — explicitly borrows the differentiation target (`nb(&v)`).
const DA: &str = "module Repro.Da\n\
     import Repro.Lib (nb)\n\
     def shim_a(v: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(nb(&v), cast(0, int32)))\n\
     def trig_a(v: tensor[3, f32]) -> tensor[3, f32] = grad(shim_a)(v)\n";

/// grad #2 — structurally distinct, also explicitly borrows.
const DB: &str = "module Repro.Db\n\
     import Repro.Lib (nb)\n\
     def shim_b(v: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(nb(&v), nb(&v)), cast(0, int32)))\n\
     def trig_b(v: tensor[3, f32]) -> tensor[3, f32] = grad(shim_b)(v)\n";

/// Run `chelis check <dir>` over a reef package and parse the
/// `{"files":[...]}` JSON. The style gate is disabled per the
/// documented integration-test convention: these fixtures synthesize
/// ad-hoc Surf to exercise linearity behavior, and the explicit `&`
/// IS the point of the fixture.
fn check_package(root: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(["check", "src/"])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("check output must be json: {e}\n{output:?}"))
}

/// Pull the entry for a single source file out of the files map.
fn file_entry<'a>(json: &'a Value, suffix: &str) -> &'a Value {
    json["files"]
        .as_array()
        .expect("files array")
        .iter()
        .find(|f| f["file"].as_str().is_some_and(|p| p.ends_with(suffix)))
        .unwrap_or_else(|| panic!("no entry for {suffix} in {json}"))
}

/// CLAUDE.md contract invariant: a perfect score and an empty error
/// vector must travel together. Assert BOTH per file.
fn assert_file_clean(json: &Value, suffix: &str) {
    let entry = file_entry(json, suffix);
    assert_eq!(
        entry["report"]["score"], 1,
        "{suffix} must check clean (score 1): {entry}"
    );
    let errors = entry["report"]["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{suffix} report has no errors array: {entry}"));
    assert!(
        errors.is_empty(),
        "{suffix} reports score 1 so its error list must be empty: {entry}"
    );
}

fn error_messages(json: &Value, suffix: &str) -> Vec<String> {
    file_entry(json, suffix)["report"]["errors"]
        .as_array()
        .map(|errs| {
            errs.iter()
                .map(|e| {
                    format!(
                        "{}: {}",
                        e["kind"].as_str().unwrap_or(""),
                        e["message"].as_str().unwrap_or("")
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

// ── The issue's headline reproducer ──────────────────────────────────

/// Cross-module form — the verbatim 3-file package from the issue
/// body. This is the variant the downstream tracker (School) left
/// unconfirmed when archiving its entry; pinning it green is the
/// upstream confirmation that closes the issue.
#[test]
fn issue_329_cross_module_two_explicit_borrow_grads_check_clean() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/lib.ch"), LIB);
    write_file(&root.join("src/da.ch"), DA);
    write_file(&root.join("src/db.ch"), DB);

    let json = check_package(root);
    for file in ["lib.ch", "da.ch", "db.ch"] {
        assert_file_clean(&json, file);
    }
}

/// Same-module form — all five defs in one file (the issue reports
/// this reproduced identically).
#[test]
fn issue_329_same_module_two_explicit_borrow_grads_check_clean() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(
        &root.join("src/all.ch"),
        "module Repro.All\n\
         sig nb: &tensor[a, p] -> tensor[a, p]\n\
         def nb(x) = relu(x)\n\
         def shim_a(v: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(nb(&v), cast(0, int32)))\n\
         def trig_a(v: tensor[3, f32]) -> tensor[3, f32] = grad(shim_a)(v)\n\
         def shim_b(v: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(nb(&v), nb(&v)), cast(0, int32)))\n\
         def trig_b(v: tensor[3, f32]) -> tensor[3, f32] = grad(shim_b)(v)\n",
    );

    let json = check_package(root);
    assert_file_clean(&json, "all.ch");
}

// ── Stays-clean controls from the issue's trigger matrix ─────────────

/// One explicit-`&` grad alone (the matrix's "1 grad" row).
#[test]
fn issue_329_single_explicit_borrow_grad_checks_clean() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/lib.ch"), LIB);
    write_file(&root.join("src/da.ch"), DA);

    let json = check_package(root);
    for file in ["lib.ch", "da.ch"] {
        assert_file_clean(&json, file);
    }
}

/// Mixed: one explicit `&`, one auto-borrow (matrix row "mixed").
#[test]
fn issue_329_mixed_explicit_and_auto_borrow_grads_check_clean() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/lib.ch"), LIB);
    write_file(&root.join("src/da.ch"), DA);
    write_file(&root.join("src/db.ch"), &DB.replace("nb(&v)", "nb(v)"));

    let json = check_package(root);
    for file in ["lib.ch", "da.ch", "db.ch"] {
        assert_file_clean(&json, file);
    }
}

/// Two grads with structurally IDENTICAL helper bodies (matrix row
/// "identical helper bodies dedup").
#[test]
fn issue_329_identical_helper_bodies_check_clean() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/lib.ch"), LIB);
    write_file(&root.join("src/da.ch"), DA);
    write_file(
        &root.join("src/db.ch"),
        &DA.replace("Repro.Da", "Repro.Db")
            .replace("shim_a", "shim_b")
            .replace("trig_a", "trig_b"),
    );

    let json = check_package(root);
    for file in ["lib.ch", "da.ch", "db.ch"] {
        assert_file_clean(&json, file);
    }
}

/// Owned-parameter verb, any count (matrix row "owned-parameter
/// verbs"): the verb takes its tensor by value, the shims call it
/// without `&`.
#[test]
fn issue_329_owned_param_verb_grads_check_clean() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(
        &root.join("src/lib.ch"),
        "module Repro.Lib\n\
         export (nb)\n\
         sig nb: tensor[a, p] -> tensor[a, p]\n\
         def nb(x) = relu(x)\n",
    );
    write_file(&root.join("src/da.ch"), &DA.replace("nb(&v)", "nb(v)"));
    write_file(&root.join("src/db.ch"), &DB.replace("nb(&v)", "nb(v)"));

    let json = check_package(root);
    for file in ["lib.ch", "da.ch", "db.ch"] {
        assert_file_clean(&json, file);
    }
}

// ── Explicit-`&` / auto-borrow equivalence (check + eval lanes) ──────

/// Both call forms must check identically clean: two explicit-`&`
/// grads in one unit, and the auto-borrow twin of the same unit.
#[test]
fn issue_329_explicit_and_auto_borrow_check_parity() {
    for explicit in [true, false] {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let (da, db) = if explicit {
            (DA.to_string(), DB.to_string())
        } else {
            (DA.replace("nb(&v)", "nb(v)"), DB.replace("nb(&v)", "nb(v)"))
        };
        write_file(&root.join("reef.toml"), &reef_toml());
        write_file(&root.join("src/lib.ch"), LIB);
        write_file(&root.join("src/da.ch"), &da);
        write_file(&root.join("src/db.ch"), &db);

        let json = check_package(root);
        for file in ["lib.ch", "da.ch", "db.ch"] {
            assert_file_clean(&json, file);
        }
    }
}

/// Two distinct explicit-`&` grads in ONE unit, evaluated: the exact
/// trigger shape from the issue must also produce correct gradients,
/// not merely check clean — and the auto-borrow twin must produce
/// the SAME values.
///
/// Analytic oracles:
///   d/dv sum(relu(v))        at [2, -1]    = [1, 0]
///   d/dv sum(relu(v)^2)      at [1, -2, 3] = 2·relu(v)·step(v) = [2, 0, 6]
const EVAL_EXPLICIT: &str = "module Repro.Issue329Eval\n\
     sig nb: &tensor[a, p] -> tensor[a, p]\n\
     def nb(x) = relu(x)\n\
     def shim_a(v: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(nb(&v), cast(0, int32)))\n\
     def shim_b(v: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(nb(&v), nb(&v)), cast(0, int32)))\n\
     out_a = grad(shim_a)(to_tensor([cast(2.0, f32), cast(-1.0, f32)]))\n\
     out_b = grad(shim_b)(to_tensor([cast(1.0, f32), cast(-2.0, f32), cast(3.0, f32)]))\n";

const GRAD_A: [f64; 2] = [1.0, 0.0];
const GRAD_B: [f64; 3] = [2.0, 0.0, 6.0];

/// Parse a printed `name = tensor(shape=[..], data=[..])` line.
fn parse_named_tensor(stdout: &str, name: &str) -> Vec<f64> {
    let prefix = format!("{name} = tensor(");
    let line = stdout
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("output does not contain `{prefix}` line:\n{stdout}"));
    let data_marker = "data=[";
    let start = line
        .find(data_marker)
        .unwrap_or_else(|| panic!("no `data=[` in line: {line}"))
        + data_marker.len();
    let end = line[start..]
        .find(']')
        .unwrap_or_else(|| panic!("no closing `]` after data: {line}"));
    line[start..start + end]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("numeric"))
        .collect()
}

fn eval_gradients(source: &str, stem: &str) -> (Vec<f64>, Vec<f64>) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{stem}: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    (
        parse_named_tensor(&stdout, "out_a"),
        parse_named_tensor(&stdout, "out_b"),
    )
}

#[test]
fn issue_329_explicit_and_auto_borrow_grads_agree_numerically() {
    let (explicit_a, explicit_b) = eval_gradients(EVAL_EXPLICIT, "issue329_explicit");
    let auto_src = EVAL_EXPLICIT.replace("nb(&v)", "nb(v)");
    let (auto_a, auto_b) = eval_gradients(&auto_src, "issue329_auto");

    assert_eq!(
        explicit_a, GRAD_A,
        "explicit-& relu grad must match the analytic step function"
    );
    assert_eq!(
        explicit_b, GRAD_B,
        "explicit-& mul-form grad must match 2·relu(v)·step(v)"
    );
    assert_eq!(
        explicit_a, auto_a,
        "explicit `&v` and auto-borrow must produce identical relu gradients"
    );
    assert_eq!(
        explicit_b, auto_b,
        "explicit `&v` and auto-borrow must produce identical mul-form gradients"
    );
}

// ── True negative parity ─────────────────────────────────────────────

/// A genuinely invalid borrow — `&` of a tensorless ADT — placed in
/// the SAME unit as two explicit-`&` grads must still fail with
/// `InvalidBorrow` (the linearity rule is not weakened), and the
/// message must carry the real source site (`(at surf:a..b)`), not
/// the unlocalizable `(offset 0)` that let chelis#329's report
/// misattribute a unit-level error to the grads.
#[test]
fn issue_329_genuine_invalid_borrow_still_fails_with_real_site() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/lib.ch"), LIB);
    write_file(&root.join("src/da.ch"), DA);
    write_file(&root.join("src/db.ch"), DB);
    write_file(
        &root.join("src/bad.ch"),
        "module Repro.Bad\n\
         type Counter =\n\
           | Counter { value: int64 }\n\
         sig peek: &Counter -> bool\n\
         def peek(c) = true\n\
         def trip(c: Counter) -> bool = peek(&c)\n",
    );

    let json = check_package(root);
    let msgs = error_messages(&json, "bad.ch");
    assert!(
        msgs.iter().any(|m| m.contains(
            "InvalidBorrow: borrowed arguments must be tensor or tensor-carrying values"
        )),
        "tensorless ADT borrow must still produce InvalidBorrow; got {msgs:?}"
    );
    let borrow_msg = msgs
        .iter()
        .find(|m| m.contains("borrowed arguments must be tensor"))
        .expect("InvalidBorrow message present");
    assert!(
        borrow_msg.contains("(at surf:"),
        "InvalidBorrow must name its real source site so a unit-level \
         error is localizable to the offending file (chelis#329's \
         misattribution vector); got: {borrow_msg}"
    );
    assert!(
        !borrow_msg.contains("(offset 0)"),
        "InvalidBorrow must not report the structural zero offset; got: {borrow_msg}"
    );
}
