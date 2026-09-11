//! chelis#709 - `with seed` / `with device` bodies are not type-checked
//! (no `handle-effect` case in infer.rs), plus the two executed escalations
//! from the 2026-07-16 sweep and the #721 round-trip bug found alongside.
//!
//! ## The scoping, confirmed by execution
//!
//! A wrapper battery (locked below) shows the hole is EXACTLY `with seed` /
//! `with device` bodies: the same ill-typed expression is caught inside
//! let/if/match/lambda/pipe/tuple/list/grad/vmap/jit, and the seed/device
//! HANDLER expressions are checked too. #709's blast-radius claim holds.
//!
//! ## The escalations (fixed by chelis#731 Phase 1)
//!
//! * The masked error used to reach a RUNNABLE binary: `with seed(42i64) {
//!   cast(5, int64) }` from an `-> f32` fn passed check (score 1), built, ran,
//!   and printed `5` in BOTH lanes. The handle-effect checker case now returns
//!   the body's type, so the declared return type is enforced and the build
//!   rejects it with a type diagnostic (`masked_return_type_violation...`).
//! * `lower_handle_effect`'s catch-all was live via `.dp`: rewriting
//!   `effect: random` to a bogus kind still checked at score 1 and built a
//!   working binary - the unknown kind and its handler silently dropped
//!   (the #703 shape). The checker case now string-matches the two known kinds
//!   with a loud `MalformedForm` else, so a bogus kind is rejected at check
//!   (`unknown_effect_kind_is_rejected`).
//!
//! ## chelis#721 (found while probing the controls; fixed, locked below)
//!
//! `chelis eval` could not ingest the canonical Deep of a NULLARY fn. The
//! real chain (verified against origin/main): a nullary def whose body is
//! DAG-lowerable folds to a constant that lands in `tensor_bindings`, whose
//! precedence in `eval_var` returns that Tensor BEFORE `resolve_top_level`,
//! so `(app (var f))` applied the folded scalar and died `value is not
//! callable`, while check scored 1 and the compiled lane ran the same file.
//! Fixed in `eval_app`: an application whose callee names a `(fn …)`-bodied
//! top-level def (and is not a local binding) resolves the def directly to
//! its Closure, bypassing only the tensor_bindings shadow. Unary defs always
//! round-tripped (locked as the control).

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const MASKED_ERROR: &str = "add(cast(1.0, f32), cast(2, int64))";

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `chelis check` score for a program written to a file with the given
/// extension (".ch" or ".dp").
fn check_score(program: &str, ext: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("check must emit JSON: {e}"));
    parsed["score"].as_f64().expect("numeric score")
}

/// `chelis deep` output for a surf program.
fn deep_of(program: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", path.to_str().unwrap()])
        .output()
        .expect("chelis deep should run");
    assert!(out.status.success(), "chelis deep must succeed");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Run `chelis eval --file` on a program with the given extension.
fn eval_with_ext(program: &str, ext: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Build to C and (if it builds) link + run. Distinguishes build rejection
/// from a runnable binary, which is what the escalation rows assert on.
enum CLane {
    Rejected(String),
    Ran(String),
}

fn c_lane_outcome(program: &str, ext: &str, name: &str) -> CLane {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}{ext}"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    if !built.status.success() {
        return CLane::Rejected(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return CLane::Rejected(format!("emitted C failed to compile: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    CLane::Ran(String::from_utf8_lossy(&run.stdout).into_owned())
}

// ===========================================================================
// CONTROLS: the hole is exactly with seed / with device. Everything else
// catches the error, including the handler expressions.
// ===========================================================================

/// Every non-handle-effect wrapper catches the masked error. If any row here
/// starts scoring 1, a new #709-class hole has opened.
#[test]
fn wrapper_constructs_catch_the_masked_error() {
    let cases: &[(&str, String)] = &[
        ("bare", format!("def f() -> f32 = {MASKED_ERROR}\n")),
        (
            "let_body",
            format!("def f() -> f32 = let x = {MASKED_ERROR} in x\n"),
        ),
        (
            "if_then",
            format!("def f(c: bool) -> f32 = if c then {MASKED_ERROR} else 1.0\n"),
        ),
        (
            "lambda_body",
            format!("def f() -> f32 = (fn (v: f32) -> f32 = {MASKED_ERROR})(1.0)\n"),
        ),
        (
            "pipe_stage",
            format!("def f() -> f32 = 1.0 |> fn (v: f32) -> f32 = {MASKED_ERROR}\n"),
        ),
        (
            "tuple_elem",
            format!("def f() -> (f32, f32) = ({MASKED_ERROR}, 1.0)\n"),
        ),
        (
            "list_elem",
            format!("def f() -> [f32] = [{MASKED_ERROR}, 1.0]\n"),
        ),
        (
            "match_arm",
            format!("def f(c: bool) -> f32 = match c {{ true => {MASKED_ERROR}, false => 1.0 }}\n"),
        ),
        (
            "grad_callee",
            format!("def g(x: f32) -> f32 = {MASKED_ERROR}\ndef f(x: f32) -> f32 = grad(g)(x)\n"),
        ),
        (
            "vmap_lambda",
            format!(
                "def f(t: tensor[4, f32]) -> tensor[4, f32] = \
                 vmap(fn (v: f32) -> f32 = {MASKED_ERROR})(t)\n"
            ),
        ),
        (
            "jit_callee",
            format!("def g(x: f32) -> f32 = {MASKED_ERROR}\ndef f(x: f32) -> f32 = jit(g)(x)\n"),
        ),
    ];
    for (name, program) in cases {
        let score = check_score(program, ".ch");
        assert!(
            score < 1.0,
            "{name}: the masked error must be caught (score < 1), got score {score}"
        );
    }
}

/// The HANDLER expressions of with seed / with device ARE checked - the
/// hole is only the body. Bounds #709 from the other side.
#[test]
fn handler_expressions_are_checked() {
    let score = check_score(
        "def f() -> f32 = with seed(\"not a seed\") { 1.0 }\n",
        ".ch",
    );
    assert!(score < 1.0, "a string seed must be rejected, got {score}");
    let score = check_score("def f() -> f32 = with device(42) { 1.0 }\n", ".ch");
    assert!(score < 1.0, "an int device must be rejected, got {score}");
}

/// chelis#721 control: a UNARY def's canonical Deep round-trips through eval.
#[test]
fn unary_fn_deep_roundtrips_through_eval() {
    let dp = deep_of("def f(x: f32) -> f32 = add(x, 1.5)\nout = print(f(1.0))\n");
    let stdout = eval_with_ext(&dp, ".dp").expect("unary deep must evaluate");
    assert!(stdout.contains("2.5"), "got: {stdout}");
}

// ===========================================================================
// chelis#709 - the holes
// ===========================================================================

/// chelis#731 Phase 1: the `handle-effect` checker case now checks the `with
/// seed` body, so the masked error inside it is caught (score < 1). Was
/// `#[ignore]`d red when the body was unchecked. The seed carries the required
/// `i64` suffix so the ONLY error is the body's (isolating what this pins).
#[test]
fn with_seed_body_is_type_checked() {
    let score = check_score(
        &format!("def f() -> f32 = with seed(42i64) {{ {MASKED_ERROR} }}\n"),
        ".ch",
    );
    assert!(
        score < 1.0,
        "the ill-typed with seed body must be caught, got score {score}"
    );
}

/// chelis#731 Phase 1: same, through `with device`. The device body is now
/// checked, so the masked error is caught.
#[test]
fn with_device_body_is_type_checked() {
    let score = check_score(
        &format!("def f() -> f32 = with device(\"gpu:0\") {{ {MASKED_ERROR} }}\n"),
        ".ch",
    );
    assert!(
        score < 1.0,
        "the ill-typed with device body must be caught, got score {score}"
    );
}

/// Positive parity: a WELL-TYPED `with seed` body (with the required `i64` seed
/// suffix) checks clean (score 1). The handle-effect case returns the body's
/// type, so a correct body is accepted, not just rejected.
#[test]
fn with_seed_well_typed_body_checks_clean() {
    let score = check_score(
        "def f() -> f32 = with seed(42i64) { add(cast(1.0, f32), cast(2.0, f32)) }\n",
        ".ch",
    );
    assert!(
        (score - 1.0).abs() < 1e-9,
        "a well-typed with seed body must check clean, got score {score}"
    );
}

/// Positive parity for `with device`: a well-typed body checks clean.
#[test]
fn with_device_well_typed_body_checks_clean() {
    let score = check_score(
        "def f() -> f32 = with device(\"gpu:0\") { add(cast(1.0, f32), cast(2.0, f32)) }\n",
        ".ch",
    );
    assert!(
        (score - 1.0).abs() < 1e-9,
        "a well-typed with device body must check clean, got score {score}"
    );
}

/// Negative parity for the §C1.5 seed-suffix rule (chelis#731 / chelis#771):
/// an UNSUFFIXED integer literal seed is a type error, even with a well-typed
/// body. This is the reject-diagnostic half chelis#771 left to Phase 1; it
/// unblocks the parked cross-lane RNG atom (chelis#735). The seeded-authoring
/// half stays with #735.
#[test]
fn unsuffixed_seed_literal_is_rejected() {
    let score = check_score(
        "def f() -> f32 = with seed(42) { add(cast(1.0, f32), cast(2.0, f32)) }\n",
        ".ch",
    );
    assert!(
        score < 1.0,
        "an unsuffixed seed literal must be rejected, got score {score}"
    );
}

/// [05-RNG-1] admits signed int64 seed bits in both source representations.
#[test]
fn negative_int64_seed_literal_is_accepted() {
    let score = check_score(
        "(module {} m.main (def {} out (handle-effect {effect: random} \
         (lit {type: (t-prim {} int64)} -1) (lit {type: (t-prim {} f32)} 2.5))))\n",
        ".dp",
    );
    assert!(
        score == 1.0,
        "a negative int64 seed literal must be accepted, got score {score}"
    );
}

/// chelis#731 Phase 1: `with seed(42i64) { cast(5, int64) }` from an `-> f32`
/// fn used to pass check (score 1), build, and run, printing an int64 `5` from
/// a function declared `-> f32`. The handle-effect case returns the body's type,
/// so the declared return type is now enforced and the build rejects it with a
/// type diagnostic before any backend sees it. Was `#[ignore]`d red.
#[test]
fn masked_return_type_violation_does_not_reach_a_binary() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "def f() -> f32 = with seed(42i64) { cast(5, int64) }\nout = print(f())\n";
    match c_lane_outcome(program, ".ch", "seed_masked") {
        CLane::Rejected(stderr) => assert!(
            stderr.contains("Type errors") || stderr.contains("type"),
            "the rejection must be a type diagnostic, got: {stderr}"
        ),
        CLane::Ran(stdout) => panic!(
            "an ill-typed program must not reach a runnable binary; it ran and printed: {stdout}"
        ),
    }
}

/// chelis#731 Phase 1: a `.dp` whose handle-effect carries a BOGUS effect kind
/// used to check at score 1 and build a working binary; `lower_handle_effect`'s
/// catch-all lowers the body and silently drops the unknown kind and its
/// handler (the #703 shape). Post chelis#731 P1 + chelis#730 P1 the pinned
/// section I1 interlock holds from BOTH sides: the checker's handle-effect
/// case string-matches the two known kinds (`random`/`resource`) with a
/// loud `MalformedForm` else, so an unknown kind is rejected at CHECK time
/// first - the earliest competent stage - and chelis#730's IR-lane and
/// host-lane lowering raises (census rows 9/20, the branded fatal
/// diagnostics) are defense-in-depth behind it. This test asserts the
/// surviving surface: the checker-level rejection. The `EffectKind` enum
/// that closes the future-kinds hole structurally is chelis#730 Phase 2.
/// Was `#[ignore]`d red.
#[test]
fn unknown_effect_kind_is_rejected() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let dp = deep_of("def f() -> f32 = with seed(42i64) { 2.5 }\nout = print(f())\n")
        .replace("effect: random", "effect: teleport");
    assert!(
        dp.contains("effect: teleport"),
        "probe fixture must rewrite"
    );
    match c_lane_outcome(&dp, ".dp", "bogus_effect") {
        CLane::Rejected(stderr) => assert!(
            !stderr.is_empty(),
            "rejection must carry a diagnostic naming the unknown effect kind"
        ),
        CLane::Ran(stdout) => panic!(
            "an unknown effect kind must not silently compile to a working \
             binary; it ran and printed: {stdout}"
        ),
    }
}

// ===========================================================================
// chelis#721 - a nullary def's canonical Deep round-trips through eval
//
// Before the fix these `.dp` evals died `value is not callable: Tensor(...)`:
// the def's DAG-folded constant shadowed its Closure in `tensor_bindings`.
// The print result is asserted on stdout's FIRST line exactly; eval also
// emits trailing labeled roots (`f = tensor(...)`) that the C lane does not
// (the known labeled-root parity residual recorded on the issue) - the tests
// deliberately pin only the print result so they neither depend on nor bless
// that residual.
// ===========================================================================

/// The exact print output is stdout's first line; assert on that rather than
/// `contains`, so a value that only appears in a trailing labeled root cannot
/// mask a broken print.
fn first_line(stdout: &str) -> &str {
    stdout.lines().next().unwrap_or_default()
}

/// The locked repro: the canonical Deep of the simplest constant-returning
/// nullary def now evaluates and prints its value. Was `#[ignore]`d as the
/// live #721 bug.
#[test]
fn nullary_fn_deep_roundtrips_through_eval() {
    let dp = deep_of("def f() -> f32 = 2.5\nout = print(f())\n");
    let stdout = eval_with_ext(&dp, ".dp")
        .expect("chelis#721: the canonical Deep of a nullary fn must evaluate");
    assert_eq!(first_line(&stdout), "2.5", "full stdout: {stdout}");
}

/// A nullary def applied more than once inside a larger expression: proves
/// the resolved Closure is reusable, not a one-shot, and composes.
#[test]
fn nullary_fn_applied_twice_through_eval() {
    let dp = deep_of("def f() -> f32 = 2.5\nout = print(add(f(), f()))\n");
    let stdout = eval_with_ext(&dp, ".dp").expect("nullary applied twice must evaluate");
    // chelis#732 P1 migration: integral f32 scalars render "5.0".
    assert_eq!(first_line(&stdout), "5.0", "full stdout: {stdout}");
}

/// A nullary def whose body is NOT a bare literal but still DAG-folds into
/// `tensor_bindings` (`cast(5, int64)` -> a shaped scalar root, verified via
/// the `f = tensor(...)` labeled root). It exercises the SAME tensor-root
/// shadow as the bare-lit lock through a compound body. (A scalar `add` of
/// two literals is host-gated to a Closure instead and never took this path,
/// so it is not the interesting case here.)
#[test]
fn nullary_nonliteral_fn_deep_roundtrips_through_eval() {
    let dp = deep_of("def f() -> int64 = cast(5, int64)\nout = print(f())\n");
    let stdout =
        eval_with_ext(&dp, ".dp").expect("nullary cast-bodied def must evaluate through eval");
    assert_eq!(first_line(&stdout), "5", "full stdout: {stdout}");
}

/// Negative parity for the fix's precedence guard: a LOCAL binding named `f`
/// (here the function-typed parameter of `call_f`) must still win over the
/// top-level nullary `f`. Only the `tensor_bindings` shadow is bypassed, never
/// the local-bindings lookup. If the guard regressed, `f(1.0)` would resolve
/// the nullary top-level `f` and fail `closure expected 0 args, got 1` (or
/// print `2.5`); the local lambda yields `1.0 + 100.0`.
#[test]
fn local_binding_shadows_nullary_def_in_eval() {
    let dp = deep_of(
        "def f() -> f32 = 2.5\n\
         def call_f(f: (f32) -> f32) -> f32 = f(1.0)\n\
         out = print(call_f(fn (x: f32) -> add(x, 100.0)))\n",
    );
    let stdout =
        eval_with_ext(&dp, ".dp").expect("local binding must shadow the top-level nullary def");
    // chelis#732 P1 migration: integral f32 scalars render "101.0".
    assert_eq!(first_line(&stdout), "101.0", "full stdout: {stdout}");
}

// ---------------------------------------------------------------------------
// RT #721 adversarials (fresh-context red team). Coverage the PR's tests do
// not carry: the chained-nullary and host-body variants of the resolve path,
// and the two negative guards the fix must NOT weaken — a non-fn top-level
// binding must stay non-callable, and arity errors must stay clean.
// ---------------------------------------------------------------------------

/// A nullary def that CALLS another nullary def. The inner `a()` folds to a
/// tensor-root (`a = tensor(...)`) while the outer `b` is a host closure;
/// both must resolve through the fixed application path. Was part of the
/// original shadow class.
#[test]
fn rt721_nullary_calls_nullary_through_eval() {
    let dp = deep_of("def a() -> f32 = 2.0\ndef b() -> f32 = add(a(), 1.0)\nout = print(b())\n");
    let stdout = eval_with_ext(&dp, ".dp").expect("chained nullary defs must evaluate");
    // chelis#732 P1 migration: integral f32 scalars render "3.0".
    assert_eq!(first_line(&stdout), "3.0", "full stdout: {stdout}");
}

/// A nullary def whose body is host-gated to a Closure (a scalar `add` of two
/// literals) and therefore NEVER lands in `tensor_bindings`. It exercises the
/// new path's `else`-free fn branch on a def that was never shadowed, proving
/// the fix does not disturb the always-working host-body nullary.
#[test]
fn rt721_host_bodied_nullary_still_roundtrips() {
    let dp = deep_of("def f() -> f32 = add(1.0, 1.5)\nout = print(f())\n");
    let stdout = eval_with_ext(&dp, ".dp").expect("host-body nullary must evaluate");
    assert_eq!(first_line(&stdout), "2.5", "full stdout: {stdout}");
}

/// Negative guard: a top-level NON-fn value binding (`g = to_tensor(...)`)
/// applied as `g()` must be REJECTED, never resolved into a call. The fix's
/// `matches!(.. tag == "fn")` guard is the reason a folded tensor root cannot
/// be turned into a callable. If it regressed, `g()` would print the tensor
/// or resolve it; here it must fail with a type mismatch and produce no value.
#[test]
fn rt721_toplevel_value_binding_is_not_callable() {
    let dp = deep_of("g = to_tensor([1.0])\nout = print(g())\n");
    let err = eval_with_ext(&dp, ".dp")
        .expect_err("applying a non-fn top-level value binding must be rejected");
    assert!(
        err.contains("type mismatch") && err.contains("tensor"),
        "expected a type-mismatch rejection naming the tensor, got: {err}"
    );
}

/// Negative guard: a nullary def applied with an argument is a clean arity
/// error through the fixed path, not shadow-path weirdness.
#[test]
fn rt721_nullary_applied_with_arg_is_arity_error() {
    let dp = deep_of("def f() -> f32 = 2.5\nout = print(f(3.0))\n");
    let err = eval_with_ext(&dp, ".dp").expect_err("nullary def with an arg must be rejected");
    assert!(
        err.contains("arity mismatch") && err.contains("expected 0 args, got 1"),
        "expected a clean 0-vs-1 arity error, got: {err}"
    );
}
