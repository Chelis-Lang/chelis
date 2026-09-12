//! chelis#1489: a suspended operand decision's result variable must not be
//! generalized by `let`.
//!
//! #1577 made `copy` and `cast` defer a decision on an operand that is not
//! resolved yet: the gate is suspended on the operand's type variable and hands
//! its consumer a FRESH result variable, which discharge unifies with the
//! decided type once the operand binds. Nothing tied that result variable to the
//! operand, so an unannotated `let` inside the suspending scope generalized it.
//! Every use of the bound name then got its own unconstrained instance, and the
//! later discharge bound only the original -- so the declared signature was
//! never checked against what the call actually produces.
//!
//! The consequence was a false signature that checked at score 1.0 and ran:
//!
//! ```text
//! def probe(t: tensor[4, 3, f32]) -> tensor[100, 3, f32] =
//!   apply_n(fn (v) -> { g = copy(v)
//!     g }, t)                          # checked; returned tensor[4, 3]
//! ```
//!
//! Main before #1577 rejected that program (as `copy requires tensor input, got
//! ?N` -- the wrong reason, but a sound verdict), so this was a regression. Review
//! round 4 of PR #1690 found it; PR #1690 had inherited the same result-variable
//! design for `gather`/`scatter`/`trace`.
//!
//! The repair is the one `let` already applies to a recursive group's
//! instantiation variables (`tvar_pinned`): a variable that still occurs in a
//! pending gate's result stays monomorphic until that gate discharges.

use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

fn run(subcommand: &str, source: &str) -> std::process::Output {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("probe.ch");
    fs::write(&path, source).expect("write fixture");
    let mut cmd = Command::cargo_bin("chelis").expect("binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1").arg(subcommand);
    if subcommand == "eval" {
        cmd.arg("--file");
    }
    cmd.arg(&path).output().expect("run chelis")
}

fn check_json(source: &str) -> serde_json::Value {
    let output = run("check", source);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("stdout is not JSON: {e}\n{stdout}"))
}

fn messages(report: &serde_json::Value) -> Vec<String> {
    report["errors"]
        .as_array()
        .map(|errors| {
            errors
                .iter()
                .filter_map(|e| e["message"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn kinds(report: &serde_json::Value) -> Vec<String> {
    report["errors"]
        .as_array()
        .map(|errors| {
            errors
                .iter()
                .filter_map(|e| e["kind"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// A dim-polymorphic higher-order helper: `v` is an unresolved variable while
/// the lambda body is inferred, and `n` is not fixed until the argument is.
const HELPER: &str =
    "def apply_n[b](f: tensor[n, 3, f32] -> b, x: tensor[n, 3, f32]) -> b = f(x)\n";

/// The `let`-bound form, for a given call and declared result.
fn let_bound(module: &str, call: &str, declared: &str) -> String {
    format!(
        "module {module}\n\
         {HELPER}\
         def probe(t: tensor[4, 3, f32]) -> {declared} = apply_n(fn (v) -> {{ g = {call}\n\
         \x20 g }}, t)\n"
    )
}

/// The regression itself. A false declared extent must be rejected, for both
/// gates #1577 converted.
#[test]
fn a_let_bound_deferred_result_is_checked_against_the_declaration() {
    for (call, lie) in [
        ("copy(v)", "tensor[100, 3, f32]"),
        ("cast(v, f64)", "tensor[100, 3, f64]"),
    ] {
        let report = check_json(&let_bound("Issue1489GenLie", call, lie));
        assert!(
            !messages(&report).is_empty(),
            "{call}: the true result is 4 x 3, so a declared {lie} is FALSE and must \
             be rejected. A clean report means `let` generalized the suspended \
             result variable and the declaration was never checked"
        );
        // And for the right reason: the declared extent disagrees with the one
        // the call produces. Any error at all would satisfy the check above,
        // including an unrelated one this change happened to introduce.
        assert!(
            kinds(&report).iter().any(|k| k == "DimensionMismatch"),
            "{call}: expected a DimensionMismatch against the declared {lie}; got \
             {:?}",
            messages(&report)
        );
    }
}

/// Positive parity: the TRUE declaration must still be accepted. Without this,
/// the test above would pass for a fix that simply rejects every `let`-bound
/// deferred result.
#[test]
fn a_let_bound_deferred_result_with_the_true_type_is_accepted() {
    for (call, truth) in [
        ("copy(v)", "tensor[4, 3, f32]"),
        ("cast(v, f64)", "tensor[4, 3, f64]"),
    ] {
        let report = check_json(&let_bound("Issue1489GenTruth", call, truth));
        assert!(
            messages(&report).is_empty(),
            "{call}: {truth} is the true result and must be accepted; got {:?}",
            messages(&report)
        );
    }
}

/// The bound name used more than once. Each use previously instantiated the
/// generalized result separately, so the uses could disagree with each other as
/// well as with the declaration.
#[test]
fn every_use_of_a_let_bound_deferred_result_is_the_same_type() {
    let source = format!(
        "module Issue1489GenTwice\n\
         {HELPER}\
         def probe(t: tensor[4, 3, f32]) -> (tensor[4, 3, f32], tensor[100, 3, f32]) =\n\
        \x20 apply_n(fn (v) -> {{ g = copy(v)\n\
        \x20   (g, g) }}, t)\n"
    );
    let report = check_json(&source);
    assert!(
        !messages(&report).is_empty(),
        "both components are the same `g`, so they cannot be both 4 x 3 and \
         100 x 3; a clean report means each use got its own instance"
    );
    assert!(
        kinds(&report).iter().any(|k| k == "DimensionMismatch"),
        "expected a DimensionMismatch between the two uses; got {:?}",
        messages(&report)
    );
}

/// A result that is PARTLY unified before its `let` generalizes it.
///
/// `id_dim` is dim-polymorphic, so applying it INSIDE the `let`'s right-hand
/// side mints a fresh `d'` at that level and unifies the pending result with
/// `tensor[d', 3, f32]` before the binding is generalized. The result's own type
/// variable is then gone -- it is bound -- and only `d'` stands between the
/// declaration and the call. Excluding just the top-level type variable would
/// quantify `d'`, and every use of `g` would pick its own extent.
///
/// An earlier fixture here partly unified the result AFTER the `let`, where the
/// type-variable exclusion alone already caught it, so it passed with the
/// dim-variable exclusion removed and pinned nothing. This form fails without it.
#[test]
fn a_partly_unified_pending_result_keeps_its_dims_monomorphic() {
    let source = format!(
        "module Issue1489GenPartial\n\
         {HELPER}\
         def id_dim(x: tensor[d, 3, f32]) -> tensor[d, 3, f32] = x\n\
         def probe(t: tensor[4, 3, f32]) -> tensor[100, 3, f32] =\n\
        \x20 apply_n(fn (v) -> {{ g = id_dim(copy(v))\n\
        \x20   g }}, t)\n"
    );
    let report = check_json(&source);
    assert!(
        !messages(&report).is_empty(),
        "`g` is 4 x 3 once `v` settles, so a declared 100 x 3 is FALSE; a clean \
         report means the dim variable the pending result was partly unified \
         with got generalized"
    );
    assert!(
        kinds(&report).iter().any(|k| k == "DimensionMismatch"),
        "expected a DimensionMismatch against the declared 100 x 3; got {:?}",
        messages(&report)
    );
}

/// The same door through a RANK variable. `id_rank` is rank-polymorphic, so
/// applying it inside the `let`'s right-hand side unifies the pending result
/// with `tensor[..r', f32]` for a fresh `r'`, and only `r'` is left to be
/// quantified. The rank-variable exclusion is what keeps it monomorphic.
#[test]
fn a_rank_polymorphic_pending_result_keeps_its_rank_monomorphic() {
    let source = format!(
        "module Issue1489GenRank\n\
         {HELPER}\
         def id_rank(x: tensor[..r, f32]) -> tensor[..r, f32] = x\n\
         def probe(t: tensor[4, 3, f32]) -> tensor[4, 3, 7, f32] =\n\
        \x20 apply_n(fn (v) -> {{ g = id_rank(copy(v))\n\
        \x20   g }}, t)\n"
    );
    let report = check_json(&source);
    assert!(
        !messages(&report).is_empty(),
        "`g` is rank 2 once `v` settles, so a declared rank-3 result is FALSE; a \
         clean report means the rank variable the pending result was partly \
         unified with got generalized"
    );
    assert!(
        kinds(&report).iter().any(|k| k == "DimensionMismatch"),
        "expected a DimensionMismatch against the declared rank-3 result; got \
         {:?}",
        messages(&report)
    );
}

/// The same door through a DTYPE variable. `id_dt` is dtype-polymorphic, so
/// applying it inside the `let`'s right-hand side unifies the pending result
/// with `tensor[4, 3, p']` for a fresh `p'`. A tensor's precision variable is a
/// type variable here, so it is the type-variable exclusion that keeps `p'`
/// monomorphic -- and until this test, nothing pinned that path.
#[test]
fn a_dtype_polymorphic_pending_result_keeps_its_dtype_monomorphic() {
    let source = |declared: &str| {
        format!(
            "module Issue1489GenDtype\n\
             {HELPER}\
             def id_dt[p](x: tensor[4, 3, p]) -> tensor[4, 3, p] = x\n\
             def probe(t: tensor[4, 3, f32]) -> {declared} =\n\
            \x20 apply_n(fn (v) -> {{ g = id_dt(copy(v))\n\
            \x20   g }}, t)\n"
        )
    };
    let report = check_json(&source("tensor[4, 3, f64]"));
    assert!(
        !messages(&report).is_empty(),
        "`g` is f32 once `v` settles, so a declared f64 result is FALSE; a clean \
         report means the precision variable the pending result was partly \
         unified with got generalized"
    );
    assert!(
        kinds(&report).iter().any(|k| k == "TypeMismatch"),
        "expected a TypeMismatch against the declared f64 result; got {:?}",
        messages(&report)
    );
    assert!(
        messages(&check_json(&source("tensor[4, 3, f32]"))).is_empty(),
        "the true f32 declaration must still be accepted"
    );
}

/// End to end: the accepted program's runtime shape is the one it declares.
/// The regression was only visible because the false program also RAN.
#[test]
fn the_accepted_program_runs_at_its_declared_shape() {
    let source = format!(
        "module Issue1489GenRun\n\
         {HELPER}\
         def probe(t: tensor[4, 3, f32]) -> tensor[4, 3, f32] = apply_n(fn (v) -> {{ g = copy(v)\n\
        \x20 g }}, t)\n\
         xs = to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32], \
         [7.0f32, 8.0f32, 9.0f32], [1.0f32, 1.0f32, 1.0f32]])\n\
         r = probe(xs)\n"
    );
    let output = run("eval", &source);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("shape=[4, 3]"),
        "the program declares and must return a 4 x 3 tensor; got status {:?}\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Control: the direct form, with no `let`, was never affected, and must keep
/// rejecting the false declaration.
#[test]
fn the_direct_form_still_rejects_a_false_declaration() {
    let source = format!(
        "module Issue1489GenDirect\n\
         {HELPER}\
         def probe(t: tensor[4, 3, f32]) -> tensor[100, 3, f32] = apply_n(fn (v) -> copy(v), t)\n"
    );
    assert!(
        !messages(&check_json(&source)).is_empty(),
        "the direct form must reject a false declaration"
    );
}
