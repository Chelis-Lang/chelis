//! chelis#1558: `[04-DTYPE-1]` at check time when the cast source is a
//! variable.
//!
//! `spec/04-type-system.md` [04-DTYPE-1] says a primitive type position SHALL
//! name an active primitive and that the TYPE CHECKER rejects anything else,
//! "including as a `cast` target". [04-DTYPE-2] says a binder that declares no
//! bound "remains an unconstrained type variable admitting every type, not only
//! a dtype". So `cast(<anything>, p)` under an unbounded `[p]` names a
//! non-primitive in a primitive position and the checker owes a rejection.
//!
//! PR #1545 delivered that rejection for a LITERAL operand. The classifier it
//! added (`chelis_deep::literal_source::visit_binder_literal_uses`) already
//! emitted the event for every cast whose target is a bare `(t-var {} p)`; only
//! the consuming arm's `source: Some(_)` pattern confined the rejection to
//! literals. Dropping that pattern is this change.
//!
//! Before it, the six variable-source forms below checked at 1.0 at BOTH
//! ingresses and were caught late and differently by each lane: `chelis build`
//! and `chelis eval --file` both failed at IR lowering with the [04-DTYPE-1]
//! diagnostic, while the compiler-api host interpreter accepted them and
//! actualized the target per call site. That interpreter acceptance is what
//! `chelis-compiler-api`'s `runtime::tests` rows pinned; they are inverted in
//! the same change set.
//!
//! Negative parity: a bounded binder is a legitimate primitive position under
//! [04-DTYPE-2], so `[p: Int]`, `[p: Float]` and `[p: Numeric]` must keep
//! checking clean. Those controls are also the proof that the diagnostic's
//! remedy works, rather than merely being offered.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// A binder with no bound, cast to from a non-literal source. Each entry is
/// (module, program body, the binder named in the diagnostic, its owning def).
const REJECTED_FORMS: &[(&str, &str, &str, &str)] = &[
    (
        "Param",
        "def recast[p](value: p) -> p = cast(value, p)\nout = recast(cast(7, int32))\n",
        "p",
        "recast",
    ),
    (
        "LetBound",
        "def recast[p](value: p) -> p = {\n  v = value\n  cast(v, p)\n}\nout = recast(cast(7, int32))\n",
        "p",
        "recast",
    ),
    (
        "CallResult",
        "def ident[p](value: p) -> p = value\n\
         def recast[p](value: p) -> p = cast(ident(value), p)\n\
         out = recast(cast(7, int32))\n",
        "p",
        "recast",
    ),
    (
        "InLambda",
        "def recast[p](value: p) -> p = (fn (v) -> cast(v, p))(value)\n\
         out = recast(cast(7, int32))\n",
        "p",
        "recast",
    ),
    (
        "HelperGeneric",
        "def helper[q](value: q) -> q = cast(value, q)\n\
         def outer[p](value: p) -> p = helper(value)\n\
         out = outer(cast(7, int32))\n",
        "q",
        "helper",
    ),
    (
        "SigBinder",
        "sig recast: p -> p\ndef recast(value) = cast(value, p)\nout = recast(cast(7, int32))\n",
        "p",
        "recast",
    ),
];

/// Programs that must keep checking clean. A concrete target, and the three
/// dtype-family bounds the diagnostic offers as its remedy.
const ACCEPTED_FORMS: &[(&str, &str)] = &[
    (
        "Concrete",
        "def recast(value: int32) -> f64 = cast(value, f64)\nout = recast(cast(7, int32))\n",
    ),
    (
        "BoundedInt",
        "def recast[p: Int](value: p) -> p = cast(value, p)\nout = recast(cast(7, int32))\n",
    ),
    (
        "BoundedFloat",
        "def recast[p: Float](value: p) -> p = cast(value, p)\nout = recast(cast(1.5, f32))\n",
    ),
    (
        "BoundedNumeric",
        "def recast[p: Numeric](value: p) -> p = cast(value, p)\nout = recast(cast(7, int32))\n",
    ),
    (
        "BoundedLetBound",
        "def recast[p: Int](value: p) -> p = {\n  v = value\n  cast(v, p)\n}\n\
         out = recast(cast(7, int32))\n",
    ),
];

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn chelis(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(args)
        .output()
        .expect("run chelis")
}

fn run_check(path: &Path) -> Value {
    let output = chelis(&["check", path.to_str().unwrap()]);
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn fmt_inplace(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

/// Write, canonicalize, then check the program at both ingresses. Returns the
/// Surf verdict and the Deep verdict so a caller can assert they agree.
fn check_both_ingresses(dir: &Path, module: &str, body: &str) -> (Value, Value) {
    let source = dir.join(format!("{module}.ch"));
    write_file(&source, &format!("module {module}\n{body}"));
    fmt_inplace(&source);
    let surf = run_check(&source);

    let deep = chelis(&["deep", source.to_str().unwrap()]);
    assert!(
        deep.status.success(),
        "chelis deep must succeed for {module}: {}",
        String::from_utf8_lossy(&deep.stderr)
    );
    let lowered = source.with_extension("dp");
    fs::write(&lowered, &deep.stdout).expect("write .dp");
    (surf, run_check(&lowered))
}

/// chelis#1558: **regression test**. Six non-literal sources cast to an
/// unbounded binder, each rejected at check time under [04-DTYPE-1], with the
/// diagnostic naming the offending binder and its owning declaration.
///
/// Red before this change: every one of these scored 1.0 with an empty error
/// list, at both ingresses. Proven by reverting
/// `crates/chelis-types/src/infer/binder_literal.rs` to the parent commit.
///
/// The claim is exactly these six forms at `chelis check`. No enumerator proves
/// the set of expressible non-literal sources, so the rest is unclaimed.
#[test]
fn unbounded_cast_target_is_rejected_for_every_variable_source_form() {
    let dir = tempdir().expect("tempdir");
    for (module, body, binder, owner) in REJECTED_FORMS {
        let module = format!("Issue1558{module}");
        let (surf, deep) = check_both_ingresses(dir.path(), &module, body);
        let subject =
            format!("cast target `{binder}` in `{owner}` does not name an active primitive");
        for (label, json) in [("surf", &surf), ("deep", &deep)] {
            assert_ne!(
                json["score"], 1,
                "{module} must not check clean at the {label} ingress: {json}"
            );
            assert!(
                error_messages(json)
                    .iter()
                    .any(|m| m.contains(&subject) && m.contains("04-DTYPE-1")),
                "{module} at the {label} ingress must carry the [04-DTYPE-1] rejection \
                 naming `{binder}`; got {:?}",
                error_messages(json)
            );
        }
    }
}

/// chelis#1558: **regression test** for ingress agreement. The verdict must not
/// depend on the entry point, so this asserts equality of score and error list
/// between `chelis check f.ch` and `chelis check f.dp`, not merely that each
/// rejects.
///
/// Red before this change at both ingresses, which agreed on the wrong verdict;
/// the score assertion in the test above is what turns red.
#[test]
fn unbounded_cast_target_verdict_agrees_across_both_ingresses() {
    let dir = tempdir().expect("tempdir");
    for (module, body, _, _) in REJECTED_FORMS {
        let module = format!("Issue1558Agree{module}");
        let (surf, deep) = check_both_ingresses(dir.path(), &module, body);
        assert_eq!(
            surf["score"], deep["score"],
            "{module}: ingresses must agree on score; surf={surf}, deep={deep}"
        );
        assert_eq!(
            error_messages(&surf),
            error_messages(&deep),
            "{module}: ingresses must agree on the error list"
        );
    }
}

/// chelis#1558 negative parity: **disposition lock**. A concrete cast target
/// and the three dtype-family bounds keep checking clean at both ingresses.
///
/// Green before and after. Its job is twofold: it holds the boundary of the
/// rejection above, so the change cannot degrade into "reject every binder
/// target"; and it is the evidence that the diagnostic's remedy works, since
/// the remedy it prints is exactly to declare one of these bounds.
#[test]
fn a_concrete_or_bounded_cast_target_still_checks_clean() {
    let dir = tempdir().expect("tempdir");
    for (module, body) in ACCEPTED_FORMS {
        let module = format!("Issue1558Ok{module}");
        let (surf, deep) = check_both_ingresses(dir.path(), &module, body);
        for (label, json) in [("surf", &surf), ("deep", &deep)] {
            assert_eq!(
                json["score"], 1,
                "{module} must check clean at the {label} ingress: {json}"
            );
            assert!(
                error_messages(json).is_empty(),
                "{module} at the {label} ingress must report no errors; got {:?}",
                error_messages(json)
            );
        }
    }
}

/// chelis#1558: **disposition lock** recording what a tensor source does, and
/// deliberately deciding nothing about it.
///
/// A tensor cast to an unbounded binder was already rejected before this
/// change, at 0.92 with `CastNonTensor`, "cast to a quantified scalar dtype
/// requires a numeric scalar". That is a scalar-source rule and not
/// [04-DTYPE-1]. After this change the program carries both errors, because the
/// target is independently a non-primitive.
///
/// This asserts only that both errors are present. **It decides nothing for
/// chelis#1564**, which owns whether `cast(tensor, bounded_binder)` should be
/// admitted at all; that question is about a BOUNDED target and this row's
/// target is unbounded.
#[test]
fn a_tensor_source_carries_both_the_scalar_rule_and_the_dtype_rule() {
    let dir = tempdir().expect("tempdir");
    let (surf, deep) = check_both_ingresses(
        dir.path(),
        "Issue1558TensorSource",
        "def recast[p](value: tensor[2, p]) -> tensor[2, p] = cast(value, p)\n\
         out = recast(to_tensor([cast(1, int32), cast(2, int32)]))\n",
    );
    for (label, json) in [("surf", &surf), ("deep", &deep)] {
        let messages = error_messages(json);
        assert!(
            messages
                .iter()
                .any(|m| m.contains("requires a numeric scalar")),
            "the pre-existing scalar-source rule must still fire at the {label} ingress; \
             got {messages:?}"
        );
        assert!(
            messages.iter().any(|m| m.contains("04-DTYPE-1")),
            "the [04-DTYPE-1] target rule must also fire at the {label} ingress; \
             got {messages:?}"
        );
    }
}
