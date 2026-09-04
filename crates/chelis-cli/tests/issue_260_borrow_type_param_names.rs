//! chelis#260 Site 2: the borrow diagnostic named an internal `TypeVar` id
//! (`?344`) instead of the source parameter (`t`).
//!
//! The threading the issue proposes cannot work, and the reason is worth
//! recording because it is not visible from the source. Measured on
//! `def go[t](x: t)`: the resolver mints `t` as `?343`, the instantiation for
//! the body renames it to `?344`, the borrow site defers a LATER variable
//! `?345`, and `?345` resolves back to `?344`. Four identities for one source
//! name, so a map captured at resolution and consulted at the drain misses on
//! every hop.
//!
//! What carries the name is the composition: the `Defsig` arm records the
//! resolver's map, the instantiation composes it through the fresh-variable
//! renaming, and the result is parked on the outer `Env` because the drain
//! runs per-def after body inference and never sees the instantiation.

use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

fn check_stdout(source: &str) -> String {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("probe.ch");
    fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check"])
        .arg(&path)
        .output()
        .expect("run chelis check");
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn borrow_message(stdout: &str) -> String {
    let report: serde_json::Value = serde_json::from_str(stdout)
        .unwrap_or_else(|e| panic!("stdout is not JSON: {e}\n{stdout}"));
    report["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("`errors` must be an array; got:\n{stdout}"))
        .iter()
        .find_map(|error| {
            let message = error["message"].as_str()?;
            message
                .contains("borrow requires")
                .then(|| message.to_string())
        })
        .unwrap_or_else(|| panic!("no borrow diagnostic; got:\n{stdout}"))
}

#[test]
fn a_declared_type_parameter_is_named_not_numbered() {
    let message = borrow_message(&check_stdout(concat!(
        "def go[t](x: t) -> int32 = {\n",
        "  y = &x\n",
        "  1i32\n",
        "}\n",
    )));
    assert!(
        message.contains("got `t`"),
        "the borrow diagnostic must name the source parameter; got: {message}"
    );
}

#[test]
fn a_borrow_diagnostic_carries_no_internal_type_id() {
    // The regression this issue is about. `?N` is inference bookkeeping and a
    // reader has no way to map it back to source (spec/04 [04-FIT-9]).
    let message = borrow_message(&check_stdout(concat!(
        "def go[t](x: t) -> int32 = {\n",
        "  y = &x\n",
        "  1i32\n",
        "}\n",
    )));
    assert!(
        !message.contains('?'),
        "no internal TypeVar id may reach the message; got: {message}"
    );
}

#[test]
fn the_named_parameter_is_the_borrowed_one() {
    // A single-parameter signature passes even if the renderer picks an
    // arbitrary entry from the recorded map. Two parameters make the choice
    // observable: `y` has type `b`, so naming `a` would be a confidently
    // wrong answer, which is worse than an opaque id.
    //
    // The property holds up to unification, and no further. Where the body
    // unifies two declared parameters the lookup reports the substitution's
    // representative, which tracks quantifier order rather than the borrow:
    // measured, `def go[a, b](x: a, y: b)` borrowing `x` under `a ~ b`
    // prints `b`, and the same program with the parameters declared in the
    // other order prints `a`. Both spell the one type the two names now
    // denote, so neither is false under [04-FIT-9] -- but the attribution is
    // arbitrary. Naming every co-unified parameter, the way Site 1 already
    // does for collapsed dimensions, is recorded as residual scope on
    // chelis#260 rather than done here.
    let message = borrow_message(&check_stdout(concat!(
        "def go[a, b](x: a, y: b) -> int32 = {\n",
        "  z = &y\n",
        "  1i32\n",
        "}\n",
    )));
    assert!(
        message.contains("got `b`"),
        "the borrow must name the parameter it borrowed; got: {message}"
    );
    assert!(
        !message.contains("got `a`"),
        "naming the other parameter is misattribution; got: {message}"
    );
}

#[test]
fn names_do_not_leak_between_signatures() {
    // The map is parked per definition. A second signature must report its
    // OWN parameter, not the one recorded for the first.
    let stdout = check_stdout(concat!(
        "def alpha[t](x: t) -> int32 = 1i32\n",
        "def beta[q](y: q) -> int32 = {\n",
        "  z = &y\n",
        "  1i32\n",
        "}\n",
    ));
    let message = borrow_message(&stdout);
    assert!(
        message.contains("got `q`"),
        "beta's diagnostic must name beta's parameter; got: {message}"
    );
    assert!(
        !message.contains("got `t`"),
        "alpha's parameter must not appear in beta's diagnostic; got: {message}"
    );
}

#[test]
fn a_concrete_type_still_renders_itself() {
    // Guards the renderer's non-variable arm: a concrete type has no
    // declared parameter to look up and must keep rendering itself.
    //
    // This is NOT [04-FIT-10]. `f32` is a spelling the user wrote, not an
    // inference identity, so the atom's subject is the test below.
    let message = borrow_message(&check_stdout(concat!(
        "def go(x: f32) -> int32 = {\n",
        "  y = &x\n",
        "  1i32\n",
        "}\n",
    )));
    assert!(
        message.contains("got f32"),
        "a concrete type renders itself; got: {message}"
    );
}

#[test]
fn an_inference_identity_with_no_source_name_renders_as_synthesized() {
    // spec/04 [04-FIT-10]: "Where provenance genuinely does not exist, a
    // diagnostic SHALL render the inference identity as synthesized,
    // distinguishably from a spelling the user wrote, and SHALL NOT invent a
    // source name for it."
    //
    // An unannotated parameter has no declared spelling to recover, so the
    // fallback arm is the atom's subject and `?N` is the compliant answer.
    // The residual is deliberate: Site 2 names DECLARED type parameters, and
    // this shape is what remains outside that.
    let message = borrow_message(&check_stdout(concat!(
        "def go(x) -> int32 = {\n",
        "  y = &x\n",
        "  1i32\n",
        "}\n",
    )));
    let identity = message
        .rsplit_once("got ")
        .expect("the borrow diagnostic reports what it got")
        .1
        .to_string();
    assert!(
        identity.starts_with('?'),
        "an identity with no provenance must render as synthesized; got: {message}"
    );
    assert!(
        !identity.contains('`'),
        "[04-FIT-10] forbids inventing a source name for it; got: {message}"
    );
}
