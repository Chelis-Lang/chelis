//! chelis#260 Site 1: the declared-dim rigidity diagnostics named internal
//! `DimVar` ids (`d44`, `d45`) instead of the source parameters (`n`, `m`).
//!
//! The names live in `DeepTypeResolver`'s `name -> DimVar` map, which was
//! dropped when `resolve_deep_type` returned only the `Type`. They also could
//! not be used directly: the check runs on an INSTANTIATED signature, whose
//! dim variables are freshly minted by `Env::instantiate`, so the names had to
//! survive resolution, generalization, and instantiation to be renderable.

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

#[test]
fn collapsed_dim_parameters_are_named_not_numbered() {
    let stdout = check_stdout(concat!(
        "def go[n, m](cond: bool, a: tensor[n, f32], b: tensor[m, f32]) -> tensor[n, f32] = {\n",
        "  y = cond\n",
        "  if y then a else b\n",
        "}\n",
    ));
    assert!(
        stdout.contains("distinct declared dim parameters `n` and `m`"),
        "the collapse must name both source parameters; got:\n{stdout}"
    );
}

#[test]
fn a_collapse_diagnostic_carries_no_internal_dim_ids() {
    // The regression this issue is about: the message must not fall back to
    // the `d{N}` spelling once the signature recorded its names.
    let stdout = check_stdout(concat!(
        "def go[n, m](cond: bool, a: tensor[n, f32], b: tensor[m, f32]) -> tensor[n, f32] = {\n",
        "  y = cond\n",
        "  if y then a else b\n",
        "}\n",
    ));
    let collapse = stdout
        .lines()
        .find(|line| line.contains("distinct declared dim parameters"))
        .unwrap_or(&stdout);
    assert!(
        !collapse.contains(" d0")
            && !collapse.contains(" d1")
            && !collapse.contains(" d2")
            && !collapse.contains(" d3")
            && !collapse.contains(" d4"),
        "no internal DimVar id may reach the message; got:\n{collapse}"
    );
}

#[test]
fn a_dim_parameter_pinned_to_a_literal_is_named_too() {
    // The sibling arm of the same check. It rendered no identifier at all
    // before, so a multi-parameter signature gave no clue which dim was
    // pinned.
    let stdout = check_stdout(concat!(
        "def pin[k](a: tensor[k, f32]) -> tensor[k, f32] = {\n",
        "  b = to_tensor([cast(1.0, f32), cast(2.0, f32)])\n",
        "  b\n",
        "}\n",
    ));
    assert!(
        stdout.contains("polymorphic dim parameter `k` forced to concrete Lit(2)"),
        "the pin must name the source parameter; got:\n{stdout}"
    );
}

#[test]
fn a_recursive_definition_names_its_parameters_too() {
    // A recursive `def` instantiates through the in-group branch, which
    // needs the type mapping for recursion validation. That branch must
    // carry the dim mapping as well: the collapse is the same defect and a
    // recursive signature has the same provenance, so falling back to the
    // internal id here would leave spec/04 [04-FIT-9] unsatisfied for
    // exactly the programs most likely to hit it.
    let stdout = check_stdout(concat!(
        "def go[n, m](cond: bool, a: tensor[n, f32], b: tensor[m, f32]) -> tensor[n, f32] = {\n",
        "  y = cond\n",
        "  if y then go(y, a, b) else b\n",
        "}\n",
    ));
    assert!(
        stdout.contains("distinct declared dim parameters `n` and `m`"),
        "the recursive branch must name both source parameters; got:\n{stdout}"
    );
    assert!(
        !stdout.contains(" d4") && !stdout.contains(" d5"),
        "no internal DimVar id may reach the message; got:\n{stdout}"
    );
}

#[test]
fn a_recursive_definition_pinned_to_a_literal_is_named_too() {
    // The literal-pinning arm on the same in-group instantiation branch.
    // Naming the collapse there but not the pin would leave half the check
    // rendering internal ids for the same signature.
    let stdout = check_stdout(concat!(
        "def pin[k](cond: bool, a: tensor[k, f32]) -> tensor[k, f32] = {\n",
        "  b = to_tensor([cast(1.0, f32), cast(2.0, f32)])\n",
        "  if cond then pin(false, a) else b\n",
        "}\n",
    ));
    assert!(
        stdout.contains("polymorphic dim parameter `k` forced to concrete Lit(2)"),
        "the recursive pin must name the source parameter; got:\n{stdout}"
    );
}

#[test]
fn the_return_only_rigidity_check_names_its_parameters() {
    // `check_return_only_dvars_rigid` is the adjacent rigidity arm (chelis#273).
    // It reads the same declared signature and had the same defect, so the
    // name map reaches it too rather than leaving one arm of one contract
    // rendering `d{N}`.
    let stdout = check_stdout("def f[n, m](x: tensor[n, f32]) -> tensor[m, f32] = x\n");
    assert!(
        stdout.contains(
            "return-position dim parameter `m` was unified with the distinct \
                         declared dim parameter `n`"
        ),
        "the return-only collapse must name both parameters; got:\n{stdout}"
    );
    assert!(
        !stdout.contains(" d4") && !stdout.contains(" d5"),
        "no internal DimVar id may reach the message; got:\n{stdout}"
    );
}

#[test]
fn the_return_only_literal_pin_names_its_parameter() {
    // The sibling arm of the return-only check.
    let stdout = check_stdout("def g[n, m](x: tensor[2, f32]) -> tensor[m, f32] = x\n");
    assert!(
        stdout.contains("return-position dim parameter `m` was pinned to concrete Lit(2)"),
        "the return-only pin must name the source parameter; got:\n{stdout}"
    );
}

#[test]
fn the_list_uniformity_check_names_its_parameter() {
    // The third rigidity arm (chelis#272/#276). It renders `Dim::Name`
    // through its source spelling already, so a declared dim parameter
    // beside it rendering `d{N}` was the odd one out in its own message.
    let stdout = check_stdout(concat!(
        "def h[k](flag: bool) -> List[tensor[k, f32]] = {\n",
        "  a = to_tensor([cast(1.0, f32), cast(2.0, f32)])\n",
        "  b = to_tensor([cast(1.0, f32)])\n",
        "  [a, b]\n",
        "}\n",
    ));
    assert!(
        stdout.contains("promises a uniform dim parameter `k`"),
        "the list-uniformity promise must name the source parameter; got:\n{stdout}"
    );
    assert!(
        !stdout.contains(" d4") && !stdout.contains("parameter d"),
        "no internal DimVar id may reach the message; got:\n{stdout}"
    );
}

#[test]
fn names_do_not_leak_between_signatures() {
    // The names are recorded per definition and resolved through the
    // instantiation mapping. A second signature in the same program must
    // report its OWN parameters, not the ones recorded for the first.
    let stdout = check_stdout(concat!(
        "def alpha[n, m](a: tensor[n, f32], b: tensor[m, f32]) -> tensor[n, f32] = {\n",
        "  y = b\n",
        "  a\n",
        "}\n",
        "def beta[p, q](c: bool, a: tensor[p, f32], b: tensor[q, f32]) -> tensor[p, f32] = {\n",
        "  y = c\n",
        "  if y then a else b\n",
        "}\n",
    ));
    assert!(
        stdout.contains("distinct declared dim parameters `p` and `q`"),
        "beta's collapse must name beta's parameters; got:\n{stdout}"
    );
    assert!(
        !stdout.contains("`n`") && !stdout.contains("`m`"),
        "alpha's parameter names must not appear in beta's diagnostic; got:\n{stdout}"
    );
}

#[test]
fn a_legitimately_polymorphic_signature_is_still_accepted() {
    // Positive control: naming must not change which programs are rejected.
    //
    // Asserted structurally rather than against the literal `"errors": []`.
    // chelis#886 is replacing the hand-formatted report with a serializer
    // that preserves the shape but not the spacing, and a syntax-agnostic
    // claim ("this program reports no errors") should not be coupled to the
    // whitespace of the producer that happens to emit it today.
    let stdout = check_stdout(concat!(
        "def keep[n, m](a: tensor[n, f32], b: tensor[m, f32]) -> tensor[n, f32] = {\n",
        "  y = b\n",
        "  a\n",
        "}\n",
    ));
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("stdout is not JSON: {e}\n{stdout}"));
    let errors = report["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("`errors` must be an array; got:\n{stdout}"));
    assert!(
        errors.is_empty(),
        "distinct dims used distinctly must still type-check; got errors:\n{errors:#?}"
    );
}
