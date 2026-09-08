//! chelis#1593 at the CLI, both ingresses.
//!
//! `def f(x: u8) -> u8 = x` scored 1.0 with an empty error vector, because the
//! scalar type-name desugar arm made the reserved spelling an implicitly
//! quantified `(t-var {} u8)`. The signature named a dtype the language
//! rejects and meant `forall u8. u8 -> u8`.
//!
//! The two ingresses are Surf through `chelis check`, and the desugared Deep
//! that `chelis deep` prints, fed back through `chelis check` and
//! `chelis prove`. The Deep leg matters because the defect was IN the
//! desugaring: a `.dp` produced from the Surf carried the quantifier, so the
//! Deep ingress inherited the hole rather than catching it.
//!
//! Test labels are recorded in each function's doc comment.

use assert_cmd::Command;
use std::{fs, path::Path};
use tempfile::tempdir;

/// The eight §1.1.2 spellings.
const UNSIGNED: [&str; 8] = [
    "u8", "u16", "u32", "u64", "uint8", "uint16", "uint32", "uint64",
];

fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run")
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn source_for(name: &str) -> String {
    format!("module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n")
}

fn assert_rejected(rendered: &str, name: &str, lane: &str) {
    assert!(
        rendered.contains(&format!("`{name}`"))
            && rendered.contains("unsigned integer types are deferred")
            && rendered.contains("spec/04-type-system.md §1.1.1")
            && rendered.contains("§1.1.2"),
        "{lane} must reject `{name}` with the §1.1.1 / §1.1.2 diagnostic: \
         {rendered}"
    );
}

/// REGRESSION test. Surf ingress: the scored `chelis check` surface.
#[test]
fn chelis_check_rejects_an_unsigned_scalar_signature() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    for name in UNSIGNED {
        fs::write(root.join("unsigned.ch"), source_for(name)).expect("write");
        let rendered = text(&run(root, &["check", "unsigned.ch"]));
        assert!(
            !rendered.contains("\"score\": 1,"),
            "a signature naming the reserved dtype `{name}` must not score \
             1.0: {rendered}"
        );
        assert_rejected(&rendered, name, "`chelis check` on Surf");
    }
}

/// REGRESSION test. Deep ingress: the desugared `.dp` must carry `t-prim`, and
/// checking and proving that `.dp` must reject with the same diagnostic.
#[test]
fn the_desugared_deep_carries_t_prim_and_is_rejected_at_both_deep_entry_points() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    for name in UNSIGNED {
        fs::write(root.join("unsigned.ch"), source_for(name)).expect("write");

        let deep = run(root, &["deep", "unsigned.ch"]);
        assert!(
            deep.status.success(),
            "`chelis deep` must succeed: {deep:?}"
        );
        let printed = String::from_utf8_lossy(&deep.stdout).to_string();
        assert!(
            printed.contains(&format!("(t-prim {{}} {name})")),
            "the desugared Deep must carry `{name}` as a `t-prim`: {printed}"
        );
        assert!(
            !printed.contains(&format!("(t-var {{}} {name})")),
            "the desugared Deep must not quantify `{name}`: {printed}"
        );

        fs::write(root.join("unsigned.dp"), &printed).expect("write dp");
        let checked = text(&run(root, &["check", "unsigned.dp"]));
        assert!(
            !checked.contains("\"score\": 1,"),
            "the desugared Deep must not score 1.0 for `{name}`: {checked}"
        );
        assert_rejected(&checked, name, "`chelis check` on the desugared Deep");

        let proved = text(&run(root, &["prove", "unsigned.dp"]));
        assert!(
            proved.contains("module does not type-check"),
            "`chelis prove` must refuse a module that does not type-check for \
             `{name}`: {proved}"
        );
        assert_rejected(&proved, name, "`chelis prove` on the desugared Deep");
    }
}

/// DISPOSITION LOCK. Green in both states, and the neighbour this change must
/// not disturb: a genuine lowercase type variable still quantifies, still
/// scores 1.0, and still round-trips through the Deep ingress.
#[test]
fn a_genuine_lowercase_name_still_scores_one_at_both_ingresses() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    fs::write(
        root.join("poly.ch"),
        "module P.M\nexport (f)\ndef f(x: a) -> a = x\n",
    )
    .expect("write");

    let checked = text(&run(root, &["check", "poly.ch"]));
    assert!(
        checked.contains("\"score\": 1,"),
        "a genuine type variable must still score 1.0: {checked}"
    );

    let deep = run(root, &["deep", "poly.ch"]);
    let printed = String::from_utf8_lossy(&deep.stdout).to_string();
    assert!(
        printed.contains("(t-var {} a)"),
        "`a` must still desugar to a quantified type variable: {printed}"
    );
    fs::write(root.join("poly.dp"), &printed).expect("write dp");
    let deep_checked = text(&run(root, &["check", "poly.dp"]));
    assert!(
        deep_checked.contains("\"score\": 1,"),
        "the Deep ingress must agree with the Surf ingress: {deep_checked}"
    );
}

/// DISPOSITION LOCK. Green in both states. chelis#1587's short signed
/// spellings sit one character away from these names and must keep naming
/// their primitives rather than being caught by the new rejection.
#[test]
fn the_short_signed_aliases_are_untouched() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    for (short, long) in [
        ("i8", "int8"),
        ("i16", "int16"),
        ("i32", "int32"),
        ("i64", "int64"),
    ] {
        fs::write(root.join("signed.ch"), source_for(short)).expect("write");
        let checked = text(&run(root, &["check", "signed.ch"]));
        assert!(
            checked.contains("\"score\": 1,"),
            "`{short}` must still name `{long}` and check clean: {checked}"
        );
        let printed =
            String::from_utf8_lossy(&run(root, &["deep", "signed.ch"]).stdout).to_string();
        assert!(
            printed.contains(&format!("(t-prim {{}} {long})")),
            "`{short}` must still map to `{long}`: {printed}"
        );
    }
}

/// REGRESSION test. Both binder forms at both ingresses: an explicit `[..]`
/// clause does not rebind a reserved spelling, in the scalar position or in
/// the tensor precision slot, through `chelis check` on Surf and through the
/// Deep that `chelis deep` prints.
#[test]
fn an_explicit_binder_does_not_rebind_a_reserved_name_at_either_ingress() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    for name in UNSIGNED {
        for (label, source) in [
            (
                "scalar",
                format!("module P.M\nexport (f)\ndef f[{name}](x: {name}) -> {name} = x\n"),
            ),
            (
                "tensor precision",
                format!(
                    "module P.M\nexport (f)\n\
                     def f[{name}](x: tensor[3, {name}]) -> tensor[3, {name}] = x\n"
                ),
            ),
        ] {
            fs::write(root.join("binder.ch"), &source).expect("write");

            let checked = text(&run(root, &["check", "binder.ch"]));
            assert!(
                !checked.contains("\"score\": 1,"),
                "an explicit binder must not rebind `{name}` in a {label} \
                 position: {checked}"
            );
            assert!(
                checked.contains("unsigned integer types are deferred"),
                "the {label} binder form must carry the §1.1.1 diagnostic for \
                 `{name}`: {checked}"
            );

            let printed =
                String::from_utf8_lossy(&run(root, &["deep", "binder.ch"]).stdout).to_string();
            assert!(
                printed.contains(&format!("(t-prim {{}} {name})"))
                    && !printed.contains(&format!("(t-var {{}} {name})")),
                "the desugared Deep must not quantify `{name}` in a {label} \
                 position: {printed}"
            );
            fs::write(root.join("binder.dp"), &printed).expect("write dp");
            let deep_checked = text(&run(root, &["check", "binder.dp"]));
            assert!(
                !deep_checked.contains("\"score\": 1,")
                    && deep_checked.contains("unsigned integer types are deferred"),
                "the Deep ingress must reject the {label} binder form for \
                 `{name}`: {deep_checked}"
            );
        }
    }
}

/// DISPOSITION LOCK. Green in both states. The positive control: an explicit
/// binder with an ordinary lowercase name still binds at both ingresses.
#[test]
fn an_explicit_binder_with_an_ordinary_name_still_scores_one() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    for source in [
        "module P.M\nexport (f)\ndef f[p](x: p) -> p = x\n",
        "module P.M\nexport (f)\ndef f[p](x: tensor[3, p]) -> tensor[3, p] = x\n",
    ] {
        fs::write(root.join("bound.ch"), source).expect("write");
        let checked = text(&run(root, &["check", "bound.ch"]));
        assert!(
            checked.contains("\"score\": 1,"),
            "an ordinary explicit binder must still bind: {source}\n{checked}"
        );
        let printed = String::from_utf8_lossy(&run(root, &["deep", "bound.ch"]).stdout).to_string();
        assert!(
            printed.contains("(t-var {} p)"),
            "`p` must stay a bound type variable: {printed}"
        );
    }
}
