// Regression for chelis#317: after chelis#157 / #316 (module-scoped
// constructor resolution), a module that referenced a constructor of an
// ADT declared in ANOTHER module — at a construction site or in a `match`
// pattern — WITHOUT naming that constructor in its `import` list
// `chelis check`ed clean and then SILENTLY mis-resolved the constructor to
// a foreign module's mangled tag through the registry's fuzzy
// terminal-segment fallback. The defect surfaced only at RUNTIME as a
// `non-exhaustive runtime match`.
//
// Fix (`crates/chelis-types/src/infer.rs`): a constructor reference is in
// scope only when its exact (mangled or bare-builtin) name is bound. A bare
// constructor name that resolves ONLY through `lookup_terminal_unique` is a
// constructor from another module that the importing module never pulled
// in, and is now rejected at `check` with an `unknown constructor`
// diagnostic — the same way an unknown value name is an `unbound variable`.
// The migration the issue documents is to name the used constructors in the
// import list (`import Pkg.Adt (Mode, Alpha, Beta, Gamma, classify)`); the
// declaring module's `export` list does NOT need them added, because
// exporting the TYPE already auto-exports its constructors, so they are
// importable by name.
//
// These tests drive the real `chelis check` binary over a no-dependency
// reef package so the product surface — not just the library pipeline — is
// covered, mirroring `issue_316_qualified_ctor_disambiguation.rs`.

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

const REEF_TOML: &str = "[package]\n\
     name = \"adt\"\n\
     version = \"0.1.0\"\n\
     compiler = \"=0.7.23\"\n\
     module_prefix = \"Pkg\"\n";

// The declaring module exports only `(Mode, classify)`. Exporting the TYPE
// auto-exports its constructors (Alpha/Beta/Gamma), so a consumer may import
// them BY NAME even though they are not spelled in this export list.
const ADT: &str = "module Pkg.Adt\n\
     export (Mode, classify)\n\
     type Mode = | Alpha | Beta | Gamma\n\
     def classify(m: Mode) -> i64 = match m with { | Alpha => 0 | Beta => 1 | Gamma => 2 }\n";

/// Run `chelis check <dir>` with the style gate disabled (the fixtures
/// synthesize ad-hoc Surf to exercise resolution, not formatting) and return
/// the per-file JSON map keyed by file path.
fn check_package(root: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", root.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("check output must be json: {e}\n{:?}", output))
}

/// Pull the entry for a single source file out of the `{"files":[...]}` map.
fn file_entry<'a>(json: &'a Value, suffix: &str) -> &'a Value {
    json["files"]
        .as_array()
        .expect("files array")
        .iter()
        .find(|f| f["file"].as_str().is_some_and(|p| p.ends_with(suffix)))
        .unwrap_or_else(|| panic!("no entry for {suffix} in {json}"))
}

fn error_messages(combo: &Value) -> Vec<String> {
    combo["report"]["errors"]
        .as_array()
        .map(|errs| {
            errs.iter()
                .map(|e| e["message"].as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default()
}

// ── NEGATIVE: type-only import + cross-module constructor use ────────────

#[test]
fn type_only_import_then_construct_is_unknown_constructor() {
    // The headline defect from the issue: importing only the TYPE `Mode`
    // (and `classify`) and then constructing `Alpha` must be rejected at
    // `check` as an unknown constructor, NOT pass clean and fail at runtime.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), REEF_TOML);
    write_file(&root.join("src/adt.ch"), ADT);
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.Adt (Mode, classify)\n\
         def use_alpha() -> i64 = classify(Alpha)\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_ne!(
        combo["report"]["score"], 1,
        "type-only import then construct must NOT be a perfect score: {combo}"
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("unknown constructor") && m.contains("Alpha")),
        "constructing an unimported constructor must be `unknown constructor: Alpha`; got {msgs:?} in {combo}"
    );
}

#[test]
fn type_only_import_then_match_pattern_is_unknown_constructor() {
    // The pattern dual: matching on `Alpha`/`Beta`/`Gamma` in a `match`
    // arm when only the TYPE was imported must be rejected at `check`, not
    // deferred to a runtime non-exhaustive match.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), REEF_TOML);
    write_file(&root.join("src/adt.ch"), ADT);
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.Adt (Mode, classify)\n\
         def relabel(m: Mode) -> i64 = match m with { | Alpha => 10 | Beta => 11 | Gamma => 12 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_ne!(
        combo["report"]["score"], 1,
        "type-only import then match must NOT be a perfect score: {combo}"
    );
    assert!(
        msgs.iter().any(|m| m.contains("unknown constructor")
            && (m.contains("Alpha") || m.contains("Beta") || m.contains("Gamma"))),
        "matching an unimported constructor must be `unknown constructor`; got {msgs:?} in {combo}"
    );
}

// ── POSITIVE: in-scope constructors resolve and check clean ─────────────

#[test]
fn import_by_name_construct_and_match_checks_clean() {
    // The documented migration: naming the constructors in the import list
    // brings them into scope (even though the declaring module's export list
    // names only the type). Both the construction site and the match pattern
    // then resolve and `check` clean.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), REEF_TOML);
    write_file(&root.join("src/adt.ch"), ADT);
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.Adt (Mode, Alpha, Beta, Gamma, classify)\n\
         def use_alpha() -> i64 = classify(Alpha)\n\
         def relabel(m: Mode) -> i64 = match m with { | Alpha => 10 | Beta => 11 | Gamma => 12 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_eq!(
        combo["report"]["score"], 1,
        "imported-by-name constructors must resolve and check clean: {combo}"
    );
    assert!(
        msgs.is_empty(),
        "no errors expected when the constructors are imported by name; got {msgs:?} in {combo}"
    );
}

#[test]
fn local_constructor_construct_and_match_checks_clean() {
    // Locally declared constructors are in scope without any import, at both
    // the construction site and the match pattern. Guards against
    // over-rejection: the chelis#317 tightening must not flag own-module
    // constructors.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), REEF_TOML);
    write_file(
        &root.join("src/local.ch"),
        "module Pkg.Local\n\
         type Color = | Red | Green | Blue\n\
         def first() -> Color = Red\n\
         def rank(c: Color) -> i64 = match c with { | Red => 0 | Green => 1 | Blue => 2 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "local.ch");
    let msgs = error_messages(combo);
    assert_eq!(
        combo["report"]["score"], 1,
        "local constructors must resolve and check clean: {combo}"
    );
    assert!(
        msgs.is_empty(),
        "no errors expected for own-module constructors; got {msgs:?} in {combo}"
    );
}

#[test]
fn module_qualified_constructor_checks_clean() {
    // The module-qualified form (#316) is also in scope without naming the
    // constructor in the import list: `import Pkg.Adt ()` then
    // `Pkg.Adt.Alpha` resolves through the qualified-module map, so the
    // chelis#317 guard must not reject it.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), REEF_TOML);
    write_file(&root.join("src/adt.ch"), ADT);
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.Adt ()\n\
         def use_alpha() -> i64 = Pkg.Adt.classify(Pkg.Adt.Alpha)\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_eq!(
        combo["report"]["score"], 1,
        "module-qualified constructor must resolve and check clean: {combo}"
    );
    assert!(
        msgs.is_empty(),
        "no errors expected for the module-qualified form; got {msgs:?} in {combo}"
    );
}

#[test]
fn builtin_option_constructors_check_clean() {
    // Negative-parity guard for the builtins: `Some`/`None` are bare prelude
    // constructors (registered bare, not reef-mangled), so the chelis#317
    // exact-scope check must continue to accept them rather than flag them as
    // out of scope.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), REEF_TOML);
    write_file(
        &root.join("src/opt.ch"),
        "module Pkg.Opt\n\
         def wrap() -> Option[i64] = Some(7)\n\
         def empty() -> Option[i64] = None\n\
         def unwrap_or(o: Option[i64]) -> i64 = match o with { | Some(n) => n | None => 0 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "opt.ch");
    let msgs = error_messages(combo);
    assert_eq!(
        combo["report"]["score"], 1,
        "builtin Option constructors must resolve and check clean: {combo}"
    );
    assert!(
        msgs.is_empty(),
        "no errors expected for builtin Some/None; got {msgs:?} in {combo}"
    );
}
