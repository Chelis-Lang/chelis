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

fn reef_toml() -> String {
    format!(
        "[package]\n\
         name = \"adt\"\n\
         version = \"0.1.0\"\n\
         compiler = \"={ver}\"\n\
         module_prefix = \"Pkg\"\n",
        ver = chelis_compiler_api::COMPILER_VERSION,
    )
}

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
        .unwrap_or_else(|e| panic!("check output must be json: {e}\n{output:?}"))
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
    write_file(&root.join("reef.toml"), &reef_toml());
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
    write_file(&root.join("reef.toml"), &reef_toml());
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
    write_file(&root.join("reef.toml"), &reef_toml());
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
    write_file(&root.join("reef.toml"), &reef_toml());
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
    write_file(&root.join("reef.toml"), &reef_toml());
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
    write_file(&root.join("reef.toml"), &reef_toml());
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

#[test]
fn builtin_list_constructors_check_clean() {
    // chelis#317 review (5c): `Cons`/`Nil` are also bare prelude constructors.
    // The exact-scope check must accept them at both construction and match
    // sites, the same as `Some`/`None`.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(
        &root.join("src/lst.ch"),
        "module Pkg.Lst\n\
         def one() -> List[i64] = Cons(cast(1, int64), Nil)\n\
         def head_or(xs: List[i64]) -> i64 = match xs with { | Cons(h, _) => h | Nil => 0 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "lst.ch");
    let msgs = error_messages(combo);
    assert_eq!(
        combo["report"]["score"], 1,
        "builtin List Cons/Nil constructors must resolve and check clean: {combo}"
    );
    assert!(
        msgs.is_empty(),
        "no errors expected for builtin Cons/Nil; got {msgs:?} in {combo}"
    );
}

// ── RECORD-shaped constructors (chelis#317 review item 3) ───────────────

// A record-shaped ADT whose declaring module exports only the type. The
// constructor name (`AdamState`) deliberately DIFFERS from the type name
// (`Adam`): the issue names `Adam`'s `AdamState` record constructor among the
// regressed cases, and only a distinct-named constructor is genuinely out of
// scope under a type-only import. (When the constructor shares the type's name,
// importing the type already brings the identically-mangled constructor into
// scope, so it is not the bug surface.)
const REC_ADT: &str = "module Pkg.Rec\n\
     export (Adam, use)\n\
     type Adam = | AdamState { rate: int64 }\n\
     def use(c: Adam) -> i64 = match c with { | AdamState { rate } => rate }\n";

#[test]
fn type_only_import_then_record_construct_is_unknown_constructor() {
    // The record-construction dual of the headline defect: importing only the
    // TYPE `Adam` and then constructing `AdamState { rate: .. }` must be
    // rejected at check. Record construction lowers through a separate IR
    // builder and was previously unchecked at type-check, so this site
    // mis-resolved silently.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/rec.ch"), REC_ADT);
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.Rec (Adam)\n\
         def make() -> Adam = AdamState { rate: cast(7, int64) }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_ne!(
        combo["report"]["score"], 1,
        "type-only import then record construct must NOT be a perfect score: {combo}"
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("unknown constructor") && m.contains("AdamState")),
        "constructing an unimported record constructor must be \
         `unknown constructor: AdamState`; got {msgs:?} in {combo}"
    );
}

#[test]
fn import_record_constructor_by_name_checks_clean() {
    // Over-rejection guard for the record path: naming the record constructor
    // in the import brings it into scope, so `AdamState { rate: .. }` resolves
    // and checks clean.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/rec.ch"), REC_ADT);
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.Rec (Adam, AdamState, use)\n\
         def make() -> i64 = use(AdamState { rate: cast(7, int64) })\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_eq!(
        combo["report"]["score"], 1,
        "imported-by-name record constructor must resolve and check clean: {combo}"
    );
    assert!(
        msgs.is_empty(),
        "no errors expected when the record constructor is imported by name; \
         got {msgs:?} in {combo}"
    );
}

// ── Non-unique terminal + wildcard arm (chelis#317 review item 4) ────────

#[test]
fn ambiguous_foreign_constructor_pattern_under_wildcard_is_rejected() {
    // Two modules each declare `type Mode = | Dup`, and a consumer imports
    // NEITHER constructor but matches `| Dup => .. | _ => ..`. The terminal
    // `Dup` is non-unique, so `lookup_terminal_unique` returns None and the
    // arm would push a bare unresolved name into the coverage set; the `_`
    // wildcard then suppresses the would-be `non-exhaustive` diagnostic. The
    // pattern guard must still reject `Dup` as an unknown constructor — the
    // reference resolves to neither module's constructor and must not be
    // silently accepted.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(
        &root.join("src/one.ch"),
        "module Pkg.One\n\
         export (Mode)\n\
         type Mode = | Dup\n",
    );
    write_file(
        &root.join("src/two.ch"),
        "module Pkg.Two\n\
         export (Mode)\n\
         type Mode = | Dup\n",
    );
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.One ()\n\
         import Pkg.Two ()\n\
         def label(n: i64) -> i64 = match n with { | Dup => 1 | _ => 0 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_ne!(
        combo["report"]["score"], 1,
        "ambiguous foreign constructor pattern under a wildcard must NOT score perfect: {combo}"
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("unknown constructor") && m.contains("Dup")),
        "a non-unique foreign constructor pattern must be `unknown constructor: Dup`, \
         even under a `_` wildcard arm; got {msgs:?} in {combo}"
    );
}

// ── Qualified pattern + nested pattern (chelis#317 review item 5) ────────

#[test]
fn module_qualified_constructor_pattern_checks_clean() {
    // Review item 5a: the module-qualified form (#316) is in scope in PATTERN
    // position too, without naming the constructor in the import list. Pins the
    // §P2 qualified-pattern claim against over-rejection.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/adt.ch"), ADT);
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.Adt (Mode)\n\
         def relabel(m: Mode) -> i64 = match m with { \
           | Pkg.Adt.Alpha => 10 | Pkg.Adt.Beta => 11 | Pkg.Adt.Gamma => 12 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_eq!(
        combo["report"]["score"], 1,
        "module-qualified constructor pattern must resolve and check clean: {combo}"
    );
    assert!(
        msgs.is_empty(),
        "no errors expected for the module-qualified pattern form; got {msgs:?} in {combo}"
    );
}

#[test]
fn nested_out_of_scope_constructor_pattern_is_unknown_constructor() {
    // Review item 5b: an out-of-scope constructor nested inside an in-scope
    // constructor's field pattern (`| Some(Alpha) =>`, with `Alpha`
    // type-only-imported) must still be rejected — verifies the recursive
    // `pattern_bindings` re-enters the guard on sub-patterns.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/adt.ch"), ADT);
    write_file(
        &root.join("src/consumer.ch"),
        "module Pkg.Consumer\n\
         import Pkg.Adt (Mode)\n\
         def peek(o: Option[Mode]) -> i64 = match o with { | Some(Alpha) => 1 | _ => 0 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "consumer.ch");
    let msgs = error_messages(combo);
    assert_ne!(
        combo["report"]["score"], 1,
        "nested out-of-scope constructor pattern must NOT score perfect: {combo}"
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("unknown constructor") && m.contains("Alpha")),
        "a nested out-of-scope constructor pattern must be `unknown constructor: Alpha`; \
         got {msgs:?} in {combo}"
    );
}
