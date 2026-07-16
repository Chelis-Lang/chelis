// Regression for chelis#316: two modules in one package each declare
// `type Mode = | Train | Eval`. Bare ADT constructors resolve in a single
// flat namespace, so a third module that needs both cannot import them
// unqualified — the same-named constructors collide and the program fails
// to compile (the original report saw
// `type mismatch: Pkg__demo__Demo__Dropout__Mode vs Pkg__demo__Demo__Sd__Mode`).
//
// Resolution (ask #2 in the issue): a module-qualified constructor / value
// expression. The reef name resolver already mapped `Module.Name` chains to
// each module's own internal name, and the ambiguity diagnostic already told
// users to "Qualify the reference (e.g. `Module.Eval`)" — but the parser
// rejected an uppercase segment after `.` (and a `(args)` call after any
// dotted path), so the recommended syntax did not parse. The parser fix makes
// `Demo.Dropout.Eval` and `Demo.Dropout.use(m)` parse into an `Access`/`Apply`
// chain that reef resolves to the declaring module's constructor.
//
// These tests drive the real `chelis check` binary over a no-dependency reef
// package so the product surface — not just the library pipeline — is covered.

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
         name = \"demo\"\n\
         version = \"0.1.0\"\n\
         compiler = \"={ver}\"\n\
         module_prefix = \"Demo\"\n",
        ver = chelis_compiler_api::COMPILER_VERSION,
    )
}

const DROPOUT: &str = "module Demo.Dropout\n\
     export (Mode, Train, Eval, use)\n\
     type Mode = | Train | Eval\n\
     def use(m: Mode) -> i64 = match m with { | Train => 1 | Eval => 0 }\n";

const SD: &str = "module Demo.Sd\n\
     export (Mode, Train, Eval, use)\n\
     type Mode = | Train | Eval\n\
     def use(m: Mode) -> i64 = match m with { | Train => 1 | Eval => 0 }\n";

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

#[test]
fn qualified_constructor_references_disambiguate_same_named_modes() {
    // The headline fix: `combo` reaches each module's own `Mode` constructor
    // through a qualified path, so the two `Mode` ADTs coexist in one build.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/dropout.ch"), DROPOUT);
    write_file(&root.join("src/sd.ch"), SD);
    write_file(
        &root.join("src/combo.ch"),
        "module Demo.Combo\n\
         import Demo.Dropout ()\n\
         import Demo.Sd ()\n\
         def go() -> i64 = add(Demo.Dropout.use(Demo.Dropout.Eval), Demo.Sd.use(Demo.Sd.Train))\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "combo.ch");
    let report = &combo["report"];
    assert_eq!(
        report["score"], 1,
        "qualified references must let both Modes coexist: {combo}"
    );
    assert!(
        report["errors"]
            .as_array()
            .expect("errors array")
            .is_empty(),
        "no errors expected for the qualified combo: {combo}"
    );
}

#[test]
fn unqualified_import_of_both_modes_is_still_ambiguous() {
    // Negative parity: qualification is the *escape hatch*, not a silent
    // override of the ambiguity guard. Importing both `Eval`s unqualified must
    // still be rejected with the chelis#157 ambiguity diagnostic that points
    // at qualification — exactly the path the headline test then takes.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/dropout.ch"), DROPOUT);
    write_file(&root.join("src/sd.ch"), SD);
    write_file(
        &root.join("src/combo.ch"),
        "module Demo.Combo\n\
         import Demo.Dropout (Mode, Train, Eval, use)\n\
         import Demo.Sd (Mode, Train, Eval, use)\n\
         def go() -> i64 = add(use(Train), use(Eval))\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "combo.ch");
    // A reef-level resolution failure surfaces as a top-level `error` string
    // rather than a per-file `report`.
    let blob = combo.to_string();
    assert!(
        blob.contains("ambiguous reference") && blob.contains("Eval"),
        "importing both Modes unqualified must stay ambiguous; got {combo}"
    );
}

#[test]
fn qualified_reference_to_unexported_name_is_rejected() {
    // Negative parity: a qualified path whose head names an imported module
    // but whose tail that module does not export must be rejected, not silently
    // accepted. `Demo.Dropout` does not export `Missing`. reef rejects it with
    // a `does not export` error in expression position (and identically in
    // pattern and type position — see the tests below).
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/dropout.ch"), DROPOUT);
    write_file(
        &root.join("src/combo.ch"),
        "module Demo.Combo\n\
         import Demo.Dropout ()\n\
         def go() -> i64 = Demo.Dropout.use(Demo.Dropout.Missing)\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "combo.ch");
    let blob = combo.to_string();
    assert!(
        blob.contains("does not export") && blob.contains("Missing"),
        "qualified reference to an unexported name must be rejected; got {combo}"
    );
}

#[test]
fn qualified_pattern_to_unexported_name_is_rejected() {
    // The pattern position must reject an unknown qualified leaf as loudly as
    // the expression position — not let the typo vanish into a dead `match`
    // arm. `| Demo.Dropout.Missing =>` names a constructor Dropout doesn't
    // export.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/dropout.ch"), DROPOUT);
    write_file(
        &root.join("src/combo.ch"),
        "module Demo.Combo\n\
         import Demo.Dropout ()\n\
         def go() -> i64 = match Demo.Dropout.Train with { | Demo.Dropout.Train => 1 | Demo.Dropout.Missing => 0 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "combo.ch");
    let blob = combo.to_string();
    assert!(
        blob.contains("does not export") && blob.contains("Missing"),
        "qualified pattern to an unexported constructor must be rejected; got {combo}"
    );
}

#[test]
fn qualified_constructor_patterns_match_per_module() {
    // Destructuring dual of the headline test: a `match` whose arms qualify
    // against one module (`| Demo.Dropout.Train =>`) binds that module's
    // variants, so two same-named `Mode` ADTs can be matched in one build
    // without renaming. The scrutinee is a qualified constructor expression so
    // no qualified *type* annotation is needed.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/dropout.ch"), DROPOUT);
    write_file(&root.join("src/sd.ch"), SD);
    write_file(
        &root.join("src/combo.ch"),
        "module Demo.Combo\n\
         import Demo.Dropout ()\n\
         import Demo.Sd ()\n\
         def classify_dropout() -> i64 = match Demo.Dropout.Train with { | Demo.Dropout.Train => 1 | Demo.Dropout.Eval => 0 }\n\
         def classify_sd() -> i64 = match Demo.Sd.Eval with { | Demo.Sd.Train => 1 | Demo.Sd.Eval => 0 }\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "combo.ch");
    let report = &combo["report"];
    assert_eq!(
        report["score"], 1,
        "qualified constructor patterns must type-check per module: {combo}"
    );
    assert!(
        report["errors"]
            .as_array()
            .expect("errors array")
            .is_empty(),
        "no errors expected for the qualified-pattern combo: {combo}"
    );
}

#[test]
fn qualified_type_annotation_resolves_per_module() {
    // The third qualification position: a module-qualified *type* name in an
    // annotation. `relay` pins its parameter to `Demo.Dropout.Mode` even
    // though both modules export a `Mode`, and forwards it to that module's
    // `use`.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/dropout.ch"), DROPOUT);
    write_file(&root.join("src/sd.ch"), SD);
    write_file(
        &root.join("src/combo.ch"),
        "module Demo.Combo\n\
         import Demo.Dropout ()\n\
         import Demo.Sd ()\n\
         def relay(m: Demo.Dropout.Mode) -> i64 = Demo.Dropout.use(m)\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "combo.ch");
    let report = &combo["report"];
    assert_eq!(
        report["score"], 1,
        "qualified type annotation must resolve and type-check: {combo}"
    );
    assert!(
        report["errors"]
            .as_array()
            .expect("errors array")
            .is_empty(),
        "no errors expected for the qualified-type combo: {combo}"
    );
}

#[test]
fn qualified_type_annotation_distinguishes_modules() {
    // Negative parity for the type feature: a value of `Demo.Dropout.Mode`
    // passed to `Demo.Sd.use` (which wants `Demo.Sd.Mode`) must be a type
    // mismatch — proving the qualified annotation resolves to a distinct type,
    // not a shared/erased one.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/dropout.ch"), DROPOUT);
    write_file(&root.join("src/sd.ch"), SD);
    write_file(
        &root.join("src/combo.ch"),
        "module Demo.Combo\n\
         import Demo.Dropout ()\n\
         import Demo.Sd ()\n\
         def bad(m: Demo.Dropout.Mode) -> i64 = Demo.Sd.use(m)\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "combo.ch");
    let messages: Vec<String> = combo["report"]["errors"]
        .as_array()
        .map(|errs| {
            errs.iter()
                .map(|e| e["message"].as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    assert!(
        messages.iter().any(|m| m.contains("type mismatch")
            && m.contains("Sd__Mode")
            && m.contains("Dropout__Mode")),
        "mixing two modules' qualified Mode types must be a mismatch; got {combo}"
    );
}

#[test]
fn qualified_type_to_unexported_name_is_rejected() {
    // The type position must reject an unknown qualified leaf as loudly as the
    // expression and pattern positions — not silently accept it as an opaque
    // type. `Demo.Dropout` exports `Mode`, never `Nope`.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_file(&root.join("reef.toml"), &reef_toml());
    write_file(&root.join("src/dropout.ch"), DROPOUT);
    write_file(
        &root.join("src/combo.ch"),
        "module Demo.Combo\n\
         import Demo.Dropout ()\n\
         def relay(m: Demo.Dropout.Nope) -> i64 = 0\n",
    );

    let json = check_package(root);
    let combo = file_entry(&json, "combo.ch");
    let blob = combo.to_string();
    assert!(
        blob.contains("does not export") && blob.contains("Nope"),
        "qualified type to an unexported name must be rejected; got {combo}"
    );
}
