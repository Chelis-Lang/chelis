//! chelis#1742: the runtime-extent oracle's reviewed test-target manifest must
//! agree with the sources it names.
//!
//! `scripts/runtime_extent_oracle.py` runs a fixed set of cargo commands and
//! requires each one's per-test receipts to EQUAL a reviewed list. Before this
//! test the list lived in Python tuples that only the oracle read, so a rename
//! in another pull request moved the binary and left the tuple behind; the
//! oracle then failed before it reached any corpus row, silently for everyone
//! who did not run it. chelis#1664 and chelis#1668 each did exactly that.
//!
//! This is the cheap half of the repair. It parses every source the manifest
//! names, applies the row's selector to the `#[test]` functions it finds, and
//! compares the result with the row's `expected` list, so the renaming pull
//! request fails `scripts/gate.py --fast` instead of the next oracle run. The
//! expensive half stays where it belongs: the oracle still executes the
//! binaries and reads their real receipts, which is the only thing that proves
//! the tests pass.
//!
//! What this test does NOT claim: that a test does what its name says, that a
//! `--lib` row's module path resolves the way the harness resolves it, or that
//! the selected tests pass. A syntactic name census cannot show any of those.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("chelis-types lives under <workspace>/crates")
        .to_path_buf()
}

fn manifest_path() -> PathBuf {
    workspace_root().join("scripts/runtime_extent_oracle_targets.json")
}

/// One `#[test]` function discovered in a source tree, keyed by the path the
/// harness would print for it.
type Inventory = BTreeMap<String, bool>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selector<'a> {
    All,
    Substring(&'a str),
    Exact,
}

#[derive(Debug)]
struct Row {
    phase: String,
    id: String,
    package: String,
    kind: String,
    file: String,
    selector_mode: String,
    selector_value: Option<String>,
    expected: Vec<String>,
    list_only: bool,
}

impl Row {
    fn selector(&self) -> Selector<'_> {
        match self.selector_mode.as_str() {
            "all" => Selector::All,
            "substring" => Selector::Substring(
                self.selector_value
                    .as_deref()
                    .expect("a substring selector carries its value"),
            ),
            "exact" => Selector::Exact,
            other => panic!("{}: unknown selector mode {other}", self.id),
        }
    }

    /// Whether the row's expectation is the complete selected inventory.
    ///
    /// An `all` or `substring` row selects whatever its target happens to
    /// hold, so its expectation must be an equality: that is the case a
    /// rename or an addition breaks, and the case this test exists for. A
    /// `--lib` row is held to it over the module file it names, so a test
    /// added there under a substring selector fails here rather than in the
    /// nightly oracle (chelis#2941). An `exact` row names its tests on the
    /// command line, so `cargo test --exact` already fails loudly when one of
    /// them is gone; it is checked for containment instead.
    fn requires_equality(&self) -> bool {
        !matches!(self.selector(), Selector::Exact)
    }
}

fn string_field(row: &Value, field: &str, id: &str) -> String {
    row.get(field)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("manifest row {id} has no string {field}"))
        .to_string()
}

fn parse_rows(manifest: &Value) -> Vec<Row> {
    assert_eq!(
        manifest.get("schema_version").and_then(Value::as_u64),
        Some(1),
        "unsupported target manifest schema"
    );
    let targets = manifest
        .get("targets")
        .and_then(Value::as_array)
        .expect("target manifest holds a targets array");
    assert!(!targets.is_empty(), "target manifest must not be empty");
    targets
        .iter()
        .map(|row| {
            let id = row
                .get("id")
                .and_then(Value::as_str)
                .expect("every manifest row has an id")
                .to_string();
            let selector = row
                .get("selector")
                .unwrap_or_else(|| panic!("manifest row {id} has no selector"));
            let expected = row
                .get("expected")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("manifest row {id} has no expected list"))
                .iter()
                .map(|name| {
                    name.as_str()
                        .unwrap_or_else(|| panic!("manifest row {id} expects a non-string"))
                        .to_string()
                })
                .collect::<Vec<_>>();
            Row {
                phase: string_field(row, "phase", &id),
                package: string_field(row, "package", &id),
                kind: string_field(row, "kind", &id),
                file: string_field(row, "file", &id),
                selector_mode: string_field(selector, "mode", &id),
                selector_value: selector
                    .get("value")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                expected,
                list_only: row
                    .get("list_only")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                id,
            }
        })
        .collect()
}

fn parse_source(path: &Path) -> syn::File {
    let source =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    syn::parse_file(&source).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn has_attribute(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident(name))
}

fn path_attribute(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|attr| {
        if !attr.path().is_ident("path") {
            return None;
        }
        match &attr.meta {
            syn::Meta::NameValue(pair) => match &pair.value {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(literal),
                    ..
                }) => Some(literal.value()),
                _ => None,
            },
            _ => None,
        }
    })
}

/// The conditional-compilation attribute that makes an item's presence, or
/// its `#[ignore]`, unreadable from the source alone.
///
/// `#[cfg(test)]` is the exception and returns `None`: every source a
/// manifest row names is compiled as a test binary, so that gate is always
/// on. Everything else is refused rather than guessed. `#[cfg_attr(..)]` can
/// attach `ignore` under a condition this parser cannot evaluate, and a
/// `#[cfg(..)]` on a test or on a module holding tests decides whether the
/// binary contains them at all; counting either one unconditionally would
/// make the comparison quietly wrong instead of loudly unavailable.
fn conditional_gate(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|attr| {
        if attr.path().is_ident("cfg_attr") {
            return Some("#[cfg_attr(...)]".to_string());
        }
        if !attr.path().is_ident("cfg") {
            return None;
        }
        let tokens = attr
            .meta
            .require_list()
            .map(|list| list.tokens.to_string())
            .unwrap_or_default();
        (tokens.trim() != "test").then(|| format!("#[cfg({tokens})]"))
    })
}

/// Where a file-backed `mod NAME;` declaration resolves, by Rust's own two
/// spellings, or the `#[path]` override when one is present.
fn module_source(directory: &Path, name: &str, attrs: &[syn::Attribute]) -> Option<PathBuf> {
    if let Some(relative) = path_attribute(attrs) {
        let candidate = directory.join(relative);
        return candidate.is_file().then_some(candidate);
    }
    let flat = directory.join(format!("{name}.rs"));
    if flat.is_file() {
        return Some(flat);
    }
    let nested = directory.join(name).join("mod.rs");
    nested.is_file().then_some(nested)
}

/// Collect every `#[test]` function reachable from `items`, keyed by the name
/// the test harness prints, with the value recording `#[ignore]`.
///
/// Inline `mod NAME { .. }` blocks are followed without limit: they are the
/// same file. File-backed modules are followed exactly one level, which is
/// what `capacity_census_wire.rs` needs for its three `#[path]` support
/// modules. A second level is not followed; `incomplete` records the module
/// whose contents were not read, so an equality row can refuse to compare a
/// partial inventory.
#[derive(Default)]
struct Scan {
    inventory: Inventory,
    /// Modules whose contents were not read, so the inventory is partial.
    incomplete: Vec<String>,
    /// Tests whose presence or `#[ignore]` is conditional, as (name, reason).
    /// A row refuses to compare when its selector reaches one of these.
    conditional: Vec<(String, String)>,
}

fn collect_tests(
    items: &[syn::Item],
    directory: &Path,
    prefix: &str,
    follow_files: bool,
    gate: Option<&str>,
    scan: &mut Scan,
) {
    for item in items {
        match item {
            syn::Item::Fn(function) if has_attribute(&function.attrs, "test") => {
                let name = format!("{prefix}{}", function.sig.ident);
                if let Some(reason) =
                    conditional_gate(&function.attrs).or_else(|| gate.map(str::to_string))
                {
                    scan.conditional.push((name, reason));
                    continue;
                }
                let ignored = has_attribute(&function.attrs, "ignore");
                assert!(
                    scan.inventory.insert(name.clone(), ignored).is_none(),
                    "two tests named {name}"
                );
            }
            syn::Item::Mod(module) => {
                let nested = format!("{prefix}{}::", module.ident);
                let own = conditional_gate(&module.attrs);
                let inherited = own.as_deref().or(gate);
                if let Some((_, inner)) = &module.content {
                    collect_tests(inner, directory, &nested, follow_files, inherited, scan);
                    continue;
                }
                if !follow_files {
                    scan.incomplete.push(nested);
                    continue;
                }
                let Some(source) =
                    module_source(directory, &module.ident.to_string(), &module.attrs)
                else {
                    scan.incomplete.push(nested);
                    continue;
                };
                let file = parse_source(&source);
                let parent = source
                    .parent()
                    .expect("a module source has a parent directory")
                    .to_path_buf();
                collect_tests(&file.items, &parent, &nested, false, inherited, scan);
            }
            _ => {}
        }
    }
}

/// The full inventory one manifest row's source contributes.
///
/// A `lib` row names its tests by the module path the crate gives them, whose
/// segments follow the source path below the crate's `src` directory.
/// Reconstructing that from a single file is a convention rather than a
/// resolution of the crate's module tree, so a `lib` row's equality covers the
/// file it names, not every test its substring could reach elsewhere.
fn inventory_for(root: &Path, row: &Row) -> Scan {
    let source = root.join(&row.file);
    let file = parse_source(&source);
    let directory = source
        .parent()
        .expect("a manifest source has a parent directory")
        .to_path_buf();
    let prefix = if row.kind == "lib" {
        let stem = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_else(|| panic!("{}: unreadable file stem", row.id));
        assert!(
            stem != "lib" && stem != "mod",
            "{}: a lib row must name the module's own file, not {stem}.rs",
            row.id
        );
        let source_root = root.join("crates").join(&row.package).join("src");
        let relative = source
            .strip_prefix(&source_root)
            .unwrap_or_else(|_| panic!("{}: library source is outside src", row.id))
            .with_extension("");
        let segments = relative
            .iter()
            .map(|part| part.to_str().expect("UTF-8 library module name"))
            .collect::<Vec<_>>();
        format!("{}::", segments.join("::"))
    } else {
        String::new()
    };
    let mut scan = Scan::default();
    collect_tests(&file.items, &directory, &prefix, true, None, &mut scan);
    scan
}

fn selects(row: &Row, name: &str) -> bool {
    match row.selector() {
        Selector::All => true,
        Selector::Substring(value) => name.contains(value),
        Selector::Exact => row.expected.iter().any(|expected| expected == name),
    }
}

fn check_row(root: &Path, row: &Row) -> Result<(), String> {
    let source = root.join(&row.file);
    if !source.is_file() {
        return Err(format!("{}: source {} does not exist", row.id, row.file));
    }
    if !row.file.starts_with(&format!("crates/{}/", row.package)) {
        return Err(format!(
            "{}: source {} is outside package {}",
            row.id, row.file, row.package
        ));
    }
    let Scan {
        inventory,
        incomplete,
        conditional,
    } = inventory_for(root, row);
    // A conditional test is refused when the row's selector reaches it, on
    // every row shape rather than equality rows alone: under `#[cfg_attr]`
    // the unreadable part is the `#[ignore]` check, and that check runs on
    // containment rows too. A conditional test the selector does not reach
    // changes no verdict, so it is left alone. `crates/chelis-ir/src/host.rs`
    // holds one such test behind `#[cfg(debug_assertions)]`, outside the
    // `host_actualization` row's single named test.
    let reached: Vec<String> = conditional
        .iter()
        .filter(|(name, _)| selects(row, name))
        .map(|(name, reason)| format!("{name} carries {reason}"))
        .collect();
    if !reached.is_empty() {
        return Err(format!(
            "{}: cannot read this source's test inventory, {}",
            row.id,
            reached.join("; ")
        ));
    }
    if row.requires_equality() && !incomplete.is_empty() {
        return Err(format!(
            "{}: cannot compare a complete inventory, these modules were not read: {}",
            row.id,
            incomplete.join(", ")
        ));
    }

    let selected: Vec<(&String, &bool)> = inventory
        .iter()
        .filter(|(name, _)| selects(row, name))
        .collect();

    for (name, ignored) in &selected {
        if row.list_only {
            if !**ignored {
                return Err(format!(
                    "{}: {name} is selected by an --ignored --list row but carries no #[ignore]",
                    row.id
                ));
            }
        } else if **ignored {
            // `parse_test_receipt` rejects any outcome that is not `ok`, and
            // an ignored test reports `ignored`, so this breaks the oracle
            // exactly the way a rename does.
            return Err(format!(
                "{}: {name} is selected but carries #[ignore]; the oracle reads its receipt \
                 as `ignored` and fails",
                row.id
            ));
        }
    }

    let observed: Vec<String> = selected.iter().map(|(name, _)| (*name).clone()).collect();
    if row.requires_equality() {
        if observed != row.expected {
            let stale: Vec<&String> = row
                .expected
                .iter()
                .filter(|name| !observed.contains(name))
                .collect();
            let unregistered: Vec<&String> = observed
                .iter()
                .filter(|name| !row.expected.contains(name))
                .collect();
            return Err(format!(
                "{}: selected test drift; expected but absent from the source: {stale:?}; \
                 in the source but unregistered: {unregistered:?}",
                row.id
            ));
        }
        return Ok(());
    }

    let missing: Vec<&String> = row
        .expected
        .iter()
        .filter(|name| !inventory.contains_key(*name))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{}: named tests are absent from {}: {missing:?}",
            row.id, row.file
        ));
    }
    Ok(())
}

fn fixture_row(selector: Value, expected: &[&str], list_only: bool) -> Row {
    fixture_row_over("target_fixture.rs", selector, expected, list_only)
}

fn fixture_row_over(source: &str, selector: Value, expected: &[&str], list_only: bool) -> Row {
    let mut value = serde_json::json!({
        "phase": "a",
        "id": "fixture",
        "package": "chelis-types",
        "kind": "test",
        "file": format!("crates/chelis-types/tests/fixtures/runtime_extent_manifest/{source}"),
        "selector": selector,
        "expected": expected,
    });
    if list_only {
        value["list_only"] = Value::Bool(true);
    }
    parse_rows(&serde_json::json!({"schema_version": 1, "targets": [value]}))
        .pop()
        .expect("one fixture row")
}

// --- the reviewed manifest ------------------------------------------------

#[test]
fn every_manifest_row_matches_the_source_it_names() {
    // Regression test for chelis#1742: on the head that opened the issue,
    // `wire_capacity` expected three renamed tests and missed four, and
    // `literal_claim` missed `literal_extent_example_contract`. Those are
    // exactly the two rows this comparison reports.
    let root = workspace_root();
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(manifest_path()).expect("read the target manifest"),
    )
    .expect("the target manifest is valid JSON");
    let rows = parse_rows(&manifest);
    let failures: Vec<String> = rows
        .iter()
        .filter_map(|row| check_row(&root, row).err())
        .collect();
    assert!(
        failures.is_empty(),
        "the runtime-extent oracle's target manifest disagrees with the sources it names. \
         Update scripts/runtime_extent_oracle_targets.json in the same change set:\n{}",
        failures.join("\n")
    );
}

#[test]
fn every_manifest_row_is_structurally_reviewable() {
    // Disposition lock: the fields the oracle reads and the fields this test
    // reads are the same fields, so neither reader can be satisfied by a row
    // the other would reject.
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(manifest_path()).expect("read the target manifest"),
    )
    .expect("the target manifest is valid JSON");
    let rows = parse_rows(&manifest);
    assert!(rows.len() >= 20, "the manifest lost rows: {}", rows.len());
    for row in &rows {
        assert!(
            row.phase == "a" || row.phase == "b",
            "{}: unknown phase {}",
            row.id,
            row.phase
        );
        assert!(
            row.kind == "test" || row.kind == "lib",
            "{}: unknown kind {}",
            row.id,
            row.kind
        );
        assert!(!row.expected.is_empty(), "{}: no expected tests", row.id);
        let mut sorted = row.expected.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted, row.expected,
            "{}: expected tests must be sorted and unique",
            row.id
        );
        assert!(
            !row.list_only || row.kind == "test",
            "{}: only an integration target can be listed",
            row.id
        );
    }
}

// --- rejection cases ------------------------------------------------------
//
// Each drives `check_row` over a synthetic row pointing at the checked-in
// fixture source, so the rejection comes from a real parse of a real file
// rather than from a mocked inventory. Each is a regression test for one
// drift class rather than a disposition lock: the assertion is that the
// checker REPORTS the defect, and every one of them is red without the
// corresponding arm of `check_row`.

#[test]
fn a_stale_expected_name_is_rejected() {
    // The chelis#1664 shape: a test the reviewed list still names is gone
    // from the source. The selector avoids the fixture's ignored test so
    // that this case fails for the renamed name and nothing else.
    let root = workspace_root();
    let row = fixture_row(
        serde_json::json!({"mode": "substring", "value": "_is_selected"}),
        &[
            "alpha_is_selected",
            "beta_was_renamed",
            "included::tests::epsilon_is_selected",
        ],
        false,
    );
    let error = check_row(&root, &row).expect_err("a stale name must be rejected");
    assert!(
        error.contains("expected but absent from the source: [\"beta_was_renamed\"]"),
        "{error}"
    );
    assert!(
        error.contains("in the source but unregistered: [\"beta_is_selected\"]"),
        "{error}"
    );
}

#[test]
fn a_new_test_under_a_substring_selector_must_be_registered() {
    let root = workspace_root();
    let row = fixture_row(
        serde_json::json!({"mode": "substring", "value": "selector_"}),
        &[],
        false,
    );
    let error = check_row(&root, &row).expect_err("an unregistered selected test is drift");
    assert!(
        error.contains("in the source but unregistered: [\"selector_prefix_delta\"]"),
        "{error}"
    );
}

#[test]
fn an_ignored_selected_test_is_rejected() {
    let root = workspace_root();
    let row = fixture_row(
        serde_json::json!({"mode": "substring", "value": "gamma"}),
        &["gamma_is_a_manual_gate"],
        false,
    );
    let error = check_row(&root, &row).expect_err("an ignored selected test breaks the oracle");
    assert!(error.contains("carries #[ignore]"), "{error}");
}

#[test]
fn an_ignored_row_that_selects_a_running_test_is_rejected() {
    // The inverse obligation, for the HIP manual-gate row: `--ignored --list`
    // names only tests the harness would skip, so a selected test WITHOUT
    // `#[ignore]` would never appear in the listing the oracle parses.
    let root = workspace_root();
    let row = fixture_row(
        serde_json::json!({"mode": "substring", "value": "alpha"}),
        &["alpha_is_selected"],
        true,
    );
    let error = check_row(&root, &row).expect_err("a running test cannot be listed as ignored");
    assert!(error.contains("carries no #[ignore]"), "{error}");
}

#[test]
fn an_exact_selector_whose_named_test_is_absent_is_rejected() {
    let root = workspace_root();
    let row = fixture_row(
        serde_json::json!({"mode": "exact"}),
        &["alpha_is_selected", "zeta_never_existed"],
        false,
    );
    let error = check_row(&root, &row).expect_err("an absent exact name must be rejected");
    assert!(error.contains("[\"zeta_never_existed\"]"), "{error}");
}

#[test]
fn a_conditionally_compiled_test_is_refused_rather_than_counted() {
    // Whether `present_only_under_a_feature` is in the binary depends on a
    // feature this parser cannot evaluate, so the inventory is unreadable
    // rather than short by one. Counting it either way would make an
    // equality row silently wrong.
    let root = workspace_root();
    let row = fixture_row_over(
        "conditional_fixture.rs",
        serde_json::json!({"mode": "substring", "value": "present_only"}),
        &["present_only_under_a_feature"],
        false,
    );
    let error = check_row(&root, &row).expect_err("a cfg-gated test must be refused");
    assert!(
        error.contains("present_only_under_a_feature carries #[cfg(feature"),
        "{error}"
    );
}

#[test]
fn a_conditionally_ignored_test_is_refused_rather_than_counted() {
    // `#[cfg_attr(target_os = "macos", ignore)]` attaches `#[ignore]` under a
    // condition, so the ignore check this tripwire runs cannot be answered
    // from the source. The whole row is refused, containment rows included,
    // because that check runs on those too.
    let root = workspace_root();
    let row = fixture_row_over(
        "conditional_fixture.rs",
        serde_json::json!({"mode": "exact"}),
        &["ignored_only_on_one_platform"],
        false,
    );
    let error = check_row(&root, &row).expect_err("a cfg_attr test must be refused");
    assert!(
        error.contains("ignored_only_on_one_platform carries #[cfg_attr(...)]"),
        "{error}"
    );
}

#[test]
fn a_test_under_a_conditional_module_is_refused_rather_than_counted() {
    // The gate sits on the module rather than on the test, so the refusal
    // has to be inherited down the traversal. Without the inherited arm this
    // test would be counted as unconditionally present. `#[cfg(test)] mod
    // tests` is the exception and stays readable, which the `#[path]`
    // inventory case covers.
    let root = workspace_root();
    let row = fixture_row_over(
        "conditional_fixture.rs",
        serde_json::json!({"mode": "substring", "value": "its_module_feature"}),
        &["gated_module::present_only_under_its_module_feature"],
        false,
    );
    let error = check_row(&root, &row).expect_err("a test under a cfg module must be refused");
    assert!(
        error.contains("gated_module::present_only_under_its_module_feature carries #[cfg(feature"),
        "{error}"
    );
}

#[test]
fn a_row_naming_a_missing_source_is_rejected() {
    let root = workspace_root();
    let mut row = fixture_row(
        serde_json::json!({"mode": "all"}),
        &["alpha_is_selected"],
        false,
    );
    row.file = "crates/chelis-types/tests/fixtures/runtime_extent_manifest/absent.rs".to_string();
    let error = check_row(&root, &row).expect_err("a missing source must be rejected");
    assert!(error.contains("does not exist"), "{error}");
}

#[test]
fn a_nested_library_source_retains_its_outer_module_name() {
    let root = workspace_root();
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(manifest_path()).expect("read the target manifest"),
    )
    .expect("valid manifest");
    let row = parse_rows(&manifest)
        .into_iter()
        .find(|row| row.id == "ir_signature_entry_plan")
        .expect("the nested signature entry receipt is required");
    check_row(&root, &row).expect("the full nested library path must resolve");
}

#[test]
fn a_nested_library_source_rejects_a_shortened_module_name() {
    let root = workspace_root();
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(manifest_path()).expect("read the target manifest"),
    )
    .expect("valid manifest");
    let mut row = parse_rows(&manifest)
        .into_iter()
        .find(|row| row.id == "ir_signature_entry_plan")
        .expect("the nested signature entry receipt is required");
    row.selector_mode = "exact".to_string();
    row.expected = row
        .expected
        .iter()
        .map(|name| {
            name.strip_prefix("host::")
                .expect("outer module")
                .to_string()
        })
        .collect();
    let error = check_row(&root, &row).expect_err("a shortened path must not resolve");
    assert!(error.contains("named tests are absent"), "{error}");
}

#[test]
fn a_library_substring_row_must_register_every_selected_test() {
    // The chelis#2941 shape: a `--lib` row held only to containment let two
    // tests added under its substring selector reach `main` unregistered, and
    // the drift surfaced only as a nightly oracle failure.
    let root = workspace_root();
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(manifest_path()).expect("read the target manifest"),
    )
    .expect("valid manifest");
    let mut row = parse_rows(&manifest)
        .into_iter()
        .find(|row| row.id == "ir_signature_entry_plan")
        .expect("the nested signature entry receipt is required");
    let dropped = row.expected.pop().expect("a registered test");
    let error = check_row(&root, &row).expect_err("an unregistered library test is drift");
    assert!(
        error.contains(&format!("in the source but unregistered: [{dropped:?}]")),
        "{error}"
    );
}

#[test]
fn a_path_module_contributes_its_tests_under_its_module_prefix() {
    // The mechanism `capacity_census_wire` depends on: four of the seven
    // names chelis#1664 left out of the tuple live behind `#[path]` modules,
    // so a parser that stopped at the file would have reported them as
    // unregistered forever.
    let root = workspace_root();
    let row = fixture_row(
        serde_json::json!({"mode": "all"}),
        &["alpha_is_selected"],
        false,
    );
    let scan = inventory_for(&root, &row);
    assert!(scan.incomplete.is_empty(), "{:?}", scan.incomplete);
    assert!(scan.conditional.is_empty(), "{:?}", scan.conditional);
    let inventory = scan.inventory;
    assert_eq!(
        inventory.keys().cloned().collect::<Vec<_>>(),
        vec![
            "alpha_is_selected".to_string(),
            "beta_is_selected".to_string(),
            "gamma_is_a_manual_gate".to_string(),
            "included::tests::epsilon_is_selected".to_string(),
            "selector_prefix_delta".to_string(),
        ]
    );
    assert!(
        inventory["gamma_is_a_manual_gate"],
        "the fixture's manual-gate test carries #[ignore]"
    );
    assert!(
        !inventory["alpha_is_selected"],
        "an ordinary fixture test does not"
    );
}
