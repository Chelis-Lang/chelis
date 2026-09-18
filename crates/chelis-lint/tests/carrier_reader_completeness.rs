use chelis_lint::rules::carrier_reader_completeness::CarrierReaderCompleteness;
use chelis_lint::{Context, Rule, Surface};
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

const BARE_READER: &str = r#"
use chelis_deep::Expr;
fn read(expr: &Expr) -> bool { matches!(expr, Expr::List(_, _)) }
"#;

fn check(rel: &str, source: &str) -> Vec<chelis_lint::Violation> {
    let root = Path::new("/repo");
    let path = root.join(rel);
    CarrierReaderCompleteness.check(&Context {
        root,
        path: &path,
        source: Some(source),
        surface: Surface::RustSource,
    })
}

fn write_workspace(root: &Path, source: &str) {
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = []\nresolver = \"2\"\n",
    )
    .expect("write workspace manifest");
    let source_dir = root.join("crates/chelis-types/src/infer");
    fs::create_dir_all(&source_dir).expect("create fixture source root");
    fs::write(source_dir.join("reader.rs"), source).expect("write fixture");
}

fn git(root: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .status()
        .expect("run git fixture command");
    assert!(status.success(), "git {args:?}");
}

fn init_repo(root: &Path, branch: &str) {
    git(root, &["init", "-q", "-b", branch]);
    git(root, &["config", "user.email", "lint@example.invalid"]);
    git(root, &["config", "user.name", "Lint Test"]);
}

fn commit_all(root: &Path, message: &str) {
    git(root, &["add", "."]);
    git(root, &["commit", "-qm", message]);
}

fn lint_workspace(root: &Path) -> Vec<chelis_lint::Violation> {
    let rules: Vec<Box<dyn Rule>> = vec![Box::new(CarrierReaderCompleteness)];
    chelis_lint::lint(root, &rules).expect("lint fixture workspace")
}

#[test]
fn guarded_match_arm_is_rejected() {
    let violations = check(
        "crates/chelis-types/src/infer/planted.rs",
        r#"
use chelis_deep::Expr;

fn read(expr: &Expr) -> bool {
    match expr {
        Expr::List(list, _) if list.elements.len() == 3 => true,
        _ => false,
    }
}
"#,
    );
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn guard_keeps_an_otherwise_exhaustive_match_from_escaping() {
    let violations = check(
        "crates/chelis-types/src/infer/planted.rs",
        r#"
use chelis_deep::Expr;

fn read(expr: &Expr) {
    match expr {
        Expr::List(_, _) if false => {}
        Expr::Node(_, _) | Expr::BareList(_, _) | Expr::UnknownForm(_) => {}
        Expr::Atom(_, _) | Expr::Map(_, _) | Expr::MetaExpr(_, _) => {}
        _ => {}
    }
}
"#,
    );
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn aliases_if_let_and_matches_are_rejected() {
    let violations = check(
        "crates/chelis-types/src/infer/planted.rs",
        r#"
use chelis_deep::Expr as DeepExpr;
use chelis_deep::Expr::List as LegacyList;

fn read_if(expr: &DeepExpr) -> bool {
    if let DeepExpr::List(_, _) = expr {
        return true;
    }
    matches!(expr, LegacyList(_, _))
}
"#,
    );
    assert_eq!(violations.len(), 2, "{violations:?}");
}

#[test]
fn chained_module_alias_is_rejected() {
    let violations = check(
        "crates/chelis-types/src/infer/planted.rs",
        r#"
use chelis_deep as deep;
use deep::Expr as E;
fn read(expr: &E) -> bool { matches!(expr, E::List(_, _)) }
"#,
    );
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn node_bridge_spellings_are_rejected() {
    let violations = check(
        "crates/chelis-ir/src/planted.rs",
        r#"
use chelis_deep::{Node, Node as DeepNode, Span};

fn bridge(node: DeepNode, span: Span) {
    let _ = DeepNode::to_list(node, span);
    let _ = <Node>::to_list(node, span);
}
"#,
    );
    assert_eq!(violations.len(), 2, "{violations:?}");
    assert!(
        violations
            .iter()
            .all(|violation| violation.message.contains("Node::to_list")),
        "{violations:?}"
    );
}

#[test]
fn unrelated_to_list_apis_are_allowed() {
    let source = r#"
struct Catalog;
impl Catalog { fn to_list(&self) {} }
fn to_list() {}
fn use_catalog(catalog: &Catalog) {
    catalog.to_list();
    Catalog::to_list(catalog);
    crate::to_list();
}
"#;
    assert!(check("crates/chelis-ir/src/planted.rs", source).is_empty());
}

#[test]
fn constructors_prose_and_exhaustive_matches_are_allowed() {
    let source = r#"
use chelis_deep::{Expr, List, Span};

const NOTE: &str = "Expr::List(list, _) and node.to_list(span)";

fn produce(list: List, span: Span) -> Expr {
    Expr::List(list, span)
}

fn read(expr: &Expr) -> usize {
    match expr {
        Expr::List(list, _) => list.elements.len(),
        Expr::Node(node, _) => node.children_slice().len(),
        Expr::BareList(items, _) => items.len(),
        Expr::UnknownForm(data) => data.children.len(),
        Expr::Atom(_, _) | Expr::Map(_, _) | Expr::MetaExpr(_, _) => 0,
    }
}
"#;
    assert!(check("crates/chelis-types/src/infer/planted.rs", source).is_empty());
}

#[test]
fn inline_exception_requires_a_class_and_nonempty_necessity() {
    let dir = tempdir().expect("tempdir");

    write_workspace(
        dir.path(),
        r#"
use chelis_deep::Expr;
fn read(expr: &Expr) -> bool {
    // chelis-lint: allow carrier-reader-completeness -- producer:
    matches!(expr, Expr::List(_, _))
}
"#,
    );
    let violations = lint_workspace(dir.path());
    assert_eq!(violations.len(), 1, "{violations:?}");

    write_workspace(
        dir.path(),
        r#"
use chelis_deep::Expr;
fn preserve_legacy_output(expr: &Expr) -> bool {
    // chelis-lint: allow carrier-reader-completeness -- symmetric: preserves the input carrier while rebuilding children
    matches!(expr, Expr::List(_, _))
}
"#,
    );
    let violations = lint_workspace(dir.path());
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn existing_debt_is_not_snapshotted_but_a_new_site_is_rejected() {
    let dir = tempdir().expect("tempdir");
    write_workspace(
        dir.path(),
        r#"
use chelis_deep::Expr;
fn existing(expr: &Expr) -> bool {
    matches!(expr, Expr::List(_, _))
}
"#,
    );
    init_repo(dir.path(), "main");
    commit_all(dir.path(), "baseline");

    let baseline = lint_workspace(dir.path());
    assert!(baseline.is_empty(), "{baseline:?}");

    let path = dir.path().join("crates/chelis-types/src/infer/reader.rs");
    let mut source = fs::read_to_string(&path).expect("read fixture");
    source.push_str(
        r#"
fn added(expr: &Expr) -> bool {
    matches!(expr, Expr::List(_, _))
}
"#,
    );
    fs::write(path, source).expect("add new reader");
    let violations = lint_workspace(dir.path());
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn detached_shallow_history_uses_first_parent_as_the_baseline() {
    let dir = tempdir().expect("tempdir");
    write_workspace(dir.path(), "fn baseline() {}\n");
    init_repo(dir.path(), "candidate");
    commit_all(dir.path(), "baseline");

    write_workspace(dir.path(), BARE_READER);
    commit_all(dir.path(), "candidate");
    git(dir.path(), &["checkout", "--detach", "-q", "HEAD"]);

    let violations = lint_workspace(dir.path());
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn committed_source_without_any_baseline_fails_closed() {
    let dir = tempdir().expect("tempdir");
    write_workspace(dir.path(), BARE_READER);
    init_repo(dir.path(), "candidate");
    commit_all(dir.path(), "root candidate");

    let violations = lint_workspace(dir.path());
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn deleting_other_carrier_arms_exposes_the_surviving_list_reader() {
    let dir = tempdir().expect("tempdir");
    write_workspace(
        dir.path(),
        r#"
use chelis_deep::Expr;
fn read(expr: &Expr) -> usize {
    match expr {
        Expr::List(list, _) => list.elements.len(),
        Expr::Node(_, _) | Expr::BareList(_, _) | Expr::UnknownForm(_)
        | Expr::Atom(_, _) | Expr::Map(_, _) | Expr::MetaExpr(_, _) => 0,
    }
}
"#,
    );
    init_repo(dir.path(), "main");
    commit_all(dir.path(), "baseline");

    write_workspace(
        dir.path(),
        r#"
use chelis_deep::Expr;
fn read(expr: &Expr) -> usize {
    match expr {
        Expr::List(list, _) => list.elements.len(),
    }
}
"#,
    );

    let violations = lint_workspace(dir.path());
    assert_eq!(violations.len(), 1, "{violations:?}");
}

#[test]
fn chelis_deep_owns_the_raw_representation() {
    assert!(
        check(
            "crates/chelis-deep/src/ast.rs",
            "fn read(expr: &Expr) { let Expr::List(_, _) = expr else { return; }; }",
        )
        .is_empty()
    );
}
