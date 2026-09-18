use chelis_lint::rules::carrier_reader_completeness::CarrierReaderCompleteness;
use chelis_lint::{Context, Rule, Surface};
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

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
fn node_bridge_spellings_are_rejected() {
    let violations = check(
        "crates/chelis-ir/src/planted.rs",
        r#"
use chelis_deep::{Node, Node as DeepNode, Span};

fn bridge(node: &Node, span: Span) {
    let _ = node.to_list(span);
}

fn bridge_ufcs(node: DeepNode, span: Span) {
    let _ = DeepNode::to_list(node, span);
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
    let rules: Vec<Box<dyn Rule>> = vec![Box::new(CarrierReaderCompleteness)];

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
    let violations = chelis_lint::lint(dir.path(), &rules).expect("lint empty necessity");
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
    let violations = chelis_lint::lint(dir.path(), &rules).expect("lint justified exception");
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
    for args in [
        &["init", "-q"][..],
        &["config", "user.email", "lint@example.invalid"][..],
        &["config", "user.name", "Lint Test"][..],
        &["add", "."][..],
        &["commit", "-qm", "baseline"][..],
    ] {
        let status = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(args)
            .status()
            .expect("run git fixture command");
        assert!(status.success(), "git {args:?}");
    }

    let rules: Vec<Box<dyn Rule>> = vec![Box::new(CarrierReaderCompleteness)];
    let baseline = chelis_lint::lint(dir.path(), &rules).expect("lint baseline");
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
    let violations = chelis_lint::lint(dir.path(), &rules).expect("lint changed source");
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
