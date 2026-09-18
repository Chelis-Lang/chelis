use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_workspace(root: &Path, source: &str) {
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = []\nresolver = \"2\"\n",
    )
    .expect("write workspace manifest");
    let infer = root.join("crates/chelis-types/src/infer");
    fs::create_dir_all(&infer).expect("create infer fixture");
    fs::write(infer.join("planted.rs"), source).expect("write planted source");
}

fn lint_workspace(root: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(root)
        .args(["lint", "--check", "."])
        .output()
        .expect("chelis lint --check .")
}

#[test]
fn lint_check_rejects_a_planted_guarded_expr_list_arm() {
    let dir = tempdir().expect("tempdir");
    write_workspace(
        dir.path(),
        r#"
use chelis_deep::Expr::List;

fn read(expr: &chelis_deep::Expr) -> bool {
    match expr {
        List(list, _) if list.elements.len() == 3 => true,
        _ => false,
    }
}
"#,
    );

    let output = lint_workspace(dir.path());
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("carrier-reader-completeness"),
        "missing rule diagnostic:\n{stdout}"
    );
    assert!(
        stdout.contains("crates/chelis-types/src/infer/planted.rs"),
        "missing planted path:\n{stdout}"
    );
}

#[test]
fn lint_check_accepts_a_carrier_total_reader() {
    let dir = tempdir().expect("tempdir");
    write_workspace(
        dir.path(),
        r#"
use chelis_deep::{DeepTag, Expr, ExprCarrier};

fn read(expr: &Expr) -> usize {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Tuple, _, children) => children.len(),
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => 0,
    }
}
"#,
    );

    let output = lint_workspace(dir.path());
    assert!(
        output.status.success(),
        "carrier-total reader must pass: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
