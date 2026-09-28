//! chelis#2449: C emission for an ADT `match` points the scrutinee's owner at
//! a temporary declared in the enclosing C block. That alias must end with
//! the match, or a sibling branch that releases the same owner names a
//! variable declared inside another block and the emitted C does not compile.

mod common;

use assert_cmd::Command;
use common::{build_and_run, write_file};

fn evaluate(source: &str, name: &str) -> String {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .expect("eval should run");
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8 stdout")
}

/// REGRESSION TEST. On the base sha the emitted C failed to compile with
/// "use of undeclared identifier '__adt_1'" in the `else` branch, which
/// releases `w` without matching it.
#[test]
fn an_adt_match_in_one_branch_compiles_when_the_other_branch_releases_its_scrutinee() {
    let source = r#"
type Wrap =
  | Wrap(Option[i64])
def g(w: Wrap, c: bool) -> i64 =
  if c then
    match w with {
      | Wrap(o) => 5i64
    }
  else 1i64
e = g(Wrap(Some(20i64)), true)
h = g(Wrap(None), false)
"#;
    let interpreted = evaluate(source, "adt_match_branch");
    let compiled = build_and_run(source, "adt_match_branch");
    assert_eq!(
        compiled, interpreted,
        "eval/C output must be byte-identical"
    );
    for expected in ["e = 5", "h = 1"] {
        assert!(
            interpreted.lines().any(|line| line == expected),
            "{interpreted}"
        );
    }
}
