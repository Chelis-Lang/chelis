//! chelis#759: `cast_saturate` and `cast_wrap` are Surf keywords that share
//! the `cast` node shape, as `cast_trunc` does ([05-OP-23], [05-OP-24],
//! spec/02 §1.6). The formatter keeps each rung, the call-stage pipe form
//! inserts the piped value as the source, and the Deep form carries the
//! rung's selector. A bare keyword is a parse error, like every special form.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn chelis(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .expect("chelis should run")
}

const PROGRAM: &str = "module M.Main\n\
                       def sat(x: tensor[3, f32]) -> tensor[3, i8] = x |> cast_saturate(i8)\n\
                       def wrapped(x: i64) -> i8 = cast_wrap(x, i8)\n\
                       a = print(sat(sqrt(to_tensor([90000.0, 4.0, 2.25]))))\n\
                       b = print(wrapped(300i64))\n";

#[test]
fn both_rungs_format_parse_and_evaluate() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(&path, PROGRAM);
    let formatted = chelis(&["fmt", path.to_str().unwrap()]);
    assert!(formatted.status.success(), "fmt must succeed");
    let formatted = String::from_utf8_lossy(&formatted.stdout).into_owned();
    for spelling in ["x |> cast_saturate(i8)", "cast_wrap(x, i8)"] {
        assert!(
            formatted.contains(spelling),
            "the formatter must keep `{spelling}`: {formatted}"
        );
    }

    let deep = chelis(&["deep", path.to_str().unwrap()]);
    assert!(deep.status.success(), "deep must succeed");
    let deep = String::from_utf8_lossy(&deep.stdout).into_owned();
    for selector in ["saturate)", "wrap)"] {
        assert!(
            deep.contains(selector),
            "the Deep form must carry the `{selector}` selector: {deep}"
        );
    }

    let evaluated = chelis(&["eval", "--file", path.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    assert!(
        evaluated.status.success(),
        "eval: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let mut lines = stdout.lines();
    assert_eq!(
        lines.next(),
        Some("tensor(shape=[3], data=[127, 2, 1])"),
        "300 saturates to 127: {stdout}"
    );
    assert_eq!(lines.next(), Some("44"), "300 wraps to 44 at i8: {stdout}");
}

/// The negative twin: like `cast` and `cast_trunc`, a named rung is a special
/// form, so a bare keyword is a parse error rather than a callable value.
#[test]
fn a_bare_rung_keyword_is_a_parse_error() {
    for keyword in ["cast_saturate", "cast_wrap"] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("m.ch");
        write_file(&path, &format!("module M.Main\nf = {keyword}\n"));
        let checked = chelis(&["check", path.to_str().unwrap()]);
        assert!(
            !checked.status.success(),
            "`f = {keyword}` must not parse: {}",
            String::from_utf8_lossy(&checked.stdout)
        );
    }
}
