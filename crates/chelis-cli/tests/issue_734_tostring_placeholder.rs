//! chelis#734 - `to_string` on a tensor or list compiles to the literal
//! placeholder string `"<value>"` while eval stringifies the value
//! properly. Root: `host_emit.rs:2236`'s catch-all arm (`_ =>
//! chelis_string_from_cstr("<value>")`) - the same
//! catch-all-returning-a-value shape as #682's stub, one page away from
//! it, substituting a string instead of a zero.
//!
//! This settled census row 3 of `spec/design/loud_unsupported.md`
//! (previously "unknown - probe"): LIVE. The fix arrives via that plan's
//! Phase 1 (the arm becomes Err through the emitter failure channel).
//!
//! Controls: to_string of int64/f64/bool scalars is correct in both
//! lanes, and to_string(cast(1.5, f16)) prints 1.5 in both lanes (the
//! f16 scalar resolves through the f64 formatter arm here, so #714's
//! Unknown path does not compound).

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn eval_first_line(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

fn c_first_line(program: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link failed for {name}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    String::from_utf8_lossy(&run.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Observed today: the compiled binary prints the literal `<value>`.
#[test]
#[ignore = "chelis#734: to_string(tensor) compiles to the literal string '<value>'; eval \
            prints tensor(shape=[2], data=[1.5, 2.5]). Run with \
            `cargo test -p chelis-cli --test issue_734_tostring_placeholder -- --ignored`."]
fn to_string_of_a_tensor_stringifies_in_the_compiled_lane() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[2, f32]) -> string = to_string(x)\n\
         out = print(f(to_tensor([1.5, 2.5])))\n";
    let eval_got = eval_first_line(program).expect("eval");
    assert_eq!(eval_got, "tensor(shape=[2], data=[1.5, 2.5])");
    let c_got = c_first_line(program, "ts_tensor");
    assert_eq!(
        c_got, eval_got,
        "to_string of a tensor must stringify, not emit a placeholder"
    );
}

/// Observed today: `<value>` for lists as well.
#[test]
#[ignore = "chelis#734: to_string(List[int64]) compiles to the literal string '<value>'; \
            eval prints [1, 2]. Run with \
            `cargo test -p chelis-cli --test issue_734_tostring_placeholder -- --ignored`."]
fn to_string_of_a_list_stringifies_in_the_compiled_lane() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(xs: List[int64]) -> string = to_string(xs)\n\
         out = print(f([cast(1, int64), cast(2, int64)]))\n";
    let eval_got = eval_first_line(program).expect("eval");
    assert_eq!(eval_got, "[1, 2]");
    let c_got = c_first_line(program, "ts_list");
    assert_eq!(
        c_got, eval_got,
        "to_string of a list must stringify, not emit a placeholder"
    );
}

/// The scalar arms are correct in both lanes - bounds #734 to the
/// catch-all, and locks the working surface while the #730 conversion
/// lands.
#[test]
fn to_string_scalar_arms_agree_across_lanes() {
    let rows = [
        ("to_string(cast(7, int64))", "7"),
        ("to_string(cast(1.5, f64))", "1.5"),
        ("to_string(true)", "true"),
        ("to_string(cast(1.5, f16))", "1.5"),
    ];
    let have_cc = c_toolchain_available();
    for (i, (expr, expected)) in rows.iter().enumerate() {
        let program = format!("module M.Main\ndef f() -> string = {expr}\nout = print(f())\n");
        assert_eq!(
            eval_first_line(&program).expect("eval"),
            *expected,
            "{expr}"
        );
        if have_cc {
            assert_eq!(
                c_first_line(&program, &format!("ts_ctl_{i}")),
                *expected,
                "{expr}"
            );
        }
    }
}
