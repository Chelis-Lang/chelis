//! chelis#734 - `to_string` on a tensor or list compiled to the literal
//! placeholder string `"<value>"` while eval stringifies the value
//! properly. Root: `host_emit.rs:2236`'s catch-all arm (`_ =>
//! chelis_string_from_cstr("<value>")`) - the same
//! catch-all-returning-a-value shape as #682's stub, one page away from
//! it, substituting a string instead of a zero.
//! The site now rejects loudly; chelis#1059 owns implementing compiled
//! tensor/list stringification.
//!
//! This settled census row 3 of `spec/design/loud_unsupported.md`
//! (previously "unknown - probe"): LIVE. The fix arrives via that plan's
//! Phase 1 (the arm becomes Err through the emitter failure channel).
//!
//! Controls: to_string of scalar values is correct in both lanes. The former
//! f16 "control" passed only because the C host silently widened through the
//! Unknown/f64 path; Phase 2 replaced that accidental green with an explicit
//! ABI rejection. Chelis#729 Phase 3 supplies exact reduced-float storage and
//! returns f16/bf16 to the positive, own-width observation corpus below.

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
            "--emit-c",
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

/// Historical bug: the compiled binary printed the literal `<value>`.
/// Phase 1 now rejects this unsupported container conversion loudly; the
/// ignored positive row remains the support contract.
#[test]
#[ignore = "chelis#1059: to_string(tensor) is now REJECTED loudly at build per the \
            chelis#730 plan (was chelis#734's silent '<value>' placeholder); real tensor \
            rendering un-ignores this value \
            test. Run with \
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

/// Historical bug: lists also produced `<value>`. The current compiled lane
/// rejects this unsupported container conversion loudly.
#[test]
#[ignore = "chelis#1059: to_string(List) is now REJECTED loudly at build per the \
            chelis#730 plan (was chelis#734's silent '<value>' placeholder); real list rendering \
            un-ignores this value test. Run with \
            `cargo test -p chelis-cli --test issue_734_tostring_placeholder -- --ignored`."]
fn to_string_of_a_list_stringifies_in_the_compiled_lane() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(xs: List[i64]) -> string = to_string(xs)\n\
         out = print(f([cast(1, i64), cast(2, i64)]))\n";
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
        ("to_string(cast(7, i64))", "7"),
        ("to_string(cast(1.5, f64))", "1.5"),
        ("to_string(true)", "true"),
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

/// Reduced-float scalar `to_string` used to appear green only because the C
/// host widened the value through its old unknown/f64 representation. Phase 3
/// gives both dtypes exact host storage, so this exit must now preserve their
/// own-width shortest strings and agree with eval.
#[test]
fn to_string_reduced_float_scalars_agree_across_lanes() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for (dtype, value, expected) in [("f16", "0.3333", "0.3333"), ("bf16", "0.334", "0.334")] {
        let program = format!(
            "module M.Main\ndef render() -> string = to_string(cast({value}, {dtype}))\n\
             out = print(render())\n"
        );
        let eval = eval_first_line(&program).expect("eval lane");
        let compiled = c_first_line(&program, &format!("to_string_{dtype}"));
        assert_eq!(eval, expected, "{dtype}: eval own-width rendering drift");
        assert_eq!(compiled, eval, "{dtype}: to_string lane divergence");
    }
}
