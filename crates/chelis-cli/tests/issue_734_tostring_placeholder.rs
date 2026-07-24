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
//! Controls: to_string of int64/f64/bool scalars is correct in both lanes.
//! The former f16 "control" passed only because the C host silently widened
//! the value through the Unknown/f64 path; Phase 2 replaces that accidental
//! green with an explicit ABI rejection until chelis#729 supplies real
//! reduced-float storage and rounding. The exact f16/bf16 `to_string` cells
//! are locked below as typed ABI rejections with no emitted C artifact.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use chelis_compiler_api::compiler::compile;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
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
#[ignore = "chelis#734: to_string(tensor) is now REJECTED loudly at build per the \
            chelis#730 plan (was the silent '<value>' placeholder); real tensor \
            rendering arrives with chelis#732's formatter and un-ignores this value \
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

/// Observed today: `<value>` for lists as well.
#[test]
#[ignore = "chelis#734: to_string(List) is now REJECTED loudly at build per the \
            chelis#730 plan (was the silent '<value>' placeholder); real list rendering \
            arrives with chelis#732's formatter and un-ignores this value test. Run with \
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
/// host widened the value through its old unknown/f64 representation. Until
/// chelis#729 supplies exact storage and rounding, both known logical dtypes
/// must reach the structured C-host ABI rejection; they must not widen and no
/// translation unit may be emitted.
#[test]
fn to_string_reduced_float_scalars_reject_before_artifact_emission() {
    for dtype in ["f16", "bf16"] {
        let program = format!(
            "module M.Main\ndef render() -> string = to_string(cast(1.5, {dtype}))\n\
             out = print(render())\n"
        );
        let error = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: program.clone(),
            target: CompileTarget::C,
            entry_name: None,
        })
        .expect_err("reduced-float to_string must not produce a widened C artifact");
        assert_eq!(error.stage, "compile", "{dtype}: {error:?}");
        assert_eq!(error.errors.len(), 1, "{dtype}: {error:?}");
        let diagnostic = &error.errors[0];
        assert_eq!(diagnostic.kind, "unsupported_feature", "{dtype}");
        for expected in [
            "unsupported:",
            &format!("dtype `{dtype}`"),
            "C host ABI selection",
            "(codegen:c)",
        ] {
            assert!(
                diagnostic.message.contains(expected),
                "{dtype}: missing {expected:?} in {diagnostic:?}"
            );
        }

        let dir = tempdir().expect("tempdir");
        let source_path = dir.path().join(format!("to_string_{dtype}.ch"));
        let out_dir = dir.path().join("out");
        write_file(&source_path, &program);
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                source_path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out_dir.to_str().unwrap(),
            ])
            .output()
            .expect("chelis build should run");
        assert!(!output.status.success(), "{dtype}: build must reject");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("unsupported:")
                && stderr.contains(&format!("dtype `{dtype}`"))
                && stderr.contains("C host ABI selection"),
            "{dtype}: {stderr}"
        );
        assert!(
            !out_dir.join(format!("to_string_{dtype}.c")).exists(),
            "{dtype}: rejection must occur before a C translation unit is written"
        );
    }
}
