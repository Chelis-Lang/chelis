//! chelis#3268: calling a function value that a `match` pattern bound.
//!
//! A function stored in a constructor payload, a record field or an `Option`
//! and called after a `match` binds it is well typed, and `chelis eval` runs
//! it. The C host has no first-class function-value ABI until chelis#879, so
//! `chelis build --target c` must refuse such a program with the typed,
//! target-attributed `unsupported:` rejection that cites #879, before any
//! artifact exists. It must never fail inside ownership lowering with a
//! message that names the binder's compiler-generated spelling. Programs
//! whose match binders hold non-function payloads, and contextual callback
//! parameters, keep compiling and agree with eval.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const FN_BOX: &str = "type FnBox =\n  | FnBox(i32 -> i32)\n  | NoBox\n\
                      def inc(z: i32) -> i32 = add(z, 1i32)\n";

#[derive(Debug)]
struct CArtifacts {
    _dir: tempfile::TempDir,
    out_dir: std::path::PathBuf,
}

fn build_c(program: &str, name: &str) -> Result<CArtifacts, (String, Vec<String>)> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().expect("utf-8 path"),
            "--target",
            "c",
            "--output",
            out_dir.to_str().expect("utf-8 path"),
        ])
        .output()
        .expect("chelis build should run");
    if !output.status.success() {
        let artifacts = if out_dir.is_dir() {
            std::fs::read_dir(&out_dir)
                .expect("read output directory")
                .map(|entry| {
                    entry
                        .expect("output entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect()
        } else {
            Vec::new()
        };
        return Err((
            String::from_utf8_lossy(&output.stderr).into_owned(),
            artifacts,
        ));
    }
    Ok(CArtifacts { _dir: dir, out_dir })
}

fn eval_stdout(program: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, program);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("utf-8 path")])
        .output()
        .expect("chelis eval should run");
    assert!(
        output.status.success(),
        "{name}: eval is the reference lane: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn compiled_stdout(program: &str, name: &str) -> String {
    let artifacts =
        build_c(program, name).unwrap_or_else(|(stderr, _)| panic!("{name}: C build: {stderr}"));
    let status = common::link_generated(&artifacts.out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "{name}: generated C must link: {status}");
    let run = std::process::Command::new(artifacts.out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "{name}: compiled binary failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

/// Eval runs `program` and prints `expected`; the C build refuses it with the
/// typed #879 rejection, emits nothing, and never names compiler internals.
fn assert_eval_runs_and_c_rejects_typed(program: &str, name: &str, expected: &str) {
    assert_eq!(eval_stdout(program, name).trim_end(), expected, "{name}");
    let (stderr, artifacts) =
        build_c(program, name).expect_err("C has no first-class function-value ABI (chelis#879)");
    assert_typed_function_value_rejection(name, &stderr, &artifacts);
}

fn assert_typed_function_value_rejection(name: &str, stderr: &str, artifacts: &[String]) {
    for required in [
        "unsupported:",
        "function value",
        "(codegen:c)",
        "chelis#879",
    ] {
        assert!(
            stderr.contains(required),
            "{name}: the rejection must be the typed C-target #879 diagnostic and contain \
             {required:?}:\n{stderr}"
        );
    }
    for internal in [
        "__chelis_pattern_",
        "__inl",
        "unknown callee",
        "ownership lowering",
    ] {
        assert!(
            !stderr.contains(internal),
            "{name}: the rejection must not expose {internal:?}:\n{stderr}"
        );
    }
    assert!(
        artifacts.is_empty(),
        "{name}: a rejected program must not emit partial C artifacts: {artifacts:?}"
    );
}

/// The issue's first witness: a constructor payload bound by `match` and
/// called in the arm.
#[test]
fn calling_a_constructor_payload_function_rejects_typed_on_c() {
    let program = format!(
        "{FN_BOX}def use_box(b: FnBox) -> i32 =\n  match b with {{\n    \
         | FnBox(h) => h(10i32)\n    | NoBox => 0i32\n  }}\n\
         a = use_box(FnBox(inc))\n"
    );
    assert_eval_runs_and_c_rejects_typed(&program, "fnbox_call", "a = 11");
}

/// The issue's second witness: the binder shadows a function-typed formal of
/// the same name, which C could call as a contextual callback.
#[test]
fn a_payload_binder_shadowing_a_callback_formal_rejects_typed_on_c() {
    let program = format!(
        "{FN_BOX}def use_box(f: i32 -> i32, b: FnBox) -> i32 =\n  match b with {{\n    \
         | FnBox(f) => f(10i32)\n    | NoBox => 0i32\n  }}\n\
         a = use_box(inc, FnBox(inc))\n"
    );
    assert_eval_runs_and_c_rejects_typed(&program, "fnbox_shadow", "a = 11");
}

/// A function-typed record field bound by a record pattern and called.
#[test]
fn calling_a_record_field_function_rejects_typed_on_c() {
    let program = "type CbBox =\n  | CbBox { cb: i32 -> i32 }\n\
                   def inc(z: i32) -> i32 = add(z, 1i32)\n\
                   def run(b: CbBox) -> i32 =\n  match b with {\n    \
                   | CbBox { cb } => cb(10i32)\n  }\n\
                   a = run(CbBox { cb: inc })\n";
    assert_eval_runs_and_c_rejects_typed(program, "record_field_call", "a = 11");
}

/// Every other way an arm applies its pattern-bound function reaches the same
/// rejection: an `Option` payload, a guard, an inlined higher-order callee, a
/// named list callback, and a lambda that captures the binder.
#[test]
fn every_application_route_through_a_pattern_bound_function_rejects_typed_on_c() {
    let cases = [
        (
            "option_payload_call",
            "def inc(z: i32) -> i32 = add(z, 1i32)\n\
             def use_opt(o: Option[i32 -> i32]) -> i32 =\n  match o with {\n    \
             | Some(h) => h(10i32)\n    | None => 0i32\n  }\n\
             a = use_opt(Some(inc))\n"
                .to_string(),
            "a = 11",
        ),
        (
            "guard_call",
            format!(
                "{FN_BOX}def use_box(b: FnBox) -> i32 =\n  match b with {{\n    \
                 | FnBox(h) if gt(h(0i32), 0i32) => 1i32\n    | _ => 0i32\n  }}\n\
                 a = use_box(FnBox(inc))\n"
            ),
            "a = 1",
        ),
        (
            "inlined_higher_order_call",
            format!(
                "{FN_BOX}def apply(f: i32 -> i32, x: i32) -> i32 = f(x)\n\
                 def use_box(b: FnBox) -> i32 =\n  match b with {{\n    \
                 | FnBox(h) => apply(h, 10i32)\n    | NoBox => 0i32\n  }}\n\
                 a = use_box(FnBox(inc))\n"
            ),
            "a = 11",
        ),
        (
            "named_list_callback",
            format!(
                "{FN_BOX}def use_box(b: FnBox) -> List[i32] =\n  match b with {{\n    \
                 | FnBox(h) => map(h, [1i32, 2i32])\n    | NoBox => []\n  }}\n\
                 a = use_box(FnBox(inc))\n"
            ),
            "a = [2, 3]",
        ),
        (
            "captured_by_lambda",
            format!(
                "{FN_BOX}def use_box(b: FnBox) -> List[i32] =\n  match b with {{\n    \
                 | FnBox(h) => map(fn (x: i32) -> h(x), [1i32, 2i32])\n    | NoBox => []\n  }}\n\
                 a = use_box(FnBox(inc))\n"
            ),
            "a = [2, 3]",
        ),
    ];
    for (name, program, expected) in cases {
        assert_eval_runs_and_c_rejects_typed(&program, name, expected);
    }
}

/// Negative parity: a function value stored in an ADT and returned, or
/// rebuilt from a pattern binder, without being called keeps the existing
/// #879 rejection (chelis#2739).
#[test]
fn an_uncalled_function_in_an_adt_keeps_the_typed_rejection() {
    let cases = [
        (
            "stored_and_returned",
            format!("{FN_BOX}def make() -> FnBox = FnBox(inc)\na = make()\n"),
            "make.0 = <closure>\na = FnBox(<closure>)",
        ),
        (
            "rebuilt_from_binder",
            format!(
                "{FN_BOX}def unbox(b: FnBox) -> FnBox =\n  match b with {{\n    \
                 | FnBox(h) => FnBox(h)\n    | NoBox => NoBox\n  }}\n\
                 a = unbox(FnBox(inc))\n"
            ),
            "a = FnBox(<closure>)",
        ),
    ];
    for (name, program, expected) in cases {
        assert_eval_runs_and_c_rejects_typed(&program, name, expected);
    }
}

/// Control: match binders over non-function payloads, including one that
/// shadows a formal and record-field binders, compile and agree with eval.
#[test]
fn non_function_payload_binders_compile_and_agree_with_eval() {
    let program = "type IntBox =\n  | IntBox(i32)\n  | NoInt\n\
                   type Pair =\n  | Pair { cb: i32, w: i32 }\n\
                   def use_box(f: i32, b: IntBox) -> i32 =\n  match b with {\n    \
                   | IntBox(f) => add(f, 10i32)\n    | NoInt => f\n  }\n\
                   def use_pair(p: Pair) -> i32 =\n  match p with {\n    \
                   | Pair { cb, w } => mul(cb, w)\n  }\n\
                   a = use_box(1i32, IntBox(1i32))\n\
                   b = use_box(7i32, NoInt)\n\
                   c = use_pair(Pair { cb: 6i32, w: 7i32 })\n";
    let expected = eval_stdout(program, "non_function_payloads");
    assert_eq!(expected.trim_end(), "a = 11\nb = 7\nc = 42");
    assert_eq!(
        compiled_stdout(program, "non_function_payloads"),
        expected,
        "C must print exactly what eval prints"
    );
}

/// Control: contextual callback parameters, with `i8` and `i16` signatures
/// and one called inside a match arm over a non-function payload, compile
/// and agree with eval.
#[test]
fn contextual_callback_parameters_compile_and_agree_with_eval() {
    let program = "def inc8(z: i8) -> i8 = add(z, cast(1, i8))\n\
                   def dbl16(z: i16) -> i16 = mul(z, cast(2, i16))\n\
                   def apply8(f: i8 -> i8, x: i8) -> i8 = f(x)\n\
                   def apply16(f: i16 -> i16, o: Option[i16]) -> i16 =\n  match o with {\n    \
                   | Some(v) => f(v)\n    | None => cast(0, i16)\n  }\n\
                   a = apply8(inc8, cast(6, i8))\n\
                   b = apply16(dbl16, Some(cast(151, i16)))\n\
                   c = apply16(dbl16, None)\n";
    let expected = eval_stdout(program, "contextual_callbacks");
    assert_eq!(expected.trim_end(), "a = 7\nb = 302\nc = 0");
    assert_eq!(
        compiled_stdout(program, "contextual_callbacks"),
        expected,
        "C must print exactly what eval prints"
    );
}
