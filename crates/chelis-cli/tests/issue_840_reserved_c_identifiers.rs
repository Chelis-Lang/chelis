//! chelis#840 - reserved-C-identifier mangling on the host build lane.
//!
//! A chelis-legal source def name that is a C keyword (`double`) or an
//! included typedef (`int8_t`) previously emitted verbatim into a translation
//! unit that could not compile (`static inline int32_t double(...)`), while
//! `chelis build` reported success. Authored defs now use an injective private
//! `chelis_fn_<utf8-hex>` namespace; bindings and parameters retain the
//! narrower #379 `chelis_user__` mapping (`c_ident`). The eval lane never
//! emits C and is unaffected.

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

/// `chelis build --target c`; Ok((generated .c source, live tempdir,
/// out dir)) or Err(stderr). The tempdir handle keeps the artifacts
/// alive for callers that link and run.
fn c_build_source(
    program: &str,
    name: &str,
) -> Result<(String, tempfile::TempDir, std::path::PathBuf), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
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
        .output()
        .expect("chelis build should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let source = std::fs::read_to_string(out_dir.join(format!("{name}.c")))
        .expect("the translation unit must be emitted");
    Ok((source, dir, out_dir))
}

/// Link and run the generated program, returning its first stdout line.
fn link_and_run(out_dir: &std::path::Path, name: &str) -> String {
    let status = common::link_generated(out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "cc link failed: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled binary failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// The chelis#840 red-team repro: a def named `double`, used both as a
/// direct call and as a named callback in `map`.
#[test]
fn a_def_named_double_builds_with_a_mangled_c_identifier() {
    let (source, _dir, out_dir) = c_build_source(
        "def double(x: i32) -> i32 = mul(x, 2)\n\
         out = print(double(21))\n",
        "kw_double",
    )
    .expect("a C-keyword def name must build via the #379 mapping");
    assert!(
        source.contains("chelis_fn_646f75626c65"),
        "the def must declare and reference through its private name:\n{source}"
    );
    assert!(
        !source.contains("int32_t double("),
        "the C keyword must never appear as a declarator name:\n{source}"
    );
    if c_toolchain_available() {
        assert_eq!(link_and_run(&out_dir, "kw_double"), "42");
    }
}

/// The named-callback position references the same mangled identifier.
#[test]
fn a_def_named_double_survives_the_named_callback_position() {
    let (source, _dir, out_dir) = c_build_source(
        "def double(x: i32) -> i32 = mul(x, 2)\n\
         out = print(map(double, [1, 2, 3]))\n",
        "kw_double_map",
    )
    .expect("a C-keyword named callback must build via the #379 mapping");
    assert!(
        source.contains("chelis_fn_646f75626c65"),
        "callback references must use the private emitted name:\n{source}"
    );
    assert!(
        !source.contains("int32_t double("),
        "the C keyword must never appear as a declarator name:\n{source}"
    );
    if c_toolchain_available() {
        assert_eq!(link_and_run(&out_dir, "kw_double_map"), "[2, 4, 6]");
    }
}

/// Parameter declarators and body references stay consistent for a
/// C-keyword parameter name.
#[test]
fn a_param_named_long_builds_consistently() {
    let (source, _dir, out_dir) = c_build_source(
        "def scale(long: i32, x: i32) -> i32 = mul(long, x)\n\
         out = print(scale(3, 14))\n",
        "kw_long_param",
    )
    .expect("a C-keyword parameter name must build via the #379 mapping");
    assert!(
        source.contains("chelis_user__6c6f6e67"),
        "the parameter must declare and reference through the mangled name:\n{source}"
    );
    assert!(
        !source.contains("int32_t long"),
        "the C keyword must never appear as a declarator name:\n{source}"
    );
    if c_toolchain_available() {
        assert_eq!(link_and_run(&out_dir, "kw_long_param"), "42");
    }
}

/// An included-typedef name is the same collision class as a keyword
/// (chelis#840 adds the stdint/stddef names to the shared reserved list).
#[test]
fn a_def_named_int8_t_builds_with_a_mangled_c_identifier() {
    let (source, _dir, out_dir) = c_build_source(
        "def int8_t(x: i32) -> i32 = add(x, 1)\n\
         out = print(int8_t(41))\n",
        "kw_typedef",
    )
    .expect("an included-typedef def name must build via the #379 mapping");
    assert!(
        source.contains("chelis_fn_696e74385f74"),
        "the def must declare and reference through the private name:\n{source}"
    );
    if c_toolchain_available() {
        assert_eq!(link_and_run(&out_dir, "kw_typedef"), "42");
    }
}

/// The eval lane never emits C; a C-keyword def name stays legal there.
#[test]
fn eval_still_accepts_a_def_named_double() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("kw_eval.ch");
    write_file(
        &path,
        "def double(x: i32) -> i32 = mul(x, 2)\n\
         out = print(double(21))\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval must keep accepting the name: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("42"),
        "eval must compute the right value"
    );
}

/// chelis#840 review, finding 1: a typed callback PARAMETER named like a
/// C reserved word must reference the same mangled identifier its
/// declarator used, at the direct-call site.
#[test]
fn a_reserved_callback_parameter_name_stays_consistent_at_the_call_site() {
    let (source, _dir, out_dir) = c_build_source(
        "def apply(double: i32 -> i32, x: i32) -> i32 = double(x)\n\
         def inc(y: i32) -> i32 = add(y, 1)\n\
         out = print(apply(inc, 41))\n",
        "kw_cb_param",
    )
    .expect("a reserved-word callback parameter must build via the #379 mapping");
    assert!(
        source.contains("(*chelis_user__646f75626c65)"),
        "the parameter declarator must be mangled:\n{source}"
    );
    assert!(
        !source.contains("= double("),
        "the call site must not reference the raw C keyword:\n{source}"
    );
    if c_toolchain_available() {
        assert_eq!(link_and_run(&out_dir, "kw_cb_param"), "42");
    }
}

/// chelis#840 review, finding 2: source spellings that previously collided
/// through the non-injective `chelis_user__` escape now remain distinct in
/// the private function namespace.
#[test]
fn source_spellings_that_resembled_the_old_escape_remain_distinct() {
    let (source, _dir, out_dir) = c_build_source(
        "def double(x: i32) -> i32 = mul(x, 2)\n\
         def chelis_user__double(x: i32) -> i32 = add(x, 100)\n\
         out = print(add(double(21), chelis_user__double(0)))\n",
        "kw_collision",
    )
    .expect("the private namespace is injective");
    assert!(
        source.contains("chelis_fn_646f75626c65")
            && source.contains("chelis_fn_6368656c69735f757365725f5f646f75626c65"),
        "the two source names must have distinct private emitted symbols:\n{source}"
    );
    if c_toolchain_available() {
        assert_eq!(link_and_run(&out_dir, "kw_collision"), "142");
    }
}

/// chelis#840 review, finding 2 control: the eval lane still computes
/// the collision program's value.
#[test]
fn eval_still_accepts_the_colliding_spelling_program() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("kw_collision_eval.ch");
    write_file(
        &path,
        "def double(x: i32) -> i32 = mul(x, 2)\n\
         def chelis_user__double(x: i32) -> i32 = add(x, 100)\n\
         out = print(add(double(21), chelis_user__double(0)))\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(out.status.success(), "eval accepts both names");
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("142"),
        "eval must compute 142"
    );
}

/// chelis#840 review, finding 3: the least/fast stdint families are in
/// the emitted include set and must mangle like the fixed-width names.
#[test]
fn a_def_named_int_fast8_t_builds_with_a_mangled_c_identifier() {
    let (source, _dir, out_dir) = c_build_source(
        "def int_fast8_t(x: i32) -> i32 = add(x, 1)\n\
         out = print(int_fast8_t(41))\n",
        "kw_fast_typedef",
    )
    .expect("a least/fast typedef def name must build via the #379 mapping");
    assert!(
        source.contains("chelis_fn_696e745f66617374385f74"),
        "the def must declare and reference through the private name:\n{source}"
    );
    if c_toolchain_available() {
        assert_eq!(link_and_run(&out_dir, "kw_fast_typedef"), "42");
    }
}
