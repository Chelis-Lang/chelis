//! chelis#2152: a SCALAR `cast` whose target is a dtype-family-bounded binder
//! must lower to C at every concrete instantiation.
//!
//! chelis#1418 made bounded cast targets actualize in compiled specializations,
//! but its acceptance covered tensor-producing helpers. A scalar cast to the
//! binder reaches the host-lane cast arm, which rejected it:
//!
//!   unsupported: dtype `p` on a `cast` target in host lowering ...
//!
//! This happened even when every call site is concrete. So a Float-generic
//! library breaks the C build of every consumer, f32 consumers included, while
//! check, eval and `reef build` all pass. [04-DTYPE-2] says the bound survives
//! wrappers, imports and recursive calls, and [05-OP-63] says no backend or
//! host carrier narrows the cast domain.
//!
//! Every positive program is evaluated, then built, linked and run, and the
//! two lanes must print the same thing. Each program instantiates the binder
//! at f32 AND f64, so a wrongly shared instantiation cannot pass.
//!
//! Negative parity: an unbounded binder target still rejects. [04-DTYPE-1]
//! and chelis#1558 own that rejection. Making the gate accept more must not
//! turn it into accepting everything.
use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn program(body: &str) -> String {
    format!("module Bounded.Main\nexport (main)\n{body}\n")
}

fn eval(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("cast.ch");
    fs::write(&path, source).expect("source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    assert!(
        output.status.success(),
        "eval rejected: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8")
}

fn assert_lanes_agree(body: &str, name: &str, expected: &[f64]) {
    let source = program(body);
    let interpreted = eval(&source);
    assert_eq!(
        common::parse_tensor_data(&interpreted, "main"),
        expected,
        "{name}: eval value"
    );
    let native = common::build_and_run(&source, name);
    assert_eq!(native.trim(), interpreted.trim(), "{name}: eval vs C");
}

#[test]
fn a_scalar_cast_of_an_integer_to_the_binder_lowers_at_f32_and_f64() {
    // `numel` is an i64; the cast selects the float dtype from the binder.
    assert_lanes_agree(
        "def count_as[n, p: Float](v: tensor[n, p]) -> p = cast(numel(v), p)\n\
         def main() -> tensor[2, f64] = to_tensor([\
            cast(count_as(to_tensor([1.0f32, 2.0f32, 3.0f32])), f64), \
            count_as(to_tensor([1.0f64, 2.0f64]))])",
        "scalar_int_source",
        &[3.0, 2.0],
    );
}

#[test]
fn a_scalar_cast_of_a_float_literal_to_the_binder_lowers() {
    // 16777217 is exact in f64 and rounds to 16777216 in f32, so both
    // instantiations are visible in the result.
    assert_lanes_agree(
        "def shifted[n, p: Float](v: tensor[n, p]) -> p = add(index(to_list(v), 0i64), cast(16777217.0f64, p))\n\
         def main() -> tensor[2, f64] = to_tensor([\
            cast(shifted(to_tensor([0.0f32])), f64), \
            shifted(to_tensor([0.0f64]))])",
        "scalar_literal_source",
        &[16777216.0, 16777217.0],
    );
}

#[test]
fn a_scalar_binder_cast_reached_through_a_generic_caller_lowers() {
    // The cast sits in `as_p`, which only `mean_like`, itself generic, calls.
    // The concrete dtype has to reach `as_p` through one generic-to-generic
    // call, which is the shape a real dtype-generic library has everywhere.
    assert_lanes_agree(
        "def as_p[p: Float](k: i64) -> p = cast(k, p)\n\
         def mean_like[n, p: Float](v: tensor[n, p]) -> p = div(index(to_list(v), 0i64), as_p(numel(v)))\n\
         def main() -> tensor[2, f64] = to_tensor([\
            cast(mean_like(to_tensor([6.0f32, 1.0f32, 1.0f32])), f64), \
            mean_like(to_tensor([5.0f64, 1.0f64]))])",
        "nested_generic_caller",
        &[2.0, 2.5],
    );
}

#[test]
fn a_binder_bound_only_through_tensor_types_actualizes_a_scalar_cast() {
    // `p` appears only as a tensor precision, in both the parameter and the
    // result. No scalar occurrence can bind it, so the substitution has to be
    // solved from the tensor precision itself.
    assert_lanes_agree(
        "def count_and_first[n, p: Float](v: tensor[n, p]) -> tensor[2, p] = to_tensor([cast(numel(v), p), index(to_list(v), 0i64)])\n\
         def main() -> tensor[2, f64] = add(\
            cast(count_and_first(to_tensor([1.5f32, 2.0f32, 9.0f32])), f64), \
            count_and_first(to_tensor([0.25f64])))",
        "tensor_only_binding",
        &[4.0, 1.75],
    );
}

#[test]
fn an_int_bounded_scalar_binder_target_lowers() {
    // The same inline path for an Int family. `p` is a tensor's precision, so
    // the call is inlined, not monomorphized. (With `p` only in a scalar
    // parameter the call is monomorphized and already worked, which is why
    // this test must not take that shape.)
    assert_lanes_agree(
        "def count_as[n, p: Int](v: tensor[n, p]) -> p = cast(numel(v), p)\n\
         def main() -> tensor[2, i64] = to_tensor([\
            cast(count_as(to_tensor([1i32, 2i32])), i64), \
            count_as(to_tensor([1i64, 2i64, 3i64]))])",
        "int_bounded_target",
        &[2.0, 3.0],
    );
}

#[test]
fn an_unbounded_binder_cast_target_still_rejects_in_the_build_lane() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad.ch");
    fs::write(
        &path,
        program(
            "def recast[p](value: p) -> p = cast(value, p)\n\
             def main() -> tensor[1, f64] = to_tensor([recast(1.5f64)])",
        ),
    )
    .expect("source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(dir.path().join("out"))
        .output()
        .expect("build");
    assert!(
        !output.status.success(),
        "an unbounded binder cast target was accepted by the build lane"
    );
}
