//! chelis#2937: an output-inferred dimension binder takes its value from the
//! first site that produces it.
//!
//! `spec/04-type-system.md` section 4.4.1 makes a dimension binder that only
//! the declared result names output-inferred: the body is the only place its
//! value can come from. The first site in evaluation order that produces an
//! extent for it binds it, and every later site naming it, including the
//! declared result, is a claim checked under section 4.7. A local tensor
//! ascription is such a site. Lowering used to refuse the ascription outright
//! ("cannot resolve authored extent") because only parameter axes declared a
//! binder's value.
//!
//! Each positive row runs `chelis check`, `chelis eval --file` and the compiled
//! C executable and requires the two lanes to print the same bytes. Each
//! negative row requires the same `Domain` trap on both lanes.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::{Command as StdCommand, Output};
use tempfile::{TempDir, tempdir};

/// A runtime `i64` the compiler cannot fold (3), so a claim over it is checked
/// at run time rather than refuted when the activation is lowered.
const RUNTIME_THREE: &str = "tensor_to_scalar(sum(to_tensor([2i64, 1i64]), 0i32))";

const INSERT_TRAP: &str = "numeric trap: domain in insert at i64";

fn cli() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

fn combined(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn write_fixture(dir: &TempDir, stem: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write fixture");
    path
}

fn check(path: &Path) -> Output {
    cli()
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check")
}

fn eval(path: &Path) -> Output {
    cli()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

/// Build natively for C and run the executable. A build failure is returned
/// as the build's own output, so a row can tell a lowering rejection from a
/// run-time trap.
fn build_and_run(dir: &TempDir, stem: &str, path: &Path) -> Output {
    let out_dir = dir.path().join(format!("{stem}-out"));
    let build = cli()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    if !build.status.success() {
        return build;
    }
    StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run the compiled executable")
}

/// Check, evaluate and build `source`; require both lanes to succeed, print
/// the same bytes, and print `expected` among them.
fn assert_lanes_agree(stem: &str, source: &str, expected: &str) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(&dir, stem, source);
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "{stem}: check must accept a runtime i64 size: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    assert!(
        evaluated.status.success(),
        "{stem}: eval must execute: {}",
        combined(&evaluated)
    );
    let compiled = build_and_run(&dir, stem, &path);
    assert!(
        compiled.status.success(),
        "{stem}: the C build must build and run: {}",
        combined(&compiled)
    );
    let eval_out = String::from_utf8_lossy(&evaluated.stdout).to_string();
    let c_out = String::from_utf8_lossy(&compiled.stdout).to_string();
    assert_eq!(eval_out, c_out, "{stem}: eval and C print the same bytes");
    assert!(
        eval_out.contains(expected),
        "{stem}: expected `{expected}` in {eval_out}"
    );
}

/// Check accepts `source`; both lanes then fail at run time with the same
/// rendering, which contains every line of `lines`.
fn assert_lanes_trap_identically(stem: &str, source: &str, lines: &[&str]) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(&dir, stem, source);
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "{stem}: the failure is a run-time one, so check accepts: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    let compiled = build_and_run(&dir, stem, &path);
    let eval_out = combined(&evaluated);
    let c_out = combined(&compiled);
    for (lane, output, text) in [("eval", &evaluated, &eval_out), ("c", &compiled, &c_out)] {
        assert!(!output.status.success(), "{stem}: {lane} must trap: {text}");
        assert!(
            !text.contains("out = "),
            "{stem}: {lane} must not print the trapped binding: {text}"
        );
        for line in lines {
            assert!(
                text.contains(line),
                "{stem}: {lane} must print `{line}`: {text}"
            );
        }
    }
    assert_eq!(
        eval_out.trim_end(),
        c_out.trim_end(),
        "{stem}: the two lanes render the trap identically"
    );
}

/// The chelis#2937 reproducer: school's chained broadcast with an
/// intermediate ascription naming `h`, which only the result carries.
#[test]
fn a_local_ascription_binds_an_output_inferred_binder() {
    assert_lanes_agree(
        "chw",
        "def broadcast_to_chw[c, h, w](v: &tensor[c, f32], h_dim: i64, w_dim: i64) -> tensor[c, h, w, f32] = {\n\
         \x20 step1: tensor[c, h, f32] = insert(v, 1i32, h_dim)\n\
         \x20 insert(step1, 2i32, w_dim)\n\
         }\n\
         out = [1.0f32, 2.0f32] |> to_tensor |> broadcast_to_chw(3i64, 4i64)\n",
        "out = tensor(shape=[2, 3, 4], data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
         2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])",
    );
}

/// A second local ascription of the same binder is a claim against the first,
/// and an agreeing one executes.
#[test]
fn two_agreeing_local_ascriptions_of_one_binder_execute() {
    assert_lanes_agree(
        "two_sites",
        &format!(
            "def f[c, h](v: &tensor[c, f32], k: i64) -> tensor[c, h, f32] = {{\n\
             \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 step2: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 add(step1, step2)\n\
             }}\n\
             out = [1.0f32, 2.0f32] |> to_tensor |> f({RUNTIME_THREE})\n"
        ),
        "out = tensor(shape=[2, 3], data=[2.0, 2.0, 2.0, 4.0, 4.0, 4.0])",
    );
}

/// Once bound, the output-inferred binder sizes a later `insert` by name.
#[test]
fn a_bound_output_inferred_binder_sizes_a_later_insert() {
    assert_lanes_agree(
        "later_size",
        &format!(
            "def f[c, h](v: &tensor[c, f32], k: i64) -> tensor[c, h, f32] = {{\n\
             \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 twos = 2.0f32 |> scalar_to_tensor |> insert(0i32, shape(v, 0i32)) |> insert(1i32, h)\n\
             \x20 mul(step1, twos)\n\
             }}\n\
             out = [1.0f32, 2.0f32] |> to_tensor |> f({RUNTIME_THREE})\n"
        ),
        "out = tensor(shape=[2, 3], data=[2.0, 2.0, 2.0, 4.0, 4.0, 4.0])",
    );
}

/// A later local ascription whose extent disagrees with the first site traps
/// `Domain` at its own producing operation.
#[test]
fn a_later_local_ascription_that_disagrees_traps_domain() {
    assert_lanes_trap_identically(
        "later_site",
        &format!(
            "def f[c, h](v: &tensor[c, f32], k: i64) -> tensor[c, h, f32] = {{\n\
             \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 step2: tensor[c, h, f32] = insert(v, 1i32, add(k, 1i64))\n\
             \x20 add(step1, step2)\n\
             }}\n\
             out = [1.0f32, 2.0f32] |> to_tensor |> f({RUNTIME_THREE})\n"
        ),
        &[INSERT_TRAP],
    );
}

/// The declared result is a later site too: a returned extent that disagrees
/// with the first local site traps `Domain`.
#[test]
fn a_declared_result_that_disagrees_with_the_first_site_traps_domain() {
    assert_lanes_trap_identically(
        "result_site",
        &format!(
            "def f[c, h](v: &tensor[c, f32], k: i64) -> tensor[c, h, f32] = {{\n\
             \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 _ = step1\n\
             \x20 insert(v, 1i32, add(k, 1i64))\n\
             }}\n\
             out = [1.0f32, 2.0f32] |> to_tensor |> f({RUNTIME_THREE})\n"
        ),
        &[INSERT_TRAP],
    );
}

/// A binder a parameter carries is still bound by the parameter: an
/// ascription that agrees with it executes, and one that disagrees traps with
/// the parameter's extent as the claim.
#[test]
fn a_parameter_carried_binder_is_still_bound_by_the_parameter() {
    let source = |rows: &str| {
        format!(
            "def f[c, h](v: &tensor[c, f32], m: &tensor[h, f32], k: i64) -> tensor[c, h, f32] = {{\n\
             \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 add(step1, step1)\n\
             }}\n\
             out = f(to_tensor([1.0f32, 2.0f32]), to_tensor({rows}), {RUNTIME_THREE})\n"
        )
    };
    assert_lanes_agree(
        "param_agrees",
        &source("[0.0f32, 0.0f32, 0.0f32]"),
        "out = tensor(shape=[2, 3], data=[2.0, 2.0, 2.0, 4.0, 4.0, 4.0])",
    );
    assert_lanes_trap_identically(
        "param_disagrees",
        &source("[0.0f32, 0.0f32]"),
        &["extent `h`: claimed = 2, insert axis 1 = 3", INSERT_TRAP],
    );
}

/// A first site inside a runtime `if` arm binds the binder on that arm's path
/// only; the other arm's value is that path's first site.
#[test]
fn a_first_site_inside_a_runtime_arm_binds_only_that_path() {
    assert_lanes_agree(
        "arm_site",
        &format!(
            "def f[c, h](v: &tensor[c, f32], k: i64, flag: bool) -> tensor[c, h, f32] = if flag then {{\n\
             \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 step1\n\
             }} else insert(v, 1i32, add(k, 1i64))\n\
             out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n\
             other = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, false)\n"
        ),
        "out = tensor(shape=[2, 3], data=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])\n\
         other = tensor(shape=[2, 4], data=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])",
    );
}

/// A runtime `if` keeps its body in host control flow on both lanes, so these
/// rows exercise the host lanes' own record of each binder's first site.
fn arm_source(then_body: &str, else_body: &str, calls: &str) -> String {
    format!(
        "def f[c, h](v: &tensor[c, f32], k: i64, flag: bool) -> tensor[c, h, f32] = if flag then {{\n\
         {then_body}\
         }} else {else_body}\n\
         {calls}"
    )
}

/// The first site sits inside the taken arm and the arm's value disagrees
/// with it: the declared result is a later site on that path and traps.
#[test]
fn a_declared_result_that_disagrees_with_a_first_site_in_a_runtime_arm_traps_domain() {
    assert_lanes_trap_identically(
        "arm_result_site",
        &arm_source(
            "\x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 _ = step1\n\
             \x20 insert(v, 1i32, add(k, 1i64))\n",
            "insert(v, 1i32, k)",
            &format!("out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n"),
        ),
        &[
            "extent `h`: step1 axis 1 = 3, insert axis 1 = 4",
            INSERT_TRAP,
        ],
    );
    // The arms agree with each other, so only the first site can refute the
    // returned extent.
    assert_lanes_trap_identically(
        "arm_result_site_equal_arms",
        &arm_source(
            "\x20 step1: tensor[c, h, f32] = insert(v, 1i32, add(k, 1i64))\n\
             \x20 _ = step1\n\
             \x20 insert(v, 1i32, k)\n",
            "insert(v, 1i32, k)",
            &format!("out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n"),
        ),
        &[
            "extent `h`: step1 axis 1 = 4, insert axis 1 = 3",
            INSERT_TRAP,
        ],
    );
}

/// The controls for the row above: an agreeing taken arm, and the other arm,
/// whose path has no earlier site, so its value is the first site.
#[test]
fn a_first_site_in_a_runtime_arm_binds_only_the_path_that_runs_it() {
    let source = format!(
        "def f[c, h](v: &tensor[c, f32], k: i64, j: i64, flag: bool) -> tensor[c, h, f32] = if flag then {{\n\
         \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
         \x20 _ = step1\n\
         \x20 insert(v, 1i32, j)\n\
         }} else insert(v, 1i32, add(j, 1i64))\n\
         a = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, {RUNTIME_THREE}, true)\n\
         b = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, {RUNTIME_THREE}, false)\n\
         c = f(to_tensor([1.0f32, 2.0f32]), 2i64, 2i64, true)\n"
    );
    assert_lanes_agree(
        "arm_paths",
        &source,
        "a = tensor(shape=[2, 3], data=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])\n\
         b = tensor(shape=[2, 4], data=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])\n\
         c = tensor(shape=[2, 2], data=[1.0, 1.0, 2.0, 2.0])",
    );
}

/// A later local ascription in a runtime arm is a claim against the first
/// site, reported at the later site's producer on both lanes.
#[test]
fn a_later_local_ascription_in_a_runtime_arm_that_disagrees_traps_domain() {
    assert_lanes_trap_identically(
        "arm_later_site",
        &arm_source(
            "\x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 step2: tensor[c, h, f32] = insert(v, 1i32, add(k, 1i64))\n\
             \x20 add(step1, step2)\n",
            "insert(v, 1i32, k)",
            &format!("out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n"),
        ),
        &["extent `h`: claimed = 3, insert axis 1 = 4", INSERT_TRAP],
    );
}

/// In a runtime arm, the bound binder sizes a later `insert` by name. A size
/// reads the binder's value, so it cannot disagree with the first site; the
/// row pins that the host lanes read the first site's extent for it.
#[test]
fn a_bound_binder_sizes_a_later_insert_in_a_runtime_arm() {
    assert_lanes_agree(
        "arm_later_size",
        &arm_source(
            "\x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 twos = 2.0f32 |> scalar_to_tensor |> insert(0i32, shape(v, 0i32)) |> insert(1i32, h)\n\
             \x20 mul(step1, twos)\n",
            "insert(v, 1i32, k)",
            &format!("out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n"),
        ),
        "out = tensor(shape=[2, 3], data=[2.0, 2.0, 2.0, 4.0, 4.0, 4.0])",
    );
}

/// A first site inside a block binds the binder for the rest of the
/// activation: the block's end does not unbind it, for a later ascription or
/// for the declared result.
#[test]
fn a_first_site_inside_a_block_stays_bound_after_the_block() {
    let block = "\x20 s = {\n\
                 \x20   step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
                 \x20   sum(sum(step1, 1i32), 0i32)\n\
                 \x20 }\n\
                 \x20 _ = s\n";
    let call = format!("out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n");
    assert_lanes_trap_identically(
        "block_result_site",
        &arm_source(
            &format!("{block}\x20 insert(v, 1i32, add(k, 1i64))\n"),
            "insert(v, 1i32, k)",
            &call,
        ),
        &[
            "extent `h`: step1 axis 1 = 3, insert axis 1 = 4",
            INSERT_TRAP,
        ],
    );
    assert_lanes_trap_identically(
        "block_later_site",
        &arm_source(
            &format!(
                "{block}\x20 step2: tensor[c, h, f32] = insert(v, 1i32, add(k, 1i64))\n\
                 \x20 _ = step2\n\
                 \x20 insert(v, 1i32, k)\n"
            ),
            "insert(v, 1i32, k)",
            &call,
        ),
        &["extent `h`: claimed = 3, insert axis 1 = 4", INSERT_TRAP],
    );
}

/// The declared result is checked against a first site that runs after the
/// returned value's producer, on the host lanes and in a straight-line body.
#[test]
fn a_first_site_after_the_returned_producer_still_binds_the_result() {
    let body = "\x20 r = insert(v, 1i32, add(k, 1i64))\n\
                \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
                \x20 _ = step1\n\
                \x20 r\n";
    assert_lanes_trap_identically(
        "late_site_arm",
        &arm_source(
            body,
            "insert(v, 1i32, k)",
            &format!("out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n"),
        ),
        &[
            "extent `h`: step1 axis 1 = 3, insert axis 1 = 4",
            INSERT_TRAP,
        ],
    );
    assert_lanes_trap_identically(
        "late_site_block",
        &format!(
            "def f[c, h](v: &tensor[c, f32], k: i64) -> tensor[c, h, f32] = {{\n\
             {body}\
             }}\n\
             out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE})\n"
        ),
        &[INSERT_TRAP],
    );
}

/// A `match` arm binds like an `if` arm, and a returned callee's value is
/// claimed against the caller's first site.
#[test]
fn a_first_site_in_a_match_arm_or_before_a_tail_call_is_claimed() {
    let matched = |some: &str| {
        format!(
            "def f[c, h](v: &tensor[c, f32], k: i64, o: Option[i64]) -> tensor[c, h, f32] = match o with {{\n\
             \x20 | Some(j) => {{\n\
             \x20   step1: tensor[c, h, f32] = insert(v, 1i32, j)\n\
             \x20   _ = step1\n\
             \x20   insert(v, 1i32, k)\n\
             \x20 }}\n\
             \x20 | None => insert(v, 1i32, add(k, 1i64))\n\
             }}\n\
             a = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, None)\n\
             b = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, Some({some}))\n"
        )
    };
    assert_lanes_agree(
        "match_agrees",
        &matched(RUNTIME_THREE),
        "a = tensor(shape=[2, 4], data=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])\n\
         b = tensor(shape=[2, 3], data=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])",
    );
    assert_lanes_trap_identically(
        "match_disagrees",
        &matched("2i64"),
        &[
            "extent `h`: step1 axis 1 = 2, insert axis 1 = 3",
            INSERT_TRAP,
        ],
    );
    assert_lanes_trap_identically(
        "tail_call",
        &format!(
            "def g[c, n](v: &tensor[c, f32], k: i64) -> tensor[c, n, f32] = insert(v, 1i32, k)\n\
             {}",
            arm_source(
                "\x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
                 \x20 _ = step1\n\
                 \x20 g(v, add(k, 1i64))\n",
                "g(v, k)",
                &format!("out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n"),
            )
        ),
        &[
            "extent `h`: step1 axis 1 = 3, insert axis 1 = 4",
            INSERT_TRAP,
        ],
    );
}

/// Each invocation binds its own first sites.
#[test]
fn each_invocation_binds_its_own_first_site() {
    assert_lanes_agree(
        "per_invocation",
        &arm_source(
            "\x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 step1\n",
            "insert(v, 1i32, k)",
            &format!(
                "a = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, true)\n\
                 b = f(to_tensor([1.0f32, 2.0f32]), 2i64, true)\n"
            ),
        ),
        "a = tensor(shape=[2, 3], data=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])\n\
         b = tensor(shape=[2, 2], data=[1.0, 1.0, 2.0, 2.0])",
    );
}

/// In host control flow, a local ascription naming a parameter-carried binder
/// is claimed at its own producer on both lanes (chelis#2374's named
/// residual for host regions).
#[test]
fn a_parameter_carried_binder_is_claimed_at_a_local_site_in_a_runtime_arm() {
    assert_lanes_trap_identically(
        "arm_param_site",
        &format!(
            "def f[c, h](v: &tensor[c, f32], m: &tensor[h, f32], k: i64, flag: bool) -> tensor[c, h, f32] = if flag then {{\n\
             \x20 step1: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20 add(step1, step1)\n\
             }} else insert(v, 1i32, k)\n\
             out = f(to_tensor([1.0f32, 2.0f32]), to_tensor([0.0f32, 0.0f32]), {RUNTIME_THREE}, true)\n"
        ),
        &["extent `h`: claimed = 2, insert axis 1 = 3", INSERT_TRAP],
    );
}

/// A runtime `if` inside a `let` initializer, with a first site in its `then`
/// arm. A tensor kernel would lower the `if` as a `where` that computes both
/// arms, which cannot bind `h` on one path alone, so the body runs in host
/// control flow on both lanes.
fn let_arm_source(then_value: &str, else_value: &str, tail: &str, calls: &str) -> String {
    format!(
        "def f[c, h](v: &tensor[c, f32], k: i64, flag: bool) -> tensor[c, h, f32] = {{\n\
         \x20 s = if flag then {{\n\
         \x20   a: tensor[c, h, f32] = insert(v, 1i32, k)\n\
         \x20   {then_value}\n\
         \x20 }} else {else_value}\n\
         {tail}\
         }}\n\
         {calls}"
    )
}

const SUMMED_ARM: &str = "sum(sum(a, 1i32), 0i32)";
const SUMMED_ELSE: &str = "sum(v, 0i32)";
const SCALAR_ARM: &str = "shape(a, 1i32)";
const SCALAR_ELSE: &str = "0i64";
/// The arm's value scales the result, so the result shows whether it ran.
const SCALED_TAIL: &str = "\x20 step2 = insert(v, 1i32, add(k, 1i64))\n\
     \x20 mul(step2, s |> insert(0i32, shape(v, 0i32)) |> insert(1i32, shape(step2, 1i32)))\n";
const SCALED_LATER_SITE: &str = "\x20 step2: tensor[c, h, f32] = insert(v, 1i32, add(k, 1i64))\n\
     \x20 mul(step2, s |> insert(0i32, shape(v, 0i32)) |> insert(1i32, shape(step2, 1i32)))\n";
const DISCARDED_TAIL: &str = "\x20 _ = s\n\x20 insert(v, 1i32, add(k, 1i64))\n";
const SCALAR_TAIL: &str = "\x20 insert(v, 1i32, add(add(k, 1i64), mul(s, 0i64)))\n";
const MUL_TRAP: &[&str] = &[
    "extent `h`: a axis 1 = 3, mul axis 1 = 4",
    "numeric trap: domain in mul at i64",
];
const ARM_INSERT_TRAP: &[&str] = &["extent `h`: a axis 1 = 3, insert axis 1 = 4", INSERT_TRAP];
const ONES_2X4: &str = "tensor(shape=[2, 4], data=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])";
const THREES_2X4: &str = "tensor(shape=[2, 4], data=[3.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 6.0])";

fn out_call(flag: &str) -> String {
    format!("out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, {flag})\n")
}

/// A flag the compiler cannot fold: `3 > threshold`.
fn runtime_flag(threshold: i64) -> String {
    format!("gt({RUNTIME_THREE}, {threshold}i64)")
}

/// An untaken call, then the call whose arm runs.
fn other_then_out() -> String {
    format!(
        "other = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, false)\n{}",
        out_call("true")
    )
}

/// The arm runs and binds `h` = 3; the arm's value scales the result, which
/// produces 4 and so traps as a later site.
#[test]
fn a_first_site_in_a_runtime_arm_of_a_let_initializer_claims_the_result() {
    assert_lanes_trap_identically(
        "let_arm_used",
        &let_arm_source(
            SUMMED_ARM,
            SUMMED_ELSE,
            SCALED_TAIL,
            &out_call(&runtime_flag(2)),
        ),
        MUL_TRAP,
    );
}

/// The same with the arm's scalar value discarded.
#[test]
fn a_first_site_in_a_discarded_scalar_arm_claims_the_result() {
    assert_lanes_trap_identically(
        "let_arm_scalar_discarded",
        &let_arm_source(SCALAR_ARM, SCALAR_ELSE, DISCARDED_TAIL, &out_call("true")),
        ARM_INSERT_TRAP,
    );
}

/// The same with a discarded tensor arm under a runtime flag.
#[test]
fn a_first_site_in_a_discarded_arm_claims_the_result_under_a_runtime_flag() {
    assert_lanes_trap_identically(
        "let_arm_discarded_runtime_flag",
        &let_arm_source(
            SUMMED_ARM,
            SUMMED_ELSE,
            DISCARDED_TAIL,
            &out_call(&runtime_flag(2)),
        ),
        ARM_INSERT_TRAP,
    );
}

/// The same with a later ascription, after an untaken call of the same def.
#[test]
fn a_first_site_in_a_discarded_arm_claims_a_later_ascription() {
    assert_lanes_trap_identically(
        "let_arm_discarded_later_site",
        &let_arm_source(
            SUMMED_ARM,
            SUMMED_ELSE,
            "\x20 _ = s\n\
             \x20 step2: tensor[c, h, f32] = insert(v, 1i32, add(k, 1i64))\n\
             \x20 step2\n",
            &other_then_out(),
        ),
        ARM_INSERT_TRAP,
    );
}

/// The same with the declared result, after an untaken call of the same def.
#[test]
fn a_first_site_in_a_discarded_arm_claims_the_result_after_an_untaken_call() {
    assert_lanes_trap_identically(
        "let_arm_discarded_result",
        &let_arm_source(SUMMED_ARM, SUMMED_ELSE, DISCARDED_TAIL, &other_then_out()),
        ARM_INSERT_TRAP,
    );
}

/// The controls for the rows above: the arm does not run, so it binds
/// nothing, and the declared result is the path's first site.
#[test]
fn an_untaken_scalar_arm_leaves_the_result_as_the_first_site() {
    assert_lanes_agree(
        "let_arm_untaken_result",
        &let_arm_source(SCALAR_ARM, SCALAR_ELSE, SCALAR_TAIL, &out_call("false")),
        &format!("out = {ONES_2X4}"),
    );
}

/// An untaken arm leaves a later ascription as the path's first site.
#[test]
fn an_untaken_scalar_arm_leaves_a_later_ascription_as_the_first_site() {
    assert_lanes_agree(
        "let_arm_untaken_later_site",
        &let_arm_source(
            SCALAR_ARM,
            SCALAR_ELSE,
            "\x20 step2: tensor[c, h, f32] = insert(v, 1i32, add(add(k, 1i64), mul(s, 0i64)))\n\
             \x20 step2\n",
            &out_call(&runtime_flag(5)),
        ),
        &format!("out = {ONES_2X4}"),
    );
}

/// The same when the untaken arm's value scales the result.
#[test]
fn an_untaken_tensor_arm_leaves_a_later_ascription_as_the_first_site() {
    assert_lanes_agree(
        "let_arm_untaken_used",
        &let_arm_source(
            SUMMED_ARM,
            SUMMED_ELSE,
            SCALED_LATER_SITE,
            &out_call(&runtime_flag(5)),
        ),
        &format!("out = {THREES_2X4}"),
    );
}

/// The same for two untaken calls of one def.
#[test]
fn each_untaken_call_leaves_a_later_ascription_as_the_first_site() {
    assert_lanes_agree(
        "let_arm_untaken_twice",
        &let_arm_source(
            SUMMED_ARM,
            SUMMED_ELSE,
            SCALED_LATER_SITE,
            &format!(
                "other = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE}, false)\n{}",
                out_call("false")
            ),
        ),
        &format!("other = {THREES_2X4}\nout = {THREES_2X4}"),
    );
}

/// Check accepts `source`; both lanes then refuse it before running, each
/// naming `text`.
fn assert_lanes_refuse(stem: &str, source: &str, text: &str) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(&dir, stem, source);
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "{stem}: check accepts: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    let compiled = build_and_run(&dir, stem, &path);
    for (lane, output) in [("eval", &evaluated), ("c", &compiled)] {
        let rendered = combined(output);
        assert!(
            !output.status.success(),
            "{stem}: {lane} must refuse: {rendered}"
        );
        assert!(
            rendered.contains(text) && !rendered.contains("numeric trap"),
            "{stem}: {lane} must refuse with `{text}`: {rendered}"
        );
    }
}

/// Only a binder the declared result names and no parameter names is
/// output-inferred. A binder the body alone names has no site that may bind
/// it, so a site naming it keeps the refusal, on a kernel body and across the
/// regions of a host body alike.
#[test]
fn a_binder_only_the_body_names_is_refused_on_both_lanes() {
    assert_lanes_refuse(
        "body_only_kernel",
        &format!(
            "def f[c, q](v: &tensor[c, f32], k: i64) -> tensor[c, f32] = {{\n\
             \x20 a: tensor[c, q, f32] = insert(v, 1i32, k)\n\
             \x20 b: tensor[c, q, f32] = insert(v, 1i32, add(k, 1i64))\n\
             \x20 sum(add(a, b), 1i32)\n\
             }}\n\
             out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE})\n"
        ),
        "cannot resolve authored extent `q`",
    );
    assert_lanes_refuse(
        "body_only_regions",
        &format!(
            "def f[c, q](v: &tensor[c, f32], k: i64, flag: bool) -> tensor[c, f32] = {{\n\
             \x20 s = if flag then {{\n\
             \x20   a: tensor[c, q, f32] = insert(v, 1i32, k)\n\
             \x20   sum(a, 1i32)\n\
             \x20 }} else add(v, v)\n\
             \x20 b: tensor[c, q, f32] = insert(v, 1i32, add(k, 1i64))\n\
             \x20 add(s, sum(b, 1i32))\n\
             }}\n\
             {}",
            out_call(&runtime_flag(2))
        ),
        "cannot resolve authored extent `q`",
    );
}

/// A binder an element type of a `List` parameter names is the parameter's,
/// not output-inferred, so a site naming it never binds it first. The host
/// regions have no witness for it and keep the refusal on both lanes.
#[test]
fn a_list_parameter_binder_is_never_bound_by_a_site() {
    for (stem, sites) in [
        (
            "list_binder_one_site",
            "\x20 b: tensor[n, f32] = insert(sum(c0, 0i32), 0i32, k)\n\
             \x20 if flag then add(b, b) else b\n",
        ),
        (
            "list_binder_two_sites",
            "\x20 b: tensor[n, f32] = insert(sum(c0, 0i32), 0i32, k)\n\
             \x20 d: tensor[n, f32] = insert(sum(c0, 0i32), 0i32, add(k, 1i64))\n\
             \x20 if flag then add(b, b) else add(c0, c0)\n",
        ),
    ] {
        assert_lanes_refuse(
            stem,
            &format!(
                "def f[n](xs: List[tensor[n, f32]], k: i64, flag: bool) -> tensor[n, f32] = {{\n\
                 \x20 c0 = if flag then index(xs, 0i64) else index(xs, 1i64)\n\
                 {sites}\
                 }}\n\
                 out = f([to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])], {RUNTIME_THREE}, {})\n",
                runtime_flag(2)
            ),
            "cannot resolve authored extent `n`",
        );
    }
}

/// A lambda's ascription naming the enclosing definition's parameter binder
/// is a claim, never a first site: the lambda does not bind `c` afresh.
/// Compiled C claims it against the parameter and traps. `chelis eval` has no
/// typed parameter input inside a lambda, so it keeps the refusal it gave
/// before output-inferred binders existed; neither lane prints a value.
#[test]
fn a_lambda_site_never_binds_the_enclosing_parameter_binder() {
    for (stem, body) in [
        (
            "lambda_param_site",
            "\x20   a: tensor[c, f32] = insert(sum(v, 0i32), 0i32, j)\n\
             \x20   add(j, shape(a, 0i32))\n",
        ),
        (
            "lambda_param_two_sites",
            "\x20   a: tensor[c, f32] = insert(sum(v, 0i32), 0i32, j)\n\
             \x20   b = add(j, shape(a, 0i32))\n\
             \x20   d: tensor[c, f32] = insert(sum(v, 0i32), 0i32, b)\n\
             \x20   shape(d, 0i32)\n",
        ),
    ] {
        let dir = tempdir().expect("tempdir");
        let source = format!(
            "def f[c](v: &tensor[c, f32], k: i64) -> i64 = {{\n\
             \x20 m = map(fn (j: i64) -> {{\n\
             {body}\
             \x20 }}, [k])\n\
             \x20 sum(to_tensor(m), 0i32) |> tensor_to_scalar\n\
             }}\n\
             out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE})\n"
        );
        let path = write_fixture(&dir, stem, &source);
        assert!(check(&path).status.success(), "{stem}: check accepts");
        let evaluated = eval(&path);
        let eval_out = combined(&evaluated);
        assert!(
            !evaluated.status.success()
                && eval_out.contains("cannot resolve authored extent `c`")
                && !eval_out.contains("out = "),
            "{stem}: eval must refuse rather than bind `c`: {eval_out}"
        );
        let compiled = build_and_run(&dir, stem, &path);
        let c_out = combined(&compiled);
        assert!(
            !compiled.status.success()
                && c_out.contains("extent `c`: claimed = 2, insert axis 0 = 3")
                && c_out.contains(INSERT_TRAP),
            "{stem}: C must claim `c` against the parameter: {c_out}"
        );
    }
}

/// A first site inside a block, before a runtime `if` that returns the
/// result, for a binder only a tuple, `Option` or `List` result names.
fn block_first_site_then(result: &str, later: &str, tail: &str) -> String {
    format!(
        "def f[c, h](v: &tensor[c, f32], k: i64, flag: bool) -> {result} = {{\n\
         \x20 s = {{\n\
         \x20   a: tensor[c, h, f32] = insert(v, 1i32, k)\n\
         \x20   sum(sum(a, 1i32), 0i32)\n\
         \x20 }}\n\
         \x20 if flag then {{\n\
         \x20   b: tensor[c, h, f32] = insert(v, 1i32, {later})\n\
         {tail}\
         }}\n\
         {}",
        out_call(&runtime_flag(2))
    )
}

/// The return-only set counts a binder that only a tuple result names, so
/// the block's ascription is `h`'s first site whatever the result's shape,
/// and the arm's ascription, in another host region, is a claim against it.
#[test]
fn a_tuple_result_binder_is_bound_by_a_block_and_claimed_in_an_arm() {
    assert_lanes_trap_identically(
        "tuple_block_then_arm",
        &block_first_site_then(
            "(tensor[c, h, f32], i64)",
            "add(k, 1i64)",
            "\x20   (b, 1i64)\n\
             \x20 } else (insert(v, 1i32, k), 0i64)\n",
        ),
        &["extent `h`: claimed = 3, insert axis 1 = 4", INSERT_TRAP],
    );
}

/// The same with the two sites in two blocks and no runtime `if`.
#[test]
fn a_tuple_result_binder_is_claimed_across_two_blocks() {
    assert_lanes_trap_identically(
        "tuple_two_blocks",
        &format!(
            "def f[c, h](v: &tensor[c, f32], k: i64) -> (tensor[c, h, f32], i64) = {{\n\
             \x20 s = {{\n\
             \x20   a: tensor[c, h, f32] = insert(v, 1i32, k)\n\
             \x20   sum(sum(a, 1i32), 0i32)\n\
             \x20 }}\n\
             \x20 t = {{\n\
             \x20   b: tensor[c, h, f32] = insert(v, 1i32, add(k, 1i64))\n\
             \x20   sum(sum(b, 1i32), 0i32)\n\
             \x20 }}\n\
             \x20 (insert(v, 1i32, add(k, 1i64)), 0i64)\n\
             }}\n\
             out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE})\n"
        ),
        &["extent `h`: claimed = 3, insert axis 1 = 4", INSERT_TRAP],
    );
}

/// The same for a binder only an `Option` result names.
#[test]
fn an_option_result_binder_is_bound_by_a_block_and_claimed_in_an_arm() {
    assert_lanes_trap_identically(
        "option_block_then_arm",
        &block_first_site_then(
            "Option[tensor[c, h, f32]]",
            "add(k, 1i64)",
            "\x20   Some(b)\n\
             \x20 } else None\n",
        ),
        &["extent `h`: claimed = 3, insert axis 1 = 4", INSERT_TRAP],
    );
}

/// The same for a binder only a `List` result names.
#[test]
fn a_list_result_binder_is_bound_by_a_block_and_claimed_in_an_arm() {
    assert_lanes_trap_identically(
        "list_block_then_arm",
        &block_first_site_then(
            "List[tensor[c, h, f32]]",
            "add(k, 1i64)",
            "\x20   [b]\n\
             \x20 } else []\n",
        ),
        &["extent `h`: claimed = 3, insert axis 1 = 4", INSERT_TRAP],
    );
}

/// Control: an arm site that agrees with the block's first site runs, for a
/// tuple and an `Option` result alike.
#[test]
fn an_agreeing_arm_site_of_a_non_tensor_result_binder_executes() {
    assert_lanes_agree(
        "tuple_block_then_arm_agree",
        &block_first_site_then(
            "(tensor[c, h, f32], i64)",
            "k",
            "\x20   (b, 1i64)\n\
             \x20 } else (insert(v, 1i32, k), 0i64)\n",
        ),
        "out.0 = tensor(shape=[2, 3], data=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])\nout.1 = 1",
    );
    assert_lanes_agree(
        "option_block_then_arm_agree",
        &block_first_site_then(
            "Option[tensor[c, h, f32]]",
            "k",
            "\x20   Some(b)\n\
             \x20 } else None\n",
        ),
        "out = Some(tensor(shape=[2, 3], data=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))",
    );
}

/// Known gap, pinned rather than fixed: with a rank-polymorphic signature
/// `chelis eval` runs the call through its named-axis path (chelis#338),
/// which never claims a later site, so it prints the first site's value
/// exactly as before first sites existed. Compiled C claims the later site
/// and traps, which is the section 4.4.1 answer. When eval claims it, this
/// row must become an identical-trap row.
#[test]
fn a_rank_polymorphic_later_site_is_claimed_only_by_compiled_c() {
    let stem = "rankpoly_two_sites";
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(
        &dir,
        stem,
        &format!(
            "def f[r, h](v: &tensor[..r, f32], k: i64) -> tensor[..r, h, f32] = {{\n\
             \x20 a: tensor[..r, h, f32] = insert(v, h, 3i64)\n\
             \x20 b: tensor[..r, h, f32] = insert(v, h, 4i64)\n\
             \x20 _ = b\n\
             \x20 a\n\
             }}\n\
             out = f(to_tensor([1.0f32, 2.0f32]), {RUNTIME_THREE})\n"
        ),
    );
    assert!(check(&path).status.success(), "{stem}: check accepts");
    let evaluated = eval(&path);
    let eval_out = combined(&evaluated);
    assert!(
        evaluated.status.success()
            && eval_out.contains("out = tensor(shape=[2, 3], data=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])"),
        "{stem}: eval does not claim the later site yet: {eval_out}"
    );
    let compiled = build_and_run(&dir, stem, &path);
    let c_out = combined(&compiled);
    assert!(
        !compiled.status.success()
            && c_out.contains("extent `h`: claimed = 3, insert axis 1 = 4")
            && c_out.contains(INSERT_TRAP),
        "{stem}: C claims the later site: {c_out}"
    );
}
