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
