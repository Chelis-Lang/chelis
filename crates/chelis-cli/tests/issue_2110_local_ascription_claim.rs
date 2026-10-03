//! chelis#2110: Eval and generated C enforce authored local tensor ascriptions
//! at the initializer operation with [04-NUM-9]'s typed Domain trap.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated, write_file};
use std::process::Command as StdCommand;
use tempfile::TempDir;

#[derive(Debug)]
struct LaneResult {
    success: bool,
    text: String,
}

fn combined(output: &std::process::Output) -> String {
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text
}

fn check_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run check");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn eval_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--allow-style-violations",
            "--file",
            path.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("run eval");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn c_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    assert!(
        gcc_available(),
        "the #2110 acceptance target requires an executed C lane"
    );
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    write_file(&path, source);
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            "--allow-style-violations",
            path.to_str().expect("UTF-8 path"),
            "--target",
            "c",
            "-o",
            out_dir.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("build generated C");
    assert!(
        built.status.success(),
        "{stem}: build must reach generated C: {}",
        combined(&built)
    );
    let linked = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(
        linked.success(),
        "{stem}: generated C must compile and link"
    );
    let output = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run generated binary");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn both_lanes(dir: &TempDir, stem: &str, source: &str) -> [(&'static str, LaneResult); 2] {
    let checked = check_result(dir, &format!("{stem}_check"), source);
    assert!(
        checked.success,
        "check: runtime-dependent or agreeing local ascription must be accepted: {}",
        checked.text
    );
    [
        ("eval", eval_result(dir, &format!("{stem}_eval"), source)),
        ("c", c_result(dir, &format!("{stem}_c"), source)),
    ]
}

fn assert_initializer_axis_trap(
    dir: &TempDir,
    stem: &str,
    source: &str,
    claim: &str,
    operation: &str,
    axis: usize,
) {
    let context = format!("extent `{claim}`: claimed = 2, {operation} axis {axis} = 3");
    let trap = format!("numeric trap: domain in {operation} at i64");
    for (lane, result) in both_lanes(dir, stem, source) {
        assert!(
            !result.success,
            "{lane}: local ascription must trap: {}",
            result.text
        );
        assert!(
            result.text.contains(&context),
            "{lane}: initializer owns `{context}`: {}",
            result.text
        );
        assert!(
            result.text.lines().any(|line| line == trap),
            "{lane}: exact typed trap line is missing: {}",
            result.text
        );
        assert!(
            !result.text.contains("assert") && !result.text.contains("Aborted"),
            "{lane}: this is a typed Domain trap, not a helper assertion/abort: {}",
            result.text
        );
    }
}

fn assert_initializer_trap(dir: &TempDir, stem: &str, source: &str, claim: &str, operation: &str) {
    assert_initializer_axis_trap(dir, stem, source, claim, operation, 0);
}

fn assert_pad_trap(dir: &TempDir, stem: &str, source: &str, claim: &str) {
    assert_initializer_trap(dir, stem, source, claim, "pad");
}

fn direct(body: &str) -> String {
    format!(
        "def f(x: tensor[*, f32]) -> tensor[*, f32] = {{\n  {body}\n}}\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    )
}

fn nested_same_name_ascription(required: usize) -> String {
    format!(
        "def f(x: tensor[*, f32]) -> tensor[*, f32] = {{\n  \
         y = add({{\n    \
         y: tensor[{required}, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n    \
         y\n  \
         }}, x)\n  \
         y\n\
         }}\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    )
}

fn deep_runtime_source(required: i64) -> String {
    format!(
        r#"
(defsig {{}}
  f
  (t-fn {{}}
    (t-tensor {{}} (d-name {{}} *) (t-prim {{}} f32))
    (t-tensor {{}} (d-name {{}} *) (t-prim {{}} f32))))
(def {{}}
  f
  (fn {{}}
    (params {{}} (x {{type: (t-var {{}} _)}}))
    (let {{}}
      (bind {{}}
        y
        (app {{type: (t-tensor {{}} (d-lit {{}} {required}) (t-prim {{}} f32))}}
          (var {{}} pad)
          (var {{}} x)
          (app {{}}
            (var {{}} Cons)
            (app {{}}
              (var {{}} Cons)
              (lit {{type: (t-prim {{}} i64)}} 0)
              (app {{}}
                (var {{}} Cons)
                (lit {{type: (t-prim {{}} i64)}} 0)
                (var {{}} Nil)))
            (var {{}} Nil))
          (lit {{type: (t-prim {{}} f32)}} 0.0)))
      (var {{}} y))))
(def {{}}
  out
  (app {{}}
    (var {{}} f)
    (app {{}}
      (var {{}} to_tensor)
      (app {{}}
        (var {{}} Cons)
        (lit {{type: (t-prim {{}} f32)}} 1.0)
        (app {{}}
          (var {{}} Cons)
          (lit {{type: (t-prim {{}} f32)}} 2.0)
          (app {{}}
            (var {{}} Cons)
            (lit {{type: (t-prim {{}} f32)}} 3.0)
            (var {{}} Nil)))))))
"#
    )
}

fn deep_cross_let_alias_source() -> String {
    r#"
(defsig {}
  f
  (t-fn {eff: (effects {} io)}
    (t-tensor {} (d-name {} *) (t-prim {} f32))
    (t-tensor {} (d-name {} *) (t-prim {} f32))))
(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (let {}
      (bind {}
        raw
        (app {type: (t-tensor {} (d-name {} *) (t-prim {} f32))}
          (var {} pad)
          (var {} x)
          (app {}
            (var {} Cons)
            (app {}
              (var {} Cons)
              (lit {type: (t-prim {} i64)} 0)
              (app {}
                (var {} Cons)
                (lit {type: (t-prim {} i64)} 0)
                (var {} Nil)))
            (var {} Nil))
          (lit {type: (t-prim {} f32)} 0.0)))
      (let {}
        (bind {}
          observed
          (app {}
            (var {} print)
            (lit {type: (t-prim {} string)} "between-producer-and-ascription")))
        (let {}
          (bind {}
            y
            (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} raw))
          (var {} y))))))
(def {}
  out
  (app {}
    (var {} f)
    (app {}
      (var {} to_tensor)
      (app {}
        (var {} Cons)
        (lit {type: (t-prim {} f32)} 1.0)
        (app {}
          (var {} Cons)
          (lit {type: (t-prim {} f32)} 2.0)
          (app {}
            (var {} Cons)
            (lit {type: (t-prim {} f32)} 3.0)
            (var {} Nil)))))))
"#
    .to_string()
}

fn deep_check_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    let path = dir.path().join(format!("{stem}.dp"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run Deep check");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn deep_eval_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    let path = dir.path().join(format!("{stem}.dp"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run Deep eval");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn deep_c_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    assert!(gcc_available(), "#2110 requires an executed Deep C lane");
    let path = dir.path().join(format!("{stem}.dp"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    write_file(&path, source);
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            "--deep",
            "--allow-style-violations",
            path.to_str().expect("UTF-8 path"),
            "--target",
            "c",
            "-o",
            out_dir.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("build generated C from Deep");
    assert!(
        built.status.success(),
        "{stem}: Deep build must reach generated C: {}",
        combined(&built)
    );
    let linked = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(linked.success(), "{stem}: Deep generated C must link");
    let output = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run Deep generated binary");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

#[test]
fn hand_authored_deep_runtime_ascription_checks_then_traps_on_eval_and_c() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = deep_runtime_source(2);
    let checked = deep_check_result(&dir, "deep_runtime_check", &source);
    assert!(checked.success, "Deep check: {}", checked.text);
    for (lane, result) in [
        ("eval", deep_eval_result(&dir, "deep_runtime_eval", &source)),
        ("c", deep_c_result(&dir, "deep_runtime_c", &source)),
    ] {
        assert!(!result.success, "{lane}: {}", result.text);
        assert!(
            result
                .text
                .contains("extent `2`: claimed = 2, pad axis 0 = 3"),
            "{lane}: {}",
            result.text
        );
        assert!(
            result
                .text
                .lines()
                .any(|line| line == "numeric trap: domain in pad at i64"),
            "{lane}: {}",
            result.text
        );
    }
}

#[test]
fn hand_authored_deep_agreeing_ascription_checks_and_executes_on_eval_and_c() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = deep_runtime_source(3);
    let checked = deep_check_result(&dir, "deep_agreeing_check", &source);
    assert!(checked.success, "Deep check: {}", checked.text);
    let eval = deep_eval_result(&dir, "deep_agreeing_eval", &source);
    let compiled = deep_c_result(&dir, "deep_agreeing_c", &source);
    assert!(eval.success, "Deep Eval: {}", eval.text);
    assert!(compiled.success, "Deep C: {}", compiled.text);
    assert_eq!(eval.text, compiled.text);
    assert_eq!(eval.text, "out = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n");
}

#[test]
fn hand_authored_deep_cross_let_alias_traps_before_the_intervening_effect() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = deep_cross_let_alias_source();
    let checked = deep_check_result(&dir, "deep_cross_let_alias_check", &source);
    assert!(checked.success, "Deep check: {}", checked.text);
    for (lane, result) in [
        (
            "eval",
            deep_eval_result(&dir, "deep_cross_let_alias_eval", &source),
        ),
        ("c", deep_c_result(&dir, "deep_cross_let_alias_c", &source)),
    ] {
        assert!(!result.success, "{lane}: {}", result.text);
        assert!(
            !result
                .text
                .lines()
                .any(|line| line == "between-producer-and-ascription"),
            "{lane}: the producer-owned guard must precede the nested-let effect: {}",
            result.text
        );
        assert!(
            result
                .text
                .contains("extent `2`: claimed = 2, pad axis 0 = 3"),
            "{lane}: {}",
            result.text
        );
        assert!(
            result
                .text
                .lines()
                .any(|line| line == "numeric trap: domain in pad at i64"),
            "{lane}: {}",
            result.text
        );
    }
}

#[test]
fn hand_authored_deep_static_mismatch_rejects_check_eval_and_c_build() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = deep_runtime_source(2).replace(
        "(t-tensor {} (d-name {} *) (t-prim {} f32))\n    (t-tensor {} (d-name {} *)",
        "(t-tensor {} (d-lit {} 3) (t-prim {} f32))\n    (t-tensor {} (d-name {} *)",
    );
    for (lane, result) in [
        (
            "check",
            deep_check_result(&dir, "deep_static_check", &source),
        ),
        ("eval", deep_eval_result(&dir, "deep_static_eval", &source)),
    ] {
        assert!(!result.success, "{lane}: {}", result.text);
        assert!(
            result.text.contains("tensor[2, f32]") && result.text.contains("tensor[3, f32]"),
            "{lane}: the declared and inferred extents must both survive: {}",
            result.text
        );
        assert!(
            !result.text.contains("numeric trap:"),
            "{lane}: static refutation must not reach execution: {}",
            result.text
        );
    }
    let path = dir.path().join("deep_static_build.dp");
    let out_dir = dir.path().join("deep-static-out");
    write_file(&path, &source);
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            "--deep",
            "--allow-style-violations",
            path.to_str().expect("UTF-8 path"),
            "--target",
            "c",
            "-o",
            out_dir.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("run Deep static build");
    let text = combined(&built);
    assert!(!built.status.success(), "build: {text}");
    assert!(
        text.contains("tensor[2, f32]") && text.contains("tensor[3, f32]"),
        "build must name the declared and inferred extents: {text}"
    );
    assert!(
        !text.contains("numeric trap:"),
        "build: static refutation must not reach execution: {text}"
    );
}

#[test]
fn direct_runtime_disagreement_traps_at_the_initializer_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "direct",
        &direct("y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y"),
        "2",
    );
}

#[test]
fn an_initializer_alias_keeps_the_local_claim_on_the_producing_operation() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "initializer_alias",
        &direct(
            "raw = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
             y: tensor[2, f32] = raw\n  \
             y",
        ),
        "2",
    );
}

#[test]
fn a_later_return_alias_keeps_the_local_claim_on_the_initializer() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "return_alias",
        &direct(
            "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
             returned = y\n  \
             returned",
        ),
        "2",
    );
}

#[test]
fn a_value_dead_except_for_the_obligation_still_traps() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "dead_value",
        &direct(
            "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
             x",
        ),
        "2",
    );
}

#[test]
fn an_inlined_callee_retains_its_local_ascription_obligation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def helper(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  out = helper(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    assert_pad_trap(&dir, "inlined", source, "2");
}

fn runtime_branch_inlined_helper_source(threshold: i64) -> String {
    format!(
        "def helper(x: tensor[*, f32]) -> tensor[*, f32] = {{\n  \
         y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y\n\
         }}\n\
         def f(flag: bool, x: tensor[*, f32]) -> tensor[*, f32] = \
           if flag then helper(x) else x\n\
         def run(x: tensor[*, f32]) -> tensor[*, f32] = \
           f(lt(shape(x, 0i32), {threshold}i64), x)\n\
         out = run(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    )
}

#[test]
fn an_untaken_runtime_branch_skips_an_inlined_helpers_local_ascription_on_both_lanes() {
    assert_exact_on_both_lanes(
        "runtime_branch_inlined_helper_untaken",
        &runtime_branch_inlined_helper_source(3),
    );
}

#[test]
fn a_selected_runtime_branch_enforces_an_inlined_helpers_local_ascription_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "runtime_branch_inlined_helper_selected",
        &runtime_branch_inlined_helper_source(4),
        "2",
    );
}

#[test]
fn an_invoked_local_closure_prepares_its_own_local_ascription() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  local = fn (z: tensor[*, f32]) -> {\n    \
                  y: tensor[2, f32] = pad(z, [[0i64, 0i64]], 0.0f32)\n    \
                  y\n  \
                  }\n  \
                  local(x)\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    assert_pad_trap(&dir, "local_closure", source, "2");
}

#[test]
fn an_agreeing_invoked_local_closure_executes_exactly_on_both_lanes() {
    assert_exact_on_both_lanes(
        "agreeing_local_closure",
        "def f(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
         local = fn (z: tensor[*, f32]) -> {\n    \
         y: tensor[3, f32] = pad(z, [[0i64, 0i64]], 0.0f32)\n    \
         y\n  \
         }\n  \
         local(x)\n\
         }\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
}

#[test]
fn an_uninvoked_disagreeing_local_closure_creates_no_outer_obligation() {
    assert_exact_on_both_lanes(
        "uninvoked_local_closure",
        "def f(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
         local = fn (z: tensor[*, f32]) -> {\n    \
         y: tensor[2, f32] = pad(z, [[0i64, 0i64]], 0.0f32)\n    \
         y\n  \
         }\n  \
         x\n\
         }\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
}

#[test]
fn an_untaken_static_branch_creates_no_local_ascription_obligation() {
    assert_exact_on_both_lanes(
        "untaken_static_branch",
        "def f(x: tensor[*, f32]) -> tensor[*, f32] = if false then {\n  \
         y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y\n\
         } else x\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
}

#[test]
fn an_untaken_static_match_arm_creates_no_local_ascription_obligation() {
    assert_exact_on_both_lanes(
        "untaken_static_match_arm",
        "type Choice = | First | Second\n\
         def f(choice: Choice, x: tensor[*, f32]) -> tensor[*, f32] = match choice with {\n  \
         | First => x\n  \
         | Second => {\n    \
         y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n    \
         y\n  \
         }\n\
         }\n\
         out = f(First, to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
}

#[test]
fn a_selected_static_branch_enforces_its_local_ascription_obligation() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "selected_static_branch",
        "def f(x: tensor[*, f32]) -> tensor[*, f32] = if true then {\n  \
         y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y\n\
         } else x\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
        "2",
    );
}

#[test]
fn a_selected_static_match_arm_enforces_its_local_ascription_obligation() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "selected_static_match_arm",
        "type Choice = | First | Second\n\
         def f(choice: Choice, x: tensor[*, f32]) -> tensor[*, f32] = match choice with {\n  \
         | First => x\n  \
         | Second => {\n    \
         y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n    \
         y\n  \
         }\n\
         }\n\
         out = f(Second, to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
        "2",
    );
}

#[test]
fn an_agreeing_selected_match_arm_executes_exactly_on_both_lanes() {
    assert_exact_on_both_lanes(
        "agreeing_selected_match_arm",
        "type Choice = | First | Second\n\
         def f(choice: Choice, x: tensor[*, f32]) -> tensor[*, f32] = match choice with {\n  \
         | First => x\n  \
         | Second => {\n    \
         y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n    \
         y\n  \
         }\n\
         }\n\
         out = f(Second, to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
}

fn host_effect_source(required: usize) -> String {
    format!(
        "def f(x: tensor[*, f32]) -> tensor[*, f32] ! {{ IO }} = {{\n  \
         _ = print(\"before-local-claim\")\n  \
         y: tensor[{required}, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         _ = print(\"after-local-claim\")\n  \
         y\n\
         }}\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    )
}

#[test]
fn a_host_effect_before_the_local_guard_runs_and_one_after_it_does_not() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (lane, result) in both_lanes(&dir, "host_effect_trap", &host_effect_source(2)) {
        assert!(!result.success, "{lane}: {}", result.text);
        assert_eq!(
            result
                .text
                .lines()
                .filter(|line| *line == "before-local-claim")
                .count(),
            1,
            "{lane}: {}",
            result.text
        );
        assert!(
            !result.text.lines().any(|line| line == "after-local-claim"),
            "{lane}: {}",
            result.text
        );
        assert!(
            result
                .text
                .contains("extent `2`: claimed = 2, pad axis 0 = 3"),
            "{lane}: {}",
            result.text
        );
        assert!(
            result
                .text
                .lines()
                .any(|line| line == "numeric trap: domain in pad at i64"),
            "{lane}: {}",
            result.text
        );
    }
}

#[test]
fn agreeing_host_effects_execute_in_source_order_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let [(_, eval), (_, compiled)] = both_lanes(&dir, "host_effect_agrees", &host_effect_source(3));
    for (lane, result) in [("eval", &eval), ("c", &compiled)] {
        assert!(result.success, "{lane}: {}", result.text);
        assert_eq!(
            result.text,
            "before-local-claim\nafter-local-claim\n\
             out = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n",
            "{lane}"
        );
    }
    assert_eq!(eval.text, compiled.text);
}

#[test]
fn a_host_lane_initializer_alias_keeps_pad_as_the_guard_owner() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) -> tensor[*, f32] ! { IO } = {\n  \
                  _ = print(\"before-local-alias\")\n  \
                  raw = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y: tensor[2, f32] = raw\n  \
                  _ = print(\"after-local-alias\")\n  \
                  y\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    for (lane, result) in both_lanes(&dir, "host_initializer_alias", source) {
        assert!(!result.success, "{lane}: {}", result.text);
        assert_eq!(
            result
                .text
                .lines()
                .filter(|line| *line == "before-local-alias")
                .count(),
            1,
            "{lane}: {}",
            result.text
        );
        assert!(
            !result.text.lines().any(|line| line == "after-local-alias"),
            "{lane}: {}",
            result.text
        );
        assert!(
            result
                .text
                .contains("extent `2`: claimed = 2, pad axis 0 = 3"),
            "{lane}: the alias must retain pad attribution: {}",
            result.text
        );
        assert!(
            result
                .text
                .lines()
                .any(|line| line == "numeric trap: domain in pad at i64"),
            "{lane}: {}",
            result.text
        );
    }
}

#[test]
fn an_agreeing_host_lane_initializer_alias_executes_once_in_source_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) -> tensor[*, f32] ! { IO } = {\n  \
                  _ = print(\"before-local-alias\")\n  \
                  raw = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y: tensor[3, f32] = raw\n  \
                  _ = print(\"after-local-alias\")\n  \
                  y\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    let [(_, eval), (_, compiled)] = both_lanes(&dir, "host_initializer_alias_agrees", source);
    for (lane, result) in [("eval", &eval), ("c", &compiled)] {
        assert!(result.success, "{lane}: {}", result.text);
        assert_eq!(
            result.text,
            "before-local-alias\nafter-local-alias\n\
             out = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n",
            "{lane}"
        );
    }
    assert_eq!(eval.text, compiled.text);
}

#[test]
fn a_cross_let_alias_traps_at_the_producer_before_intervening_effects() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) -> tensor[*, f32] ! { IO } = {\n  \
                  _ = print(\"before-local-alias\")\n  \
                  raw = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  _ = print(\"between-producer-and-ascription\")\n  \
                  y: tensor[2, f32] = raw\n  \
                  _ = print(\"after-local-alias\")\n  \
                  y\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    for (lane, result) in both_lanes(&dir, "host_cross_let_alias_effect_order", source) {
        assert!(!result.success, "{lane}: {}", result.text);
        assert_eq!(
            result
                .text
                .lines()
                .filter(|line| *line == "before-local-alias")
                .count(),
            1,
            "{lane}: {}",
            result.text
        );
        assert!(
            !result
                .text
                .lines()
                .any(|line| line == "between-producer-and-ascription"),
            "{lane}: the producer-owned guard must precede the intervening effect: {}",
            result.text
        );
        assert!(
            !result.text.lines().any(|line| line == "after-local-alias"),
            "{lane}: {}",
            result.text
        );
        assert!(
            result
                .text
                .contains("extent `2`: claimed = 2, pad axis 0 = 3"),
            "{lane}: the alias must retain pad attribution: {}",
            result.text
        );
        assert!(
            result
                .text
                .lines()
                .any(|line| line == "numeric trap: domain in pad at i64"),
            "{lane}: {}",
            result.text
        );
    }
}

#[test]
fn a_cross_let_alias_preserves_nested_shadowing_on_the_c_lane() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) -> tensor[*, f32] ! { IO } = {\n  \
                  marker = 1i64\n  \
                  raw = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y: tensor[3, f32] = raw\n  \
                  marker = 2i64\n  \
                  _ = print(marker)\n  \
                  y\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    let [(_, eval), (_, compiled)] = both_lanes(&dir, "host_initializer_alias_shadowing", source);
    for (lane, result) in [("eval", &eval), ("c", &compiled)] {
        assert!(result.success, "{lane}: {}", result.text);
        assert_eq!(
            result.text, "2\nout = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n",
            "{lane}"
        );
    }
    assert_eq!(eval.text, compiled.text);
}

#[test]
fn an_agreeing_nested_same_name_ascription_attaches_to_only_its_initializer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = nested_same_name_ascription(3);
    let [(_, eval), (_, compiled)] = both_lanes(&dir, "nested_same_name_agrees", &source);
    for (lane, result) in [("eval", &eval), ("c", &compiled)] {
        assert!(result.success, "{lane}: {}", result.text);
        assert_eq!(
            result.text, "out = tensor(shape=[3], data=[2.0, 4.0, 6.0])\n",
            "{lane}"
        );
        assert!(
            !result.text.contains("initializer owners"),
            "{lane}: one authored ascription must have exactly one owner: {}",
            result.text
        );
    }
    assert_eq!(eval.text, compiled.text);
}

#[test]
fn a_disagreeing_nested_same_name_ascription_traps_at_its_initializer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = nested_same_name_ascription(2);
    assert_pad_trap(&dir, "nested_same_name_disagrees", &source, "2");
}

#[test]
fn direct_top_level_grad_retains_the_forward_local_ascription_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) -> tensor[f32] = {\n  \
                  y: tensor[2, f32] = exp(neg(x))\n  \
                  sum(y, 0)\n\
                  }\n\
                  out = grad(f)(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    assert_initializer_trap(&dir, "direct_grad", source, "2", "exp");
}

#[test]
fn agreeing_direct_top_level_grad_executes_exactly_on_both_lanes() {
    assert_exact_text_on_both_lanes(
        "agreeing_direct_grad",
        "def f(x: tensor[*, f32]) -> tensor[f32] = {\n  \
         y: tensor[3, f32] = x\n  \
         sum(y, 0)\n\
         }\n\
         out = grad(f)(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
        "out = tensor(shape=[3], data=[1.0, 1.0, 1.0])\n",
    );
}

#[test]
fn direct_top_level_vmap_retains_the_shifted_local_ascription_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  out = vmap(f)(to_tensor([\n  \
                  [1.0f32, 2.0f32, 3.0f32],\n  \
                  [4.0f32, 5.0f32, 6.0f32],\n\
                  ]))\n";
    assert_initializer_axis_trap(&dir, "direct_vmap", source, "2", "pad", 1);
}

#[test]
fn agreeing_direct_top_level_vmap_executes_exactly_on_both_lanes() {
    assert_exact_text_on_both_lanes(
        "agreeing_direct_vmap",
        "def f(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
         y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y\n\
         }\n\
         out = vmap(f)(to_tensor([\n  \
         [1.0f32, 2.0f32, 3.0f32],\n  \
         [4.0f32, 5.0f32, 6.0f32],\n\
         ]))\n",
        "out = tensor(shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])\n",
    );
}

#[test]
fn a_named_local_claim_uses_its_declaring_runtime_extent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f[n](anchor: tensor[n, f32], x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[n, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  out = f(\n  \
                  to_tensor([10.0f32, 20.0f32]),\n  \
                  to_tensor([1.0f32, 2.0f32, 3.0f32]),\n\
                  )\n";
    assert_pad_trap(&dir, "named", source, "n");
}

#[test]
fn a_staged_host_partition_traps_at_its_reshape_initializer_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(source: tensor[*, f32], x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = reshape(x, [numel(source)])\n  \
                  y\n\
                  }\n\
                  out = f(\n  \
                  to_tensor([10.0f32, 20.0f32, 30.0f32]),\n  \
                  to_tensor([1.0f32, 2.0f32, 3.0f32]),\n\
                  )\n";
    assert_initializer_trap(&dir, "staged_host", source, "2", "reshape");
}

#[test]
fn a_later_same_shape_consumer_does_not_take_the_initializer_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "same_shape_consumer",
        &direct(
            "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
             add(y, y)",
        ),
        "2",
    );
}

#[test]
fn multiple_local_axis_claims_fail_in_authored_axis_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, *, f32]) -> tensor[*, *, f32] = {\n  \
                  y: tensor[2, 5, f32] = pad(\n  \
                  x,\n  \
                  [[0i64, 0i64], [0i64, 0i64]],\n  \
                  0.0f32,\n  \
                  )\n  \
                  y\n\
                  }\n\
                  out = f(to_tensor([\n  \
                  [1.0f32, 2.0f32, 3.0f32, 4.0f32],\n  \
                  [5.0f32, 6.0f32, 7.0f32, 8.0f32],\n  \
                  [9.0f32, 10.0f32, 11.0f32, 12.0f32],\n\
                  ]))\n";
    assert_pad_trap(&dir, "axis_order", source, "2");
}

#[test]
fn an_inferred_result_claim_does_not_replace_the_local_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) = {\n  \
                  y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    assert_pad_trap(&dir, "inferred_result", source, "2");
}

#[test]
fn a_static_local_disagreement_is_a_checker_error_on_every_entry_lane() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("static_disagreement.ch");
    write_file(
        &path,
        "out = {\n  \
         y: tensor[9, f32] = pad(\n  \
         to_tensor([1.0f32]),\n  \
         [[1i64, 0i64]],\n  \
         0.0f32,\n  \
         )\n  \
         y\n\
         }\n",
    );
    let out_dir = dir.path().join("static-out");
    for (lane, args) in [
        (
            "check",
            vec!["check".to_string(), path.to_str().unwrap().to_string()],
        ),
        (
            "eval",
            vec![
                "eval".to_string(),
                "--allow-style-violations".to_string(),
                "--file".to_string(),
                path.to_str().unwrap().to_string(),
            ],
        ),
        (
            "build",
            vec![
                "build".to_string(),
                "--allow-style-violations".to_string(),
                path.to_str().unwrap().to_string(),
                "--target".to_string(),
                "c".to_string(),
                "-o".to_string(),
                out_dir.to_str().unwrap().to_string(),
            ],
        ),
    ] {
        let output = Command::cargo_bin("chelis")
            .expect("chelis")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(args)
            .output()
            .expect("run lane");
        let text = combined(&output);
        assert!(!output.status.success(), "{lane}: {text}");
        assert!(text.contains("DimensionMismatch"), "{lane}: {text}");
        assert!(!text.contains("numeric trap:"), "{lane}: {text}");
    }
}

fn assert_exact_text_on_both_lanes(stem: &str, source: &str, expected: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let [(_, eval), (_, compiled)] = both_lanes(&dir, stem, source);
    assert!(eval.success, "eval: {}", eval.text);
    assert!(compiled.success, "c: {}", compiled.text);
    assert_eq!(eval.text, compiled.text);
    assert_eq!(eval.text, expected);
}

fn assert_exact_on_both_lanes(stem: &str, source: &str) {
    assert_exact_text_on_both_lanes(
        stem,
        source,
        "out = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n",
    );
}

#[test]
fn an_agreeing_local_ascription_executes_exactly_on_both_lanes() {
    assert_exact_on_both_lanes(
        "agree",
        &direct("y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y"),
    );
}

#[test]
fn inferred_metadata_does_not_create_a_local_runtime_claim() {
    assert_exact_on_both_lanes(
        "inferred",
        &direct("y = pad(x, [[0i64, 0i64]], 0.0f32)\n  y"),
    );
}

#[test]
fn a_wildcard_local_ascription_creates_no_extent_obligation() {
    assert_exact_on_both_lanes(
        "wildcard_ascription",
        &direct("y: tensor[*, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y"),
    );
}

#[test]
fn a_wildcard_host_alias_creates_no_guard_or_internal_error() {
    assert_exact_text_on_both_lanes(
        "wildcard_host_alias",
        "def f(x: tensor[*, f32]) -> tensor[*, f32] ! { IO } = {\n  \
         _ = print(\"before-wildcard\")\n  \
         raw = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y: tensor[*, f32] = raw\n  \
         _ = print(\"after-wildcard\")\n  \
         y\n\
         }\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
        "before-wildcard\nafter-wildcard\n\
         out = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n",
    );
}

#[test]
fn an_agreeing_literal_result_and_local_claim_execute_exactly() {
    assert_exact_on_both_lanes(
        "agreeing_literal_result",
        "def f(x: tensor[*, f32]) -> tensor[3, f32] = {\n  \
         y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y\n\
         }\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
}
