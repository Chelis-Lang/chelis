//! CLI execution of [05-OP-37], without relabeling compiled dropout.
mod common;

use assert_cmd::Command;
use serde_json::Value;

fn cli(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn staged_size_example_checks_exact_mask_and_rejects_a_false_result_claim() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("staged.ch");
    let source = include_str!("../../../examples/dropout_staged_claim.ch");
    for valid in [true, false] {
        let source = if valid {
            source.to_owned()
        } else {
            source
                .replace(
                    "source = to_tensor([1.0f32, 1.0f32])",
                    "source = to_tensor([1.0f32, 1.0f32, 1.0f32])",
                )
                .replace(
                    "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])",
                    "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])",
                )
        };
        std::fs::write(&file, source).unwrap();
        let path = file.to_str().unwrap();
        assert!(cli(&["fmt", "--inplace", path]).status.success());
        assert!(cli(&["lint", "--check", path]).status.success());
        let checked = cli(&["check", path]);
        assert!(
            checked.status.success(),
            "{}",
            String::from_utf8_lossy(&checked.stderr)
        );
        let checked: Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert_eq!(checked["score"], 1.0);
        assert_eq!(checked["errors"], serde_json::json!([]));
        let output = cli(&["eval", "--file", path, "--json"]);
        if valid {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result: chelis_compiler_api::schema::EvalResult =
                serde_json::from_slice(&output.stdout).unwrap();
            let chelis_compiler_api::schema::ExecutionValue::Tensor { value } =
                &result.roots[0].value
            else {
                panic!("{result:?}")
            };
            assert_eq!(value.shape, vec![2, 2]);
            assert_eq!(value.data.to_f64_lossy_vec(), vec![2.0, 0.0, 0.0, 0.0]);
        } else {
            assert!(!output.status.success());
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(
                error.contains("claimed = 2")
                    && error.contains("reshape axis 0 = 3")
                    && error.contains("numeric trap: domain in reshape at i64"),
                "{error}"
            );
        }
    }
}

#[test]
fn executable_example_survives_format_check_and_exact_eval() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("dropout.ch");
    std::fs::write(
        &file,
        include_str!("../../../examples/dropout_fixed_stream.ch"),
    )
    .unwrap();
    let path = file.to_str().unwrap();
    let formatted = cli(&["fmt", "--inplace", path]);
    assert!(
        formatted.status.success(),
        "{}",
        String::from_utf8_lossy(&formatted.stderr)
    );
    let checked = cli(&["check", path]);
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let checked: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(checked["score"], 1.0, "{checked}");
    assert_eq!(checked["errors"], serde_json::json!([]), "{checked}");
    for command in ["fmt", "lint"] {
        assert!(cli(&[command, "--check", path]).status.success());
    }
    assert!(cli(&["validate", "--surf", path]).status.success());
    let deep = cli(&["deep", path]);
    assert!(deep.status.success());
    let deep_path = dir.path().join("dropout.dp");
    std::fs::write(&deep_path, deep.stdout).unwrap();
    let surf = cli(&["surf", deep_path.to_str().unwrap()]);
    assert!(surf.status.success());
    let roundtrip_path = dir.path().join("roundtrip.ch");
    std::fs::write(&roundtrip_path, surf.stdout).unwrap();
    let roundtrip_path = roundtrip_path.to_str().unwrap();
    assert!(cli(&["fmt", "--check", roundtrip_path]).status.success());
    let roundtrip_check = cli(&["check", roundtrip_path]);
    assert!(roundtrip_check.status.success());
    let checked: Value = serde_json::from_slice(&roundtrip_check.stdout).unwrap();
    assert_eq!(checked["score"], 1.0, "{checked}");
    assert_eq!(checked["errors"], serde_json::json!([]), "{checked}");
    for source_path in [path, path, roundtrip_path] {
        let output = cli(&["eval", "--file", source_path, "--json"]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output: Value = serde_json::from_slice(&output.stdout).unwrap();
        let result: chelis_compiler_api::schema::EvalResult =
            serde_json::from_value(output).unwrap();
        let roots = result
            .roots
            .iter()
            .filter_map(|root| match &root.value {
                chelis_compiler_api::schema::ExecutionValue::Tensor { value } => {
                    Some((root.name.as_deref().unwrap(), value.data.to_f64_lossy_vec()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        // Independently computed seed 42 ordinals 0 and 1; the second
        // differs, so a fresh-mask backward or omitted-forward mutant fails.
        assert_eq!(
            roots,
            [
                ("main.0", vec![0.0, 2.0, 0.0, 0.0]),
                ("main.1", vec![2.0, 0.0, 0.0, 0.0])
            ]
        );
    }
}

#[test]
fn invalid_empty_rate_traps_in_eval_and_c_and_a_runtime_rate_builds() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("dropout.ch");
    std::fs::write(&file, "def empty() -> tensor[0, f32] = to_tensor([])\ndef invalid(x: tensor[0, f32]) -> tensor[0, f32] = dropout(x, 1.0f32)\ndef main() = with seed(42i64) { invalid(empty()) }\n").unwrap();
    let path = file.to_str().unwrap();
    assert!(cli(&["fmt", "--inplace", path]).status.success());
    let output = cli(&["eval", "--file", path, "--json"]);
    assert!(!output.status.success());
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        text.contains("numeric trap: domain in dropout at f32"),
        "{text}"
    );
    // Compiled C traps the same invalid rate at its draw, as eval does, even
    // over an empty tensor.
    let invalid_out = dir.path().join("invalid-out");
    let output = cli(&[
        "build",
        path,
        "--target",
        "c",
        "--output",
        invalid_out.to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(common::link_generated(&invalid_out, "dropout.c", "dropout").success());
    let run = std::process::Command::new(invalid_out.join("dropout"))
        .output()
        .unwrap();
    assert!(!run.status.success());
    let text = String::from_utf8_lossy(&run.stderr);
    assert!(
        text.contains("numeric trap: domain in dropout at f32"),
        "{text}"
    );
    // A runtime rate is an ordinary operand (chelis#2411).
    std::fs::write(&file, "def sample(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = with seed(42i64) { dropout(x, rate) }\n").unwrap();
    assert!(cli(&["fmt", "--inplace", path]).status.success());
    let output = cli(&[
        "build",
        path,
        "--target",
        "c",
        "--output",
        dir.path().join("out").to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// [05-RNG-1] and [05-OP-37] over f32 ones, transcribed independently of every
/// evaluator: the mask for `ordinal` under `seed` at `rate`, whose kept
/// elements are `1 / (1 - rate)` at binary32.
fn reference_dropout(seed: u64, ordinal: u64, count: u64, rate: f32) -> Vec<f64> {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    let kept = 1.0_f32 / (1.0_f32 - rate);
    (0..count)
        .map(|index| {
            let word = mix(seed ^ mix(ordinal).rotate_left(17) ^ mix(index).rotate_left(41));
            let unit = ((word >> 11) as f64 / 9007199254740992.0) as f32;
            if unit < rate { 0.0 } else { f64::from(kept) }
        })
        .collect()
}

fn reference_mask(seed: u64, ordinal: u64, count: u64) -> Vec<f64> {
    reference_dropout(seed, ordinal, count, 0.5)
}

/// A printed root's values: a tensor's data, or a scalar as one value.
fn printed_root(stdout: &str, root: &str) -> Vec<f64> {
    let prefix = format!("{root} = ");
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{root}` in:\n{stdout}"));
    if line.starts_with("tensor(") {
        return common::parse_tensor_data(stdout, root);
    }
    vec![line.trim().parse().expect("a numeric scalar")]
}

/// [05-RNG-1]: only the selected arm of a runtime `if` or `match` is
/// entered, so a `dropout` in an unselected arm takes no ordinal. A kernel
/// `where` computes both arms, so each draw there carries its arm's path
/// condition as its activation (chelis#2410). Every flag is computed from
/// data, so no lane can fold it. Rows reach the arm inline, through helpers,
/// a `match` on an ADT and on an integer, `grad`, nested arms, both arms
/// drawing, explicit `do` sequencing, and eval's named-axis route. `par` is
/// fenced by chelis#2503 until its cross-lane effects are complete.
///
/// Evidentiary status: REGRESSION TEST for the eval `grad_untaken`,
/// `named_axis_untaken` and `named_axis_taken` rows: at dcc9256c4 eval
/// refused each with the #2410 rejection. The other rows already passed
/// there, compiled C running each branch in host code; they lock that the
/// kernel `where` lowering C now selects draws the same stream.
#[test]
fn a_dropout_in_an_unselected_runtime_arm_takes_no_ordinal_in_eval_or_c() {
    let layer = "def layer(x: tensor[8, f32], training: bool) -> tensor[8, f32] ! { Random } = if training then dropout(x, 0.5f32) else x\n";
    let ones = "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])";
    let sum = "tensor_to_scalar(sum(copy(x), 0i32))";
    let not_training = format!("training = lt({sum}, 0.0f32)");
    let handled =
        |body: &str| format!("def main() =\n  with seed(7i64) {{\n    {ones}\n{body}  }}\n");
    let mask = |ordinal| reference_mask(7, ordinal, 8);
    let kept = vec![1.0; 8];
    let plus = |left: Vec<f64>, right: Vec<f64>| -> Vec<f64> {
        left.iter()
            .zip(&right)
            .map(|(left, right)| left + right)
            .collect()
    };
    let loss = |comparison: &str| {
        format!(
            "def loss(x: tensor[8, f32]) -> tensor[f32] ! {{ Random }} = if {comparison}({sum}, 0.0f32) then sum(dropout(x, 0.5f32), 0i32) else sum(x, 0i32)\n"
        )
    };
    let routed = "def layer(x: tensor[seq, f32], training: bool) -> f32 ! { Random } = tensor_to_scalar(sum(if training then dropout(x, 0.5f32) else x, seq))\n";
    let programs = [
        (
            "inline_flag",
            format!(
                "{layer}{}",
                handled(&format!(
                    "    {not_training}\n    y = layer(copy(x), training)\n    z = dropout(x, 0.5f32)\n    (y, z)\n"
                ))
            ),
            vec![kept.clone(), mask(0)],
        ),
        (
            "two_layers",
            format!(
                "{layer}def model(x: tensor[8, f32], training: bool) -> tensor[8, f32] ! {{ Random }} = layer(layer(x, training), training)\n{}",
                handled(&format!(
                    "    {not_training}\n    out = model(copy(x), training)\n    noise = dropout(x, 0.5f32)\n    (out, noise)\n"
                ))
            ),
            vec![kept.clone(), mask(0)],
        ),
        (
            "taken_then_untaken",
            format!(
                "{layer}{}",
                handled(&format!(
                    "    flag = gt({sum}, 0.0f32)\n    a = layer(copy(x), flag)\n    b = dropout(copy(x), 0.5f32)\n    c = layer(copy(x), not(flag))\n    d = dropout(x, 0.5f32)\n    (a, b, c, d)\n"
                ))
            ),
            vec![mask(0), mask(1), kept.clone(), mask(2)],
        ),
        (
            "handled_pick",
            format!(
                "def pick(x: tensor[8, f32], flag: bool) -> tensor[8, f32] =\n  with seed(7i64) {{\n    a = if flag then dropout(copy(x), 0.5f32) else copy(x)\n    b = dropout(x, 0.5f32)\n    add(a, b)\n  }}\ndef main() = {{\n  {ones}\n  flag = gt({sum}, 0.0f32)\n  t = pick(copy(x), flag)\n  f = pick(x, not(flag))\n  (t, f)\n}}\n"
            ),
            vec![plus(mask(0), mask(1)), plus(kept.clone(), mask(0))],
        ),
        (
            "match_adt",
            format!(
                "type Mode =\n  | Train\n  | Infer\ndef apply_mode(x: tensor[8, f32], m: Mode) -> tensor[8, f32] ! {{ Random }} =\n  match m with {{\n    | Train => dropout(x, 0.5f32)\n    | Infer => x\n  }}\n{}",
                handled(&format!(
                    "    m = if gt({sum}, 100.0f32) then Train else Infer\n    y = apply_mode(copy(x), m)\n    z = dropout(x, 0.5f32)\n    (y, z)\n"
                ))
            ),
            vec![kept.clone(), mask(0)],
        ),
        (
            "match_int",
            format!(
                "def pick(x: tensor[8, f32], k: i64) -> tensor[8, f32] ! {{ Random }} =\n  match k with {{\n    | 0 => dropout(x, 0.5f32)\n    | _ => x\n  }}\n{}",
                handled(&format!(
                    "    k = cast({sum}, i64)\n    y = pick(copy(x), k)\n    z = dropout(x, 0.5f32)\n    (y, z)\n"
                ))
            ),
            vec![kept.clone(), mask(0)],
        ),
        (
            "grad_untaken",
            format!(
                "{}{}",
                loss("lt"),
                handled(
                    "    g = grad(loss)(copy(x))\n    after = dropout(x, 0.5f32)\n    (g, after)\n"
                )
            ),
            vec![kept.clone(), mask(0)],
        ),
        (
            "grad_taken",
            format!(
                "{}{}",
                loss("gt"),
                handled(
                    "    g = grad(loss)(copy(x))\n    after = dropout(x, 0.5f32)\n    (g, after)\n"
                )
            ),
            vec![mask(0), mask(1)],
        ),
        (
            "two_entries",
            format!(
                "def pick(x: tensor[8, f32], flag: bool) -> tensor[8, f32] ! {{ Random }} = if flag then dropout(x, 0.5f32) else x\ndef first(x: tensor[8, f32], flag: bool) -> tensor[8, f32] ! {{ Random }} = pick(x, flag)\ndef second(x: tensor[8, f32], flag: bool) -> tensor[8, f32] ! {{ Random }} = pick(x, not(flag))\n{}",
                handled(&format!(
                    "    flag = gt({sum}, 0.0f32)\n    p = first(copy(x), flag)\n    q = second(copy(x), flag)\n    r = dropout(x, 0.5f32)\n    (p, q, r)\n"
                ))
            ),
            vec![mask(0), kept.clone(), mask(1)],
        ),
        (
            "nested_arm",
            format!(
                "def pick(x: tensor[8, f32], a: bool, b: bool) -> tensor[8, f32] ! {{ Random }} = if a then if b then dropout(x, 0.5f32) else x else x\n{}",
                handled(&format!(
                    "    s = {sum}\n    y = pick(copy(x), gt(s, 0.0f32), lt(s, 0.0f32))\n    z = dropout(x, 0.5f32)\n    (y, z)\n"
                ))
            ),
            vec![kept.clone(), mask(0)],
        ),
        (
            "both_arms_draw",
            format!(
                "def pick(x: tensor[8, f32], flag: bool) -> tensor[8, f32] ! {{ Random }} = if flag then dropout(x, 0.5f32) else dropout(x, 0.25f32)\n{}",
                handled(&format!(
                    "    y = pick(copy(x), lt({sum}, 0.0f32))\n    z = dropout(x, 0.5f32)\n    (y, z)\n"
                ))
            ),
            vec![reference_dropout(7, 0, 8, 0.25), mask(1)],
        ),
        (
            "do_untaken",
            format!(
                "{layer}{}",
                handled(&format!(
                    "    {not_training}\n    p = do {{ layer(copy(x), training); layer(copy(x), training) }}\n    z = dropout(x, 0.5f32)\n    (p, z)\n"
                ))
            ),
            vec![kept.clone(), mask(0)],
        ),
        (
            "named_axis_untaken",
            format!(
                "{routed}{}",
                handled(&format!(
                    "    {not_training}\n    y = layer(copy(x), training)\n    z = dropout(x, 0.5f32)\n    (y, z)\n"
                ))
            ),
            vec![vec![8.0], mask(0)],
        ),
        (
            "named_axis_taken",
            format!(
                "{routed}{}",
                handled(&format!(
                    "    training = gt({sum}, 0.0f32)\n    y = layer(copy(x), training)\n    z = dropout(x, 0.5f32)\n    (y, z)\n"
                ))
            ),
            vec![vec![mask(0).iter().sum()], mask(1)],
        ),
    ];
    assert_ne!(mask(0), mask(1));
    assert_ne!(mask(0), reference_dropout(7, 0, 8, 0.25));
    let mut failures = Vec::new();
    for (name, source, expected) in programs {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(format!("{name}.ch"));
        std::fs::write(&file, &source).unwrap();
        let path = file.to_str().unwrap();
        assert!(cli(&["fmt", "--inplace", path]).status.success(), "{name}");
        let eval = cli(&["eval", "--file", path]);
        let eval = if eval.status.success() {
            String::from_utf8(eval.stdout).unwrap()
        } else {
            failures.push(format!(
                "{name} eval: {}",
                String::from_utf8_lossy(&eval.stderr)
            ));
            continue;
        };
        let compiled = common::build_and_run(&std::fs::read_to_string(&file).unwrap(), name);
        // Printed decimals round-trip at binary32, the roots' dtype.
        let bits = |values: &[f64]| {
            values
                .iter()
                .map(|value| (*value as f32).to_bits())
                .collect::<Vec<_>>()
        };
        for (index, expected) in expected.iter().enumerate() {
            let root = format!("main.{index}");
            for (lane, stdout) in [("eval", &eval), ("C", &compiled)] {
                let actual = printed_root(stdout, &root);
                if bits(&actual) != bits(expected) {
                    failures.push(format!("{name} {lane} {root}: {actual:?} != {expected:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
