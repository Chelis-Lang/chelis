//! CLI execution of [05-OP-37], without relabeling compiled dropout.
mod common;

use assert_cmd::Command;
use common::key_ref;
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
            // key_ref.py: the right half of split_key(key_from_seed(42)) at 0.5.
            assert_eq!(value.data.to_f64_lossy_vec(), vec![2.0, 2.0, 0.0, 0.0]);
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
        // Independently computed by key_ref.py from the two halves of
        // split_key(key_from_seed(42)); they differ, so a fresh-mask backward
        // or omitted-forward mutant fails.
        assert_eq!(
            roots,
            [
                ("main.0", vec![2.0, 0.0, 0.0, 2.0]),
                ("main.1", vec![2.0, 2.0, 0.0, 0.0])
            ]
        );
    }
}

#[test]
fn invalid_empty_rate_traps_in_eval_and_c_and_a_runtime_rate_builds() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("dropout.ch");
    std::fs::write(&file, "def empty() -> tensor[0, f32] = to_tensor([])\ndef invalid(k: key, x: tensor[0, f32]) -> tensor[0, f32] = dropout(k, x, 1.0f32)\ndef main() = invalid(key_from_seed(42i64), empty())\n").unwrap();
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
    std::fs::write(&file, "def sample(k: key, x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(k, x, rate)\n").unwrap();
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

/// [05-OP-37] over f32 ones, keyed by `key` at `rate`, from
/// `common::key_ref`'s transcription of the spec text.
fn reference_dropout(key: u64, count: usize, rate: f32) -> Vec<f64> {
    key_ref::dropout_f32(key, &vec![1.0; count], rate)
        .into_iter()
        .map(f64::from)
        .collect()
}

fn reference_mask(key: u64) -> Vec<f64> {
    reference_dropout(key, 8, 0.5)
}

/// The Chelis bindings of `count` keys chained from `key_from_seed(seed)`:
/// `(k0, r0) = split_key(key_from_seed(seed))`, then `(kj, rj) =
/// split_key(r{j-1})`; the last remainder is dropped.
fn key_chain_source(seed: i64, count: usize) -> String {
    (0..count)
        .map(|j| {
            let parent = if j == 0 {
                format!("key_from_seed({seed}i64)")
            } else {
                format!("r{}", j - 1)
            };
            let rest = if j + 1 == count {
                "_".to_string()
            } else {
                format!("r{j}")
            };
            format!("    (k{j}, {rest}) = split_key({parent})\n")
        })
        .collect()
}

/// The reference words of [`key_chain_source`]'s keys.
fn key_chain(seed: i64, count: usize) -> Vec<u64> {
    let mut rest = key_ref::key_from_seed(seed);
    (0..count)
        .map(|_| {
            let (key, next) = key_ref::split(rest);
            rest = next;
            key
        })
        .collect()
}

/// The transcription in `common::key_ref` reproduces `key_ref.py`'s printed
/// worked values and `slice2_ref.py`'s f32 draws, so every expected draw
/// in this crate derived from it is the reference's.
#[test]
fn worked_values_match_key_ref_py() {
    let seven = key_ref::key_from_seed(7);
    assert_eq!(key_ref::key_from_seed(-1), 0xffff_ffff_ffff_ffff);
    assert_eq!(
        key_ref::split(seven),
        (0xaa38_9617_2f9a_3213, 0x8fd0_6b2e_7bad_8630)
    );
    assert_eq!(key_ref::fold_in(seven, 3), 0x53c6_f7e8_3810_b049);
    assert_eq!(key_ref::fold_in(seven, -1), 0x45c8_0b55_7fb9_4ddb);
    assert_eq!(
        key_ref::split_n(seven, 3),
        [
            0x25ea_33e6_1c10_576f,
            0x7071_24fb_ecd5_f054,
            0x8239_3615_3a56_5205
        ]
    );
    assert_eq!(key_ref::derive(seven, 2), 0x4ed9_4e35_099b_b63d);
    assert_eq!(key_ref::word(seven, 0), 0x2065_4588_fcd2_5740);
    assert_eq!(key_ref::unit(seven, 0), 0.12654528231070938);
    let bits = |values: Vec<f32>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(key_ref::uniform_f32(seven, 4, 0.0, 1.0)),
        [0x3e019516, 0x3f56526f, 0x3f2c55af, 0x3f1a0770]
    );
    assert_eq!(
        bits(key_ref::uniform_f32(seven, 4, 2.0, 5.0)),
        [0x40184bf4, 0x40905eea, 0x4080a022, 0x40738594]
    );
    assert_eq!(
        bits(key_ref::dropout_f32(seven, &[1.0, 2.0, 3.0, 4.0], 0.5)),
        [0x00000000, 0x40800000, 0x40c00000, 0x41000000]
    );
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

/// Keyed `dropout` under runtime control flow, the key form of chelis#2410:
/// only the selected arm of a runtime `if` or `match` reaches the result, and
/// each draw reads exactly the key its arm is given, so an unselected arm's
/// draw changes no other draw. A kernel `where` computes both arms, each arm's
/// draw reading its own key, and selects one. Every flag is computed from
/// data, so no lane can fold it. Rows reach the arm inline, through helpers,
/// a `match` on an ADT and on an integer, `grad`, nested arms, both arms
/// drawing from one key, explicit `do` sequencing, and eval's named-axis
/// route; `handled_pick` calls one helper twice with equal keys and opposite
/// flags, so its unselected arm must leave the helper's second draw
/// unchanged. `par` is fenced by chelis#2503 until its cross-lane effects are
/// complete.
///
/// Evidentiary status: DISPOSITION LOCK of the explicit-key semantics
/// (chelis#2413), ported from the counter-stream version of this test, which
/// pinned that an unselected arm took no ordinal. Every expected draw comes
/// from `common::key_ref`, pinned by `worked_values_match_key_ref_py`.
#[test]
fn a_keyed_dropout_under_a_runtime_arm_reads_only_its_own_key_in_eval_and_c() {
    let layer = "def layer(k: key, x: tensor[8, f32], training: bool) -> tensor[8, f32] = if training then dropout(k, x, 0.5f32) else x\n";
    let ones = "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])";
    let sum = "tensor_to_scalar(sum(copy(x), 0i32))";
    let not_training = format!("training = lt({sum}, 0.0f32)");
    let keyed = |count: usize, body: &str| {
        format!(
            "def main() = {{\n{}    {ones}\n{body}}}\n",
            key_chain_source(7, count)
        )
    };
    let keys = key_chain(7, 4);
    let mask = |index: usize| reference_mask(keys[index]);
    let kept = vec![1.0; 8];
    let plus = |left: Vec<f64>, right: Vec<f64>| -> Vec<f64> {
        left.iter()
            .zip(&right)
            .map(|(left, right)| left + right)
            .collect()
    };
    let loss = |comparison: &str| {
        format!(
            "def loss(k: key, x: tensor[8, f32]) -> tensor[f32] = if {comparison}({sum}, 0.0f32) then sum(dropout(k, x, 0.5f32), 0i32) else sum(x, 0i32)\n"
        )
    };
    let routed = "def layer(k: key, x: tensor[seq, f32], training: bool) -> f32 = tensor_to_scalar(sum(if training then dropout(k, x, 0.5f32) else x, seq))\n";
    let (pick_first, pick_second) = key_ref::split(key_ref::key_from_seed(7));
    let (model_first, model_second) = key_ref::split(keys[0]);
    let programs = [
        (
            "inline_flag",
            format!(
                "{layer}{}",
                keyed(
                    2,
                    &format!(
                        "    {not_training}\n    y = layer(k0, copy(x), training)\n    z = dropout(k1, x, 0.5f32)\n    (y, z)\n"
                    )
                )
            ),
            vec![kept.clone(), mask(1)],
        ),
        (
            "two_layers",
            format!(
                "{layer}def model(k: key, x: tensor[8, f32], training: bool) -> tensor[8, f32] = {{\n  (inner, outer) = split_key(k)\n  layer(outer, layer(inner, x, training), training)\n}}\n{}",
                keyed(
                    2,
                    &format!(
                        "    {not_training}\n    out = model(k0, copy(x), training)\n    noise = dropout(k1, x, 0.5f32)\n    (out, noise)\n"
                    )
                )
            ),
            vec![kept.clone(), mask(1)],
        ),
        (
            "two_layers_taken",
            format!(
                "{layer}def model(k: key, x: tensor[8, f32], training: bool) -> tensor[8, f32] = {{\n  (inner, outer) = split_key(k)\n  layer(outer, layer(inner, x, training), training)\n}}\n{}",
                keyed(
                    2,
                    &format!(
                        "    training = gt({sum}, 0.0f32)\n    out = model(k0, copy(x), training)\n    noise = dropout(k1, x, 0.5f32)\n    (out, noise)\n"
                    )
                )
            ),
            vec![
                key_ref::dropout_f32(
                    model_second,
                    &key_ref::dropout_f32(model_first, &[1.0; 8], 0.5),
                    0.5,
                )
                .into_iter()
                .map(f64::from)
                .collect(),
                mask(1),
            ],
        ),
        (
            "taken_then_untaken",
            format!(
                "{layer}{}",
                keyed(
                    4,
                    &format!(
                        "    flag = gt({sum}, 0.0f32)\n    a = layer(k0, copy(x), flag)\n    b = dropout(k1, copy(x), 0.5f32)\n    c = layer(k2, copy(x), not(flag))\n    d = dropout(k3, x, 0.5f32)\n    (a, b, c, d)\n"
                    )
                )
            ),
            vec![mask(0), mask(1), kept.clone(), mask(3)],
        ),
        (
            "handled_pick",
            format!(
                "def pick(k: key, x: tensor[8, f32], flag: bool) -> tensor[8, f32] = {{\n  (first, second) = split_key(k)\n  a = if flag then dropout(first, copy(x), 0.5f32) else copy(x)\n  b = dropout(second, x, 0.5f32)\n  add(a, b)\n}}\ndef main() = {{\n  {ones}\n  flag = gt({sum}, 0.0f32)\n  t = pick(key_from_seed(7i64), copy(x), flag)\n  f = pick(key_from_seed(7i64), x, not(flag))\n  (t, f)\n}}\n"
            ),
            vec![
                plus(reference_mask(pick_first), reference_mask(pick_second)),
                plus(kept.clone(), reference_mask(pick_second)),
            ],
        ),
        (
            "match_adt",
            format!(
                "type Mode =\n  | Train\n  | Infer\ndef apply_mode(k: key, x: tensor[8, f32], m: Mode) -> tensor[8, f32] =\n  match m with {{\n    | Train => dropout(k, x, 0.5f32)\n    | Infer => x\n  }}\n{}",
                keyed(
                    2,
                    &format!(
                        "    m = if gt({sum}, 100.0f32) then Train else Infer\n    y = apply_mode(k0, copy(x), m)\n    z = dropout(k1, x, 0.5f32)\n    (y, z)\n"
                    )
                )
            ),
            vec![kept.clone(), mask(1)],
        ),
        (
            "match_int",
            format!(
                "def pick(k: key, x: tensor[8, f32], n: i64) -> tensor[8, f32] =\n  match n with {{\n    | 0 => dropout(k, x, 0.5f32)\n    | _ => x\n  }}\n{}",
                keyed(
                    2,
                    &format!(
                        "    n = cast({sum}, i64)\n    y = pick(k0, copy(x), n)\n    z = dropout(k1, x, 0.5f32)\n    (y, z)\n"
                    )
                )
            ),
            vec![kept.clone(), mask(1)],
        ),
        (
            "grad_untaken",
            format!(
                "{}{}",
                loss("lt"),
                keyed(
                    2,
                    "    g = grad(loss, wrt=x)(k0, copy(x))\n    after = dropout(k1, x, 0.5f32)\n    (g, after)\n"
                )
            ),
            vec![kept.clone(), mask(1)],
        ),
        (
            "grad_taken",
            format!(
                "{}{}",
                loss("gt"),
                keyed(
                    2,
                    "    g = grad(loss, wrt=x)(k0, copy(x))\n    after = dropout(k1, x, 0.5f32)\n    (g, after)\n"
                )
            ),
            vec![mask(0), mask(1)],
        ),
        (
            "two_entries",
            format!(
                "def pick(k: key, x: tensor[8, f32], flag: bool) -> tensor[8, f32] = if flag then dropout(k, x, 0.5f32) else x\ndef first(k: key, x: tensor[8, f32], flag: bool) -> tensor[8, f32] = pick(k, x, flag)\ndef second(k: key, x: tensor[8, f32], flag: bool) -> tensor[8, f32] = pick(k, x, not(flag))\n{}",
                keyed(
                    3,
                    &format!(
                        "    flag = gt({sum}, 0.0f32)\n    p = first(k0, copy(x), flag)\n    q = second(k1, copy(x), flag)\n    r = dropout(k2, x, 0.5f32)\n    (p, q, r)\n"
                    )
                )
            ),
            vec![mask(0), kept.clone(), mask(2)],
        ),
        (
            "nested_arm",
            format!(
                "def pick(k: key, x: tensor[8, f32], a: bool, b: bool) -> tensor[8, f32] = if a then if b then dropout(k, x, 0.5f32) else x else x\n{}",
                keyed(
                    2,
                    &format!(
                        "    s = {sum}\n    y = pick(k0, copy(x), gt(s, 0.0f32), lt(s, 0.0f32))\n    z = dropout(k1, x, 0.5f32)\n    (y, z)\n"
                    )
                )
            ),
            vec![kept.clone(), mask(1)],
        ),
        (
            "both_arms_draw",
            format!(
                "def pick(k: key, x: tensor[8, f32], flag: bool) -> tensor[8, f32] = if flag then dropout(k, x, 0.5f32) else dropout(k, x, 0.25f32)\n{}",
                keyed(
                    2,
                    &format!(
                        "    y = pick(k0, copy(x), lt({sum}, 0.0f32))\n    z = dropout(k1, x, 0.5f32)\n    (y, z)\n"
                    )
                )
            ),
            vec![reference_dropout(keys[0], 8, 0.25), mask(1)],
        ),
        (
            "do_untaken",
            format!(
                "{layer}{}",
                keyed(
                    3,
                    &format!(
                        "    {not_training}\n    p = do {{ layer(k0, copy(x), training); layer(k1, copy(x), training) }}\n    z = dropout(k2, x, 0.5f32)\n    (p, z)\n"
                    )
                )
            ),
            vec![kept.clone(), mask(2)],
        ),
        (
            "named_axis_untaken",
            format!(
                "{routed}{}",
                keyed(
                    2,
                    &format!(
                        "    {not_training}\n    y = layer(k0, copy(x), training)\n    z = dropout(k1, x, 0.5f32)\n    (y, z)\n"
                    )
                )
            ),
            vec![vec![8.0], mask(1)],
        ),
        (
            "named_axis_taken",
            format!(
                "{routed}{}",
                keyed(
                    2,
                    &format!(
                        "    training = gt({sum}, 0.0f32)\n    y = layer(k0, copy(x), training)\n    z = dropout(k1, x, 0.5f32)\n    (y, z)\n"
                    )
                )
            ),
            vec![vec![mask(0).iter().sum()], mask(1)],
        ),
    ];
    // Each discriminating pair differs, so a lane that reads the wrong key,
    // the wrong arm or the wrong rate fails.
    assert_ne!(mask(0), mask(1));
    assert_ne!(mask(0), kept);
    assert_ne!(mask(0), reference_dropout(keys[0], 8, 0.25));
    assert_ne!(reference_mask(pick_first), reference_mask(pick_second));
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
