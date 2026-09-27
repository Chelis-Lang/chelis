//! [05-OP-69]/[05-RNG-2]: source admission and both executable lanes preserve
//! signed seed bits through `key_from_seed` (chelis#1803, in the explicit-key
//! form of chelis#2413). Expected draws come from `common::key_ref`.
use assert_cmd::Command;
use std::path::Path;
use std::process::Output;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn cli(args: &[&str]) -> Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .args(args)
        .output()
        .unwrap()
}

fn success(output: Output) -> String {
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

fn canonical(path: &Path) {
    success(cli(&["fmt", "--inplace", path.to_str().unwrap()]));
    success(cli(&["lint", "--check", path.to_str().unwrap()]));
}

// Pin the [05-RNG-2] uniform draws over [0, 1) of both halves of
// `split_key(key_from_seed(seed))`: the stored value is the unit value
// rounded to the dtype.
fn expected(seed: i64, narrow: bool) -> Vec<u64> {
    let (first, second) = common::key_ref::split(common::key_ref::key_from_seed(seed));
    [first, second]
        .into_iter()
        .flat_map(|key| (0u64..8).map(move |index| bits(common::key_ref::unit(key, index), narrow)))
        .collect()
}

fn bits(value: f64, narrow: bool) -> u64 {
    if narrow {
        u64::from((value as f32).to_bits())
    } else {
        value.to_bits()
    }
}

#[test]
fn canonical_signed_constants_execute_exact_split_draws_in_surf_and_deep() {
    for (seed_expr, seed) in [
        ("-1i64", -1),
        ("-9223372036854775808i64", i64::MIN),
        ("9223372036854775807i64", i64::MAX),
        ("4294967295i64", 4_294_967_295),
        ("1i64", 1),
    ] {
        for (dtype, narrow) in [("f32", true), ("f64", false)] {
            let dir = tempdir().unwrap();
            let surf = dir.path().join("sample.ch");
            let zeros = std::array::from_fn::<_, 8, _>(|_| format!("0.0{dtype}")).join(", ");
            common::write_file(
                &surf,
                &format!(
                    "template = to_tensor([{zeros}])\n\
                 sampled = {{\n\
                   (a, b) = split_key(key_from_seed({seed_expr}))\n\
                   first = uniform_like(a, template, 0.0f32, 1.0f32)\n\
                   second = uniform_like(b, template, 0.0f32, 1.0f32)\n\
                   concat([first, second], 0i32)\n }}\n"
                ),
            );
            canonical(&surf);
            let deep = dir.path().join("sample.dp");
            common::write_file(&deep, &success(cli(&["deep", surf.to_str().unwrap()])));
            canonical(&deep);
            for path in [&surf, &deep] {
                let checked = success(cli(&["check", path.to_str().unwrap()]));
                let checked: serde_json::Value = serde_json::from_str(&checked).unwrap();
                assert_eq!(checked["score"], 1);
                assert_eq!(checked["errors"], serde_json::json!([]));
                let eval = success(cli(&["eval", "--file", path.to_str().unwrap()]));
                let out_dir = dir.path().join(path.extension().unwrap());
                success(cli(&[
                    "build",
                    path.to_str().unwrap(),
                    "--target",
                    "c",
                    "--output",
                    out_dir.to_str().unwrap(),
                ]));
                assert!(common::link_generated(&out_dir, "sample.c", "sample").success());
                let compiled = success(
                    std::process::Command::new(out_dir.join("sample"))
                        .output()
                        .unwrap(),
                );
                for (lane, output) in [("eval", eval), ("C", compiled)] {
                    let bits = common::parse_tensor_data(&output, "sampled")
                        .into_iter()
                        .map(|value| bits(value, narrow))
                        .collect::<Vec<_>>();
                    assert_eq!(bits, expected(seed, narrow), "{seed_expr} {path:?} {lane}");
                }
            }
        }
    }
}

/// A seed that is not an `i64` is refused before execution in every command;
/// `key_from_seed` takes any `i64` value, constant or computed.
#[test]
fn wrong_dtype_seeds_reject_before_execution_and_computed_i64_seeds_check() {
    let program = |seed: &str| {
        format!(
            "def sample(runtime: i64) -> tensor[2, f32] = uniform_like(key_from_seed({seed}), \
             to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)\n"
        )
    };
    for seed in ["-1i32", "1.0f64", "true", "\"seed\""] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rejected.ch");
        common::write_file(&path, &program(seed));
        canonical(&path);
        for command in ["check", "eval", "build"] {
            let args = if command == "eval" {
                vec![command, "--file", path.to_str().unwrap()]
            } else {
                vec![command, path.to_str().unwrap()]
            };
            let output = cli(&args);
            assert!(!output.status.success(), "{seed} {command}: {output:?}");
            let diagnostics = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                diagnostics.contains("expected i64"),
                "{seed}: {diagnostics}"
            );
        }
    }
    for seed in [
        "runtime",
        "add(1i64, 2i64)",
        "cast(1.5f64, i64)",
        "neg(-1i64)",
        "cast(-1i32, i64)",
        "cast_trunc(-1.75f64, i64)",
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("accepted.ch");
        common::write_file(&path, &program(seed));
        canonical(&path);
        let checked = success(cli(&["check", path.to_str().unwrap()]));
        let checked: serde_json::Value = serde_json::from_str(&checked).unwrap();
        assert_eq!(
            checked["errors"],
            serde_json::json!([]),
            "{seed}: {checked}"
        );
    }
}

/// A local `neg` that shadows the builtin reaches `key_from_seed` as the
/// closure it is: the key is that of `add(1, runtime)`, never of the builtin
/// `neg(1)`, in eval and in C. A top-level `neg` is refused as builtin
/// shadowing.
#[test]
fn a_shadowed_neg_in_a_seed_draws_from_the_closures_value() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shadow.ch");
    common::write_file(
        &path,
        "def sample(runtime: i64) -> tensor[2, f32] = {\n\
         \x20 neg = fn (value: i64) -> add(value, runtime)\n\
         \x20 uniform_like(key_from_seed(neg(1i64)), to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)\n\
         }\n\
         def main() = sample(5i64)\n",
    );
    canonical(&path);
    let expected = common::key_ref::uniform_f32(common::key_ref::key_from_seed(6), 2, 0.0, 1.0);
    assert_ne!(
        expected,
        common::key_ref::uniform_f32(common::key_ref::key_from_seed(-1), 2, 0.0, 1.0)
    );
    let eval = success(cli(&["eval", "--file", path.to_str().unwrap()]));
    let out_dir = dir.path().join("out");
    success(cli(&[
        "build",
        path.to_str().unwrap(),
        "--target",
        "c",
        "--output",
        out_dir.to_str().unwrap(),
    ]));
    assert!(common::link_generated(&out_dir, "shadow.c", "shadow").success());
    let compiled = success(
        std::process::Command::new(out_dir.join("shadow"))
            .output()
            .unwrap(),
    );
    for (lane, output) in [("eval", eval), ("C", compiled)] {
        let actual = common::parse_tensor_data(&output, "main")
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{lane}");
    }

    let top_level = dir.path().join("top_level.ch");
    common::write_file(
        &top_level,
        "def neg(value: i64) -> i64 = add(value, 1i64)\n def sample() -> key = key_from_seed(neg(1i64))",
    );
    canonical(&top_level);
    let output = cli(&["check", top_level.to_str().unwrap()]);
    assert!(!output.status.success(), "{output:?}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["errors"].as_array().unwrap().len(), 1, "{report}");
    assert_eq!(report["errors"][0]["kind"], "BuiltinShadowing", "{report}");
}

#[test]
fn signed_seed_dropout_evaluator_preserves_canonical_masks_of_both_split_halves() {
    use chelis_compiler_api::schema::{EvalResult, ExecutionValue};
    for seed in [-1i64, i64::MIN, i64::MAX] {
        let dir = tempdir().unwrap();
        let surf = dir.path().join("dropout.ch");
        common::write_file(
            &surf,
            &format!(
                "def main() = {{\n\
             (a, b) = split_key(key_from_seed({seed}i64))\n\
             x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
             (dropout(a, x, 0.5f32), dropout(b, x, 0.5f32))\n }}\n"
            ),
        );
        canonical(&surf);
        let deep = dir.path().join("dropout.dp");
        common::write_file(&deep, &success(cli(&["deep", surf.to_str().unwrap()])));
        canonical(&deep);
        let (first, second) = common::key_ref::split(common::key_ref::key_from_seed(seed));
        for path in [&surf, &deep] {
            let output = success(cli(&["eval", "--file", path.to_str().unwrap(), "--json"]));
            let result: EvalResult = serde_json::from_str(&output).unwrap();
            assert_eq!(result.roots.len(), 2, "{result:?}");
            for (half, (root, key)) in result.roots.iter().zip([first, second]).enumerate() {
                let ExecutionValue::Tensor { value } = &root.value else {
                    panic!("{root:?}");
                };
                let expected = common::key_ref::dropout_f32(key, &[1.0; 8], 0.5)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>();
                assert_eq!(value.shape, [8]);
                let actual = value
                    .data
                    .to_f64_lossy_vec()
                    .into_iter()
                    .map(|x| (x as f32).to_bits())
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "seed {seed}, split half {half}, {path:?}");
            }
        }
    }
}
