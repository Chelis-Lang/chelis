//! [05-RNG-1]: source admission and both executable lanes preserve signed seed bits.
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

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn splitmix64(value: u64) -> u64 {
    mix64(value.wrapping_add(0x9e3779b97f4a7c15))
}

// Pin the [05-RNG-1] uniform stream over [0, 1), including the next call
// ordinal: the stored value is the unit value rounded to the dtype.
fn expected(seed: i64, narrow: bool) -> Vec<u64> {
    (0u64..2)
        .flat_map(|ordinal| {
            (0u64..8).map(move |index| {
                let word = splitmix64(
                    seed as u64
                        ^ splitmix64(ordinal).rotate_left(17)
                        ^ splitmix64(index).rotate_left(41),
                );
                bits((word >> 11) as f64 / 9_007_199_254_740_992.0, narrow)
            })
        })
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
fn canonical_signed_constants_execute_exact_first_and_next_draw_in_surf_and_deep() {
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
                 sampled = with seed({seed_expr}) {{\n\
                   first = uniform_like(template, 0.0f32, 1.0f32)\n\
                   second = uniform_like(template, 0.0f32, 1.0f32)\n\
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

#[test]
fn wrong_dtype_runtime_and_nonstatic_seed_controls_reject_before_execution() {
    for seed in [
        "-1i32",
        "1.0f64",
        "true",
        "\"seed\"",
        "runtime",
        "add(1i64, 2i64)",
        "cast(1.5f64, i64)",
        "neg(-9223372036854775808i64)",
        "neg(-1i64)",
        "cast(-1i32, i64)",
        "cast_trunc(-1.75f64, i64)",
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rejected.ch");
        common::write_file(
            &path,
            &format!("def sample(runtime: i64) -> f32 = with seed({seed}) {{ 1.0f32 }}\n"),
        );
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
            assert!(diagnostics.contains("seed"), "{seed}: {diagnostics}");
        }
    }
}

#[test]
fn shadowed_neg_cannot_substitute_a_constant_for_a_runtime_seed() {
    for (source, kind) in [
        (
            "def sample(runtime: i64) -> f32 = {\n neg = fn (value: i64) -> add(value, runtime)\n with seed(neg(1i64)) { 1.0f32 }\n }",
            "TypeMismatch",
        ),
        (
            "def neg(value: i64) -> i64 = add(value, 1i64)\n def sample() -> f32 = with seed(neg(1i64)) { 1.0f32 }",
            "BuiltinShadowing",
        ),
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shadow.ch");
        common::write_file(&path, source);
        canonical(&path);
        let output = cli(&["check", path.to_str().unwrap()]);
        assert!(!output.status.success(), "{source}: {output:?}");
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["errors"].as_array().unwrap().len(), 1, "{report}");
        assert_eq!(report["errors"][0]["kind"], kind, "{report}");
        if kind == "TypeMismatch" {
            assert!(
                report["errors"][0]["message"]
                    .as_str()
                    .unwrap()
                    .contains("shadowed `neg`"),
                "{report}"
            );
        }
    }
}

#[test]
fn signed_seed_dropout_evaluator_preserves_canonical_masks_and_next_draw() {
    use chelis_compiler_api::schema::{EvalResult, ExecutionValue};
    let splitmix = splitmix64;
    for seed in [-1i64, i64::MIN, i64::MAX] {
        let dir = tempdir().unwrap();
        let surf = dir.path().join("dropout.ch");
        common::write_file(
            &surf,
            &format!(
                "def main() = with seed({seed}i64) {{\n\
             x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
             (dropout(x, 0.5f32), dropout(x, 0.5f32))\n }}\n"
            ),
        );
        canonical(&surf);
        let deep = dir.path().join("dropout.dp");
        common::write_file(&deep, &success(cli(&["deep", surf.to_str().unwrap()])));
        canonical(&deep);
        for path in [&surf, &deep] {
            let output = success(cli(&["eval", "--file", path.to_str().unwrap(), "--json"]));
            let result: EvalResult = serde_json::from_str(&output).unwrap();
            assert_eq!(result.roots.len(), 2, "{result:?}");
            for (ordinal, root) in result.roots.iter().enumerate() {
                let ExecutionValue::Tensor { value } = &root.value else {
                    panic!("{root:?}");
                };
                let expected = (0..8)
                    .map(|index| {
                        let word = splitmix(
                            seed as u64
                                ^ splitmix(ordinal as u64).rotate_left(17)
                                ^ splitmix(index).rotate_left(41),
                        );
                        let unit = ((word >> 11) as f64 / 9_007_199_254_740_992.0) as f32;
                        if unit < 0.5 {
                            0.0f32.to_bits()
                        } else {
                            2.0f32.to_bits()
                        }
                    })
                    .collect::<Vec<_>>();
                assert_eq!(value.shape, [8]);
                let actual = value
                    .data
                    .to_f64_lossy_vec()
                    .into_iter()
                    .map(|x| (x as f32).to_bits())
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "seed {seed}, ordinal {ordinal}, {path:?}");
            }
        }
    }
}
