//! [05-RNG-1] and [05-OP-8]: straight-line `uniform_like` draws produce the
//! spec's words and sampler arithmetic, bit for bit, in `chelis eval`, in
//! compiled C, and in the DAG evaluator (whole-program lowering and
//! fixed-control plans). The stream programs take no `vmap`, whose ordinals
//! are chelis#2409's; a draw in an unselected arm takes no ordinal and
//! validates nothing (chelis#2410).
//!
//! The reference below transcribes the two atoms from the spec text. It never
//! calls an evaluator, a lowering, or a `chelis_types` sampler, so a lane that
//! drifts from the spec cannot also move the expectation (chelis#2408).
use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

const DTYPES: [(&str, Prim); 4] = [
    ("f32", Prim::F32),
    ("f64", Prim::F64),
    ("f16", Prim::F16),
    ("bf16", Prim::Bf16),
];
const SEEDS: [i64; 3] = [42, -1, 7];
/// One entry per call ordinal under a handler, in source order.
const DRAWS: [(f32, f32, &str, &str); 3] = [
    (2.0, 5.0, "2.0f32", "5.0f32"),
    (-1.0, 3.0, "-1.0f32", "3.0f32"),
    (0.0, 1.0, "0.0f32", "1.0f32"),
];
const LEN: u64 = 8;

// [05-RNG-1]'s `splitmix64`: add the golden gamma, then the two xor-shift
// multiplies and the final xor-shift, all modulo 2^64.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// [05-RNG-1]'s exact unit value for seed `seed`, call ordinal `c` and flat
/// element index `i`. The high 53 bits over 2^53 is exact in f64.
fn spec_unit(seed: i64, c: u64, i: u64) -> f64 {
    let word =
        splitmix64((seed as u64) ^ splitmix64(c).rotate_left(17) ^ splitmix64(i).rotate_left(41));
    (word >> 11) as f64 / (1_u64 << 53) as f64
}

/// The stored element bits of [05-OP-8]'s sampler arithmetic at each dtype,
/// over f32 bounds. [05-OP-8] requires bounds of the template's dtype; the
/// checker still admits only f32 bounds (chelis#1295), so these rows check the
/// arithmetic, not the bound signature. f64 is one f64 fused multiply-add over
/// the exactly widened bounds; f32 is one f32 fused multiply-add of
/// `round_f32(u)`; f16 and bf16 take that f32 result and narrow it once.
fn spec_bits(prim: Prim, seed: i64, c: u64, i: u64, low: f32, high: f32) -> u64 {
    let unit = spec_unit(seed, c, i);
    let narrow = (high - low).mul_add(unit as f32, low);
    match prim {
        Prim::F64 => {
            let (low, high) = (f64::from(low), f64::from(high));
            (high - low).mul_add(unit, low).to_bits()
        }
        Prim::F32 => u64::from(narrow.to_bits()),
        Prim::F16 => u64::from(half::f16::from_f32(narrow).to_bits()),
        Prim::Bf16 => u64::from(half::bf16::from_f32(narrow).to_bits()),
        other => unreachable!("{other:?}"),
    }
}

fn expected_draws(prim: Prim, seed: i64) -> Vec<u64> {
    DRAWS
        .iter()
        .enumerate()
        .flat_map(|(c, &(low, high, _, _))| {
            (0..LEN).map(move |i| spec_bits(prim, seed, c as u64, i, low, high))
        })
        .collect()
}

/// Stored bits of a value that a lane printed or widened exactly to f64.
fn stored_bits(prim: Prim, value: f64) -> u64 {
    match prim {
        Prim::F64 => value.to_bits(),
        Prim::F32 => u64::from((value as f32).to_bits()),
        Prim::F16 => u64::from(half::f16::from_f64(value).to_bits()),
        Prim::Bf16 => u64::from(half::bf16::from_f64(value).to_bits()),
        other => unreachable!("{other:?}"),
    }
}

fn binding(dtype: &str, seed_index: usize) -> String {
    format!("d_{dtype}_{seed_index}")
}

fn template_literal(dtype: &str, len: u64) -> String {
    let zeros = (0..len)
        .map(|_| format!("0.0{dtype}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("to_tensor([{zeros}])")
}

/// Every dtype and seed as a top-level handled binding drawing `DRAWS` in
/// order, so the three ordinals 0, 1 and 2 each carry distinct bounds.
fn stream_program() -> String {
    let mut program = String::new();
    for (dtype, _) in DTYPES {
        program.push_str(&format!(
            "template_{dtype} = {}\n",
            template_literal(dtype, LEN)
        ));
    }
    for (dtype, _) in DTYPES {
        for (index, seed) in SEEDS.iter().enumerate() {
            program.push_str(&format!(
                "{} = with seed({seed}i64) {{\n",
                binding(dtype, index)
            ));
            for (ordinal, (_, _, low, high)) in DRAWS.iter().enumerate() {
                program.push_str(&format!(
                    "  u{ordinal} = uniform_like(template_{dtype}, {low}, {high})\n"
                ));
            }
            program.push_str("  concat([u0, u1, u2], 0i32)\n}\n");
        }
    }
    program
}

/// Four f64 draws of four elements over [0, 1) under one handler: element
/// `i` of row `c` is exactly [05-RNG-1]'s unit value for `(42, c, i)`.
const SYMMETRY_PROGRAM: &str = "template = to_tensor([0.0f64, 0.0f64, 0.0f64, 0.0f64])\n\
square = with seed(42i64) {\n\
  r0 = uniform_like(template, 0.0f32, 1.0f32)\n\
  r1 = uniform_like(template, 0.0f32, 1.0f32)\n\
  r2 = uniform_like(template, 0.0f32, 1.0f32)\n\
  r3 = uniform_like(template, 0.0f32, 1.0f32)\n\
  concat([r0, r1, r2, r3], 0i32)\n\
}\n";

fn cli(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .unwrap()
}

fn succeeded(output: std::process::Output, what: &str) -> String {
    assert!(
        output.status.success(),
        "{what} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// `chelis eval --json`: the execution wire carries each stored element's
/// exact bits as hexadecimal.
fn eval_lane(program: &str, names: &[String]) -> Vec<Vec<u64>> {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stream.ch");
    common::write_file(&path, program);
    let stdout = succeeded(
        cli(&["eval", "--file", path.to_str().unwrap(), "--json"]),
        "chelis eval",
    );
    let json: Value = serde_json::from_str(&stdout).unwrap();
    let roots = json["roots"].as_array().expect("roots");
    names
        .iter()
        .map(|name| {
            let root = roots
                .iter()
                .find(|root| root["name"].as_str() == Some(name))
                .unwrap_or_else(|| panic!("no eval root `{name}`: {json}"));
            root["value"]["value"]["data"]["bits"]
                .as_array()
                .unwrap_or_else(|| panic!("`{name}` is not a tensor: {root}"))
                .iter()
                .map(|bits| u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap())
                .collect()
        })
        .collect()
}

/// `chelis build`, the host C toolchain, and the compiled program's printed
/// roots. Each printed decimal round-trips at its stored dtype.
fn c_lane(program: &str, names: &[(String, Prim)]) -> Vec<Vec<u64>> {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stream.ch");
    let out_dir = dir.path().join("stream-out");
    common::write_file(&path, program);
    succeeded(
        cli(&[
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ]),
        "chelis build",
    );
    assert!(common::link_generated(&out_dir, "stream.c", "stream").success());
    let stdout = succeeded(
        std::process::Command::new(out_dir.join("stream"))
            .output()
            .unwrap(),
        "compiled program",
    );
    names
        .iter()
        .map(|(name, prim)| {
            common::parse_tensor_data(&stdout, name)
                .into_iter()
                .map(|value| stored_bits(*prim, value))
                .collect()
        })
        .collect()
}

fn checked_program(program: &str) -> chelis_types::CheckedProgram {
    let parsed = chelis_surf::parser::parse_str(program).unwrap();
    let desugared = chelis_surf::desugar::desugar_program(&parsed).unwrap();
    let checked = chelis_types::check_ir_program(&desugared)
        .unwrap_or_else(|result| panic!("check failed: {:?}", result.errors));
    let checked = chelis_effects::check_program(&checked)
        .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
    chelis_types::check_linearity(&checked)
        .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"))
}

/// `def sample(x) ! { Random }` drawing each `(low, high)` from `x` in order
/// and concatenating the draws. The seed comes from the lowering or execution
/// context, so ordinal 0 is the first draw.
fn sample_source(dtype: &str, len: u64, draws: &[(&str, &str)]) -> String {
    let mut source = format!(
        "def sample(x: tensor[{len}, {dtype}]) -> tensor[{}, {dtype}] ! {{ Random }} = {{\n",
        draws.len() as u64 * len
    );
    for (ordinal, (low, high)) in draws.iter().enumerate() {
        source.push_str(&format!("  u{ordinal} = uniform_like(x, {low}, {high})\n"));
    }
    let names = (0..draws.len())
        .map(|ordinal| format!("u{ordinal}"))
        .collect::<Vec<_>>()
        .join(", ");
    source.push_str(&format!("  concat([{names}], 0i32)\n}}\n"));
    source
}

fn sample_body(checked: &chelis_types::CheckedProgram) -> chelis_deep::Expr {
    fn children(expr: &chelis_deep::Expr) -> &[chelis_deep::Expr] {
        match expr {
            chelis_deep::Expr::Node(node, _) => node.children_slice(),
            _ => panic!("tagged checked expression"),
        }
    }
    let sample = checked
        .exprs()
        .iter()
        .find(|expr| expr.tag() == Some(chelis_deep::tag::DeepTag::Def))
        .map(|def| children(def)[1].clone())
        .expect("checked sample definition");
    assert_eq!(sample.tag(), Some(chelis_deep::tag::DeepTag::Fn));
    children(&sample)[1].clone()
}

/// The two DAG evaluator lanes for `sample` under `seed`:
/// - "DAG": the key-operand lowering, whose draw keys take their ordinals
///   from the evaluation's inherited frame;
/// - "plan": a fixed-control evaluation plan, whose draws take their key from
///   the executing frame's `(seed, ordinal)`.
fn dag_lane(source: &str, prim: Prim, len: u64, seed: i64) -> (&'static str, Vec<u64>) {
    let checked = checked_program(source);
    let body = sample_body(&checked);
    let inputs: UnordMap<String, chelis_ir::dag::TensorType> = [(
        "x".to_string(),
        chelis_ir::dag::TensorType {
            dims: vec![chelis_ir::dag::DimInfo::Lit(len as usize)],
            precision: prim,
        },
    )]
    .into_iter()
    .collect();
    let template = chelis_ir::eval::TensorValue::from_storage(
        vec![len as usize],
        finalize_tensor("test", prim, RawTensor::Float(vec![0.0; len as usize])).unwrap(),
    );
    let bits = |value: &chelis_ir::eval::TensorValue| {
        value
            .to_f64_lossy_vec()
            .into_iter()
            .map(|value| stored_bits(prim, value))
            .collect::<Vec<_>>()
    };
    let dag = chelis_ir::lower::try_lower_subexpr_program(
        &body,
        inputs,
        UnordMap::new(),
        UnordMap::new(),
    )
    .unwrap();
    let mut frame = chelis_ir::eval::RandomFrame::inherited(seed as u64, 0);
    let values =
        chelis_ir::eval::eval_tensor_roots_with_frame(&dag, dag.roots(), &mut frame, |_| {
            Some(template.clone())
        })
        .unwrap();
    let next = frame.inherited_counter().expect("an inherited frame");
    assert!(next > 0, "the draw keys consumed no ordinal");
    ("DAG", bits(&values[&dag.roots()[0]]))
}

fn hex(bits: &[u64]) -> Vec<String> {
    bits.iter().map(|bits| format!("{bits:x}")).collect()
}

#[test]
fn the_reference_reproduces_the_independent_python_transcription() {
    // `rng_ref.py uniform SEED ORDINAL N LOW HIGH DTYPE`, the spec-only
    // reference the chelis#2413 randomness assessment wrote from the spec
    // text with exact rational arithmetic. Three cells, two dtypes, a
    // negative seed and a negative bound.
    for (prim, seed, ordinal, index, low, high, printed) in [
        (Prim::F32, 42, 0, 4, 2.0, 5.0, 2.093_725_7_f64),
        (Prim::F32, 42, 1, 0, 2.0, 5.0, 4.739_254_5),
        (Prim::F32, -1, 1, 3, -3.0, -1.0, -2.427_545_8),
        (Prim::F64, -1, 0, 3, 2.0, 5.0, 2.057_865_892_938_189_2),
        (Prim::F32, 7, 2, 1, 0.0, 1.0, 0.875_940_56),
    ] {
        assert_eq!(
            spec_bits(prim, seed, ordinal, index, low, high),
            stored_bits(prim, printed),
            "{prim:?} seed {seed} ordinal {ordinal} index {index}"
        );
    }
}

#[test]
fn uniform_like_draws_the_spec_stream_in_eval_c_and_the_dag_evaluator() {
    assert!(
        common::gcc_available(),
        "C toolchain required; no lane may skip"
    );
    let program = stream_program();
    let cells = DTYPES
        .iter()
        .flat_map(|&(dtype, prim)| {
            SEEDS
                .iter()
                .enumerate()
                .map(move |(index, &seed)| (binding(dtype, index), dtype, prim, seed))
        })
        .collect::<Vec<_>>();
    let names = cells
        .iter()
        .map(|(name, _, prim, _)| (name.clone(), *prim))
        .collect::<Vec<_>>();
    let plain_names = names
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let eval = eval_lane(&program, &plain_names);
    let compiled = c_lane(&program, &names);
    let draws = DRAWS
        .iter()
        .map(|(_, _, low, high)| (*low, *high))
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    for (row, (name, dtype, prim, seed)) in cells.iter().enumerate() {
        let (prim, seed) = (*prim, *seed);
        let expected = expected_draws(prim, seed);
        let dag = dag_lane(&sample_source(dtype, LEN, &draws), prim, LEN, seed);
        for (lane, actual) in [("eval", &eval[row]), ("C", &compiled[row]), (dag.0, &dag.1)] {
            if *actual != expected {
                failures.push(format!(
                    "{lane} {name} (seed {seed}):\n  actual   {:?}\n  expected {:?}",
                    hex(actual),
                    hex(&expected)
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // The reference itself discriminates seeds and ordinals.
    assert_ne!(expected_draws(Prim::F32, 42), expected_draws(Prim::F32, -1));
    let first = expected_draws(Prim::F32, 42);
    assert_ne!(first[..LEN as usize], first[2 * LEN as usize..]);
}

#[test]
fn draw_c_element_i_is_not_draw_i_element_c() {
    // chelis#2408: the retired mixing folded the ordinal and the element
    // index into the same word as `c*G ^ i*G`, so element i of draw c equalled
    // element c of draw i and every diagonal element equalled one seed-only
    // value. [05-RNG-1] rotates the two by different amounts.
    assert!(
        common::gcc_available(),
        "C toolchain required; no lane may skip"
    );
    const ROWS: usize = 4;
    let names = [("square".to_string(), Prim::F64)];
    let dag = dag_lane(
        &sample_source("f64", ROWS as u64, &[("0.0f32", "1.0f32"); ROWS]),
        Prim::F64,
        ROWS as u64,
        42,
    );
    let lanes = [
        (
            "eval",
            eval_lane(SYMMETRY_PROGRAM, &["square".to_string()]).remove(0),
        ),
        ("C", c_lane(SYMMETRY_PROGRAM, &names).remove(0)),
        dag,
    ];
    for (lane, row) in lanes {
        let square = row
            .iter()
            .map(|bits| f64::from_bits(*bits))
            .collect::<Vec<_>>();
        assert_eq!(square.len(), ROWS * ROWS, "{lane}");
        let at = |c: usize, i: usize| square[c * ROWS + i].to_bits();
        let mut mirrored = Vec::new();
        for c in 0..ROWS {
            for i in (c + 1)..ROWS {
                if at(c, i) == at(i, c) {
                    mirrored.push((c, i));
                }
            }
        }
        assert!(
            mirrored.is_empty(),
            "{lane}: draw c element i mirrors draw i element c at (c, i) = {mirrored:?}"
        );
        let diagonal = (0..ROWS).map(|c| at(c, c)).collect::<Vec<_>>();
        for (c, bits) in diagonal.iter().enumerate() {
            assert_eq!(
                diagonal.iter().filter(|other| *other == bits).count(),
                1,
                "{lane}: diagonal element {c} repeats across draws: {diagonal:x?}"
            );
        }
        for c in 0..ROWS {
            for i in 0..ROWS {
                assert_eq!(
                    at(c, i),
                    spec_unit(42, c as u64, i as u64).to_bits(),
                    "{lane}: draw {c} element {i}"
                );
            }
        }
    }
}

/// [05-RNG-1]'s draw key for ordinal `c` of a handler seeded `seed`: the
/// word's seed-and-ordinal half, which the HIP lane passes to its kernel.
fn spec_key(seed: i64, c: u64) -> u64 {
    (seed as u64) ^ splitmix64(c).rotate_left(17)
}

/// The HIP lane computes a `with seed` region's draw key at emission and
/// passes it to its device kernel. A tensor entry whose own handler draws
/// takes the HIP DAG path, and both generated entry points, the host entry
/// and its device twin, pass the key of the region's ordinal 0. The build
/// emits source only, so no GPU or HIP compiler is needed. The per-entry
/// ordinal reset for a region with several draws is the HIP emitter's
/// `each_entry_point_keys_its_scoped_draws_from_ordinal_zero`.
///
/// Evidentiary status: REGRESSION TEST. At 3b5f029d8 the build refused every
/// draw ("DAG path does not support tensor precision `key`").
#[test]
fn a_hip_tensor_entry_passes_its_handlers_draw_key_to_both_entry_points() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("noisy.ch");
    std::fs::write(
        &source,
        "def noisy(x: tensor[8, f32]) -> tensor[8, f32] = with seed(7i64) { add(x, uniform_like(copy(x), 0.0f32, 1.0f32)) }\n",
    )
    .unwrap();
    let out = dir.path().join("out");
    succeeded(
        cli(&[
            "build",
            source.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out.to_str().unwrap(),
        ]),
        "HIP build",
    );
    let emitted = std::fs::read_to_string(out.join("noisy_hip.cpp")).unwrap();
    let entry_keys = |signature: &str| {
        let body = emitted
            .split_once(signature)
            .unwrap_or_else(|| panic!("no `{signature}` in:\n{emitted}"))
            .1;
        let body = body.split("\nextern \"C\"").next().unwrap();
        body.lines()
            .filter_map(|line| {
                let (name, value) = line.trim().split_once(" = ")?;
                name.strip_prefix("unsigned long long t")?
                    .strip_suffix("_key")?;
                value.strip_suffix("ULL;")?.parse::<u64>().ok()
            })
            .collect::<Vec<_>>()
    };
    let expected = vec![spec_key(7, 0)];
    assert_eq!(
        entry_keys("extern \"C\" void noisy("),
        expected,
        "host entry"
    );
    assert_eq!(
        entry_keys("extern \"C\" void noisy_device("),
        expected,
        "device entry"
    );
}

/// [05-OP-8] in compiled C: bounds that are not finite, reversed, or whose
/// width overflows at the arithmetic dtype trap Domain as `uniform_like`
/// before the draw produces a value. The non-finite bounds come from run-time
/// arithmetic, so no literal gate can catch them first.
///
/// Evidentiary status: COVERAGE LOCK, not a regression test: compiled C traps
/// these at 3b5f029d8 too. Removing the emitted bound check makes every row
/// return a value instead.
#[test]
fn compiled_c_traps_invalid_run_time_uniform_bounds_before_the_draw() {
    assert!(
        common::gcc_available(),
        "C toolchain required; no lane may skip"
    );
    let setup = "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n    s = tensor_to_scalar(sum(copy(x), 0i32))";
    let rows = [
        (
            "infinite_high",
            "hi = div(1.0f32, sub(s, s))\n    uniform_like(x, 0.0f32, hi)",
        ),
        (
            "nan_high",
            "hi = div(sub(s, s), sub(s, s))\n    uniform_like(x, 0.0f32, hi)",
        ),
        ("reversed", "uniform_like(x, s, 1.0f32)"),
        (
            "overflowing_width",
            "uniform_like(x, mul(s, -7.5e37f32), mul(s, 7.5e37f32))",
        ),
    ];
    for (name, body) in rows {
        let dir = tempdir().unwrap();
        let source = dir.path().join(format!("{name}.ch"));
        std::fs::write(
            &source,
            format!("def main() =\n  with seed(7i64) {{\n    {setup}\n    {body}\n  }}\n"),
        )
        .unwrap();
        let out = dir.path().join("out");
        succeeded(
            cli(&[
                "build",
                source.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out.to_str().unwrap(),
            ]),
            name,
        );
        assert!(common::link_generated(&out, &format!("{name}.c"), name).success());
        let run = std::process::Command::new(out.join(name)).output().unwrap();
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(!run.status.success(), "{name} returned a value");
        assert!(
            stderr.contains("numeric trap: domain in uniform_like at f32"),
            "{name}: {stderr}"
        );
        assert!(run.stdout.is_empty(), "{name} printed a draw");
    }
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

/// One expected root: exact binary32 values, or a binary32 reduction whose
/// summation order the reference does not model.
enum Root {
    Exact(Vec<f64>),
    Sum(f64),
}

/// [05-RNG-1] enters only the selected arm of a runtime `if` or `match`, so a
/// `uniform_like` in an unselected arm neither validates its bounds nor
/// takes an ordinal (chelis#2410). A kernel `where` computes both arms, so
/// each draw there carries its arm's path condition as its activation. The
/// flags are computed from data, so no lane can fold them. Rows cover run-time
/// bounds that would trap if validated, eval's named-axis route, a local
/// ascription, a nested arm, a `match` arm, `grad` through an unselected arm,
/// and chelis#2410's own reproducer.
///
/// Evidentiary status: REGRESSION TEST. At dcc9256c4 compiled C trapped on
/// `helper_invalid_bounds` and shifted the later draw of
/// `helper_valid_bounds`, `ascribed_helper` and `issue_2410` by one ordinal,
/// and eval trapped on `routed_invalid_bounds` and shifted the later draw of
/// `routed_valid_bounds` and `routed_literal`. The other rows passed there.
#[test]
fn a_uniform_like_in_an_unselected_arm_takes_no_ordinal_in_eval_or_c() {
    assert!(
        common::gcc_available(),
        "C toolchain required; no lane may skip"
    );
    let unit = |c: u64, len: u64| {
        (0..len)
            .map(|i| {
                f64::from(f32::from_bits(
                    spec_bits(Prim::F32, 7, c, i, 0.0, 1.0) as u32
                ))
            })
            .collect::<Vec<_>>()
    };
    let ones = |len| vec![1.0; len];
    let x8 = "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])";
    let sum = "tensor_to_scalar(sum(copy(x), 0i32))";
    let handled =
        |body: &str| format!("def main() =\n  with seed(7i64) {{\n    {x8}\n{body}  }}\n");
    let noisy_helper = "def layer(x: tensor[8, f32], noisy: bool, eps: f32) -> tensor[8, f32] ! { Random } = if noisy then add(copy(x), uniform_like(x, neg(eps), eps)) else x\n";
    let routed = |bounds: &str| {
        format!(
            "def layer(x: tensor[seq, f32], noisy: bool, eps: f32) -> f32 ! {{ Random }} = tensor_to_scalar(sum(if noisy then uniform_like(x, {bounds}) else x, seq))\n"
        )
    };
    let routed_body = |comparison: &str, eps: &str| {
        format!(
            "    s = {sum}\n    noisy = {comparison}(s, 0.0f32)\n    y = layer(copy(x), noisy, {eps})\n    z = uniform_like(x, 0.0f32, 1.0f32)\n    (y, z)\n"
        )
    };
    let pick = "def pick(x: tensor[8, f32], flag: bool) -> tensor[8, f32] ! { Random } = if flag then uniform_like(x, 0.0f32, 1.0f32) else x\n";
    let rows = [
        (
            "helper_invalid_bounds",
            format!(
                "{noisy_helper}{}",
                handled(&format!(
                    "    s = {sum}\n    layer(x, lt(s, 0.0f32), sub(0.0f32, s))\n"
                ))
            ),
            vec![("main", Root::Exact(ones(8)))],
        ),
        (
            "inline_invalid_bounds",
            handled(&format!(
                "    s = {sum}\n    eps = sub(0.0f32, s)\n    if lt(s, 0.0f32) then uniform_like(x, neg(eps), eps) else x\n"
            )),
            vec![("main", Root::Exact(ones(8)))],
        ),
        (
            "helper_valid_bounds",
            format!(
                "{noisy_helper}{}",
                handled(&format!(
                    "    s = {sum}\n    y = layer(copy(x), lt(s, 0.0f32), s)\n    z = uniform_like(x, 0.0f32, 1.0f32)\n    (y, z)\n"
                ))
            ),
            vec![
                ("main.0", Root::Exact(ones(8))),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
        (
            "routed_invalid_bounds",
            format!(
                "{}{}",
                routed("neg(eps), eps"),
                handled(&routed_body("lt", "sub(0.0f32, s)"))
            ),
            vec![
                ("main.0", Root::Exact(vec![8.0])),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
        (
            "routed_valid_bounds",
            format!(
                "{}{}",
                routed("neg(eps), eps"),
                handled(&routed_body("lt", "s"))
            ),
            vec![
                ("main.0", Root::Exact(vec![8.0])),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
        (
            "routed_literal",
            format!(
                "{}{}",
                routed("0.0f32, 1.0f32"),
                handled(&routed_body("lt", "s"))
            ),
            vec![
                ("main.0", Root::Exact(vec![8.0])),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
        (
            "routed_literal_taken",
            format!(
                "{}{}",
                routed("0.0f32, 1.0f32"),
                handled(&routed_body("gt", "s"))
            ),
            vec![
                ("main.0", Root::Sum(unit(0, 8).iter().sum())),
                ("main.1", Root::Exact(unit(1, 8))),
            ],
        ),
        (
            "ascribed_inline",
            handled(&format!(
                "    noisy = lt({sum}, 0.0f32)\n    y: tensor[8, f32] = if noisy then uniform_like(copy(x), 0.0f32, 1.0f32) else copy(x)\n    z = uniform_like(x, 0.0f32, 1.0f32)\n    (y, z)\n"
            )),
            vec![
                ("main.0", Root::Exact(ones(8))),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
        (
            "ascribed_helper",
            format!(
                "{pick}{}",
                handled(&format!(
                    "    y: tensor[8, f32] = pick(copy(x), lt({sum}, 0.0f32))\n    z = uniform_like(x, 0.0f32, 1.0f32)\n    (y, z)\n"
                ))
            ),
            vec![
                ("main.0", Root::Exact(ones(8))),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
        (
            "issue_2410",
            format!(
                "{pick}{}",
                handled(&format!(
                    "    flag = gt({sum}, 0.0f32)\n    a = pick(copy(x), flag)\n    b = pick(copy(x), not(flag))\n    unused = uniform_like(copy(x), 0.0f32, 1.0f32)\n    c = uniform_like(x, 0.0f32, 1.0f32)\n    (a, b, c)\n"
                ))
            ),
            vec![
                ("main.0", Root::Exact(unit(0, 8))),
                ("main.1", Root::Exact(ones(8))),
                ("main.2", Root::Exact(unit(2, 8))),
            ],
        ),
        (
            "nested_arm",
            format!(
                "def pick(x: tensor[8, f32], a: bool, b: bool) -> tensor[8, f32] ! {{ Random }} = if a then if b then uniform_like(x, 0.0f32, 1.0f32) else x else x\n{}",
                handled(&format!(
                    "    s = {sum}\n    y = pick(copy(x), gt(s, 0.0f32), lt(s, 0.0f32))\n    z = uniform_like(x, 0.0f32, 1.0f32)\n    (y, z)\n"
                ))
            ),
            vec![
                ("main.0", Root::Exact(ones(8))),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
        (
            "match_arm",
            format!(
                "type Mode =\n  | Train\n  | Infer\ndef apply_mode(x: tensor[8, f32], m: Mode) -> tensor[8, f32] ! {{ Random }} =\n  match m with {{\n    | Train => uniform_like(x, 0.0f32, 1.0f32)\n    | Infer => x\n  }}\n{}",
                handled(&format!(
                    "    m = if gt({sum}, 100.0f32) then Train else Infer\n    y = apply_mode(copy(x), m)\n    z = uniform_like(x, 0.0f32, 1.0f32)\n    (y, z)\n"
                ))
            ),
            vec![
                ("main.0", Root::Exact(ones(8))),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
        (
            "grad_untaken_invalid_bounds",
            format!(
                "def loss(x: tensor[8, f32]) -> tensor[f32] ! {{ Random }} = {{\n  s = {sum}\n  if lt(s, 0.0f32) then sum(add(copy(x), uniform_like(x, s, neg(s))), 0i32) else sum(x, 0i32)\n}}\n{}",
                handled(
                    "    g = grad(loss)(copy(x))\n    after = uniform_like(x, 0.0f32, 1.0f32)\n    (g, after)\n"
                )
            ),
            vec![
                ("main.0", Root::Exact(ones(8))),
                ("main.1", Root::Exact(unit(0, 8))),
            ],
        ),
    ];
    assert_ne!(unit(0, 8), unit(1, 8));
    let bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| (*value as f32).to_bits())
            .collect::<Vec<_>>()
    };
    let mut failures = Vec::new();
    for (name, source, expected) in rows {
        let dir = tempdir().unwrap();
        let path = dir.path().join(format!("{name}.ch"));
        common::write_file(&path, &source);
        let eval = cli(&["eval", "--file", path.to_str().unwrap()]);
        if !eval.status.success() {
            failures.push(format!(
                "{name} eval: {}",
                String::from_utf8_lossy(&eval.stderr)
            ));
            continue;
        }
        let eval = String::from_utf8(eval.stdout).unwrap();
        let out_dir = dir.path().join("out");
        succeeded(
            cli(&[
                "build",
                path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out_dir.to_str().unwrap(),
            ]),
            name,
        );
        assert!(common::link_generated(&out_dir, &format!("{name}.c"), name).success());
        let run = std::process::Command::new(out_dir.join(name))
            .output()
            .unwrap();
        if !run.status.success() {
            failures.push(format!(
                "{name} C: {}",
                String::from_utf8_lossy(&run.stderr)
            ));
            continue;
        }
        let compiled = String::from_utf8(run.stdout).unwrap();
        for (root, expected) in &expected {
            for (lane, stdout) in [("eval", &eval), ("C", &compiled)] {
                let actual = printed_root(stdout, root);
                let matches = match expected {
                    Root::Exact(values) => bits(&actual) == bits(values),
                    Root::Sum(total) => actual.len() == 1 && (actual[0] - total).abs() < 1e-4,
                };
                if !matches {
                    failures.push(format!("{name} {lane} {root}: {actual:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
