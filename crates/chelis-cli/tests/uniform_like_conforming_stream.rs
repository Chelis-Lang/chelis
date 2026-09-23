//! [05-RNG-1] and [05-OP-8]: straight-line `uniform_like` draws produce the
//! spec's words and sampler arithmetic, bit for bit, in `chelis eval`, in
//! compiled C, and in the DAG evaluator (whole-program lowering and
//! fixed-control plans). The programs take no `vmap` or unselected arm, whose
//! ordinals are chelis#2409's and chelis#2410's.
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
            chelis_deep::Expr::List(list, _) => &list.elements[2..],
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
/// - "DAG": the legacy lowering, whose draws carry a key fixed at lowering
///   time from the lowering context's seed and counter;
/// - "plan": a fixed-control evaluation plan, whose draws take their key from
///   the executing frame's `(seed, ordinal)`.
fn dag_lanes(source: &str, prim: Prim, len: u64, seed: i64) -> [(&'static str, Vec<u64>); 2] {
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
    let (legacy, next) = chelis_ir::lower::try_lower_subexpr_program_with_random_state_progress(
        &body,
        inputs.clone(),
        UnordMap::new(),
        UnordMap::new(),
        Some(seed as u64),
        0,
    )
    .unwrap();
    assert!(next > 0, "the legacy lowering consumed no ordinal");
    let values = chelis_ir::eval::eval_tensor(
        &legacy,
        &[("x".to_string(), template.clone())].into_iter().collect(),
    )
    .unwrap();
    let legacy_bits = bits(&values[&legacy.roots()[0]]);
    let mut context =
        chelis_ir::evaluation::RandomExecutionContext::new(chelis_ir::host::RandomLoweringState {
            seed: Some(seed as u64),
            counter: 0,
        });
    let plan = chelis_ir::lower::try_lower_subexpr_evaluation_plan(
        &body,
        inputs,
        UnordMap::new(),
        UnordMap::new(),
        &context,
    )
    .unwrap();
    let values = chelis_ir::eval::eval_tensor_plan_with_strict(&plan, &mut context, |_| {
        Some(template.clone())
    })
    .unwrap();
    assert_eq!(
        context.state().counter,
        next,
        "the plan and the lowering agree on ordinals"
    );
    let plan_bits = bits(&values[&plan.dag_for_inspection().roots()[0]]);
    [("DAG", legacy_bits), ("plan", plan_bits)]
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
        let [dag, plan] = dag_lanes(&sample_source(dtype, LEN, &draws), prim, LEN, seed);
        for (lane, actual) in [
            ("eval", &eval[row]),
            ("C", &compiled[row]),
            (dag.0, &dag.1),
            (plan.0, &plan.1),
        ] {
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
    let [dag, plan] = dag_lanes(
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
        plan,
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
