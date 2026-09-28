//! chelis#2413, spec/04 section 4.7: a def whose body is the explicit-key form
//! keeps its declared result claim. The claim belongs to the primitive that
//! produced the returned tensor, here the draw, so a wrongly sized result
//! traps in compiled C exactly as it does in `chelis eval --file` and in the
//! DAG evaluator, and the correctly sized twin returns the reference draw in
//! every lane.
//!
//! On main a `with seed` body lost the claim in compiled C: C returned the
//! wrongly sized tensor where eval trapped. The switch deletes `with seed`,
//! and this file locks the key form that replaces it in every lane the
//! program reaches (asserted on each row, not assumed):
//!
//! - `chelis eval --file` on the whole program `out = f(..)`;
//! - the whole program's host C: `chelis build`, whose `main` calls `f`'s
//!   host function. For a pure body that function wraps the draw's tensor
//!   kernel, which checks the claim the helper took over; for a body with
//!   effects, a host frame passes its declared claim into that kernel;
//! - `f` selected with `compile_for_execution`: its scalar `key` parameter
//!   keeps it a host entry (`NotTensorSignature`), one C function taking a
//!   `chelis_key` and the tensor;
//! - for a pure body, `g`, which calls `f` with a key of its own and is the
//!   only way to put `f` under a selected root with tensor inputs: the
//!   evaluator on `eval_selected(g)` and `g`'s selected C entry, which is
//!   the four-argument Tensor entry for both bodies. For a draw from the
//!   parameter or from `split_key`, `g` is a Tensor-lane root and the
//!   evaluator is the DAG evaluator. Tensor key operations preserve the
//!   selected root's tensor path through destructuring.
//!
//! Expected draws come from the `key_ref` transcription of
//! `briefs/switch-design-probes/key_ref.py`, never from a lane. Every row is
//! collected before the assertion, so one red row never hides a sibling.

mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

use assert_cmd::Command;
use chelis_compiler_api::compiler::{EntryLaneDecline, compile_for_execution, eval_selected};
use chelis_compiler_api::schema::{
    CompileRequest, CompileTarget, EvalRequest, ExecutionValue, SourceKind, TensorValue,
};
use chelis_types::types::Lane;
use common::key_ref;
use std::collections::BTreeMap;
use std::fs;

const INPUT: [f32; 3] = [1.0, 2.0, 3.0];
const SEED: i64 = 7;
const RATE: f32 = 0.5;
const CALL: &str = "out = f(key_from_seed(7i64), to_tensor([1.0, 2.0, 3.0]))\n";

/// `g` calls `f` with key 7. It sums the draw, because a Tensor entry has no
/// C representation for a runtime-extent result or input (chelis#600), and a
/// literal result extent would add a claim of `g`'s own that traps with the
/// same text as `f`'s. Its input is literal for the same reason.
const SELECTOR: &str =
    "def g(x: tensor[3, f32]) -> tensor[f32] = sum(f(key_from_seed(7i64), x), 0i32)\n";

/// One key-form body. `{n}` is `f`'s declared extent; every body returns a
/// three-element draw, so `{n}` = 3 agrees and `{n}` = 2 traps.
struct Shape {
    name: &'static str,
    def: &'static str,
    /// The key the draw consumes, from the reference.
    key: fn() -> u64,
    /// Whether `f`'s body prints before and after the draw (a Host-lane
    /// body).
    effects: bool,
    /// The lane `eval_selected(g)` reaches, for a pure body; `g` is not
    /// built for a body with effects.
    g_lane: Option<Lane>,
}

fn seed_key() -> u64 {
    key_ref::key_from_seed(SEED)
}

fn left_half() -> u64 {
    key_ref::split(key_ref::key_from_seed(SEED)).0
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "draw_from_a_key_parameter",
        def: "def f(k: key, x: tensor[*, f32]) -> tensor[{n}, f32] = dropout(k, x, 0.5f32)\n",
        key: seed_key,
        effects: false,
        g_lane: Some(Lane::Tensor),
    },
    Shape {
        name: "split_then_draw",
        def: "def f(k: key, x: tensor[*, f32]) -> tensor[{n}, f32] = {\n  \
              (a, b) = split_key(k)\n  \
              dropout(a, x, 0.5f32)\n}\n",
        key: left_half,
        effects: false,
        // `split_key` now lowers both halves into the tensor DAG, so this
        // selector reaches the tensor lane with the same result-claim guard.
        g_lane: Some(Lane::Tensor),
    },
    Shape {
        name: "draw_from_a_key_parameter_between_effects",
        def: "def f(k: key, x: tensor[*, f32]) -> tensor[{n}, f32] ! { IO } = {\n  \
              _ = print(\"before\")\n  \
              r = dropout(k, x, 0.5f32)\n  \
              _ = print(\"after\")\n  \
              r\n}\n",
        key: seed_key,
        effects: true,
        g_lane: None,
    },
    Shape {
        name: "split_then_draw_between_effects",
        def: "def f(k: key, x: tensor[*, f32]) -> tensor[{n}, f32] ! { IO } = {\n  \
              _ = print(\"before\")\n  \
              (a, b) = split_key(k)\n  \
              r = dropout(a, x, 0.5f32)\n  \
              _ = print(\"after\")\n  \
              r\n}\n",
        key: left_half,
        effects: true,
        g_lane: None,
    },
];

fn def_source(shape: &Shape, extent: usize) -> String {
    shape.def.replace("{n}", &extent.to_string())
}

fn program(shape: &Shape, extent: usize) -> String {
    format!("{}{CALL}", def_source(shape, extent))
}

/// `f`'s reference draw.
fn draw(shape: &Shape) -> Vec<f32> {
    key_ref::dropout_f32((shape.key)(), &INPUT, RATE)
}

/// `g`'s reference result: the sum of `f`'s reference draw.
fn draw_sum(shape: &Shape) -> Vec<f32> {
    vec![draw(shape).iter().sum()]
}

/// The lines a failing result claim produces, in order, wherever they occur.
fn trap_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| line.contains("extent `") || line.contains("numeric trap:"))
        .map(|line| line.trim_start_matches("error: ").to_string())
        .collect()
}

fn claim_trap() -> Vec<String> {
    vec![
        "extent `2`: claimed = 2, dropout axis 0 = 3".to_string(),
        "numeric trap: domain in dropout at i64".to_string(),
    ]
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// One selected run: the lane or ABI it reached, when the run shows it, and
/// `Ok(values)` or `Err(trap lines)`.
type Outcome = (Option<String>, Result<Vec<f32>, Vec<String>>);

/// `x`, the selection's only runtime input.
fn bindings() -> BTreeMap<String, TensorValue> {
    let x: TensorValue = serde_json::from_value(serde_json::json!({
        "shape": [INPUT.len()],
        "data": {"dtype": "f32", "bits": INPUT.iter().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>()},
    }))
    .expect("an f32 tensor binding");
    BTreeMap::from([("x".to_string(), x)])
}

/// `eval_selected(g)`. The lane shows only on a returned result.
fn evaluator(shape: &Shape, extent: usize) -> Outcome {
    let source = format!("{}{SELECTOR}", def_source(shape, extent));
    let outcome = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: bindings(),
        },
        &["g".to_string()],
    );
    match outcome {
        Ok(result) => {
            let lane = result
                .manifest
                .entries
                .iter()
                .find(|entry| entry.name == "g")
                .map(|entry| format!("{:?}", entry.lane));
            let [root] = result.roots.as_slice() else {
                panic!("one root: {:?}\n{source}", result.roots)
            };
            let ExecutionValue::Tensor { value } = &root.value else {
                panic!("a tensor result: {:?}\n{source}", root.value)
            };
            assert!(
                value.shape.is_empty(),
                "`g` returns a rank-0 tensor:\n{source}"
            );
            let values = (0..value.data.len())
                .map(|index| value.data.element_f64_lossy(index) as f32)
                .collect();
            (lane, Ok(values))
        }
        Err(error) => (
            None,
            Err(trap_lines(
                &error
                    .errors
                    .iter()
                    .map(|error| error.message.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )),
        ),
    }
}

/// A C driver for the selected entry on [`INPUT`] (and key 7 for `f`); it
/// prints the result's f32 bits. `tensor_abi` selects the four-argument
/// Tensor entry, otherwise the host entry's one C function.
fn driver(name: &str, tensor_abi: bool, symbol: &str) -> String {
    let fill = INPUT
        .iter()
        .enumerate()
        .map(|(index, value)| format!("x_data[{index}] = {value:?}f;"))
        .collect::<Vec<_>>()
        .join(" ");
    let call = if tensor_abi {
        format!(
            "chelis_tensor *inputs[1] = {{x}};\n    \
             chelis_tensor *outputs[1] = {{NULL}};\n    \
             {symbol}(inputs, 1, outputs, 1);\n    \
             chelis_tensor_release(x);\n    \
             chelis_tensor *out = outputs[0];\n"
        )
    } else if name == "f" {
        format!("chelis_tensor *out = {symbol}(chelis_key_from_seed({SEED}), x);\n")
    } else {
        format!("chelis_tensor *out = {symbol}(x);\n")
    };
    format!(
        "\nint main(void) {{\n    \
         int64_t n = {count};\n    \
         chelis_tensor *x = chelis_alloc(1, &n, CHELIS_DTYPE_F32);\n    \
         chelis_tensor_write *x_guard = chelis_tensor_begin_write(x);\n    \
         float *x_data = (float *)chelis_tensor_write_view(x_guard).data;\n    \
         {fill}\n    \
         chelis_tensor_end_write(x_guard);\n    \
         {call}    \
         chelis_read_view view = chelis_tensor_read_view(out);\n    \
         printf(\"out =\");\n    \
         for (int64_t i = 0; i < view.count; ++i) {{\n        \
         float value = ((const float *)view.data)[i];\n        \
         uint32_t bits;\n        \
         memcpy(&bits, &value, sizeof bits);\n        \
         printf(\" %08x\", bits);\n    \
         }}\n    \
         printf(\"\\n\");\n    \
         chelis_tensor_release(out);\n    \
         return 0;\n}}\n",
        count = INPUT.len(),
    )
}

/// The selected entry `name` (`compile_for_execution`), linked against the
/// runtime a `chelis build` of the whole program stages, and run natively.
/// The ABI reached is `Tensor` when the entry-scoped lane claimed the
/// compilation, otherwise `Host` (with the decline).
fn entry_c(shape: &Shape, extent: usize, name: &str) -> Outcome {
    let source = if name == "g" {
        format!("{}{SELECTOR}", def_source(shape, extent))
    } else {
        def_source(shape, extent)
    };
    let artifact = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.clone(),
        target: CompileTarget::C,
        entry_name: Some(name.into()),
    })
    .unwrap_or_else(|error| panic!("`{name}` compiles as an entry: {error:?}\n{source}"));
    let tensor_abi = artifact.entry_lane_decline.is_none();
    let (abi, symbol) = if tensor_abi {
        let inputs = artifact
            .inputs
            .iter()
            .map(|input| (input.name.as_str(), input.dtype.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(inputs, [("x", "f32")], "{source}");
        ("Tensor".to_string(), artifact.host_entry_name.clone())
    } else {
        let decline = match &artifact.entry_lane_decline {
            Some(EntryLaneDecline::NotTensorSignature { entry }) => {
                format!("NotTensorSignature({entry})")
            }
            other => format!("{other:?}"),
        };
        (format!("Host {decline}"), common::authored_c_symbol(name))
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("completion.ch");
    fs::write(&path, program(shape, extent)).expect("fixture");
    let staged = dir.path().join("c");
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--allow-style-violations"])
        .arg(&path)
        .args(["--target", "c", "-o"])
        .arg(&staged)
        .output()
        .expect("build");
    assert!(built.status.success(), "{built:?}");
    let stem = &artifact.compile_result.entry_name;
    let mut generated = None;
    for file in &artifact.compile_result.files {
        if file.path == format!("{stem}.c") {
            generated = Some(file.contents.clone());
        } else if file.path == format!("{stem}.h") {
            fs::write(staged.join(&file.path), &file.contents).expect("entry header");
        }
    }
    let generated = generated.unwrap_or_else(|| panic!("no `{stem}.c` in the artifact"));
    assert!(
        !generated.contains("\nint main(void)"),
        "`{name}` compiled as a whole program, not as its selected entry:\n{source}"
    );
    fs::write(
        staged.join("entry.c"),
        format!(
            "#include <string.h>\n{generated}{}",
            driver(name, tensor_abi, &symbol)
        ),
    )
    .expect("entry source");
    assert!(common::link_generated(&staged, "entry.c", "entry").success());
    let output = std::process::Command::new(staged.join("entry"))
        .output()
        .expect("execute the selected entry");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        return (Some(abi), Err(trap_lines(&stderr)));
    }
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("out ="))
        .unwrap_or_else(|| panic!("no result line:\n{stdout}{stderr}"));
    let values = line
        .split_whitespace()
        .map(|word| f32::from_bits(u32::from_str_radix(word, 16).expect("hex bits")))
        .collect();
    (Some(abi), Ok(values))
}

/// One lane of the whole program through the CLI (`native` selects the
/// whole program's host C, otherwise `chelis eval --file`).
fn whole_program(shape: &Shape, native: bool, rows: &mut Vec<String>) {
    let lane = if native {
        "whole-program C"
    } else {
        "eval --file"
    };
    let failing = program(shape, 2);
    let (ok, output) = result_claims::run(&failing, native);
    if ok
        || trap_lines(&output) != claim_trap()
        || output.contains("out =")
        || output.contains("after")
    {
        rows.push(format!(
            "{} ({lane}): the wrong size must trap with {:?} before any later effect\n{failing}\n{output}",
            shape.name,
            claim_trap()
        ));
    }
    let agreeing = program(shape, 3);
    let (ok, output) = result_claims::run(&agreeing, native);
    let returned = output
        .lines()
        .any(|line| line.starts_with("out = tensor(shape=[3], data="));
    let values = returned.then(|| common::parse_tensor_data(&output, "out"));
    let reference: Vec<f64> = draw(shape).into_iter().map(f64::from).collect();
    if !ok || values.as_ref() != Some(&reference) || output.contains("numeric trap:") {
        rows.push(format!(
            "{} ({lane}): the right size must return {reference:?}\n{agreeing}\n{output}",
            shape.name
        ));
    }
    if shape.effects && !(output.contains("before") && output.contains("after")) {
        rows.push(format!(
            "{} ({lane}): both effects must run\n{output}",
            shape.name
        ));
    }
}

/// A selected lane: the wrong size traps with the claim, the right size
/// returns `reference`, and the lane reached is `lane` wherever the run shows
/// it.
fn selected(
    shape: &Shape,
    row: &str,
    run: &dyn Fn(usize) -> Outcome,
    lane: &str,
    reference: &[f32],
    rows: &mut Vec<String>,
) {
    for extent in [2, 3] {
        let (reached, outcome) = run(extent);
        if let Some(reached) = reached
            && reached != lane
        {
            rows.push(format!(
                "{} ({row}, extent {extent}): reached {reached}, not {lane}",
                shape.name
            ));
        }
        match (extent, outcome) {
            (2, Err(lines)) if lines == claim_trap() => {}
            (3, Ok(values)) if bits(&values) == bits(reference) => {}
            (extent, outcome) => rows.push(format!(
                "{} ({row}, extent {extent}): expected {}, got {outcome:?}",
                shape.name,
                if extent == 2 {
                    format!("the trap {:?}", claim_trap())
                } else {
                    format!("{reference:?}")
                }
            )),
        }
    }
}

// LOCK. At the base of this file (a53c21349) every lane already traps on the
// wrong size and returns the reference draw on the right one; the `with seed`
// form that lost the claim on main no longer parses. The rows pin that the
// key form reaches the claim in each lane: the helper's transferred claim
// for a pure body, the host frame's declared claim for one with effects.
#[test]
fn a_key_form_body_keeps_its_declared_result_claim_in_every_lane() {
    assert!(common::gcc_available(), "the compiled C lanes must execute");
    let mut rows = Vec::new();
    for shape in SHAPES {
        whole_program(shape, false, &mut rows);
        whole_program(shape, true, &mut rows);
        selected(
            shape,
            "selected C entry `f`",
            &|extent| entry_c(shape, extent, "f"),
            "Host NotTensorSignature(f)",
            &draw(shape),
            &mut rows,
        );
        let Some(g_lane) = shape.g_lane else {
            continue;
        };
        let evaluator_row = match g_lane {
            Lane::Tensor => "DAG evaluator on `g`",
            _ => "host interpreter on `g`",
        };
        selected(
            shape,
            evaluator_row,
            &|extent| evaluator(shape, extent),
            &format!("{g_lane:?}"),
            &draw_sum(shape),
            &mut rows,
        );
        selected(
            shape,
            "C Tensor entry `g`",
            &|extent| entry_c(shape, extent, "g"),
            "Tensor",
            &draw_sum(shape),
            &mut rows,
        );
    }
    assert!(
        rows.is_empty(),
        "{} row(s) failed:\n\n{}",
        rows.len(),
        rows.join("\n\n")
    );
}
