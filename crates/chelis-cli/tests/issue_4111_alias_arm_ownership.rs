//! chelis#4111: a join arm that returns an alias of a binding the enclosing
//! scope reads later must not consume that binding.
//!
//! The reported program bound `scores` to a block whose rank-three reshape
//! read extents gathered from a tensor, aliased it as `input` inside a second
//! block whose `if` returned `input` on one arm, and read `shape(scores, 0)`
//! afterwards. `chelis check` and `chelis eval` accepted it; `chelis build`
//! failed with `owner %12 in `forward` b4 is not live`. Ownership lowering had
//! recorded `scores` at the depth its block minted it, below its three nested
//! bindings, so the arm treated the enclosing binding as its own and moved it.
//!
//! Every program here runs through `chelis eval --json` (dtype and payload)
//! and through the native executable built from the same file (payload), and
//! both are compared to independently computed values. Each program binds two
//! roots whose inputs differ in length and in the gathered extents, one taking
//! the alias path and one the fresh path.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

fn cli(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_chelis"))
        .args(arguments)
        .output()
        .expect("execute chelis")
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn write(directory: &Path, name: &str, source: &str) -> PathBuf {
    let path = directory.join(format!("{name}.ch"));
    let program = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .unwrap_or_else(|error| panic!("{name} parses: {error:?}\n{source}"));
    std::fs::write(&path, chelis_surf::format::format_program(&program)).unwrap();
    for command in ["fmt", "lint"] {
        let output = cli(&[command, "--check", path.to_str().unwrap()]);
        assert!(output.status.success(), "{name}: {}", text(&output));
    }
    path
}

fn check_report(path: &Path) -> Value {
    let output = cli(&["check", path.to_str().unwrap()]);
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("check emits JSON: {}", text(&output)))
}

fn build(path: &Path, output: &Path) -> Output {
    cli(&[
        "build",
        path.to_str().unwrap(),
        "--target",
        "c",
        "-o",
        output.to_str().unwrap(),
    ])
}

/// One expected root: its name, the eval dtype, and its payload.
enum Payload {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl Payload {
    fn dtype(&self) -> &'static str {
        match self {
            Payload::F32(_) => "f32",
            Payload::I64(_) => "int64",
        }
    }

    fn len(&self) -> usize {
        match self {
            Payload::F32(values) => values.len(),
            Payload::I64(values) => values.len(),
        }
    }

    /// The executable's printed form: f32 elements print with a decimal
    /// point (`2.0`), i64 elements without.
    fn native(&self) -> String {
        match self {
            Payload::F32(values) => values
                .iter()
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>()
                .join(", "),
            Payload::I64(values) => values
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        }
    }

    /// The evaluator's JSON tensor data must carry this dtype and payload.
    fn assert_eval(&self, data: &Value, context: &str) {
        assert_eq!(data["dtype"], self.dtype(), "{context}");
        match self {
            Payload::F32(values) => {
                let bits: Vec<String> = values
                    .iter()
                    .map(|value| format!("{:08x}", value.to_bits()))
                    .collect();
                assert_eq!(data["bits"], serde_json::json!(bits), "{context}");
            }
            Payload::I64(values) => {
                assert_eq!(data["values"], serde_json::json!(values), "{context}");
            }
        }
    }
}

/// Check, evaluate and build `source`, and require both lanes to produce
/// `expected` for every root.
fn assert_lanes_agree(
    directory: &TempDir,
    label: &str,
    source: &str,
    expected: &[(&str, Payload)],
) {
    let path = write(directory.path(), label, source);
    let report = check_report(&path);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{label}: {report:#}");
    assert_eq!(
        report["errors"],
        serde_json::json!([]),
        "{label}: {report:#}"
    );

    let json = cli(&["eval", "--file", path.to_str().unwrap(), "--json"]);
    assert!(json.status.success(), "{label}: {}", text(&json));
    let report: Value = serde_json::from_slice(&json.stdout).unwrap();
    let roots = report["roots"].as_array().unwrap();
    for (name, payload) in expected {
        let root = roots
            .iter()
            .find(|root| root["name"] == *name)
            .unwrap_or_else(|| panic!("{label}: no root {name}: {report:#}"));
        let tensor = &root["value"]["value"];
        assert_eq!(
            tensor["shape"],
            serde_json::json!([payload.len()]),
            "{label} {name}"
        );
        payload.assert_eval(&tensor["data"], &format!("{label} {name}"));
    }

    let native_directory = directory.path().join(format!("{label}-native"));
    let built = build(&path, &native_directory);
    assert!(built.status.success(), "{label}: {}", text(&built));
    let native = Command::new(native_directory.join(label)).output().unwrap();
    assert!(native.status.success(), "{label}: {}", text(&native));
    let expected_stdout: String = expected
        .iter()
        .map(|(name, payload)| {
            format!(
                "{name} = tensor(shape=[{}], data=[{}])\n",
                payload.len(),
                payload.native()
            )
        })
        .collect();
    assert_eq!(
        String::from_utf8(native.stdout).unwrap(),
        expected_stdout,
        "{label}: the native payload must equal the evaluator's"
    );
}

/// The reported program, verbatim but for a second root whose gathered
/// extents make the leading axis 2 rather than 1.
#[test]
fn the_reported_program_builds_and_agrees_with_eval_at_two_shapes() {
    let source = "module Repro\n\
sig forward[n]: tensor[n, f32] -> tensor[3, i64] -> tensor[1, i64]\n\
def forward(data: tensor[n, f32], target: tensor[3, i64]) = {\n\
  scores = {\n\
    k0 = tensor_to_scalar(reshape(gather(target, to_tensor([0], i64), cast(0, i32)), []))\n\
    k1 = tensor_to_scalar(reshape(gather(target, to_tensor([1], i64), cast(0, i32)), []))\n\
    k2 = tensor_to_scalar(reshape(gather(target, to_tensor([2], i64), cast(0, i32)), []))\n\
    reshape(data, [k0, k1, k2])\n\
  }\n\
  chosen = {\n\
    input = scores\n\
    if eq(shape(input, cast(0, i32)), cast(0, i64)) then input else exp(input)\n\
  }\n\
  extent = shape(scores, cast(0, i32))\n\
  to_tensor([extent])\n\
}\n\
out = forward(to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], f32), to_tensor([1, 2, 3], i64))\n\
wide = forward(to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], f32), to_tensor([2, 2, 2], i64))\n";
    let directory = tempfile::tempdir().unwrap();
    assert_lanes_agree(
        &directory,
        "issue_4111",
        source,
        &[
            ("out", Payload::I64(vec![1])),
            ("wide", Payload::I64(vec![2])),
        ],
    );
}

/// How `scores` is produced: the reported block with gathered rank-three
/// extents, the same block at rank two, the same reshape bound directly, a
/// block reshaping to static extents, and a block producing a copy scalar.
#[derive(Clone, Copy)]
enum Producer {
    BlockRuntimeRank3,
    BlockRuntimeRank2,
    LetRuntimeRank3,
    BlockStatic,
    BlockScalar,
}

/// Where the alias of `scores` is returned: either `if` arm, an option
/// `match` arm, or a plain block's result.
#[derive(Clone, Copy)]
enum Consumer {
    IfThen,
    IfElse,
    Match,
    Block,
}

const FIRST: &str =
    "tensor_to_scalar(reshape(gather(data, to_tensor([0], i64), cast(0, i32)), []))";

fn extent(index: usize) -> String {
    format!(
        "k{index} = tensor_to_scalar(reshape(gather(target, to_tensor([{index}], i64), cast(0, i32)), []))"
    )
}

fn source(producer: Producer, consumer: Consumer) -> String {
    let padding = ["p0 = 0i64", "p1 = 1i64", "p2 = 2i64"].map(String::from);
    let (signature, value, inner, block, result) = match producer {
        Producer::BlockRuntimeRank3 | Producer::LetRuntimeRank3 => (
            "sig forward[n]: tensor[n, f32] -> tensor[3, i64] -> tensor[n, f32]",
            "reshape(data, [k0, k1, k2])".to_string(),
            (0..3).map(extent).collect::<Vec<_>>(),
            matches!(producer, Producer::BlockRuntimeRank3),
            "reshape(add(chosen, scores), [shape(data, cast(0, i32))])",
        ),
        Producer::BlockRuntimeRank2 => (
            "sig forward[n]: tensor[n, f32] -> tensor[3, i64] -> tensor[n, f32]",
            "reshape(data, [k0, k1])".to_string(),
            (0..2).map(extent).collect(),
            true,
            "reshape(add(chosen, scores), [shape(data, cast(0, i32))])",
        ),
        Producer::BlockStatic => (
            "sig forward: tensor[6, f32] -> tensor[3, i64] -> tensor[6, f32]",
            "reshape(data, [2i64, 3i64, 1i64])".to_string(),
            padding.to_vec(),
            true,
            "reshape(add(chosen, scores), [6i64])",
        ),
        Producer::BlockScalar => (
            "sig forward[n]: tensor[n, f32] -> tensor[3, i64] -> tensor[1, f32]",
            FIRST.to_string(),
            padding.to_vec(),
            true,
            "to_tensor([add(chosen, scores)])",
        ),
    };
    let data = match producer {
        Producer::BlockStatic => "data: tensor[6, f32]",
        _ => "data: tensor[n, f32]",
    };
    let producer = if block {
        let lines: String = inner.iter().map(|line| format!("    {line}\n")).collect();
        format!("  scores = {{\n{lines}    {value}\n  }}\n")
    } else {
        let lines: String = inner.iter().map(|line| format!("  {line}\n")).collect();
        format!("{lines}  scores = {value}\n")
    };
    let fresh = "mul(input, input)";
    let consumer = match consumer {
        Consumer::IfThen => format!("if flag then input else {fresh}"),
        Consumer::IfElse => format!("if not(flag) then {fresh} else input"),
        Consumer::Match => format!(
            "pick: Option[bool] = if flag then Some(true) else None\n    \
             match pick with {{\n      \
             | None => {fresh}\n      \
             | Some(taken) => if taken then input else {fresh}\n    \
             }}"
        ),
        Consumer::Block => "input".to_string(),
    };
    format!(
        "module Probe\n\
         {signature}\n\
         def forward({data}, target: tensor[3, i64]) = {{\n\
         \x20 flag = gt({FIRST}, 1.5)\n\
         {producer}\
         \x20 chosen = {{\n\
         \x20   input = scores\n\
         \x20   {consumer}\n\
         \x20 }}\n\
         \x20 {result}\n\
         }}\n"
    )
}

/// Two roots: `low` starts at 1.0, so `flag` is false; `high` starts at 2.0,
/// so `flag` is true. The runtime producers also differ in length and in the
/// gathered extents.
fn roots(producer: Producer) -> String {
    let (low, high) = match producer {
        Producer::BlockStatic => (
            "to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], f32), to_tensor([2, 3, 1], i64)",
            "to_tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], f32), to_tensor([2, 3, 1], i64)",
        ),
        Producer::BlockRuntimeRank2 => (
            "to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], f32), to_tensor([2, 3, 0], i64)",
            "to_tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], f32), to_tensor([4, 2, 0], i64)",
        ),
        _ => (
            "to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], f32), to_tensor([2, 3, 1], i64)",
            "to_tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], f32), to_tensor([1, 4, 2], i64)",
        ),
    };
    format!("low = forward({low})\nhigh = forward({high})\n")
}

/// The expected root payload, computed here rather than read from either
/// lane: the alias path returns `scores`, so the result is `2x`; the fresh
/// path returns `x * x`, so the result is `x * x + x`.
fn expected(producer: Producer, consumer: Consumer, high: bool) -> Payload {
    let data: Vec<f32> = match (producer, high) {
        (_, false) => (1..=6).map(|value| value as f32).collect(),
        (Producer::BlockStatic, true) => (2..=7).map(|value| value as f32).collect(),
        (_, true) => (2..=9).map(|value| value as f32).collect(),
    };
    let flag = data[0] > 1.5;
    let alias = match consumer {
        // `if_else` tests `not(flag)` and returns the alias on its else arm.
        Consumer::IfThen | Consumer::IfElse | Consumer::Match => flag,
        Consumer::Block => true,
    };
    let elements = match producer {
        Producer::BlockScalar => vec![data[0]],
        _ => data,
    };
    Payload::F32(
        elements
            .into_iter()
            .map(|x| if alias { x + x } else { x * x + x })
            .collect(),
    )
}

/// Every producer against every alias consumer, each with a later read of
/// `scores` that the result depends on, through both lanes at two inputs.
#[test]
fn every_alias_consumer_leaves_the_binding_live_for_its_later_read() {
    let directory = tempfile::tempdir().unwrap();
    let producers = [
        ("block_rank3", Producer::BlockRuntimeRank3),
        ("block_rank2", Producer::BlockRuntimeRank2),
        ("let_rank3", Producer::LetRuntimeRank3),
        ("block_static", Producer::BlockStatic),
        ("block_scalar", Producer::BlockScalar),
    ];
    let consumers = [
        ("if_then", Consumer::IfThen),
        ("if_else", Consumer::IfElse),
        ("match", Consumer::Match),
        ("block", Consumer::Block),
    ];
    for (producer_name, producer) in producers {
        for (consumer_name, consumer) in consumers {
            let label = format!("{producer_name}_{consumer_name}");
            let program = format!("{}{}", source(producer, consumer), roots(producer));
            assert_lanes_agree(
                &directory,
                &label,
                &program,
                &[
                    ("low", expected(producer, consumer, false)),
                    ("high", expected(producer, consumer, true)),
                ],
            );
        }
    }
}

/// Negative control: a `drop` leaves nothing a copy could repair (spec/04
/// §8.3), so a read of `scores` after a block drops its alias stays rejected
/// at check and at build, for a block result and for a direct binding.
#[test]
fn a_read_after_the_alias_is_dropped_is_still_rejected() {
    let directory = tempfile::tempdir().unwrap();
    for (label, block) in [("drop_block", true), ("drop_let", false)] {
        let producer = if block {
            "  scores = {\n    p0 = 0i64\n    p1 = 1i64\n    p2 = 2i64\n    exp(data)\n  }\n"
        } else {
            "  scores = exp(data)\n"
        };
        let source = format!(
            "module Probe\n\
             sig forward[n]: tensor[n, f32] -> tensor[1, i64]\n\
             def forward(data: tensor[n, f32]) = {{\n\
             {producer}\
             \x20 released = {{\n\
             \x20   input = scores\n\
             \x20   drop(input)\n\
             \x20 }}\n\
             \x20 extent = shape(scores, cast(0, i32))\n\
             \x20 to_tensor([extent])\n\
             }}\n\
             out = forward(to_tensor([1.0, 2.0], f32))\n"
        );
        let path = write(directory.path(), label, &source);
        let report = check_report(&path);
        assert!(
            report["score"].as_f64().is_some_and(|score| score < 1.0),
            "{label}: {report:#}"
        );
        let messages: Vec<&str> = report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|error| error["message"].as_str())
            .collect();
        assert!(
            messages
                .iter()
                .any(|message| message.contains("was already consumed by call to `drop`")),
            "{label}: {messages:?}"
        );
        let built = build(&path, &directory.path().join(format!("{label}-native")));
        assert!(!built.status.success(), "{label} must not build");
    }
}
