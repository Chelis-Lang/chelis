//! chelis#4105, [05-OP-15]: `argmax_reduce` and `argmin_reduce` return i64
//! indices with the reduced axis removed, through every lane.
//!
//! The reported witness composed a computed input reshape, an i32 cast and a
//! conditional index reduction followed by a result reshape. `chelis check`
//! and `chelis eval` agreed on i64 `[1, 2]`, while the compiled lane stored
//! the indices as i32 and then copied the i64 destination byte count from
//! that buffer, so two rows read back as `[8589934593, 0]`.
//!
//! Each program here is executed three ways from one source: `chelis eval
//! --json` (dtype and payload), the native executable built from the same
//! file (payload), and the same symbolic function compiled once as a static
//! library and called from C at several runtime shapes (dtype tag, rank,
//! count and payload). Batch one, batch two, a wider batch, a one-column
//! sequence and, for the conditional forms, an empty batch all run through
//! that one compiled function.

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

/// One runtime input: binding name, `[batch, sequence]`, row-major i32 data,
/// and the expected argmax and argmin indices.
struct Case {
    name: &'static str,
    shape: [i64; 2],
    data: &'static [i32],
    argmax: &'static [i64],
    argmin: &'static [i64],
}

const CASES: [Case; 5] = [
    Case {
        name: "one",
        shape: [1, 3],
        data: &[1, 9, 2],
        argmax: &[1],
        argmin: &[0],
    },
    Case {
        name: "two",
        shape: [2, 3],
        data: &[1, 9, 2, 2, 1, 9],
        argmax: &[1, 2],
        argmin: &[0, 1],
    },
    // Distinct indices in every row, and a first-index tie in the last.
    Case {
        name: "three",
        shape: [3, 4],
        data: &[4, 3, 2, 1, 1, 2, 3, 4, 2, 7, 7, 0],
        argmax: &[0, 3, 1],
        argmin: &[3, 0, 3],
    },
    Case {
        name: "column",
        shape: [2, 1],
        data: &[5, 6],
        argmax: &[0, 0],
        argmin: &[0, 0],
    },
    Case {
        name: "zero",
        shape: [0, 3],
        data: &[],
        argmax: &[],
        argmin: &[],
    },
];

/// The cases a form executes. An unconditional reduction over an empty batch
/// is excluded: the compiled lane traps `Domain` there while the evaluator
/// returns the empty result (chelis#4107), independent of index storage. The
/// conditional forms route the empty batch around the reduction, as the
/// reported witness does, and run every case.
fn cases(conditional: bool) -> impl Iterator<Item = &'static Case> {
    CASES
        .iter()
        .filter(move |case| conditional || case.shape[0] != 0)
}

fn expected<'a>(op: &str, case: &'a Case) -> &'a [i64] {
    if op == "argmax_reduce" {
        case.argmax
    } else {
        case.argmin
    }
}

const COMPUTED: &str = "reshape(reshape(x, [floor_div(mul(shape(x, 0), shape(x, 1)), shape(x, 1)), shape(x, 1)]), [shape(x, 0), shape(x, 1)])";

/// The definition under test. `composed` adds the computed input reshape;
/// `conditional` adds the reported empty-batch branch.
fn definition(op: &str, composed: bool, conditional: bool, result: &str, axis: i64) -> String {
    let operand = if composed { COMPUTED } else { "x" };
    let reduced = format!("reshape({op}(y, {axis}), [shape(x, 0)])");
    let body = if conditional {
        format!(
            "if eq(shape(x, 0), 0i64) then reshape((to_tensor([]) : tensor[0, {result}]), [shape(x, 0)]) else {reduced}"
        )
    } else {
        reduced
    };
    format!(
        "module Probe.Index\n\
         sig pool[b, s]: tensor[b, s, i32] -> tensor[b, {result}]\n\
         def pool(x: tensor[b, s, i32]) -> tensor[b, {result}] = {{\n\
         \x20 y = cast({operand}, i32)\n\
         \x20 {body}\n\
         }}\n"
    )
}

fn literal(case: &Case) -> String {
    if case.shape[0] == 0 {
        return format!(
            "reshape((to_tensor([]) : tensor[0, i32]), [0i64, {}i64])",
            case.shape[1]
        );
    }
    let rows = case
        .data
        .chunks(case.shape[1] as usize)
        .map(|row| {
            let row = row.iter().map(i32::to_string).collect::<Vec<_>>();
            format!("[{}]", row.join(", "))
        })
        .collect::<Vec<_>>();
    format!("to_tensor([{}], i32)", rows.join(", "))
}

fn bindings(conditional: bool) -> String {
    cases(conditional)
        .map(|case| format!("out_{} = pool({})\n", case.name, literal(case)))
        .collect()
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

fn assert_checks(path: &Path) {
    let report = check_report(path);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report:#}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report:#}");
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

fn render(values: &[i64]) -> String {
    values
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// `chelis eval --json` and the native executable from the same source.
fn assert_eval_and_executable(
    directory: &TempDir,
    label: &str,
    op: &str,
    conditional: bool,
    source: &str,
) {
    let path = write(directory.path(), label, source);
    assert_checks(&path);

    let json = cli(&["eval", "--file", path.to_str().unwrap(), "--json"]);
    assert!(json.status.success(), "{label}: {}", text(&json));
    let report: Value = serde_json::from_slice(&json.stdout).unwrap();
    let roots = report["roots"].as_array().unwrap();
    for case in cases(conditional) {
        let name = format!("out_{}", case.name);
        let root = roots
            .iter()
            .find(|root| root["name"] == name.as_str())
            .unwrap_or_else(|| panic!("{label}: no root {name}: {report:#}"));
        let tensor = &root["value"]["value"];
        let want = expected(op, case);
        assert_eq!(
            tensor["shape"],
            serde_json::json!([want.len()]),
            "{label} {name}"
        );
        assert_eq!(tensor["data"]["dtype"], "int64", "{label} {name}");
        assert_eq!(
            tensor["data"]["values"],
            serde_json::json!(want),
            "{label} {name}"
        );
    }

    let native_directory = directory.path().join(format!("{label}-native"));
    let built = build(&path, &native_directory);
    assert!(built.status.success(), "{label}: {}", text(&built));
    let native = Command::new(native_directory.join(label)).output().unwrap();
    assert!(native.status.success(), "{label}: {}", text(&native));
    let expected_stdout: String = cases(conditional)
        .map(|case| {
            let want = expected(op, case);
            format!(
                "out_{} = tensor(shape=[{}], data=[{}])\n",
                case.name,
                want.len(),
                render(want)
            )
        })
        .collect();
    assert_eq!(
        String::from_utf8(native.stdout).unwrap(),
        expected_stdout,
        "{label}: native payload must equal the evaluator's"
    );
}

/// One symbolic library function, called from C at every runtime shape.
fn assert_library_storage(
    directory: &TempDir,
    label: &str,
    op: &str,
    conditional: bool,
    source: &str,
) {
    let path = write(directory.path(), label, source);
    assert_checks(&path);
    let out = directory.path().join(format!("{label}-library"));
    let built = build(&path, &out);
    assert!(built.status.success(), "{label}: {}", text(&built));
    assert!(
        String::from_utf8_lossy(&built.stdout).contains("Built static library"),
        "{label}: {}",
        text(&built)
    );
    let symbol: String = "pool".bytes().map(|byte| format!("{byte:02x}")).collect();
    let mut calls = String::new();
    for case in cases(conditional) {
        let data = if case.data.is_empty() {
            "NULL".to_string()
        } else {
            let values = case
                .data
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("(const int32_t[]){{{values}}}")
        };
        calls.push_str(&format!(
            "    run(\"{}\", {}, {}, {data});\n",
            case.name, case.shape[0], case.shape[1]
        ));
    }
    let driver = format!(
        r#"#include <stdint.h>
#include <stdio.h>
#include "chelis_runtime.h"
#include "{label}.h"

static void run(const char *name, int64_t batch, int64_t sequence, const int32_t *data) {{
    int64_t shape[2] = {{batch, sequence}};
    chelis_tensor *x = chelis_tensor_entry_borrow(
        2, shape, CHELIS_DTYPE_I32, data, batch * sequence * (int64_t)sizeof(int32_t));
    chelis_tensor *out = chelis_fn_{symbol}(x);
    chelis_read_view view = chelis_tensor_read_view(out);
    printf("%s dtype=%s rank=%d count=%lld data=[", name,
           view.dtype == CHELIS_DTYPE_I64 ? "i64" : "other",
           (int)chelis_tensor_rank(out), (long long)view.count);
    if (view.dtype == CHELIS_DTYPE_I64) {{
        for (int64_t i = 0; i < view.count; ++i) {{
            printf("%s%lld", i ? ", " : "", (long long)((const int64_t *)view.data)[i]);
        }}
    }}
    printf("]\n");
    chelis_tensor_release(out);
}}

int main(void) {{
{calls}    return 0;
}}
"#
    );
    std::fs::write(out.join("driver.c"), driver).unwrap();
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    )
    .expect("declared C directories");
    let driver = out.join("driver.c");
    let library = out.join(format!("lib{label}.a"));
    let runtime = out.join("libchelis_runtime.a");
    let product = out.join("driver");
    let linked = chelis_backend_c::toolchain::tool_command(&toolchain.compiler)
        .args(chelis_backend_c::toolchain::link_args(
            &toolchain.compile_flags,
            &[driver.as_os_str(), library.as_os_str(), runtime.as_os_str()],
            &toolchain.link_flags,
            product.as_os_str(),
        ))
        .output()
        .unwrap();
    assert!(linked.status.success(), "{label}: {}", text(&linked));
    let run = Command::new(&product).output().unwrap();
    assert!(run.status.success(), "{label}: {}", text(&run));
    let expected_stdout: String = cases(conditional)
        .map(|case| {
            let want = expected(op, case);
            format!(
                "{} dtype=i64 rank=1 count={} data=[{}]\n",
                case.name,
                want.len(),
                render(want)
            )
        })
        .collect();
    assert_eq!(
        String::from_utf8(run.stdout).unwrap(),
        expected_stdout,
        "{label}: one compiled symbolic function must return i64 indices at every shape"
    );
}

fn forms() -> Vec<(&'static str, bool, bool)> {
    vec![
        ("direct", false, false),
        ("direct_conditional", false, true),
        ("composed", true, false),
        ("composed_conditional", true, true),
    ]
}

fn assert_all_forms(op: &str) {
    let directory = tempfile::tempdir().unwrap();
    for (form, composed, conditional) in forms() {
        let definition = definition(op, composed, conditional, "i64", 1);
        let label = format!("{}_{form}", op.trim_end_matches("_reduce"));
        assert_eval_and_executable(
            &directory,
            &format!("{label}_bindings"),
            op,
            conditional,
            &format!("{definition}{}", bindings(conditional)),
        );
        assert_library_storage(
            &directory,
            &format!("{label}_library"),
            op,
            conditional,
            &definition,
        );
    }
}

#[test]
fn argmax_returns_i64_indices_in_every_lane_and_runtime_shape() {
    assert_all_forms("argmax_reduce");
}

#[test]
fn argmin_returns_i64_indices_in_every_lane_and_runtime_shape() {
    assert_all_forms("argmin_reduce");
}

/// Negative controls from the issue: the checker rejects an out-of-range
/// axis and an i32 result declaration for the composed witness, for both
/// index reductions.
#[test]
fn invalid_axis_and_i32_result_declarations_reject_at_check() {
    let directory = tempfile::tempdir().unwrap();
    for op in ["argmax_reduce", "argmin_reduce"] {
        for (label, source, needle) in [
            (
                "axis",
                definition(op, true, true, "i64", 2),
                format!("{op} argument 2, axis 2: expected axis in -2..2 for rank 2 tensor, got 2"),
            ),
            (
                "result",
                definition(op, true, true, "i32", 1),
                "tensor precision mismatch: i32 vs i64".to_string(),
            ),
        ] {
            let name = format!("{}_{label}", op.trim_end_matches("_reduce"));
            let path = write(directory.path(), &name, &source);
            let report = check_report(&path);
            assert!(
                report["score"].as_f64().is_some_and(|score| score < 1.0),
                "{name}: {report:#}"
            );
            let messages: Vec<&str> = report["errors"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|error| error["message"].as_str())
                .collect();
            assert!(
                messages.iter().any(|message| message.contains(&needle)),
                "{name}: expected `{needle}`, got {messages:?}"
            );
            let built = build(&path, &directory.path().join(format!("{name}-native")));
            assert!(!built.status.success(), "{name} must not build");
        }
    }
}
