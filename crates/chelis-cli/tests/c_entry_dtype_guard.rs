//! chelis#2490, [04-NUM-11]: a generated C public entry compares each supplied
//! tensor's dtype tag with the declared parameter dtype before it reads any
//! element. A mismatch traps `Domain` in `load` at the declared dtype under
//! [04-NUM-9], after a context line naming the input and both dtypes. It is
//! never reinterpreted: before this guard every mismatched row below ran to
//! completion, reading the supplied storage at the declared dtype.
//!
//! One program carries every host entry kind that takes a tensor: a tensor
//! result, a tuple result, a record result, and a tensor nested in a tuple
//! parameter. The direct four-argument DAG entry is covered beside the other
//! direct-entry guards in `chelis-backend-c/tests/exec_compile.rs`.

mod common;

use assert_cmd::Command;
use common::{authored_c_symbol, gcc_available, link_generated};
use std::fs;
use std::path::Path;

const SOURCE: &str = "def ints(x: tensor[3, i64]) -> tensor[3, i64] = x + x\n\
def floats(x: tensor[3, f64]) -> tensor[3, f64] = x + x\n\
def singles(x: tensor[3, f32]) -> tensor[3, f32] = x + x\n\
def int_pair(x: tensor[3, i64]) -> (tensor[3, i64], tensor[3, i64]) = (x + x, x * x)\n\
type IntRecord =\n  | IntRecord { twice: tensor[3, i64], square: tensor[3, i64] }\n\
def int_record(x: tensor[3, i64]) -> IntRecord = IntRecord { twice: x + x, square: x * x }\n\
def nested(p: (tensor[3, i64], tensor[3, i64])) -> tensor[3, i64] = p.0 + p.1\n";

/// How the harness hands one supplied tensor to one entry and releases the
/// result. The C local `x` is the supplied tensor.
#[derive(Clone, Copy)]
enum Entry {
    Tensor(&'static str),
    Tuple(&'static str),
    Record(&'static str),
    NestedInTuple(&'static str),
}

impl Entry {
    fn call(self) -> String {
        match self {
            Entry::Tensor(name) => format!(
                "chelis_tensor *result = {}(x);\nchelis_tensor_release(result);",
                authored_c_symbol(name)
            ),
            Entry::Tuple(name) => format!(
                "chelis_tuple *result = {}(x);\nchelis_tuple_release(result);",
                authored_c_symbol(name)
            ),
            Entry::Record(name) => format!(
                "chelis_adt *result = {}(x);\nchelis_adt_release(result);",
                authored_c_symbol(name)
            ),
            Entry::NestedInTuple(name) => format!(
                "chelis_tensor_retain(x); chelis_tensor_retain(x);\n\
                 chelis_value items[2] = {{ chelis_value_take_tensor(x), chelis_value_take_tensor(x) }};\n\
                 chelis_tuple *p = chelis_tuple_from_values(items, 2);\n\
                 chelis_value_release(items[0]); chelis_value_release(items[1]);\n\
                 chelis_tensor *result = {}(p);\n\
                 chelis_tensor_release(result); chelis_tuple_release(p);",
                authored_c_symbol(name)
            ),
        }
    }
}

fn build(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("entry.ch");
    fs::write(&path, SOURCE).expect("source");
    let out = dir.join("c");
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--allow-style-violations"])
        .arg(&path)
        .args(["--target", "c", "-o"])
        .arg(&out)
        .output()
        .expect("build C");
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    out
}

/// Link a harness that supplies a zero-filled rank-1 tensor of dtype
/// `CHELIS_DTYPE_{supplied}` to `entry`, and return (exit success, stdout plus
/// stderr).
fn run(out: &Path, name: &str, entry: Entry, supplied: &str) -> (bool, String) {
    let harness = format!(
        "#define main generated_main\n#include \"entry.c\"\n#undef main\n\
         int main(void) {{\n\
         chelis_tensor *x = chelis_alloc(1, (int64_t[]){{3}}, CHELIS_DTYPE_{supplied});\n\
         {}\n\
         chelis_tensor_release(x);\n\
         puts(\"completed\"); return 0;\n}}\n",
        entry.call()
    );
    let file = format!("{name}.c");
    fs::write(out.join(&file), harness).expect("harness");
    assert!(link_generated(out, &file, name).success(), "{name} link");
    let run = std::process::Command::new(out.join(name))
        .output()
        .expect("run");
    (
        run.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        ),
    )
}

#[test]
fn mismatched_dtypes_trap_at_every_host_entry_kind() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = build(dir.path());
    // Every row runs before any assertion, so a failure names each entry kind
    // the guard misses rather than only the first.
    let mut failures = Vec::new();
    for (name, entry, supplied, declared) in [
        ("f64_for_i64", Entry::Tensor("ints"), "F64", "i64"),
        ("i32_for_i64", Entry::Tensor("ints"), "I32", "i64"),
        ("i64_for_f64", Entry::Tensor("floats"), "I64", "f64"),
        ("bool_for_f32", Entry::Tensor("singles"), "BOOL", "f32"),
        ("f16_for_f32", Entry::Tensor("singles"), "F16", "f32"),
        ("tuple", Entry::Tuple("int_pair"), "F64", "i64"),
        ("record", Entry::Record("int_record"), "F64", "i64"),
        ("nested", Entry::NestedInTuple("nested"), "F64", "i64"),
    ] {
        // Each supplied tag's language spelling is its lowercase name.
        let actual = supplied.to_lowercase();
        let (succeeded, output) = run(&out, name, entry, supplied);
        let names_parameter =
            matches!(entry, Entry::NestedInTuple(_)) || output.contains("input `x` expected dtype");
        if succeeded
            || output.contains("completed")
            || !output.contains(&format!("expected dtype {declared}, got {actual}"))
            || !output.contains(&format!("numeric trap: domain in load at {declared}\n"))
            || !names_parameter
        {
            failures.push(format!("{name}: exit success {succeeded}: {output}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn matching_dtypes_complete_at_every_host_entry_kind() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = build(dir.path());
    for (name, entry, supplied) in [
        ("i64_ints", Entry::Tensor("ints"), "I64"),
        ("f64_floats", Entry::Tensor("floats"), "F64"),
        ("f32_singles", Entry::Tensor("singles"), "F32"),
        ("i64_pair", Entry::Tuple("int_pair"), "I64"),
        ("i64_record", Entry::Record("int_record"), "I64"),
        ("i64_nested", Entry::NestedInTuple("nested"), "I64"),
    ] {
        let (succeeded, output) = run(&out, name, entry, supplied);
        assert!(succeeded, "{name}: {output}");
        assert!(output.contains("completed"), "{name}: {output}");
        assert!(!output.contains("numeric trap"), "{name}: {output}");
    }
}
