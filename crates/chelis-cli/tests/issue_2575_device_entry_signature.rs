//! chelis#2575: a device target (`--target hip`, `--target metal`) must not
//! drop a definition whose parameter its single DAG entry cannot carry.
//!
//! chelis#2522 made the C lane give such a definition (a string, data-type,
//! container, tuple or unit parameter) its own host function. HIP and Metal
//! selected their DAG entry instead, emitting a zero-input `<stem>` entry and
//! dropping `f` with its authored signature. They now take their host
//! backend, which keeps `f` as C does.
//!
//! Negative parity: a parameter the DAG entry does carry (a tensor, or a
//! numeric scalar as a rank-0 input) leaves the program on the DAG entry.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

use common::{authored_c_symbol, write_file};

const STEM: &str = "entry";

struct Built {
    _dir: TempDir,
    out_dir: PathBuf,
}

fn build(target: &str, source: &str) -> Built {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{STEM}.ch"));
    let out_dir = dir.path().join(format!("{target}-out"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            target,
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build runs");
    assert!(
        output.status.success(),
        "`--target {target}` rejected:\n{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Built { _dir: dir, out_dir }
}

fn read(out_dir: &Path, file: &str) -> String {
    fs::read_to_string(out_dir.join(file))
        .unwrap_or_else(|error| panic!("missing emitted `{file}`: {error}"))
}

/// The program's own header declarations, without the artifact envelope.
fn header_declarations(target: &str, built: &Built) -> Vec<String> {
    read(&built.out_dir, &format!("{STEM}_{target}.h"))
        .lines()
        .filter(|line| !line.starts_with("/*") && !line.starts_with('#') && !line.is_empty())
        .map(str::to_string)
        .collect()
}

fn source_file(target: &str) -> String {
    match target {
        "hip" => format!("{STEM}_hip.cpp"),
        "metal" => format!("{STEM}_metal.mm"),
        other => panic!("not a device target: {other}"),
    }
}

/// The emitted host source compiles as an object: C++ for HIP, which with no
/// device helper includes no HIP header, and Objective-C++ for Metal, which
/// needs Apple clang.
fn compiles_as_object(target: &str, built: &Built) {
    let compiler = match target {
        "hip" => std::env::var("CXX").unwrap_or_else(|_| "c++".to_string()),
        "metal" if cfg!(target_os = "macos") => "clang++".to_string(),
        _ => return,
    };
    let compiled = StdCommand::new(&compiler)
        .current_dir(&built.out_dir)
        .args(["-O1", "-c", &source_file(target), "-o", "entry.o"])
        .output()
        .unwrap_or_else(|error| panic!("`{compiler}` must run: {error}"));
    assert!(
        compiled.status.success(),
        "the {target} host source does not compile:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

fn assert_keeps_authored_signature(target: &str, parameter: &str, c_parameter: &str) {
    let source = format!("def f({parameter}) -> tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])\n");
    let built = build(target, &source);
    let declarations = header_declarations(target, &built);
    let expected = format!("chelis_tensor* {}({c_parameter});", authored_c_symbol("f"));
    assert_eq!(
        declarations,
        [expected],
        "`--target {target}` must keep `f({parameter})` as its authored entry"
    );
    compiles_as_object(target, &built);
}

fn assert_stays_a_dag_entry(target: &str, source: &str) {
    let built = build(target, source);
    let declarations = header_declarations(target, &built);
    assert!(
        declarations.first().is_some_and(|line| line
            == &format!(
                "extern \"C\" void {STEM}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
            )),
        "`--target {target}` must keep this program on its DAG entry: {declarations:?}"
    );
}

const PARAMETERS: [(&str, &str); 5] = [
    ("s: string", "chelis_string s"),
    ("xs: List[i64]", "chelis_list* xs"),
    ("p: (i64, i64)", "chelis_tuple* p"),
    ("d: Dict[string, i64]", "chelis_dict* d"),
    ("u: unit", "int u"),
];

#[test]
fn hip_keeps_every_definition_whose_parameter_the_dag_entry_cannot_carry() {
    for (parameter, c_parameter) in PARAMETERS {
        assert_keeps_authored_signature("hip", parameter, c_parameter);
    }
}

#[test]
fn metal_keeps_every_definition_whose_parameter_the_dag_entry_cannot_carry() {
    for (parameter, c_parameter) in PARAMETERS {
        assert_keeps_authored_signature("metal", parameter, c_parameter);
    }
}

/// A data-type parameter, the class the issue names beside strings.
#[test]
fn a_data_type_parameter_keeps_its_definition_on_both_device_targets() {
    let source = "type Pair =\n  | Pair { a: i64, b: i64 }\n\
                  def f(p: Pair) -> tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])\n";
    for target in ["hip", "metal"] {
        let built = build(target, source);
        let declarations = header_declarations(target, &built);
        let symbol = authored_c_symbol("f");
        assert!(
            declarations
                .iter()
                .any(|line| line.starts_with("chelis_tensor* ")
                    && line.contains(&format!("{symbol}("))),
            "`--target {target}` must keep `f(p: Pair)`: {declarations:?}"
        );
        compiles_as_object(target, &built);
    }
}

/// The dropped parameter beside a carried tensor parameter: the DAG entry
/// would keep `x` and silently lose `s`. The emitted source is not compiled
/// here: its C tensor helper spells `restrict`, which C++ rejects (chelis#2595).
#[test]
fn a_string_parameter_beside_a_tensor_parameter_keeps_both() {
    let source = "def f(s: string, x: tensor[3, f32]) -> tensor[3, f32] = (x + x)\n";
    for target in ["hip", "metal"] {
        let built = build(target, source);
        let declarations = header_declarations(target, &built);
        let expected = format!(
            "chelis_tensor* {}(chelis_string s, chelis_tensor* x);",
            authored_c_symbol("f")
        );
        assert_eq!(declarations, [expected], "`--target {target}`");
    }
}

/// Negative parity: a numeric scalar is a rank-0 DAG input, and an unused
/// one is dropped from the entry exactly as an unused tensor is.
#[test]
fn hip_keeps_carried_parameters_on_its_dag_entry() {
    assert_stays_a_dag_entry(
        "hip",
        "def f(n: i64) -> tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])\n",
    );
    assert_stays_a_dag_entry(
        "hip",
        "def f(x: tensor[3, f32]) -> tensor[3, f32] = (x + x)\n",
    );
}

#[test]
fn metal_keeps_a_tensor_parameter_on_its_dag_entry() {
    assert_stays_a_dag_entry(
        "metal",
        "def f(x: tensor[3, f32]) -> tensor[3, f32] = (x + x)\n",
    );
}
