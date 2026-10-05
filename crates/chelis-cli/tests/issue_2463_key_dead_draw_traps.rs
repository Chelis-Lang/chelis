//! chelis#2463 in key form: a draw whose result is discarded still traps.
//!
//! spec/06 §5.2 lets dead-code elimination remove a potentially trapping
//! node only when the trap still occurs, and [05-OP-37]/[05-OP-8] trap
//! `Domain` on an invalid rate before any element is drawn. #2463's four
//! witnesses were written against the retired `with seed` handler; here each
//! is a keyed draw whose result nothing reads. Every witness must trap with
//! its typed trap text in `chelis eval` and in compiled C; the library
//! witness also traps evaluated inside its reef package, the
//! `compile_reef_context` + `eval_in_context` lane (spec/03 §4.4: a
//! binding's initializer is evaluated whether or not the binding is read).

use assert_cmd::Command;
use chelis_compiler_api::{compile_reef_context, eval_in_context};
use std::path::Path;

const DROPOUT_DOMAIN: &str = "numeric trap: domain in dropout at f32";
const CAST_OVERFLOW: &str = "numeric trap: overflow in cast at i8";

/// Witness 1: a selected definition whose only draw is dead and has an
/// invalid rate.
const DEAD_INVALID_DRAW: &str = "def selected(k: key, x: tensor[4, f32]) -> tensor[4, f32] = {
  dead = dropout(k, x, 1.0f32)
  x
}
def main() -> tensor[4, f32] = selected(key_from_seed(7i64), to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";

/// Witness 3: a value declaration holding an invalid draw, referenced dead
/// from a function that is reached only through `grad`.
const DEAD_VALUE_THROUGH_GRAD: &str =
    "sampled = dropout(key_from_seed(9i64), to_tensor([1.0f32, 1.0f32]), 1.0f32)
def f(v: tensor[4, f32]) -> f32 = {
  dead = sampled
  tensor_to_scalar(sum(v, 0i32))
}
def main() -> tensor[4, f32] = grad(f)(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))
";

/// Witness 4 (K2): the trap is in the data input of a discarded `dropout`.
const TRAP_IN_DEAD_DROPOUT_INPUT: &str = "def selected(k: key, x: tensor[4, f32]) -> tensor[4, f32] = {
  dead = dropout(k, cast(cast(add(x, to_tensor([1000.0f32, 1000.0f32, 1000.0f32, 1000.0f32])), i8), f32), 0.5f32)
  x
}
def main() -> tensor[4, f32] = selected(key_from_seed(7i64), to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";

/// K3: the trap is in the template of a discarded `uniform_like`.
const TRAP_IN_DEAD_UNIFORM_TEMPLATE: &str = "def selected(k: key, x: tensor[4, f32]) -> tensor[4, f32] = {
  dead = uniform_like(k, cast(cast(add(x, to_tensor([1000.0f32, 1000.0f32, 1000.0f32, 1000.0f32])), i8), f32), 0.0f32, 1.0f32)
  x
}
def main() -> tensor[4, f32] = selected(key_from_seed(7i64), to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";

/// Witness 2: the library module whose exported value declaration is an
/// invalid draw.
const LIBRARY_DRAW: &str = "module Drawlib.Draw
export (sampled)
sampled = dropout(key_from_seed(1i64), to_tensor([1.0f32, 1.0f32]), 1.0f32)
";

/// Witness 2: the client, whose only reference to the library value is dead.
const LIBRARY_CLIENT: &str = "module App.Entry
import Drawlib.Draw (sampled)
def main() -> tensor[2, f32] = {
  dead = sampled
  to_tensor([1.0f32, 1.0f32])
}
";

fn chelis(directory: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// The file is canonical and lint-clean, so the style gate cannot be what
/// rejects it.
fn assert_canonical(directory: &Path, path: &str, source: &str) {
    let formatted = chelis(directory, &["fmt", "--inplace", path]);
    assert!(formatted.status.success(), "{}", text(&formatted));
    assert_eq!(
        std::fs::read_to_string(directory.join(path)).unwrap(),
        source,
        "the witness must already be canonical Surf"
    );
    let linted = chelis(directory, &["lint", "--check", path]);
    assert!(linted.status.success(), "{}", text(&linted));
}

fn assert_eval_traps(directory: &Path, path: &str, trap: &str) {
    let output = chelis(directory, &["eval", "--file", path, "--json"]);
    let output_text = text(&output);
    assert!(
        !output.status.success(),
        "eval did not trap:\n{output_text}"
    );
    assert!(output_text.contains(trap), "{output_text}");
}

fn assert_c_traps(directory: &Path, path: &str, stem: &str, trap: &str) {
    let out = directory.join(format!("{stem}-out"));
    let built = chelis(
        directory,
        &[
            "build",
            path,
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ],
    );
    assert!(built.status.success(), "{}", text(&built));
    let run = std::process::Command::new(out.join(stem)).output().unwrap();
    let run_text = text(&run);
    assert!(
        !run.status.success(),
        "compiled C did not trap:\n{run_text}"
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains(trap),
        "{run_text}"
    );
}

fn assert_single_file_witness_traps(stem: &str, source: &str, trap: &str) {
    let directory = tempfile::tempdir().unwrap();
    let path = format!("{stem}.ch");
    std::fs::write(directory.path().join(&path), source).unwrap();
    assert_canonical(directory.path(), &path, source);
    assert_eval_traps(directory.path(), &path, trap);
    assert_c_traps(directory.path(), &path, stem, trap);
}

#[test]
fn a_dead_invalid_draw_in_a_selected_def_traps_in_eval_and_c() {
    assert_single_file_witness_traps("dead_draw", DEAD_INVALID_DRAW, DROPOUT_DOMAIN);
}

#[test]
fn a_dead_value_declaration_reached_only_through_grad_traps_in_eval_and_c() {
    assert_single_file_witness_traps("through_grad", DEAD_VALUE_THROUGH_GRAD, DROPOUT_DOMAIN);
}

#[test]
fn a_trap_in_a_discarded_dropout_input_traps_in_eval_and_c() {
    assert_single_file_witness_traps("dropout_input", TRAP_IN_DEAD_DROPOUT_INPUT, CAST_OVERFLOW);
}

#[test]
fn a_trap_in_a_discarded_uniform_template_traps_in_eval_and_c() {
    assert_single_file_witness_traps(
        "uniform_template",
        TRAP_IN_DEAD_UNIFORM_TEMPLATE,
        CAST_OVERFLOW,
    );
}

fn write(path: &Path, contents: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

/// Evidentiary status: REGRESSION TEST for the `eval_in_context` assertion,
/// which returned `[1, 1]` at 727e74b41; the C assertion is a disposition
/// lock.
#[test]
fn a_dead_reference_to_a_library_draw_value_traps_in_context_and_c() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("app");
    let compiler = env!("CARGO_PKG_VERSION");
    write(
        &root.join("reef.toml"),
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\ncompiler = \"={compiler}\"\n\
             module_prefix = \"App\"\n\n[dependencies]\ndrawlib = {{ path = \"./drawlib\" }}\n"
        ),
    );
    write(
        &root.join("drawlib/reef.toml"),
        &format!(
            "[package]\nname = \"drawlib\"\nversion = \"0.1.0\"\ncompiler = \"={compiler}\"\n\
             module_prefix = \"Drawlib\"\n"
        ),
    );
    write(
        &root.join("reef.lock"),
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\n\n[[dependencies]]\n\
             name = \"drawlib\"\nversion = \"0.1.0\"\ncompiler = \"={compiler}\"\n\
             archive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\n\
             kind = \"path\"\npath = \"./drawlib\"\n"
        ),
    );
    write(&root.join("drawlib/src/draw.ch"), LIBRARY_DRAW);
    write(&root.join("src/entry.ch"), LIBRARY_CLIENT);
    assert_canonical(&root, "drawlib/src/draw.ch", LIBRARY_DRAW);
    assert_canonical(&root, "src/entry.ch", LIBRARY_CLIENT);
    let context =
        compile_reef_context(Path::new(""), &root, &chelis_std_bundle::EMBEDDED_RUNTIME).unwrap();
    let error = eval_in_context(&context, LIBRARY_CLIENT)
        .map(|result| format!("{:?}", result.roots))
        .expect_err("eval_in_context did not trap");
    assert!(
        error
            .errors
            .iter()
            .any(|error| error.message == DROPOUT_DOMAIN),
        "{error:?}"
    );
    assert_c_traps(&root, "src/entry.ch", "entry", DROPOUT_DOMAIN);
}
