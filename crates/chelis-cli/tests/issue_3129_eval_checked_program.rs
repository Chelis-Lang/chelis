//! chelis#3129: `chelis eval --file X.ch` evaluates the program it checked.
//!
//! The single-file eval path type-checked the parsed declarations, then
//! printed them back to Surf and evaluated a second parse of that text. A
//! printer defect (chelis#3128) therefore changed the answer: the checked
//! program was the 11.0 one and the evaluated program was the 1.0 one.
//!
//! Three controls. The value control pins the #3128 witness. The parity
//! control requires `eval --file X.ch` to equal `deep X.ch` followed by
//! `eval --file X.dp` over a small corpus, one program of which links the
//! bundled chelis-std. The structural control does not depend on any printer
//! defect: no function in the CLI or the prove crate both calls the Surf
//! printer and hands a program to an evaluator entry point, so printer output
//! can never again be the program a caller runs.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use syn::visit::Visit;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// The #3128 witness: the Surf printer drops the parentheses around the `if`,
/// so its re-parse pipes only the else-branch.
const PIPE_OF_IF: &str = "\
def inner(x: f32) -> f32 = x + 10.0

def main() -> f32 = (if true then 1.0 else 3.0) |> inner
";

const MATCH_AND_BLOCK: &str = "\
type Shape =
  | Circle(f64)
  | Square(f64)

def area(s: Shape) -> f64 = match s with {
  | Circle(r) => 3.0f64 * r * r
  | Square(w) => w * w
}

def main() -> f64 = {
  a = area(Circle(2.0f64))
  b = area(Square(3.0f64))
  a + b
}
";

/// Imports, so the single-file program is linked against chelis-std.
const LINKED_STD: &str = "\
import Std.Scalar (abs)

def gap(x: f32, y: f32) -> f32 = abs(x - y)

def main() -> f32 = gap(2.5, 7.0) |> gap(1.0)
";

const TENSOR_AND_VALUE_ROOT: &str = "\
def scale(t: tensor[3, f32], k: f32) -> tensor[3, f32] = t * expand(to_tensor([k]), 0i32, 3i64)

def main() -> tensor[3, f32] = scale(to_tensor([1.0, 2.0, 3.0], f32), 2.0)

offset = 4i64 + 5i64
";

fn chelis(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .expect("chelis runs")
}

fn success_stdout(output: std::process::Output, what: &str) -> String {
    assert!(
        output.status.success(),
        "{what} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

fn eval_surf(dir: &Path, name: &str, source: &str) -> String {
    let path = dir.join(format!("{name}.ch"));
    write_file(&path, source);
    success_stdout(
        chelis(&["eval", "--file", path.to_str().unwrap()]),
        &format!("eval --file {name}.ch"),
    )
}

fn eval_through_deep(dir: &Path, name: &str) -> String {
    let surf = dir.join(format!("{name}.ch"));
    let deep = dir.join(format!("{name}.dp"));
    let printed = success_stdout(
        chelis(&["deep", surf.to_str().unwrap()]),
        &format!("deep {name}.ch"),
    );
    write_file(&deep, &printed);
    success_stdout(
        chelis(&["eval", "--file", deep.to_str().unwrap()]),
        &format!("eval --file {name}.dp"),
    )
}

#[test]
fn eval_file_runs_the_checked_pipe_of_if() {
    let dir = tempdir().expect("tempdir");
    assert_eq!(
        eval_surf(dir.path(), "pipe_of_if", PIPE_OF_IF).trim_end(),
        "main = 11.0"
    );
}

#[test]
fn eval_file_agrees_with_deep_then_eval() {
    let dir = tempdir().expect("tempdir");
    for (name, source) in [
        ("pipe_of_if", PIPE_OF_IF),
        ("match_and_block", MATCH_AND_BLOCK),
        ("linked_std", LINKED_STD),
        ("tensor_and_value_root", TENSOR_AND_VALUE_ROOT),
    ] {
        let direct = eval_surf(dir.path(), name, source);
        assert!(
            !direct.trim().is_empty(),
            "{name}: eval --file printed no roots"
        );
        assert_eq!(
            direct,
            eval_through_deep(dir.path(), name),
            "{name}: eval --file X.ch and deep X.ch | eval --file X.dp disagree"
        );
    }
}

/// Entry points that check and then evaluate or lower the program they are
/// handed as text or as a parsed module (`eval`, `prepare_eval`,
/// `eval_in_context` and their variants, the CLI's text wrappers over them,
/// the obligation engine's source entry, and `compiler::lower`, which the
/// prove Beacon lane fed printed Surf until chelis#3172).
const TEXT_EVALUATORS: &[&str] = &[
    "eval",
    "eval_for_target",
    "eval_selected",
    "eval_selected_for_target",
    "eval_many",
    "prepare_eval",
    "eval_in_context",
    "eval_in_context_for_target",
    "eval_in_context_with_bindings",
    "eval_many_in_context",
    "prepare_eval_in_context",
    "check_in_context",
    "try_eval_for_target",
    "try_eval_result_for_target",
    "eval_bool_with_bindings",
    "run_surf_source_obligations",
    "lower",
];

#[derive(Default)]
struct CallNames {
    called: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for CallNames {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func
            && let Some(last) = path.path.segments.last()
        {
            self.called.insert(last.ident.to_string());
        }
        syn::visit::visit_expr_call(self, call);
    }
}

/// Every function body in `file` (free functions and methods, closures
/// included) that calls both `format_program` and a text evaluator.
fn printer_fed_evaluators(file: &Path) -> Vec<String> {
    struct Functions<'p> {
        file: &'p Path,
        found: Vec<String>,
    }
    impl Functions<'_> {
        fn inspect(&mut self, name: &syn::Ident, block: &syn::Block) {
            let mut calls = CallNames::default();
            calls.visit_block(block);
            if !calls.called.contains("format_program") {
                return;
            }
            for evaluator in TEXT_EVALUATORS {
                if calls.called.contains(*evaluator) {
                    self.found.push(format!(
                        "{}::{name} prints Surf and calls {evaluator}",
                        self.file.display()
                    ));
                }
            }
        }
    }
    impl<'ast> Visit<'ast> for Functions<'_> {
        fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
            self.inspect(&item.sig.ident, &item.block);
            syn::visit::visit_item_fn(self, item);
        }
        fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
            self.inspect(&item.sig.ident, &item.block);
            syn::visit::visit_impl_item_fn(self, item);
        }
    }
    let source = fs::read_to_string(file).expect("read source");
    let syntax = syn::parse_file(&source)
        .unwrap_or_else(|error| panic!("{} does not parse: {error}", file.display()));
    let mut functions = Functions {
        file,
        found: Vec::new(),
    };
    functions.visit_file(&syntax);
    functions.found
}

fn rust_sources(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read source dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension() == Some(OsStr::new("rs")) {
            out.push(path);
        }
    }
}

#[test]
fn no_cli_or_prove_function_evaluates_printed_surf() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir");
    let mut sources = Vec::new();
    for owner in ["chelis-cli", "chelis-prove"] {
        rust_sources(&crates.join(owner).join("src"), &mut sources);
    }
    sources.sort();
    // The scan must see the functions it guards.
    for guarded in [
        "main.rs",
        "obligation_run.rs",
        "property_runner.rs",
        "std_graph.rs",
        "beacon.rs",
    ] {
        assert!(
            sources
                .iter()
                .any(|path| path.file_name() == Some(OsStr::new(guarded))),
            "scan did not reach {guarded}"
        );
    }
    let found = sources
        .iter()
        .flat_map(|path| printer_fed_evaluators(path))
        .collect::<Vec<_>>();
    assert!(
        found.is_empty(),
        "a function evaluates or lowers the Surf printer's output instead of the program \
         it holds (chelis#3129, chelis#3172):\n{}",
        found.join("\n")
    );
}
