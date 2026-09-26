//! Rule D of the key-liveness model (#2413; spec/03 §4.4, spec/06 §5.2,
//! [05-RNG-1]) in three lanes: a selection executes every node a selected
//! root reaches, plus every potentially trapping node its entered
//! declarations own, and a node under a false activation checks nothing.
//!
//! The lanes: E is the DAG evaluator (`eval_selected`, `compiler::eval`,
//! `eval_in_context_with_bindings`) on a Tensor-lane root; C is the compiled
//! C of the selected entry, run natively (`compile_for_execution` without
//! top-level values, or `compile_for_execution_in_context`, which is
//! entry-scoped); H is the host interpreter on a Host-lane root. A monolithic
//! C build of a source with top-level values is the whole-program lane,
//! which initializes every value, so [`run_c`] refuses it as a witness.
//!
//! Lane routing at the reviewed head, asserted on each valid twin rather
//! than assumed: a value declaration built with `scalar_to_tensor` is
//! Tensor lane and one built with `to_tensor` is Host lane; a def whose
//! runtime `if` compares with `lt` stays a Tensor-lane root and a C kernel,
//! so the arm rows spell a taken arm `lt(0.0f32, s)` and an untaken one
//! `lt(s, 0.0f32)`. Every row is collected before the assertion, so one red
//! row never hides a sibling.
#![allow(deprecated)]
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

mod ownership_support;

use chelis_compiler_api::compiler::{
    CompiledExecutionArtifact, CompilerError, compile_for_execution,
    compile_for_execution_in_context, eval, eval_selected,
};
use chelis_compiler_api::context::CompiledContext;
use chelis_compiler_api::schema::{
    CompileRequest, CompileTarget, EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context, eval_in_context_with_bindings};
use chelis_ir::eval::TensorValue as DagTensor;
use chelis_types::types::{Lane, Prim};
use chelis_types::{RawTensor, finalize_tensor};
use chelis_unord::UnordMap;
use std::collections::BTreeMap;

const DROPOUT_DOMAIN: &str = "numeric trap: domain in dropout at f32";
const CAST_OVERFLOW: &str = "numeric trap: overflow in cast at i32";
const MUL_OVERFLOW: &str = "numeric trap: overflow in mul at i32";
const FLOOR_DIV_ZERO: &str = "numeric trap: division by zero in floor_div at i32";

/// The two trapping initializers every value-declaration row runs: a draw
/// with an invalid rate and an out-of-range `cast`.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Draw,
    Cast,
}

const KINDS: [Kind; 2] = [Kind::Draw, Kind::Cast];

impl Kind {
    fn trap(self) -> &'static str {
        match self {
            Kind::Draw => DROPOUT_DOMAIN,
            Kind::Cast => CAST_OVERFLOW,
        }
    }

    /// `sampled`, in the Tensor lane (`scalar_to_tensor`) or the Host lane
    /// (`to_tensor`), trapping when `traps` and in range otherwise.
    fn sampled(self, host: bool, traps: bool) -> String {
        let data = |value: &str| {
            if host {
                format!("to_tensor([{value}, 1.0f32])")
            } else {
                format!("scalar_to_tensor({value})")
            }
        };
        match self {
            Kind::Draw => format!(
                "sampled = dropout(key_from_seed(9i64), {}, {})\n",
                data("1.0f32"),
                if traps { "1.0f32" } else { "0.5f32" }
            ),
            Kind::Cast => format!(
                "sampled = cast({}, i32)\n",
                data(if traps { "3000000000.0f32" } else { "3.0f32" })
            ),
        }
    }
}

fn ones(count: usize) -> String {
    std::iter::repeat_n("1.0f32", count)
        .collect::<Vec<_>>()
        .join(", ")
}

fn f32_input(count: usize) -> TensorValue {
    TensorValue {
        shape: vec![count as i64],
        data: wire_values::storage_f32(vec![1.0; count]),
    }
}

fn i32_input(value: i32) -> TensorValue {
    TensorValue {
        shape: vec![1],
        data: serde_json::from_value(serde_json::json!({"dtype": "int32", "values": [value]}))
            .unwrap(),
    }
}

/// `x`, 32 ones: the Tensor-lane rows' only input.
fn x32() -> BTreeMap<String, TensorValue> {
    BTreeMap::from([("x".into(), f32_input(32))])
}

/// `x`, one 1.0: the #2466 and #2563 rows' input.
fn x1() -> BTreeMap<String, TensorValue> {
    BTreeMap::from([("x".into(), f32_input(1))])
}

fn select(
    source: &str,
    root: &str,
    bindings: BTreeMap<String, TensorValue>,
) -> Result<EvalResult, CompilerError> {
    eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            bindings,
        },
        &[root.into()],
    )
}

fn values(result: &EvalResult, name: &str) -> Vec<f64> {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no root `{name}`: {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => (0..value.data.len())
            .map(|index| value.data.element_f64_lossy(index))
            .collect(),
        ExecutionValue::Scalar { value } => vec![value.get().as_f64_lossy()],
        other => panic!("{other:?}"),
    }
}

fn lane(result: &EvalResult, name: &str) -> Option<Lane> {
    result
        .manifest
        .entries
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| entry.lane)
}

/// Collects one row's disagreement with its expectation.
#[derive(Default)]
struct Rows(Vec<String>);

impl Rows {
    /// `outcome` must be the typed trap `trap`.
    fn traps(&mut self, row: &str, outcome: Result<EvalResult, CompilerError>, trap: &str) {
        match outcome {
            Ok(result) => self.0.push(format!("{row}: returned {:?}", result.roots)),
            Err(error) if error.errors.iter().any(|error| error.message == trap) => {}
            Err(error) => self.0.push(format!(
                "{row}: expected `{trap}`, got {:?}",
                error.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            )),
        }
    }

    /// `outcome` must return `expected` for `root`, from `lane` when named.
    fn returns(
        &mut self,
        row: &str,
        outcome: Result<EvalResult, CompilerError>,
        root: &str,
        expected: &[f64],
        expected_lane: Option<Lane>,
    ) {
        match outcome {
            Ok(result) => {
                if values(&result, root) != expected {
                    self.0
                        .push(format!("{row}: `{root}` = {:?}", values(&result, root)));
                }
                if let Some(expected_lane) = expected_lane
                    && lane(&result, root) != Some(expected_lane)
                {
                    self.0.push(format!(
                        "{row}: `{root}` ran in {:?}, not {expected_lane:?}",
                        lane(&result, root)
                    ));
                }
            }
            Err(error) => self.0.push(format!(
                "{row}: {:?}",
                error.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            )),
        }
    }

    /// A C run must end as `expected`: `Ok(())` returned, `Err(trap)` trapped.
    fn c(&mut self, row: &str, outcome: Result<(), String>, expected: Result<(), &str>) {
        match (outcome, expected) {
            (Ok(()), Ok(())) => {}
            (Err(stderr), Err(trap)) if stderr.contains(trap) => {}
            (outcome, expected) => self
                .0
                .push(format!("{row}: C ended {outcome:?}, expected {expected:?}")),
        }
    }

    /// `main`'s C entry takes only `x` and returns.
    fn c_only_x(&mut self, row: &str, artifact: &CompiledExecutionArtifact) {
        let inputs = artifact
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>();
        if inputs == ["x"] {
            self.c(row, run_c(artifact, "main", 0), Ok(()));
        } else {
            self.0.push(format!("{row}: C entry takes {inputs:?}"));
        }
    }

    fn assert_empty(self) {
        assert!(self.0.is_empty(), "\n{}", self.0.join("\n"));
    }
}

/// A reef package `app` whose dependency `mylib` holds `library` as
/// `Mylib.Math`, compiled as a context, with its decoded copy.
fn dependency_contexts(library: &str) -> [CompiledContext; 2] {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("app");
    let write = |path: &str, contents: &str| {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    };
    write(
        "reef.toml",
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\n\
             module_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n"
        ),
    );
    write(
        "src/main.ch",
        "module App.Main\n\ndef placeholder() -> i32 = 0i32\n",
    );
    write(
        "mylib/reef.toml",
        &format!(
            "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\n\
             module_prefix = \"Mylib\"\n"
        ),
    );
    write("mylib/src/math.ch", library);
    write(
        "reef.lock",
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\n\n[[dependencies]]\n\
             name = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\n\
             archive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\n\
             kind = \"path\"\npath = \"./mylib\"\n"
        ),
    );
    let context = compile_reef_context(directory.path(), &root)
        .unwrap_or_else(|error| panic!("{library}\n{error:?}"));
    let decoded = CompiledContext::decode(&context.encode().unwrap()).unwrap();
    [context, decoded]
}

/// Run a selected C entry natively. Each `f32` input is the driver's
/// `input(n)`, whose element `i` is `2i - 3`; each `i32` input holds
/// `i32_fill`. Returns `Ok(())` when the entry returns and the trap text
/// when the program aborts. A whole-program artifact (a `main` of its own)
/// initializes every top-level value and is no witness for a selection, so
/// it is refused.
fn run_c(artifact: &CompiledExecutionArtifact, entry: &str, i32_fill: i32) -> Result<(), String> {
    let stem = &artifact.compile_result.entry_name;
    let file = |path: String| {
        artifact
            .compile_result
            .files
            .iter()
            .find(|file| file.path == path)
            .unwrap_or_else(|| panic!("no generated `{path}`"))
            .contents
            .clone()
    };
    let program = ownership_support::GeneratedProgram::new(
        file(format!("{stem}.c")),
        file(format!("{stem}.h")),
    );
    assert!(
        !program.contains("\nint main(void)"),
        "`{entry}` compiled as a whole program ({:?}), not as its selected entry",
        artifact.entry_lane_decline
    );
    let mut driver = format!(
        "static chelis_tensor *input_i32(int64_t n) {{\n    chelis_tensor *x = chelis_alloc(1, &n, CHELIS_DTYPE_I32);\n    chelis_tensor_write *guard = chelis_tensor_begin_write(x);\n    chelis_write_view view = chelis_tensor_write_view(guard);\n    for (int64_t i = 0; i < n; ++i) ((int32_t *)view.data)[i] = {i32_fill};\n    chelis_tensor_end_write(guard);\n    return x;\n}}\nint main(void) {{\n"
    );
    let mut inputs = Vec::new();
    for (index, spec) in artifact.inputs.iter().enumerate() {
        let size = match spec.dims.as_slice() {
            [dim] => serde_json::to_value(dim).unwrap()["size"]
                .as_u64()
                .unwrap_or_else(|| panic!("`{entry}` input `{}` is not static", spec.name)),
            dims => panic!("`{entry}` input `{}` has rank {}", spec.name, dims.len()),
        };
        let make = if spec.dtype == "f32" {
            "input"
        } else {
            "input_i32"
        };
        driver.push_str(&format!("    chelis_tensor *in{index} = {make}({size});\n"));
        inputs.push(format!("in{index}"));
    }
    let direct = program.header().contains("chelis_fn_");
    if direct {
        // A Host-lane entry is one C function taking its tensors in order.
        // Its artifact lists no inputs; every such entry here takes one
        // `tensor[1, f32]` per parameter.
        let declaration = program.declaration(entry);
        let arity = declaration.split_once('(').map_or(0, |(_, parameters)| {
            parameters.matches("chelis_tensor").count()
        });
        assert!(inputs.is_empty() || inputs.len() == arity, "{declaration}");
        for index in inputs.len()..arity {
            driver.push_str(&format!("    chelis_tensor *in{index} = input(1);\n"));
            inputs.push(format!("in{index}"));
        }
        driver.push_str(&format!(
            "    chelis_tensor *out = {}({});\n    chelis_tensor_release(out);\n",
            program.symbol(entry),
            inputs.join(", ")
        ));
    } else {
        let outputs = artifact.outputs.len();
        let list = if inputs.is_empty() {
            "NULL".to_string()
        } else {
            driver.push_str(&format!(
                "    chelis_tensor *inputs[] = {{{}}};\n",
                inputs.join(", ")
            ));
            "inputs".to_string()
        };
        driver.push_str(&format!(
            "    chelis_tensor *outputs[{}] = {{NULL}};\n    {}({list}, {}, outputs, {outputs});\n    for (int i = 0; i < {outputs}; ++i) chelis_tensor_release(outputs[i]);\n",
            outputs.max(1),
            artifact.host_entry_name,
            inputs.len()
        ));
    }
    for input in &inputs {
        driver.push_str(&format!("    chelis_tensor_release({input});\n"));
    }
    driver.push_str("    return 0;\n}\n");
    match std::panic::catch_unwind(|| ownership_support::run_failure_stderr(&program, &driver)) {
        Ok(stderr) => Err(stderr),
        Err(_) => {
            ownership_support::balanced(&ownership_support::run(&program, &driver));
            Ok(())
        }
    }
}

fn compile_c(source: &str, entry: &str) -> CompiledExecutionArtifact {
    compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some(entry.into()),
    })
    .unwrap_or_else(|error| panic!("{source}\n{error:?}"))
}

fn compile_c_in(context: &CompiledContext, source: &str, entry: &str) -> CompiledExecutionArtifact {
    compile_for_execution_in_context(context, source, CompileTarget::C, Some(entry))
        .unwrap_or_else(|error| panic!("{source}\n{error:?}"))
}

/// `selected(x)`, whose runtime `if` reads `sampled` in a dead binding of
/// its `then` arm, which runs exactly when `taken` (the sum of `x` is
/// positive for both the evaluator's ones and the C driver's input).
fn arm_def(taken: bool) -> String {
    let condition = if taken {
        "lt(0.0f32, s)"
    } else {
        "lt(s, 0.0f32)"
    };
    format!(
        "def selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  if {condition} then {{\n    dead = sampled\n    x\n  }} else x\n}}\n"
    )
}

/// The Host-lane twin of [`arm_def`]: a nullary `main` over constant data.
fn host_arm_def(taken: bool) -> String {
    let condition = if taken {
        "lt(0.0f32, s)"
    } else {
        "lt(s, 0.0f32)"
    };
    format!(
        "def main() -> tensor[32, f32] = {{\n  x = to_tensor([{}])\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  if {condition} then {{\n    dead = sampled\n    x\n  }} else x\n}}\n",
        ones(32)
    )
}

/// The client of a dependency exporting `sampled`, selecting [`arm_def`].
fn arm_client(taken: bool) -> String {
    format!(
        "module App.Eval\nimport Mylib.Math (sampled)\n\n{}",
        arm_def(taken)
    )
}

/// (a) A taken arm's dead reference to a trapping value declaration runs its
/// initializer, inlined at the reference under the arm's activation, so it
/// traps: in the DAG evaluator (the value beside the selected def) and in
/// the in-context C entry (the value in a dependency). The valid twin
/// returns `x` from the Tensor lane.
///
/// Evidentiary status: DISPOSITION LOCK. Both lanes trapped at b003b1610,
/// where the reference entered the value's declaration unconditionally.
#[test]
fn a_taken_arms_dead_reference_to_a_value_declaration_traps_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    for kind in KINDS {
        let evaluate = |traps: bool| {
            select(
                &format!("{}{}", kind.sampled(false, traps), arm_def(true)),
                "selected",
                x32(),
            )
        };
        rows.returns(
            &format!("E {kind:?} valid twin"),
            evaluate(false),
            "selected",
            &[1.0; 32],
            Some(Lane::Tensor),
        );
        rows.traps(&format!("E {kind:?}"), evaluate(true), kind.trap());
        for context in &dependency_contexts(&format!(
            "module Mylib.Math\nexport (sampled)\n\n{}",
            kind.sampled(false, true)
        )) {
            let artifact = compile_c_in(context, &arm_client(true), "selected");
            rows.c(
                &format!("C {kind:?}"),
                run_c(&artifact, "selected", 0),
                Err(kind.trap()),
            );
        }
    }
    rows.assert_empty();
}

/// (a) in the host interpreter: a Host-lane `main` whose taken arm reads a
/// trapping value declaration in a dead binding, and a Host-lane value root
/// applying [`arm_def`] in a context whose dependency exports the value.
/// Rule D enters the value, so both trap. The valid twin returns `x` from
/// the Host lane.
///
/// Evidentiary status: REGRESSION TEST. Red at ad0abe6a9: the host runs the
/// body as a DAG kernel, which never demanded the dead value, and
/// initializes beforehand only the values the body reaches on every path
/// (`ProgramScope::reached_by_call` skips `if` and `match` arms), so the
/// taken arm's value was never initialized. The kernel's lowering now
/// inlines a trapping initializer at the reference, under the arm's
/// activation.
#[test]
fn a_taken_arms_dead_reference_to_a_value_declaration_traps_in_the_host_lane() {
    let mut rows = Rows::default();
    for kind in KINDS {
        let evaluate = |traps: bool| {
            select(
                &format!("{}{}", kind.sampled(true, traps), host_arm_def(true)),
                "main",
                BTreeMap::new(),
            )
        };
        rows.returns(
            &format!("H {kind:?} valid twin"),
            evaluate(false),
            "main",
            &[1.0; 32],
            Some(Lane::Host),
        );
        rows.traps(&format!("H {kind:?}"), evaluate(true), kind.trap());
        for context in &dependency_contexts(&format!(
            "module Mylib.Math\nexport (sampled)\n\n{}",
            kind.sampled(false, true)
        )) {
            let source = format!(
                "{}answer = selected(to_tensor([{}]))\n",
                arm_client(true),
                ones(32)
            );
            rows.traps(
                &format!("H in context {kind:?}"),
                eval_in_context_with_bindings(context, &source, BTreeMap::new()),
                kind.trap(),
            );
        }
    }
    rows.assert_empty();
}

/// (b) The untaken twin of (a): the arm's activation is false, so its dead
/// reference checks nothing ([05-RNG-1]) and the selection returns `x`.
///
/// Evidentiary status: REGRESSION TEST. Red at ad0abe6a9 in both lanes:
/// lowering recorded the arm's reference without its activation, so the
/// value was entered unconditionally and trapped. The reference now inlines
/// the initializer under the arm's activation, and the value's own nodes run
/// only when it is selected.
#[test]
fn an_untaken_arms_dead_reference_to_a_value_declaration_runs_nothing_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    for kind in KINDS {
        rows.returns(
            &format!("E {kind:?}"),
            select(
                &format!("{}{}", kind.sampled(false, true), arm_def(false)),
                "selected",
                x32(),
            ),
            "selected",
            &[1.0; 32],
            None,
        );
        for context in &dependency_contexts(&format!(
            "module Mylib.Math\nexport (sampled)\n\n{}",
            kind.sampled(false, true)
        )) {
            let artifact = compile_c_in(context, &arm_client(false), "selected");
            rows.c(
                &format!("C {kind:?}"),
                run_c(&artifact, "selected", 0),
                Ok(()),
            );
        }
    }
    rows.assert_empty();
}

/// (b) in the host interpreter: the untaken arm's value is not initialized,
/// in a Host-lane `main` and in a context whose dependency exports it.
///
/// Evidentiary status: DISPOSITION LOCK (green at b003b1610, because the
/// host never initialized an arm's value); it keeps the fix of the taken-arm
/// host test from initializing an untaken arm's value.
#[test]
fn an_untaken_arms_dead_reference_to_a_value_declaration_runs_nothing_in_the_host_lane() {
    let mut rows = Rows::default();
    for kind in KINDS {
        rows.returns(
            &format!("H {kind:?}"),
            select(
                &format!("{}{}", kind.sampled(true, true), host_arm_def(false)),
                "main",
                BTreeMap::new(),
            ),
            "main",
            &[1.0; 32],
            Some(Lane::Host),
        );
        for context in &dependency_contexts(&format!(
            "module Mylib.Math\nexport (sampled)\n\n{}",
            kind.sampled(false, true)
        )) {
            let source = format!(
                "{}answer = selected(to_tensor([{}]))\n",
                arm_client(false),
                ones(32)
            );
            rows.returns(
                &format!("H in context {kind:?}"),
                eval_in_context_with_bindings(context, &source, BTreeMap::new()),
                "answer",
                &[1.0; 32],
                Some(Lane::Host),
            );
        }
    }
    rows.assert_empty();
}

/// `f`, a function whose body names `sampled` in a dead binding.
const F: &str = "def f(v: tensor[32, f32]) -> tensor[32, f32] = {\n  dead = sampled\n  v\n}\n";

/// (c) `g = f` names a function without applying it, so it enters nothing
/// and `sampled` does not run; applying the alias (`g = f` then `g(x)`) or
/// `f` itself inlines `f`'s body, which names `sampled`, so both trap. Rows:
/// the DAG evaluator on a value root and on a function root, and a
/// dependency exporting `f` (with `sampled` private) in the in-context
/// evaluator and the in-context C entry.
///
/// Evidentiary status: DISPOSITION LOCK at b003b1610 for every row (the
/// unapplied rows' regression evidence is
/// `a_function_named_as_a_value_and_not_applied_runs_nothing_in_any_lane`).
#[test]
fn an_unapplied_function_enters_nothing_and_its_applied_twins_trap_in_the_evaluator_and_c() {
    let shapes = [
        ("unapplied", "{\n  g = f\n  x\n}", false),
        ("applied through its alias", "{\n  g = f\n  g(x)\n}", true),
        ("applied directly", "f(x)", true),
    ];
    let mut rows = Rows::default();
    for kind in KINDS {
        let sampled = kind.sampled(false, true);
        for (shape, body, traps) in shapes {
            let value_root = format!(
                "x: tensor[32, f32] = x\n{sampled}{F}selected = {}\n",
                body.replace("x\n}", "copy(x)\n}")
                    .replace("(x)", "(copy(x))")
            );
            let function_root = format!(
                "{sampled}{F}def selected(x: tensor[32, f32]) -> tensor[32, f32] = {body}\n"
            );
            for (root_kind, source) in
                [("value root", value_root), ("function root", function_root)]
            {
                let row = format!("E {kind:?} {root_kind}, {shape}");
                let outcome = select(&source, "selected", x32());
                if traps {
                    rows.traps(&row, outcome, kind.trap());
                } else {
                    rows.returns(&row, outcome, "selected", &[1.0; 32], Some(Lane::Tensor));
                }
            }
            for context in
                &dependency_contexts(&format!("module Mylib.Math\nexport (f)\n\n{sampled}{F}"))
            {
                let client = format!(
                    "module App.Eval\nimport Mylib.Math (f)\n\ndef main(x: tensor[32, f32]) -> tensor[32, f32] = {body}\n"
                );
                let row = format!("{kind:?} in context, {shape}");
                let outcome = eval_in_context_with_bindings(context, &client, x32());
                let artifact = compile_c_in(context, &client, "main");
                let ran = run_c(&artifact, "main", 0);
                if traps {
                    rows.traps(&format!("E {row}"), outcome, kind.trap());
                    rows.c(&format!("C {row}"), ran, Err(kind.trap()));
                } else {
                    rows.returns(
                        &format!("E {row}"),
                        outcome,
                        "main",
                        &[1.0; 32],
                        Some(Lane::Tensor),
                    );
                    rows.c(&format!("C {row}"), ran, Ok(()));
                }
            }
        }
    }
    rows.assert_empty();
}

/// (c) in the host interpreter: a Host-lane `main` over a Host-lane
/// `sampled`. `g = f` alone runs nothing; applying `f` through a local alias
/// (called, through an alias of the alias, or as a pipe stage) or directly
/// runs its body, which names `sampled`, so it traps.
///
/// Evidentiary status: REGRESSION TEST for the three alias rows, which
/// returned `t` at b003b1610: the host initializes the values a kernel body
/// reaches, and its walk (`ProgramScope`'s `ReachWalk`) treated an applied
/// local name as reaching nothing, even when the local names a top-level
/// function. The unapplied and direct rows are disposition locks.
#[test]
fn a_function_applied_through_a_local_alias_initializes_what_it_names_in_the_host_lane() {
    let shapes = [
        ("unapplied", "  g = f\n  t\n", false),
        ("called through its alias", "  g = f\n  g(t)\n", true),
        (
            "called through an alias of its alias",
            "  g = f\n  h = g\n  h(t)\n",
            true,
        ),
        ("piped through its alias", "  g = f\n  t |> g\n", true),
        ("called directly", "  f(t)\n", true),
    ];
    let mut rows = Rows::default();
    for kind in KINDS {
        for (shape, body, traps) in shapes {
            let evaluate = |trapping: bool| {
                select(
                    &format!(
                        "{}{F}def main() -> tensor[32, f32] = {{\n  t = to_tensor([{}])\n{body}}}\n",
                        kind.sampled(true, trapping),
                        ones(32)
                    ),
                    "main",
                    BTreeMap::new(),
                )
            };
            let row = format!("H {kind:?}, {shape}");
            rows.returns(
                &format!("{row}, valid twin"),
                evaluate(false),
                "main",
                &[1.0; 32],
                Some(Lane::Host),
            );
            if traps {
                rows.traps(&row, evaluate(true), kind.trap());
            } else {
                rows.returns(&row, evaluate(true), "main", &[1.0; 32], Some(Lane::Host));
            }
        }
    }
    rows.assert_empty();
}

/// #2466's P1 in its reviewer's shape: `g`, never called, discards an
/// integer `mul` of its own parameter `z`.
const G: &str = "def g(z: tensor[1, i32]) -> tensor[1, i32] = {\n  t = mul(&z, &z)\n  z\n}\n";

const MAIN: &str = "def main(x: tensor[1, f32]) -> tensor[1, f32] = add(&x, &x)\n";

/// `main`, discarding a call of `g` on `100000 * x`, whose square overflows.
const MAIN_CALLS_G: &str = "def main(x: tensor[1, f32]) -> tensor[1, f32] = {\n  d = g(cast(mul(&x, to_tensor([100000.0f32])), i32))\n  add(&x, &x)\n}\n";

/// (d) #2466's P1: an uncalled function's discarded overflow is not part of
/// a selection that does not enter it, so selecting `main` needs only `x`
/// (never "missing required input `z`") and returns `2x`: through
/// `eval_selected`, through `compiler::eval` of every root, in context with
/// `g` in the snippet, in an imported dependency the snippet leaves unused,
/// and in a dependency the snippet does not import; and as a C entry, whose
/// only input is `x`, beside `g` and in context. The same `g`, selected or
/// called, still traps on its overflow in every lane. The host rows run a
/// Host-lane `main` beside `g`.
///
/// Evidentiary status: REGRESSION TEST. At c05287eb (#2466's head) every
/// evaluator row that selects `main`, the `eval_selected` and in-context
/// `g called` rows included, failed "missing required input `z`", and every
/// in-context C entry took `["z", "x"]`. The monolithic C rows, the host
/// rows and the rows selecting `g` are disposition locks: they held there.
#[test]
fn an_uncalled_functions_discarded_overflow_needs_no_input_and_traps_when_entered() {
    let mut rows = Rows::default();
    let beside = format!("{G}{MAIN}");
    rows.returns(
        "E main beside g",
        select(&beside, "main", x1()),
        "main",
        &[2.0],
        Some(Lane::Tensor),
    );
    rows.returns(
        "E every root beside g",
        eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: beside.clone(),
            bindings: x1(),
        }),
        "main",
        &[2.0],
        Some(Lane::Tensor),
    );
    rows.traps(
        "E g selected",
        select(
            &beside,
            "g",
            BTreeMap::from([("z".into(), i32_input(100000))]),
        ),
        MUL_OVERFLOW,
    );
    rows.returns(
        "E g selected, in range",
        select(&beside, "g", BTreeMap::from([("z".into(), i32_input(3))])),
        "g",
        &[3.0],
        Some(Lane::Tensor),
    );
    let calls = format!("{G}{MAIN_CALLS_G}");
    rows.traps("E g called", select(&calls, "main", x1()), MUL_OVERFLOW);

    rows.c_only_x("C main beside g", &compile_c(&beside, "main"));
    rows.c(
        "C g selected",
        run_c(&compile_c(&beside, "g"), "g", 100000),
        Err(MUL_OVERFLOW),
    );
    rows.c(
        "C g called",
        run_c(&compile_c(&calls, "main"), "main", 0),
        Err(MUL_OVERFLOW),
    );

    let host = |body: &str| {
        format!(
            "{G}def main() -> tensor[1, f32] = {{\n  x = to_tensor([1.0f32])\n{body}  add(&x, &x)\n}}\n"
        )
    };
    rows.returns(
        "H main beside g",
        select(&host(""), "main", BTreeMap::new()),
        "main",
        &[2.0],
        Some(Lane::Host),
    );
    rows.traps(
        "H g called",
        select(
            &host("  d = g(to_tensor([100000i32]))\n"),
            "main",
            BTreeMap::new(),
        ),
        MUL_OVERFLOW,
    );

    for context in &dependency_contexts(&format!(
        "module Mylib.Math\nexport (g, h)\n\n{G}def h(z: tensor[1, f32]) -> tensor[1, f32] = add(&z, &z)\n"
    )) {
        for (row, snippet) in [
            ("not imported", format!("module App.Eval\n\n{MAIN}")),
            (
                "imported, unused",
                format!("module App.Eval\nimport Mylib.Math (g)\n\n{MAIN}"),
            ),
            ("in the snippet", format!("module App.Eval\n\n{G}{MAIN}")),
        ] {
            rows.returns(
                &format!("E in context, g {row}"),
                eval_in_context_with_bindings(context, &snippet, x1()),
                "main",
                &[2.0],
                Some(Lane::Tensor),
            );
            rows.c_only_x(
                &format!("C in context, g {row}"),
                &compile_c_in(context, &snippet, "main"),
            );
        }
        let calls = format!("module App.Eval\nimport Mylib.Math (g)\n\n{MAIN_CALLS_G}");
        rows.traps(
            "E in context, g called",
            eval_in_context_with_bindings(context, &calls, x1()),
            MUL_OVERFLOW,
        );
        rows.c(
            "C in context, g called",
            run_c(&compile_c_in(context, &calls, "main"), "main", 0),
            Err(MUL_OVERFLOW),
        );
    }
    rows.assert_empty();
}

/// One of #2563's shapes: an arm holding a discarded or consumed float
/// `cast` out of `i32`'s range, or an integer `floor_div` by the zero in
/// `d`, with or without `grad`.
struct ArmShape {
    name: String,
    source: String,
    integer: bool,
    /// The lane `eval_selected` runs the root in: a `gt` condition makes the
    /// def a Host-lane root, an `lt` one keeps it in the Tensor lane.
    lane: Lane,
    /// The C row runs unless the entry is a `grad` transform in the Tensor
    /// lane, which the compiled-execution lane refuses (chelis#1138).
    c: bool,
    expected: Vec<f64>,
}

/// #2563's shapes, their arm taken or not. `x` is 1.0 for the evaluator and
/// -3.0 for the C driver and `d` is 0, so each condition decides the same
/// arm in both lanes. The reviewer's `gt` spellings are the Host-lane rows;
/// the float shapes are repeated with `lt` for the Tensor lane. The integer
/// shapes have no Tensor-lane row: there `d` is read only through the
/// discarded arm, which `a_dead_let_on_a_parameter_read_nowhere_else_traps_in_the_evaluator`
/// shows fails before any trap.
fn arm_shapes(taken: bool) -> Vec<ArmShape> {
    let cast = "cast(mul(&x, to_tensor([1e30f32])), i32)";
    let discarded = |name: &str, condition: &str| {
        format!(
            "def {name}(x: tensor[1, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(&x, 0i32))\n  r = if {condition} then {{\n    dead = {cast}\n    sum(&x, 0i32)\n  }} else sum(&x, 0i32)\n  sum(x, 0i32)\n}}\n"
        )
    };
    let consumed = |name: &str, condition: &str| {
        format!(
            "def {name}(x: tensor[1, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(&x, 0i32))\n  r = if {condition} then cast({cast}, f32) else x\n  sum(r, 0i32)\n}}\n"
        )
    };
    let grad = "def selected(x: tensor[1, f32]) -> tensor[1, f32] = grad(loss)(x)\n";
    let mut shapes = Vec::new();
    for (lane, condition) in [
        (
            Lane::Host,
            if taken {
                "gt(s, -5.0f32)"
            } else {
                "gt(s, 5.0f32)"
            },
        ),
        (
            Lane::Tensor,
            if taken {
                "lt(s, 5.0f32)"
            } else {
                "lt(5.0f32, s)"
            },
        ),
    ] {
        for (name, source, gradient) in [
            ("discarded cast", discarded("selected", condition), false),
            ("consumed cast", consumed("selected", condition), false),
            (
                "discarded cast under grad",
                format!("{}{grad}", discarded("loss", condition)),
                true,
            ),
            (
                "consumed cast under grad",
                format!("{}{grad}", consumed("loss", condition)),
                true,
            ),
        ] {
            shapes.push(ArmShape {
                name: format!("{name}, {lane:?} lane"),
                source,
                integer: false,
                lane,
                c: lane == Lane::Host || !gradient,
                expected: vec![1.0],
            });
        }
    }
    let condition = if taken { "gt(s, -1i32)" } else { "gt(s, 0i32)" };
    shapes.push(ArmShape {
        name: "discarded floor_div, Host lane".into(),
        source: format!(
            "def selected(x: tensor[1, f32], d: tensor[1, i32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(&d, 0i32))\n  r = if {condition} then {{\n    q = floor_div(to_tensor([7i32]), &d)\n    sum(&x, 0i32)\n  }} else sum(&x, 0i32)\n  sum(x, 0i32)\n}}\n"
        ),
        integer: true,
        lane: Lane::Host,
        c: true,
        expected: vec![1.0],
    });
    shapes.push(ArmShape {
        name: "consumed floor_div, Host lane".into(),
        source: format!(
            "def selected(x: tensor[1, f32], d: tensor[1, i32]) -> tensor[i32] = {{\n  s = tensor_to_scalar(sum(&d, 0i32))\n  r = if {condition} then floor_div(to_tensor([7i32]), &d) else d\n  sum(r, 0i32)\n}}\n"
        ),
        integer: true,
        lane: Lane::Host,
        c: true,
        expected: vec![0.0],
    });
    shapes
}

fn arm_bindings(shape: &ArmShape) -> BTreeMap<String, TensorValue> {
    let mut bindings = x1();
    if shape.integer {
        bindings.insert("d".into(), i32_input(0));
    }
    bindings
}

/// (e) #2563: an operation in an untaken `if` arm checks nothing, discarded
/// or consumed, a float `cast` or an integer `floor_div`, with and without
/// `grad` ([05-RNG-1] with spec/06 §5.2), through `eval_selected` (Host-lane
/// and Tensor-lane roots) and the selected C entry.
///
/// Evidentiary status: HELD RED at b003b1610 in every row (the arm's
/// trapping node is kept live without its activation); the expectation is
/// Rule D's. The fix is the activation-carrying trap seed (phase 2).
#[test]
fn an_untaken_arms_operation_checks_nothing_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    for shape in arm_shapes(false) {
        rows.returns(
            &format!("E {}", shape.name),
            select(&shape.source, "selected", arm_bindings(&shape)),
            "selected",
            &shape.expected,
            Some(shape.lane),
        );
        if shape.c {
            rows.c(
                &format!("C {}", shape.name),
                run_c(&compile_c(&shape.source, "selected"), "selected", 0),
                Ok(()),
            );
        }
    }
    rows.assert_empty();
}

/// The positive control for (e): the same operations in a taken arm trap in
/// both lanes, so the untaken rows cannot pass by dropping the check.
///
/// Evidentiary status: DISPOSITION LOCK (green at b003b1610).
#[test]
fn a_taken_arms_operation_traps_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    for shape in arm_shapes(true) {
        let trap = if shape.integer {
            FLOOR_DIV_ZERO
        } else {
            CAST_OVERFLOW
        };
        rows.traps(
            &format!("E {}", shape.name),
            select(&shape.source, "selected", arm_bindings(&shape)),
            trap,
        );
        if shape.c {
            // The C runtime words an integer division by zero its own way.
            let c_trap = if shape.integer {
                "division"
            } else {
                CAST_OVERFLOW
            };
            rows.c(
                &format!("C {}", shape.name),
                run_c(&compile_c(&shape.source, "selected"), "selected", 0),
                Err(c_trap),
            );
        }
    }
    rows.assert_empty();
}

/// spec/03 §4.4: a `let` initializer is evaluated whether or not the
/// binding is read, so a dead non-draw binding's out-of-range `cast` traps
/// in a selected def: in the DAG evaluator, in its C entry, and in the host
/// interpreter. Each valid twin returns the def's value from its lane.
///
/// Evidentiary status: DISPOSITION LOCK (green at b003b1610).
#[test]
fn a_dead_non_draw_let_initializer_traps_in_every_lane() {
    let mut rows = Rows::default();
    let tensor = |factor: &str| {
        format!(
            "def selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n  dead = cast(mul(sum(copy(x), 0i32), scalar_to_tensor({factor})), i32)\n  x\n}}\n"
        )
    };
    rows.returns(
        "E valid twin",
        select(&tensor("3.0f32"), "selected", x32()),
        "selected",
        &[1.0; 32],
        Some(Lane::Tensor),
    );
    rows.traps(
        "E",
        select(&tensor("3000000000.0f32"), "selected", x32()),
        CAST_OVERFLOW,
    );
    rows.c(
        "C valid twin",
        run_c(&compile_c(&tensor("3.0f32"), "selected"), "selected", 0),
        Ok(()),
    );
    rows.c(
        "C",
        run_c(
            &compile_c(&tensor("3000000000.0f32"), "selected"),
            "selected",
            0,
        ),
        Err(CAST_OVERFLOW),
    );
    let host = |factor: &str| {
        format!(
            "def main() -> tensor[2, f32] = {{\n  x = to_tensor([1.0f32, 1.0f32])\n  dead = cast(mul(sum(copy(x), 0i32), scalar_to_tensor({factor})), i32)\n  x\n}}\n"
        )
    };
    rows.returns(
        "H valid twin",
        select(&host("3.0f32"), "main", BTreeMap::new()),
        "main",
        &[1.0; 2],
        Some(Lane::Host),
    );
    rows.traps(
        "H",
        select(&host("3000000000.0f32"), "main", BTreeMap::new()),
        CAST_OVERFLOW,
    );
    rows.assert_empty();
}

/// spec/03 §4.4 on a parameter the def reads only in the dead binding:
/// `selected(x, y)` discards `cast(3000000000.0 * y, i32)`. With `y = 1.0` bound
/// the selection traps; with `y = 0.0` it returns `sum(x)` from the Tensor
/// lane. The C entry takes `y` and traps on the driver's -3.0.
///
/// Evidentiary status: RED at b003b1610 in the DAG evaluator, which fails
/// both bindings with "missing required input `y`" though `y` is bound: the
/// manifest's required inputs (`required_inputs_for_dag_root` in
/// `compiler.rs`) walk only the selected root's data ancestors, the binding
/// filter in `eval_compiled` drops every other binding, and the evaluator's
/// live mask then seeds the dead `cast`, whose `Load` of `y` finds nothing.
/// It fails the same way at c05287eb. The C row is a disposition lock.
#[test]
fn a_dead_let_on_a_parameter_read_nowhere_else_traps_in_the_evaluator() {
    let source = "def selected(x: tensor[1, f32], y: tensor[1, f32]) -> tensor[f32] = {\n  dead = cast(mul(&y, to_tensor([3000000000.0f32])), i32)\n  sum(x, 0i32)\n}\n";
    let bound = |y: f32| {
        BTreeMap::from([
            ("x".into(), f32_input(1)),
            (
                "y".into(),
                TensorValue {
                    shape: vec![1],
                    data: wire_values::storage_f32(vec![y]),
                },
            ),
        ])
    };
    let mut rows = Rows::default();
    rows.returns(
        "E y = 0",
        select(source, "selected", bound(0.0)),
        "selected",
        &[1.0],
        Some(Lane::Tensor),
    );
    rows.traps(
        "E y = 1",
        select(source, "selected", bound(1.0)),
        CAST_OVERFLOW,
    );
    rows.c(
        "C",
        run_c(&compile_c(source, "selected"), "selected", 0),
        Err(CAST_OVERFLOW),
    );
    rows.assert_empty();
}

/// `xs` holding `values`: the C driver's `input(n)` holds `2i - 3`, so a
/// one-row `xs` is `[-3.0]` in both lanes.
fn xs(values: &[f32]) -> BTreeMap<String, TensorValue> {
    BTreeMap::from([(
        "xs".into(),
        TensorValue {
            shape: vec![values.len() as i64],
            data: wire_values::storage_f32(values.to_vec()),
        },
    )])
}

/// A vmapped row's arm over `rows` rows: the cast of a row scaled by 1e9
/// overflows `i32` for -3.0 and fits for -1.0. A `gt` condition keeps the
/// entry in the Host lane, whose C entry the driver runs.
fn vmapped_arm(condition: &str, rows: usize) -> String {
    format!(
        "def row(x: tensor[f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(copy(x))\n  if {condition} then cast(cast(mul(&x, scalar_to_tensor(1000000000.0f32)), i32), f32) else x\n}}\n\ndef selected(xs: tensor[{rows}, f32]) -> tensor[{rows}, f32] = vmap(row)(xs)\n"
    )
}

/// The vmap batching pass (spec/10 §3): an `if` over a vmapped row gives each
/// row its own activation, so a row whose arm is not taken checks nothing
/// while a taken row checks. In the evaluator row 0 (-3.0) is untaken and
/// would overflow while row 1 (-1.0) is taken and fits; the C entry runs the
/// one row -3.0. The taken twin takes -3.0 and traps.
///
/// Evidentiary status: REGRESSION TEST. At eb608d063 the untaken row's cast
/// traps in both lanes.
#[test]
fn a_vmapped_arm_checks_only_in_the_rows_that_take_it_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    let untaken = "gt(s, -2.0f32)";
    rows.returns(
        "E untaken row",
        select(&vmapped_arm(untaken, 2), "selected", xs(&[-3.0, -1.0])),
        "selected",
        &[-3.0, -1_000_000_000.0],
        None,
    );
    rows.c(
        "C untaken row",
        run_c(
            &compile_c(&vmapped_arm(untaken, 1), "selected"),
            "selected",
            0,
        ),
        Ok(()),
    );
    let taken = "gt(-2.0f32, s)";
    rows.traps(
        "E taken row",
        select(&vmapped_arm(taken, 2), "selected", xs(&[-3.0, -1.0])),
        CAST_OVERFLOW,
    );
    rows.c(
        "C taken row",
        run_c(
            &compile_c(&vmapped_arm(taken, 1), "selected"),
            "selected",
            0,
        ),
        Err(CAST_OVERFLOW),
    );
    rows.assert_empty();
}

/// A vmap call in a runtime `if` arm over `xs = [-3.0]`: the row's cast of
/// the row scaled by 1e9 overflows.
fn vmap_in_arm(condition: &str) -> String {
    format!(
        "def row(x: tensor[f32]) -> tensor[f32] = cast(cast(mul(&x, scalar_to_tensor(1000000000.0f32)), i32), f32)\n\ndef selected(xs: tensor[1, f32]) -> tensor[1, f32] = {{\n  s = tensor_to_scalar(sum(copy(xs), 0i32))\n  if {condition} then vmap(row)(xs) else xs\n}}\n"
    )
}

/// The vmap batching pass at its call site (spec/10 §3): a vmapped body
/// spliced into an arm runs under the arm's activation, so when the arm is
/// not taken no row checks. The taken twin traps. The compiled-execution
/// lane refuses a transform entry (chelis#1138), so this is the evaluator's.
///
/// Evidentiary status: REGRESSION TEST. At eb608d063 the untaken arm's
/// vmapped cast traps.
#[test]
fn a_vmap_in_an_untaken_arm_checks_nothing_in_the_evaluator() {
    let mut rows = Rows::default();
    rows.returns(
        "E untaken arm",
        select(&vmap_in_arm("gt(s, 0.0f32)"), "selected", xs(&[-3.0])),
        "selected",
        &[-3.0],
        None,
    );
    rows.traps(
        "E taken arm",
        select(&vmap_in_arm("gt(s, -5.0f32)"), "selected", xs(&[-3.0])),
        CAST_OVERFLOW,
    );
    rows.assert_empty();
}

/// `selected` inlines `h`, whose runtime `if` arm applies `vmap(grad(f))`
/// to `xs = [-3.0]` (the evaluator's binding and the C driver's input) and
/// whose else arm is the total `xs`. `f`'s body discards a checking node
/// that traps on the row: the row scaled by 1e30 cast to `i32` (overflow),
/// or, when `integer`, 7 floor-divided by the zero the row gives scaled by
/// 0. `selected` itself names no transform, so the compiled-execution lane
/// takes it as its entry (chelis#1138 refuses an entry whose own body does).
fn vmap_grad_in_arm(condition: &str, integer: bool) -> String {
    let dead = if integer {
        "floor_div(scalar_to_tensor(7i32), cast(mul(&x, scalar_to_tensor(0.0f32)), i32))"
    } else {
        "cast(mul(&x, scalar_to_tensor(1e30f32)), i32)"
    };
    format!(
        "def f(x: tensor[f32]) -> tensor[f32] = {{\n  dead = {dead}\n  mul(&x, &x)\n}}\n\ndef h(xs: tensor[1, f32]) -> tensor[1, f32] = {{\n  s = tensor_to_scalar(sum(copy(xs), 0i32))\n  if {condition} then vmap(grad(f))(xs) else xs\n}}\n\ndef selected(xs: tensor[1, f32]) -> tensor[1, f32] = h(xs)\n"
    )
}

/// The `vmap(grad(...))` call site (spec/10 §3): the differentiated,
/// vmapped body spliced into an arm runs under the arm's activation, so
/// when the arm is not taken no row checks, for a float `cast` and for an
/// integer `floor_div`, in the DAG evaluator and the selected C entry. An
/// `lt` condition keeps `selected` in the Tensor lane and a `gt` one puts
/// it in the Host lane, whose `h` lowers the same `if`; both are asserted.
/// The taken twins trap in both lanes.
///
/// Evidentiary status: REGRESSION TEST for the untaken rows. At 224414e1f
/// every untaken row traps with its taken twin's trap, in both lanes. The
/// taken rows are a disposition lock (green at 224414e1f).
#[test]
fn a_vmap_of_grad_in_an_untaken_arm_checks_nothing_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    for (lane, untaken, taken) in [
        (Lane::Tensor, "lt(0.0f32, s)", "lt(s, 0.0f32)"),
        (Lane::Host, "gt(s, 0.0f32)", "gt(s, -5.0f32)"),
    ] {
        // The C runtime words an integer division by zero its own way.
        for (kind, integer, trap, c_trap) in [
            ("cast", false, CAST_OVERFLOW, CAST_OVERFLOW),
            ("floor_div", true, FLOOR_DIV_ZERO, "division"),
        ] {
            let untaken = vmap_grad_in_arm(untaken, integer);
            let taken = vmap_grad_in_arm(taken, integer);
            rows.returns(
                &format!("E untaken arm, {kind}, {lane:?} lane"),
                select(&untaken, "selected", xs(&[-3.0])),
                "selected",
                &[-3.0],
                Some(lane),
            );
            rows.c(
                &format!("C untaken arm, {kind}, {lane:?} lane"),
                run_c(&compile_c(&untaken, "selected"), "selected", 0),
                Ok(()),
            );
            rows.traps(
                &format!("E taken arm, {kind}, {lane:?} lane"),
                select(&taken, "selected", xs(&[-3.0])),
                trap,
            );
            rows.c(
                &format!("C taken arm, {kind}, {lane:?} lane"),
                run_c(&compile_c(&taken, "selected"), "selected", 0),
                Err(c_trap),
            );
        }
    }
    rows.assert_empty();
}

/// An integer chain the fusion pass would join, `add(mul(d, d), d)`, in a
/// runtime `if` arm: `d = 100000` overflows `mul` at `i32`.
fn integer_chain_arm(condition: &str) -> String {
    format!(
        "def selected(x: tensor[1, f32], d: tensor[1, i32]) -> tensor[i32] = {{\n  s = tensor_to_scalar(sum(&d, 0i32))\n  r = if {condition} then add(mul(&d, &d), &d) else d\n  sum(r, 0i32)\n}}\n"
    )
}

/// The fusion pass (spec/10 §3): a fused kernel runs under one owner, and a
/// checking operation under an activation keeps its own kernel, so an
/// untaken arm's integer chain checks nothing. The taken twin traps.
///
/// Evidentiary status: REGRESSION TEST. At eb608d063 the untaken arm's
/// `mul` traps in both lanes.
#[test]
fn an_untaken_arms_integer_chain_checks_nothing_in_the_evaluator_and_c() {
    let bindings = || {
        let mut bindings = x1();
        bindings.insert("d".into(), i32_input(100_000));
        bindings
    };
    let mut rows = Rows::default();
    let untaken = integer_chain_arm("gt(s, 200000i32)");
    rows.returns(
        "E untaken arm",
        select(&untaken, "selected", bindings()),
        "selected",
        &[100_000.0],
        None,
    );
    rows.c(
        "C untaken arm",
        run_c(&compile_c(&untaken, "selected"), "selected", 100_000),
        Ok(()),
    );
    let taken = integer_chain_arm("gt(s, 0i32)");
    rows.traps(
        "E taken arm",
        select(&taken, "selected", bindings()),
        MUL_OVERFLOW,
    );
    rows.c(
        "C taken arm",
        run_c(&compile_c(&taken, "selected"), "selected", 100_000),
        Err(MUL_OVERFLOW),
    );
    rows.assert_empty();
}

const SUM_OVERFLOW: &str = "numeric trap: overflow in sum at i32";

/// One kind of check decisions section 11 gates (spec/10 section 3.2): its
/// trapping instance in the `then` arm of `selected`'s runtime `if`, whose
/// `{c}` is the condition. `x` holds `n` ones for the evaluator and the C
/// driver's `2i - 3`, whose sums (1: -3, 4: 0, 6: 12) stay under 100 like
/// the evaluator's, so each condition decides the same arm in both lanes. An
/// `lt` condition keeps `selected` a Tensor-lane root and a C kernel unless
/// something else in it routes it to the Host lane: a runtime list (the
/// bounds of `shrink` and `pad`, a computed `reshape` target) or a `fail`
/// message string. An empty reduced axis is a `tensor[0, f32]` input, since
/// the C lane refuses a runtime-sized `insert` (chelis#600); a runtime
/// integer is the rank-0 `d`.
struct GatedKind {
    name: &'static str,
    source: &'static str,
    n: usize,
    /// The rank-0 `i64` input `d`, where the source declares one.
    d: Option<i64>,
    /// The lane `eval_selected` runs `selected` in. A Host-lane row runs
    /// the `if` as control flow, so its untaken arm never reaches the gate;
    /// [`every_gated_kind_checks_only_in_a_taken_arm_in_the_dag_evaluator`]
    /// drives the DAG evaluator for every kind.
    lane: Lane,
    /// `selected` when the arm is not taken, in the evaluator.
    expected: &'static [f64],
    /// The evaluator's typed trap when it is.
    trap: &'static str,
    /// A fragment of the C lane's trap, or `None` where the selected C
    /// entry refuses the source: a runtime-extent result (a runtime-bounded
    /// movement, a computed reshape target) has no C representation in a
    /// Tensor-lane entry (chelis#600). `exec_compile`'s
    /// `a_gated_movement_or_extent_claim_checks_only_where_its_activation_holds_in_eval_and_c`
    /// covers those kinds' C gates on hand-built graphs.
    c_trap: Option<&'static str>,
}

const TAKEN: &str = "lt(s, 100.0f32)";
const UNTAKEN: &str = "lt(100.0f32, s)";

/// Every kind the evaluator and the C lane now gate beyond operand values:
/// an integer reduction (consumed and discarded), an empty reduced axis
/// (`max_reduce`, `argmax_reduce`), runtime movement bounds (`shrink`,
/// `stride`, `pad`), a call's named extent claim, a callee's literal and
/// named result claims, and a guarded abort in a `grad` body spliced into
/// the arm. The literal result claim is `tensor[2, 2, f32]` over a
/// `reshape` whose target the tensor-entry lowering folds to 3, so it is a
/// result claim, not a `CheckedReshapeExtent`.
const GATED_KINDS: [GatedKind; 11] = [
    GatedKind {
        name: "consumed integer sum",
        source: "def selected(x: tensor[1, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then cast(sum(to_tensor([2000000000i32, 2000000000i32]), 0i32), f32) else sum(x, 0i32)\n}\n",
        n: 1,
        d: None,
        lane: Lane::Tensor,
        expected: &[1.0],
        trap: SUM_OVERFLOW,
        c_trap: Some(SUM_OVERFLOW),
    },
    GatedKind {
        name: "discarded integer sum",
        source: "def selected(x: tensor[1, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  r = if {c} then {\n    dead = sum(to_tensor([2000000000i32, 2000000000i32]), 0i32)\n    sum(&x, 0i32)\n  } else sum(&x, 0i32)\n  sum(x, 0i32)\n}\n",
        n: 1,
        d: None,
        lane: Lane::Tensor,
        expected: &[1.0],
        trap: SUM_OVERFLOW,
        c_trap: Some(SUM_OVERFLOW),
    },
    GatedKind {
        name: "empty max_reduce",
        source: "def selected(x: tensor[1, f32], e: tensor[0, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then max_reduce(e, 0i32) else sum(x, 0i32)\n}\n",
        n: 1,
        d: None,
        lane: Lane::Tensor,
        expected: &[1.0],
        trap: "numeric trap: domain in max_reduce at f32",
        c_trap: Some("numeric trap: domain in max_reduce at f32"),
    },
    GatedKind {
        name: "empty argmax_reduce",
        source: "def selected(x: tensor[1, f32], e: tensor[0, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then cast(argmax_reduce(e, 0i32), f32) else sum(x, 0i32)\n}\n",
        n: 1,
        d: None,
        lane: Lane::Tensor,
        expected: &[1.0],
        trap: "numeric trap: domain in argmax_reduce at i64",
        c_trap: Some("numeric trap: domain in argmax_reduce at i64"),
    },
    GatedKind {
        name: "shrink past the end",
        source: "def selected(x: tensor[4, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then sum(shrink(&x, [[1i64, add(shape(&x, 0i32), 3i64)]]), 0i32) else sum(x, 0i32)\n}\n",
        n: 4,
        d: None,
        lane: Lane::Host,
        expected: &[4.0],
        trap: "numeric trap: domain in shrink at i64",
        c_trap: None,
    },
    GatedKind {
        name: "stride of zero",
        source: "def selected(x: tensor[4, f32], d: tensor[i64]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then sum(stride(&x, tensor_to_scalar(d)), 0i32) else sum(x, 0i32)\n}\n",
        n: 4,
        d: Some(0),
        lane: Lane::Tensor,
        expected: &[4.0],
        trap: "numeric trap: domain in stride at i64",
        c_trap: None,
    },
    GatedKind {
        name: "negative pad",
        source: "def selected(x: tensor[4, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then sum(pad(&x, [[sub(shape(&x, 0i32), 5i64), 0i64]], 0.0f32), 0i32) else sum(x, 0i32)\n}\n",
        n: 4,
        d: None,
        lane: Lane::Host,
        expected: &[4.0],
        trap: "must be a non-negative integer",
        c_trap: None,
    },
    GatedKind {
        name: "call's named extent claim",
        source: "def g[n](a: tensor[n, f32], b: tensor[n, f32]) -> tensor[f32] = sum(a, 0i32)\n\ndef selected(x: tensor[4, f32], d: tensor[i64]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then g(stride(&x, tensor_to_scalar(d)), copy(x)) else sum(x, 0i32)\n}\n",
        n: 4,
        d: Some(2),
        lane: Lane::Tensor,
        expected: &[4.0],
        trap: "numeric trap: domain in load at i64",
        c_trap: None,
    },
    GatedKind {
        name: "callee's literal result claim",
        source: "def g[n](y: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(y, [floor_div(shape(y, 0i32), 2i64), 2i64])\n\ndef selected(x: tensor[6, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then sum(sum(g(copy(x)), 0i32), 0i32) else sum(x, 0i32)\n}\n",
        n: 6,
        d: None,
        lane: Lane::Host,
        expected: &[6.0],
        trap: "numeric trap: domain in reshape at i64",
        c_trap: None,
    },
    GatedKind {
        name: "callee's named result claim",
        source: "def g[n](x: tensor[n, f32]) -> tensor[n, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\ndef selected(x: tensor[4, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if {c} then sum(g(copy(x)), 0i32) else sum(x, 0i32)\n}\n",
        n: 4,
        d: None,
        lane: Lane::Host,
        expected: &[4.0],
        trap: "numeric trap: domain in shrink at i64",
        c_trap: None,
    },
    GatedKind {
        name: "guarded fail in a grad body",
        source: "def loss(x: tensor[1, f32]) -> tensor[f32] = if lt(tensor_to_scalar(sum(&x, 0i32)), 50.0f32) then fail(\"guard tripped\") else sum(x, 0i32)\n\ndef h(x: tensor[1, f32]) -> tensor[1, f32] = {\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  if {c} then grad(loss)(x) else x\n}\n\ndef selected(x: tensor[1, f32]) -> tensor[1, f32] = h(x)\n",
        n: 1,
        d: None,
        lane: Lane::Host,
        expected: &[1.0],
        trap: "guard tripped",
        c_trap: Some("guard tripped"),
    },
];

impl GatedKind {
    fn source(&self, condition: &str) -> String {
        self.source.replace("{c}", condition)
    }

    /// `x`, the empty `e` and the rank-0 `d` the source declares.
    fn bindings(&self) -> BTreeMap<String, TensorValue> {
        let mut bindings = BTreeMap::from([("x".into(), f32_input(self.n))]);
        if self.source.contains("e: tensor[0, f32]") {
            bindings.insert("e".into(), f32_input(0));
        }
        if let Some(d) = self.d {
            bindings.insert(
                "d".into(),
                TensorValue {
                    shape: vec![],
                    data: serde_json::from_value(
                        serde_json::json!({"dtype": "int64", "values": [d]}),
                    )
                    .unwrap(),
                },
            );
        }
        bindings
    }

    /// [`Self::bindings`] as the DAG evaluator's inputs.
    fn dag_inputs(&self) -> UnordMap<String, DagTensor> {
        let f32s = |count: usize| {
            DagTensor::from_storage(
                vec![count],
                finalize_tensor("x", Prim::F32, RawTensor::Float(vec![1.0; count])).unwrap(),
            )
        };
        let mut inputs = UnordMap::new();
        inputs.insert("x".to_string(), f32s(self.n));
        if self.source.contains("e: tensor[0, f32]") {
            inputs.insert("e".to_string(), f32s(0));
        }
        if let Some(d) = self.d {
            inputs.insert(
                "d".to_string(),
                DagTensor::from_storage(
                    vec![],
                    finalize_tensor("d", Prim::Int64, RawTensor::Int(vec![d])).unwrap(),
                ),
            );
        }
        inputs
    }
}

/// `selected` of `source` lowered as a tensor entry (the lowering the host
/// lane's kernels and `--target hip` use), whatever lane the router would
/// give it, and evaluated by the DAG evaluator: the root's values, or the
/// evaluator's error.
fn dag_evaluator(source: &str, inputs: &UnordMap<String, DagTensor>) -> Result<Vec<f64>, String> {
    let declarations = chelis_surf::parser::parse_str(source).expect("parse");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("desugar");
    let checked = chelis_types::check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("check: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects");
    let checked = chelis_types::check_linearity(&checked).expect("linearity");
    let dag = chelis_ir::host::lower_named_tensor_entry_dag(&checked, "selected")
        .unwrap_or_else(|| panic!("`selected` lowers as a tensor entry:\n{source}"));
    let root = *dag.roots().last().expect("a root");
    let values = chelis_ir::eval::eval_tensor(&dag, inputs)?;
    Ok(values[&root].to_f64_lossy_vec())
}

/// `outcome` must fail with a message containing `trap`.
fn fails_with(rows: &mut Rows, row: &str, outcome: Result<EvalResult, CompilerError>, trap: &str) {
    match outcome {
        Ok(result) => rows.0.push(format!("{row}: returned {:?}", result.roots)),
        Err(error)
            if error
                .errors
                .iter()
                .any(|error| error.message.contains(trap)) => {}
        Err(error) => rows.0.push(format!(
            "{row}: expected `{trap}`, got {:?}",
            error.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
        )),
    }
}

/// Decisions section 11 for the kinds beyond operand values: in an untaken
/// arm each checks nothing, through `eval_selected` and the selected C entry,
/// and the arm's value is the untaken twin's (`lower_if`, and the `grad`
/// splice for the guarded abort). The lanes agree row by row.
///
/// Each row's lane is pinned ([`GatedKind::lane`]): the Tensor-lane rows
/// reach the DAG evaluator and a Tensor-lane C kernel, whose `if` is a
/// `Where` under activations; a Host-lane row runs its `if` as control flow.
///
/// Evidentiary status: per row, see the report of ks5-h2e (REGRESSION TEST
/// where the row fails at 224414e1f, DISPOSITION LOCK where it passes).
#[test]
fn an_untaken_arms_extent_bound_and_reduction_checks_do_nothing_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    for kind in &GATED_KINDS {
        let source = kind.source(UNTAKEN);
        rows.returns(
            &format!("E {}", kind.name),
            select(&source, "selected", kind.bindings()),
            "selected",
            kind.expected,
            Some(kind.lane),
        );
        if kind.c_trap.is_some() {
            rows.c(
                &format!("C {}", kind.name),
                run_c(&compile_c(&source, "selected"), "selected", 0),
                Ok(()),
            );
        }
    }
    rows.assert_empty();
}

/// The positive control: taken, each kind traps in both lanes, so the
/// untaken rows cannot pass by dropping the check.
///
/// Evidentiary status: DISPOSITION LOCK (each row traps at 224414e1f).
#[test]
fn a_taken_arms_extent_bound_and_reduction_checks_trap_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    for kind in &GATED_KINDS {
        let source = kind.source(TAKEN);
        fails_with(
            &mut rows,
            &format!("E {}", kind.name),
            select(&source, "selected", kind.bindings()),
            kind.trap,
        );
        if let Some(c_trap) = kind.c_trap {
            rows.c(
                &format!("C {}", kind.name),
                run_c(&compile_c(&source, "selected"), "selected", 0),
                Err(c_trap),
            );
        }
    }
    rows.assert_empty();
}

/// The DAG evaluator for every gated kind, the Host-lane ones included:
/// `selected` lowered as a tensor entry (its `if` a `Where` under the arm's
/// activation, `lower_if`, and the `grad` splice for the guarded abort) and
/// evaluated directly. Untaken, each kind returns the `else` value; taken,
/// it traps with the kind's trap. `chelis-cli`'s
/// `issue_2563_untaken_arm_eval_file` has the `chelis eval --file` rows.
///
/// Evidentiary status: at 224414e1f the untaken rows of every kind but the
/// discarded integer sum and the negative pad trap (REGRESSION TEST), and
/// the taken discarded integer sum returns 1.0 (REGRESSION TEST: it was not
/// a seed); the other rows are a disposition lock. The two result-claim
/// rows were added by ks5-h2f: their untaken rows trap at eaa5f3306 and
/// 224414e1f, where the result-claim guard ignored the arm's activation
/// (REGRESSION TEST).
#[test]
fn every_gated_kind_checks_only_in_a_taken_arm_in_the_dag_evaluator() {
    let mut failures = Vec::new();
    for kind in &GATED_KINDS {
        let inputs = kind.dag_inputs();
        match dag_evaluator(&kind.source(UNTAKEN), &inputs) {
            Ok(values) if values == kind.expected => {}
            outcome => failures.push(format!("untaken {}: {outcome:?}", kind.name)),
        }
        match dag_evaluator(&kind.source(TAKEN), &inputs) {
            Err(error) if error.contains(kind.trap) => {}
            outcome => failures.push(format!("taken {}: {outcome:?}", kind.name)),
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A vmapped row's arm over `rows` rows whose checking node is an integer
/// `sum`: the row scaled by 7e8 cast to `i32`, inserted twice along a new
/// axis and summed. Row -3.0 overflows the sum (and not the cast); rows
/// -1.0 and 1.0 fit. A `gt` condition keeps the entry in the Host lane,
/// whose C entry the driver runs on the one row `[-3.0]`.
fn vmapped_sum_arm(condition: &str, rows: usize) -> String {
    format!(
        "def row(x: tensor[f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(copy(x))\n  if {condition} then cast(sum(insert(cast(mul(&x, scalar_to_tensor(700000000.0f32)), i32), 0i32, 2i64), 0i32), f32) else x\n}}\n\ndef selected(xs: tensor[{rows}, f32]) -> tensor[{rows}, f32] = vmap(row)(xs)\n"
    )
}

/// The vmap batching pass for an integer reduction (spec/10 §3.2): each row
/// of a vmapped arm's `sum` checks only where that row takes the arm. Row
/// -3.0 does not take it and would overflow; rows -1.0 and 1.0 take it and
/// sum to -1.4e9 and 1.4e9, in the evaluator; the C entry runs the one row
/// -3.0 and returns its `else` value. The taken twin (row -3.0 taking the
/// arm) traps in both lanes.
///
/// Evidentiary status: DISPOSITION LOCK (every row passes at 224414e1f).
#[test]
fn a_vmapped_arms_integer_sum_checks_only_in_the_rows_that_take_it_in_the_evaluator_and_c() {
    let mut rows = Rows::default();
    let untaken = vmapped_sum_arm("gt(s, -2.0f32)", 3);
    rows.returns(
        "E untaken row",
        select(&untaken, "selected", xs(&[-3.0, -1.0, 1.0])),
        "selected",
        &[-3.0, -1_400_000_000.0, 1_400_000_000.0],
        None,
    );
    rows.c(
        "C untaken row",
        run_c(
            &compile_c(&vmapped_sum_arm("gt(s, -2.0f32)", 1), "selected"),
            "selected",
            0,
        ),
        Ok(()),
    );
    let taken = "gt(-2.0f32, s)";
    rows.traps(
        "E taken row",
        select(
            &vmapped_sum_arm(taken, 3),
            "selected",
            xs(&[-3.0, -1.0, 1.0]),
        ),
        SUM_OVERFLOW,
    );
    rows.c(
        "C taken row",
        run_c(
            &compile_c(&vmapped_sum_arm(taken, 1), "selected"),
            "selected",
            0,
        ),
        Err(SUM_OVERFLOW),
    );
    rows.assert_empty();
}

/// Every per-row activation gate in `source`'s C: the gated node's id, the
/// activation tensor it reads, and its row loop's text.
fn per_row_gates(source: &str) -> Vec<(String, String, String)> {
    let mut gates = Vec::new();
    for line in source.lines() {
        let Some(rest) = line.trim().strip_prefix("const int __act_row_") else {
            continue;
        };
        let (id, rest) = rest.split_once(" = (((const uint8_t*)").expect(line);
        let (mask, rest) = rest.split_once("_data)[").expect(line);
        assert_eq!(rest, format!("__row_{id}] != 0);"), "{line}");
        let header = format!("for (int64_t __row_{id} = 0;");
        let start = source.find(&header).expect(&header);
        let mut depth = 0usize;
        let mut end = start;
        for (offset, character) in source[start..].char_indices() {
            match character {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + offset + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        gates.push((
            id.to_string(),
            mask.to_string(),
            source[start..end].to_string(),
        ));
    }
    gates
}

/// One activation read per row (decisions section 11): under `vmap` a
/// checking node's activation is one bool per row, and its C kernel reads
/// each row's byte once, in its row loop, never once per element. For the
/// vmapped `cast` and integer `sum` arms, every per-row gate's loop reads
/// the activation tensor exactly once (`[__row_N]`), and every element it
/// checks reads `__act_row_N` instead. The only other read of the mask for
/// that node is the any-row reduction, also one byte per row.
///
/// Evidentiary status: REGRESSION TEST. At 224414e1f the C lane has no
/// per-row gate for these nodes (no `__act_row_` in either kernel).
#[test]
fn a_vmapped_arms_c_kernel_reads_each_rows_activation_once() {
    for (kind, source, checked) in [
        (
            "cast",
            vmapped_arm("gt(s, -2.0f32)", 3),
            "(int32_t)((int64_t)((double)(((__act_row_",
        ),
        (
            "integer sum",
            vmapped_sum_arm("gt(s, -2.0f32)", 3),
            "__sum_level_",
        ),
    ] {
        let artifact = compile_c(&source, "selected");
        let stem = &artifact.compile_result.entry_name;
        let c = &artifact
            .compile_result
            .files
            .iter()
            .find(|file| file.path == format!("{stem}.c"))
            .expect("generated C")
            .contents;
        let gates = per_row_gates(c);
        assert!(
            gates.iter().any(|(_, _, body)| body.contains(checked)),
            "{kind}: no per-row gate around its check:\n{c}"
        );
        for (id, mask, body) in &gates {
            let read = format!("{mask}_data)[");
            assert_eq!(
                body.matches(&read).count(),
                1,
                "{kind}: node {id}'s row loop reads the mask once per row:\n{body}"
            );
            let (row_read, elements) = body
                .split_once(&format!("_data)[__row_{id}] != 0);"))
                .expect("the row read opens the row loop");
            assert!(row_read.ends_with(mask), "{kind}: {body}");
            assert!(
                elements.contains(&format!("__act_row_{id}")),
                "{kind}: node {id}'s elements read the row's flag:\n{body}"
            );
            assert_eq!(
                c.matches(&format!(
                    "__act_{id} |= (((const uint8_t*){read}__r] != 0);"
                ))
                .count(),
                1,
                "{kind}: node {id}'s any-row reduction:\n{c}"
            );
        }
    }
}
