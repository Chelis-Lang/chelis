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
use chelis_types::types::Lane;
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

/// (a) A taken arm's dead reference to a trapping value declaration enters
/// it, so its initializer traps: in the DAG evaluator (the value beside
/// the selected def) and in the in-context C entry (the value in a
/// dependency). The valid twin returns `x` from the Tensor lane.
///
/// Evidentiary status: DISPOSITION LOCK. Both lanes trapped at b003b1610;
/// the reference is recorded by lowering and entered unconditionally.
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
/// Evidentiary status: RED at b003b1610, left red: the host runs the body
/// as a DAG kernel, which never demands the dead value, and initializes
/// beforehand only the values the body reaches on every path
/// (`ProgramScope::reached_by_call` skips `if` and `match` arms), so the
/// taken arm's value is never initialized. Closing it needs the arm's
/// activation on the reference, which is held for phase 2.
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
/// Evidentiary status: HELD RED at b003b1610 in both lanes: lowering records
/// the arm's reference without its activation, so the value is entered
/// unconditionally and traps. The expectation is Rule D's; the fix is the
/// activation-carrying reference (phase 2).
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
/// Evidentiary status: DISPOSITION LOCK (green at b003b1610, for the reason
/// the taken-arm host test is red); it keeps the phase-2 fix of that test
/// from initializing an untaken arm's value.
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
        run_c(&compile_c(&vmapped_arm(untaken, 1), "selected"), "selected", 0),
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
        run_c(&compile_c(&vmapped_arm(taken, 1), "selected"), "selected", 0),
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
