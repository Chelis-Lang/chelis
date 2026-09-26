//! A local tensor ascription's claim (spec/04 §4.7) is checked only where the
//! ascription's `let` executes: spec/10 §3.2's "a node whose activation is
//! false checks nothing" (decisions section 11, #2413, #2440).
//!
//! The claim's activation has one carrier, the owner activation of the node
//! that carries the claim. Inside a `grad` or `vmap` body spliced into a
//! runtime `if` arm that owner activation is the call site's, while the
//! enclosing arm predicates the body's own lowering sees are none, so a claim
//! gated by the branch path alone checked in an untaken arm. Under `vmap` an
//! arm's activation is per row, and a claim about an extent every row shares
//! checks when any row takes the arm.
//!
//! The lanes: E is `eval_selected`, whose DAG evaluator runs the kernel that
//! holds each runtime `if`, computing both arms; C is the selected C entry,
//! compiled and run natively on the same inputs. The compiled lane refuses a
//! kernel over the runtime extent `*` (chelis#600), so its rows use the
//! per-row shapes, whose extent is a `pad` by a runtime count. Each row is
//! collected before the assertion, so one red row never hides a sibling.
#![allow(deprecated)]
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

mod ownership_support;

use chelis_compiler_api::compiler::{
    CompiledExecutionArtifact, CompilerError, compile_for_execution, eval_selected,
};
use chelis_compiler_api::schema::{
    CompileRequest, CompileTarget, EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use chelis_types::types::Lane;
use std::collections::BTreeMap;

/// The local claim's trap line ([04-NUM-9]); its context line names the claim.
const PAD_CLAIM: &str = "numeric trap: domain in pad at i64";
const CLAIM_CONTEXT: &str = "extent `3`: claimed = 3, pad axis";

/// One named input: its shape and its row-major `f32` data.
struct Input {
    name: &'static str,
    shape: Vec<i64>,
    data: Vec<f32>,
}

const ROW: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];

fn x() -> Vec<Input> {
    vec![Input {
        name: "x",
        shape: vec![4],
        data: ROW.to_vec(),
    }]
}

/// `xs`, one row per entry of `rows`.
fn xs(rows: &[[f32; 4]]) -> Vec<Input> {
    vec![Input {
        name: "xs",
        shape: vec![rows.len() as i64, 4],
        data: rows.iter().flatten().copied().collect(),
    }]
}

fn scalar(name: &'static str, value: f32) -> Vec<Input> {
    vec![Input {
        name,
        shape: Vec::new(),
        data: vec![value],
    }]
}

fn bindings(inputs: &[Input]) -> BTreeMap<String, TensorValue> {
    inputs
        .iter()
        .map(|input| {
            (
                input.name.to_string(),
                TensorValue {
                    shape: input.shape.clone(),
                    data: wire_values::storage_f32(input.data.clone()),
                },
            )
        })
        .collect()
}

fn select(source: &str, inputs: &[Input]) -> Result<EvalResult, CompilerError> {
    eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            bindings: bindings(inputs),
        },
        &["selected".into()],
    )
}

fn values(result: &EvalResult) -> Vec<f64> {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("selected"))
        .unwrap_or_else(|| panic!("no root `selected`: {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => (0..value.data.len())
            .map(|index| value.data.element_f64_lossy(index))
            .collect(),
        ExecutionValue::Scalar { value } => vec![value.get().as_f64_lossy()],
        other => panic!("{other:?}"),
    }
}

fn lane(result: &EvalResult) -> Option<Lane> {
    result
        .manifest
        .entries
        .iter()
        .find(|entry| entry.name == "selected")
        .map(|entry| entry.lane)
}

fn compile_c(source: &str) -> CompiledExecutionArtifact {
    compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some("selected".into()),
    })
    .unwrap_or_else(|error| panic!("{source}\n{error:?}"))
}

/// Run the selected C entry natively on `inputs`, in order. `Ok` holds the
/// `f32` output's elements; `Err` holds stderr when the program aborts.
fn run_c(artifact: &CompiledExecutionArtifact, inputs: &[Input]) -> Result<Vec<f64>, String> {
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
        "`selected` compiled as a whole program ({:?}), not as its selected entry",
        artifact.entry_lane_decline
    );
    let mut driver = String::from(
        "static chelis_tensor *make_f32(int64_t rank, const int64_t *shape, const float *values, int64_t n) {\n    chelis_tensor *x = chelis_alloc(rank, shape, CHELIS_DTYPE_F32);\n    chelis_tensor_write *guard = chelis_tensor_begin_write(x);\n    chelis_write_view view = chelis_tensor_write_view(guard);\n    for (int64_t i = 0; i < n; ++i) ((float *)view.data)[i] = values[i];\n    chelis_tensor_end_write(guard);\n    return x;\n}\nstatic void print_f32(const chelis_tensor *x) {\n    chelis_read_view view = chelis_tensor_read_view(x);\n    assert(view.dtype == CHELIS_DTYPE_F32);\n    for (int64_t i = 0; i < (int64_t)view.count; ++i) printf(\"%.9g\\n\", (double)((const float *)view.data)[i]);\n}\nint main(void) {\n",
    );
    let mut names = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let shape = input
            .shape
            .iter()
            .map(i64::to_string)
            .chain(std::iter::once("0".into()))
            .collect::<Vec<_>>()
            .join(", ");
        let data = input
            .data
            .iter()
            .map(|value| format!("{value:?}f"))
            .collect::<Vec<_>>()
            .join(", ");
        driver.push_str(&format!(
            "    const int64_t shape{index}[] = {{{shape}}};\n    const float data{index}[] = {{{data}}};\n    chelis_tensor *in{index} = make_f32({}, shape{index}, data{index}, {});\n",
            input.shape.len(),
            input.data.len()
        ));
        names.push(format!("in{index}"));
    }
    if program.header().contains("chelis_fn_") {
        // A Host-lane entry is one C function taking its tensors in order.
        driver.push_str(&format!(
            "    chelis_tensor *out = {}({});\n    print_f32(out);\n    chelis_tensor_release(out);\n",
            program.symbol("selected"),
            names.join(", ")
        ));
    } else {
        let outputs = artifact.outputs.len();
        driver.push_str(&format!(
            "    chelis_tensor *inputs[] = {{{}}};\n    chelis_tensor *outputs[{}] = {{NULL}};\n    {}(inputs, {}, outputs, {outputs});\n    print_f32(outputs[0]);\n    for (int i = 0; i < {outputs}; ++i) chelis_tensor_release(outputs[i]);\n",
            names.join(", "),
            outputs.max(1),
            artifact.host_entry_name,
            names.len()
        ));
    }
    for name in &names {
        driver.push_str(&format!("    chelis_tensor_release({name});\n"));
    }
    driver.push_str("    return 0;\n}\n");
    match std::panic::catch_unwind(|| ownership_support::run_failure_stderr(&program, &driver)) {
        Ok(stderr) => Err(stderr),
        Err(_) => {
            let (summary, stdout) = ownership_support::run_with_stdout(&program, &driver);
            ownership_support::balanced(&summary);
            Ok(stdout
                .lines()
                .map(|line| line.parse::<f64>().expect("printed f32"))
                .collect())
        }
    }
}

/// Collects each row's disagreement with its expectation.
#[derive(Default)]
struct Rows(Vec<String>);

impl Rows {
    /// E must return `expected` for `selected`, from `expected_lane` when named.
    fn e_returns(
        &mut self,
        row: &str,
        outcome: Result<EvalResult, CompilerError>,
        expected: &[f64],
        expected_lane: Option<Lane>,
    ) {
        match outcome {
            Ok(result) => {
                if values(&result) != expected {
                    self.0
                        .push(format!("E {row}: returned {:?}", values(&result)));
                }
                if let Some(expected_lane) = expected_lane
                    && lane(&result) != Some(expected_lane)
                {
                    self.0.push(format!(
                        "E {row}: ran in {:?}, not {expected_lane:?}",
                        lane(&result)
                    ));
                }
            }
            Err(error) => self.0.push(format!(
                "E {row}: {:?}",
                error.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            )),
        }
    }

    /// E must trap with a message containing `trap`.
    fn e_traps(&mut self, row: &str, outcome: Result<EvalResult, CompilerError>, trap: &str) {
        match outcome {
            Ok(result) => self
                .0
                .push(format!("E {row}: returned {:?}", values(&result))),
            Err(error) if error.errors.iter().any(|e| e.message.contains(trap)) => {}
            Err(error) => self.0.push(format!(
                "E {row}: expected `{trap}`, got {:?}",
                error.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            )),
        }
    }

    /// C must end as `expected`: `Ok(values)` returned, `Err(text)` aborted
    /// with stderr containing `text`.
    fn c(&mut self, row: &str, source: &str, inputs: &[Input], expected: Result<&[f64], &str>) {
        let outcome = run_c(&compile_c(source), inputs);
        match (&outcome, expected) {
            (Ok(values), Ok(expected)) if values == expected => {}
            (Err(stderr), Err(trap)) if stderr.contains(trap) => {}
            _ => self
                .0
                .push(format!("C {row}: ended {outcome:?}, expected {expected:?}")),
        }
    }

    fn assert_empty(self) {
        assert!(self.0.is_empty(), "\n{}", self.0.join("\n"));
    }
}

/// An arm's condition over `s = sum(x) = 0`, `(untaken, taken)`. The
/// selected root is a Host-lane one, and the evaluator runs the `g` or
/// `selected` it calls as a tensor kernel, lowering the `if` into both arms
/// and a `Where` (at 224414e1f the untaken rows trap there).
const CONDITION: (&str, &str) = ("lt(0.5f32, s)", "lt(s, 0.5f32)");

/// `selected` calls `g`, whose runtime `if` arm applies `grad(h)` to `x`;
/// `h` ascribes the extent-4 `pad` of `x` as extent 3. The compiled lane
/// refuses a Tensor-lane entry over the runtime extent `*` (chelis#600), so
/// the C rows of this shape are the per-row ones below.
fn grad_in_arm(condition: &str) -> String {
    format!(
        "def h(x: tensor[*, f32]) -> tensor[f32] = {{\n  y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  sum(y, 0i32)\n}}\ndef g(x: tensor[*, f32]) -> tensor[*, f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  if {condition} then grad(h)(x) else x\n}}\ndef selected(x: tensor[*, f32]) -> tensor[*, f32] = g(x)\n"
    )
}

/// The `vmap` twin of [`grad_in_arm`]: the arm maps `r`, which holds the
/// same ascription, over two copies of `x`.
fn vmap_in_arm(condition: &str) -> String {
    format!(
        "def r(x: tensor[*, f32]) -> tensor[f32] = {{\n  y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  sum(y, 0i32)\n}}\ndef g(xs: tensor[2, *, f32]) -> tensor[2, f32] = {{\n  s = tensor_to_scalar(sum(sum(copy(xs), 0i32), 0i32))\n  if {condition} then vmap(r)(xs) else sum(xs, 1i32)\n}}\ndef selected(xs: tensor[2, *, f32]) -> tensor[2, f32] = g(xs)\n"
    )
}

/// A runtime `if` arm ascribing a `pad` of `x`; `before` produces it ahead
/// of the arm, otherwise the arm produces it.
fn ascription_in_arm(condition: &str, before: bool) -> String {
    let (produced, ascribed) = if before {
        ("  z = pad(&x, [[0i64, 0i64]], 0.0f32)\n", "z")
    } else {
        ("", "pad(&x, [[0i64, 0i64]], 0.0f32)")
    };
    format!(
        "def selected(x: tensor[*, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n{produced}  if {condition} then {{\n    y: tensor[3, f32] = {ascribed}\n    sum(y, 0i32)\n  }} else sum(x, 0i32)\n}}\n"
    )
}

/// A `grad` or `vmap` body spliced into an untaken arm checks no local
/// ascription it holds, in the evaluator (a rank-0 activation: the call
/// site's). An arm's ascription of a value produced before the arm checks
/// nothing either, and neither does the control whose arm produces it.
///
/// Evidentiary status: REGRESSION TEST for the `grad` and `vmap` rows (at
/// 224414e1f each traps on the claim, which carried no activation in the
/// body) and the produced-before row (at 224414e1f the guard sat at the
/// producer, before the arm's activation existed, and the evaluator failed
/// with "local extent guard activation at node N is not available").
/// DISPOSITION LOCK for the produced-in-arm control.
#[test]
fn an_untaken_arms_local_ascription_checks_nothing_in_the_evaluator() {
    let (untaken, _) = CONDITION;
    let mut rows = Rows::default();
    rows.e_returns(
        "grad body",
        select(&grad_in_arm(untaken), &x()),
        &ROW.map(f64::from),
        None,
    );
    rows.e_returns(
        "vmap body",
        select(&vmap_in_arm(untaken), &xs(&[ROW, ROW])),
        &[0.0, 0.0],
        None,
    );
    rows.e_returns(
        "produced before the arm",
        select(&ascription_in_arm(untaken, true), &x()),
        &[0.0],
        None,
    );
    rows.e_returns(
        "produced in the arm",
        select(&ascription_in_arm(untaken, false), &x()),
        &[0.0],
        None,
    );
    rows.assert_empty();
}

/// The positive control: the same ascriptions in a taken arm trap on the
/// claim, with its context line, in the evaluator.
///
/// Evidentiary status: DISPOSITION LOCK (green at 224414e1f) for the `grad`,
/// `vmap` and produced-in-arm rows; REGRESSION TEST for the produced-before
/// row, which failed with the unavailable-activation error at 224414e1f.
#[test]
fn a_taken_arms_local_ascription_traps_in_the_evaluator() {
    let (_, taken) = CONDITION;
    let mut rows = Rows::default();
    for (row, source, inputs) in [
        ("grad body", grad_in_arm(taken), x()),
        ("vmap body", vmap_in_arm(taken), xs(&[ROW, ROW])),
        (
            "produced before the arm",
            ascription_in_arm(taken, true),
            x(),
        ),
        ("produced in the arm", ascription_in_arm(taken, false), x()),
    ] {
        rows.e_traps(row, select(&source, &inputs), PAD_CLAIM);
        rows.e_traps(row, select(&source, &inputs), CLAIM_CONTEXT);
    }
    rows.assert_empty();
}

/// `p = shape(x, 0) - 4`, which is 0 in every row (the evaluator and C
/// substitute zeros for an inactive row's integer operands, which gives the
/// same 0), and `pad(x, [[0, p]])` has the runtime extent 4 the ascription
/// claims is 3. A static `x` keeps the shape inside the compiled lane.
fn padded(indent: &str) -> String {
    format!(
        "{indent}p = sub(shape(&x, 0i32), 4i64)\n{indent}y: tensor[3, f32] = pad(&x, [[0i64, p]], 0.0f32)\n"
    )
}

/// Where the ascription sits inside `vmap(r)(xs)`: in `r`'s own arm, in a
/// `grad` body the arm applies, in a `vmap` body the arm applies, or in the
/// arm over a value produced before it. A row takes the arm when its sum is
/// above `threshold`.
#[derive(Clone, Copy, Debug)]
enum PerRow {
    Arm,
    GradBody,
    VmapBody,
    ProducedBefore,
}

impl PerRow {
    const ALL: [PerRow; 4] = [
        PerRow::Arm,
        PerRow::GradBody,
        PerRow::VmapBody,
        PerRow::ProducedBefore,
    ];

    fn source(self, threshold: &str) -> String {
        match self {
            PerRow::Arm => format!(
                "def r(x: tensor[4, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  if lt({threshold}, s) then {{\n{}    sum(y, 0i32)\n  }} else sum(x, 0i32)\n}}\ndef selected(xs: tensor[2, 4, f32]) -> tensor[2, f32] = vmap(r)(xs)\n",
                padded("    ")
            ),
            PerRow::GradBody => format!(
                "def h(x: tensor[4, f32]) -> tensor[f32] = {{\n{}  sum(y, 0i32)\n}}\ndef r(x: tensor[4, f32]) -> tensor[4, f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  if lt({threshold}, s) then grad(h)(x) else x\n}}\ndef selected(xs: tensor[2, 4, f32]) -> tensor[2, 4, f32] = vmap(r)(xs)\n",
                padded("  ")
            ),
            PerRow::VmapBody => format!(
                "def q(x: tensor[4, f32]) -> tensor[f32] = {{\n{}  sum(y, 0i32)\n}}\ndef r(x: tensor[1, 4, f32]) -> tensor[1, f32] = {{\n  s = tensor_to_scalar(sum(sum(copy(x), 0i32), 0i32))\n  if lt({threshold}, s) then vmap(q)(x) else sum(x, 1i32)\n}}\ndef selected(xs: tensor[2, 1, 4, f32]) -> tensor[2, 1, f32] = vmap(r)(xs)\n",
                padded("  ")
            ),
            PerRow::ProducedBefore => format!(
                "def r(x: tensor[4, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  p = sub(shape(&x, 0i32), 4i64)\n  z = pad(&x, [[0i64, p]], 0.0f32)\n  if lt({threshold}, s) then {{\n    y: tensor[3, f32] = z\n    sum(y, 0i32)\n  }} else sum(x, 0i32)\n}}\ndef selected(xs: tensor[2, 4, f32]) -> tensor[2, f32] = vmap(r)(xs)\n"
            ),
        }
    }

    /// Row 0 sums to 0 and row 1 to 4.
    fn inputs(self) -> Vec<Input> {
        let data = ROW.iter().copied().chain([1.0; 4]).collect();
        let shape = match self {
            PerRow::VmapBody => vec![2, 1, 4],
            _ => vec![2, 4],
        };
        vec![Input {
            name: "xs",
            shape,
            data,
        }]
    }

    /// The value when no row takes the arm.
    fn untaken(self) -> Vec<f64> {
        match self {
            PerRow::GradBody => ROW.iter().copied().chain([1.0; 4]).map(f64::from).collect(),
            _ => vec![0.0, 4.0],
        }
    }
}

/// A per-row activation: under `vmap` each row has its own. The claim is
/// about an extent every row shares, so it checks when any row takes the arm
/// and not when none does, in the evaluator and C, wherever the ascription
/// sits.
///
/// Evidentiary status: REGRESSION TEST for the no-row rows (at 224414e1f
/// each traps on the claim in both lanes, or, produced before the arm,
/// fails the evaluator with the unavailable activation); DISPOSITION LOCK
/// for the rest, except that the produced-before evaluator rows are also
/// regression tests.
#[test]
fn a_vmapped_arms_local_ascription_checks_only_when_a_row_takes_the_arm() {
    let mut rows = Rows::default();
    for shape in PerRow::ALL {
        let none = shape.source("8.0f32");
        let row = format!("{shape:?}, no row");
        rows.e_returns(&row, select(&none, &shape.inputs()), &shape.untaken(), None);
        rows.c(&row, &none, &shape.inputs(), Ok(&shape.untaken()));
        for (threshold, which) in [("2.0f32", "row 1"), ("-2.0f32", "both rows")] {
            let source = shape.source(threshold);
            let row = format!("{shape:?}, {which}");
            rows.e_traps(&row, select(&source, &shape.inputs()), PAD_CLAIM);
            rows.c(&row, &source, &shape.inputs(), Err(PAD_CLAIM));
        }
    }
    rows.assert_empty();
}

/// `h` aborts on a negative input through its own runtime `if`
/// ([05-OP-68]'s guarded abort).
const ABORTS: &str = "def h(x: tensor[f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(copy(x))\n  if lt(s, 0.0f32) then fail(\"negative row\") else mul(&x, &x)\n}\n";

/// Where the abort sits: `selected` calls `g`, whose runtime `if` arm
/// applies `grad(h)` or maps `h` with `vmap` (a rank-0 activation), or
/// `selected` maps `r`, whose arm applies `grad(h)` or maps `h` (a per-row
/// activation). An arm is taken when its input's sum is above `threshold`.
#[derive(Clone, Copy, Debug)]
enum Abort {
    GradInArm,
    VmapInArm,
    GradInMappedArm,
    VmapInMappedArm,
}

impl Abort {
    const ALL: [Abort; 4] = [
        Abort::GradInArm,
        Abort::VmapInArm,
        Abort::GradInMappedArm,
        Abort::VmapInMappedArm,
    ];

    fn source(self, threshold: &str) -> String {
        let arm = |ty: &str, sum: &str, apply: &str| {
            format!(
                "  s = tensor_to_scalar({sum})\n  if lt({threshold}, s) then {apply} else x\n}}\n"
            ) + &format!("def selected(xs: {ty}) -> {ty} = ")
        };
        match self {
            Abort::GradInArm => format!(
                "{ABORTS}def g(x: tensor[f32]) -> tensor[f32] = {{\n{}g(xs)\n",
                arm("tensor[f32]", "copy(x)", "grad(h)(x)")
            ),
            Abort::VmapInArm => format!(
                "{ABORTS}def g(x: tensor[2, f32]) -> tensor[2, f32] = {{\n{}g(xs)\n",
                arm("tensor[2, f32]", "sum(copy(x), 0i32)", "vmap(h)(x)")
            ),
            Abort::GradInMappedArm => format!(
                "{ABORTS}def r(x: tensor[f32]) -> tensor[f32] = {{\n{}vmap(r)(xs)\n",
                arm("tensor[2, f32]", "copy(x)", "grad(h)(x)")
            ),
            Abort::VmapInMappedArm => format!(
                "{ABORTS}def r(x: tensor[1, f32]) -> tensor[1, f32] = {{\n{}vmap(r)(xs)\n",
                arm("tensor[2, 1, f32]", "sum(copy(x), 0i32)", "vmap(h)(x)")
            ),
        }
    }

    /// `xs`: the rank-0 shapes read -3 (the abort's predicate holds); the
    /// mapped ones read `[-3, 1]`, where row 0 would abort and row 1 would
    /// not.
    fn inputs(self) -> Vec<Input> {
        let (shape, data) = match self {
            Abort::GradInArm => (vec![], vec![-3.0]),
            Abort::VmapInArm => (vec![2], vec![-3.0, 1.0]),
            Abort::GradInMappedArm => (vec![2], vec![-3.0, 1.0]),
            Abort::VmapInMappedArm => (vec![2, 1], vec![-3.0, 1.0]),
        };
        vec![Input {
            name: "xs",
            shape,
            data,
        }]
    }

    /// `(threshold, which arms or rows are taken, expected)`. With row 1
    /// alone taking the arm the abort's own predicate holds only in row 0,
    /// whose arm is not taken, so nothing fires; `grad(h)` at 1 is 2 and `h`
    /// at 1 is 1.
    fn rows(self) -> Vec<(&'static str, &'static str, Result<Vec<f64>, &'static str>)> {
        match self {
            Abort::GradInArm => vec![
                ("8.0f32", "untaken", Ok(vec![-3.0])),
                ("-5.0f32", "taken", Err("negative row")),
            ],
            Abort::VmapInArm => vec![
                ("8.0f32", "untaken", Ok(vec![-3.0, 1.0])),
                ("-5.0f32", "taken", Err("negative row")),
            ],
            Abort::GradInMappedArm => vec![
                ("8.0f32", "no row", Ok(vec![-3.0, 1.0])),
                ("-2.0f32", "row 1", Ok(vec![-3.0, 2.0])),
                ("-5.0f32", "both rows", Err("negative row")),
            ],
            Abort::VmapInMappedArm => vec![
                ("8.0f32", "no row", Ok(vec![-3.0, 1.0])),
                ("-2.0f32", "row 1", Ok(vec![-3.0, 1.0])),
                ("-5.0f32", "both rows", Err("negative row")),
            ],
        }
    }
}

/// Item 2: a guarded abort in a `grad` or `vmap` body spliced into an arm
/// fires only where the arm is taken, in the evaluator, whose owner gate
/// reads the abort's activation (the call site's, per row under `vmap`).
///
/// Evidentiary status: REGRESSION TEST for the untaken, no-row and row-1
/// rows (at 224414e1f the abort fires, its fire condition conjoining only
/// the body's own branch path); DISPOSITION LOCK for the taken rows.
#[test]
fn a_guarded_abort_in_a_transform_body_fires_only_where_its_arm_is_taken_in_the_evaluator() {
    let mut rows = Rows::default();
    for shape in Abort::ALL {
        for (threshold, which, expected) in shape.rows() {
            let source = shape.source(threshold);
            let row = format!("{shape:?}, {which}");
            match expected {
                Ok(values) => rows.e_returns(&row, select(&source, &shape.inputs()), &values, None),
                Err(trap) => rows.e_traps(&row, select(&source, &shape.inputs()), trap),
            }
        }
    }
    rows.assert_empty();
}

/// The C rows of item 2, for every shape but `VmapInMappedArm`, whose
/// rank-2 abort condition the C lane's ownership lowering refuses ("must be
/// rank-0, or rank-1 when mapped over a batch axis"). The C lane gates a
/// guarded abort on its owner activation only once `emit_guarded_fail`
/// emits `if (act && fires)`, the C gate the sibling helper ks5-h2b owns;
/// until that merges, the untaken, no-row and row-1 rows abort here.
#[test]
fn a_guarded_abort_in_a_transform_body_fires_only_where_its_arm_is_taken_in_c() {
    let mut rows = Rows::default();
    for shape in [Abort::GradInArm, Abort::VmapInArm, Abort::GradInMappedArm] {
        for (threshold, which, expected) in shape.rows() {
            let source = shape.source(threshold);
            let row = format!("{shape:?}, {which}");
            match &expected {
                Ok(values) => rows.c(&row, &source, &shape.inputs(), Ok(values)),
                Err(trap) => rows.c(&row, &source, &shape.inputs(), Err(trap)),
            }
        }
    }
    rows.assert_empty();
}

/// A shift in a runtime `if` arm, consumed or discarded, by the count
/// `n = -1` ([04-NUM-13]: a negative count traps).
fn shift_in_arm(condition: &str, dead: bool) -> String {
    let arm = if dead {
        "{\n    dead = shl(1i32, n)\n    x\n  }"
    } else {
        "mul(x, scalar_to_tensor(cast(shl(1i32, n), f32)))"
    };
    format!(
        "def selected(x: tensor[f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(copy(x))\n  n = cast(s, i32)\n  if {condition} then {arm} else x\n}}\n"
    )
}

const SHIFT_TRAP: &str = "shift amount must be non-negative, got -1";

/// Item 3: `shl` and `shr` are scalar-only builtins with no RISC node, so a
/// def that shifts is a Host-lane root in every spelling; the host
/// interpreter and the host C entry run only the taken arm. An untaken arm's
/// negative count checks nothing and the taken twin traps, except that the
/// host C entry drops a discarded shift (the `dead` taken row), a lost trap
/// this suite records rather than fixes.
///
/// Evidentiary status: DISPOSITION LOCK (green at 224414e1f); the lane
/// assertion is the finding that no shift reaches the DAG evaluator.
#[test]
fn a_shift_in_an_untaken_arm_checks_nothing_and_its_consumed_taken_twin_traps() {
    let mut rows = Rows::default();
    let input = || scalar("x", -1.0);
    for dead in [false, true] {
        for (untaken, taken) in [
            ("lt(0.5f32, s)", "lt(s, 0.5f32)"),
            ("gt(s, 0.5f32)", "gt(s, -5.0f32)"),
        ] {
            let row = format!("dead {dead}, {untaken}");
            let source = shift_in_arm(untaken, dead);
            rows.e_returns(&row, select(&source, &input()), &[-1.0], Some(Lane::Host));
            rows.c(&row, &source, &input(), Ok(&[-1.0]));
            let source = shift_in_arm(taken, dead);
            rows.e_traps(&row, select(&source, &input()), SHIFT_TRAP);
            if !dead {
                rows.c(&row, &source, &input(), Err(SHIFT_TRAP));
            }
        }
    }
    rows.assert_empty();
}
