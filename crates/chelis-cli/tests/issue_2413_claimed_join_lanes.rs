//! chelis#2413 (#2586 round 3), spec/10 §3.2 with [05-OP-53]: at a runtime
//! `if` join every lane returns what the taken arm computes when that value
//! satisfies the join's declared type, or traps with the typed extent message
//! when the taken arm's claim is false. An untaken arm's claim never sizes
//! the join. A join whose arms' extents lowering does not prove equal (one
//! origin, or one C identity) is refused with the typed chelis#2583 refusal
//! instead of being sized from either arm; it is never a silent value.
//!
//! The matrix is generated, not sampled: the claimed arm in the `then` or
//! `else` position; its claim from a callee's result type, a local ascription,
//! or an `expand`'s unit claim; the claimed extent 0, a literal equal to the
//! extent the arm computes, a different literal, or a runtime extent equal to
//! or different from (0) that extent; the claimed arm taken or untaken; the
//! other arm unclaimed and either independent of the claimed one (`x[0..3]`)
//! or computed from the same value outside the join, so that the two arms'
//! extents have one origin (for a result claim and an ascription). Each
//! cell's expectation is computed from its parameters, never read from a
//! lane: an independent other arm's extent is not proven equal to the
//! claimed arm's, so those cells expect the typed refusal in every lane
//! (the host interpreter included: it runs `selected` as the kernel C
//! emits, and a kernel's lowering failure is its error, never a
//! fall-through to the interpreter).
//!
//! Lanes: the host interpreter (`chelis eval --file` on `main`), the DAG
//! evaluator (`selected` lowered as a tensor entry, as `exec_compile` runs
//! it; where it declines to lower, the same body's kernel lowering states
//! why), and the whole program's C (`chelis build`, linked and run). A cell
//! C built at 096daea8c (the 40 shared-origin cells) must still build; a C
//! run must agree. Every cell is collected before the assertion, so one red
//! cell never hides a sibling.
//!
//! Evidentiary status: REGRESSION TEST for the three silent values the
//! round-2b reviewer found at dfaefd9a2 ([`the_reviewers_unproven_joins_are_the_taken_arm_or_refused`]:
//! a nested join returned an empty tensor in the host interpreter and the
//! DAG evaluator, a `vmap` row join `[0, 0, 0]` in the DAG evaluator), and
//! for the three shared-origin untaken `else` cells whose claim is 0, which
//! returned an empty tensor at 44c9b23e7; DISPOSITION LOCK for every other
//! cell, for the typed refusal of the 60 independent-arm cells, and for the
//! C build rule (C built the same 40 cells at 096daea8c, 592dc55ce and
//! dfaefd9a2).
use assert_cmd::Command;
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_types::types::Prim;
use chelis_types::{RawTensor, finalize_tensor};
use chelis_unord::UnordMap;
use std::path::Path;

#[path = "common/mod.rs"]
mod common;

/// `x`: `[-3, -1, ..., 59]`, summing to 896, so `lt(0.0f32, s)` holds.
fn x32() -> Vec<f32> {
    (0..32).map(|index| (2 * index - 3) as f32).collect()
}

/// The unclaimed arm, `x[0..3]`.
const OTHER: &str = "shrink(copy(x), [[0i64, sub(shape(&x, 0i32), 29i64)]])";
/// The claimed arm's computation for a result claim or an ascription,
/// `x[1..4]`: extent 3.
const CLAIMED_SHRINK: &str = "shrink(copy(x), [[1i64, sub(shape(&x, 0i32), 28i64)]])";

#[derive(Clone, Copy, Debug)]
enum Source {
    Result,
    Ascription,
    Unit,
}

/// The claimed extent, and for a runtime one the extent the input carries.
#[derive(Clone, Copy, Debug)]
enum Claim {
    Literal(usize),
    Runtime(usize),
}

/// The extent the claimed arm computes for a result claim or an ascription.
const COMPUTED: usize = 3;

const CLAIMS: [Claim; 5] = [
    Claim::Literal(0),
    Claim::Literal(COMPUTED),
    Claim::Literal(5),
    Claim::Runtime(COMPUTED),
    Claim::Runtime(0),
];

struct Cell {
    name: String,
    source: String,
    /// The DAG evaluator's inputs to `selected`.
    bindings: UnordMap<String, TensorValue>,
    expected: Expected,
    /// The arms' extents are proven one extent: they are computed from one
    /// value outside the join, which C names as one extent. At 096daea8c C
    /// built exactly these cells, so a refusal of one is a regression, not a
    /// loud answer.
    proven: bool,
    /// The lanes that must refuse a join lowering does not prove: in the
    /// generated matrix the host interpreter and the DAG evaluator, which
    /// run `selected` as a kernel. C's whole-program build interprets a body
    /// its kernel lowering refuses (chelis#1515), so it may return the
    /// expected value.
    refused_in: &'static [&'static str],
}

#[derive(Debug)]
enum Expected {
    Value(Vec<f32>),
    /// The claimed extent and the extent the taken arm computed.
    Trap {
        claimed: usize,
        actual: usize,
    },
}

impl Claim {
    fn extent(self) -> usize {
        match self {
            Claim::Literal(extent) | Claim::Runtime(extent) => extent,
        }
    }
}

fn cell(then_claimed: bool, claim: Claim, taken: bool, source: Source, shared: bool) -> Cell {
    let x = x32();
    // With a shared origin both arms are computed from `y = x[1..4]`, outside
    // the join: the claimed arm is `y + y`, the other `-y`.
    let doubled = x[1..4].iter().map(|v| v + v).collect::<Vec<_>>();
    let negated = x[1..4].iter().map(|v| -v).collect::<Vec<_>>();
    let (other, other_value, computation) = if shared {
        ("neg(copy(y))", negated, "add(copy(y), y)")
    } else {
        (OTHER, x[0..3].to_vec(), CLAIMED_SHRINK)
    };
    let literal = |values: &[f32]| {
        let elements = values
            .iter()
            .map(|value| format!("{value:?}f32"))
            .collect::<Vec<_>>();
        format!("to_tensor([{}])", elements.join(", "))
    };
    let input = |shape: usize, prim: Prim, raw: RawTensor| {
        TensorValue::from_storage(vec![shape], finalize_tensor("input", prim, raw).unwrap())
    };
    let mut bindings = UnordMap::new();
    bindings.insert(
        "x".to_owned(),
        input(
            32,
            Prim::F32,
            RawTensor::Float(x.iter().map(|&v| f64::from(v)).collect()),
        ),
    );
    let mut params = String::new();
    let mut binders = "";
    let mut args = String::new();
    let mut helper = String::new();
    let mut pre = if shared {
        "  y = shrink(copy(x), [[1i64, sub(shape(&x, 0i32), 28i64)]])\n".to_owned()
    } else {
        String::new()
    };
    if let Claim::Runtime(extent) = claim {
        // A runtime extent: an empty tensor inserted to `3 - d` elements,
        // `d` read from the data, so no stage folds it.
        params.push_str(if matches!(source, Source::Ascription) {
            ", like: tensor[n, f32]"
        } else {
            ", like: tensor[*, f32]"
        });
        if matches!(source, Source::Ascription) {
            binders = "[n]";
        }
        args.push_str(&format!(
            ", insert(scalar_to_tensor(0.0f32), 0i32, sub(shape(&w, 0i32), {}i64))",
            COMPUTED - extent
        ));
        bindings.insert(
            "like".to_owned(),
            input(extent, Prim::F32, RawTensor::Float(vec![0.0; extent])),
        );
    }
    // A unit claim holds exactly when the claim is the arm's own extent: the
    // broadcast operand then has one element, otherwise two.
    let holds = claim.extent() == COMPUTED;
    let unit = if holds { 1 } else { 2 };
    let (arm, claimed_value, trap) = match source {
        Source::Result => {
            let operand = if shared { "y" } else { "copy(x)" };
            let (declared, call) = match claim {
                Claim::Literal(extent) => (
                    format!("def claim(y: tensor[*, f32]) -> tensor[{extent}, f32]"),
                    format!("claim({operand})"),
                ),
                Claim::Runtime(_) => (
                    "def claim[n](y: tensor[*, f32], like: tensor[n, f32]) -> tensor[n, f32]"
                        .to_owned(),
                    format!("claim({operand}, copy(like))"),
                ),
            };
            let body = if shared {
                "add(copy(y), y)"
            } else {
                "shrink(y, [[1i64, sub(shape(&y, 0i32), 28i64)]])"
            };
            helper = format!("{declared} = {body}\n");
            (
                call,
                if shared { doubled } else { x[1..4].to_vec() },
                (claim.extent(), COMPUTED),
            )
        }
        Source::Ascription => {
            let extent = match claim {
                Claim::Literal(extent) => extent.to_string(),
                Claim::Runtime(_) => "n".to_owned(),
            };
            (
                format!("{{\n    e: tensor[{extent}, f32] = {computation}\n    e\n  }}"),
                if shared { doubled } else { x[1..4].to_vec() },
                (claim.extent(), COMPUTED),
            )
        }
        Source::Unit => {
            params.push_str(", i: tensor[1, i64]");
            args.push_str(&format!(", to_tensor([{unit}i64])"));
            bindings.insert(
                "i".to_owned(),
                input(1, Prim::Int64, RawTensor::Int(vec![unit])),
            );
            pre += "  t = shrink(copy(x), [[1i64, add(1i64, tensor_to_scalar(sum(copy(i), 0i32)))]])\n";
            let size = match claim {
                Claim::Literal(extent) => format!("{extent}i64"),
                Claim::Runtime(_) => "shape(&like, 0i32)".to_owned(),
            };
            (
                format!("expand(t, 0i32, {size})"),
                vec![x[1]; claim.extent()],
                (1, unit as usize),
            )
        }
    };
    let condition = if then_claimed == taken {
        "lt(0.0f32, s)"
    } else {
        "lt(s, 0.0f32)"
    };
    let (then_arm, else_arm) = if then_claimed {
        (arm.as_str(), other)
    } else {
        (other, arm.as_str())
    };
    let main = match claim {
        Claim::Literal(_) => format!(
            "def main() -> tensor[*, f32] = selected({}{args})\n",
            literal(&x)
        ),
        Claim::Runtime(_) => format!(
            "def main() -> tensor[*, f32] = {{\n  w = shrink({}, [[0i64, tensor_to_scalar(sum(to_tensor([3i64]), 0i32))]])\n  selected({}{args})\n}}\n",
            literal(&x),
            literal(&x)
        ),
    };
    let source_text = format!(
        "{helper}def selected{binders}(x: tensor[32, f32]{params}) -> tensor[*, f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n{pre}  if {condition} then {then_arm} else {else_arm}\n}}\n{main}"
    );
    let expected = if !taken {
        Expected::Value(other_value)
    } else if holds {
        Expected::Value(claimed_value)
    } else {
        Expected::Trap {
            claimed: trap.0,
            actual: trap.1,
        }
    };
    Cell {
        name: format!(
            "{source:?} claim {claim:?} in {} arm, {}{}",
            if then_claimed { "then" } else { "else" },
            if taken { "taken" } else { "untaken" },
            if shared { ", shared origin" } else { "" }
        ),
        source: source_text,
        bindings,
        expected,
        proven: shared,
        refused_in: if shared { &[] } else { &["H", "E"] },
    }
}

fn cells() -> Vec<Cell> {
    let mut cells = Vec::new();
    for shared in [false, true] {
        // A unit claim's extent is the `expand` size, which no other arm
        // shares.
        let sources: &[Source] = if shared {
            &[Source::Result, Source::Ascription]
        } else {
            &[Source::Result, Source::Ascription, Source::Unit]
        };
        for then_claimed in [true, false] {
            for claim in CLAIMS {
                for taken in [true, false] {
                    for &source in sources {
                        cells.push(cell(then_claimed, claim, taken, source, shared));
                    }
                }
            }
        }
    }
    cells
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// What a lane did: a value (shape and elements), a trap or error text, or
/// a refusal to lower or build it, with its text.
#[derive(Debug)]
enum Outcome {
    Value(Vec<f32>),
    Trapped(String),
    Refused(String),
}

/// The typed refusal of a join whose arms' extents are not proven equal
/// (`join_condition_extents` in `chelis-ir`'s lowering).
fn is_unproven_join_refusal(message: &str) -> bool {
    message.contains("whose arms' extents on axis ")
        && message.contains("are not proven equal")
        && message.contains("unimplemented chelis#2583: ")
}

/// `main = tensor(shape=[k], data=[..])` as printed by eval and the C main.
fn printed(values: &[f32]) -> String {
    let elements = values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>();
    format!(
        "main = tensor(shape=[{}], data=[{}])",
        values.len(),
        elements.join(", ")
    )
}

fn parse_printed(stdout: &str) -> Option<Vec<f32>> {
    let line = stdout.lines().find(|line| line.starts_with("main = "))?;
    let data = line.split_once("data=[")?.1.strip_suffix("])")?;
    let values = if data.is_empty() {
        Vec::new()
    } else {
        data.split(", ")
            .map(|value| value.parse().ok())
            .collect::<Option<Vec<f32>>>()?
    };
    (printed(&values) == line).then_some(values)
}

fn host(directory: &Path, stem: &str, source: &str) -> Outcome {
    std::fs::write(directory.join(format!("{stem}.ch")), source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(directory)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", &format!("{stem}.ch")])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    match parse_printed(&stdout) {
        Some(values) if output.status.success() => Outcome::Value(values),
        _ if is_unproven_join_refusal(&text(&output)) => Outcome::Refused(text(&output)),
        _ => Outcome::Trapped(text(&output)),
    }
}

fn compiled(directory: &Path, stem: &str, source: &str) -> Outcome {
    let path = directory.join(format!("{stem}.ch"));
    std::fs::write(&path, source).unwrap();
    let out_dir = directory.join(format!("{stem}-out"));
    let built = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(&out_dir)
        .output()
        .unwrap();
    if !built.status.success() {
        return Outcome::Refused(text(&built));
    }
    let linked = common::link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(linked.success(), "{stem}: link failed: {linked}");
    let run = std::process::Command::new(out_dir.join(stem))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&run.stdout);
    match parse_printed(&stdout) {
        Some(values) if run.status.success() => Outcome::Value(values),
        _ => Outcome::Trapped(text(&run)),
    }
}

fn dag(cell: &Cell) -> Outcome {
    let declarations = chelis_surf::parser::parse_str(&cell.source).expect("Surf parse");
    let checked = chelis_types::check_ir_program(
        &chelis_surf::desugar::desugar_program(&declarations).expect("Surf desugar"),
    )
    .unwrap_or_else(|report| panic!("type check failed: {:?}", report.errors));
    // A fatal lowering diagnostic (one raised inside a transform body)
    // unwinds with the diagnostic as its payload.
    let lowered = std::panic::catch_unwind(|| {
        chelis_ir::host::lower_named_tensor_entry_dag(&checked, "selected")
    });
    let dag = match lowered {
        Ok(Some(dag)) => dag,
        Ok(None) => {
            let kernel = chelis_ir::host::host_def_kernel(
                &chelis_ir::host::HostLoweringSession::new(&checked),
                "selected",
            );
            return Outcome::Refused(match kernel {
                Err(diagnostic) => diagnostic.to_string(),
                Ok(_) => "`selected` does not lower as a tensor entry".to_owned(),
            });
        }
        Err(payload) => match payload.downcast::<chelis_ir::lower::LowerDiagnostic>() {
            Ok(diagnostic) => return Outcome::Refused(diagnostic.to_string()),
            Err(payload) => std::panic::resume_unwind(payload),
        },
    };
    let root = *dag.roots().last().expect("a root");
    match eval_tensor(&dag, &cell.bindings) {
        Err(message) => Outcome::Trapped(message),
        Ok(values) if values[&root].shape.len() == 1 => Outcome::Value(
            values[&root]
                .to_f64_lossy_vec()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        ),
        Ok(values) => Outcome::Trapped(format!("result shape {:?}", values[&root].shape)),
    }
}

/// Whether `outcome` is `expected`: the same elements, or the typed extent
/// trap stating the claimed and the computed extent (`extent `0`: claimed =
/// 0, shrink axis 0 = 3`; a runtime claim that the callee's operand also
/// sizes is checked at the call, `extent `n`: like axis 0 = 0, y axis 0 =
/// 3`). A proven join admits no refusal. One lowering does not prove admits
/// the typed refusal, or in C any build refusal, and in the cell's
/// `refused_in` lanes only that.
fn agrees(cell: &Cell, lane: &str, outcome: &Outcome) -> bool {
    if let Outcome::Refused(message) = outcome {
        return !cell.proven && (lane == "C" || is_unproven_join_refusal(message));
    }
    if cell.refused_in.contains(&lane) {
        return false;
    }
    match (&cell.expected, outcome) {
        (Expected::Value(values), Outcome::Value(got)) => values == got,
        (Expected::Trap { claimed, actual }, Outcome::Trapped(message)) => {
            message.contains("numeric trap: domain in ")
                && message.contains("extent `")
                && message.contains(&format!(" = {claimed},"))
                && message.contains(&format!("axis 0 = {actual}"))
        }
        _ => false,
    }
}

/// Each cell's disagreeing lanes.
fn disagreements(cells: &[Cell]) -> Vec<String> {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, cell) in cells.iter().enumerate() {
        let stem = format!("cell_{index}");
        let lanes = [
            ("H", host(directory.path(), &stem, &cell.source)),
            ("E", dag(cell)),
            ("C", compiled(directory.path(), &stem, &cell.source)),
        ];
        for (lane, outcome) in lanes {
            if !agrees(cell, lane, &outcome) {
                failures.push(format!(
                    "{} [{lane}]: expected {:?}, got {outcome:?}",
                    cell.name, cell.expected
                ));
            }
        }
    }
    failures
}

/// Every cell of the matrix, in every lane: the taken arm's value, or its
/// claim's typed trap; never another value.
#[test]
fn every_lane_returns_the_taken_arm_or_its_claims_trap_at_a_claimed_join() {
    let cells = cells();
    assert_eq!(cells.len(), 100);
    let failures = disagreements(&cells);
    assert!(
        failures.is_empty(),
        "{} of {} cell lanes disagree:\n{}",
        failures.len(),
        cells.len() * 3,
        failures.join("\n")
    );
}

/// A declared extent on the join itself, by a local ascription or by the
/// result type, is a claim on an active node: the taken arm's three elements
/// violate the declared 0, so every lane traps with the typed extent message,
/// whatever the untaken arm claims, or refuses the join: these arms' extents
/// (a runtime `shrink` against a claimed 0) are not proven equal.
///
/// Evidentiary status: REGRESSION TEST: at 44c9b23e7 the host interpreter
/// returned `0.0` and an empty tensor, the join's extent read from the
/// untaken arm's zeros.
#[test]
fn a_declared_join_extent_is_checked_against_the_taken_arm() {
    let x = x32();
    let elements = x
        .iter()
        .map(|value| format!("{value:?}f32"))
        .collect::<Vec<_>>();
    let literal = format!("to_tensor([{}])", elements.join(", "));
    let helper = "def zero(y: tensor[*, f32]) -> tensor[0, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 31i64)]])\n";
    let join = format!("if lt(0.0f32, s) then {OTHER} else zero(x)");
    let cells = [
        (
            "ascription on the join",
            format!(
                "{helper}def selected(x: tensor[32, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  r: tensor[0, f32] = {join}\n  sum(r, 0i32)\n}}\ndef main() -> tensor[f32] = selected({literal})\n"
            ),
        ),
        (
            "result type of the join",
            format!(
                "{helper}def selected(x: tensor[32, f32]) -> tensor[0, f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  {join}\n}}\ndef main() -> tensor[0, f32] = selected({literal})\n"
            ),
        ),
    ]
    .map(|(name, source)| {
        let mut bindings = UnordMap::new();
        bindings.insert(
            "x".to_owned(),
            TensorValue::from_storage(
                vec![32],
                finalize_tensor(
                    "input",
                    Prim::F32,
                    RawTensor::Float(x.iter().map(|&v| f64::from(v)).collect()),
                )
                .unwrap(),
            ),
        );
        Cell {
            name: name.to_owned(),
            source,
            bindings,
            expected: Expected::Trap {
                claimed: 0,
                actual: 3,
            },
            proven: false,
            refused_in: &[],
        }
    });
    let failures = disagreements(&cells);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The round-2b reviewer's joins whose arms' extents are not proven equal:
/// a nested `if` whose inner `then` arm claims 0 (the taken inner `else`
/// holds `x[0..3]`), and a `vmap` row function whose arms hold 5 and 3
/// elements, the `else` arm unclaimed or claimed 3 by a callee's result
/// type. Every lane returns the taken arm's value or refuses the join.
///
/// Evidentiary status: REGRESSION TEST: at dfaefd9a2 the nested join
/// returned an empty tensor in the host interpreter and the DAG evaluator,
/// and each `vmap` join `[0, 0, 0]` in the DAG evaluator, the condition
/// sized as the larger arm.
#[test]
fn the_reviewers_unproven_joins_are_the_taken_arm_or_refused() {
    let x = x32();
    let elements = x
        .iter()
        .map(|value| format!("{value:?}f32"))
        .collect::<Vec<_>>();
    let literal = format!("to_tensor([{}])", elements.join(", "));
    let rows = [
        [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
        [9.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let matrix = format!(
        "to_tensor([{}])",
        rows.iter()
            .map(|row| {
                let row = row
                    .iter()
                    .map(|value| format!("{value:?}f32"))
                    .collect::<Vec<_>>();
                format!("[{}]", row.join(", "))
            })
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Each row's sum decides its arm: the first five elements where it is
    // positive, else the first three.
    let row_sums = rows
        .iter()
        .map(|row| {
            let kept = if row.iter().sum::<f32>() > 0.0 { 5 } else { 3 };
            row[..kept].iter().sum::<f32>()
        })
        .collect::<Vec<_>>();
    let vmap_join = |else_arm: &str, helper: &str| {
        format!(
            "{helper}def rowf(r: tensor[8, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(copy(r), 0i32))\n  sum(if lt(0.0f32, s) then shrink(copy(r), [[0i64, sub(shape(&r, 0i32), 3i64)]]) else {else_arm}, 0i32)\n}}\ndef selected(xs: tensor[3, 8, f32]) -> tensor[3, f32] = vmap(rowf)(xs)\ndef main() -> tensor[3, f32] = selected({matrix})\n"
        )
    };
    let cells = [
        (
            "nested join, inner then arm claimed 0",
            format!(
                "def zero(y: tensor[*, f32]) -> tensor[0, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 31i64)]])\ndef selected(x: tensor[32, f32]) -> tensor[*, f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  if lt(0.0f32, s) then (if lt(s, 0.0f32) then zero(copy(x)) else {OTHER}) else zero(x)\n}}\ndef main() -> tensor[*, f32] = selected({literal})\n"
            ),
            ("x", vec![32], x.clone()),
            x[0..3].to_vec(),
        ),
        (
            "vmap row join, arms of 5 and 3",
            vmap_join(
                "shrink(copy(r), [[0i64, sub(shape(&r, 0i32), 5i64)]])",
                "",
            ),
            ("xs", vec![3, 8], rows.concat()),
            row_sums.clone(),
        ),
        (
            "vmap row join, else arm claimed 3",
            vmap_join(
                "three(r)",
                "def three(y: tensor[*, f32]) -> tensor[3, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 5i64)]])\n",
            ),
            ("xs", vec![3, 8], rows.concat()),
            row_sums,
        ),
    ]
    .map(|(name, source, (input, shape, values), expected)| {
        let mut bindings = UnordMap::new();
        bindings.insert(
            input.to_owned(),
            TensorValue::from_storage(
                shape,
                finalize_tensor(
                    "input",
                    Prim::F32,
                    RawTensor::Float(values.iter().map(|&v| f64::from(v)).collect()),
                )
                .unwrap(),
            ),
        );
        Cell {
            name: name.to_owned(),
            source,
            bindings,
            expected: Expected::Value(expected),
            proven: false,
            refused_in: &[],
        }
    });
    let failures = disagreements(&cells);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
