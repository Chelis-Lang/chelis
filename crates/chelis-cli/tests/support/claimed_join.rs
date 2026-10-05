use assert_cmd::Command;
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_types::types::Prim;
use chelis_types::{RawTensor, finalize_tensor};
use chelis_unord::UnordMap;
use std::path::Path;


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
        .args(["build", "--emit-c", path.to_str().unwrap(), "--target", "c", "--output"])
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
