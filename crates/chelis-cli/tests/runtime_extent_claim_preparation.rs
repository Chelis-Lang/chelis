//! #1277 preparation: one fixture set, current observations and pending contract.
//!
//! The normal test locks measured gaps; it is NOT a claim that they are fixed.
//! Run the ignored `claimed_extent_contract` test explicitly before moving a
//! row to acceptance. It reports all failed cells rather than stopping at one.
//! The manual command and remaining owners live in runtime_extents.md C5.
mod common;

use common::{gcc_available, link_generated};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};
use tempfile::tempdir;

#[derive(Clone)]
struct Input {
    dims: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Clone)]
struct Case {
    id: String,
    issue: u32,
    source: String,
    // Authored expectation, independent of both compiler lanes and the baseline.
    signature: Option<&'static str>,
    expected: Expected,
    exported: Option<Vec<Input>>,
}

#[derive(Clone)]
enum Expected {
    Tensor(Vec<usize>, Vec<f64>),
    Domain(&'static str, &'static [&'static str]),
    Reject(&'static str),
}

fn vector(n: usize) -> Input {
    Input {
        dims: vec![n],
        values: (1..=n).map(|v| v as f32).collect(),
    }
}

fn literal(input: &Input) -> String {
    if input.dims.is_empty() {
        return format!("scalar_to_tensor({:.1}f32)", input.values[0]);
    }
    assert_eq!(input.dims.len(), 1, "call matrix inputs are rank zero/one");
    format!(
        "to_tensor([{}])",
        input
            .values
            .iter()
            .map(|v| format!("{v:.1}f32"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn call_matrix(
    cases: &mut Vec<Case>,
    family: &str,
    issue: u32,
    def: &str,
    signature: &'static str,
    inputs: Vec<Input>,
    result: Expected,
) {
    let good = matches!(result, Expected::Tensor(..));
    let call = format!(
        "f({})",
        inputs.iter().map(literal).collect::<Vec<_>>().join(", ")
    );
    for (form, suffix) in [
        ("export", String::new()),
        ("binding", format!("out = {call}\n")),
        ("root", format!("def main() = {call}\n")),
    ] {
        cases.push(Case {
            id: format!(
                "{family}.{form}.{}",
                if good { "satisfied" } else { "mismatch" }
            ),
            issue,
            source: format!("{def}\n{suffix}"),
            signature: Some(signature),
            expected: result.clone(),
            exported: (form == "export").then(|| inputs.clone()),
        });
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let seed = Input {
        dims: vec![],
        values: vec![7.0],
    };
    for good in [true, false] {
        call_matrix(
            &mut cases,
            "named",
            1374,
            "def f(b: tensor[f32], x: tensor[rows, f32], y: tensor[cols, f32]) -> tensor[rows, f32] = insert(b, 0i32, shape(y, 0i32))",
            "(tensor[f32], tensor[rows, f32], tensor[cols, f32]) -> tensor[rows, f32]",
            vec![seed.clone(), vector(if good { 3 } else { 2 }), vector(3)],
            if good {
                Expected::Tensor(vec![3], vec![7.0; 3])
            } else {
                Expected::Domain("load", &["x axis 0 = 2", "y axis 0 = 3"])
            },
        );
        call_matrix(
            &mut cases,
            "foreign",
            1376,
            "def f(x: tensor[rows, f32], y: tensor[cols, f32]) -> tensor[rows, cols, f32] = insert(x, 1i32, shape(x, 0i32))",
            "(tensor[rows, f32], tensor[cols, f32]) -> tensor[rows, cols, f32]",
            vec![vector(2), vector(if good { 2 } else { 3 })],
            if good {
                Expected::Tensor(vec![2, 2], vec![1.0, 1.0, 2.0, 2.0])
            } else {
                Expected::Domain("load", &["x axis 0 = 2", "y axis 0 = 3"])
            },
        );
        call_matrix(
            &mut cases,
            "literal",
            1377,
            "def f(b: tensor[f32], x: tensor[rows, f32]) -> tensor[4, f32] = insert(b, 0i32, shape(x, 0i32))",
            "(tensor[f32], tensor[rows, f32]) -> tensor[4, f32]",
            vec![seed.clone(), vector(if good { 4 } else { 5 })],
            if good {
                Expected::Tensor(vec![4], vec![7.0; 4])
            } else {
                Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"])
            },
        );
        call_matrix(
            &mut cases,
            "shrink",
            1397,
            "def f(x: tensor[rows, f32]) -> tensor[2, f32] = shrink(x, [[1i64, shape(x, 0i32)]])",
            "(tensor[rows, f32]) -> tensor[2, f32]",
            vec![vector(if good { 3 } else { 4 })],
            if good {
                Expected::Tensor(vec![2], vec![2.0, 3.0])
            } else {
                Expected::Domain("shrink", &["claimed = 2", "shrink axis 0 = 3"])
            },
        );
    }
    // Keep the issues' original polymorphic n/m binders as well as the
    // concrete named-dimension controls above; changing spelling must not
    // silently change which claim-preservation seam the reproducer tests.
    let polymorphic: Vec<_> = cases
        .iter()
        .filter(|case| case.issue == 1374 || case.issue == 1376)
        .map(|case| {
            let mut case = case.clone();
            case.id = format!("polymorphic.{}", case.id);
            case.source = case.source.replace("rows", "n").replace("cols", "m");
            case.signature = Some(if case.issue == 1374 {
                "(tensor[f32], tensor[d0, f32], tensor[d1, f32]) -> tensor[d0, f32]"
            } else {
                "(tensor[d0, f32], tensor[d1, f32]) -> tensor[d0, d1, f32]"
            });
            case
        })
        .collect();
    cases.extend(polymorphic);
    let mut add = |id: &str, issue, source: &str, signature, expected| {
        cases.push(Case {
            id: id.into(),
            issue,
            source: source.into(),
            signature,
            expected,
            exported: None,
        })
    };
    for (id, bound) in [("vmap.shape", "shape(x, 0i32)"), ("vmap.literal", "3i64")] {
        add(
            id,
            1378,
            &format!(
                "def g(x: tensor[n, f32]) -> tensor[m, f32] = shrink(x, [[1i64, {bound}]])\ndef main() = vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n"
            ),
            None,
            Expected::Tensor(vec![2, 2], vec![2.0, 3.0, 5.0, 6.0]),
        );
    }
    add(
        "vmap.element_bound",
        1378,
        "def g(x: tensor[n, f32]) -> tensor[m, f32] = { end = cast(tensor_to_scalar(sum(x, 0i32)), int64)\n shrink(x, [[0i64, end]]) }\nout = vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n",
        None,
        Expected::Reject("batch_varying_extent"),
    );
    add(
        "wildcard.root",
        1397,
        "def g(x: tensor[n, f32]) -> tensor[m, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\ndef main() = g(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
        None,
        Expected::Tensor(vec![2], vec![2.0, 3.0]),
    );
    for (id, body) in [
        (
            "record.direct",
            "expand(to_tensor([0.25f32]), 0i32, shape(inp.q, 0i32))",
        ),
        (
            "record.alias",
            "{ q = inp.q\n expand(to_tensor([0.25f32]), 0i32, shape(q, 0i32)) }",
        ),
    ] {
        add(
            id,
            1266,
            &format!(
                "type Inputs = | Inputs {{ q: tensor[batch, 4, f32] }}\nsig f: Inputs -> tensor[batch, f32]\ndef f(inp: Inputs) = {body}\nout = f(Inputs {{ q: to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]) }})\n"
            ),
            None,
            Expected::Tensor(vec![2], vec![0.25, 0.25]),
        );
        add(
            &format!("{id}.bad_axis"),
            1266,
            &format!(
                "type Inputs = | Inputs {{ q: tensor[batch, 4, f32] }}\nsig f: Inputs -> tensor[batch, f32]\ndef f(inp: Inputs) = {}\n",
                body.replace("shape(inp.q, 0i32)", "shape(inp.q, 2i32)")
                    .replace("shape(q, 0i32)", "shape(q, 2i32)")
            ),
            None,
            Expected::Reject("out of bounds"),
        );
    }
    for (id, size) in [
        ("broadcast.literal", "3i64"),
        ("broadcast.symbolic", "shape(xs, 0i32)"),
    ] {
        let dim = if id == "broadcast.literal" { "3" } else { "n" };
        add(
            id,
            1619,
            &format!(
                "sig f: tensor[{dim}, f32] -> tensor[{dim}, bool]\ndef f(xs) = gt(xs, expand(to_tensor([1.5f32]), 0i32, {size}))\nout = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
            ),
            None,
            Expected::Tensor(vec![3], vec![0.0, 1.0, 1.0]),
        );
    }
    add(
        "broadcast.folded",
        1619,
        "xs = to_tensor([1.0f32, 2.0f32, 3.0f32])\nout = gt(xs, expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32)))\n",
        None,
        Expected::Tensor(vec![3], vec![0.0, 1.0, 1.0]),
    );
    add(
        "broadcast.non_unit",
        1619,
        "def f(x: tensor[2, f32]) -> tensor[3, f32] = expand(x, 0i32, 3i64)\n",
        None,
        Expected::Reject("got 2: expand broadcasts a size-1 axis"),
    );
    add(
        "scope.independent",
        1566,
        "def total(x: &tensor[seq, f32]) -> tensor[f32] = sum(x, 0i32)\ndef use2(x: &tensor[batch, seq, f32]) -> tensor[f32] = sum(sum(x, 0i32), 0i32)\na = total(to_tensor([1.0f32, 2.0f32, 3.0f32]))\nb = use2(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\nout = add(a, b)\n",
        None,
        Expected::Tensor(vec![], vec![16.0]),
    );
    for (id, n, expected) in [
        (
            "scope.unread.satisfied",
            2,
            Expected::Tensor(vec![], vec![3.0]),
        ),
        (
            "scope.unread.mismatch",
            3,
            Expected::Domain("load", &["x axis 0 = 2", "p axis 0 = 3"]),
        ),
    ] {
        add(
            id,
            1374,
            &format!(
                "def f(x: tensor[extent, f32], p: tensor[extent, f32]) -> tensor[f32] = sum(x, 0i32)\nout = f(to_tensor([1.0f32, 2.0f32]), {})\n",
                literal(&vector(n))
            ),
            Some("(tensor[extent, f32], tensor[extent, f32]) -> tensor[f32]"),
            expected,
        );
    }
    // Current resolved controls; these do not claim a non-expand #1512 census.
    add(
        "validation.sum.good",
        1512,
        "def f(x: tensor[1, f32]) -> tensor[f32] = sum(expand(x, 0i32, 3i64), 0i32)\nout = f(to_tensor([2.0f32]))\n",
        None,
        Expected::Tensor(vec![], vec![6.0]),
    );
    add(
        "validation.sum.axis",
        1512,
        "def f(x: tensor[1, f32]) -> tensor[f32] = sum(expand(x, 0i32, 3i64), 1i32)\n",
        None,
        Expected::Reject("out of bounds"),
    );
    add(
        "validation.permute.good",
        1512,
        "def f(x: tensor[2, f32]) -> tensor[2, 3, f32] = permute(insert(x, 0i32, 3i64), 1i32, 0i32)\nout = f(to_tensor([1.0f32, 2.0f32]))\n",
        None,
        Expected::Tensor(vec![2, 3], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
    );
    add(
        "validation.permute.result",
        1512,
        "def f(x: tensor[2, f32]) -> tensor[3, 2, f32] = permute(insert(x, 0i32, 3i64), 1i32, 0i32)\n",
        None,
        Expected::Reject("doesn't match declared signature"),
    );
    cases
}

fn run(binary: &Path, args: &[&str], cwd: &Path) -> std::process::Output {
    Command::new(binary)
        .args(args)
        .current_dir(cwd)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .output()
        .expect("run fixture command")
}

fn receipt(stage: &str, output: &std::process::Output) -> Value {
    json!({"stage":stage, "success":output.status.success(),
        "stdout":String::from_utf8_lossy(&output.stdout).trim(),
        "stderr":String::from_utf8_lossy(&output.stderr).trim()})
}

fn driver(inputs: &[Input]) -> (String, Vec<String>) {
    let mut text = String::from(
        "\n#include <stdio.h>\n#include <stdlib.h>\nint main(int argc, char **argv) {\n(void)argc;\n(void)argv;\n",
    );
    let mut argv = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let values = input
            .values
            .iter()
            .map(|v| format!("{v:.1}f"))
            .collect::<Vec<_>>()
            .join(",");
        text.push_str(&format!(
            "float data{i}[] = {{{values}}};\nint64_t shape{i}[{}];\n",
            input.dims.len().max(1)
        ));
        for (axis, dim) in input.dims.iter().enumerate() {
            argv.push(dim.to_string());
            text.push_str(&format!(
                "shape{i}[{axis}] = strtoll(argv[{}], NULL, 10);\n",
                argv.len()
            ));
        }
        text.push_str(&format!("chelis_tensor *a{i} = chelis_tensor_entry_borrow({}, shape{i}, CHELIS_DTYPE_F32, data{i}, sizeof(data{i}));\n", input.dims.len()));
    }
    let args = (0..inputs.len())
        .map(|i| format!("a{i}"))
        .collect::<Vec<_>>()
        .join(",");
    text.push_str(&format!("chelis_tensor *result = f({args});\n"));
    text.push_str(
        r#"printf("out = tensor(shape=[");
for (int32_t axis = 0; axis < chelis_tensor_rank(result); ++axis) {
    if (axis) printf(", ");
    printf("%lld", (long long)chelis_tensor_shape(result, axis));
}
printf("], data=[");
chelis_read_view view = chelis_tensor_read_view(result);
if (view.dtype != CHELIS_DTYPE_F32) return 12;
for (int64_t i = 0; i < view.count; ++i) {
    if (i) printf(", ");
    printf("%.9g", (double)((const float *)view.data)[i]);
}
printf("])\n");
chelis_tensor_release(result);
return 0;
}
"#,
    );
    (text, argv)
}

fn observe(case: &Case) -> Value {
    let dir = tempdir().expect("fixture directory");
    let path = dir.path().join("fixture.ch");
    fs::write(&path, &case.source).expect("write fixture");
    let binary = Path::new(env!("CARGO_BIN_EXE_chelis"));
    let check = run(
        binary,
        &["check", "--show-inferred", path.to_str().unwrap()],
        dir.path(),
    );
    let report: Value = serde_json::from_slice(&check.stdout)
        .expect("checker JSON, not a style/parse transport failure");
    assert_eq!(
        report["components"]["parse"], 1,
        "fixture must parse: {}: {report}",
        case.id
    );
    assert!(
        report["unresolved_names"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "fixture names must resolve: {}: {report}",
        case.id
    );
    let signatures = report["inferred_signatures"]
        .as_array()
        .expect("signatures")
        .iter()
        .map(|s| {
            (
                s["function"].as_str().unwrap().to_string(),
                s["checked_signature"].clone(),
            )
        })
        .collect::<serde_json::Map<String, Value>>();
    let checker = json!({"success":check.status.success(), "score":report["score"],
        "errors":report["errors"], "signatures":signatures});
    let eval = if case.exported.is_none() {
        receipt(
            "execute",
            &run(
                binary,
                &["eval", "--timeout", "20", "--file", path.to_str().unwrap()],
                dir.path(),
            ),
        )
    } else {
        Value::Null
    };
    let out = dir.path().join("out");
    let build = run(
        binary,
        &[
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            out.to_str().unwrap(),
        ],
        dir.path(),
    );
    let compiled = if !build.status.success() {
        receipt("build", &build)
    } else {
        let c_path = out.join("fixture.c");
        let mut source = fs::read_to_string(&c_path).expect("generated C");
        let args = if let Some(inputs) = &case.exported {
            assert!(
                !source.contains("int main("),
                "export fixture unexpectedly contains an entry"
            );
            let (driver, args) = driver(inputs);
            source.push_str(&driver);
            fs::write(&c_path, &source).expect("append exported runtime caller");
            args
        } else {
            Vec::new()
        };
        if !source.contains("int main(") {
            json!({"stage":"no_entry", "success":false, "stdout":"", "stderr":"accepted root has no C entry"})
        } else {
            assert!(
                link_generated(&out, "fixture.c", "fixture").success(),
                "fixture link failed: {}",
                case.id
            );
            receipt(
                "execute",
                &Command::new(out.join("fixture"))
                    .args(args)
                    .current_dir(dir.path())
                    .output()
                    .expect("execute C"),
            )
        }
    };
    // Generated diagnostics can name the task-owned temporary path. Normalize
    // only that path; trap lines, node identities, values and source names stay.
    let raw = json!({"issue":case.issue, "check":checker, "eval":eval, "c":compiled}).to_string();
    serde_json::from_str(&raw.replace(&dir.path().display().to_string(), "<fixture>"))
        .expect("normalized observation")
}

fn tensor(stdout: &str, name: &str) -> Option<(Vec<usize>, Vec<f64>)> {
    let prefix = format!("{name} = ");
    let line = stdout.lines().find_map(|line| line.strip_prefix(&prefix))?;
    // The CLI renders a rank-zero tensor as its scalar value.
    if let Ok(value) = line.parse::<f64>() {
        return Some((vec![], vec![value]));
    }
    if !line.starts_with("tensor(shape=[") {
        return None;
    }
    let shape = line.split("shape=[").nth(1)?.split(']').next()?;
    let data = line.split("data=[").nth(1)?.split(']').next()?;
    let shape = if shape.trim().is_empty() {
        Vec::new()
    } else {
        shape
            .split(',')
            .map(|x| x.trim().parse())
            .collect::<Result<Vec<_>, _>>()
            .ok()?
    };
    let data = if data.trim().is_empty() {
        Vec::new()
    } else {
        data.split(',')
            .map(|x| match x.trim() {
                "false" => Ok(0.0),
                "true" => Ok(1.0),
                n => n.parse(),
            })
            .collect::<Result<Vec<_>, _>>()
            .ok()?
    };
    Some((shape, data))
}

// Context is information, not the canonical trap line's fixed bytes. Ignore
// punctuation/whitespace, but retain each source/axis/value association and
// the sign of an extent. Separate records can appear in either order.
fn context_has_record(stderr: &str, record: &str) -> bool {
    fn words(text: &str) -> Vec<&str> {
        text.split(|c: char| !c.is_ascii_alphanumeric() && !matches!(c, '_' | '-' | '.'))
            .filter(|word| !word.is_empty())
            .collect()
    }
    let required = words(record);
    assert!(
        !required.is_empty(),
        "a trap context record must convey information"
    );
    words(stderr)
        .windows(required.len())
        .any(|seen| seen == required)
}

fn contract_failures(case: &Case, observation: &Value) -> Vec<String> {
    let mut failures = Vec::new();
    let check = &observation["check"];
    if let Expected::Reject(message) = case.expected {
        let errors = check["errors"].as_array().expect("errors");
        if check["success"] != false
            || errors.is_empty()
            || !errors
                .iter()
                .any(|e| e["message"].as_str().is_some_and(|m| m.contains(message)))
        {
            failures.push(format!(
                "{}.check: expected rejection containing {message:?}",
                case.id
            ));
        }
        for lane in ["eval", "c"] {
            let run = &observation[lane];
            if !run.is_null()
                && (run["success"] != false
                    || !run["stderr"]
                        .as_str()
                        .is_some_and(|text| text.contains(message)))
            {
                failures.push(format!(
                    "{}.{}: expected the checker rejection, not execution",
                    case.id, lane
                ));
            }
        }
        return failures;
    }
    if check["success"] != true || check["score"] != 1 || check["errors"] != json!([]) {
        failures.push(format!("{}.check: expected score 1 and no errors", case.id));
    }
    if let Some(signature) = case.signature
        && check["signatures"]["f"] != signature
    {
        failures.push(format!(
            "{}.signature: expected {signature}, observed {}",
            case.id, check["signatures"]["f"]
        ));
    }
    for lane in if case.exported.is_some() {
        &['c'][..]
    } else {
        &['e', 'c'][..]
    } {
        let lane = if *lane == 'e' { "eval" } else { "c" };
        let run = &observation[lane];
        let stdout = run["stdout"].as_str().unwrap();
        let stderr = run["stderr"].as_str().unwrap();
        let satisfied = match &case.expected {
            Expected::Tensor(shape, data) => {
                run["stage"] == "execute"
                    && run["success"] == true
                    && tensor(
                        stdout,
                        if case.source.contains("def main()") {
                            "main"
                        } else {
                            "out"
                        },
                    )
                    .as_ref()
                        == Some(&(shape.clone(), data.clone()))
            }
            Expected::Domain(op, context) => {
                run["stage"] == "execute"
                    && run["success"] == false
                    && stderr
                        .lines()
                        .any(|line| line == format!("numeric trap: domain in {op} at int64"))
                    && context
                        .iter()
                        .all(|record| context_has_record(stderr, record))
            }
            Expected::Reject(_) => unreachable!(),
        };
        if !satisfied {
            failures.push(format!("{}.{}: contract not met: {run}", case.id, lane));
        }
    }
    failures
}

fn collect() -> (Value, Vec<String>) {
    assert!(
        gcc_available(),
        "the fixture suite requires a working host C toolchain; no lane may skip"
    );
    let mut observed = serde_json::Map::new();
    let mut failures = Vec::new();
    for case in cases() {
        let result = observe(&case);
        failures.extend(contract_failures(&case, &result));
        assert!(
            observed.insert(case.id, result).is_none(),
            "duplicate fixture id"
        );
    }
    eprintln!(
        "EXTENT CLAIM FIXTURES: {} cases, {} unmet contract cells",
        observed.len(),
        failures.len()
    );
    (Value::Object(observed), failures)
}

#[test]
fn current_observations_are_explicit_and_do_not_claim_acceptance() {
    let (observed, failures) = collect();
    let expected: Value =
        serde_json::from_str(include_str!("fixtures/runtime_extent_claim_baseline.json"))
            .expect("baseline JSON");
    let changed: Vec<_> = observed
        .as_object()
        .unwrap()
        .iter()
        .filter(|(id, value)| expected.get(*id) != Some(*value))
        .map(|(id, _)| id)
        .collect();
    assert!(
        changed.is_empty()
            && observed.as_object().unwrap().len() == expected.as_object().unwrap().len(),
        "behavior changed in {changed:?}: compare each cell with its contract before updating the issue-owned baseline"
    );
    assert!(
        !failures.is_empty(),
        "all contracts now pass: retire the preparation baseline and attach acceptance receipts"
    );
}

#[test]
#[ignore = "manual pending acceptance: runtime_extents.md C5; expected red until B2b repairs"]
fn claimed_extent_contract() {
    let (_, failures) = collect();
    assert!(
        failures.is_empty(),
        "{} contract cells remain:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// #1377: a literal result claim retains its required failure after inlining.
#[test]
fn literal_result_claim_contract() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let fixtures: Vec<_> = cases()
        .into_iter()
        .filter(|case| case.issue == 1377)
        .collect();
    assert_eq!(fixtures.len(), 6, "export/binding/root, satisfied/mismatch");
    let failures: Vec<_> = fixtures
        .iter()
        .flat_map(|case| contract_failures(case, &observe(case)))
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Independent expected-value and rejection evidence for the executable example.
#[test]
fn literal_extent_example_contract() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let source = include_str!("../../../examples/literal_extent_claim.ch");
    for good in [true, false] {
        let source = if good {
            source.to_owned()
        } else {
            let changed = source.replace("3.0f32, 4.0f32]", "3.0f32, 4.0f32, 5.0f32]");
            assert_ne!(
                source, changed,
                "negative example must change the actual extent"
            );
            changed
        };
        let case = Case {
            id: format!("literal.example.{good}"),
            issue: 1377,
            source,
            signature: None,
            exported: None,
            expected: if good {
                Expected::Tensor(vec![4], vec![7.0; 4])
            } else {
                Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"])
            },
        };
        let observation = observe(&case);
        assert_eq!(
            observation["check"]["signatures"]["fill_four"],
            "(tensor[f32], tensor[rows, f32]) -> tensor[4, f32]"
        );
        let failures = contract_failures(&case, &observation);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

/// [04] §4.7 and [06] §5.2: a call's runtime obligation survives another
/// inlining boundary and remains observable when its result is discarded.
#[test]
fn literal_claim_transport_survives_nested_and_unused_calls() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut fixtures = Vec::new();
    for good in [true, false] {
        let x = vector(if good { 4 } else { 5 });
        let seed = Input {
            dims: vec![],
            values: vec![7.0],
        };
        let definitions = "def g(b: tensor[f32], x: tensor[rows, f32]) -> tensor[4, f32] = insert(b, 0i32, shape(x, 0i32))\ndef f(b: tensor[f32], x: tensor[rows, f32]) -> tensor[4, f32] = g(b, x)";
        call_matrix(
            &mut fixtures,
            "literal.nested",
            1377,
            definitions,
            "(tensor[f32], tensor[rows, f32]) -> tensor[4, f32]",
            vec![seed.clone(), x.clone()],
            if good {
                Expected::Tensor(vec![4], vec![7.0; 4])
            } else {
                Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"])
            },
        );
        fixtures.push(Case {
            id: format!("literal.unused.{}", if good { "satisfied" } else { "mismatch" }),
            issue: 1377,
            source: format!("{definitions}\ndef main() -> tensor[1, f32] = {{\n  _ = f({}, {})\n  to_tensor([9.0f32])\n}}\n", literal(&seed), literal(&x)),
            signature: Some("(tensor[f32], tensor[rows, f32]) -> tensor[4, f32]"),
            exported: None,
            expected: if good { Expected::Tensor(vec![1], vec![9.0]) }
            else { Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"]) },
        });
        fixtures.push(Case {
            id: format!("literal.unused_existing.{}", if good { "satisfied" } else { "mismatch" }),
            issue: 1377,
            source: format!("{definitions}\ndef main() -> tensor[f32] = {{\n  b = {}\n  _ = f(b, {})\n  b\n}}\n", literal(&seed), literal(&x)),
            signature: Some("(tensor[f32], tensor[rows, f32]) -> tensor[4, f32]"),
            exported: None,
            expected: if good { Expected::Tensor(vec![], vec![7.0]) }
            else { Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"]) },
        });
        for (id, bindings, source) in [
            ("literal.tensor_alias", "y = x", "y"),
            ("literal.tensor_alias_chain", "y = x\n  z = y", "z"),
            ("literal.tensor_alias_borrow", "y = x", "&y"),
        ] {
            call_matrix(
                &mut fixtures,
                id,
                1377,
                &format!(
                    "def f(b: tensor[f32], x: tensor[rows, f32]) -> tensor[4, f32] = {{\n  {bindings}\n  insert(b, 0i32, shape({source}, 0i32))\n}}"
                ),
                "(tensor[f32], tensor[rows, f32]) -> tensor[4, f32]",
                vec![seed.clone(), x.clone()],
                if good {
                    Expected::Tensor(vec![4], vec![7.0; 4])
                } else {
                    Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"])
                },
            );
        }
        call_matrix(
            &mut fixtures,
            "literal.tensor_alias_shadow",
            1377,
            "def f(b: tensor[f32], x: tensor[rows, f32], z: tensor[cols, f32]) -> tensor[4, f32] = {\n  y = x\n  y = z\n  insert(b, 0i32, shape(y, 0i32))\n}",
            "(tensor[f32], tensor[rows, f32], tensor[cols, f32]) -> tensor[4, f32]",
            vec![seed, vector(if good { 5 } else { 4 }), x],
            if good {
                Expected::Tensor(vec![4], vec![7.0; 4])
            } else {
                Expected::Domain("load", &["claimed = 4", "z axis 0 = 5"])
            },
        );
    }
    assert_eq!(fixtures.len(), 34);
    let failures: Vec<_> = fixtures
        .iter()
        .flat_map(|case| contract_failures(case, &observe(case)))
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// #1619's executable exit: original comparison witnesses, exported calls,
/// top-level bindings, an inlined main, and a refuted runtime unit operand.
#[test]
fn singleton_broadcast_contract() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut fixtures: Vec<_> = cases()
        .into_iter()
        .filter(|case| case.issue == 1619)
        .collect();
    call_matrix(
        &mut fixtures,
        "broadcast.call",
        1619,
        "def f(xs: tensor[n, f32]) -> tensor[n, f32] = mul(xs, expand(to_tensor([5.0f32]), 0i32, shape(xs, 0i32)))",
        "(tensor[d0, f32]) -> tensor[d0, f32]",
        vec![vector(3)],
        Expected::Tensor(vec![3], vec![5.0, 10.0, 15.0]),
    );
    // Unit-claim preservation after inlining is B2b-1's transport obligation.
    // These calls exercise the runtime parameter before that transformation.
    for good in [true, false] {
        let b = Input {
            dims: vec![if good { 1 } else { 2 }],
            values: if good { vec![5.0] } else { vec![5.0, 6.0] },
        };
        let def = "def f(b: tensor[unit, f32], xs: tensor[n, f32]) -> tensor[n, f32] = mul(xs, expand(b, 0i32, shape(xs, 0i32)))";
        for exported in [true, false] {
            fixtures.push(Case {
                id: format!(
                    "broadcast.unit.{}.{}",
                    if exported { "export" } else { "binding" },
                    if good { "satisfied" } else { "mismatch" }
                ),
                issue: 1619,
                source: if exported {
                    def.to_string()
                } else {
                    format!("{def}\nout = f({}, {})\n", literal(&b), literal(&vector(3)))
                },
                signature: Some("(tensor[unit, f32], tensor[d0, f32]) -> tensor[d0, f32]"),
                expected: if good {
                    Expected::Tensor(vec![3], vec![5.0, 10.0, 15.0])
                } else {
                    Expected::Domain("load", &["claimed = 1", "b axis 0 = 2"])
                },
                exported: exported.then(|| vec![b.clone(), vector(3)]),
            });
        }
    }
    assert_eq!(fixtures.len(), 11);
    let mut failures = Vec::new();
    for fixture in fixtures {
        failures.extend(contract_failures(&fixture, &observe(&fixture)));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn acceptance_cannot_be_satisfied_by_equal_wrong_answers_or_missing_roots() {
    let case = Case {
        id: "negative-control".into(),
        issue: 1377,
        source: String::new(),
        signature: None,
        expected: Expected::Tensor(vec![4], vec![7.0; 4]),
        exported: None,
    };
    let run = json!({"stage":"execute", "success":true, "stdout":"out = tensor(shape=[5], data=[7, 7, 7, 7, 7])", "stderr":""});
    let mut observed = json!({"check":{"success":true,"score":1,"errors":[]},"eval":run,"c":run});
    assert_eq!(contract_failures(&case, &observed).len(), 2);
    for lane in ["eval", "c"] {
        observed[lane]["stdout"] = "out = tensor(shape=[4], data=[7, 7, 7, 7])".into();
    }
    assert!(contract_failures(&case, &observed).is_empty());
    observed["c"]["stage"] = "no_entry".into();
    assert_eq!(contract_failures(&case, &observed).len(), 1);
    observed["eval"]["stdout"] = "".into();
    assert_eq!(contract_failures(&case, &observed).len(), 2);
}

#[test]
fn claims_and_traps_require_their_own_evidence() {
    let mut case = Case {
        id: "claim-control".into(),
        issue: 1377,
        source: String::new(),
        signature: Some("() -> tensor[4, f32]"),
        expected: Expected::Tensor(vec![4], vec![7.0; 4]),
        exported: None,
    };
    let ok = json!({"stage":"execute", "success":true, "stdout":"out = tensor(shape=[4], data=[7, 7, 7, 7])", "stderr":""});
    let mut observed = json!({"check":{"success":true,"score":1,"errors":[],"signatures":{"f":"() -> tensor[*, f32]"}},"eval":ok,"c":ok});
    assert_eq!(
        contract_failures(&case, &observed).len(),
        1,
        "a correct value does not restore an erased declaration"
    );
    observed["check"]["signatures"]["f"] = "() -> tensor[4, f32]".into();
    assert!(contract_failures(&case, &observed).is_empty());
    case.expected = Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"]);
    for lane in ["eval", "c"] {
        observed[lane] = json!({"stage":"execute", "success":false,"stdout":"",
            "stderr":"extent `4`: claimed = 4, x axis 0 = 5\nnumeric trap: domain in load at int64"});
    }
    assert!(contract_failures(&case, &observed).is_empty());
    observed["eval"]["stderr"] =
        "extent `4`: claimed = 4, x axis 0 = 50\nnumeric trap: domain in load at int64".into();
    assert_eq!(
        contract_failures(&case, &observed).len(),
        1,
        "50 is not the observed extent 5"
    );
    observed["eval"] = observed["c"].clone();
    observed["c"]["stage"] = "build".into();
    assert_eq!(
        contract_failures(&case, &observed).len(),
        1,
        "build refusal is not the runtime guard"
    );
    observed["eval"]["stderr"] = "numeric trap: domain in load at int64".into();
    assert_eq!(
        contract_failures(&case, &observed).len(),
        2,
        "source and observed-value context is required"
    );
}

#[test]
fn trap_context_keeps_source_axis_and_signed_value_together() {
    let case = cases()
        .into_iter()
        .find(|case| case.id == "literal.export.mismatch")
        .unwrap();
    let mut observed = observe(&case);
    // Isolate diagnostic acceptance from the separately recorded signature
    // erasure. This does not claim that the compiler preserved the signature.
    observed["check"]["signatures"]["f"] = case.signature.unwrap().into();
    assert!(contract_failures(&case, &observed).is_empty());
    for context in [
        "claimed = 4, x axis 1 = 5",
        "claimed = 5, x axis 0 = 4",
        "claimed = 4, x axis 0 = -5",
        "claimed = 4, y axis 0 = 5",
        "claimed = 4, x axis 0 = 50",
    ] {
        observed["c"]["stderr"] =
            format!("{context}\nnumeric trap: domain in load at int64").into();
        assert_eq!(
            contract_failures(&case, &observed).len(),
            1,
            "wrong context accepted: {context}"
        );
    }
    observed["c"]["stderr"] =
        "x axis 0: 5\nclaimed: 4\nnumeric trap: domain in load at int64".into();
    assert!(
        contract_failures(&case, &observed).is_empty(),
        "context punctuation and record order are not normative"
    );
}
