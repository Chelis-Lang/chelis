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
    TensorF32Bits(Vec<usize>, Vec<u32>),
    Domain(&'static str, &'static [&'static str]),
    TargetDivisionByZero,
    ExactTargetDivisionByZero,
    EntryShapeMismatch(&'static str),
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
    let good = matches!(result, Expected::Tensor(..) | Expected::TensorF32Bits(..));
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

fn driver(inputs: &[Input], header: &str) -> (String, Vec<String>) {
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
    if header
        .lines()
        .any(|line| line.starts_with("chelis_tensor* f("))
    {
        text.push_str(&format!("chelis_tensor *result = f({args});\n"));
    } else {
        // A single pure tensor definition is exported through the named
        // kernel ABI. Exercise that public entry rather than inventing a
        // host wrapper which the compiler did not emit.
        assert_eq!(
            header.trim(),
            "void fixture(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);",
            "unknown exported fixture ABI"
        );
        text.push_str(&format!(
            "chelis_tensor *inputs[] = {{{args}}};\nchelis_tensor *result = NULL;\nfixture(inputs, {}, &result, 1);\n",
            inputs.len()
        ));
    }
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
    observe_with_dependency(case, None)
}

fn observe_with_dependency(case: &Case, library: Option<&str>) -> Value {
    observe_host_lane(case, library, "c", false)
}

fn observe_host_lane(case: &Case, library: Option<&str>, target: &str, api: bool) -> Value {
    let dir = tempdir().expect("fixture directory");
    if let Some(library) = library {
        let version = chelis_compiler_api::COMPILER_VERSION;
        fs::create_dir_all(dir.path().join("mylib/src")).unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(
            dir.path().join("src/placeholder.ch"),
            "module App.Placeholder\ndef placeholder() -> int32 = 0i32\n",
        )
        .unwrap();
        fs::write(dir.path().join("reef.toml"), format!("[package]\nname = \"app\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\nmodule_prefix = \"App\"\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n")).unwrap();
        fs::write(dir.path().join("mylib/reef.toml"), format!("[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\nmodule_prefix = \"Mylib\"\n")).unwrap();
        fs::write(
            dir.path().join("mylib/src/claims.ch"),
            format!("module Mylib.Claims\nexport (f)\n{library}\n"),
        )
        .unwrap();
        fs::write(dir.path().join("reef.lock"), format!("[package]\nname = \"app\"\nversion = \"0.1.0\"\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n")).unwrap();
    }
    let path = dir.path().join(if library.is_some() {
        "src/fixture.ch"
    } else {
        "fixture.ch"
    });
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
        report["components"]["parse"], 1.0,
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
            if api { "c" } else { target },
            "-o",
            out.to_str().unwrap(),
        ],
        dir.path(),
    );
    let compiled = if !build.status.success() {
        receipt("build", &build)
    } else {
        let generated = if api {
            use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
            // The CLI invocation above supplies runtime headers/archive only.
            // Every compiled source byte below comes from the API artifact.
            match chelis_compiler_api::compiler::compile(CompileRequest {
                source_kind: SourceKind::Surf,
                source: case.source.clone(),
                target: CompileTarget::Hip,
                entry_name: None,
            }) {
                Ok(artifact) => {
                    let source = artifact
                        .files
                        .iter()
                        .find(|f| f.path.ends_with("_hip.cpp"))
                        .expect("HIP source")
                        .contents
                        .clone();
                    let header = artifact
                        .files
                        .iter()
                        .find(|f| f.path.ends_with("_hip.h"))
                        .expect("HIP header")
                        .contents
                        .clone();
                    for file in artifact.files {
                        fs::write(out.join(file.path), file.contents).unwrap();
                    }
                    Ok((source, header))
                }
                Err(error) => Err(format!("{error:?}")),
            }
        } else {
            let (source, header) = if target == "hip" {
                ("fixture_hip.cpp", "fixture_hip.h")
            } else {
                ("fixture.c", "fixture.h")
            };
            Ok((
                fs::read_to_string(out.join(source)).expect("generated source"),
                fs::read_to_string(out.join(header)).expect("generated header"),
            ))
        };
        let (mut source, header) = match generated {
            Ok(files) => files,
            Err(error) => {
                return json!({"issue":case.issue, "check":checker, "eval":eval,
                "c":{"stage":"build", "success":false, "stdout":"", "stderr":error}});
            }
        };
        if target == "hip" {
            assert!(
                !source.contains("__global__") && !source.contains("hipLaunchKernelGGL"),
                "{} must retain host C",
                case.id
            );
        }
        let c_path = out.join("fixture.c");
        fs::write(&c_path, &source).expect("write selected host source");
        let args = if let Some(inputs) = &case.exported {
            // A nullary helper also creates an observation entry. Exercise
            // f through the independent exported caller below.
            source = source.replacen("int main(", "int fixture_generated_main(", 1);
            let (driver, args) = driver(inputs, &header);
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
    if check["success"] != true || check["score"] != 1.0 || check["errors"] != json!([]) {
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
            Expected::TensorF32Bits(shape, bits) => {
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
                    .is_some_and(|(actual_shape, data)| {
                        actual_shape == *shape
                            && data
                                .iter()
                                .map(|value| (*value as f32).to_bits())
                                .collect::<Vec<_>>()
                                == *bits
                    })
            }
            Expected::TargetDivisionByZero => {
                // The existing C integer helper uses its older diagnostic.
                // Assert each lane's arithmetic failure independently: an
                // earlier reshape claim failure does not satisfy this case.
                run["stage"] == "execute"
                    && run["success"] == false
                    && stderr.lines().any(|line| {
                        line.strip_prefix("error: ").unwrap_or(line)
                            == if lane == "eval" {
                                "numeric trap: division by zero in floor_div at int64"
                            } else {
                                "integer division or remainder by zero"
                            }
                    })
            }
            Expected::ExactTargetDivisionByZero => {
                run["stage"] == "execute"
                    && run["success"] == false
                    && stderr.lines().any(|line| {
                        line.strip_prefix("error: ").unwrap_or(line)
                            == "numeric trap: division by zero in floor_div at int64"
                    })
            }
            Expected::EntryShapeMismatch(context) => {
                run["stage"] == "execute" && run["success"] == false && stderr.contains(context)
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
        "behavior changed in {changed:?}: compare each cell with its contract before updating the issue-owned baseline. Observations: {}",
        serde_json::to_string_pretty(
            &changed
                .iter()
                .map(|id| ((*id).clone(), observed[*id].clone()))
                .collect::<serde_json::Map<_, _>>()
        )
        .unwrap()
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
    let mut observed = json!({"check":{"success":true,"score":1.0,"errors":[]},"eval":run,"c":run});
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
    let mut observed = json!({"check":{"success":true,"score":1.0,"errors":[],"signatures":{"f":"() -> tensor[*, f32]"}},"eval":ok,"c":ok});
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

#[test]
fn helper_signature_guard_order_contract() {
    let mut fixtures = Vec::new();
    let seed = Input {
        dims: vec![],
        values: vec![7.0],
    };
    for (z, a, expected) in [
        (4, 3, Expected::Tensor(vec![4, 3], vec![7.0; 12])),
        (
            5,
            3,
            Expected::Domain("load", &["claimed = 4", "z axis 0 = 5"]),
        ),
        (
            5,
            6,
            Expected::Domain("load", &["claimed = 4", "z axis 0 = 5"]),
        ),
        (
            4,
            6,
            Expected::Domain("load", &["claimed = 3", "a axis 0 = 6"]),
        ),
    ] {
        call_matrix(
            &mut fixtures,
            &format!("helper_order.multi.{z}.{a}"),
            1277,
            "def f(b: tensor[f32], z: tensor[rows, f32], a: tensor[cols, f32]) -> tensor[4, 3, f32] = insert(insert(b, 0i32, shape(a, 0i32)), 0i32, shape(z, 0i32))",
            "(tensor[f32], tensor[rows, f32], tensor[cols, f32]) -> tensor[4, 3, f32]",
            vec![seed.clone(), vector(z), vector(a)],
            expected,
        );
    }
    for (x, z, expected) in [
        (4, 4, Expected::Tensor(vec![], vec![7.0])),
        (
            4,
            5,
            Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"]),
        ),
        (
            5,
            6,
            Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"]),
        ),
    ] {
        call_matrix(
            &mut fixtures,
            &format!("helper_order.repeated.{x}.{z}"),
            1277,
            "def g(b: tensor[f32], x: tensor[n, f32]) -> tensor[4, f32] = insert(b, 0i32, shape(x, 0i32))\ndef f(b: tensor[f32], x: tensor[rows, f32], z: tensor[cols, f32]) -> tensor[f32] = {\n  _ = g(b, x)\n  _ = g(b, z)\n  b\n}",
            "(tensor[f32], tensor[rows, f32], tensor[cols, f32]) -> tensor[f32]",
            vec![seed.clone(), vector(x), vector(z)],
            expected,
        );
    }
    for (actual, expected) in [
        (4, Expected::Tensor(vec![4], vec![7.0; 4])),
        (
            5,
            Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"]),
        ),
    ] {
        call_matrix(
            &mut fixtures,
            &format!("helper_order.scalar_alias.{actual}"),
            1277,
            "def f(b: tensor[f32], x: tensor[rows, f32]) -> tensor[4, f32] = {\n  n = shape(x, 0i32)\n  m = n\n  insert(b, 0i32, m)\n}",
            "(tensor[f32], tensor[rows, f32]) -> tensor[4, f32]",
            vec![seed.clone(), vector(actual)],
            expected,
        );
    }
    for (z, a, q, expected) in [
        (2, 3, 2, Expected::Tensor(vec![2, 3, 2], vec![7.0; 12])),
        (
            5,
            6,
            5,
            Expected::Domain("load", &["claimed = 2", "z axis 0 = 5"]),
        ),
        (
            2,
            6,
            5,
            Expected::Domain("load", &["claimed = 3", "a axis 0 = 6"]),
        ),
        (
            2,
            6,
            2,
            Expected::Domain("load", &["claimed = 3", "a axis 0 = 6"]),
        ),
        (
            2,
            3,
            5,
            Expected::Domain("load", &["claimed = 2", "q axis 0 = 5"]),
        ),
    ] {
        call_matrix(
            &mut fixtures,
            &format!("interleaved.{z}.{a}.{q}"),
            1277,
            "def f(b: tensor[f32], z: tensor[rows, f32], a: tensor[cols, f32], q: tensor[depth, f32]) -> tensor[2, 3, 2, f32] = insert(insert(insert(b, 0i32, shape(q, 0i32)), 0i32, shape(a, 0i32)), 0i32, shape(z, 0i32))",
            "(tensor[f32], tensor[rows, f32], tensor[cols, f32], tensor[depth, f32]) -> tensor[2, 3, 2, f32]",
            vec![
                Input {
                    dims: vec![],
                    values: vec![7.0],
                },
                vector(z),
                vector(a),
                vector(q),
            ],
            expected,
        );
    }
    let start = fixtures.len();
    for (q, r, expected) in [
        (3, 2, Expected::Tensor(vec![], vec![18.0])),
        (
            4,
            5,
            Expected::Domain("load", &["a axis 0 = 3", "q axis 0 = 4"]),
        ),
        (
            3,
            5,
            Expected::Domain("load", &["z axis 0 = 2", "r axis 0 = 5"]),
        ),
    ] {
        call_matrix(
            &mut fixtures,
            &format!("helper_order.named.{q}.{r}"),
            1277,
            "def f(z: tensor[rows, f32], a: tensor[cols, f32], q: tensor[cols, f32], r: tensor[rows, f32]) -> tensor[f32] = add(sum(add(z, r), 0i32), sum(add(a, q), 0i32))",
            "(tensor[rows, f32], tensor[cols, f32], tensor[cols, f32], tensor[rows, f32]) -> tensor[f32]",
            vec![vector(2), vector(3), vector(q), vector(r)],
            expected,
        );
    }
    for (a, q, expected) in [
        (
            3,
            2,
            Expected::Tensor(vec![2, 3], vec![2.0, 2.0, 2.0, 4.0, 4.0, 4.0]),
        ),
        (
            6,
            5,
            Expected::Domain("load", &["claimed = 3", "a axis 0 = 6"]),
        ),
        (
            3,
            5,
            Expected::Domain("load", &["z axis 0 = 2", "q axis 0 = 5"]),
        ),
    ] {
        call_matrix(
            &mut fixtures,
            &format!("helper_order.mixed.{a}.{q}"),
            1277,
            "def f(z: tensor[rows, f32], a: tensor[cols, f32], q: tensor[rows, f32]) -> tensor[rows, 3, f32] = insert(add(z, q), 1i32, shape(a, 0i32))",
            "(tensor[rows, f32], tensor[cols, f32], tensor[rows, f32]) -> tensor[rows, 3, f32]",
            vec![vector(2), vector(a), vector(q)],
            expected,
        );
    }
    let bindings = fixtures
        .drain(start..)
        .filter(|case| case.exported.is_none() && !case.source.contains("def main()"))
        .collect::<Vec<_>>();
    fixtures.extend(bindings);
    let example = include_str!("../../../examples/ordered_extent_claims.ch");
    for (id, source, expected) in [
        (
            "match",
            example.to_owned(),
            Expected::Tensor(vec![4, 3], vec![7.0; 12]),
        ),
        (
            "mismatch",
            example
                .replace("3.0f32, 4.0f32])", "3.0f32, 4.0f32, 5.0f32])")
                .replace(
                    "2.0f32, 3.0f32]))",
                    "2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]))",
                ),
            Expected::Domain("load", &["claimed = 4", "z axis 0 = 5"]),
        ),
    ] {
        fixtures.push(Case {
            id: format!("helper_order.example.{id}"),
            issue: 1277,
            source,
            signature: None,
            expected,
            exported: None,
        });
    }
    let mut failures = Vec::new();
    for case in &fixtures {
        let observed = observe(case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(case, &observed));
    }
    assert_eq!(fixtures.len(), 50);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Result claims follow axis provenance, including an inferred helper result.
/// Each call route checks the declared type independently of either runtime.
#[test]
fn static_reshape_folding_accepts_producing_source_expressions() {
    producing_source_expression_contract("floor_div", 1);
}

#[test]
fn remainder_reshape_claims_preserve_producing_source_expressions() {
    producing_source_expression_contract("mod", 4);
}

fn producing_source_expression_contract(op: &str, divisor: i64) {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [2, 3] {
        let source = literal(&vector(n));
        for (kind, helper, operand) in [
            ("literal", String::new(), source.clone()),
            ("helper", format!("def g() = {source}\n"), "g()".to_owned()),
            (
                "checked_helper",
                "def g(y: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(y, [floor_div(shape(y, 0i32), 2i64), 2i64])\n".to_owned(),
                format!("g({})", literal(&vector(n * 2))),
            ),
            (
                "inferred_computed_helper",
                "def g(y: tensor[n, f32]) = reshape(y, [floor_div(shape(y, 0i32), 2i64), 2i64])\n".to_owned(),
                format!("g({})", literal(&vector(n * 2))),
            ),
        ] {
            call_matrix(
                &mut cases,
                &format!("static_reshape_expression.{op}.{kind}.x{n}"),
                1686,
                &format!("{helper}def f(x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [{op}(shape({operand}, 0i32), {divisor}i64), 2i64])"),
                "(tensor[d0, f32]) -> tensor[2, 2, f32]",
                vec![vector(4)],
                if n == 2 { Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]) }
                else { Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"]) },
            );
        }
    }
    assert_eq!(cases.len(), 24);
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
        if case.source.contains("def main()")
            && observed["check"]["signatures"]["main"] != "() -> tensor[2, 2, f32]"
        {
            failures.push(format!("{}: declared main shape changed", case.id));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn remainder_reshape_claims_preserve_dynamic_target() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [2, 3] {
        call_matrix(
            &mut cases,
            &format!("remainder_reshape_dynamic.x{n}"),
            1686,
            "def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [mod(shape(source, 0i32), 4i64), 2i64])",
            "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
            vec![vector(n), vector(n * 2)],
            if n == 2 {
                Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            } else {
                Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
            },
        );
    }
    assert_eq!(cases.len(), 6);
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Spec/04 §4.7.3 admits any correctly typed int64 target producer. A host
/// boundary must retain the claim just as the tensor-DAG boundary does.
#[test]
fn host_produced_reshape_targets_preserve_declared_claims() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for (producer, declarations, target) in [
        ("bitwise", "", "bitand(shape(source, 0i32), 3i64)"),
        ("metadata", "", "numel(source)"),
        ("list", "", "len(to_list(source))"),
        (
            "helper",
            "def size(y: tensor[m, f32]) -> int64 = numel(y)\n",
            "size(source)",
        ),
        (
            "scalar_branch",
            "",
            "if eq(shape(source, 0i32), 2i64) then 2i64 else 3i64",
        ),
    ] {
        for n in [2, 3] {
            for (wrapper, body) in [
                ("direct", format!("reshape(x, [{target}, 2i64])")),
                (
                    "bound_copy",
                    format!(
                        "{{\n  size = {target}\n  value = reshape(x, [size, 2i64])\n  copy(value)\n}}"
                    ),
                ),
            ] {
                call_matrix(
                    &mut cases,
                    &format!("host_target.{producer}.{wrapper}.x{n}"),
                    1686,
                    &format!(
                        "{declarations}def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {body}"
                    ),
                    "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
                    vec![vector(n), vector(n * 2)],
                    if n == 2 {
                        Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
                    } else {
                        Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
                    },
                );
            }
        }
    }
    assert_eq!(cases.len(), 60);
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn staged_reshape_sources_preserve_captures_and_order() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [2, 3] {
        for (kind, declarations, body) in [
            (
                "local_list",
                "",
                "{\n  items = to_list(source)\n  size = len(items)\n  reshape(x, [size, 2i64])\n}",
            ),
            (
                "tensor_capture",
                "",
                "{\n  doubled = add(source, source)\n  reshape(x, [bitand(numel(doubled), 3i64), 2i64])\n}",
            ),
            (
                "scalar_capture_shadow",
                "def g(source: int64, x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [bitand(source, 3i64), 2i64])\n",
                "g(shape(source, 0i32), x)",
            ),
            (
                "scalar_helper_alias",
                "def extent(y: tensor[m, f32]) -> int64 = numel(y)\n",
                "{\n  target = extent\n  reshape(x, [target(source), 2i64])\n}",
            ),
            (
                "scalar_helper_alias_nested",
                "def extent(y: tensor[m, f32]) -> int64 = numel(y)\n",
                "{\n  target = extent\n  reshape(x, [bitand(target(source), 3i64), 2i64])\n}",
            ),
            (
                "scalar_helper_alias_shadow",
                "def extent(y: tensor[m, f32]) -> int64 = numel(y)\n",
                "{\n  target = extent\n  extent = to_list(source)\n  reshape(x, [bitand(target(source), len(extent)), 2i64])\n}",
            ),
            (
                "repeated_calls",
                "def g(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [bitand(numel(source), 3i64), 2i64])\n",
                "{\n  first = g(source, x)\n  second = g(source, x)\n  copy(first)\n}",
            ),
        ] {
            call_matrix(
                &mut cases,
                &format!("staged_source.{kind}.x{n}"),
                1686,
                &format!(
                    "{declarations}def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {body}"
                ),
                "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
                vec![vector(n), vector(n * 2)],
                if n == 2 {
                    Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
                } else {
                    Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
                },
            );
        }
        call_matrix(
            &mut cases,
            &format!("staged_source.untaken_failure.x{n}"),
            1686,
            "def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [if eq(numel(source), 2i64) then 2i64 else floor_div(numel(source), sub(numel(source), numel(source))), 2i64])",
            "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
            vec![vector(n), vector(n * 2)],
            if n == 2 {
                Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            } else {
                Expected::ExactTargetDivisionByZero
            },
        );
        call_matrix(
            &mut cases,
            &format!("staged_source.discarded.x{n}"),
            1686,
            "def g(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [bitand(numel(source), 3i64), 2i64])\ndef f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[n, f32] = {\n  ignored = g(source, x)\n  x\n}",
            "(tensor[d0, f32], tensor[d1, f32]) -> tensor[d1, f32]",
            vec![vector(n), vector(n * 2)],
            if n == 2 {
                Expected::Tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0])
            } else {
                Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
            },
        );
    }
    for (kind, targets) in [
        (
            "host_then_dag",
            "bitand(shape(source, 0i32), 3i64), floor_div(shape(x, 0i32), sub(shape(x, 0i32), shape(x, 0i32)))",
        ),
        (
            "dag_then_host",
            "floor_div(shape(x, 0i32), sub(shape(x, 0i32), shape(x, 0i32))), bitand(shape(source, 0i32), 3i64)",
        ),
    ] {
        call_matrix(
            &mut cases,
            &format!("staged_source.{kind}"),
            1686,
            &format!(
                "def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [{targets}])"
            ),
            "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
            vec![vector(3), vector(6)],
            Expected::TargetDivisionByZero,
        );
    }
    assert_eq!(cases.len(), 60);
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Spec/04 §4.7.3 and [05-OP-50]: packing a local tuple and changing a
/// rank-zero value's scalar surface must preserve its actual extent source.
#[test]
fn staged_sources_preserve_tuple_captures() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [4, 6] {
        for (kind, body) in [
            ("tuple", "{\n  sizes = (floor_div(numel(x), 2i64), 2i64)\n  reshape(x, [sizes.0, sizes.1])\n}"),
            ("nested_tuple", "{\n  sizes = ((floor_div(numel(x), 2i64), 2i64), x)\n  reshape(sizes.1, [sizes.0.0, sizes.0.1])\n}"),
            ("tuple_host_consumer", "{\n  sizes = (floor_div(numel(x), 2i64), 2i64)\n  reshape(x, [bitand(sizes.0, 3i64), sizes.1])\n}"),
        ] {
            call_matrix(
                &mut cases,
                &format!("staged_source.{kind}.x{n}"),
                1686,
                &format!("def f(x: tensor[n, f32]) -> tensor[2, 2, f32] = {body}"),
                "(tensor[d0, f32]) -> tensor[2, 2, f32]",
                vec![vector(n)],
                if n == 4 { Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]) }
                else { Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"]) },
            );
        }
    }
    assert_eq!(cases.len(), 18);
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn staged_sources_preserve_scalar_and_tensor_views() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [4, 6] {
        let mut input = vector(n);
        if n == 6 {
            input.values[5] = 8.0; // sum 23; 23 & 3 is the incompatible extent 3.
        }
        for (kind, body) in [
            ("scalar_view", "{\n  total = tensor_to_scalar(sum(x, 0i32))\n  size = bitand(cast(total, int64), 3i64)\n  reshape(x, [size, 2i64])\n}"),
            ("retained_views", "{\n  tensor_total = sum(x, 0i32)\n  total = tensor_to_scalar(tensor_total)\n  tensor_again = scalar_to_tensor(total)\n  size = mul(bitand(cast(total, int64), 3i64), mul(numel(tensor_total), numel(tensor_again)))\n  reshape(x, [size, 2i64])\n}"),
        ] {
            call_matrix(
                &mut cases,
                &format!("staged_source.{kind}.x{n}"),
                1686,
                &format!("def f(x: tensor[n, f32]) -> tensor[2, 2, f32] = {body}"),
                "(tensor[d0, f32]) -> tensor[2, 2, f32]",
                vec![input.clone()],
                if n == 4 { Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]) }
                else { Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"]) },
            );
        }
    }
    assert_eq!(cases.len(), 12);
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn staged_graph_segments_preserve_eager_sources() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [2, 3] {
        call_matrix(
            &mut cases,
            &format!("staged_source.eager_unused.x{n}"),
            1686,
            "def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {\n  unused = floor_div(shape(x, 0i32), sub(3i64, shape(source, 0i32)))\n  reshape(x, [bitand(numel(source), 3i64), 2i64])\n}",
            "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
            vec![vector(n), vector(n * 2)],
            if n == 2 {
                Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            } else {
                Expected::TargetDivisionByZero
            },
        );
    }
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn staged_reshape_sources_preserve_signature_witnesses() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [2, 3] {
        call_matrix(
            &mut cases,
            &format!("staged_source.signature_scope.x{n}"),
            1686,
            "def g(source: tensor[rows, f32], x: tensor[n, f32]) -> tensor[rows, 2, f32] = reshape(x, [bitand(numel(source), 3i64), 2i64])\ndef f(unread: tensor[rows, f32], source: tensor[m, f32], x: tensor[n, f32]) -> tensor[rows, 2, f32] = copy(g(source, x))",
            // The checked type expresses the equality required by the nested
            // result. The original parameters must still supply two witnesses.
            "(tensor[rows, f32], tensor[rows, f32], tensor[d0, f32]) -> tensor[rows, 2, f32]",
            vec![vector(2), vector(n), vector(n * 2)],
            if n == 2 {
                Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            } else {
                Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
            },
        );
    }
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn staged_sources_preserve_handled_random_progress() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for seed in [42u64, 43] {
        for n in [2, 3] {
            for (kind, target, draw) in [
                ("native", "bitand(numel(source), 3i64)", 1u64),
                (
                    "mixed",
                    "if eq(numel(source), 2i64) then floor_div(numel(uniform_like(x, 2.0f32, 5.0f32)), 2i64) else 3i64",
                    2u64,
                ),
            ] {
                let expected = if n == 2 {
                    // Reference the numeric sampler directly, independently of
                    // either compiler lane's staging, seed and draw scheduling.
                    let effective_seed = seed ^ draw.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    Expected::TensorF32Bits(
                        vec![2, 2],
                        (0..4)
                            .map(|index| {
                                (chelis_types::uniform_sample(
                                    chelis_types::types::Prim::F32,
                                    2.0,
                                    5.0,
                                    effective_seed,
                                    index,
                                )
                                .unwrap()
                                .as_f64_lossy() as f32)
                                    .to_bits()
                            })
                            .collect(),
                    )
                } else {
                    Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
                };
                call_matrix(
                    &mut cases,
                    &format!("staged_source.random.{kind}.seed{seed}.x{n}"),
                    1686,
                    &format!(
                        "def g(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] ! {{ Random }} = {{\n  first = uniform_like(x, 2.0f32, 5.0f32)\n  size = {target}\n  second = uniform_like(x, 2.0f32, 5.0f32)\n  reshape(second, [size, 2i64])\n}}\ndef f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = with seed({seed}i64) {{ g(source, x) }}"
                    ),
                    "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
                    vec![vector(n), vector(n * 2)],
                    expected,
                );
            }
        }
    }
    let mut failures = Vec::new();
    for case in cases {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn remainder_claims_preserve_hip_host_cli_and_api_execution() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [2, 3] {
        call_matrix(
            &mut cases,
            &format!("remainder_hip_host.x{n}"),
            1686,
            "def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [mod(shape(source, 0i32), 4i64), 2i64])",
            "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
            vec![vector(n), vector(n * 2)],
            if n == 2 {
                Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            } else {
                Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
            },
        );
    }
    let mut failures = Vec::new();
    for case in cases {
        for api in [false, true] {
            let observed = observe_host_lane(&case, None, "hip", api);
            println!("{} api={api}: {}", case.id, observed);
            failures.extend(
                contract_failures(&case, &observed)
                    .into_iter()
                    .map(|f| format!("api={api}: {f}")),
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn staged_claims_preserve_hip_host_cli_and_api_execution() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [2, 3] {
        call_matrix(
            &mut cases,
            &format!("staged_hip_host.x{n}"),
            1686,
            "def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [bitand(numel(source), 3i64), 2i64])",
            "(tensor[d0, f32], tensor[d1, f32]) -> tensor[2, 2, f32]",
            vec![vector(n), vector(n * 2)],
            if n == 2 {
                Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            } else {
                Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
            },
        );
    }
    let mut failures = Vec::new();
    for case in cases {
        for api in [false, true] {
            let observed = observe_host_lane(&case, None, "hip", api);
            println!("{} api={api}: {}", case.id, observed);
            failures.extend(contract_failures(&case, &observed));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn static_reshape_folding_preserves_declaring_input_contract() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut cases = Vec::new();
    for n in [2, 3] {
        call_matrix(
            &mut cases,
            &format!("static_reshape_source.x{n}"),
            1686,
            "def f(unread: tensor[2, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [mod(shape(unread, 0i32), 4i64), 2i64])",
            "(tensor[2, f32], tensor[d0, f32]) -> tensor[2, 2, f32]",
            vec![vector(n), vector(4)],
            if n == 2 {
                Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            } else {
                Expected::Domain("load", &["claimed = 2", "unread axis 0 = 3"])
            },
        );
    }
    assert_eq!(cases.len(), 6);
    let mut failures = Vec::new();
    for mut case in cases {
        if case.id.contains("x3.") {
            case.expected = if case.exported.is_some() {
                Expected::EntryShapeMismatch("input `unread` axis 0 expected 2, got 3")
            } else {
                // Concrete incompatible arguments are static type errors;
                // only the externally supplied input reaches an ABI check.
                Expected::Reject("dimension mismatch")
            };
        }
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
        if !matches!(case.expected, Expected::Reject(_))
            && case.source.contains("def main()")
            && observed["check"]["signatures"]["main"] != "() -> tensor[2, 2, f32]"
        {
            failures.push(format!("{}: declared main shape changed", case.id));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn computed_claim_complete_shape_list_precedes_guards() {
    // Spec/04 §4.7 places local guards at the introducing operation;
    // §4.7.3 requires the complete shape list to evaluate first. Both a
    // matching and mismatching first axis must reach a later target failure.
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut fixtures = Vec::new();
    for (kind, claim, second, n, divisor, denominator, signature, expected) in [
        (
            "literal_later_trap",
            "4",
            2,
            4,
            2,
            "sub(shape(x, 0i32), 4i64)",
            "(tensor[d0, f32]) -> tensor[4, 2, f32]",
            Expected::TargetDivisionByZero,
        ),
        (
            "literal_positive",
            "4",
            2,
            8,
            2,
            "sub(shape(x, 0i32), 4i64)",
            "(tensor[d0, f32]) -> tensor[4, 2, f32]",
            Expected::Tensor(vec![4, 2], (1..=8).map(f64::from).collect()),
        ),
        (
            "literal_claim",
            "4",
            2,
            6,
            2,
            "sub(shape(x, 0i32), 4i64)",
            "(tensor[d0, f32]) -> tensor[4, 2, f32]",
            Expected::Domain("reshape", &["claimed = 4", "reshape axis 0 = 3"]),
        ),
        (
            "named_bad_first_later_trap",
            "n",
            2,
            4,
            2,
            "sub(shape(x, 0i32), 4i64)",
            "(tensor[d0, f32]) -> tensor[d0, 2, f32]",
            Expected::TargetDivisionByZero,
        ),
        (
            "named_good_first_later_trap",
            "n",
            2,
            4,
            1,
            "sub(shape(x, 0i32), 4i64)",
            "(tensor[d0, f32]) -> tensor[d0, 2, f32]",
            Expected::TargetDivisionByZero,
        ),
        (
            "named_positive",
            "n",
            1,
            4,
            1,
            "shape(x, 0i32)",
            "(tensor[d0, f32]) -> tensor[d0, 1, f32]",
            Expected::Tensor(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "named_claim",
            "n",
            1,
            4,
            2,
            "shape(x, 0i32)",
            "(tensor[d0, f32]) -> tensor[d0, 1, f32]",
            Expected::Domain("reshape", &["claimed = 4", "reshape axis 0 = 2"]),
        ),
    ] {
        let reshape = format!(
            "reshape(x, [floor_div(shape(x, 0i32), {divisor}i64), floor_div(shape(x, 0i32), {denominator})])"
        );
        for (wrapper, body) in [
            ("direct", reshape.clone()),
            ("copy", format!("copy({reshape})")),
        ] {
            let mut routes = Vec::new();
            call_matrix(
                &mut routes,
                &format!("complete_shape_list.{kind}.{wrapper}"),
                1686,
                &format!("def f(x: tensor[n, f32]) -> tensor[{claim}, {second}, f32] = {body}"),
                signature,
                vec![vector(n)],
                expected.clone(),
            );
            for case in routes {
                let rows = if claim == "n" { n } else { 4 };
                let main = case
                    .source
                    .contains("def main()")
                    .then(|| format!("() -> tensor[{rows}, {second}, f32]"));
                fixtures.push((case, main));
            }
        }
    }
    // Only the first result axis is claimed. If the result is discarded,
    // retaining that claim must also retain the later unclaimed target's
    // computation, including a possible arithmetic failure.
    for n in [4, 8] {
        let mut routes = Vec::new();
        call_matrix(
            &mut routes,
            &format!("complete_shape_list.discarded.x{n}"),
            1686,
            "def g(x: tensor[n, f32]) -> tensor[4, *, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), floor_div(shape(x, 0i32), sub(shape(x, 0i32), 4i64))])\ndef f(x: tensor[n, f32]) -> tensor[f32] = {\n  discarded = g(x)\n  scalar_to_tensor(9.0f32)\n}",
            "(tensor[d0, f32]) -> tensor[f32]",
            vec![vector(n)],
            if n == 4 {
                Expected::TargetDivisionByZero
            } else {
                Expected::Tensor(vec![], vec![9.0])
            },
        );
        for case in routes {
            let main = case
                .source
                .contains("def main()")
                .then(|| "() -> tensor[f32]".to_owned());
            fixtures.push((case, main));
        }
    }
    assert_eq!(fixtures.len(), 48);
    let mut failures = Vec::new();
    for (case, main_signature) in fixtures {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
        if let Some(signature) = main_signature
            && observed["check"]["signatures"]["main"] != signature
        {
            failures.push(format!(
                "{} main signature: {} != {signature}",
                case.id, observed["check"]["signatures"]["main"]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn computed_claim_result_graph_contract() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut fixtures = Vec::new();
    for (kind, claim, divisor, n, signature, expected) in [
        (
            "literal",
            "2",
            2,
            4,
            "(tensor[d0, f32]) -> tensor[2, 2, f32]",
            Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "literal",
            "2",
            2,
            6,
            "(tensor[d0, f32]) -> tensor[2, 2, f32]",
            Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"]),
        ),
        (
            "named",
            "n",
            1,
            4,
            "(tensor[d0, f32]) -> tensor[d0, 1, f32]",
            Expected::Tensor(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "named",
            "n",
            2,
            4,
            "(tensor[d0, f32]) -> tensor[d0, 2, f32]",
            Expected::Domain("reshape", &["claimed = 4", "reshape axis 0 = 2"]),
        ),
    ] {
        let reshape =
            format!("reshape(x, [floor_div(shape(x, 0i32), {divisor}i64), {divisor}i64])");
        for (form, helper, body) in [
            ("copy", String::new(), format!("copy({reshape})")),
            ("neg", String::new(), format!("neg({reshape})")),
            (
                "static_if",
                String::new(),
                format!("if true then {reshape} else {reshape}"),
            ),
            (
                "helper",
                format!("def g(x: tensor[n, f32]) = {reshape}\n"),
                "g(x)".to_owned(),
            ),
            (
                "alias_helper",
                format!("def g(x: tensor[n, f32]) = copy({reshape})\n"),
                "{\n  r = g(x)\n  alias = r\n  neg(neg(alias))\n}".to_owned(),
            ),
        ] {
            let expected = match (&expected, form) {
                (Expected::Tensor(dims, values), "neg") => {
                    Expected::Tensor(dims.clone(), values.iter().map(|v| -v).collect())
                }
                _ => expected.clone(),
            };
            let mut routes = Vec::new();
            call_matrix(
                &mut routes,
                &format!("result_graph.{kind}.{form}.{divisor}.x{n}"),
                1686,
                &format!(
                    "{helper}def f(x: tensor[n, f32]) -> tensor[{claim}, {divisor}, f32] = {body}"
                ),
                signature,
                vec![vector(n)],
                expected,
            );
            for case in routes {
                let main_claim = if kind == "literal" { 2 } else { n };
                let main = case
                    .source
                    .contains("def main()")
                    .then(|| format!("() -> tensor[{main_claim}, {divisor}, f32]"));
                fixtures.push((case, main));
            }
        }
    }
    // The inner n witnesses x; the outer n witnesses an unread argument.
    // The inferred helper must not resolve the outer requirement in its own scope.
    for (rows, cols, expected) in [
        (2, 4, Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
        (
            3,
            4,
            Expected::Domain("reshape", &["claimed = 3", "reshape axis 0 = 2"]),
        ),
    ] {
        let mut routes = Vec::new();
        call_matrix(
            &mut routes,
            &format!("result_graph.distinct_witnesses.{rows}.{cols}"),
            1686,
            "def g(x: tensor[n, f32]) = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])\ndef f(unread: tensor[n, f32], x: tensor[m, f32]) -> tensor[n, 2, f32] = copy(g(x))",
            "(tensor[d0, f32], tensor[d1, f32]) -> tensor[d0, 2, f32]",
            vec![vector(rows), vector(cols)],
            expected,
        );
        for case in routes {
            let main = case
                .source
                .contains("def main()")
                .then(|| format!("() -> tensor[{rows}, 2, f32]"));
            fixtures.push((case, main));
        }
    }
    // Static branch pruning must not attach a claim to the unexecuted producer.
    let mut routes = Vec::new();
    call_matrix(
        &mut routes,
        "result_graph.untaken",
        1686,
        "def bad(x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])\ndef f(x: tensor[n, f32]) -> tensor[2, 2, f32] = if false then bad(x) else to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]])",
        "(tensor[d0, f32]) -> tensor[2, 2, f32]",
        vec![vector(6)],
        Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
    );
    for case in routes {
        let main = case
            .source
            .contains("def main()")
            .then(|| "() -> tensor[2, 2, f32]".to_owned());
        fixtures.push((case, main));
    }
    assert_eq!(fixtures.len(), 69);
    let mut failures = Vec::new();
    for (case, main_signature) in fixtures {
        let observed = observe(&case);
        println!("{}: {}", case.id, observed);
        failures.extend(contract_failures(&case, &observed));
        if let Some(signature) = main_signature
            && observed["check"]["signatures"]["main"] != signature
        {
            failures.push(format!(
                "{} main signature: {} != {signature}",
                case.id, observed["check"]["signatures"]["main"]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// #1686/#1687 retain the original #1375/#597/#1619 host exits. These
/// expectations come from section 4.7 and [05-MOV-1], never from lane agreement.
#[test]
fn omitted_extent_claim_contract() {
    assert!(gcc_available(), "C toolchain required; no lane may skip");
    let mut fixtures = Vec::new();
    {
        let mut add =
            |family: &str, issue, def: &str, signature, inputs, expected, main_signature: &str| {
                let mut routes = Vec::new();
                call_matrix(&mut routes, family, issue, def, signature, inputs, expected);
                for case in routes {
                    let main = case
                        .source
                        .contains("def main()")
                        .then(|| main_signature.to_owned());
                    fixtures.push((case, main));
                }
            };
        for (kind, claim, signature) in [
            ("named", "n", "(tensor[d0, f32]) -> tensor[d0, 2, f32]"),
            ("literal", "2", "(tensor[d0, f32]) -> tensor[2, 2, f32]"),
        ] {
            for n in [4, 6] {
                let expected = match (kind, n) {
                    ("literal", 4) => Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
                    ("literal", 6) => Expected::Domain("reshape", &["claimed = 2", "axis 0 = 3"]),
                    ("named", 4) => Expected::Domain("reshape", &["claimed = 4", "axis 0 = 2"]),
                    ("named", 6) => Expected::Domain("reshape", &["claimed = 6", "axis 0 = 3"]),
                    _ => unreachable!(),
                };
                let main_extent = if kind == "literal" { 2 } else { n };
                for (form, body) in [
                    (
                        "direct",
                        "reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])",
                    ),
                    (
                        "alias",
                        "{\n  k = floor_div(shape(x, 0i32), 2i64)\n  target = k\n  reshape(x, [target, 2i64])\n}",
                    ),
                    (
                        "result_alias",
                        "{\n  result = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])\n  alias = result\n  alias\n}",
                    ),
                ] {
                    add(
                        &format!("omitted.reshape.{kind}.{form}.x{n}"),
                        1686,
                        &format!("def f(x: tensor[n, f32]) -> tensor[{claim}, 2, f32] = {body}"),
                        signature,
                        vec![vector(n)],
                        expected.clone(),
                        &format!("() -> tensor[{main_extent}, 2, f32]"),
                    );
                }
                let g = format!(
                    "def g(x: tensor[n, f32]) -> tensor[{claim}, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])"
                );
                add(
                    &format!("omitted.reshape.{kind}.nested.x{n}"),
                    1686,
                    &format!("{g}\ndef f(x: tensor[n, f32]) -> tensor[{claim}, 2, f32] = g(x)"),
                    signature,
                    vec![vector(n)],
                    expected.clone(),
                    &format!("() -> tensor[{main_extent}, 2, f32]"),
                );
                add(
                    &format!("omitted.reshape.{kind}.discarded.x{n}"),
                    1686,
                    &format!(
                        "{g}\ndef f(x: tensor[n, f32]) -> tensor[n, f32] = {{\n  _ = g(x)\n  x\n}}"
                    ),
                    "(tensor[d0, f32]) -> tensor[d0, f32]",
                    vec![vector(n)],
                    if kind == "literal" && n == 4 {
                        Expected::Tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0])
                    } else {
                        expected
                    },
                    &format!("() -> tensor[{n}, f32]"),
                );
            }
        }
        // A named claim can agree with independently computed arithmetic.
        // Retain the operation, rather than replacing it with a literal reshape.
        for n in [4, 6] {
            add(
                &format!("omitted.reshape.named.match.x{n}"),
                1686,
                "def f(x: tensor[n, f32]) -> tensor[n, 1, f32] = reshape(x, [floor_div(shape(x, 0i32), 1i64), 1i64])",
                "(tensor[d0, f32]) -> tensor[d0, 1, f32]",
                vec![vector(n)],
                Expected::Tensor(vec![n, 1], (1..=n).map(|v| v as f64).collect()),
                &format!("() -> tensor[{n}, 1, f32]"),
            );
        }
        for good in [true, false] {
            let b = Input {
                dims: vec![if good { 1 } else { 2 }],
                values: if good { vec![5.0] } else { vec![5.0, 6.0] },
            };
            let mismatch = Expected::Domain("load", &["claimed = 1", "b axis 0 = 2"]);
            for (form, body) in [
                ("literal", "expand(b, 0i32, 3i64)"),
                ("let", "{\n  k = 3i64\n  expand(b, 0i32, k)\n}"),
                (
                    "alias",
                    "{\n  operand = b\n  alias = operand\n  expand(alias, 0i32, 3i64)\n}",
                ),
            ] {
                add(
                    &format!("omitted.unit.{form}"),
                    1687,
                    &format!("def f(b: tensor[unit, f32]) -> tensor[3, f32] = {body}"),
                    "(tensor[unit, f32]) -> tensor[3, f32]",
                    vec![b.clone()],
                    if good {
                        Expected::Tensor(vec![3], vec![5.0; 3])
                    } else {
                        mismatch.clone()
                    },
                    "() -> tensor[3, f32]",
                );
            }
            let g = "def g(b: tensor[unit, f32], xs: tensor[n, f32]) -> tensor[n, f32] = mul(xs, expand(b, 0i32, shape(xs, 0i32)))";
            for (form, def, discarded) in [
                ("shape", g.replace("def g(", "def f("), false),
                (
                    "nested",
                    format!(
                        "{g}\ndef f(b: tensor[unit, f32], xs: tensor[n, f32]) -> tensor[n, f32] = g(b, xs)"
                    ),
                    false,
                ),
                (
                    "discarded",
                    format!(
                        "{g}\ndef f(b: tensor[unit, f32], xs: tensor[n, f32]) -> tensor[n, f32] = {{\n  _ = g(b, xs)\n  xs\n}}"
                    ),
                    true,
                ),
            ] {
                add(
                    &format!("omitted.unit.{form}"),
                    1687,
                    &def,
                    "(tensor[unit, f32], tensor[d0, f32]) -> tensor[d0, f32]",
                    vec![b.clone(), vector(3)],
                    if good {
                        Expected::Tensor(
                            vec![3],
                            if discarded {
                                vec![1.0, 2.0, 3.0]
                            } else {
                                vec![5.0, 10.0, 15.0]
                            },
                        )
                    } else {
                        mismatch.clone()
                    },
                    "() -> tensor[3, f32]",
                );
            }
        }
        for (rows, cols, expected) in [
            (2, 4, Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
            (
                3,
                4,
                Expected::Domain("reshape", &["claimed = 3", "reshape axis 0 = 2"]),
            ),
            (
                2,
                6,
                Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"]),
            ),
        ] {
            add(
                &format!("omitted.unread.{rows}.{cols}"),
                1686,
                "def f(unread: tensor[rows, f32], x: tensor[cols, f32]) -> tensor[rows, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])",
                "(tensor[rows, f32], tensor[cols, f32]) -> tensor[rows, 2, f32]",
                vec![vector(rows), vector(cols)],
                expected,
                "() -> tensor[rows, 2, f32]",
            );
        }
        for divisor in [1, 2] {
            let definition = format!(
                "def g(x: tensor[n, f32]) -> tensor[n, 1, f32] = reshape(x, [floor_div(shape(x, 0i32), 1i64), 1i64])\ndef h(x: tensor[n, f32]) -> tensor[n, {divisor}, f32] = reshape(x, [floor_div(shape(x, 0i32), {divisor}i64), {divisor}i64])\ndef f(a: tensor[left, f32], b: tensor[right, f32]) -> tensor[f32] = add(sum(sum(g(a), 1i32), 0i32), sum(sum(h(b), 1i32), 0i32))"
            );
            add(
                &format!("omitted.separate_scopes.{divisor}"),
                1686,
                &definition,
                "(tensor[left, f32], tensor[right, f32]) -> tensor[f32]",
                vec![vector(4), vector(6)],
                if divisor == 1 {
                    Expected::Tensor(vec![], vec![31.0])
                } else {
                    Expected::Domain("reshape", &["claimed = 6", "reshape axis 0 = 3"])
                },
                "() -> tensor[f32]",
            );
        }
    }
    assert_eq!(
        fixtures.len(),
        117,
        "three routes for every positive/negative operation form"
    );
    let example = include_str!("../../../examples/checked_runtime_extents.ch");
    for (id, source, expected) in [
        (
            "match",
            example.to_owned(),
            Expected::Tensor(vec![2, 2], vec![2.0, 4.0, 6.0, 8.0]),
        ),
        (
            "unit",
            example.replace("to_tensor([2.0f32])", "to_tensor([2.0f32, 3.0f32])"),
            Expected::Domain("load", &["claimed = 1", "b axis 0 = 2"]),
        ),
        (
            "reshape",
            example.replace("3.0f32, 4.0f32])", "3.0f32, 4.0f32, 5.0f32, 6.0f32])"),
            Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"]),
        ),
    ] {
        fixtures.push((
            Case {
                id: format!("checked.example.{id}"),
                issue: 1277,
                source,
                signature: None,
                expected,
                exported: None,
            },
            Some("() -> tensor[2, 2, f32]".to_owned()),
        ));
    }
    assert_eq!(fixtures.len(), 120);
    let mut failures = Vec::new();
    for (case, main_signature) in &fixtures {
        let observation = observe(case);
        println!("{}: {}", case.id, observation);
        failures.extend(contract_failures(case, &observation));
        if case.id.starts_with("checked.example.") {
            let expected = "(tensor[unit, f32], tensor[d0, f32]) -> tensor[2, 2, f32]";
            let actual = &observation["check"]["signatures"]["pair_and_scale"];
            if actual != expected {
                failures.push(format!(
                    "{}.signature: expected {expected}, observed {actual}",
                    case.id
                ));
            }
        }
        if let Some(expected) = main_signature {
            let observed = &observation["check"]["signatures"]["main"];
            if observed != expected {
                failures.push(format!(
                    "{}.main.signature: expected {expected}, observed {observed}",
                    case.id
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn imported_checked_extent_contract() {
    let mut failures = Vec::new();
    for good in [true, false] {
        for (family, library, arguments, expected) in [
            (
                "staged",
                "def f(x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [bitand(floor_div(numel(x), 2i64), 3i64), 2i64])",
                literal(&vector(if good { 4 } else { 6 })),
                if good {
                    Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
                } else {
                    Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
                },
            ),
            (
                "reshape",
                "def f(x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])",
                literal(&vector(if good { 4 } else { 6 })),
                if good {
                    Expected::Tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
                } else {
                    Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"])
                },
            ),
            (
                "unit",
                "def f(b: tensor[unit, f32]) -> tensor[3, f32] = expand(b, 0i32, 3i64)",
                if good {
                    "to_tensor([5.0f32])".into()
                } else {
                    "to_tensor([5.0f32, 6.0f32])".into()
                },
                if good {
                    Expected::Tensor(vec![3], vec![5.0; 3])
                } else {
                    Expected::Domain("load", &["claimed = 1", "b axis 0 = 2"])
                },
            ),
            (
                "literal",
                "def f(x: tensor[n, f32]) -> tensor[4, f32] = insert(scalar_to_tensor(7.0f32), 0i32, shape(x, 0i32))",
                literal(&vector(if good { 4 } else { 5 })),
                if good {
                    Expected::Tensor(vec![4], vec![7.0; 4])
                } else {
                    Expected::Domain("load", &["claimed = 4", "x axis 0 = 5"])
                },
            ),
        ] {
            for (form, prefix) in [("binding", "out ="), ("main", "def main() =")] {
                let case = Case {
                    id: format!("import.{family}.{form}.{good}"),
                    issue: 1277,
                    source: format!(
                        "module App.Fixture\nimport Mylib.Claims (f)\n{prefix} f({arguments})\n"
                    ),
                    signature: None,
                    expected: expected.clone(),
                    exported: None,
                };
                let observation = observe_with_dependency(&case, Some(library));
                println!("{}: {}", case.id, observation);
                failures.extend(contract_failures(&case, &observation));
                let (signature, result) = match family {
                    "reshape" | "staged" => ("(tensor[d0, f32]) -> tensor[2, 2, f32]", "2, 2"),
                    "unit" => ("(tensor[unit, f32]) -> tensor[3, f32]", "3"),
                    "literal" => ("(tensor[d0, f32]) -> tensor[4, f32]", "4"),
                    _ => unreachable!(),
                };
                for (name, expected) in [
                    ("pkg__mylib__Mylib__Claims__f", signature.to_owned()),
                    (
                        "pkg__app__App__Fixture__main",
                        format!("() -> tensor[{result}, f32]"),
                    ),
                ] {
                    if name.ends_with("__main") && form != "main" {
                        continue;
                    }
                    let actual = &observation["check"]["signatures"][name];
                    if actual != &expected {
                        failures.push(format!(
                            "{}.signature {name}: expected {expected}, observed {actual}",
                            case.id
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn checked_extent_transforms_contract() {
    let mut failures = Vec::new();
    let mut count = 0;
    for family in ["reshape", "unit"] {
        for good in [true, false] {
            let (definition, signature, input, matrix, mapped, gradient, domain) = if family
                == "reshape"
            {
                (
                    "def f(x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])",
                    "(tensor[d0, f32]) -> tensor[2, 2, f32]",
                    vector(if good { 4 } else { 6 }),
                    if good {
                        "to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]])"
                    } else {
                        "to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32], [7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32]])"
                    },
                    Expected::Tensor(vec![2, 2, 2], (1..=8).map(f64::from).collect()),
                    Expected::Tensor(vec![4], vec![1.0; 4]),
                    Expected::Domain("reshape", &["claimed = 2", "reshape axis 0 = 3"]),
                )
            } else {
                (
                    "def f(b: tensor[unit, f32]) -> tensor[3, f32] = expand(b, 0i32, 3i64)",
                    "(tensor[unit, f32]) -> tensor[3, f32]",
                    Input {
                        dims: vec![if good { 1 } else { 2 }],
                        values: if good { vec![5.0] } else { vec![5.0, 6.0] },
                    },
                    if good {
                        "to_tensor([[5.0f32], [7.0f32]])"
                    } else {
                        "to_tensor([[5.0f32, 6.0f32], [7.0f32, 8.0f32]])"
                    },
                    Expected::Tensor(vec![2, 3], vec![5.0, 5.0, 5.0, 7.0, 7.0, 7.0]),
                    Expected::Tensor(vec![1], vec![3.0]),
                    Expected::Domain("load", &["claimed = 1", "b axis 0 = 2"]),
                )
            };
            for transform in ["vmap", "grad", "masked_grad"] {
                let (extra, call, expected) = if transform == "vmap" {
                    (
                        String::new(),
                        format!("vmap(f)({matrix})"),
                        if good {
                            mapped.clone()
                        } else if family == "reshape" {
                            Expected::Domain("reshape", &["claimed = 2", "reshape axis 1 = 3"])
                        } else {
                            Expected::Domain("load", &["claimed = 1", "b axis 1 = 2"])
                        },
                    )
                } else {
                    let reduction = if family == "reshape" {
                        "sum(sum(f(x), 1i32), 0i32)"
                    } else {
                        "sum(f(x), 0i32)"
                    };
                    let body = if transform == "masked_grad" {
                        format!("mul({reduction}, scalar_to_tensor(0.0f32))")
                    } else {
                        reduction.into()
                    };
                    let extra = format!("\ndef loss(x: tensor[n, f32]) -> tensor[f32] = {body}\n");
                    let positive = if transform == "masked_grad" {
                        Expected::Tensor(input.dims.clone(), vec![0.0; input.values.len()])
                    } else {
                        gradient.clone()
                    };
                    (
                        extra,
                        format!("grad(loss)({})", literal(&input)),
                        if good { positive } else { domain.clone() },
                    )
                };
                for (form, prefix) in [("binding", "out ="), ("main", "def main() =")] {
                    let case = Case {
                        id: format!("transform.{family}.{transform}.{form}.{good}"),
                        issue: 1277,
                        source: format!("{definition}\n{extra}{prefix} {call}\n"),
                        signature: Some(signature),
                        expected: expected.clone(),
                        exported: None,
                    };
                    let observation = observe(&case);
                    println!("{}: {}", case.id, observation);
                    failures.extend(contract_failures(&case, &observation));
                    if form == "main" {
                        let dims = if transform == "vmap" {
                            if family == "reshape" {
                                "2, 2, 2".to_owned()
                            } else {
                                "2, 3".to_owned()
                            }
                        } else if family == "unit" {
                            // A named dimension retains its declaring spelling
                            // in the signature. Values and failures below use
                            // independent actual singleton/nonunit arguments.
                            "unit".to_owned()
                        } else {
                            input.dims[0].to_string()
                        };
                        let expected = format!("() -> tensor[{dims}, f32]");
                        let actual = &observation["check"]["signatures"]["main"];
                        if actual != &expected {
                            failures.push(format!(
                                "{}.main.signature: expected {expected}, observed {actual}",
                                case.id
                            ));
                        }
                    }
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 24);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
