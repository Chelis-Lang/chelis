//! Public acceptance surface for spec/02 [02-PIPE-1..3].
use assert_cmd::Command;
use std::{fs, path::Path};
use tempfile::tempdir;

fn run(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .unwrap()
}
fn deep(path: &Path) -> String {
    let output = run(&["deep", "--flat", path.to_str().unwrap()]);
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn device_handler_operands_survive_check_eval_fmt_and_decompilation() {
    use chelis_compiler_api::schema::ExecutionValue;
    let dir = tempdir().unwrap();
    let path = dir.path().join("handler.ch");
    for expression in [
        "with device(\"cpu\") { 3i32 } |> inc",
        "(with device(\"cpu\") { 3i32 }) |> inc",
        "3i32 |> with device(\"cpu\") { inc }",
        "3i32 |> (with device(\"cpu\") { inc })",
        "inc(with device(\"cpu\") { 3i32 })",
        "(with device(\"cpu\") { inc })(3i32)",
    ] {
        fs::write(
            &path,
            format!("def inc(x: i32) -> i32 = x + 1i32\nout = {expression}\n"),
        )
        .unwrap();
        let formatted = run(&["fmt", path.to_str().unwrap()]);
        assert!(formatted.status.success(), "{expression}: {formatted:?}");
        fs::write(&path, &formatted.stdout).unwrap();
        let again = run(&["fmt", path.to_str().unwrap()]);
        assert!(again.status.success(), "{expression}: {again:?}");
        assert_eq!(again.stdout, formatted.stdout);
        let checked = run(&["check", path.to_str().unwrap()]);
        assert!(checked.status.success(), "{expression}: {checked:?}");
        let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert_eq!(report["score"], 1.0);
        assert_eq!(report["errors"], serde_json::json!([]));
        let evaluated = run(&["eval", "--json", "--file", path.to_str().unwrap()]);
        assert!(evaluated.status.success(), "{expression}: {evaluated:?}");
        let report: serde_json::Value = serde_json::from_slice(&evaluated.stdout).unwrap();
        let roots = report["roots"].as_array().unwrap();
        assert_eq!(roots.len(), 1);
        let ExecutionValue::Scalar { value } =
            serde_json::from_value(roots[0]["value"].clone()).unwrap()
        else {
            panic!("expected tagged scalar")
        };
        assert_eq!(value.get().prim().name(), "i32");
        assert_eq!(value.get().as_f64_lossy(), 4.0);
        let dp = dir.path().join("handler.dp");
        fs::write(&dp, deep(&path)).unwrap();
        let surface = run(&["surf", dp.to_str().unwrap()]);
        assert!(surface.status.success(), "{surface:?}");
        fs::write(&path, &surface.stdout).unwrap();
        let canonical = run(&["fmt", path.to_str().unwrap()]);
        assert!(canonical.status.success(), "{canonical:?}");
        assert_eq!(canonical.stdout, surface.stdout);
    }
    for expression in [
        "3i32 with { a: 1i32 } |> inc",
        "with device(\"cpu\") { 3i32 + 1i32 |> inc }",
        "with device(\"cpu\") 3i32 |> inc",
    ] {
        fs::write(&path, format!("out = {expression}\n")).unwrap();
        let rejected = run(&["check", path.to_str().unwrap()]);
        assert!(!rejected.status.success(), "{expression}: {rejected:?}");
        let report: serde_json::Value = serde_json::from_slice(&rejected.stdout).unwrap();
        assert_eq!(report["components"]["parse"], 0.0);
        assert!(!report["errors"].as_array().unwrap().is_empty());
    }
}

#[test]
fn named_cast_pipe_stages_preserve_values_and_rejected_dtype_pairs() {
    use chelis_compiler_api::schema::ExecutionValue;
    let dir = tempdir().unwrap();
    let path = dir.path().join("named-cast.ch");
    for (keyword, seed, expected) in [
        ("cast_saturate", "128i32", 127.0),
        ("cast_wrap", "128i32", -128.0),
        ("cast_saturate", "(-129i32)", -128.0),
        ("cast_wrap", "(-129i32)", 127.0),
        ("cast_trunc", "1.9f32", 1.0),
    ] {
        fs::write(
            &path,
            format!("piped = {seed} |> {keyword}(i8)\ncalled = {keyword}({seed}, i8)\n"),
        )
        .unwrap();
        let output = run(&["eval", "--json", "--file", path.to_str().unwrap()]);
        assert!(output.status.success(), "{keyword}: {output:?}");
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let roots = report["roots"].as_array().unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0]["value"], roots[1]["value"]);
        let ExecutionValue::Scalar { value } =
            serde_json::from_value(roots[0]["value"].clone()).unwrap()
        else {
            panic!("expected exact tagged scalar")
        };
        assert_eq!(value.get().prim().name(), "i8");
        assert_eq!(value.get().as_f64_lossy(), expected);
    }
    for (keyword, seed, target) in [
        ("cast_wrap", "1.5f32", "i8"),
        ("cast_trunc", "1i32", "i8"),
        ("cast_saturate", "1i32", "bool"),
    ] {
        for expression in [
            format!("{seed} |> {keyword}({target})"),
            format!("{keyword}({seed}, {target})"),
        ] {
            fs::write(&path, format!("out = {expression}\n")).unwrap();
            let output = run(&["check", path.to_str().unwrap()]);
            assert!(!output.status.success(), "{expression}: {output:?}");
            let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert!(
                report["errors"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|error| error["message"].as_str().unwrap().contains(keyword)),
                "{expression}: {report}"
            );
        }
    }
}
#[test]
fn pipe_cast_and_call_cast_have_the_same_bits_and_canonical_source() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("cast.ch");
    fs::write(
        &source,
        "piped = 0.1 |> cast(f64)\ncalled = cast(0.1, f64)\nrounded = 0.1f32 |> cast(f64)\n",
    )
    .unwrap();
    let result = run(&["eval", "--json", "--file", source.to_str().unwrap()]);
    assert!(result.status.success(), "{result:?}");
    let result: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    let roots = result["roots"].as_array().unwrap();
    assert_eq!(roots[0]["value"], roots[1]["value"]);
    assert_ne!(roots[0]["value"], roots[2]["value"]);
    let raw = deep(&source);
    assert!(!raw.contains("(pipe ") && !raw.contains("surf_pipe_stage"));
    let dp = dir.path().join("cast.dp");
    fs::write(&dp, raw).unwrap();
    let surf = run(&["surf", dp.to_str().unwrap()]);
    assert!(surf.status.success(), "{surf:?}");
    let surf = String::from_utf8(surf.stdout).unwrap();
    assert!(!surf.contains("|>"));
    let back = dir.path().join("back.ch");
    fs::write(&back, &surf).unwrap();
    let fmt = run(&["fmt", back.to_str().unwrap()]);
    assert!(fmt.status.success(), "{fmt:?}");
    assert_eq!(fmt.stdout, surf.as_bytes());
}

#[test]
fn authored_stage_errors_retain_original_source_locations() {
    let dir = tempdir().unwrap();
    for (name, source, kind, stage) in [
        (
            "type",
            "def f(x: bool) -> bool = x\nout = 1i32 |> f\n",
            "PrecisionMismatch",
            "f\n",
        ),
        (
            "arity",
            "def f(x: i32, y: i32) -> i32 = x + y\nout = 1i32 |> f\n",
            "ArityMismatch",
            "f\n",
        ),
        (
            "ownership",
            "def f(x: tensor[2, f32]) -> tensor[2, f32] = x |> realize |> add(x)\n",
            "UseAfterConsume",
            "x)\n",
        ),
    ] {
        let path = dir.path().join(format!("{name}.ch"));
        fs::write(&path, source).unwrap();
        let result = run(&["check", path.to_str().unwrap()]);
        assert!(!result.status.success(), "{name}: {result:?}");
        let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        let error = report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|error| error["kind"] == kind)
            .expect("expected diagnostic kind");
        let offset = source.rfind(stage).unwrap();
        assert_eq!(error["span"]["offset"], offset, "{source}\n{error}");
        assert!(
            error["span_id"]
                .as_str()
                .unwrap()
                .starts_with(&format!("surf:{offset}.."))
        );
        assert!(source.contains("|>"));
        let build = run(&["build", "--allow-style-violations", path.to_str().unwrap()]);
        assert!(!build.status.success());
        let rendered = String::from_utf8_lossy(&build.stderr);
        assert!(
            rendered.contains("|>"),
            "diagnostic must render authored pipe: {rendered}"
        );
        assert!(!rendered.contains("__chelis_pipe"));
    }
}

#[test]
fn persisted_pipe_nodes_are_rejected_with_version_and_cause() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("old.dp");
    fs::write(&path, "(def {} out (pipe {} (var {} x) (var {} f)))").unwrap();
    for command in ["surf", "fmt", "check"] {
        let output = run(&[command, path.to_str().unwrap()]);
        assert!(!output.status.success(), "{command}");
        let output = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.contains("pipe") && output.contains("0.20"),
            "{command}: {output}"
        );
    }
}

#[cfg(unix)]
#[test]
fn migration_requires_old_dtype_evidence_and_checks_the_entire_batch_before_writing() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempdir().unwrap();
    let python = std::process::Command::new("uv")
        .args(["python", "find", "3.11"])
        .output()
        .unwrap();
    assert!(python.status.success());
    let python = String::from_utf8(python.stdout).unwrap();
    let baseline = dir.path().join("old-compiler");
    // Frozen output from the previous grammar: its seed is f32, not f64.
    let receipt = r#"(def {} out (pipe {} (lit {span: "surf:6..9", type: (t-prim {} f32)} 0.1) (fn {surf_pipe_stage: "call-first"} (params {} p) (cast {} (var {} p) (t-prim {} f64)))))"#;
    fs::write(
        &baseline,
        format!(
            "#!{}\nimport sys\nsys.stdout.write({receipt:?})\n",
            python.trim()
        ),
    )
    .unwrap();
    fs::set_permissions(&baseline, fs::Permissions::from_mode(0o755)).unwrap();
    let first = dir.path().join("first.ch");
    let second = dir.path().join("second.ch");
    let source = "out = 0.1 |> cast(f64)\n";
    fs::write(&first, source).unwrap();
    fs::write(&second, "out = 0.2 |> cast(f64)\n").unwrap();
    let args = [
        "migrate",
        "pipes",
        "--baseline-compiler",
        baseline.to_str().unwrap(),
    ];
    let printed = run(&[args.as_slice(), &[first.to_str().unwrap()]].concat());
    assert!(printed.status.success(), "{printed:?}");
    assert_eq!(printed.stdout, b"out = 0.1f32 |> cast(f64)\n");
    let batch = run(&[
        args.as_slice(),
        &[
            "--inplace",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
        ],
    ]
    .concat());
    assert!(!batch.status.success());
    assert!(String::from_utf8_lossy(&batch.stderr).contains("Deep differs"));
    assert_eq!(fs::read_to_string(&first).unwrap(), source);
    assert_eq!(
        fs::read_to_string(&second).unwrap(),
        "out = 0.2 |> cast(f64)\n"
    );
    let inplace = run(&[args.as_slice(), &["--inplace", first.to_str().unwrap()]].concat());
    assert!(inplace.status.success(), "{inplace:?}");
    assert_eq!(
        fs::read_to_string(&first).unwrap(),
        "out = 0.1f32 |> cast(f64)\n"
    );
    assert!(
        run(&[args.as_slice(), &["--check", first.to_str().unwrap()]].concat())
            .status
            .success()
    );
    fs::write(
        &baseline,
        format!(
            "#!{}\nprint('(def {{}} out (var {{}} x))')\n",
            python.trim()
        ),
    )
    .unwrap();
    fs::write(&first, source).unwrap();
    let missing = run(&[args.as_slice(), &["--inplace", first.to_str().unwrap()]].concat());
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("no literal dtype"));
    assert_eq!(fs::read_to_string(&first).unwrap(), source);

    for (metadata, key) in [
        ("span: true", "span"),
        ("span: \"a\", span: \"b\"", "span"),
        ("type: true", "type"),
        ("surf_literal_style: \"unsuffixed\"", "surf_literal_style"),
        ("surf_binding_type: \"inferred\"", "surf_binding_type"),
    ] {
        let malformed = receipt.replace(
            "(fn {surf_pipe_stage:",
            &format!("(fn {{{metadata}, surf_pipe_stage:"),
        );
        fs::write(&baseline, format!(
            "#!{}\nimport sys\nfrom pathlib import Path\nsys.stdout.write({malformed:?} if Path(sys.argv[-1]).name == 'second.ch' else {receipt:?})\n",
            python.trim()
        )).unwrap();
        fs::write(&first, source).unwrap();
        fs::write(&second, source).unwrap();
        let rejected = run(&[
            args.as_slice(),
            &[
                "--inplace",
                first.to_str().unwrap(),
                second.to_str().unwrap(),
            ],
        ]
        .concat());
        assert!(
            !rejected.status.success(),
            "accepted malformed {key}: {rejected:?}"
        );
        assert!(
            String::from_utf8_lossy(&rejected.stderr).contains(key),
            "{rejected:?}"
        );
        assert_eq!(fs::read_to_string(&first).unwrap(), source);
        assert_eq!(fs::read_to_string(&second).unwrap(), source);
    }
    for (owner, pattern) in [
        ("fn", "(fn {surf_pipe_stage: \"call-first\"}"),
        ("pipe", "(pipe {}"),
    ] {
        for replacement in ["true", "(var {} x)", "()", ""] {
            let malformed = receipt.replace(pattern, &format!("({owner} {replacement}"));
            fs::write(&baseline, format!(
                "#!{}\nimport sys\nfrom pathlib import Path\nsys.stdout.write({malformed:?} if Path(sys.argv[-1]).name == 'second.ch' else {receipt:?})\n",
                python.trim()
            )).unwrap();
            fs::write(&first, source).unwrap();
            fs::write(&second, source).unwrap();
            let rejected = run(&[
                args.as_slice(),
                &[
                    "--inplace",
                    first.to_str().unwrap(),
                    second.to_str().unwrap(),
                ],
            ]
            .concat());
            assert!(
                !rejected.status.success(),
                "accepted non-map {owner}: {rejected:?}"
            );
            assert!(
                String::from_utf8_lossy(&rejected.stderr).contains("metadata map"),
                "{rejected:?}"
            );
            assert_eq!(fs::read_to_string(&first).unwrap(), source);
            assert_eq!(fs::read_to_string(&second).unwrap(), source);
        }
    }
}

#[test]
fn long_generated_pipe_chain_desugars_without_native_stack_overflow() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("long.ch");
    fs::write(&path, format!("out = x{}\n", " |> f".repeat(1000))).unwrap();
    let output = deep(&path);
    assert!(!output.contains("(pipe "));
    assert_eq!(output.matches("(app ").count(), 1000);
    let dp = dir.path().join("long.dp");
    fs::write(&dp, &output).unwrap();
    let surf = run(&["surf", dp.to_str().unwrap()]);
    assert!(surf.status.success(), "{surf:?}");
    let canonical = dir.path().join("calls.ch");
    fs::write(&canonical, &surf.stdout).unwrap();
    let fmt = run(&["fmt", canonical.to_str().unwrap()]);
    assert!(fmt.status.success(), "{fmt:?}");
    assert_eq!(fmt.stdout, surf.stdout);
    assert_eq!(deep(&canonical).matches("(app ").count(), 1000);
}

#[test]
fn opaque_property_injection_uses_normalized_callable_context() {
    let dir = tempdir().unwrap();
    for carried in ["f |> Hold", "Hold(f)"] {
        for (predicate, expected_status) in
            [("p.value >= 0.0", "passed"), ("p.value < 0.0", "failed")]
        {
            let source = format!(
                "module Review.Probe\n\
                 @opaque\n\
                 @invariant(p) (p.value >= 0.0)\n\
                 type Positive = | Positive {{value: f32}}\n\
                 type Holder[a] = | Hold(a)\n\
                 def f(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, x), 0i32)\n\
                 held = {carried}\n\
                 chosen = match held with {{ | Hold(g) => g }}\n\
                 derivative = grad(chosen, wrt=x)\n\
                 @property positive forall (p: Positive): ({predicate})\n"
            );
            let path = dir.path().join("property.ch");
            fs::write(&path, source).unwrap();
            let checked = run(&["check", path.to_str().unwrap()]);
            assert!(checked.status.success(), "{checked:?}");
            let checked: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
            assert_eq!(checked["score"], 1.0);
            assert!(checked["errors"].as_array().unwrap().is_empty());
            let proved = run(&[
                "prove",
                "--json",
                "--tier",
                "fuzz-only",
                "--samples",
                "1",
                path.to_str().unwrap(),
            ]);
            let records: Vec<serde_json::Value> = String::from_utf8(proved.stdout)
                .unwrap()
                .lines()
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_str(line).expect("every stdout record is JSON"))
                .collect();
            let property = records
                .iter()
                .find(|record| record["kind"] == "property" && record["name"] == "positive")
                .expect("named property result");
            assert_eq!(property["status"], expected_status, "{carried}: {property}");
            assert_eq!(property["proof_tier"], "fuzz");
            assert_eq!(property["samples"], 1);
            assert_eq!(proved.status.success(), expected_status == "passed");
            let summary = records
                .iter()
                .find(|record| record["kind"] == "summary")
                .expect("summary");
            assert_eq!(summary["errors"], 0);
        }
    }
}
