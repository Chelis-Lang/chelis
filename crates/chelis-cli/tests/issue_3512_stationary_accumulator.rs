//! spec/04 §4.7 and runtime_extents.md C6.6: reuse requires the same
//! immutable accumulator, obligation and witness source. Callback effects
//! and independent claims still execute; zero iterations check no callback.
mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

fn run(body: &str, native: bool, size: usize) -> (bool, String) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("size");
    std::fs::write(&input, "x".repeat(size)).unwrap();
    let prelude = r#"type Box[n] =
  | Box { v: tensor[n, f32] }
def size_from(path: string) -> i64 ! { IO } = string_len(read_file(path))
def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def make[n](size: i64) -> Box[n] = Box { v: produce(size) }
def width[n](b: Box[n]) -> i64 = match b with { | Box { v } => shape(v, 0i32) }
"#;
    let source = format!("{prelude}{body}").replace("PATH", &input.to_string_lossy());
    if !native {
        return result_claims::run(&source, false);
    }
    use assert_cmd::Command;
    assert!(common::gcc_available(), "compiled C must execute");
    let path = dir.path().join("stationary.ch");
    std::fs::write(&path, &source).unwrap();
    let destination = dir.path().join("c");
    let built = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--emit-c", "--allow-style-violations"])
        .arg(&path)
        .args(["--target", "c", "-o"])
        .arg(&destination)
        .output()
        .unwrap();
    assert!(built.status.success(), "{source}\n{built:?}");
    // Instrument the actual emitted entry guard, then compile, link and run.
    // This counts executed successful-check branches, not emitted text.
    let c_path = destination.join("stationary.c");
    let c = std::fs::read_to_string(&c_path).unwrap();
    let mut counted = String::new();
    let named_receipt = c.contains("__stationary_entry_states_");
    let mut guard_indent: Option<String> = None;
    for line in c.lines() {
        counted.push_str(line);
        counted.push('\n');
        if line
            .trim_start()
            .starts_with("if (!__stationary_entry_checked_")
        {
            guard_indent = Some(line[..line.len() - line.trim_start().len()].to_string());
            counted.push_str("fputs(\"__entry_check__\\n\", stderr);\n");
        } else if named_receipt
            && guard_indent
                .as_ref()
                .is_some_and(|indent| line == format!("{indent}}}"))
        {
            // Read the same scoped state that a following local/result guard
            // reads, on every callback invocation, including reused entries.
            counted.push_str("fprintf(stderr, \"__witness__=%lld\\n\", (long long)__chelis_entry_named_states[0].value);\n");
            guard_indent = None;
        }
    }
    std::fs::write(c_path, counted).unwrap();
    assert!(common::link_generated(&destination, "stationary.c", "stationary").success());
    let output = std::process::Command::new(destination.join("stationary"))
        .output()
        .unwrap();
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

fn observe(combinator: &str, call: &str) -> String {
    format!(
        "{}({call})",
        if combinator == "fold" { "width" } else { "len" }
    )
}

#[test]
fn stationary_claims_preserve_callback_effects_and_named_witnesses_on_both_lanes() {
    for native in [false, true] {
        for combinator in ["fold", "scan"] {
            let call = format!(
                r#"{combinator}(fn (acc: Box[n], i: i64) -> {{
  _ = print("tick")
  acc
}}, make(size), range(0i64, 3i64))"#
            );
            let body = format!(
                "def run[n](size: i64) -> i64 ! {{ IO }} = {}\nout = run(size_from(\"PATH\"))\n",
                observe(combinator, &call)
            );
            let (ok, output) = run(&body, native, 5);
            assert!(ok, "{output}");
            if native {
                assert_eq!(
                    output
                        .lines()
                        .filter(|line| *line == "__entry_check__")
                        .count(),
                    1,
                    "{output}"
                );
                assert_eq!(
                    output
                        .lines()
                        .filter(|line| *line == "__witness__=5")
                        .count(),
                    3,
                    "{output}"
                );
            }
            assert_eq!(
                output.lines().filter(|line| *line == "tick").count(),
                3,
                "{output}"
            );
            assert!(
                output.contains(if combinator == "fold" {
                    "out = 5"
                } else {
                    "out = 3"
                }),
                "{output}"
            );
        }
    }
}

#[test]
fn stationary_seed_is_checked_on_first_invocation_and_empty_loops_check_none() {
    for native in [false, true] {
        for combinator in ["fold", "scan"] {
            for steps in [0, 2] {
                let call = format!(
                    r#"{combinator}(fn (acc: Box[3], i: i64) -> {{
  _ = print("tick")
  acc
}}, make(size_from("PATH")), range(0i64, {steps}i64))"#
                );
                let body = format!("out = {}\n", observe(combinator, &call));
                let (ok, output) = run(&body, native, 5);
                assert_eq!(ok, steps == 0, "{output}");
                if native {
                    assert_eq!(
                        output
                            .lines()
                            .filter(|line| *line == "__entry_check__")
                            .count(),
                        usize::from(steps != 0),
                        "{output}"
                    );
                }
                assert!(!output.lines().any(|line| line == "tick"), "{output}");
                if steps != 0 {
                    assert!(
                        output.contains("numeric trap: domain in load at i64"),
                        "{output}"
                    );
                }
            }
        }
    }
}

#[test]
fn stationary_list_fold_checks_every_seed_element_before_the_body() {
    for native in [false, true] {
        for dimension in ["3", "n"] {
            let body = format!(
                r#"def run[n](size: i64) -> i64 ! {{ IO }} = len(fold(fn (acc: List[tensor[{dimension}, f32]], i: i64) -> {{
  _ = print("tick")
  acc
}}, [produce(3i64), produce(size)], range(0i64, 2i64)))
out = run(size_from("PATH"))
"#
            );
            let (ok, output) = run(&body, native, 5);
            assert!(!ok, "{output}");
            assert!(
                output.contains("numeric trap: domain in load at i64"),
                "{output}"
            );
            assert!(!output.lines().any(|line| line == "tick"), "{output}");
            let (ok, output) = run(&body, native, 3);
            assert!(ok, "{output}");
            assert!(output.contains("out = 2"), "{output}");
            assert_eq!(
                output.lines().filter(|line| *line == "tick").count(),
                2,
                "{output}"
            );
        }
    }
}

#[test]
fn changed_or_shadowed_accumulators_keep_their_next_entry_check() {
    for native in [false, true] {
        for combinator in ["fold", "scan"] {
            for result in [
                "make(size_from(\"PATH\"))",
                "{ acc = make(size_from(\"PATH\"))\n acc }",
            ] {
                let call = format!(
                    "{combinator}(fn (acc: Box[3], i: i64) -> {{\n _ = print(\"tick\")\n {result}\n}}, make(3i64), range(0i64, 2i64))"
                );
                let body = format!("out = {}\n", observe(combinator, &call));
                let (ok, output) = run(&body, native, 5);
                assert!(!ok, "{output}");
                assert_eq!(
                    output.lines().filter(|line| *line == "tick").count(),
                    1,
                    "{output}"
                );
                assert!(
                    output.contains("numeric trap: domain in load at i64"),
                    "{output}"
                );
            }
        }
    }
}

#[test]
fn independent_item_result_and_local_claims_remain_checked() {
    let cases = [
        (
            r#"def run[n](size: i64) -> i64 = width(fold(fn (acc: Box[n], t: tensor[n, f32]) -> acc, make(size), [produce(3i64)]))
out = run(size_from("PATH"))
"#,
            3,
            "load",
        ),
        (
            r#"def run[n](size: i64) -> i64 = len(scan(fn (acc: Box[n], t: tensor[n, f32]) -> acc, make(size), [produce(3i64)]))
out = run(size_from("PATH"))
"#,
            1,
            "load",
        ),
        (
            r#"def keep[n](size: i64) -> Box[n] = fold(fn (acc: Box[n], i: i64) -> acc, make(size), range(0i64, 2i64))
def insist(size: i64) -> Box[3] = keep(size)
out = width(insist(size_from("PATH")))
"#,
            3,
            "fold",
        ),
        (
            r#"def keep[n](size: i64) -> List[Box[n]] = scan(fn (acc: Box[n], i: i64) -> acc, make(size), range(0i64, 2i64))
def insist(size: i64) -> List[Box[3]] = keep(size)
out = len(insist(size_from("PATH")))
"#,
            2,
            "scan",
        ),
        (
            r#"def run(size: i64) -> i64 ! { IO } = width(fold(fn (acc: Box[3], i: i64) -> {
  bad: tensor[3, f32] = produce(size)
  _ = print(shape(bad, 0i32))
  acc
}, make(3i64), range(0i64, 2i64)))
out = run(size_from("PATH"))
"#,
            3,
            "insert",
        ),
        (
            r#"def run(size: i64) -> i64 ! { IO } = len(scan(fn (acc: Box[3], i: i64) -> {
  bad: tensor[3, f32] = produce(size)
  _ = print(shape(bad, 0i32))
  acc
}, make(3i64), range(0i64, 2i64)))
out = run(size_from("PATH"))
"#,
            2,
            "insert",
        ),
    ];
    for native in [false, true] {
        for (body, expected, operation) in cases {
            let (ok, output) = run(body, native, 5);
            assert!(!ok, "{output}");
            assert!(
                output.contains(&format!("numeric trap: domain in {operation} at i64")),
                "{output}"
            );
            assert!(!output.contains("out ="), "{output}");
            let (ok, output) = run(body, native, 3);
            assert!(ok, "{output}");
            assert!(output.contains(&format!("out = {expected}")), "{output}");
        }
    }
}
