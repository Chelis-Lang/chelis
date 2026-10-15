//! Growing accumulator entry claims retain spec/04 §4.7 entry placement.
//! In particular, a fresh bad component is checked at the next invocation,
//! and does not create a callback-result obligation on the final iteration.
mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

fn chain_source(path: &str, steps: usize, claim: &str) -> String {
    format!(r#"type Chain[n] =
  | End {{ v: tensor[n, f32] }}
  | Link {{ v: tensor[n, f32], next: Chain[n] }}
def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def run[n](size: i64, steps: i64) -> i64 ! {{ IO }} = {{
  _ = fold(fn (acc: Chain[{claim}], i: i64) -> {{
  _ = print("before")
  v = produce(size)
  _ = print("after")
  Link {{ v, next: acc }}
}}, End {{ v: produce(3i64) }}, range(0i64, steps))
  42i64
}}
out = run(string_len(read_file("{path}")), {steps}i64)
"#)
}

#[test]
fn growing_chain_entry_preserves_effects_and_final_iteration() {
    for native in [false, true] {
        for claim in ["3", "n"] {
            for steps in [0, 1, 2] {
                for size in [3, 5] {
                    let dir = tempfile::tempdir().expect("fixture directory");
                    let path = dir.path().join("size");
                    std::fs::write(&path, "x".repeat(size)).expect("opaque size");
                    let source = chain_source(path.to_str().expect("path"), steps, claim);
                    let (success, output) = result_claims::run(&source, native);
                    let bad = size == 5 && steps == 2;
                    assert_eq!(success, !bad, "{source}\n{output}");
                    let callbacks = if bad { 1 } else { steps };
                    assert_eq!(output.lines().filter(|line| *line == "before").count(), callbacks, "{output}");
                    assert_eq!(output.lines().filter(|line| *line == "after").count(), callbacks, "{output}");
                    if bad {
                        assert!(output.contains("numeric trap: domain in load at i64"), "{output}");
                        if claim == "n" {
                            assert!(output.contains("acc.Link.v axis 0 = 5, acc.Link.next.End.v axis 0 = 3"), "{output}");
                        }
                    } else {
                        assert!(output.contains("out = 42"), "{output}");
                    }
                }
            }
        }
    }
}

fn list_source(path: &str, steps: usize, claim: &str, prepend: bool, scan: bool) -> String {
    let step = if prepend { "concat([v], acc)" } else { "append(acc, v)" };
    let combinator = if scan { "scan" } else { "fold" };
    let generic = if claim == "n" { "[n]" } else { "" };
    format!(r#"def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def run{generic}(seed: List[tensor[{claim}, f32]], size: i64, steps: i64) -> i64 ! {{ IO }} = len({combinator}(fn (acc: List[tensor[{claim}, f32]], i: i64) -> {{
  _ = print("before")
  v = produce(size)
  _ = print("after")
  {step}
}}, seed, range(0i64, steps)))
out = run([produce(3i64)], string_len(read_file("{path}")), {steps}i64)
"#)
}

#[test]
fn growing_list_entry_preserves_order_and_final_iteration() {
    for native in [false, true] {
        for claim in ["3", "n"] {
            for prepend in [false, true] {
                for scan in [false, true] {
                    for (size, steps) in [(5, 0), (5, 1), (5, 2), (3, 2)] {
                        let dir = tempfile::tempdir().expect("fixture directory");
                        let path = dir.path().join("size");
                        std::fs::write(&path, "x".repeat(size)).expect("opaque size");
                        let source = list_source(path.to_str().expect("path"), steps, claim, prepend, scan);
                        let (success, output) = result_claims::run(&source, native);
                        let bad = size == 5 && steps == 2;
                        assert_eq!(success, !bad, "{source}\n{output}");
                        let callbacks = if bad { 1 } else { steps };
                        assert_eq!(output.lines().filter(|line| *line == "before").count(), callbacks, "{output}");
                        assert_eq!(output.lines().filter(|line| *line == "after").count(), callbacks, "{output}");
                        if bad {
                            assert!(output.contains("numeric trap: domain in load at i64"), "{output}");
                            if claim == "n" {
                                let (first, second) = if prepend { (5, 3) } else { (3, 5) };
                                assert!(output.contains(&format!("acc[0] axis 0 = {first}, acc[1] axis 0 = {second}")), "{output}");
                            }
                        } else {
                            let length = if scan { steps } else { steps + 1 };
                            assert!(output.contains(&format!("out = {length}")), "{output}");
                        }
                    }
                }
            }
        }
    }
}
