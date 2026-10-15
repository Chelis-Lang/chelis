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


fn counted_native(source: &str) -> (u64, u64) {
    use assert_cmd::Command;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("growing.ch");
    std::fs::write(&path, source).unwrap();
    let destination = dir.path().join("c");
    let built = Command::cargo_bin("chelis").unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--emit-c", "--allow-style-violations"])
        .arg(&path).args(["--target", "c", "-o"]).arg(&destination).output().unwrap();
    assert!(built.status.success(), "{source}\n{built:?}");
    let c_path = destination.join("growing.c");
    let mut c = std::fs::read_to_string(&c_path).unwrap();
    let include = c.rfind("#include ").expect("generated includes");
    let header_end = include + c[include..].find('\n').unwrap() + 1;
    // Count executed extent reads and rank admission, not elapsed time or
    // emitted check text. Runtime allocation internals are outside this shim.
    c.insert_str(header_end, r#"
static uint64_t __probe_entry = 0;
static uint64_t __probe_origin = 0;
static int64_t __probe_shape(const chelis_tensor *x, int32_t axis) {
    ++__probe_entry;
    return chelis_tensor_shape(x, axis);
}
static int32_t __probe_rank(const chelis_tensor *x) {
    ++__probe_entry;
    return chelis_tensor_rank(x);
}
static chelis_value __probe_field(const chelis_adt *x, int64_t slot) {
    ++__probe_entry;
    return chelis_adt_get_field(x, slot);
}
static chelis_value __probe_index(const chelis_list *x, int64_t slot) {
    ++__probe_entry;
    return chelis_list_index(x, slot);
}
#define chelis_tensor_shape __probe_shape
#define chelis_tensor_rank __probe_rank
#define chelis_adt_get_field __probe_field
#define chelis_list_index __probe_index
"#);
    c = c.replace("static __chelis_host_result_origin *__chelis_host_result_origin_alloc(__chelis_host_result_origin_arena *arena, int64_t child_count) {",
        "static __chelis_host_result_origin *__chelis_host_result_origin_alloc(__chelis_host_result_origin_arena *arena, int64_t child_count) { ++__probe_origin;");
    c = c.replace("static void __chelis_host_result_origin_arena_destroy(__chelis_host_result_origin_arena *arena) {",
        "static void __chelis_host_result_origin_arena_destroy(__chelis_host_result_origin_arena *arena) { fprintf(stderr, \"__work__=%llu,%llu\\n\", (unsigned long long)__probe_entry, (unsigned long long)__probe_origin);");
    c = c.replace("__chelis_entry_named_state *state = &states[i];", "__chelis_entry_named_state *state = &states[i]; ++__probe_entry;");
    c = c.replace("    __chelis_entry_claim_step *step = (__chelis_entry_claim_step *)malloc(sizeof *step);", "    ++__probe_entry; __chelis_entry_claim_step *step = (__chelis_entry_claim_step *)malloc(sizeof *step);");
    let mut counted = String::new();
    for line in c.lines() {
        if line.contains(".ordinal] = __chelis_entry_named_states[i];") {
            let count: u64 = line.split("i < ").nth(1).unwrap().split(';').next().unwrap().parse().unwrap();
            counted.push_str(&format!("__probe_entry += {};\n", 2 * count));
        }
        counted.push_str(line);
        counted.push('\n');
    }
    std::fs::write(&c_path, counted).unwrap();
    assert!(common::link_generated(&destination, "growing.c", "growing").success());
    let output = std::process::Command::new(destination.join("growing")).output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let output = String::from_utf8_lossy(&output.stderr);
    let counts = output.lines().filter_map(|line| line.strip_prefix("__work__=")).last().expect("executed counters");
    let (entry, origin) = counts.split_once(',').unwrap();
    (entry.parse().unwrap(), origin.parse().unwrap())
}

#[test]
fn compiled_growing_claim_reads_are_linear_with_a_quadratic_counterfactual() {
    for collection in ["chain", "list"] {
        for claim in ["3", "n"] {
            for scan in [false, true] {
                for steps in [100, 400] {
                    let combine = if scan { "scan" } else { "fold" };
                    let source = if collection == "chain" {
                        format!("type Chain[n] =\n | End {{ v: tensor[n, f32] }}\n | Link {{ v: tensor[n, f32], next: Chain[n] }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef run[n](steps: i64) -> i64 = {{\n _ = {combine}(fn (acc: Chain[{claim}], i: i64) -> Link {{ v: produce(3i64), next: acc }}, End {{ v: produce(3i64) }}, range(0i64, steps))\n steps\n}}\nout = run({steps}i64)\n")
                    } else {
                        let generic = if claim == "n" { "[n]" } else { "" };
                        format!("def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef run{generic}(seed: List[tensor[{claim}, f32]], steps: i64) -> i64 = len({combine}(fn (acc: List[tensor[{claim}, f32]], i: i64) -> append(acc, produce(3i64)), seed, range(0i64, steps)))\nout = run([produce(3i64)], {steps}i64)\n")
                    };
                    let (work, origins) = counted_native(&source);
                    let counterfactual = source.replace("next: acc", "next: if eq(i, 0i64) then acc else acc")
                        .replace("append(acc,", "append(if eq(i, 0i64) then acc else acc,");
                    let (unproved, old_origins) = counted_native(&counterfactual);
                    eprintln!("C {collection} {combine} claim={claim} K={steps}: reads {work}, unproved {unproved}; origins {origins}/{old_origins}");
                    assert!(work <= 40 * steps as u64 + 40, "{source}\n{work}");
                    assert!(unproved >= steps as u64 * steps as u64, "counterfactual must retain full entry work: {unproved}");
                    assert!(origins <= 10 * steps as u64 + 20, "origin construction work is measured separately: {origins}");
                }
            }
        }
    }
}
