//! Growing accumulator entry claims retain spec/04 §4.7 entry placement.
//! In particular, a fresh bad component is checked at the next invocation,
//! and does not create a callback-result obligation on the final iteration.
mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

fn chain_source(path: &str, steps: usize, claim: &str) -> String {
    format!(
        r#"type Chain[n] =
  | End {{ v: tensor[n, f32] }}
  | Link {{ v: tensor[n, f32], next: Chain[n] }}
def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def run[n](size: i64, steps: i64) -> i64 ! {{ IO }} = {{
  _ = fold(fn (acc: Chain[{claim}], i: i64) -> {{
  _ = print("before")
  old = acc
  older = old
  v = produce(size)
  _ = print("after")
  Link {{ v, next: older }}
}}, End {{ v: produce(3i64) }}, range(0i64, steps))
  42i64
}}
out = run(string_len(read_file("{path}")), {steps}i64)
"#
    )
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
                    assert_eq!(
                        output.lines().filter(|line| *line == "before").count(),
                        callbacks,
                        "{output}"
                    );
                    assert_eq!(
                        output.lines().filter(|line| *line == "after").count(),
                        callbacks,
                        "{output}"
                    );
                    if bad {
                        assert!(
                            output.contains("numeric trap: domain in load at i64"),
                            "{output}"
                        );
                        if claim == "n" {
                            assert!(
                                output.contains(
                                    "acc.Link.v axis 0 = 5, acc.Link.next.End.v axis 0 = 3"
                                ),
                                "{output}"
                            );
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
    let step = if prepend {
        "concat([v], older)"
    } else {
        "append(older, v)"
    };
    let combinator = if scan { "scan" } else { "fold" };
    let generic = if claim == "n" { "[n]" } else { "" };
    format!(
        r#"def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def run{generic}(seed: List[tensor[{claim}, f32]], size: i64, steps: i64) -> i64 ! {{ IO }} = len({combinator}(fn (acc: List[tensor[{claim}, f32]], i: i64) -> {{
  _ = print("before")
  old = acc
  older = old
  v = produce(size)
  _ = print("after")
  {step}
}}, seed, range(0i64, steps)))
out = run([produce(3i64)], string_len(read_file("{path}")), {steps}i64)
"#
    )
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
                        let source =
                            list_source(path.to_str().expect("path"), steps, claim, prepend, scan);
                        let (success, output) = result_claims::run(&source, native);
                        let bad = size == 5 && steps == 2;
                        assert_eq!(success, !bad, "{source}\n{output}");
                        let callbacks = if bad { 1 } else { steps };
                        assert_eq!(
                            output.lines().filter(|line| *line == "before").count(),
                            callbacks,
                            "{output}"
                        );
                        assert_eq!(
                            output.lines().filter(|line| *line == "after").count(),
                            callbacks,
                            "{output}"
                        );
                        if bad {
                            assert!(
                                output.contains("numeric trap: domain in load at i64"),
                                "{output}"
                            );
                            if claim == "n" {
                                let (first, second) = if prepend { (5, 3) } else { (3, 5) };
                                assert!(
                                    output.contains(&format!(
                                        "acc[0] axis 0 = {first}, acc[1] axis 0 = {second}"
                                    )),
                                    "{output}"
                                );
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
    let c_path = destination.join("growing.c");
    let mut c = std::fs::read_to_string(&c_path).unwrap();
    let include = c.rfind("#include ").expect("generated includes");
    let header_end = include + c[include..].find('\n').unwrap() + 1;
    // Count executed extent reads and rank admission, not elapsed time or
    // emitted check text. Runtime allocation internals are outside this shim.
    c.insert_str(
        header_end,
        r#"
static uint64_t __probe_entry = 0;
static uint64_t __probe_origin = 0;
static uint64_t __probe_origin_bytes = 0;
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
static int64_t __probe_length(const chelis_list *x) {
    ++__probe_entry;
    return chelis_list_len(x);
}
static chelis_value __probe_index(const chelis_list *x, int64_t slot) {
    ++__probe_entry;
    return chelis_list_index(x, slot);
}
#define chelis_tensor_shape __probe_shape
#define chelis_tensor_rank __probe_rank
#define chelis_adt_get_field __probe_field
#define chelis_list_len __probe_length
#define chelis_list_index __probe_index
"#,
    );
    c = c.replace("static __chelis_host_result_origin *__chelis_host_result_origin_alloc(__chelis_host_result_origin_arena *arena, int64_t child_count) {",
        "static __chelis_host_result_origin *__chelis_host_result_origin_alloc(__chelis_host_result_origin_arena *arena, int64_t child_count) { ++__probe_origin; __probe_origin_bytes += sizeof(__chelis_host_result_origin) + (size_t)child_count * sizeof(const __chelis_host_result_origin *);");
    c = c.replace("static void __chelis_host_result_origin_arena_destroy(__chelis_host_result_origin_arena *arena) {",
        "static void __chelis_host_result_origin_arena_destroy(__chelis_host_result_origin_arena *arena) { fprintf(stderr, \"__work__=%llu,%llu\\n\", (unsigned long long)__probe_entry, (unsigned long long)__probe_origin); fprintf(stderr, \"__memory__=%llu\\n\", (unsigned long long)__probe_origin_bytes);");
    c = c.replace(
        "__chelis_entry_named_state *state = &states[i];",
        "__chelis_entry_named_state *state = &states[i]; ++__probe_entry;",
    );
    c = c.replace("    __chelis_entry_claim_step *step = (__chelis_entry_claim_step *)malloc(sizeof *step);", "    ++__probe_entry; __chelis_entry_claim_step *step = (__chelis_entry_claim_step *)malloc(sizeof *step);");
    let mut counted = String::new();
    for line in c.lines() {
        if line.trim_start().starts_with("if (") && line.contains("stationary_entry_checked") {
            counted.push_str("++__probe_entry;\n");
        }
        if line.contains(".ordinal] = __chelis_entry_named_states[i];") {
            let count: u64 = line
                .split("i < ")
                .nth(1)
                .unwrap()
                .split(';')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            counted.push_str(&format!("__probe_entry += {};\n", 2 * count));
        }
        counted.push_str(line);
        counted.push('\n');
    }
    std::fs::write(&c_path, counted).unwrap();
    assert!(common::link_generated(&destination, "growing.c", "growing").success());
    let output = std::process::Command::new(destination.join("growing"))
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let output = String::from_utf8_lossy(&output.stderr);
    let counts = output
        .lines()
        .filter_map(|line| line.strip_prefix("__work__="))
        .next_back()
        .expect("executed counters");
    let memory = output
        .lines()
        .filter_map(|line| line.strip_prefix("__memory__="))
        .next_back()
        .expect("origin allocation bytes");
    eprintln!("native invocation cumulative allocated origin bytes: {memory}");
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
                        format!(
                            "type Chain[n] =\n | End {{ v: tensor[n, f32] }}\n | Link {{ v: tensor[n, f32], next: Chain[n] }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef run[n](steps: i64) -> i64 = {{\n _ = {combine}(fn (acc: Chain[{claim}], i: i64) -> Link {{ v: produce(3i64), next: acc }}, End {{ v: produce(3i64) }}, range(0i64, steps))\n steps\n}}\nout = run({steps}i64)\n"
                        )
                    } else {
                        let generic = if claim == "n" { "[n]" } else { "" };
                        format!(
                            "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef run{generic}(seed: List[tensor[{claim}, f32]], steps: i64) -> i64 = len({combine}(fn (acc: List[tensor[{claim}, f32]], i: i64) -> append(acc, produce(3i64)), seed, range(0i64, steps)))\nout = run([produce(3i64)], {steps}i64)\n"
                        )
                    };
                    let (work, origins) = counted_native(&source);
                    let counterfactual = source
                        .replace("next: acc", "next: if eq(i, 0i64) then acc else acc")
                        .replace("append(acc,", "append(if eq(i, 0i64) then acc else acc,");
                    let (unproved, old_origins) = counted_native(&counterfactual);
                    eprintln!(
                        "C {collection} {combine} claim={claim} K={steps}: reads {work}, unproved {unproved}; origins {origins}/{old_origins}"
                    );
                    assert!(work <= 40 * steps as u64 + 40, "{source}\n{work}");
                    assert!(
                        unproved >= steps as u64 * steps as u64,
                        "counterfactual must retain full entry work: {unproved}"
                    );
                    assert!(
                        origins <= 10 * steps as u64 + 20,
                        "origin construction work is measured separately: {origins}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_lexically_shadowed_list_operation_cannot_supply_delta_evidence() {
    for native in [false, true] {
        for size in [3, 5] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("size");
            std::fs::write(&path, "x".repeat(size)).unwrap();
            let source = list_source(path.to_str().unwrap(), 2, "3", false, false)
            .replace("= len(fold", "= {\n append = fn (xs: List[tensor[*, f32]], x: tensor[*, f32]) -> [produce(size), produce(3i64)]\n len(fold")
            .replace("range(0i64, steps)))", "range(0i64, steps)))\n}");
            let (success, output) = result_claims::run(&source, native);
            assert_eq!(success, size == 3, "{source}\n{output}");
            if size == 5 {
                assert!(
                    output.contains("numeric trap: domain in load at i64"),
                    "{output}"
                );
                assert_eq!(
                    output.lines().filter(|line| *line == "before").count(),
                    1,
                    "{output}"
                );
            }
        }
    }
}

fn late_component_source(
    path: &str,
    steps: usize,
    claim: &str,
    scan: bool,
    late_seed: bool,
) -> String {
    let combine = if scan { "scan" } else { "fold" };
    let seed_size = if late_seed { "size" } else { "3i64" };
    let fresh_size = if late_seed { "3i64" } else { "size" };
    format!(
        r#"type Chain[n] =
  | End {{ first: tensor[n, f32], second: tensor[n, f32] }}
  | Link {{ first: tensor[n, f32], second: tensor[n, f32], next: Chain[n] }}
def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def run[n](size: i64, steps: i64) -> i64 ! {{ IO }} = {{
  _ = {combine}(fn (acc: Chain[{claim}], i: i64) -> {{
    _ = print("before")
    first = produce(3i64)
    second = produce({fresh_size})
    _ = print("after")
    Link {{ first, second, next: acc }}
  }}, End {{ first: produce(3i64), second: produce({seed_size}) }}, range(0i64, steps))
  42i64
}}
out = run(string_len(read_file("{path}")), {steps}i64)
"#
    )
}

#[test]
fn growing_claims_check_late_seed_and_fresh_components_at_their_entry() {
    for native in [false, true] {
        for claim in ["3", "n"] {
            for scan in [false, true] {
                for late_seed in [false, true] {
                    for size in [3, 5] {
                        for steps in [0, 1, 2] {
                            let dir = tempfile::tempdir().unwrap();
                            let path = dir.path().join("size");
                            std::fs::write(&path, "x".repeat(size)).unwrap();
                            let source = late_component_source(
                                path.to_str().unwrap(),
                                steps,
                                claim,
                                scan,
                                late_seed,
                            );
                            let (success, output) = result_claims::run(&source, native);
                            let bad = size == 5 && steps > usize::from(!late_seed);
                            assert_eq!(success, !bad, "{source}\n{output}");
                            let effects = if bad { usize::from(!late_seed) } else { steps };
                            assert_eq!(
                                output.lines().filter(|line| *line == "before").count(),
                                effects,
                                "{output}"
                            );
                            assert_eq!(
                                output.lines().filter(|line| *line == "after").count(),
                                effects,
                                "{output}"
                            );
                            if bad {
                                assert!(
                                    output.contains("numeric trap: domain in load at i64"),
                                    "{output}"
                                );
                                assert!(
                                    output.contains(if late_seed {
                                        "acc.End.second"
                                    } else {
                                        "acc.Link.second"
                                    }),
                                    "{output}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn inherited_named_witnesses_replay_in_observation_order() {
    for native in [false, true] {
        for size in [3, 5] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("size");
            std::fs::write(&path, "x".repeat(size)).unwrap();
            let source = format!(
                r#"type Chain[a, b] =
  | End {{ b: tensor[b, f32], a: tensor[a, f32] }}
  | Link {{ a: tensor[a, f32], b: tensor[b, f32], next: Chain[a, b] }}
def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def run[a, b](size: i64) -> i64 ! {{ IO }} = {{
  _ = fold(fn (acc: Chain[a, b], i: i64) -> {{
    _ = print("before")
    first = produce(if eq(i, 0i64) then 3i64 else size)
    second = produce(if eq(i, 0i64) then 3i64 else size)
    _ = print("after")
    Link {{ a: first, b: second, next: acc }}
  }}, End {{ b: produce(3i64), a: produce(3i64) }}, range(0i64, 3i64))
  42i64
}}
out = run(string_len(read_file("{}")))
"#,
                path.to_str().unwrap()
            );
            let (success, output) = result_claims::run(&source, native);
            assert_eq!(success, size == 3, "{source}\n{output}");
            if size == 5 {
                assert!(
                    output.contains(
                        "extent `a`: acc.Link.a axis 0 = 5, acc.Link.next.Link.a axis 0 = 3"
                    ),
                    "{output}"
                );
                assert_eq!(
                    output.lines().filter(|line| *line == "after").count(),
                    2,
                    "{output}"
                );
            }
        }
    }
}

#[test]
fn top_level_builtin_shadowing_is_rejected_before_delta_lowering() {
    for size in [3, 5] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("size");
        std::fs::write(&path, "x".repeat(size)).unwrap();
        let source = list_source(path.to_str().unwrap(), 2, "3", false, false)
            .replace("def run(", "def append(xs: List[tensor[*, f32]], x: tensor[*, f32]) -> List[tensor[*, f32]] = [x, produce(3i64)]\ndef run(");
        let (success, output) = result_claims::run(&source, false);
        assert!(!success, "{source}\n{output}");
        assert!(output.contains("BuiltinShadowing"), "{output}");
        assert_eq!(
            output.lines().filter(|line| *line == "after").count(),
            0,
            "{output}"
        );
    }
}

#[test]
fn aliases_and_selected_callables_retain_complete_entry_validation() {
    for native in [false, true] {
        for route in ["append_step", "alias", "selected"] {
            // C ownership lowering does not admit these computed callees;
            // the direct named call is executed on both lanes.
            if native && route != "append_step" {
                continue;
            }
            for size in [3, 5] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("size");
                std::fs::write(&path, "x".repeat(size)).unwrap();
                let mut source = list_source(path.to_str().unwrap(), 2, "3", false, false)
                .replace("def run(", "def append_step(xs: List[tensor[*, f32]], x: tensor[*, f32]) -> List[tensor[*, f32]] = append(xs, x)\ndef run(");
                if route != "append_step" {
                    let value = if route == "alias" {
                        "append_step"
                    } else {
                        "if gt(size, 4i64) then append_step else append_step"
                    };
                    source = source
                        .replace("= len(fold", &format!("= {{\n step = {value}\n len(fold"))
                        .replace("range(0i64, steps)))", "range(0i64, steps)))\n}");
                }
                let intrinsic_call = "append(older, v)";
                assert_eq!(
                    source.matches(intrinsic_call).count(),
                    1,
                    "the fixture must contain exactly one callback construction: {source}"
                );
                let routed_call = if route == "append_step" {
                    "append_step(older, v)"
                } else {
                    "step(older, v)"
                };
                source = source.replacen(intrinsic_call, routed_call, 1);
                assert_eq!(source.matches(routed_call).count(), 1, "{source}");
                assert!(!source.contains(intrinsic_call), "{source}");
                let (success, output) = result_claims::run(&source, native);
                assert_eq!(success, size == 3, "{source}\n{output}");
                if size == 5 {
                    assert!(
                        output.contains("numeric trap: domain in load at i64"),
                        "{output}"
                    );
                    assert!(
                        output.contains(if native {
                            "input `acc[1]` axis 0 expected 3, got 5"
                        } else {
                            "acc[1] axis 0 = 5"
                        }),
                        "{output}"
                    );
                    assert_eq!(
                        output.lines().filter(|line| *line == "after").count(),
                        1,
                        "{output}"
                    );
                }
            }
        }
    }
}

#[test]
fn named_definition_accumulator_claim_has_independent_compositional_receipts() {
    for claim in ["3", "n"] {
        for outer in ["*", "3"] {
            for combine in ["fold", "scan"] {
                for direct in [false, true] {
                    if direct && outer != "*" {
                        continue;
                    }
                    for unproved in [false, true] {
                        for seed_size in [1, 100] {
                            for steps in [100, 400] {
                                let generic = if claim == "n" { "[n]" } else { "" };
                                let callback = if direct {
                                    "step".to_owned()
                                } else {
                                    format!(
                                        "fn (acc: List[tensor[{outer}, f32]], i: i64) -> step(acc, i)"
                                    )
                                };
                                let body = if unproved {
                                    "{\n old = if i == 0i64 then acc else acc\n append(old, produce(3i64))\n}"
                                } else {
                                    "append(acc, produce(3i64))"
                                };
                                let source = format!(
                                    "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step{generic}(acc: List[tensor[{claim}, f32]], i: i64) -> List[tensor[*, f32]] = {body}\ndef run(seed: List[tensor[3, f32]], steps: i64) -> i64 = len({combine}({callback}, seed, range(0i64, steps)))\nout = run(map(fn (i: i64) -> produce(3i64), range(0i64, {seed_size}i64)), {steps}i64)\n"
                                );
                                let (work, origins) = counted_native(&source);
                                eprintln!(
                                    "C named definition {combine} direct={direct} outer={outer} claim={claim} unproved={unproved} N={seed_size} K={steps}: reads {work}; origins {origins}"
                                );
                                if unproved {
                                    assert!(work >= steps as u64 * steps as u64 / 2, "{work}");
                                } else {
                                    assert!(
                                        work <= 96 * (steps + seed_size) as u64 + 100,
                                        "independent named entries have bounded work: {work}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn named_accumulator_entries_preserve_empty_and_final_invalid_effects() {
    // §4.7: first entry precedes the body; a bad newly appended component
    // fails at the next entry, with no invented final-result obligation.
    for native in [false, true] {
        for combine in ["fold", "scan"] {
            for direct in [false, true] {
                for steps in [0, 1, 2] {
                    for bad_seed in [false, true] {
                        let callback = if direct {
                            "step"
                        } else {
                            "fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i)"
                        };
                        let source = format!(
                            "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{\n _ = print(\"body\")\n append(acc, produce(5i64))\n}}\ndef run(seed: List[tensor[*, f32]]) -> i64 ! {{ IO }} = {{\n _ = {combine}({callback}, seed, range(0i64, {steps}i64))\n 42i64\n}}\nout = run([produce({}i64)])\n",
                            if bad_seed { 5 } else { 3 }
                        );
                        let (success, output) = result_claims::run(&source, native);
                        let failed = steps > 0 && (bad_seed || steps > 1);
                        assert_eq!(success, !failed, "{source}\n{output}");
                        let bodies = if steps == 0 || bad_seed { 0 } else { 1 };
                        assert_eq!(
                            output.lines().filter(|line| *line == "body").count(),
                            bodies,
                            "{source}\n{output}"
                        );
                        if failed {
                            assert!(
                                output.contains("numeric trap: domain in load at i64"),
                                "{output}"
                            );
                        } else {
                            assert!(output.contains("out = 42"), "{output}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn stationary_named_accumulator_has_bounded_seed_and_step_work() {
    for combine in ["fold", "scan"] {
        for direct in [false, true] {
            for seed in [100, 400] {
                let callback = if direct {
                    "step"
                } else {
                    "fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i)"
                };
                let source = format!(
                    "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] = acc\ndef run(seed: List[tensor[3, f32]]) -> i64 = len({combine}({callback}, seed, range(0i64, 400i64)))\nout = run(map(fn (i: i64) -> produce(3i64), range(0i64, {seed}i64)))\n"
                );
                let (work, _) = counted_native(&source);
                eprintln!("C stationary named {combine} direct={direct} N={seed} K=400: {work}");
                assert!(work <= 40 * (seed + 400) as u64 + 100, "{work}");
            }
        }
    }
}

#[test]
fn preceding_independent_callback_entry_has_its_own_bounded_receipt() {
    for steps in [100, 400] {
        let source = format!(
            "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] = append(acc, produce(3i64))\ndef run(seed: List[tensor[3, f32]]) -> i64 = len(fold(fn (acc: List[tensor[3, f32]], i: i64) -> step(acc, i), seed, range(0i64, {steps}i64)))\nout = run([produce(3i64)])\n"
        );
        let (work, _) = counted_native(&source);
        eprintln!("C preceding independent entry K={steps}: {work}");
        assert!(work <= 40 * steps as u64 + 100, "{work}");
    }
}

#[test]
fn named_prepend_keeps_new_first_witness_and_independent_result_claim() {
    for native in [false, true] {
        for combine in ["fold", "scan"] {
            for aliased in [false, true] {
                for result_claim in [false, true] {
                    for steps in [0, 1, 2] {
                        let result = if result_claim { "3" } else { "*" };
                        let dir = tempfile::tempdir().unwrap();
                        let path = dir.path().join("size");
                        std::fs::write(&path, "xxxxx").unwrap();
                        let prefix = if aliased {
                            " old = acc\n older = old\n _ = print(\"alias\")\n"
                        } else {
                            ""
                        };
                        let inherited = if aliased { "older" } else { "acc" };
                        let source = format!(
                            "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\nsize = string_len(read_file(\"{}\"))\ndef step[n](acc: List[tensor[n, f32]], i: i64) -> List[tensor[{result}, f32]] ! {{ IO }} = {{\n _ = print(\"body\")\n {prefix} concat([produce(size)], {inherited})\n}}\ndef run(seed: List[tensor[3, f32]]) -> i64 ! {{ IO }} = {{\n _ = {combine}(fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i), seed, range(0i64, {steps}i64))\n 42i64\n}}\nout = run([produce(3i64)])\n",
                            path.display()
                        );
                        let (success, output) = result_claims::run(&source, native);
                        let failed = steps > 0 && (result_claim || steps > 1);
                        assert_eq!(success, !failed, "{source}\n{output}");
                        assert_eq!(
                            output.lines().filter(|line| *line == "body").count(),
                            usize::from(steps > 0),
                            "{output}"
                        );
                        assert_eq!(
                            output.lines().filter(|line| *line == "alias").count(),
                            usize::from(aliased && steps > 0),
                            "{output}"
                        );
                        if failed {
                            assert!(
                                output.contains(if result_claim {
                                    "numeric trap: domain in concat at i64"
                                } else {
                                    "numeric trap: domain in load at i64"
                                }),
                                "{source}\n{output}"
                            );
                            if !result_claim {
                                assert!(
                                    output.contains("acc[0] axis 0 = 5, acc[1] axis 0 = 3"),
                                    "{output}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn recursive_named_calls_cannot_borrow_the_loop_callsite_receipt() {
    for native in [false, true] {
        for combine in ["fold", "scan"] {
            for steps in [0, 1, 2] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("size");
                std::fs::write(&path, "xxxxx").unwrap();
                let source = format!(
                    "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\nsize = string_len(read_file(\"{}\"))\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{\n _ = print(\"before\")\n _ = if i > 0i64 then len(step([produce(size)], 0i64)) else 0i64\n _ = print(\"after\")\n append(acc, produce(3i64))\n}}\ndef run(seed: List[tensor[3, f32]]) -> i64 ! {{ IO }} = {{\n _ = {combine}(fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i), seed, range(0i64, {steps}i64))\n 42i64\n}}\nout = run([produce(3i64)])\n",
                    path.display()
                );
                let (success, output) = result_claims::run(&source, native);
                for unproved in [false, true] {
                    let wrapped =
                        result_claims::run(&forwarding_wrapper(&source, unproved), native);
                    assert_eq!((success, &output), (wrapped.0, &wrapped.1));
                }
                assert_eq!(success, steps < 2, "{source}\n{output}");
                assert_eq!(
                    output.lines().filter(|line| *line == "before").count(),
                    steps,
                    "{output}"
                );
                assert_eq!(
                    output.lines().filter(|line| *line == "after").count(),
                    steps.min(1),
                    "{output}"
                );
                if steps == 2 {
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
fn named_nominal_chain_receipts_keep_literal_and_scoped_binder_obligations() {
    for direct in [true, false] {
        for claim in ["3", "n"] {
            for combine in ["fold", "scan"] {
                for steps in [100, 400] {
                    let generic = if claim == "n" { "[n]" } else { "" };
                    let callback = if direct {
                        "step"
                    } else {
                        "fn (acc: Chain[tensor[*, f32]], i: i64) -> step(acc, i)"
                    };
                    let loop_expr = format!("{combine}({callback}, seed, range(0i64, {steps}i64))");
                    let output = if combine == "fold" {
                        format!("head({loop_expr})")
                    } else {
                        format!("len({loop_expr})")
                    };
                    let source = format!(
                        "type Chain[T] =\n | End {{ v: T }}\n | Link {{ v: T, next: Chain[T] }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step{generic}(acc: Chain[tensor[{claim}, f32]], i: i64) -> Chain[tensor[*, f32]] = Link {{ v: produce(3i64), next: acc }}\ndef head(c: Chain[tensor[*, f32]]) -> i64 = match c with {{ | End {{ v }} => shape(v, 0i32) | Link {{ v, next }} => shape(v, 0i32) }}\ndef run(seed: Chain[tensor[3, f32]]) -> i64 = {output}\nout = run(End {{ v: produce(3i64) }})\n"
                    );
                    let (work, origins) = counted_native(&source);
                    eprintln!(
                        "C named Chain direct={direct} claim={claim} {combine} K={steps}: {work}; origins={origins}"
                    );
                    assert!(work <= 64 * steps as u64 + 100, "{source}\n{work}");
                }
            }
        }
    }
}

#[test]
fn captured_shadowed_callee_keeps_its_own_formal_and_effects() {
    for native in [false, true] {
        for direct in [false, true] {
            for steps in [0, 1, 2] {
                for seed in [3, 5] {
                    let callback = if direct {
                        "step"
                    } else {
                        "fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i)"
                    };
                    let source = format!(
                        "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{\n _ = print(\"global\")\n append(acc, produce(3i64))\n}}\ndef run(seed: List[tensor[*, f32]], size: i64) -> i64 ! {{ IO }} = {{\n step = fn (acc: List[tensor[5, f32]], i: i64) -> {{\n _ = print(\"shadow\")\n append(acc, produce(size))\n }}\n _ = fold({callback}, seed, range(0i64, {steps}i64))\n 42i64\n}}\nout = run([produce({seed}i64)], 3i64)\n"
                    );
                    let (success, output) = result_claims::run(&source, native);
                    let failed = steps > 0 && (seed == 3 || steps > 1);
                    assert_eq!(success, !failed, "{source}\n{output}");
                    assert_eq!(
                        output.lines().filter(|line| *line == "global").count(),
                        0,
                        "{output}"
                    );
                    assert_eq!(
                        output.lines().filter(|line| *line == "shadow").count(),
                        usize::from(seed == 5 && steps > 0),
                        "{output}"
                    );
                    if failed {
                        assert!(
                            output.contains("numeric trap: domain in load at i64"),
                            "{output}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn named_empty_seed_binds_only_at_the_first_populated_entry() {
    for native in [false, true] {
        for claim in ["3", "n"] {
            for prepend in [false, true] {
                for combine in ["fold", "scan"] {
                    for steps in [0, 1, 2, 3] {
                        let generic = if claim == "n" { "[n]" } else { "" };
                        let construction = if prepend {
                            "concat([v], acc)"
                        } else {
                            "append(acc, v)"
                        };
                        let source = format!(
                            "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step{generic}(acc: List[tensor[{claim}, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{\n _ = print(\"body\")\n v = produce(if i == 0i64 then 5i64 else 3i64)\n {construction}\n}}\ndef run(seed: List[tensor[3, f32]]) -> i64 ! {{ IO }} = {{\n _ = {combine}(fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i), seed, range(0i64, {steps}i64))\n 42i64\n}}\nout = run([])\n"
                        );
                        let (success, output) = result_claims::run(&source, native);
                        let first_bad = if claim == "3" { 2 } else { 3 };
                        assert_eq!(success, steps < first_bad, "{source}\n{output}");
                        assert_eq!(
                            output.lines().filter(|line| *line == "body").count(),
                            steps.min(first_bad - 1),
                            "{output}"
                        );
                        if steps >= first_bad {
                            assert!(
                                output.contains("numeric trap: domain in load at i64"),
                                "{output}"
                            );
                            if claim == "n" {
                                assert!(
                                    output.contains(if prepend {
                                        "acc[0] axis 0 = 3, acc[1] axis 0 = 5"
                                    } else {
                                        "acc[0] axis 0 = 5, acc[1] axis 0 = 3"
                                    }),
                                    "{output}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn named_entry_receipts_restore_the_callee_witness_for_local_claims() {
    for native in [false, true] {
        for outer in ["*", "3"] {
            for combine in ["fold", "scan"] {
                for steps in [0, 1, 2] {
                    let dir = tempfile::tempdir().unwrap();
                    let path = dir.path().join("size");
                    std::fs::write(&path, "xxxxx").unwrap();
                    let bare_source = format!(
                        "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef independent[m](seed: List[tensor[m, f32]], size: i64) -> i64 = {{\n v: tensor[m, f32] = produce(size)\n len(seed)\n}}\ndef step[m](acc: List[tensor[m, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{\n _ = print(\"before\")\n _ = independent([produce(5i64)], 5i64)\n size = string_len(read_file(\"{}\"))\n v: tensor[m, f32] = produce(if i == 0i64 then 3i64 else size)\n _ = print(\"after\")\n old = acc\n older = old\n append(older, v)\n}}\ndef run(seed: List[tensor[3, f32]]) -> i64 ! {{ IO }} = {{\n _ = {combine}(fn (acc: List[tensor[{outer}, f32]], i: i64) -> step(acc, i), seed, range(0i64, {steps}i64))\n 42i64\n}}\nout = run([produce(3i64)])\n",
                        path.display()
                    );
                    for source in [bare_source.clone(), forwarding_wrapper(&bare_source, false)] {
                        let counterfactual = forwarding_wrapper(&bare_source, true);
                        let (old_success, old_output) = result_claims::run(&counterfactual, native);
                        let (success, output) = result_claims::run(&source, native);
                        assert_eq!(
                            (success, &output),
                            (old_success, &old_output),
                            "proved/unproved differ native={native}: {source}"
                        );
                        assert_eq!(success, steps < 2, "{source}\n{output}");
                        assert_eq!(
                            output.lines().filter(|line| *line == "before").count(),
                            steps,
                            "{output}"
                        );
                        assert_eq!(
                            output.lines().filter(|line| *line == "after").count(),
                            steps.min(1),
                            "{output}"
                        );
                        if steps == 2 {
                            assert!(
                                output.contains("numeric trap: domain in insert at i64"),
                                "{output}"
                            );
                        }
                        if steps == 1 {
                            let independent_bad =
                                source.replace("[produce(5i64)], 5i64)", "[produce(5i64)], 6i64)");
                            let (success, output) = result_claims::run(&independent_bad, native);
                            assert!(!success, "{output}");
                            assert!(
                                output.contains("extent `m`: claimed = 5, insert axis 0 = 6"),
                                "{output}"
                            );
                            assert_eq!(
                                output.lines().filter(|line| *line == "before").count(),
                                1,
                                "{output}"
                            );
                            assert!(!output.lines().any(|line| line == "after"), "{output}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn growing_receipts_do_not_discharge_independent_caller_or_result_claims() {
    for native in [false, true] {
        for combine in ["fold", "scan"] {
            for owner in ["caller", "result"] {
                for size in [3, 5] {
                    let dir = tempfile::tempdir().unwrap();
                    let path = dir.path().join("size");
                    std::fs::write(&path, "x".repeat(size)).unwrap();
                    let claimed = if combine == "fold" {
                        "List[tensor[3, f32]]"
                    } else {
                        "List[List[tensor[3, f32]]]"
                    };
                    let wildcard = claimed.replace("tensor[3", "tensor[*");
                    let output_type = if owner == "result" {
                        claimed.to_owned()
                    } else {
                        wildcard
                    };
                    let end = if owner == "caller" {
                        "out = consume(grow([produce(3i64)], size))"
                    } else {
                        "out = len(grow([produce(3i64)], size))"
                    };
                    let source = format!(
                        "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef consume(xs: {claimed}) -> i64 ! {{ IO }} = {{\n _ = print(\"consume\")\n len(xs)\n}}\ndef grow(seed: List[tensor[3, f32]], size: i64) -> {output_type} ! {{ IO }} = {combine}(fn (acc: List[tensor[3, f32]], i: i64) -> {{\n _ = print(\"body\")\n append(acc, produce(size))\n}}, seed, range(0i64, 1i64))\nsize = string_len(read_file(\"{}\"))\n{end}\n",
                        path.display()
                    );
                    let named = source.replace("def consume", "def step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] ! { IO } = { _ = print(\"body\")\n append(acc, produce(size)) }\ndef consume")
                        .replace("fn (acc: List[tensor[3, f32]], i: i64) -> {\n _ = print(\"body\")\n append(acc, produce(size))\n}", "fn (acc: List[tensor[3, f32]], i: i64) -> { old = acc\n item = i\n step(old, item) }");
                    let size_binding =
                        format!("size = string_len(read_file(\"{}\"))\n", path.display());
                    let named = named.replace(&size_binding, "").replacen(
                        "def step",
                        &format!("{size_binding}def step"),
                        1,
                    );
                    for source in [source, named] {
                        let (success, output) = result_claims::run(&source, native);
                        assert_eq!(success, size == 3, "{source}\n{output}");
                        assert_eq!(
                            output.lines().filter(|line| *line == "body").count(),
                            1,
                            "{output}"
                        );
                        if size == 5 {
                            eprintln!("independent {owner} {combine} native={native}: {output}");
                            assert!(
                                output.contains(&format!(
                                    "numeric trap: domain in {} at i64",
                                    if owner == "caller" { "load" } else { combine }
                                )),
                                "{output}"
                            );
                            assert_eq!(
                                output.lines().filter(|line| *line == "consume").count(),
                                0,
                                "{output}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn growing_named_receipts_restore_dependent_local_witnesses() {
    for native in [false, true] {
        for scan in [false, true] {
            for size in [3, 5] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("size");
                std::fs::write(&path, "x".repeat(size)).unwrap();
                let source = list_source(path.to_str().unwrap(), 2, "n", false, scan)
                    .replace("v = produce(size)", "bad: tensor[n, f32] = produce(if eq(i, 0i64) then 3i64 else size)\n  _ = print(shape(bad, 0i32))\n  v = produce(3i64)");
                let (success, output) = result_claims::run(&source, native);
                assert_eq!(success, size == 3, "{source}\n{output}");
                assert_eq!(
                    output.lines().filter(|line| *line == "before").count(),
                    2,
                    "{output}"
                );
                assert_eq!(
                    output.lines().filter(|line| *line == "after").count(),
                    if size == 3 { 2 } else { 1 },
                    "{output}"
                );
                if size == 5 {
                    assert!(
                        output.contains("numeric trap: domain in insert at i64"),
                        "{output}"
                    );
                }
            }
        }
    }
}

#[test]
fn original_linked_chain_witness_has_bounded_native_claim_work() {
    for formal in ["n", "unclaimed"] {
        for steps in [100, 400] {
            let source = format!(
                "type Chain[n] =\n | End {{ v: tensor[n, f32] }}\n | Link {{ v: tensor[n, f32], next: Chain[n] }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef grow[n](size: i64, k: i64) -> Chain[n] = fold(fn (acc: Chain[{formal}], i: i64) -> Link {{ v: produce(size), next: acc }}, End {{ v: produce(size) }}, range(0i64, k))\ndef head_width[n](c: Chain[n]) -> i64 = match c with {{ | End {{ v }} => shape(v, 0i32) | Link {{ v, next }} => shape(v, 0i32) }}\nout = head_width(grow(3i64, {steps}i64))\n"
            );
            let source = if formal == "unclaimed" {
                source
                    .replace("type Chain[n]", "type Chain[T]")
                    .replace("v: tensor[n, f32]", "v: T")
                    .replace("next: Chain[n]", "next: Chain[T]")
                    .replace("Chain[unclaimed]", "Chain[tensor[*, f32]]")
                    .replace("Chain[n]", "Chain[tensor[*, f32]]")
                    .replace("def grow[n]", "def grow")
                    .replace("def head_width[n]", "def head_width")
            } else {
                source
            };
            let (work, origins) = counted_native(&source);
            eprintln!(
                "C original Chain formal={formal} K={steps}: {work} claim work units; origins={origins}"
            );
            assert!(work <= 60 * steps as u64 + 60, "{work}");
        }
    }
}

#[test]
fn empty_list_seeds_bind_the_first_observed_named_witness_at_its_entry() {
    for native in [false, true] {
        for claim in ["3", "n"] {
            for scan in [false, true] {
                for steps in [0, 1, 2, 3] {
                    for size in [3, 5] {
                        let dir = tempfile::tempdir().unwrap();
                        let path = dir.path().join("size");
                        std::fs::write(&path, "x".repeat(size)).unwrap();
                        let source = list_source(path.to_str().unwrap(), steps, claim, false, scan)
                            .replace("out = run([produce(3i64)]", "out = run([]")
                            .replace(
                                "v = produce(size)",
                                "v = produce(if eq(i, 0i64) then size else 3i64)",
                            );
                        let (success, output) = result_claims::run(&source, native);
                        let first_bad_entry = if claim == "3" { 2 } else { 3 };
                        let bad = size == 5 && steps >= first_bad_entry;
                        assert_eq!(success, !bad, "{source}\n{output}");
                        let callbacks = if bad { first_bad_entry - 1 } else { steps };
                        assert_eq!(
                            output.lines().filter(|line| *line == "before").count(),
                            callbacks,
                            "{output}"
                        );
                        assert_eq!(
                            output.lines().filter(|line| *line == "after").count(),
                            callbacks,
                            "{output}"
                        );
                        if bad {
                            assert!(
                                output.contains("numeric trap: domain in load at i64"),
                                "{output}"
                            );
                            if claim == "n" {
                                assert!(
                                    output.contains("acc[0] axis 0 = 5, acc[1] axis 0 = 3"),
                                    "{output}"
                                );
                            }
                        } else {
                            assert!(output.contains(&format!("out = {steps}")), "{output}");
                        }
                    }
                }
            }
        }
    }
}

fn immutable_alias_source(
    claim: &str,
    entry: &str,
    combine: &str,
    steps: usize,
    operation: &str,
) -> String {
    let generic = if claim == "n" { "[n]" } else { "" };
    let construction = match operation {
        "append" => "append(older, produce(3i64))",
        "prefix" => "concat([produce(3i64)], older)",
        "suffix" => "concat(older, [produce(3i64)])",
        _ => unreachable!(),
    };
    let body = format!("{{ old = acc\n older = old\n {construction} }}");
    let callback = match entry {
        "inline" => format!("fn (acc: List[tensor[{claim}, f32]], i: i64) -> {body}"),
        "direct" => "step".to_owned(),
        "composed" => "fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i)".to_owned(),
        _ => unreachable!(),
    };
    format!(
        "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step{generic}(acc: List[tensor[{claim}, f32]], i: i64) -> List[tensor[*, f32]] = {body}\ndef run{generic}(seed: List[tensor[{claim}, f32]]) -> i64 = len({combine}({callback}, seed, range(0i64, {steps}i64)))\nout = run([produce(3i64)])\n"
    )
}

#[test]
fn immutable_local_aliases_retain_bounded_accumulator_entry_work() {
    let mut excessive = Vec::new();
    for operation in ["append", "prefix", "suffix"] {
        for claim in ["3", "n"] {
            for entry in ["inline", "direct", "composed"] {
                for combine in ["fold", "scan"] {
                    for steps in [100, 400] {
                        let source =
                            immutable_alias_source(claim, entry, combine, steps, operation);
                        let (work, origins) = counted_native(&source);
                        eprintln!(
                            "C immutable aliases {operation} {entry} {combine} claim={claim} K={steps}: {work}; origins={origins}"
                        );
                        if work > 64 * steps as u64 + 100 {
                            excessive.push(format!(
                                "{operation} {entry} {combine} {claim} K={steps}: {work}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(excessive.is_empty(), "{}", excessive.join("\n"));
}

fn immutable_nominal_alias_source(
    collection: &str,
    claim: &str,
    entry: &str,
    combine: &str,
    steps: usize,
) -> String {
    let generic = if claim == "n" { "[n]" } else { "" };
    let (declaration, seed, construction) = if collection == "Chain" {
        (
            "type Chain[T] =\n | End { v: T }\n | Link { v: T, next: Chain[T] }",
            "End { v: produce(3i64) }",
            "Link { v: produce(3i64), next: older }",
        )
    } else {
        (
            "type Box[T] =\n | Base { v: T }\n | Wrap { inner: Box[T] }",
            "Base { v: produce(3i64) }",
            "Wrap { inner: older }",
        )
    };
    let formal = format!("{collection}[tensor[{claim}, f32]]");
    let result = format!("{collection}[tensor[*, f32]]");
    let body = format!("{{ old = acc\n older = old\n {construction} }}");
    let callback = match entry {
        "inline" => format!("fn (acc: {formal}, i: i64) -> {body}"),
        "direct" => "step".to_owned(),
        "composed" => format!("fn (acc: {result}, i: i64) -> step(acc, i)"),
        _ => unreachable!(),
    };
    format!(
        "{declaration}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step{generic}(acc: {formal}, i: i64) -> {result} = {body}\ndef run{generic}(seed: {formal}) -> i64 = {{ _ = {combine}({callback}, seed, range(0i64, {steps}i64))\n {steps}i64 }}\nout = run({seed})\n"
    )
}

#[test]
fn immutable_nominal_aliases_retain_bounded_accumulator_entry_work() {
    let mut excessive = Vec::new();
    for collection in ["Chain", "Box"] {
        for claim in ["3", "n"] {
            for entry in ["inline", "direct", "composed"] {
                for combine in ["fold", "scan"] {
                    for steps in [100, 400] {
                        let source = immutable_nominal_alias_source(
                            collection, claim, entry, combine, steps,
                        );
                        let (work, origins) = counted_native(&source);
                        eprintln!(
                            "C immutable {collection} aliases {entry} {combine} claim={claim} K={steps}: {work}; origins={origins}"
                        );
                        if work > 64 * steps as u64 + 100 {
                            excessive.push(format!(
                                "{collection} {entry} {combine} {claim} K={steps}: {work}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(excessive.is_empty(), "{}", excessive.join("\n"));
}

#[test]
fn immutable_alias_entries_check_seed_before_bound_effects() {
    for native in [false, true] {
        for claim in ["3", "n"] {
            for direct in [false, true] {
                for steps in [0, 1, 2] {
                    for size in [3, 5] {
                        let dir = tempfile::tempdir().unwrap();
                        let path = dir.path().join("size");
                        std::fs::write(&path, "x".repeat(size)).unwrap();
                        let generic = if claim == "n" { "[n]" } else { "" };
                        let body = "{ old = acc\n _ = print(\"alias\")\n older = old\n append(older, produce(3i64)) }";
                        let callback = if direct {
                            "step".to_owned()
                        } else {
                            format!("fn (acc: List[tensor[{claim}, f32]], i: i64) -> {body}")
                        };
                        let source = format!(
                            "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step{generic}(acc: List[tensor[{claim}, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {body}\ndef run{generic}(seed: List[tensor[*, f32]]) -> i64 ! {{ IO }} = {{ _ = fold({callback}, seed, range(0i64, {steps}i64))\n 42i64 }}\nout = run([produce(3i64), produce(string_len(read_file(\"{}\")))])\n",
                            path.display()
                        );
                        let (success, output) = result_claims::run(&source, native);
                        let bad = size == 5 && steps > 0;
                        assert_eq!(success, !bad, "{source}\n{output}");
                        assert_eq!(
                            output.lines().filter(|line| *line == "alias").count(),
                            if bad { 0 } else { steps },
                            "{output}"
                        );
                        if bad {
                            assert!(
                                output.contains("numeric trap: domain in load at i64"),
                                "{output}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn single_local_aliases_have_bounded_work_and_equal_value_controls_do_not() {
    for collection in ["List", "Chain", "Box"] {
        for claim in ["3", "n"] {
            for steps in [100, 400] {
                let source = if collection == "List" {
                    immutable_alias_source(claim, "direct", "fold", steps, "append")
                } else {
                    immutable_nominal_alias_source(collection, claim, "direct", "fold", steps)
                }
                .replace(" older = old\n", "")
                .replace("older", "old");
                for unproved in [false, true] {
                    let source = if unproved {
                        source.replace("old = acc", "old = if i == 0i64 then acc else acc")
                    } else {
                        source.clone()
                    };
                    let (work, origins) = counted_native(&source);
                    eprintln!(
                        "C single alias {collection} claim={claim} K={steps} unproved={unproved}: {work}; origins={origins}"
                    );
                    if unproved {
                        assert!(
                            work >= steps as u64 * steps as u64 / 2,
                            "full equal-value entry walk: {work}"
                        );
                    } else {
                        assert!(
                            work <= 64 * steps as u64 + 100,
                            "single original-root alias: {work}"
                        );
                    }
                }
            }
        }
    }
}

fn forwarding_wrapper_source(
    seed_size: usize,
    steps: usize,
    hops: usize,
    unproved: bool,
) -> String {
    let first = if unproved {
        "if i == 0i64 then acc else acc"
    } else {
        "acc"
    };
    let mut bindings = format!("old0 = {first}\n item0 = i\n");
    for hop in 1..hops {
        bindings.push_str(&format!(
            "old{hop} = old{}\n item{hop} = item{}\n",
            hop - 1,
            hop - 1
        ));
    }
    format!(
        "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] = append(acc, produce(3i64))\ndef run(seed: List[tensor[3, f32]]) -> i64 = len(fold(fn (acc: List[tensor[*, f32]], i: i64) -> {{ {bindings} step(old{}, item{}) }}, seed, range(0i64, {steps}i64)))\nout = run(map(fn (i: i64) -> produce(3i64), range(0i64, {seed_size}i64)))\n",
        hops - 1,
        hops - 1
    )
}

#[test]
fn lexical_forwarding_wrappers_have_bounded_native_entry_work() {
    let mut excessive = Vec::new();
    for hops in [1, 2] {
        for seed in [100, 400] {
            for steps in [100, 400] {
                for unproved in [false, true] {
                    let source = forwarding_wrapper_source(seed, steps, hops, unproved);
                    let (work, origins) = counted_native(&source);
                    eprintln!(
                        "forwarding wrapper C N={seed} K={steps} L={hops} unproved={unproved}: work={work} origins={origins}"
                    );
                    if unproved {
                        assert!(work >= (seed * steps) as u64, "{work}");
                    } else if work > 96 * (seed + steps) as u64 + 100 {
                        excessive.push(format!("N={seed} K={steps} L={hops}: {work}"));
                    }
                }
            }
        }
    }
    assert!(excessive.is_empty(), "{}", excessive.join("\n"));
}

#[test]
fn lexical_forwarding_preserves_weak_strong_entry_and_final_invalid_effects() {
    for native in [false, true] {
        for outer in ["*", "3"] {
            for steps in [0, 1, 2] {
                let dir = tempfile::tempdir().unwrap();
                let size_path = dir.path().join("size");
                std::fs::write(&size_path, "xxxxx").unwrap();
                let source = format!(
                    "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\nsize = string_len(read_file(\"{}\"))\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{ _ = print(\"body\")\n append(acc, produce(size)) }}\ndef run(seed: List[tensor[3, f32]]) -> i64 ! {{ IO }} = {{ _ = fold(fn (acc: List[tensor[{outer}, f32]], i: i64) -> {{ old = acc\n item = i\n _ = print(\"wrapper\")\n step(old, item) }}, seed, range(0i64, {steps}i64))\n 42i64 }}\nout = run([produce(3i64)])\n",
                    size_path.display()
                );
                let (success, output) = result_claims::run(&source, native);
                assert_eq!(success, steps < 2, "{source}\n{output}");
                let wrappers = if steps < 2 {
                    steps
                } else if outer == "*" {
                    2
                } else {
                    1
                };
                assert_eq!(
                    output.lines().filter(|line| *line == "wrapper").count(),
                    wrappers,
                    "{output}"
                );
                assert_eq!(
                    output.lines().filter(|line| *line == "body").count(),
                    usize::from(steps > 0),
                    "{output}"
                );
                if !success {
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
fn lexical_forwarding_nominal_fold_scan_retains_independent_named_entries() {
    for collection in ["Chain", "Box"] {
        for claim in ["3", "n"] {
            for combine in ["fold", "scan"] {
                for hops in [1, 2] {
                    for steps in [100, 400] {
                        let source = immutable_nominal_alias_source(
                            collection, claim, "composed", combine, steps,
                        );
                        let mut aliases = "old0 = acc\n item0 = i\n".to_owned();
                        for hop in 1..hops {
                            aliases.push_str(&format!(
                                "old{hop} = old{}\n item{hop} = item{}\n",
                                hop - 1,
                                hop - 1
                            ));
                        }
                        let source = source.replace(
                            "step(acc, i)",
                            &format!("{{ {aliases} step(old{}, item{}) }}", hops - 1, hops - 1),
                        );
                        let (work, origins) = counted_native(&source);
                        eprintln!(
                            "forwarding nominal C {collection}/{claim}/{combine} K={steps} L={hops}: work={work} origins={origins}"
                        );
                        assert!(work <= 96 * steps as u64 + 200, "{source}\n{work}");
                    }
                }
            }
        }
    }
}

#[test]
fn lexical_forwarding_prior_same_callee_call_keeps_independent_entry_effects() {
    for native in [false, true] {
        for strong in [false, true] {
            for unproved in [false, true] {
                let outer = if strong { "3" } else { "*" };
                let alias = if unproved {
                    "if i == 0i64 then acc else acc"
                } else {
                    "acc"
                };
                let source = format!(
                    "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{ _ = print(\"callee\")\n append(acc, produce(3i64)) }}\ndef run(seed: List[tensor[3, f32]]) -> i64 ! {{ IO }} = len(fold(fn (acc: List[tensor[{outer}, f32]], i: i64) -> {{ old = {alias}\n item = i\n _ = print(\"prefix\")\n _ = step(old, item)\n _ = print(\"between\")\n step(old, item) }}, seed, range(0i64, 2i64)))\nout = run([produce(3i64)])\n"
                );
                let (success, output) = result_claims::run(&source, native);
                assert!(success, "{source}\n{output}");
                let effects = output
                    .lines()
                    .filter(|line| matches!(*line, "prefix" | "callee" | "between"))
                    .collect::<Vec<_>>();
                assert_eq!(
                    effects,
                    [
                        "prefix", "callee", "between", "callee", "prefix", "callee", "between",
                        "callee"
                    ],
                    "{output}"
                );
                assert!(output.contains("out = 3"), "{output}");
            }
        }
    }
}

#[test]
fn lexical_forwarding_empty_and_bad_seed_retain_first_independent_entry() {
    for native in [false, true] {
        for outer in ["*", "3"] {
            for steps in [0, 1, 2] {
                for unproved in [false, true] {
                    let dir = tempfile::tempdir().unwrap();
                    let path = dir.path().join("size");
                    std::fs::write(&path, "xxxxx").unwrap();
                    let alias = if unproved {
                        "if i == 0i64 then acc else acc"
                    } else {
                        "acc"
                    };
                    let source = format!(
                        "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{ _ = print(\"body\")\n append(acc, produce(3i64)) }}\ndef run(seed: List[tensor[*, f32]]) -> i64 ! {{ IO }} = {{ _ = fold(fn (acc: List[tensor[{outer}, f32]], i: i64) -> {{ old = {alias}\n item = i\n _ = print(\"wrapper\")\n step(old, item) }}, seed, range(0i64, {steps}i64))\n 42i64 }}\nout = run([produce(string_len(read_file(\"{}\")))])\n",
                        path.display()
                    );
                    let (success, output) = result_claims::run(&source, native);
                    assert_eq!(success, steps == 0, "{source}\n{output}");
                    assert_eq!(
                        output.lines().filter(|line| *line == "body").count(),
                        0,
                        "{output}"
                    );
                    assert_eq!(
                        output.lines().filter(|line| *line == "wrapper").count(),
                        usize::from(steps > 0 && outer == "*"),
                        "{output}"
                    );
                    if !success {
                        assert!(
                            output.contains("numeric trap: domain in load at i64"),
                            "{output}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn lexical_forwarding_unproved_item_and_shadowed_callee_keep_full_semantics() {
    for native in [false, true] {
        for mode in ["item", "callee", "acc", "item_shadow"] {
            let prefix = match mode {
                "item" => "old = acc\n item = if i == 0i64 then i else i",
                "callee" => {
                    "old = acc\n item = i\n step = fn (xs: List[tensor[*, f32]], n: i64) -> xs"
                }
                "acc" => "old = acc\n acc = [produce(3i64)]\n item = i",
                "item_shadow" => "old = acc\n i = 0i64\n item = i",
                _ => unreachable!(),
            };
            let source = forwarding_wrapper_source(3, 2, 1, false)
                .replace("old0 = acc\n item0 = i\n", &format!("{prefix}\n"))
                .replace("step(old0, item0)", "step(old, item)");
            let (success, output) = result_claims::run(&source, native);
            assert!(success, "{source}\n{output}");
            let expected = if mode == "callee" { 3 } else { 5 };
            assert!(output.contains(&format!("out = {expected}")), "{output}");
        }
    }
}

fn forwarding_wrapper(source: &str, unproved: bool) -> String {
    let root = if unproved {
        "if i == 0i64 then acc else acc"
    } else {
        "acc"
    };
    source.replace(
        "-> step(acc, i)",
        &format!(
            "-> {{ old = {root}\n older = old\n item = i\n item2 = item\n step(older, item2) }}"
        ),
    )
}

#[test]
fn lexical_forwarding_example_preserves_the_independent_callee_claim() {
    let source = include_str!("../../../examples/forwarded_accumulator_entry.ch");
    let negative = source
        .replacen("acc: List[tensor[3, f32]]", "acc: List[tensor[5, f32]]", 1)
        .replacen(
            "[1.0f32, 2.0f32, 3.0f32]",
            "[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]",
            1,
        );
    for native in [false, true] {
        let (success, output) = result_claims::run(source, native);
        assert!(success && output.contains("out = 5"), "{output}");
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("contradictory.ch");
    std::fs::write(&path, negative).unwrap();
    let rejected = assert_cmd::Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let diagnostic = format!(
        "{}{}",
        String::from_utf8_lossy(&rejected.stdout),
        String::from_utf8_lossy(&rejected.stderr)
    );
    assert!(diagnostic.contains("DimensionMismatch"), "{diagnostic}");
}

#[test]
fn lexical_forwarding_stationary_named_entries_keep_seed_work_bounded() {
    for seed in [100, 400] {
        for steps in [100, 400] {
            for hops in [1, 2] {
                for unproved in [false, true] {
                    let source = forwarding_wrapper_source(seed, steps, hops, unproved)
                        .replace("append(acc, produce(3i64))", "acc");
                    let (work, origins) = counted_native(&source);
                    eprintln!(
                        "stationary forwarding C N={seed} K={steps} L={hops} unproved={unproved}: work={work} origins={origins}"
                    );
                    if unproved {
                        assert!(work >= (seed * steps) as u64, "{work}");
                    } else {
                        assert!(work <= 96 * (seed + steps) as u64 + 100, "{work}");
                    }
                }
            }
        }
    }
}

#[test]
fn lexical_forwarding_nominal_initial_size_and_steps_have_separate_entry_bounds() {
    for collection in ["Chain", "Box"] {
        for seed in [100, 400] {
            for steps in [100, 400] {
                let source =
                    immutable_nominal_alias_source(collection, "3", "composed", "fold", steps);
                let (base, construction) = if collection == "Chain" {
                    ("End { v }", "Link { v, next: acc }")
                } else {
                    ("Base { v }", "Wrap { inner: acc }")
                };
                let builder = format!(
                    "def make_seed[T](v: T, count: i64) -> {collection}[T] = fold(fn (acc: {collection}[T], i: i64) -> {construction}, {base}, range(0i64, count))\n"
                );
                let source = source.replace("out = run(", &format!("{builder}out = run("));
                let original = if collection == "Chain" {
                    "End { v: produce(3i64) }"
                } else {
                    "Base { v: produce(3i64) }"
                };
                let source = source.replace(
                    &format!("out = run({original})"),
                    &format!("out = run(make_seed(produce(3i64), {seed}i64))"),
                );
                for unproved in [false, true] {
                    let source = forwarding_wrapper(&source, unproved);
                    let (work, origins) = counted_native(&source);
                    eprintln!(
                        "nominal initial-size forwarding {collection} N={seed} K={steps} unproved={unproved}: work={work} origins={origins}"
                    );
                    if unproved {
                        assert!(work >= (seed * steps) as u64, "{work}");
                    } else {
                        assert!(work <= 96 * (seed + steps) as u64 + 200, "{work}");
                    }
                }
            }
        }
    }
}

#[test]
fn lexical_forwarding_does_not_discharge_an_earlier_identical_call_entry() {
    for seed in [100, 400] {
        for steps in [100, 400] {
            let source = forwarding_wrapper_source(seed, steps, 2, false).replace(
                "step(old1, item1) }",
                "_ = step(old1, item1)\n step(old1, item1) }",
            );
            let (work, origins) = counted_native(&source);
            eprintln!(
                "independent earlier forwarding call N={seed} K={steps}: work={work} origins={origins}"
            );
            assert!(
                work >= (seed * steps) as u64,
                "independent entry was skipped: {work}"
            );
        }
    }
}

#[test]
fn lexical_forwarding_keeps_independent_wrapper_local_claims_in_order() {
    for native in [false, true] {
        for combine in ["fold", "scan"] {
            for steps in [0, 1, 2] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("size");
                std::fs::write(&path, "xxxxx").unwrap();
                let source = format!(
                    "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\nsize = string_len(read_file(\"{}\"))\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] ! {{ IO }} = {{ _ = print(\"body\")\n append(acc, produce(size)) }}\ndef run(seed: List[tensor[3, f32]]) -> i64 ! {{ IO }} = {{ _ = {combine}(fn (acc: List[tensor[*, f32]], i: i64) -> {{ _ = print(\"prefix\")\n old: List[tensor[3, f32]] = acc\n _ = print(\"local\")\n item = i\n step(old, item) }}, seed, range(0i64, {steps}i64))\n 42i64 }}\nout = run([produce(3i64)])\n",
                    path.display()
                );
                let unproved = source.replace(
                    "old: List[tensor[3, f32]] = acc",
                    "old: List[tensor[3, f32]] = if i == 0i64 then acc else acc",
                );
                let actual = result_claims::run(&source, native);
                let baseline = result_claims::run(&unproved, native);
                assert_eq!(actual, baseline, "{source}");
                let (success, output) = actual;
                assert_eq!(success, steps < 2, "{output}");
                assert_eq!(
                    output.lines().filter(|line| *line == "prefix").count(),
                    steps,
                    "{output}"
                );
                assert_eq!(
                    output.lines().filter(|line| *line == "local").count(),
                    steps.min(1),
                    "{output}"
                );
                assert_eq!(
                    output.lines().filter(|line| *line == "body").count(),
                    steps.min(1),
                    "{output}"
                );
                if !success {
                    // Preserve the independently reproduced complete-check
                    // producer in each lane; the unproved path above is exact.
                    let producer = if native { "append" } else { "load" };
                    assert!(
                        output.contains(&format!("numeric trap: domain in {producer} at i64")),
                        "{output}"
                    );
                }
            }
        }
    }
}
