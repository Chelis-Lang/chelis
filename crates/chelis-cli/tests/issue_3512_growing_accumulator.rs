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
  v = produce(size)
  _ = print("after")
  Link {{ v, next: acc }}
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
        "concat([v], acc)"
    } else {
        "append(acc, v)"
    };
    let combinator = if scan { "scan" } else { "fold" };
    let generic = if claim == "n" { "[n]" } else { "" };
    format!(
        r#"def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def run{generic}(seed: List[tensor[{claim}, f32]], size: i64, steps: i64) -> i64 ! {{ IO }} = len({combinator}(fn (acc: List[tensor[{claim}, f32]], i: i64) -> {{
  _ = print("before")
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
static chelis_value __probe_index(const chelis_list *x, int64_t slot) {
    ++__probe_entry;
    return chelis_list_index(x, slot);
}
#define chelis_tensor_shape __probe_shape
#define chelis_tensor_rank __probe_rank
#define chelis_adt_get_field __probe_field
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
        .last()
        .expect("executed counters");
    let memory = output
        .lines()
        .filter_map(|line| line.strip_prefix("__memory__="))
        .last()
        .expect("origin allocation bytes");
    eprintln!("native invocation origin bytes retained until teardown: {memory}");
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
                source = source.replace(
                    "append(acc, v)",
                    if route == "append_step" {
                        "append_step(acc, v)"
                    } else {
                        "step(acc, v)"
                    },
                );
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
fn named_definition_accumulator_claim_is_a_measured_full_walk_residue() {
    for combine in ["fold", "scan"] {
        for steps in [100, 400] {
            let source = format!(
                "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[3, f32]], i: i64) -> List[tensor[*, f32]] = append(acc, produce(3i64))\ndef run(seed: List[tensor[3, f32]], steps: i64) -> i64 = len({combine}(fn (acc: List[tensor[*, f32]], i: i64) -> step(acc, i), seed, range(0i64, steps)))\nout = run([produce(3i64)], {steps}i64)\n"
            );
            let (work, origins) = counted_native(&source);
            eprintln!("C named definition {combine} K={steps}: reads {work}; origins {origins}");
            assert!(
                work >= steps as u64 * steps as u64,
                "named-call entries retain complete checks: {work}"
            );
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
