//! spec/04 §4.7: reclaiming private provenance preserves producer attribution,
//! independent claims and effects. Metadata follows live values/constructions,
//! rather than retaining dead loop temporaries until the invocation returns.
mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

use assert_cmd::Command;
use std::path::PathBuf;

// The allocator ledger observes actual allocation sizes and frees. Payload
// allocation in the separately linked runtime is outside these two families.
const ALLOCATOR_PROBE: &str = r#"
typedef struct __probe_allocation {
    void *pointer;
    size_t bytes;
    int family;
    struct __probe_allocation *next;
} __probe_allocation;
static __probe_allocation *__probe_allocations = NULL;
static size_t __probe_current[3] = {0};
static size_t __probe_peak[3] = {0};
static int __probe_family(const char *function) {
    if (strstr(function, "__chelis_host_result_origin") != NULL) return 1;
    if (strstr(function, "__chelis_host_result_claim") != NULL ||
        strcmp(function, "__chelis_project_host_result_claims") == 0 ||
        strcmp(function, "__chelis_join_host_result_claim") == 0) return 2;
    return 0;
}
static void __probe_record(void *pointer, size_t bytes, int family) {
    if (pointer == NULL || family == 0) return;
    __probe_allocation *record = malloc(sizeof *record);
    if (record == NULL) abort();
    *record = (__probe_allocation){ pointer, bytes, family, __probe_allocations };
    __probe_allocations = record;
    __probe_current[family] += bytes;
    if (__probe_current[family] > __probe_peak[family]) __probe_peak[family] = __probe_current[family];
}
static __probe_allocation **__probe_find(void *pointer) {
    __probe_allocation **at = &__probe_allocations;
    while (*at != NULL && (*at)->pointer != pointer) at = &(*at)->next;
    return at;
}
static void *__probe_malloc(size_t bytes, const char *function) {
    void *pointer = malloc(bytes);
    __probe_record(pointer, bytes, __probe_family(function));
    return pointer;
}
static void *__probe_calloc(size_t count, size_t bytes, const char *function) {
    void *pointer = calloc(count, bytes);
    if (bytes != 0 && count > SIZE_MAX / bytes) abort();
    __probe_record(pointer, count * bytes, __probe_family(function));
    return pointer;
}
static void *__probe_realloc(void *pointer, size_t bytes, const char *function) {
    __probe_allocation **at = __probe_find(pointer);
    __probe_allocation *old = *at;
    int family = old != NULL ? old->family : __probe_family(function);
    void *next = realloc(pointer, bytes);
    if (next == NULL && bytes != 0) return NULL;
    if (old != NULL) {
        *at = old->next;
        __probe_current[family] -= old->bytes;
        free(old);
    }
    __probe_record(next, bytes, family);
    return next;
}
static void __probe_free(void *pointer) {
    __probe_allocation **at = __probe_find(pointer);
    __probe_allocation *old = *at;
    if (old != NULL) {
        *at = old->next;
        __probe_current[old->family] -= old->bytes;
        free(old);
    }
    free(pointer);
}
static void __probe_report(void) {
    fprintf(stderr, "__live_metadata__=%zu,%zu,%zu,%zu\n",
        __probe_current[1], __probe_peak[1], __probe_current[2], __probe_peak[2]);
}
static void __probe_before_teardown(void) {
    fprintf(stderr, "__before_metadata_teardown__=%zu,%zu\n",
        __probe_current[1], __probe_current[2]);
}
#define malloc(bytes) __probe_malloc((bytes), __func__)
#define calloc(count, bytes) __probe_calloc((count), (bytes), __func__)
#define realloc(pointer, bytes) __probe_realloc((pointer), (bytes), __func__)
#define free(pointer) __probe_free((pointer))
"#;

#[derive(Debug)]
struct LiveMetadata {
    before_teardown_origin: usize,
    before_teardown_claim: usize,
    current_origin: usize,
    peak_origin: usize,
    current_claim: usize,
    peak_claim: usize,
}

fn live_metadata(name: &str, source: &str) -> LiveMetadata {
    live_metadata_with_sanitizers(name, source, false)
}

fn live_metadata_with_sanitizers(name: &str, source: &str, sanitized: bool) -> LiveMetadata {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/issue3457");
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let source_path = dir.join("lifetime.ch");
    std::fs::write(&source_path, source).unwrap();
    let destination = dir.join("c");
    let built = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--emit-c", "--allow-style-violations"])
        .arg(&source_path)
        .args(["--target", "c", "-o"])
        .arg(&destination)
        .output()
        .unwrap();
    assert!(built.status.success(), "{source}\n{built:?}");
    let path = destination.join("lifetime.c");
    let mut c = std::fs::read_to_string(&path).unwrap();
    let include = c.rfind("#include ").unwrap();
    let header_end = include + c[include..].find('\n').unwrap() + 1;
    c.insert_str(header_end, ALLOCATOR_PROBE);
    let destroy = "static void __chelis_host_result_origin_arena_destroy(__chelis_host_result_origin_arena *arena) {";
    assert!(c.contains(destroy));
    c = c.replace(
        destroy,
        &format!("{destroy}\n    __probe_before_teardown();"),
    );
    assert!(c.contains("int main(void) {"));
    c = c.replace(
        "int main(void) {",
        "int main(void) { atexit(__probe_report);",
    );
    std::fs::write(&path, c).unwrap();
    if sanitized {
        let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
            chelis_backend_c::toolchain::CodegenRequirements {
                wants_openmp: true,
                needs_blas: false,
            },
        );
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(destination.join(chelis_runtime_bundle::RECEIPT_FILE_NAME)).unwrap(),
        )
        .unwrap();
        let archive = receipt["archive"].as_str().expect("staged archive path");
        assert_eq!(std::path::Path::new(archive).components().count(), 1);
        let archive = destination.join(archive);
        use sha2::{Digest, Sha256};
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(&archive).unwrap())),
            receipt["archive_sha256"]
                .as_str()
                .expect("staged archive digest")
        );
        let linked = std::process::Command::new(toolchain.compiler)
            .current_dir(&destination)
            .args([
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
            ])
            .args(toolchain.compile_flags)
            .arg("lifetime.c")
            .arg(&archive)
            .args(toolchain.link_flags)
            .args(["-o", "lifetime"])
            .status()
            .unwrap();
        assert!(linked.success());
    } else {
        assert!(common::link_generated(&destination, "lifetime.c", "lifetime").success());
    }
    let output = std::process::Command::new(destination.join("lifetime"))
        .env(
            "ASAN_OPTIONS",
            "detect_stack_use_after_return=1:detect_leaks=0",
        )
        .output()
        .unwrap();
    std::fs::write(dir.join("stdout.txt"), &output.stdout).unwrap();
    std::fs::write(dir.join("stderr.txt"), &output.stderr).unwrap();
    assert!(output.status.success(), "{source}\n{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("out = 3"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    let counts = stderr
        .lines()
        .find_map(|line| line.strip_prefix("__live_metadata__="))
        .unwrap();
    let counts = counts
        .split(',')
        .map(|n| n.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let before = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("__before_metadata_teardown__="))
        .next_back()
        .unwrap();
    let before = before
        .split(',')
        .map(|n| n.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let measured = LiveMetadata {
        before_teardown_origin: before[0],
        before_teardown_claim: before[1],
        current_origin: counts[0],
        peak_origin: counts[1],
        current_claim: counts[2],
        peak_claim: counts[3],
    };
    eprintln!("{name}: {measured:?}");
    assert_eq!(
        measured.current_origin, 0,
        "invocation releases every origin"
    );
    assert_eq!(
        measured.current_claim, 0,
        "invocation releases every projected frame"
    );
    measured
}

fn fixed_box_source(claimed: bool, fresh: bool, steps: usize) -> String {
    let claim = if claimed { "3" } else { "*" };
    let value = if fresh { "produce(size)" } else { "value(acc)" };
    format!(
        "type Box[T] =\n | Box {{ v: T }}\ndef value[T](b: Box[T]) -> T = match b with {{ | Box {{ v }} => v }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: Box[tensor[{claim}, f32]], i: i64, size: i64) -> Box[tensor[{claim}, f32]] = Box {{ v: {value} }}\ndef run(size: i64) -> i64 = {{\n b = fold(fn (acc: Box[tensor[{claim}, f32]], i: i64) -> step(acc, i, size), Box {{ v: produce(size) }}, range(0i64, {steps}i64))\n shape(value(b), 0i32)\n}}\nout = run(3i64)\n"
    )
}

#[test]
fn fixed_live_box_origins_and_projected_frames_do_not_accumulate_iterations() {
    let mut all_measurements = Vec::new();
    for claimed in [false, true] {
        for fresh in [false, true] {
            let measurements = [100, 400, 1600, 6400].map(|steps| {
                let name = format!("box-claimed{claimed}-fresh{fresh}-k{steps}");
                live_metadata(&name, &fixed_box_source(claimed, fresh, steps))
            });
            all_measurements.push(measurements);
        }
    }
    for measurements in all_measurements {
        for later in &measurements[1..] {
            assert!(
                later.peak_origin <= measurements[0].peak_origin + 1024,
                "{measurements:?}"
            );
            assert!(
                later.peak_claim <= measurements[0].peak_claim + 1024,
                "{measurements:?}"
            );
            assert!(
                later.before_teardown_origin <= measurements[0].before_teardown_origin + 1024,
                "{measurements:?}"
            );
            assert!(
                later.before_teardown_claim <= measurements[0].before_teardown_claim + 1024,
                "{measurements:?}"
            );
        }
    }
}

#[test]
fn fixed_live_list_reconstruction_has_bounded_metadata() {
    let mut all_measurements = Vec::new();
    for claimed in [false, true] {
        let claim = if claimed { "3" } else { "*" };
        let measured = [100, 400, 1600].map(|steps| {
            let source = format!("def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: List[tensor[{claim}, f32]], i: i64) -> List[tensor[{claim}, f32]] = [produce(3i64), produce(3i64), produce(3i64)]\ndef run(seed: List[tensor[{claim}, f32]]) -> i64 = len(fold(step, seed, range(0i64, {steps}i64)))\nout = run([produce(3i64), produce(3i64), produce(3i64)])\n");
            live_metadata(&format!("list-claimed{claimed}-k{steps}"), &source)
        });
        all_measurements.push(measured);
    }
    for measured in all_measurements {
        for later in &measured[1..] {
            assert!(
                later.peak_origin <= measured[0].peak_origin + 1024,
                "{measured:?}"
            );
            assert!(
                later.peak_claim <= measured[0].peak_claim + 1024,
                "{measured:?}"
            );
        }
    }
}

#[test]
fn escaped_growing_chain_keeps_necessary_live_origins() {
    // Negative counterpart to the bounded fixed-state oracle: these origins
    // remain reachable through the returned chain and must not be discarded.
    let measured = [100, 400].map(|steps| {
        let source = format!("type Chain[T] =\n | End {{ v: T }}\n | Link {{ v: T, next: Chain[T] }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef first[T](c: Chain[T]) -> T = match c with {{ | End {{ v }} => v | Link {{ v, next }} => v }}\ndef grow(seed: Chain[tensor[*, f32]]) -> Chain[tensor[*, f32]] = fold(fn (acc: Chain[tensor[*, f32]], i: i64) -> Link {{ v: produce(3i64), next: acc }}, seed, range(0i64, {steps}i64))\nout = shape(first(grow(End {{ v: produce(3i64) }})), 0i32)\n");
        live_metadata(&format!("escaped-chain-k{steps}"), &source)
    });
    assert!(
        measured[1].peak_origin > measured[0].peak_origin,
        "{measured:?}"
    );
}

#[test]
fn returned_prefix_suffix_view_keeps_backing_origins_after_callee_exit() {
    let source = "def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef selected(size: i64) -> List[tensor[*, f32]] = {\n xs = [produce(5i64), produce(size), produce(7i64)]\n ys = skip(xs, 1i64)\n take(ys, 1i64)\n}\nout = shape(index(selected(3i64), 0i64), 0i32)\n";
    live_metadata_with_sanitizers("returned-prefix-suffix", source, true);
    // The retained selected component still attributes its independent
    // mismatching claim to insert after both view-producing scopes retire.
    let negative = source.replace(
        "out = shape(index(selected(3i64), 0i64), 0i32)",
        "def reject(size: i64) -> tensor[5, f32] = index(selected(size), 0i64)\nout = reject(3i64)",
    );
    for native in [false, true] {
        let (success, output) = result_claims::run(&negative, native);
        assert!(!success, "{output}");
        assert!(
            output.contains("numeric trap: domain in insert at i64"),
            "{output}"
        );
    }
}

#[test]
fn tensor_tuple_and_nested_nominal_construction_stays_bounded() {
    for (kind, ty, seed, body, result) in [
        (
            "tensor",
            "tensor[3, f32]",
            "produce(3i64)",
            "produce(3i64)",
            "shape(b, 0i32)",
        ),
        (
            "tuple",
            "(tensor[3, f32], tensor[3, f32])",
            "(produce(3i64), produce(3i64))",
            "(acc.1, acc.0)",
            "shape(b.0, 0i32)",
        ),
        (
            "nested",
            "Box[Box[tensor[3, f32]]]",
            "Box { v: Box { v: produce(3i64) } }",
            "Box { v: value(acc) }",
            "shape(value(value(b)), 0i32)",
        ),
    ] {
        for claimed in [false, true] {
            let ty = if claimed {
                ty.to_owned()
            } else {
                ty.replace("tensor[3, f32]", "tensor[*, f32]")
            };
            let measured = [0, 100, 400, 1600].map(|steps| {
            let source = format!("type Box[T] =\n | Box {{ v: T }}\ndef value[T](b: Box[T]) -> T = match b with {{ | Box {{ v }} => v }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef run() -> i64 = {{\n b = fold(fn (acc: {ty}, i: i64) -> {body}, {seed}, range(0i64, {steps}i64))\n {result}\n}}\nout = run()\n");
            live_metadata_with_sanitizers(&format!("{kind}-claimed{claimed}-k{steps}"), &source, steps == 1600)
        });
            for later in &measured[2..] {
                assert!(
                    later.peak_origin <= measured[1].peak_origin + 1024,
                    "{kind}: {measured:?}"
                );
                assert!(
                    later.peak_claim <= measured[1].peak_claim + 1024,
                    "{kind}: {measured:?}"
                );
            }
        }
    }
}

#[test]
fn retired_loop_metadata_preserves_empty_next_entry_and_final_invalid_schedule() {
    for native in [false, true] {
        for combine in ["fold", "scan"] {
            for (seed, steps, succeeds, effects) in [
                (3, 0, true, 0),
                (3, 1, true, 1),
                (3, 2, false, 1),
                (5, 0, true, 0),
                (5, 1, false, 0),
            ] {
                let source = format!(
                    "type Box[T] =\n | Box {{ v: T }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef run(seed: Box[tensor[*, f32]], steps: i64) -> i64 ! {{ IO }} = {{\n _ = {combine}(fn (acc: Box[tensor[3, f32]], i: i64) -> {{\n _ = print(\"step\")\n Box {{ v: produce(5i64) }}\n }}, seed, range(0i64, steps))\n 42i64\n}}\nout = run(Box {{ v: produce({seed}i64) }}, {steps}i64)\n"
                );
                let (success, output) = result_claims::run(&source, native);
                assert_eq!(
                    success, succeeds,
                    "native={native} {combine} seed={seed} steps={steps}: {output}"
                );
                assert_eq!(
                    output.lines().filter(|line| *line == "step").count(),
                    effects,
                    "{output}"
                );
                if succeeds {
                    assert!(output.contains("out = 42"), "{output}");
                } else {
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
fn distinct_logical_owners_keep_shared_origin_edges_after_one_payload_drops() {
    let source = "type Box[T] =\n | Box { v: T }\ndef value[T](b: Box[T]) -> T = match b with { | Box { v } => v }\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef aliases(seed: Box[tensor[3, f32]]) -> (Box[tensor[3, f32]], Box[tensor[3, f32]]) = (seed, seed)\ndef run(size: i64) -> i64 = {\n pair = aliases(Box { v: produce(size) })\n a = pair.0\n b = pair.1\n _ = drop(a)\n shape(value(b), 0i32)\n}\nout = run(3i64)\n";
    live_metadata_with_sanitizers("shared-logical-owners", source, true);
    for native in [false, true] {
        let (success, output) = result_claims::run(
            &source.replace("out = run(3i64)", "out = run(5i64)"),
            native,
        );
        assert!(!success, "{output}");
        assert!(
            output.contains("numeric trap: domain in load at i64"),
            "{output}"
        );
    }
}

#[test]
fn initial_list_size_and_completed_iterations_have_separate_storage_bounds() {
    for seed_length in [10, 100] {
        let seed = std::iter::repeat_n("produce(3i64)", seed_length)
            .collect::<Vec<_>>()
            .join(", ");
        let measured = [100, 400].map(|steps| {
            let source = format!("def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef run() -> i64 = {{\n b = fold(fn (acc: List[tensor[3, f32]], i: i64) -> [produce(3i64), produce(3i64), produce(3i64)], [{seed}], range(0i64, {steps}i64))\n len(b)\n}}\nout = run()\n");
            live_metadata(&format!("initial-n{seed_length}-k{steps}"), &source)
        });
        assert_eq!(
            measured[0].peak_origin, measured[1].peak_origin,
            "{measured:?}"
        );
        assert_eq!(
            measured[0].peak_claim, measured[1].peak_claim,
            "{measured:?}"
        );
    }
}

#[test]
fn metadata_identifiers_cannot_shadow_authored_parameters_or_sidecars() {
    for (case, names) in [
        ("suffix", vec!["x", "x_root"]),
        (
            "reserved",
            vec![
                "x",
                "x_root",
                "__origin_scope_0",
                "__chelis_function_origin_scope",
                "__chelis_global_origin_root_0",
                "__chelis_global_origin_root___chelis_result_origin_x",
                "__chelis_global_origin_scope_0",
                "__chelis_global_origin_function_scope",
                "__chelis_host_result_origin_root",
                "__chelis_host_result_origin_scope",
                "__chelis_host_result_origin_assign",
                "__chelis_host_result_origin_scope_enter",
                "__chelis_host_result_origin_scope_leave",
                "__chelis_global_origin_root",
                "__chelis_global_origin_scope",
                "__chelis_global_origin_assign",
                "__chelis_global_origin_retain",
                "__chelis_global_origin_retire",
                "__chelis_global_origin_release",
                "__chelis_global_origin_root_enter",
                "__chelis_global_origin_scope_enter",
                "__chelis_global_origin_scope_leave",
            ],
        ),
    ] {
        let parameters = names
            .iter()
            .map(|name| format!("{name}: Box[tensor[3, f32]]"))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = names
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i == 1 {
                    "Box { v: produce(size) }"
                } else {
                    "Box { v: produce(3i64) }"
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "type Box[T] =\n | Box {{ v: T }}\ndef value[T](b: Box[T]) -> T = match b with {{ | Box {{ v }} => v }}\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef both({parameters}) -> i64 = shape(value(x_root), 0i32)\ndef run(size: i64) -> i64 = both({arguments})\nout = run(3i64)\n"
        );
        let (success, output) = result_claims::run(&source, false);
        assert!(success && output.contains("out = 3"), "{output}");
        live_metadata_with_sanitizers(&format!("identifiers-{case}"), &source, true);
        for native in [false, true] {
            let (success, output) = result_claims::run(
                &source.replace("out = run(3i64)", "out = run(5i64)"),
                native,
            );
            assert!(!success, "{output}");
            assert!(
                output.contains("numeric trap: domain in load at i64"),
                "{output}"
            );
        }
    }
}

#[test]
fn reconstruction_keeps_independent_claim_failure_and_effect_order() {
    // Negative counterpart: a result claim on a freshly produced component
    // remains an insert trap before the later effect, including after reuse.
    for native in [false, true] {
        let source = "type Box[T] =\n | Box { v: T }\ndef produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\ndef step(acc: Box[tensor[3, f32]], i: i64) -> Box[tensor[3, f32]] ! { IO } = {\n _ = print(\"before\")\n b: Box[tensor[3, f32]] = Box { v: produce(if i == 0i64 then 3i64 else 5i64) }\n _ = print(\"after\")\n b\n}\ndef run(seed: Box[tensor[3, f32]]) -> i64 ! { IO } = {\n _ = fold(step, seed, range(0i64, 2i64))\n 42i64\n}\nout = run(Box { v: produce(3i64) })\n";
        let (success, output) = result_claims::run(source, native);
        assert!(!success, "{output}");
        assert!(
            output.contains("numeric trap: domain in insert at i64"),
            "{output}"
        );
        assert_eq!(
            output.lines().filter(|line| *line == "before").count(),
            2,
            "{output}"
        );
        assert_eq!(
            output.lines().filter(|line| *line == "after").count(),
            1,
            "{output}"
        );
    }
}
