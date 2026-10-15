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
    assert!(common::link_generated(&destination, "lifetime.c", "lifetime").success());
    let output = std::process::Command::new(destination.join("lifetime"))
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
        .last()
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
            let measurements = [100, 400, 1600].map(|steps| {
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
