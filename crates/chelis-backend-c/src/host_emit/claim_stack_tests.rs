//! Execute the emitted result-claim helpers on a long invocation chain.

use super::append_host_result_claim_checks;
use std::fs;
use std::process::Command;

const PREFIX: &str = r#"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct __chelis_host_result_axis {
    int64_t axis;
    int64_t required;
    const char *claim;
    const char *source;
    int64_t source_axis;
} __chelis_host_result_axis;

typedef struct __chelis_host_result_claim {
    const struct __chelis_host_result_claim *next;
    int64_t rank;
    int64_t count;
    const __chelis_host_result_axis *axes;
    int outer_claims_first;
} __chelis_host_result_claim;

typedef struct chelis_tensor { volatile int64_t rank, extent; } chelis_tensor;

__attribute__((noinline)) static int64_t chelis_tensor_rank(const chelis_tensor *value) {
    return value->rank;
}
__attribute__((noinline)) static int64_t chelis_tensor_shape(const chelis_tensor *value, int64_t axis) {
    if (axis != 0) abort();
    return value->extent;
}
static void chelis_numeric_trap(const char *message) {
    fprintf(stderr, "trap: %s\n", message);
    exit(91);
}
"#;

const SUFFIX: &str = r#"
int main(int argc, char **argv) {
    if (argc != 2) return 97;
    int mode = atoi(argv[1]);
    chelis_tensor value = { 1, 1 };
    int64_t observations[1][3] = {{ 0, 0, 1 }};
    if (mode == 5) {
        const size_t count = 1000000;
        __chelis_host_result_claim *claims = calloc(count, sizeof(*claims));
        if (claims == NULL) return 98;
        const __chelis_host_result_axis axis = { 0, 1, "passing", "source", 0 };
        for (size_t i = 0; i < count; ++i) {
            claims[i].next = i + 1 < count ? &claims[i + 1] : NULL;
            claims[i].rank = 1;
            claims[i].count = 1;
            claims[i].axes = &axis;
            claims[i].outer_claims_first = 1;
        }
        __chelis_check_host_result_claims(claims, &value, "probe", "mismatch");
        __chelis_check_host_result_extent_claims(claims, 1, observations, 1, "probe", "mismatch");
        free(claims);
        return 0;
    }
    if (mode < 1 || mode > 7) return 99;
    __chelis_host_result_axis axes[4] = {
        { 0, 3, "A", "source", 0 },
        { 0, mode == 1 || mode == 3 ? 2 : 1, "B", "source", 0 },
        { 0, 4, "C", "source", 0 },
        { 0, mode == 1 || mode == 3 || mode == 6 || mode == 7 ? 5 : 1, "D", "source", 0 },
    };
    __chelis_host_result_claim claims[4] = {0};
    for (int i = 0; i < 4; ++i) {
        claims[i].next = i + 1 < 4 ? &claims[i + 1] : NULL;
        claims[i].rank = 1;
        claims[i].count = 1;
        claims[i].axes = &axes[i];
        claims[i].outer_claims_first = i == 0 || i == 2;
    }
    if (mode <= 2 || mode == 6) {
        __chelis_check_host_result_claims(claims, &value, "probe", "mismatch");
    } else {
        __chelis_check_host_result_extent_claims(claims, 1, observations, 1, "probe", "mismatch");
    }
    return 96;
}
"#;

#[test]
fn result_claim_helpers_preserve_order_and_do_not_recurse_through_claim_chains() {
    let mut helpers = Vec::new();
    append_host_result_claim_checks(&mut helpers);
    let source = format!("{PREFIX}\n{}\n{SUFFIX}", helpers.join("\n"));
    let probe = tempfile::tempdir().expect("probe directory");
    let source_path = probe.path().join("claim_check.c");
    let binary_path = probe.path().join("claim_check");
    fs::write(&source_path, source).expect("write emitted helper probe");
    let toolchain = crate::toolchain::test_toolchain(Default::default());
    let compile = Command::new(&toolchain.compiler)
        .args(&toolchain.compile_flags)
        .arg(&source_path)
        .args(&toolchain.link_flags)
        .arg("-o")
        .arg(&binary_path)
        .output()
        .expect("compile emitted helper probe");
    assert!(
        compile.status.success(),
        "C probe compilation failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    for (mode, first_claim) in [(1, "B"), (2, "C"), (3, "B"), (4, "C"), (6, "D"), (7, "D")] {
        let output = Command::new(&binary_path)
            .arg(mode.to_string())
            .output()
            .expect("run ordered claim probe");
        assert_eq!(output.status.code(), Some(91), "mode {mode}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("extent `{first_claim}`")),
            "mode {mode}: unexpected first claim: {stderr}"
        );
    }

    let long = Command::new(&binary_path)
        .arg("5")
        .output()
        .expect("run long claim chain");
    assert!(
        long.status.success(),
        "a long claim chain must complete without stack exhaustion: {long:?}"
    );
}
