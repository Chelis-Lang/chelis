//! chelis#687 - the rejected-cells corpus, seeded by chelis#730 Phase 0
//! (spec/design/loud_unsupported.md section C2 / Phase 0 item 3) and moved
//! onto chelis#732's shared exact comparator in Phase 3.
//!
//! One table-driven file collecting the EXISTING loud-failure locks - the
//! HIP narrow-float rejection, the Metal f64 rejection, the Metal rank-2
//! abort stub, and the runtime aborts - so the Phase 1 message migration
//! to the section C2 `unsupported:` format has a single file to update.
//!
//! The strings asserted here mirror (never replace) their original locks;
//! per B2.1 the originals keep their expected text until the Phase 1
//! freeze-point change, which must update both in the same change set:
//!
//! - HIP: `narrow_dtype_matrix.rs::hip_rejects_f16_bf16_compute_ops_cleanly`
//! - Metal: `metal_dtype_emission_and_bool_add.rs::
//!   metal_rejects_f64_with_a_specific_diagnostic` and
//!   `::metal_rank2_fallback_is_a_named_abort_stub`
//! - C-host reduced-float selection: typed build rejection before the
//!   former widened scalar path can reach runtime
//! - runtime int-div guard: `ws2b_numeric_identifier_divergence.rs`'s
//!   `INT_DIV_ZERO_DIAGNOSTIC` rows (chelis#387 family)
//!
//! Each current diagnostic record is compared byte-for-byte through the same
//! comparator entrypoint as the value corpus. Emitted fallback source keeps
//! exact marker locks because it is an artifact rather than one diagnostic
//! record. The typed full-record diagnostic shape remains chelis#730 Phase 3
//! work; exact text does not pretend to prove that pending structure.
//!
//! The chelis#730 Phase 3 authority migration revises the human rendering so
//! the validated deliberate/unimplemented kind and citation are visible.
//! Existing substring locks continue to cover the stable diagnostic payload;
//! `rejection_authority.rs` byte-checks the new authority-bearing grammar.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use chelis_types::agreement::compare_exact_observations;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn assert_emission_fragment(observation: &str, expected: &str, context: &str) {
    let candidate = observation
        .find(expected)
        .map(|start| &observation[start..start + expected.len()])
        .unwrap_or("");
    compare_exact_observations(context, expected, candidate)
        .unwrap_or_else(|error| panic!("{context}: {error}; complete observation: {observation}"));
}

/// `chelis build` to `target`; (ok, stderr, concatenated emitted files).
fn build_target(program: &str, name: &str, target: &str) -> (bool, String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            target,
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    let mut emitted = String::new();
    if out_dir.is_dir() {
        for entry in std::fs::read_dir(&out_dir).expect("read out dir") {
            let p = entry.expect("entry").path();
            if p.is_file()
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                emitted.push_str(&text);
            }
        }
    }
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        emitted,
    )
}

/// Build to C, link, run; (run_exit_ok, stdout, stderr). Panics on build or
/// link failure - every runtime row in this corpus BUILDS and aborts at run
/// time (that is what makes it the runtime abort row).
fn c_run(program: &str, name: &str) -> (bool, String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link failed for {name}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    (
        run.status.success(),
        String::from_utf8_lossy(&run.stdout).into_owned(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
    )
}

// ===========================================================================
// The build-lane rejection rows (emission-only; no toolchain needed).
// ===========================================================================

/// (name, program, target, exact stderr rejection).
const BUILD_REJECTION_ROWS: &[(&str, &str, &str, &str)] = &[
    (
        "hip_f16_compute",
        "def f(a: tensor[4, f16], b: tensor[4, f16]) -> tensor[4, f16] = add(a, b)\n",
        "hip",
        "error: unsupported: `chelis build --target hip` admits `f16` only on tensor load/store \
         nodes and on `BlasMatmul` operands today (`hipblasGemmEx` with an f32 accumulator, \
         WS-A3). Node 2 carries op Add which has no bf16/f16 kernel template yet \
         (chelis-backend-hip emit::dtype_kernel_suffix). See spec/04-type-system.md §5.7.1.\n",
    ),
    (
        "hip_bf16_compute",
        "def f(a: tensor[4, bf16], b: tensor[4, bf16]) -> tensor[4, bf16] = add(a, b)\n",
        "hip",
        "error: unsupported: `chelis build --target hip` admits `bf16` only on tensor load/store \
         nodes and on `BlasMatmul` operands today (`hipblasGemmEx` with an f32 accumulator, \
         WS-A3). Node 2 carries op Add which has no bf16/f16 kernel template yet \
         (chelis-backend-hip emit::dtype_kernel_suffix). See spec/04-type-system.md §5.7.1.\n",
    ),
    (
        "metal_f64",
        "def f(a: tensor[4, f64], b: tensor[4, f64]) -> tensor[4, f64] = add(a, b)\n",
        "metal",
        "error: unsupported: `chelis build --target metal` rejects f64 (node 0): Apple Silicon \
         GPUs lack FP64 ALUs; use `--target c` or `--target hip` for f64 workloads. See \
         spec/04-type-system.md §1.1.3.\n",
    ),
    // -- chelis#730 Phase 1 rows: the converted census sites, each pinned
    // to the branded section C2 rendering. --------------------------------
    (
        "c_stub_tensor_scan",
        "def gen() -> tensor[5, f32] = \
         tensor_scan(0.0, fn (prev: f32, i: int64) -> add(prev, 1.0), cast(5, int64))\n\
         out = gen()\n",
        "c",
        "error: unsupported: builtin `tensor_scan` on `chelis build --target c` host emission \
         (codegen:c); deliberate [05-HOST-1]: host-runtime builders are intentionally excluded \
         from compiled targets; run under `chelis eval` or `chelis test`, or rewrite the caller \
         to use tensor-lane primitives\n",
    ),
    (
        "c_stub_scalar_floor",
        "def f(x: f32) -> f32 = floor(x)\nout = f(3.5)\n",
        "c",
        "error: unsupported: builtin `floor` on `chelis build` host emission (codegen:c); \
         deliberate [04-TOT-2]: the checked builtin vocabulary and C expression vocabulary \
         disagree; no fallback expression is permitted\n",
    ),
    (
        "c_to_string_tensor",
        "def f(x: tensor[2, f32]) -> string = to_string(x)\n\
         out = f(to_tensor([1.5, 2.5]))\n",
        "c",
        "error: unsupported: `to_string` of a `Tensor(TensorType { dims: [Lit(2)], precision: \
         F32 })`-typed value on `chelis build` host emission (codegen:c); unimplemented \
         chelis#1059: the compiled lane stringifies int64/f32/f64/bool/string scalars only \
         today; chelis#1059 owns compiled tensor/list rendering (the former `<value>` \
         placeholder is chelis#734)\n",
    ),
    (
        "c_int64_max_reduce",
        "def f(x: tensor[4, int64]) -> tensor[int64] = max_reduce(x, 0)\n\
         out = f(to_tensor([cast(1, int64), cast(4, int64), cast(2, int64), \
         cast(3, int64)]))\n",
        "c",
        "error: unsupported: op `max_reduce` on `int64` tensors in the C DAG emitter (node 1) \
         (codegen:c); unimplemented chelis#729: the C reduce kernels are f32-hardcoded today \
         (WS-A1/F1); cast to f32 before the reduction. The target capability table owns non-f32 \
         widening\n",
    ),
    (
        "c_int_tensor_cos",
        "def run(x: tensor[4, int32]) -> tensor[4, int32] = cos(x)\n\
         out = run(to_tensor([cast(1, int32), cast(2, int32), cast(3, int32), \
         cast(4, int32)]))\n",
        "c",
        "error: unsupported: builtin `cos` on tensor operands in `chelis build` host emission \
         (no tensor emission arm for this op) (codegen:c); deliberate [04-TOT-2]: a checked \
         tensor operation must route through the typed DAG lane; the C host scalar lane has no \
         fallback tensor expression\n",
    ),
    (
        "c_nonliteral_window",
        "def f(x: tensor[6, f32], w: int32, s: int32) -> tensor[5, f32] = \
         reduce_window_max(x, [w], [s])\n\
         out = f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0]), 2, 1)\n",
        "c",
        "error: Lowering error: unsupported: a non-literal window list for `reduce_window_max` \
         on the compiled-backend lowering of `reduce_window_*` (lowering); unimplemented \
         chelis#1058: window and stride lists must be integer literals for the compiled lane \
         today; a runtime-parameterized window previously lowered to a silent no-op; \
         chelis#1058 owns compiled runtime-list support at source span `surf:86..89`\n",
    ),
    (
        "c_to_tensor_narrow_dtype",
        "def f() -> tensor[2, f16] = to_tensor([cast(2049.0, f16), cast(0.75, f16)])\n\
         out = print(f())\n",
        "c",
        "error: unsupported: dtype `f16` on C host ABI selection (codegen:c); unimplemented \
         chelis#714: the C-host scalar ABI has no exact representation for this dtype; no \
         alternate dtype is permitted\n",
    ),
    (
        "hip_int64_neg",
        "def f(x: tensor[4, int64]) -> tensor[4, int64] = neg(x)\n",
        "hip",
        "error: unsupported: dtype `int64` on a HIP kernel family with f32/f64 variants only \
         (codegen:hip); unimplemented chelis#689: this op has no typed HIP kernel for the \
         operand dtype; the former silent F32 fallback emitted a corrupting kernel \
         (chelis#689). Cast to f32/f64, or use the ops with typed templates (add/mul/div and \
         the i8/i16 promoted sum)\n",
    ),
];

/// Every build-lane rejected cell fails the build and carries its pinned
/// diagnostic text.
#[test]
fn rejected_cells_fail_the_build_with_their_pinned_diagnostics() {
    for (name, program, target, expected) in BUILD_REJECTION_ROWS {
        let (ok, stderr, _) = build_target(program, name, target);
        assert!(!ok, "{name}: the build must be rejected");
        compare_exact_observations(&format!("{name} build rejection"), expected, &stderr)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

/// The Metal rank-2 fallback is a SELF-NAMING abort stub in the emitted
/// source - the loud fallback shape section C1 asks for. Build succeeds;
/// the loudness lives in the emission.
#[test]
fn metal_rank2_abort_stub_names_itself_in_the_emission() {
    let (ok, stderr, emitted) = build_target(
        "def f(a: tensor[2, 2, f32], b: tensor[2, 2, f32]) -> tensor[2, 2, f32] = add(a, b)\n",
        "metal_rank2_corpus",
        "metal",
    );
    assert!(
        ok,
        "rank-2 metal must build (the stub is loud, not fatal): {stderr}"
    );
    for expect in ["fallback stub", "abort()"] {
        assert_emission_fragment(&emitted, expect, "metal rank-2 fallback emission fragment");
    }
}

// ===========================================================================
// The runtime abort rows (need a host C toolchain; each row BUILDS and
// aborts at run time with its pinned message and a nonzero exit).
// ===========================================================================

/// (name, program, exact stderr of the abort).
const RUNTIME_ABORT_ROWS: &[(&str, &str, &str)] = &[(
    // chelis#387 family: the portable integer div-by-zero guard, with a
    // runtime-computed divisor so nothing constant-folds it away.
    "int_div_by_zero",
    "def d(x: int64, y: int64, z: int64) -> int64 = trunc_div(x, sub(y, z))\n\
         out = d(cast(7, int64), cast(5, int64), cast(5, int64))\n",
    "integer division or remainder by zero\n",
)];

/// Every runtime rejected cell aborts (nonzero exit) with its pinned
/// message on stderr, and never prints a result value.
#[test]
fn runtime_rejected_cells_abort_with_their_pinned_diagnostics() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (name, program, expect) in RUNTIME_ABORT_ROWS {
        let (ran_ok, stdout, stderr) = c_run(program, name);
        assert!(
            !ran_ok,
            "{name}: the compiled binary must abort, not exit 0; stdout: {stdout}"
        );
        compare_exact_observations(&format!("{name} runtime abort"), expect, &stderr)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        // The nonzero exit + branded stderr above carry this test. (A
        // previous `!stdout.contains("out =")` guard was vacuous:
        // compiled binaries print bare values, never an `out =` prefix -
        // PR #746 review.)
    }
}
