//! Red-team adversarial coverage for the WS-M1 Metal dtype expansion.
//!
//! Focus: silent-corruption / spec-divergence surfaces that the existing
//! tests do not pin. Each test names the spec section it enforces in the
//! assertion message so a failure points at the contract, not the test
//! mechanics.
//!
//! These tests run in default CI on every platform (no Apple SDK or GPU
//! required); they assert on the emitted source-string structure, not on
//! `clang++` accepting it. The real `clang++` compile gate is the
//! `gpu_correctness.rs` manual oracle.

use chelis_backend_metal::codegen_metal;
use chelis_backend_metal::dtype as metal_dtype;
use chelis_backend_metal::kernels;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn vec_prec(n: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: p,
    }
}

fn mat_prec(r: usize, c: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: p,
    }
}

fn cube_prec(a: usize, b: usize, c: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: p,
    }
}

fn build_add_dag(p: Prim) -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_prec(4, p),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_prec(4, p),
        None,
    );
    let s = dag.add_node(RiscOp::Add, vec![a, b], vec_prec(4, p), None);
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![s],
        vec_prec(4, p),
        None,
    );
    dag.add_root(stored);
    dag
}

fn build_matmul_dag(p: Prim) -> Dag {
    let m = 4usize;
    let k = 4usize;
    let n = 4usize;
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_prec(m, k, p),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_prec(k, n, p),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(n),
        },
        vec![a],
        cube_prec(m, k, n, p),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(m),
        },
        vec![b],
        cube_prec(m, k, n, p),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], cube_prec(m, k, n, p), None);
    let acc = metal_dtype::sum_accumulator(p);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: acc,
        },
        vec![mul],
        mat_prec(m, n, p),
        None,
    );
    dag.add_root(sum);
    dag
}

// ===========================================================================
// FINDING #1 (BLOCKER): bf16 matmul accumulator is `bfloat`, spec says `f32`
//
// spec/04-type-system.md §5.7.1: "bf16 operands -> f32 accumulator,
// downcast to operand precision". The tiled MSL kernel currently
// declares `bfloat acc = ...` instead of `float acc = ...`. Long-K
// inner-product summation in bf16-only arithmetic loses bits relative
// to the spec contract on every dispatch.
// ===========================================================================

#[test]
fn bf16_matmul_accumulator_is_f32_per_spec_5_7_1() {
    let kernel_src = kernels::matmul_tiled_kernel_for("k_test_bf16_acc", Prim::Bf16);
    // Per spec/04-type-system.md §5.7.1, bf16 matmul accumulator must be
    // f32. The tiled kernel must declare its `acc` and the threadgroup
    // tile-product accumulator as `float`, not `bfloat`. A `bfloat` acc
    // silently downgrades the bf16 operand path to bf16 inner-product
    // summation, losing precision relative to the spec contract.
    assert!(
        kernel_src.contains("float acc"),
        "spec/04-type-system.md §5.7.1: bf16 matmul accumulator must be f32. \
         Kernel declares operand-precision acc instead:\n{kernel_src}"
    );
    assert!(
        !kernel_src.contains("bfloat acc"),
        "bf16 matmul kernel must not use a bfloat accumulator (§5.7.1 violation):\n{kernel_src}"
    );
}

// ===========================================================================
// FINDING #2 (BLOCKER): host-side .mm uses `sizeof(half)` and `sizeof(bfloat)`
// but `half` and `bfloat` are MSL-only types, not host C++ types.
//
// `half` and `bfloat` are MSL (Metal Shading Language) types defined
// inside `<metal_stdlib>`; the host side of the .mm is plain C++ and
// has no such typedefs. clang++ will fail to compile `sizeof(half)` and
// `sizeof(bfloat)` host-side (verified vs the MPS f16 wrapper which
// uses `sizeof(uint16_t)` for exactly this reason). This means f16 and
// bf16 builds CANNOT compile against any real Apple toolchain.
// ===========================================================================

#[test]
fn f16_emit_does_not_use_msl_only_type_in_host_sizeof() {
    let dag = build_add_dag(Prim::F16);
    let result = codegen_metal(&dag, "f16_add");
    let src = &result.mm_source;
    // Detect host-side `sizeof(half)` outside the embedded MSL kernel
    // raw-string literals. Strip out everything between
    // `@R"MSL(` and `)MSL"` first so we only look at host code.
    let mut host_only = String::new();
    let mut i = 0;
    let bytes = src.as_bytes();
    while i < bytes.len() {
        if let Some(start) = src[i..].find("@R\"MSL(") {
            host_only.push_str(&src[i..i + start]);
            let after_open = i + start + "@R\"MSL(".len();
            if let Some(end) = src[after_open..].find(")MSL\"") {
                i = after_open + end + ")MSL\"".len();
            } else {
                break;
            }
        } else {
            host_only.push_str(&src[i..]);
            break;
        }
    }
    assert!(
        !host_only.contains("sizeof(half)"),
        "f16 host-side .mm uses `sizeof(half)`, but `half` is an MSL-only \
         type not visible to host C++ (clang++ will fail to compile this; \
         see the MPS f16 wrapper which correctly uses `sizeof(uint16_t)`):\n\n{host_only}"
    );
}

#[test]
fn bf16_emit_does_not_use_msl_only_type_in_host_sizeof() {
    let dag = build_add_dag(Prim::Bf16);
    let result = codegen_metal(&dag, "bf16_add");
    let src = &result.mm_source;
    let mut host_only = String::new();
    let mut i = 0;
    let bytes = src.as_bytes();
    while i < bytes.len() {
        if let Some(start) = src[i..].find("@R\"MSL(") {
            host_only.push_str(&src[i..i + start]);
            let after_open = i + start + "@R\"MSL(".len();
            if let Some(end) = src[after_open..].find(")MSL\"") {
                i = after_open + end + ")MSL\"".len();
            } else {
                break;
            }
        } else {
            host_only.push_str(&src[i..]);
            break;
        }
    }
    assert!(
        !host_only.contains("sizeof(bfloat)"),
        "bf16 host-side .mm uses `sizeof(bfloat)`, but `bfloat` is an \
         MSL-only type not visible to host C++ (clang++ will fail to \
         compile this; the f16 MPS wrapper uses `sizeof(uint16_t)` for \
         the same reason):\n\n{host_only}"
    );
}

// ===========================================================================
// FINDING #3 (BLOCKER / SPEC-DIVERGENCE): Runtime header has no Apple7+
// detection for bf16; spec §1.1.3 mandates a clean diagnostic naming
// the GPU family.
//
// spec/04-type-system.md §1.1.3 (bf16 on Metal: requires Apple7+ GPU
// family): "The Metal runtime surfaces this as a clean diagnostic at
// pipeline creation time, not as a silent kernel-load failure. The
// required diagnostic when bf16 pipeline creation fails on a pre-Apple7
// device is: 'bf16 requires Apple7+ GPU family (M3 or later); detected
// device family is Apple{N}.'"
//
// `chelis_metal_runtime.h` has no `MTLGPUFamilyApple7` / `supportsFamily:`
// check anywhere, and `chelis_metal_get_pipeline` aborts with a generic
// "MSL compile failed" / "function not found" message, not the spec-
// pinned diagnostic.
// ===========================================================================

#[test]
fn metal_runtime_header_detects_apple7_for_bf16_per_spec_1_1_3() {
    let header = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(chelis_backend_metal::runtime_dir())
            .join("chelis_metal_runtime.h"),
    )
    .expect("read chelis_metal_runtime.h");
    assert!(
        header.contains("MTLGPUFamilyApple7") || header.contains("supportsFamily"),
        "spec/04-type-system.md §1.1.3 bf16 entry requires the runtime to \
         surface a clean diagnostic at pipeline creation when targeting a \
         pre-Apple7 device. The runtime header has no GPU family detection \
         (`MTLGPUFamilyApple7` / `supportsFamily:`); bf16 pipeline failures \
         currently fall through to the generic `MSL compile failed` abort."
    );
    assert!(
        header.contains("Apple7+ GPU family") || header.contains("bf16 requires Apple7+"),
        "spec/04-type-system.md §1.1.3 pins the diagnostic text \
         'bf16 requires Apple7+ GPU family (M3 or later); detected device \
         family is Apple{{N}}.'; the runtime header does not contain this \
         spec-pinned string."
    );
}

// ===========================================================================
// FINDING #4 (BLOCKER): No MPS f16 wrapper guard against pre-Apple7 GPU
// dispatch.
//
// The MPS f16 wrapper sits inside `#if __METAL_VERSION__ >= 320 ||
// !defined(__METAL_VERSION__)` -- a *compile-time* guard that always
// admits host-side compilation since `__METAL_VERSION__` is undefined
// in host C++. There's no runtime check; the wrapper is callable on
// every device. The comment claims this is "purely a defensive
// belt-and-braces" but a comment is not a runtime check.
//
// The MPS f16 path is documented per spec §1.1.3 as available "on every
// shipping toolchain"; the contract is that f16 admits at pipeline
// creation (different from the bf16 surface). This is OK because the
// MPS f16 path doesn't depend on `bfloat`, but the comment overstates
// the guarantee. Kept as informational; the BLOCKER for f16 is the
// host-side `sizeof(half)`.
// ===========================================================================

// ===========================================================================
// FINDING #5 (SPEC-DIVERGENCE): The IR validation pass is named in the
// spec as one of three required f64-rejection entry points, but it does
// not exist as a separate pass.
//
// spec/04-type-system.md §1.1.3 (f64 on Metal): "This rejection is
// enforced at the CLI gate, at the IR validation pass, and defensively
// at the codegen entry point."
//
// Inventory:
//   - CLI gate: `reject_unsupported_metal_ops` -> exists
//   - codegen entry: `Emitter::require_metal_admissible` -> exists
//   - IR validation pass: NOT found in `chelis-validate` or
//     `chelis-effects::validate_build_target` (which is for device
//     pinning only). The "three surfaces" claim in §1.1.3 only resolves
//     to two surfaces today.
// ===========================================================================

#[test]
fn ir_validation_pass_for_metal_f64_exists_per_spec_1_1_3() {
    // Search the validate / effects / IR crates for ANY function that
    // walks the DAG and rejects f64 on the metal target. If none
    // exists, the spec's "three surfaces" claim is unsupported by
    // shipped code.
    let validate_src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("chelis-validate")
            .join("src")
            .join("lib.rs"),
    )
    .expect("read chelis-validate/src/lib.rs");
    let effects_src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("chelis-effects")
        .join("src");
    let mut effects_combined = String::new();
    if let Ok(entries) = std::fs::read_dir(&effects_src_dir) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                effects_combined.push_str(&content);
            }
        }
    }
    let mentions_f64_metal =
        |s: &str| -> bool { s.contains("FP64") || (s.contains("metal") && s.contains("f64")) };
    let validate_has = mentions_f64_metal(&validate_src);
    let effects_has = mentions_f64_metal(&effects_combined);
    assert!(
        validate_has || effects_has,
        "spec/04-type-system.md §1.1.3 names three f64-rejection entry \
         points (CLI gate + IR validation pass + codegen entry). Only the \
         CLI gate (`reject_unsupported_metal_ops` in chelis-cli) and the \
         codegen entry (`Emitter::require_metal_admissible`) exist; no IR \
         validation pass for f64 on Metal lives in chelis-validate or \
         chelis-effects."
    );
}

// ===========================================================================
// FINDING #6 (NEGATIVE-COVERAGE-GAP): No host-side mapping table from
// MSL precision -> a real C++ type so the emitter can stop hardcoding
// `sizeof(half)` etc. With the host-only test in #2/#3 above, this is
// the structural fix point: the emitter needs a `host_sizeof_expr(prec)`
// helper that returns `"sizeof(uint16_t)"` for half / bfloat, etc.
//
// Smoke check: the `metal_elem_size` helper is the right abstraction
// to drive that emission, and it returns 2 for both f16 and bf16 today.
// We assert that the helper exists and is what the emitter SHOULD be
// calling; the structural fix is to thread it through `emit_load`,
// `emit_const`, etc. instead of `sizeof({msl_ty})`.
// ===========================================================================

#[test]
fn metal_elem_size_provides_host_safe_byte_widths() {
    // This isn't a bug per se -- it's the affordance the emitter
    // ignores. Pinning here so the structural fix in #2/#3 has a
    // documented call target.
    assert_eq!(metal_dtype::metal_elem_size(Prim::F16), 2);
    assert_eq!(metal_dtype::metal_elem_size(Prim::Bf16), 2);
    assert_eq!(metal_dtype::metal_elem_size(Prim::Int8), 1);
    assert_eq!(metal_dtype::metal_elem_size(Prim::Int16), 2);
    assert_eq!(metal_dtype::metal_elem_size(Prim::Int32), 4);
    assert_eq!(metal_dtype::metal_elem_size(Prim::Int64), 8);
    assert_eq!(metal_dtype::metal_elem_size(Prim::Bool), 1);
}

// ===========================================================================
// FINDING #7 (NEGATIVE-COVERAGE-GAP): bf16 matmul kernel does not
// promote operands to f32 before multiplying.
//
// Even after fixing the accumulator type (Finding #1), spec §5.7.1
// requires the *inner product* to compute in the accumulator
// precision. The current kernel does `acc += tileA[...] * tileB[...]`
// where both operands are bfloat; the multiplication produces bfloat
// and is implicitly converted to float. That partial-product loses
// precision before accumulation, which is not what `f32 accumulator`
// means. The correct kernel pattern is `acc += float(tileA[...]) *
// float(tileB[...])`.
// ===========================================================================

#[test]
fn bf16_matmul_promotes_operands_to_f32_before_multiply_per_spec_5_7_1() {
    let kernel_src = kernels::matmul_tiled_kernel_for("k_test_bf16_promo", Prim::Bf16);
    // The inner-product line should explicitly promote operands. Look
    // for any of the canonical promotion forms.
    let has_explicit_promotion = kernel_src.contains("(float)tileA")
        || kernel_src.contains("float(tileA")
        || kernel_src.contains("(float)(tileA");
    assert!(
        has_explicit_promotion,
        "spec/04-type-system.md §5.7.1: bf16 matmul partial products must \
         be computed in the f32 accumulator precision. The kernel \
         multiplies bfloat operands and only widens the *result*; this \
         loses precision in the partial product before accumulation. \
         Kernel:\n{kernel_src}"
    );
}

// ===========================================================================
// FINDING #8 (BLOCKER): tiled-matmul tile-load fallback uses operand
// identity, not accumulator identity, and fallback cast targets the
// operand type when accumulator should differ.
//
// Even the spec-compliant fix to f32 accumulator wouldn't resolve the
// tile-load fallback line:
//     tileA[...] = (... in-bounds) ? A[...] : ({ty}){identity};
// Where `{ty}` is the operand type (e.g., `bfloat`) and `{identity}`
// resolves at *operand* precision. After the §5.7.1 fix to acc:f32,
// the tile-load identity should still match the tile element type
// (which is operand precision), so the fallback line itself is fine.
// Pinning the test to confirm the tile element type stays at operand
// precision after the fix; flagging because the kernel template
// confused `acc` and `tileA[lid.y][lid.x]` types.
// ===========================================================================

#[test]
fn bf16_matmul_tile_storage_stays_at_operand_precision() {
    let kernel_src = kernels::matmul_tiled_kernel_for("k_test_bf16_tile", Prim::Bf16);
    // Tiles cache operand bytes; their type must be the operand type so
    // the threadgroup memory footprint matches the buffer footprint.
    // The accumulator promotion happens at the multiply, not in the
    // tile cache. This pins the operand-precision tile shape.
    assert!(
        kernel_src.contains("threadgroup bfloat tileA")
            && kernel_src.contains("threadgroup bfloat tileB"),
        "bf16 matmul tile storage should remain at operand (bfloat) \
         precision; promotion is at the multiply per §5.7.1:\n{kernel_src}"
    );
}

// ===========================================================================
// FINDING #9 (NEGATIVE-COVERAGE-GAP): Existing dtype_matrix tests don't
// have the negative twin for "f64 reaches msl_type from a non-CLI path
// without panicking". Pin it.
// ===========================================================================

#[test]
#[should_panic(expected = "Metal backend rejects f64")]
fn metal_elem_size_panics_on_f64() {
    let _ = metal_dtype::metal_elem_size(Prim::F64);
}

#[test]
#[should_panic(expected = "Metal backend rejects f64")]
fn runtime_dtype_tag_panics_on_f64() {
    let _ = metal_dtype::runtime_dtype_tag(Prim::F64);
}

#[test]
#[should_panic(expected = "Metal backend rejects f64")]
fn kernel_suffix_panics_on_f64() {
    let _ = metal_dtype::kernel_suffix(Prim::F64);
}

// ===========================================================================
// FINDING #10 (NEGATIVE-COVERAGE-GAP): Mixed-precision MPS dispatch is
// not pinned -- a regression that called `chelis_metal_mps_gemm_f32`
// with f16 buffers would silently corrupt. The dispatch site picks the
// helper from `info.precision`, but no test pins that f16 cannot land
// in the f32 helper.
// ===========================================================================

#[test]
fn matmul_dispatch_helper_matches_operand_precision() {
    // f32 -> f32 helper, f16 -> f16 helper, bf16 -> tiled kernel.
    // Pin each so a future refactor that swaps the dispatch table
    // surfaces here.
    let f32_src = codegen_metal(&build_matmul_dag(Prim::F32), "mm_f32").mm_source;
    assert!(f32_src.contains("chelis_metal_mps_gemm_f32("));
    assert!(!f32_src.contains("chelis_metal_mps_gemm_f16("));

    let f16_src = codegen_metal(&build_matmul_dag(Prim::F16), "mm_f16").mm_source;
    assert!(f16_src.contains("chelis_metal_mps_gemm_f16("));
    assert!(!f16_src.contains("chelis_metal_mps_gemm_f32("));

    let bf16_src = codegen_metal(&build_matmul_dag(Prim::Bf16), "mm_bf16").mm_source;
    assert!(!bf16_src.contains("chelis_metal_mps_gemm_"));
    assert!(bf16_src.contains("kernel void k_matmul_bf16_"));
}

// ===========================================================================
// FINDING #11 (NEGATIVE-COVERAGE-GAP): The MPS f16 host-side wrapper
// uses `sizeof(uint16_t)` for the rowBytes computation -- but on a
// system where `_Float16` differs in size from `uint16_t` (none today
// on Apple Silicon, but a regression worth pinning) the contract would
// silently drift. Lock the MPS wrappers to the byte-width contract.
// ===========================================================================

#[test]
fn mps_f16_wrapper_uses_uint16_t_for_row_bytes_not_msl_half() {
    let header = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(chelis_backend_metal::runtime_dir())
            .join("chelis_metal_runtime.h"),
    )
    .expect("read chelis_metal_runtime.h");
    // Inside `chelis_metal_mps_gemm_f16`, row-bytes calc must use a
    // host-visible 2-byte type (`uint16_t`) since `half` is MSL-only.
    // Skip past the comment block that mentions the function name; the
    // actual function definition starts at `static inline void
    // chelis_metal_mps_gemm_f16(`.
    let f16_helper_start = header
        .find("static inline void chelis_metal_mps_gemm_f16")
        .expect("MPS f16 helper definition missing");
    let f16_helper_end = header[f16_helper_start..]
        .find("\n}\n")
        .expect("MPS f16 helper unterminated");
    let f16_helper = &header[f16_helper_start..f16_helper_start + f16_helper_end];
    assert!(
        f16_helper.contains("sizeof(uint16_t)"),
        "chelis_metal_mps_gemm_f16 must use sizeof(uint16_t) for row \
         bytes (host has no `half` type):\n{f16_helper}"
    );
    assert!(
        !f16_helper.contains("sizeof(half)"),
        "chelis_metal_mps_gemm_f16 must NOT reference MSL-only `half` \
         type host-side:\n{f16_helper}"
    );
}

// ===========================================================================
// FINDING #12 (NEGATIVE-COVERAGE-GAP): No proof that the new dtype
// matmul codegen does NOT silently emit the integer matmul kernel.
//
// dtype_matrix.rs already pins `matmul_int_rejected_at_codegen_falls
// _through_to_stub`, but does not pin that the *emitted source* contains
// the F1 diagnostic in the abort message; otherwise an int matmul that
// silently emits the stub gives the user no clue why it didn't run.
// ===========================================================================

#[test]
fn integer_matmul_stub_carries_meaningful_abort_message() {
    let dag = build_matmul_dag(Prim::Int32);
    let result = codegen_metal(&dag, "mm_i32");
    let src = &result.mm_source;
    // Either the stub is reached AND it carries the func name, OR the
    // emit_dag is expected to surface a structured F1 error to the
    // caller (which the CLI translates to a diagnostic). Today only the
    // first lands. Without a meaningful error string in the stub, the
    // user sees the bare "stub: codegen for `mm_i32` not yet implemented"
    // and has to dig through spec/04-type-system.md to find why an int
    // matmul didn't compile.
    let in_stub = src.contains("M1 fallback stub");
    let mentions_int_or_f1 = src.contains("int") || src.contains("§5.7.2") || src.contains("F1");
    assert!(
        in_stub,
        "integer matmul must fall through to stub (defense in depth):\n{src}"
    );
    assert!(
        mentions_int_or_f1,
        "integer matmul stub gives no hint why; should reference spec \
         §5.7.2 or the F1 guard so users see why the build is empty:\n{src}"
    );
}

// ===========================================================================
// FINDING #13 (NEGATIVE-COVERAGE-GAP): Apple-Silicon `long` is 64 bit
// but a Windows / Linux clang++ port (LLP64 / LP64) would surface
// `sizeof(long) != 8` if the Metal backend ever ran cross-platform.
//
// Today Metal only runs on Apple Silicon so `sizeof(long) == 8`
// matches CHELIS_I64. Pin the assumption in a test so a future
// cross-platform attempt surfaces the divergence.
// ===========================================================================

#[test]
fn int64_emit_uses_msl_long_with_apple_silicon_assumption() {
    let dag = build_add_dag(Prim::Int64);
    let result = codegen_metal(&dag, "i64_add");
    let src = &result.mm_source;
    assert!(
        src.contains("device const long* a") && src.contains("device long* out"),
        "int64 add kernel must use MSL `long` for buffers:\n{src}"
    );
    // sizeof(long) on macOS LP64 = 8, matches int64 width. Document
    // the platform assumption.
    assert_eq!(
        std::mem::size_of::<i64>(),
        8,
        "i64 width invariant changed; Metal backend assumes 8-byte long"
    );
}

// ===========================================================================
// FINDING #14 (NEGATIVE-COVERAGE-GAP): no test pins that the bf16
// matmul kernel's accumulator-precision change shows up in BOTH the
// `acc` declaration AND the `acc +=` accumulator update with explicit
// promotion. Pin the dual.
// ===========================================================================

#[test]
fn bf16_matmul_accumulator_update_promotes_both_acc_and_partial() {
    let kernel_src = kernels::matmul_tiled_kernel_for("k_test_bf16_dual", Prim::Bf16);
    // Required after the §5.7.1 fix:
    //   1. `float acc = 0.0f` (acc declared at f32)
    //   2. accumulator update uses explicit promotion of partial
    let acc_at_f32 = kernel_src.contains("float acc = 0.0f") || kernel_src.contains("float acc =");
    let promoted_partial =
        kernel_src.contains("(float)tileA") || kernel_src.contains("float(tileA");
    assert!(
        acc_at_f32 && promoted_partial,
        "bf16 matmul kernel must declare `float acc = ...` AND promote \
         `tileA * tileB` partial product to float before `+=` per spec \
         §5.7.1. Current kernel:\n{kernel_src}"
    );
}
