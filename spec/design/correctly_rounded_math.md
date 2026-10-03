# Correctly rounded transcendentals: one kernel set for every lane

This document describes how Chelis implements [05-OP-46]'s rule that `exp`, `log`,
`sin`, `cos`, `tan`, `atan`, `tanh`, and `sqrt` are correctly rounded at [04-NUM-8]'s
arithmetic width, together with the build-profile and floating-point-environment rules
of `spec/08-backends.md` §7 that keep the rest of the arithmetic from varying by host.
The numbered spec decides the semantics; this document decides the implementation and
its sequence. The evidence and the prior-art survey behind the decision are in
`docs/investigations/borrowed_numerics_2957.md`; the tracking issue is
[chelis#2957](https://github.com/Chelis-Lang/chelis/issues/2957).

## 1. Problem

Before this design, Chelis took its transcendentals from outside the compiler: Rust
`std` (`f32::exp` and friends, documented as having unspecified precision that may vary
even within one execution) in the evaluator, the host C library's `expf`/`exp` in
emitted C, Apple Accelerate vForce or Sleef for vectorized tensor kernels on some
hosts, the C compiler's constant folder when an argument was visible at compile time,
and whatever floating-point environment the process inherited. The consequences were
measured in the sub-issues of #2957: up to 32 ULP between `chelis eval` and
`chelis build` for `tanh` at f32, three different values from one built program for one
input depending on what the optimizer could see, and different bits on different
machines from one compiler build.

An N-ULP tolerance cannot repair this. A compound such as `softmax`, `gelu`, or
`normal_cdf` amplifies leaf error without a safe finite bound, and two libraries that
are each within 1 ULP of the truth can still differ from each other on every
platform-version pair. The only definition of `exp` that admits no lane choice is the
correctly rounded one: the exact real value rounded once. If every lane computes that,
the algorithm, library, and compiler that produced it stop mattering, because they all
must produce the same bits.

## 2. Rejected alternatives

- **Widen the [05-OBS-3] tolerances and count libm and the C compiler as part of the
  target.** Gives up cross-machine determinism and has no safe bound for compounds.
- **Fix instances** (pin `normal_cdf`'s `exp`, drop vForce, clean flags). Rust `std`
  and glibc stay unspecified and version-dependent, so the class stays open.
- **Pinned graphs over `+ - * /`, `sqrt`, and FMA as the semantics** (the 2026-08-25
  direction in #1311). The graph's coefficients would become language semantics, so any
  later accuracy improvement would be a breaking change, `chelis prove` would need an
  error envelope per graph, and standard range reduction needs exponent manipulation and
  tables beyond the named primitive set.
- **An approximate tier now.** A separately named, explicitly approximate family (for
  example `exp_approx`) is the likely route for GPU special-function units, but it is
  not specified: adding a new name later breaks no program, and no released lane needs
  it.

## 3. Kernel source

### 3.1 Choice: CORE-MATH, vendored

The kernels come from the [CORE-MATH project](https://core-math.gitlabpages.inria.fr/)
(INRIA), MIT licence, source at <https://gitlab.inria.fr/core-math/core-math>. Every
function the rule names exists at both widths, each a standalone C file defining
`cr_<name>`:

| function | binary32 | binary64 | binary64 worst-case corpus shipped |
|---|---|---|---|
| `exp` | `binary32/exp/expf.c` | `binary64/exp/exp.c` | `exp.wc` |
| `log` | `binary32/log/logf.c` | `binary64/log/log.c` (+ `dint.h`) | `log.wc` |
| `sin` | `binary32/sin/sinf.c` | `binary64/sin/sin.c` | `sin.wc` |
| `cos` | `binary32/cos/cosf.c` | `binary64/cos/cos.c` | `cos.wc` |
| `tan` | `binary32/tan/tanf.c` | `binary64/tan/tan.c` | `tan.wc` |
| `atan` | `binary32/atan/atanf.c` | `binary64/atan/atan.c` | `atan.wc` |
| `tanh` | `binary32/tanh/tanhf.c` | `binary64/tanh/tanh.c` | `tanh.wc` |

`sqrt` needs no kernel: IEEE 754 requires the hardware square root to be correctly
rounded, and the C lanes already emit it under the strict profile.

LLVM libc is the alternative considered. Its documentation marks the binary32 forms and
binary64 `exp`, `log`, `sin`, `cos`, and `tan` correctly rounded in all modes, but
records binary64 `atan` as "1 ULP" and has no binary64 `tanh`. Its implementations are
C++ over its own support headers, which the C backend would have to emit as C++ or link
prebuilt. CORE-MATH covers the full matrix in plain C, so it is the single source; no
width needs a second library or the [05-OP-46] parenthetical for a missing kernel.

### 3.2 Portability facts and the constraints they impose

Read from the upstream sources:

- **Language.** C99/C11 with GNU extensions: `__builtin_fma`, `__builtin_fmaf`,
  `__builtin_roundeven`, `__builtin_expect`, and `unsigned __int128` (binary32 `sin`,
  `cos`, `tan`; binary64 `sin`, `cos`, `tan`). GCC and Clang compile them; MSVC does not
  (no `__int128`). Chelis's C lanes target GCC and Clang only, on 64-bit hosts, so this
  is within the supported set. A Windows/MSVC target would need its own decision.
- **Architecture paths.** binary64 `exp` and `tanh` use SSE intrinsics under
  `#if defined(__x86_64__)` with portable fallbacks; several files use inline
  assembly for `roundeven` only when the builtin is unavailable. Both arms must be
  covered by the oracle on their architectures (§8).
- **FMA.** The kernels call `__builtin_fma`. Where the target has no hardware FMA in its
  baseline (x86-64 without `-mfma`, which the strict profile forbids adding from the
  host CPU), the compiler emits a call to the C library's `fma`, which C17 7.12.13.1
  requires to round once, so values are unaffected; the cost is speed (open question 2).
- **Rounding mode.** The kernels read the dynamic rounding mode (`fegetround`) in a few
  binary64 paths and otherwise assume arithmetic rounds in the current mode. Chelis
  pins round-to-nearest-even at every entry point (§6), so `-frounding-math` is not
  needed.
- **Evaluation method.** The code assumes `FLT_EVAL_METHOD == 0` (no x87 extended
  precision). The vendored wrapper rejects any other value with `#error`.
- **Internal precision.** Most binary32 kernels evaluate in binary64 internally and
  round once. [04-NUM-8] defines a correctly rounded primitive by its result value,
  so that internal precision is not an operation width; the prohibition on computing
  at f64 and narrowing still applies to every operation of a graph.
- **State.** No thread-local or global mutable state; tables are `static const`.
  `errno` and inexact-flag support are off by default and stay off.
- **NaN results.** Kernels return `x + x` or similar for a NaN operand, which propagates
  payloads. The Chelis wrapper canonicalizes every NaN result to [04-NUM-2]'s pattern.
- **Name collisions.** The files cannot be concatenated into one translation unit: they
  redefine the same file-scope typedefs, tables, and helpers (`b32u32_u`, `b64u64_u`,
  `tb`, `rbig`, and others; checked by compiling the concatenation).

A sampled probe on macOS arm64 with Apple clang 21, `-std=c11 -O2 -ffp-contract=off
-fno-fast-math`, compiled all fourteen files without warnings and matched mpmath's
correctly rounded values on 31,505 inputs (3,000 random finite binary32 and 1,500 random
finite binary64 per function, plus `tanh` witnesses near zero). That is a smoke test,
not the oracle of §8.

### 3.3 Vendoring

- `crates/chelis-crmath/vendor/core-math/` holds the fourteen upstream files and their
  support headers unmodified, the MIT `LICENSE`, and a `VENDOR.toml` naming the upstream
  commit and each file's SHA-256.
- `scripts/vendor_core_math.py` (Python with tests and a `[[path_rule]]`) generates one
  checked-in amalgamation, `crmath_amalgamation.c`. For each kernel it:
  1. prefixes every file-scope identifier with a per-kernel namespace
     (`chelis_cr_expf__`), using the identifier list from a Clang AST dump taken at
     vendoring time, so kernels coexist in one translation unit;
  2. gives every definition `static` linkage;
  3. adds a `static` entry `chelis_cr_<name>` that calls the kernel and canonicalizes a
     NaN result.
  The amalgamation begins with guards: `#error` when `__FAST_MATH__`,
  `__FINITE_MATH_ONLY__`, or `FLT_EVAL_METHOD != 0` is in effect. `--check` regenerates
  in memory and fails on any byte difference, so the vendored inputs and the
  amalgamation cannot drift.
- The amalgamation is the only kernel text in the repository. Every lane compiles these
  exact bytes.

## 4. Lane wiring

### 4.1 Evaluator, constant folding, prove, and bindings

`chelis-crmath` is a leaf workspace crate whose `build.rs` compiles the amalgamation
with the `cc` crate (already a workspace build dependency through
`tree-sitter-chelis`) under explicit `-std=c11 -O2 -ffp-contract=off -fno-fast-math`
flags, placed after any ambient flags so they win, and with the guards above failing the
build if ambient flags still enable fast math. A small companion C file, compiled in the
same unit, exposes non-static `chelis_crmath_ffi_*` shims to Rust. Those symbols live
only inside the Rust binary; they are not a published C ABI and appear in no header.

The crate's Rust API is per dtype and closed: `exp_f32`, `exp_f64`, ..., `tanh_f64`,
plus `f16`/`bf16` forms that widen exactly to f32, call the f32 kernel, and finalize
once through the existing `half` narrowing, as [04-NUM-8] requires. Every consumer calls
this API and nothing else:

- `crates/chelis-types/src/dtype_semantics.rs` (scalar and activation evaluation);
- `crates/chelis-ir/src/optimize.rs` (constant folding computes at the declared width
  through the same kernels, so a folded literal and a run-time value cannot differ);
- `crates/chelis-compiler-api/src/runtime/host_ops.rs` (host softmax and the other
  host-lane numerics);
- `crates/chelis-prove/src/concrete_eval.rs` and `contracts.rs` (§7).

The alternative, a Rust port, was rejected because it would be a second implementation
whose agreement with the emitted C would rest on the oracle alone, and every upstream
fix would have to be ported by hand.

Closure is structural rather than an inventory: the workspace `clippy.toml` lists
`f32::{exp,ln,sin,cos,tan,atan,tanh}` and the `f64` forms (and `exp2`, `powf`, and the
other libm-backed methods) under `disallowed-methods`, so a new call outside
`chelis-crmath` fails the gate. Test code that needs a reference value uses MPFR-derived
fixtures, not `std`.

### 4.2 Built programs

The C backend emits the amalgamation's kernels into the generated translation unit as
`static` functions, only those the program uses, and calls `chelis_cr_expf` and its
siblings directly. The text comes from `chelis-crmath` through `include_str!`, so the
emitted kernels are byte-identical to the evaluator's. Because they are `static`, a
static library built by `chelis build` exports no new symbol and no published header
gains a bare `float` or `double` signature, which keeps the change inside the
§C6 numeric-surface rule as #2979 binds it. `chelis_math.h` no longer selects a math
library; it is reduced to what it still has to declare or removed from the published
header roots, and the capacity census is rerun against the result.

Prebuilt kernels in the runtime archive were considered: they would remove the host C
compiler from transcendental values entirely, but they would export bare-float symbols
from the carried archive, which #2979 forbids. With static kernels the host compiler
does compile them, and correct rounding plus the strict profile is what makes its choice
irrelevant; §8's cross-host oracle checks that claim on the supported compilers.

Emission changes:

- `host_emit.rs`: every libm name (`expf`, `exp`, `logf`, ..., `tanhf`, `tanh`) becomes
  the `chelis_cr_*` call. The hand-written host activation helpers are replaced by one
  emitter driven by the same §3.3 lowering the evaluator uses, so `sigmoid`, `silu`,
  `gelu`, and `softmax` have one definition per lane pair rather than two hand copies.
- `emit.rs`: `vforce_func`, `sleef_macro`, the Sleef `simd_step_expr` arms, and the
  `emit_fused_reduce` math routes are removed; fused kernels call the scalar
  `chelis_cr_*` functions per element. The `sleef` cargo feature, `MathLib`,
  `MathLib::detect`, and `CodegenOptions::math_lib_override` are deleted, along with
  the tests that pinned the Sleef and vForce paths.
- `tier2.rs`: `tanh` stops lowering to `2*sigmoid(2x)-1`; the IR gains a `Tanh` unary
  primitive with the [05-OP-46] adjoint `g*(1-y*y)`. `gelu` lowers to `x*sigmoid(2u)`
  with spec/05 §3.3's exact spelling of `u`, not to `0.5*x*(1+tanh(u))`. The two are
  equal over the reals, but the tanh spelling cancels for negative `x`: with a
  correctly rounded f32 `tanh` it measured 176 ULP at `x=-3`, 4,354 ULP at `x=-4`, and
  `0` instead of `-8.4e-11` at `x=-6`. The sigmoid spelling has no cancellation; its
  remaining error (measured up to 148 ULP at `x=-9.336` and 37 ULP at `x=-5.476` on a
  dense f32 sweep) comes from the rounding of `u` amplified by `exp(2u)`. Because the
  graph is pinned, that error is the same in every lane. `sigmoid` and `silu` keep
  their graphs (measured at most 2.3 ULP with exact leaves).

Vectorized kernels are admissible later only as vectorized forms of these kernels that
pass the exhaustive binary32 oracle; vendor vector libraries are not correctly rounded
and are excluded.

### 4.3 GPU lanes

HIP and Metal are `scope:experimental`, and their implementation is deferred until
after the CPU lanes. Until a device lane has Chelis-owned kernels that pass the same
oracle, a transcendental in device code is rejected under [05-UNS-1] with a typed
diagnostic, never computed with `ocml`, MSL built-ins, or `precise::` forms, and the
rejection is listed on the known-issues page (#1170).

Metal is harder than a missing kernel. MSL permits a device to flush f32 subnormals
and to round f32 arithmetic toward zero independently of the fast-math setting, and
it has no f64. Turning fast math off therefore does not make Metal reach CPU bits, and
this design makes no such claim: an f64 operation, and any f32 operation whose bits
the device cannot guarantee, stays rejected on Metal. Whether a device-kernel route
exists for f32 on Metal is an open question for the deferred GPU work. The other GPU
items in #2968 and #2969 (NaN-dropping reductions, the f16 max/min identity,
`atomicAdd` scatter order) are outside this design and stay open.

## 5. Build profile and environment (#2962)

`spec/08-backends.md` §7 makes the compile and link invocation a function of the declared
target. In `crates/chelis-backend-c/src/toolchain.rs` and the native build driver:

- `-march=native` is removed; the target's CPU baseline is part of the declared target.
- Every native tool runs with a cleared environment plus an allowlist (`PATH` for tool
  resolution, `TMPDIR`, the SDK variables Apple's driver needs, and the explicit
  compiler selection). `CFLAGS`, `CPPFLAGS`, `LDFLAGS`, `CCC_OVERRIDE_OPTIONS`,
  `NIX_CFLAGS_COMPILE`, `NIX_LDFLAGS`, and OpenMP variables are not on it.
- The profile is `-O2 -ffp-contract=off -fno-fast-math` plus the target's required
  flags, the same strict profile `cross_lane_gate.md` PD2 defines, now the product
  default.
- A selected compiler that does not honor the profile fails the build. The amalgamation's
  `#error` guards catch fast-math and excess-precision injection at compile time. A
  canary translation unit compiled with the exact argv detects contraction (a
  non-inlinable `a*b+c` on operands where the fused and unfused results differ) and
  flush-to-zero; it runs when the target can execute on the build host, and the build
  fails if it reports a violation.

## 6. Floating-point environment and NaN canonicalization (#2964)

- **Entry points.** The emitted process entry, every exported static-library function,
  and every binding call save the caller's floating-point control register (x86
  MXCSR, arm64 FPCR), set round-to-nearest-even with FTZ and DAZ off by writing that
  register (`fesetround` alone leaves FTZ and DAZ untouched), and restore the saved
  value on every return path, a trap return included. The helper lives in the runtime as a pair of internal functions over an
  opaque saved-state carrier, so no numeric value crosses it.
- **Evaluator.** Each thread that computes sets the same state once, and a debug
  assertion checks it at kernel entry.
- **NaN.** Every float operation in both lanes finalizes a NaN result to the canonical
  pattern, not only subtraction as before. The kernels' wrapper does this for the
  transcendentals; the evaluator's binop and unop arms and the emitted C's arithmetic
  helpers do it for the rest, where the hardware's default NaN differs (x86 produces
  `0xffc00000` for an invalid operation).

## 7. `chelis prove` under correct rounding (#2965)

A correctly rounded operation is modelled as the exact real function rounded once:
`fl_p(f(x))`, the same shape as IEEE `+ - * /` and `sqrt`. Properties transfer as
follows:

- **Transfer by construction.** Rounding is monotone, so monotonicity of the real
  function carries to the float operation in its weak form (`x <= y` implies
  `exp(x) <= exp(y)`). Exact values at representable points carry (`exp(0) = 1`,
  `log(1) = 0`, `sin(0) = 0`). Range bounds at representable endpoints carry
  (`-1 <= tanh(x) <= 1`). Odd and even symmetry carry because round-to-nearest-even is
  symmetric.
- **Do not transfer.** Strict inequalities that depend on magnitude: `exp(x) > 0` is
  false at f32 for `x` below about -103.97, where the correctly rounded result is `+0`.
  The standard contract `std.exp.positivity` is therefore restated either as a
  real-model property, labelled as such, or as `exp(x) >= 0` for the float operation;
  the implementation slice chooses and records which (open question 3).
- **Error envelope.** Where a proof needs the real value, the float result lies within
  half an ULP of it at the arithmetic width, with the f16/bf16 composition adding the
  final storage rounding.

The fuzz discharges in `contracts.rs` and the concrete evaluator stop computing f32
properties at f64 through host libm: they call `chelis-crmath` at the declared width,
and the `normal_cdf` fuzz runs the shipped `contracts::normal_cdf` graph rather than its
own Abramowitz-Stegun copy.

## 8. Test plan

Spec-first: each stub below is written before the code it tests, and each positive test
has a negative partner.

**Kernel oracle (`chelis-crmath`).**

1. Per-PR canary: the #2957 witnesses (the `normal_cdf`, `sigmoid`, `tanh`, `silu`,
   `gelu`, and `softmax` inputs from #2952, #2959, and #2971) plus every special case of
   [05-OP-46] (`+-0`, `+-inf`, NaN with payload and sign, subnormal results, overflow
   thresholds), as MPFR-derived bit patterns checked into fixtures. Negative partner: the
   same fixtures fail when the wrapper is bypassed and Rust `std` is called (proves the
   canary distinguishes correct rounding from libm on at least one witness per function).
2. Per-PR binary64 worst cases: a deterministic sample of CORE-MATH's shipped `.wc`
   corpora for the seven functions, compared bitwise with MPFR-derived expectations.
3. Manual or nightly gate: every binary32 input of each of the seven functions, through
   the `chelis-crmath` API and through a built executable, compared bitwise with MPFR.
   Entered in `docs/manual_gates.md` with its command, the success condition "zero
   mismatches over 2^32 inputs per function per lane", and its owner (#2957).
4. f16 and bf16: exhaustive (65,536 inputs each) per PR, checking the composition
   "correctly rounded at f32, then one finalization". Negative partner: a planted
   direct-to-storage rounding is reported on every input where the two differ.
5. NaN canonicalization: payload-carrying and negative NaN operands give the canonical
   pattern from every kernel; partner: the raw upstream kernel does not.

**Lane parity.**

6. One input as a scalar literal, a folded constant, a run-time scalar, a run-time tensor
   (contiguous, permuted, rank-0), and a fused-kernel element gives identical bits in
   eval and in a built executable, for each function and each float dtype. Partner: the
   test fails if the C fold or the IR fold is redirected to `std`.
7. `gelu` at `x = -3, -4, -5, -6, -9.336` matches the MPFR-evaluated value of the pinned
   `x*sigmoid(2u)` graph bit for bit in both lanes; partner: the `0.5*x*(1+tanh(u))`
   spelling fails it. Eval and C are bit-equal on `sigmoid`, `silu`, `gelu`, `softmax`, and `normal_cdf`
   over the #2952, #2959, and #2971 probe sets, and on `tanh` near zero (`1e-3`, `1e-5`),
   where the old lowering was 620 and 14,932 ULP off.
8. Emitted-C scan: generated C for a corpus that uses every transcendental contains no
   libm, Sleef, or vForce identifier and no `#include` of `Accelerate` for math. Partner:
   a planted `expf` call fails it.

**Profile and environment.**

9. Environment injection: a build with `CFLAGS=-ffast-math`, `CCC_OVERRIDE_OPTIONS`
   adding `-ffast-math`, `NIX_CFLAGS_COMPILE`, or a `CHELIS_CC` wrapper that appends
   `-ffp-contract=fast` either fails with the profile diagnostic or produces bit-identical
   output; a clean build is the positive control.
10. FP environment: a host C driver that sets FTZ/DAZ and `FE_UPWARD`, then calls an
    exported function of a built static library on inputs with subnormal and inexact
    results, gets bits identical to eval and finds its own FTZ/DAZ/rounding state
    restored afterwards, including after a trapping call.
11. Cross-host: an x86_64-linux GCC build and a macOS arm64 Clang build of one corpus
    produce identical observation bytes (nightly).

**Prove.**

12. Fuzz discharges evaluate at the declared width through `chelis-crmath`; a planted
    property that holds at f64 but fails at f32 is refuted. The restated exp-positivity
    contract is checked at `x = -104` f32.

**Spec mirror.**

13. `OP_TOLERANCES` is empty, `AgreementOp` has only `Exact`, and the generated
    [05-OBS-3] table block has no rows; the existing byte-for-byte tripwire covers the
    mirror.
14. A check (the #2967 oracle,
    `agreement_tolerance::no_numbered_chapter_or_design_doc_states_an_agreement_tolerance`)
    fails if a numbered chapter, a `spec/registry/` file, or a design document under
    `spec/design/` other than archived ones states a numeric cross-lane agreement
    tolerance (a named absolute or relative tolerance constant, a nonzero ULP bound, or
    a blanket epsilon) outside the generated [05-OBS-3] block.

## 9. Implementation slices

One pull request; each slice is a commit. The CPU lanes (eval and C) come first; GPU
implementation is deferred and only fenced.

Three slices are implemented on parallel branches and combined into the pull request.
They do not depend on the kernels:

| slice | content | sub-issues it closes (oracle) |
|---|---|---|
| impl-profile | Strict product profile, environment allowlist, canary TU (§5). | #2962 (test 9) |
| impl-evalwidth | Eval f64 funnels at declared width (softmax, cumsum, clamp, scatter-add) and clamp trap parity. | #2971, #2972, #2973 (their own oracles) |
| impl-fpenv | Entry-point environment save/set/restore and canonical NaN in every float arm (§6). | #2964 (test 10 and the NaN unit test) |

The kernel slices follow, in order:

| # | slice | sub-issues it closes (oracle) |
|---|---|---|
| S0 | Spec amendments and this document. | none alone; supplies #2967's text items 1, 2, and 4 and the manifest alignment of item 3 |
| K1 | `chelis-crmath`: vendored CORE-MATH, the amalgamation script and its `--check`, the Rust API, NaN canonicalization, guards; tests 1-5 (with test 3 a manual gate); the `disallowed-methods` lint. | none alone |
| K2 | Eval wiring: `dtype_semantics.rs`, IR constant folding, host ops through `chelis-crmath`; `Tanh` IR primitive and adjoint; `gelu` respelled; host activations derived from the §3.3 lowering. | none alone |
| K3 | C wiring: static kernel emission, libm names replaced, host activations from one definition; vForce, Sleef, `MathLib`, and the `sleef` feature removed; `chelis_math.h` rewritten and census rerun. | #2952, #2958, #2959, #2961, #2963 (tests 6-8, 11); #2966's Sleef items only, so #2966 stays open |
| K4 | Prove: fuzz evaluator and contracts at declared width; restated exp contract. | #2965 (test 12) |
| S8 | `OP_TOLERANCES` emptied, `AgreementOp` reduced, [05-OBS-3] block regenerated; the tolerance-statement check. | #2967 (tests 13-14) |
| G1 | GPU fence: device transcendentals and Metal f64 rejected under [05-UNS-1]; known-issues rows. Device kernels deferred. | none closed; Part of #2968 and #2969 |
| S9 | `docs/book` backends page, `docs/manual_gates.md` rows, `changelog.d` fragment (`changed.breaking`: transcendental and compound results change bits wherever the old lanes were not correctly rounded, and `gelu` changes spelling). | none |

After K3, #1311's question (pin a reference graph for embedded transcendentals) is
answered by correct rounding instead; whether to close it as superseded is a tracker
decision.

## 10. Risks

- **Compiler trust.** Static kernels are compiled by the user's C compiler. Correct
  rounding makes any correct compilation agree, and the strict profile removes the
  value-changing options, but a miscompile in an untested compiler version would not be
  caught by the oracle. The cross-host oracle covers the CI compilers; others are
  covered by the same trust Chelis already places in the compiler for `+` and `*`.
- **Speed.** Per-element scalar calls replace vForce on macOS tensor `exp`, `log`, and
  `sin`, and x86-64 baseline builds call a software `fma`. Correctness comes first;
  vectorized kernels return only through the exhaustive oracle.
- **Upstream changes.** A CORE-MATH update is a vendoring change rerun through tests 1-5
  and the exhaustive gate before it lands. Because results are correctly rounded, an
  upstream algorithm change cannot change Chelis values unless it is a bug.

## 11. Open questions

1. Whether `chelis_math.h` survives in reduced form or leaves the published header
   roots entirely (K3 decides from what it still has to declare; the census rerun is
   the check either way).
2. Whether the x86-64 target's declared CPU baseline should include hardware FMA
   (x86-64-v3), which would remove the software `fma` call from binary64 kernels at the
   cost of not running on older CPUs. Values do not depend on the answer.
3. Whether `std.exp.positivity` becomes a labelled real-model property or the float
   property `exp(x) >= 0` (§7).
