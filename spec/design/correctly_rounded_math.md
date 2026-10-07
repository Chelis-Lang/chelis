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
  `#if defined(__x86_64__)` with portable fallbacks. The amalgamation keeps only the
  portable arm of each (§3.3), so every architecture compiles the same C; the oracle
  still runs on each architecture (§8), because the compiler lowers that C differently
  per target.
- **Integer rounding.** Six kernels round a reduced argument to an integer with ties to
  even: binary32 `sin`, `cos`, `tan` and binary64 `exp`, `erfc` through a
  `roundeven_finite` helper, binary64 `sin` through a direct `__builtin_roundeven` call.
  Upstream's helper is that builtin on GCC 10 and Clang 17, inline assembly or a
  `round`-based fallback elsewhere. On baseline x86-64, and on aarch64 with GCC 10, the
  builtin lowers to a call to the C library's `roundeven`. musl and glibc before 2.25 do
  not provide it. §3.3 therefore replaces all six with one helper around
  `__builtin_rint`. The portable `copysign((|x| + 2^52) - 2^52, x)` gives the same
  values, but it makes most of these kernels about 20% slower on AArch64.
- **FMA.** The kernels call `__builtin_fma`. Where the target has no hardware FMA in its
  baseline (x86-64 without `-mfma`, which the strict profile forbids adding from the
  host CPU), the compiler emits a call to the C library's `fma`, which C17 7.12.13.1
  requires to round once, so values are unaffected; the cost is speed (open question 1).
- **Rounding mode.** The kernels read the dynamic rounding mode (`fegetround`) in a few
  binary64 paths and otherwise assume arithmetic rounds in the current mode. Chelis
  pins round-to-nearest-even at every entry point (§6), so the amalgamation keeps only
  the round-to-nearest case of each such switch and `-frounding-math` is not needed.
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
     NaN result;
  4. reduces the kernel text to the generated-C contract every built unit must
     satisfy, since `chelis build` emits these bytes (§4.2). It drops what the contract
     forbids and Chelis never observes: `<fenv.h>`, the `FENV_ACCESS` pragma,
     floating-point exception raises, the `errno` blocks, the `noinline`/`cold`
     attributes, and the x86-64 SSE intrinsic arms, keeping the portable arm beside
     each. The `unsigned _BitInt(128)` conditionals keep their `unsigned __int128` arm,
     the same arithmetic. Every definition, the entries included, is written
     `static`. The script fails if a forbidden token or an include outside the
     generated-C allowlist survives; that allowlist admits the ISO C headers
     `<stdint.h>` and `<float.h>` for the kernels. Values are unchanged: every dropped
     arm computes the bits of the arm that stays;
  5. replaces each upstream `roundeven_finite` definition, and binary64 `sin`'s direct
     `__builtin_roundeven` call, with one helper that returns `__builtin_rint (x)`. In
     the round-to-nearest-even mode that Chelis pins at every entry (§6), `rint`
     (C17 7.12.9.4) gives `roundeven(x)` for every finite `x`, signed zero included.
     On AArch64 the builtin is one `frintx` instruction, and GCC inlines it on baseline
     x86-64. Clang on baseline x86-64 calls the C library's `rint`, which every C99 C
     library has, musl included. The script fails if any `roundeven` builtin or call
     survives, or if a `roundeven_finite` is not this helper.
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
library and declares nothing; it stays a published header root so existing includes
resolve, and the capacity census is rerun against the result.

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
  `sqrtf` and `sqrt` are the one libm name kept deliberately: C Annex F (F.3) binds
  them to the IEEE 754 square root, which is correctly rounded, so libm already gives
  the [05-OP-46] result and both lanes only finalize its NaN.
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
`docs/book` backends page lists the rejection.

Metal is harder than a missing kernel. MSL permits a device to flush f32 subnormals
and to round f32 arithmetic toward zero independently of the fast-math setting, and
it has no f64. Turning fast math off therefore does not make Metal reach CPU bits, and
this design makes no such claim: codegen rejects every f64 operation on Metal under
[04-TGT-1], and rejects `sqrt` on both device lanes alongside the transcendentals,
because MSL compiles it under fast math and the HIP runtime compile does not pin a
correctly rounded square root. The fence covers those operations only: f32 division and
reciprocal on Metal also compile under MSL's default fast math, which does not promise
correct rounding, and the fence does not reject them. That gap is part of #2968, and a Metal
artifact makes no CPU-bits promise until it closes. Whether a device-kernel route
exists for f32 on Metal is an open question for the deferred GPU work. The other GPU
items in #2968 and #2969 (NaN-dropping reductions, the f16 max/min identity,
`atomicAdd` scatter order) are outside this design and stay open.

## 5. Build profile and environment (#2962)

`spec/08-backends.md` §7 makes the compile and link invocation a function of the declared
target. In `crates/chelis-backend-c/src/toolchain.rs` and the native build driver:

- `-march=native` is removed; the target's CPU baseline is part of the declared target.
- Every native tool runs with a cleared environment plus the allowlist
  `toolchain::TOOL_ENVIRONMENT`, which is exactly `PATH` (tool resolution) and `TMPDIR`
  (driver intermediates). The compiler is selected by its path, not through the
  environment. `CFLAGS`, `CPPFLAGS`, `LDFLAGS`, `CCC_OVERRIDE_OPTIONS`,
  `NIX_CFLAGS_COMPILE`, `NIX_LDFLAGS`, locale and OpenMP variables are not on it, and on
  macOS neither are the caller's `SDKROOT` nor `DEVELOPER_DIR`. On macOS the build sets
  `SDKROOT` itself to the SDK of the `xcode-select` default (`xcrun --show-sdk-path`
  under the allowlist), the one the `/usr/bin` shims choose. Another Xcode is chosen
  by naming its `clang` in `CHELIS_CC`; that compiler has no SDK of its own, and it
  compiles against this one.
- The profile is `-O2 -ffp-contract=off -fno-fast-math` plus the target's required
  flags, the same strict profile `cross_lane_gate.md` PD2 defines, now the product
  default.
- A selected compiler that does not honor the profile fails the build. A wrapper script
  is opaque on the command line, so `toolchain::verify_compiler` observes the compiler
  itself with the exact profile argv, in three steps. The predefined macros refuse fast
  math (`__FAST_MATH__`), finite-only math (`__FINITE_MATH_ONLY__`), and a dropped `-O2`
  (`__OPTIMIZE__`). The amalgamation's `#error` guards refuse the same two modes and
  excess-precision evaluation (`FLT_EVAL_METHOD` other than 0) when the canary compiles.
  The canary then runs, and any bit it prints that differs from the profile refuses the
  compiler, naming each broken obligation, its spec text, and its first broken row. A
  compiler that cannot build the canary at all is refused with its own diagnostic and
  without the claim that something added flags: the canary carries every kernel, so a
  builtin or C library function a kernel calls unconditionally would make every native
  build fail where it is missing. The kernels therefore round to an integer through
  §3.3's helper. Every supported GCC and Clang has its builtin, and every C99 C library
  has `rint`, the one function that the helper can call.
- A selected compiler whose C library is not the one the carried runtime archive was
  built for fails the build before anything compiles (spec/08 §7). The archive comes
  from the compiler's own build, so its C library is that build's Rust target
  environment (`gnu` or `musl`). After the profile's macro check, `verify_compiler`
  reads the compiler's C library from its `<stdint.h>`, which defines `__GLIBC__`
  under glibc and not under musl; C libraries that also define it for compatibility,
  such as uClibc-ng, read as glibc.

**The obligation table.** The canary is generated, not hand-written. `chelis_crmath::profile`
holds a closed list of the profile's obligations, each tied to the text it enforces:
correct rounding ([05-OP-46]); IEEE exceptional values ([05-OP-46], [04-NUM-2]); the
canonical NaN, signed zeros, and gradual underflow ([04-NUM-2]); one rounding at the
arithmetic width ([04-NUM-8]); and no contraction and no value-changing optimization
(spec/08 §7). It also holds a closed list of the float primitives generated code computes,
each at its operand width with the C expression generated code uses for it: the four
arithmetic operations, `sqrt`, the explicit `fma`, the comparisons, the f64-to-f32 and
f32-to-f64 conversions, the f16 and bf16 storage conversions, the fourteen kernels, and
the expression shapes value-changing optimizations rewrite (`a*b + c`, `(a + b) - a`,
`a / 3`, `a + 0`, `a - a`, `a * 0`, `a == a`, `a > MAX`, with their literal operands as
in generated code). Each primitive names the value classes its rows must cover: NaN
operands with a payload or a sign, invalid operations, infinite operands, signed zeros,
subnormal operands and results, and exact ties. A row's classes are read from its bits,
except a tie, which the generator proves is an exact midpoint. The rows are three
MPFR-derived fixtures: `canary.txt` and `binary64_worst_cases.txt` for the kernels, and
`profile_obligations.txt` for every other primitive.

**Generator and check mode.** `scripts/vendor_core_math.py obligations` writes
`profile_obligations.txt` from MPFR at each result width's precision, exponent range, and
subnormalization, each written operation rounded once; `obligations --check` fails when
the checked-in fixture differs from what the generator produces. The C canary has no
checked-in form to go stale: `profile::canary_driver` generates its `main` from the
primitive list on every check, and `toolchain.rs` places it after the kernel text and
generated code's own NaN-finalization helpers (`fp_env.rs`), so the canary compiles the
bytes generated units carry. Every operand is read from standard input at run time, so
nothing folds. The same rows drive the Rust lanes' tests, which check coverage by class
and that every row detects the rewrite it witnesses (a fused `a*b + c`, a reassociated or
reciprocal form, a NaN passed through unfinalized). The f16 and bf16 storage conversions
have no canary row because a compiler flag cannot reach their integer bit arithmetic;
their rows are checked in the runtime, the evaluator, and generated C's conversion
helpers instead (§8, test 15).

**Accepted flags.** The check accepts a flag exactly when the compiler passes the macro
checks and the canary built with the flag reproduces every obligation row; the wrapper
tests assert that rule, and assert byte-identical canary assembly only for flags that
change no instruction on either clang or gcc. `-fassociative-math` alone (clang and gcc
reassociate only once signed zeros and trapping are also given up) and
`-fexcess-precision=fast` on SSE2 and AArch64 (where `FLT_EVAL_METHOD` is 0) change no
instruction. `-fno-trapping-math` (status flags are unobservable and the profile installs
no trap) changes gcc's code but no row, and `-ffp-contract=on` is refused where it
contracts (clang) and accepted where ISO C mode treats it as `off` (gcc).
`-ffp-contract=fast` and every other flag in the wrapper tests are refused.

**Cost and memo.** Compiling the canary with every kernel costs about half a second; the
run is negligible. A process checks each compiler once: an accepted compiler is
remembered in memory, keyed by its resolved path, `--version` line, profile argv, and the
executable's size and modification time, so a wrapper edited in place is checked again.
Nothing is cached across processes, because a persisted acceptance could not observe a
change the key does not capture (a compiler configuration file, a file a wrapper reads),
and spec/08 requires such a compiler to fail the build.

## 6. Floating-point environment and NaN canonicalization (#2964)

- **Entry points.** The emitted process entry, every exported static-library function,
  and every binding call save the caller's floating-point control register (x86
  MXCSR, arm64 FPCR), set round-to-nearest-even with FTZ and DAZ off by writing that
  register (`fesetround` alone leaves FTZ and DAZ untouched), and restore the saved
  value on every return path, a trap return included. The helper lives in the runtime as a pair of internal functions over an
  opaque saved-state carrier, so no numeric value crosses it.
- **Compiler and evaluator.** Literal finalization, constant folding, evaluation, and
  differentiation run in the host's thread, so every public `chelis-compiler-api`
  function opens with the runtime's `FpEnvGuard`, which nests and restores the caller's
  state when dropped. An architecture test derives the public surface from the crate's
  `lib.rs` and fails on any public function, or function re-exported from another
  crate, that skips the guard.
- **OpenMP workers.** Entry writes only the calling thread's register, and a host's
  OpenMP pool threads keep whatever state they last had. Each emitted parallel loop is
  therefore an `omp for` inside a bare `omp parallel` region in which every
  participating thread enters before its share and leaves after it. The
  generated-artifact contract rejects the combined `omp parallel for` form.
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
  The standard contract `std.exp.positivity` is therefore restated as the float
  property `exp(x) >= 0`, which holds at every width; a real-model `exp(x) > 0` would
  describe no value a lane computes.
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
   adding `-ffast-math`, `NIX_CFLAGS_COMPILE`, or a `CHELIS_CC` wrapper that appends a
   value-changing flag either fails with the profile diagnostic or produces bit-identical
   output; a clean build is the positive control. One compiler-check test per
   wrapper flag (`toolchain::tests::verify_compiler_wrapper_*`) covers fast math,
   finite-only math, `-O0`, `-fno-honor-nans`, `-fno-honor-infinities`, `-ffp-contract=fast`,
   unsafe math, reassociation, reciprocal math, `-fno-signed-zeros`,
   `-ffp-model=fast`, and `-ffp-eval-method=double` (refused), and the accepted flags of
   §5 (accepted exactly when every row passes). `crmath_profile.rs` checks the obligation table itself.
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

**Storage conversions.**

15. Storage conversions: every f16 and bf16 conversion (the runtime header's
    `chelis_f32_to_f16` and `chelis_f32_to_bf16`, the runtime's f32 and f64 accumulator
    finalization, the evaluator's `{f16,bf16}_from_f64_rne`, and generated C's
    `chelis_host_f64_to_{f16,bf16}`) is compared with an integer round-to-nearest-even
    reference (`profile::storage_reference`, itself checked against the MPFR rows). Per
    PR: every storage-conversion row, every f32 within two ulps of a rounding boundary,
    and at every boundary the f64 midpoint, one ulp either side, and a relative 2^-30
    either side. Manual gate: every f32 input, and its exact f64 widening, with zero
    mismatches (`docs/manual_gates.md`).

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
| K3 | C wiring: static kernel emission from the contract-clean amalgamation, libm names rejected, host activations from one definition; vForce, Sleef, `MathLib`, the `sleef` feature, and `pow` removed; `chelis_math.h` declares nothing; census rerun. | #2952, #2958, #2959, #2961, #2963 (tests 6-8, 11); #2966's Sleef items only, so #2966 stays open |
| K4 | Prove: fuzz evaluator and contracts at declared width; `std.exp.positivity` restated as `exp(x) >= 0`. | #2965 (test 12) |
| S8 | `OP_TOLERANCES` emptied, `AgreementOp` reduced to `Exact`, [05-OBS-3] block regenerated; the tolerance-statement check (test 14) and the design passages it found corrected. | #2967 (tests 13-14) |
| G1 | GPU fence: HIP and Metal codegen reject device transcendentals, directly or in a fused chain, under [05-UNS-1] with [05-OP-46] as authority; Metal f64 keeps its [04-TGT-1] rejection. The fence is documented on the `docs/book` backends page. Device kernels deferred. | none closed; Part of #2968 and #2969 |
| S9 | `docs/book` backends page, `docs/manual_gates.md` rows, `changelog.d` fragment (`changed.breaking`: transcendental and compound results change bits wherever the old lanes were not correctly rounded, `tanh` becomes a primitive, `gelu` changes spelling, f32 contract verdicts can now fail, and the serialized graph schema moves from 23 to 24). | none |

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

1. Whether the x86-64 target's declared CPU baseline should include hardware FMA
   (x86-64-v3), which would remove the software `fma` call from binary64 kernels at the
   cost of not running on older CPUs. Values do not depend on the answer.

## 12. The error functions, `normal_cdf`, and `gelu`

spec/05 makes `gelu` the exact Gaussian error linear unit `x * Phi(x)`, with `Phi` the
standard normal CDF, and keeps the tanh approximation as a separate operation,
`gelu_tanh`, for models trained with it (GPT-2 and its descendants). The same `Phi` graph
(spec/05 §3.3) is the [05-OP-48] builtin `standard_normal_cdf`, and the [05-OP-35] stdlib
`normal_cdf` calls it, replacing the Abramowitz-and-Stegun polynomial, so the language
has one standard normal CDF. Both rest on two new [05-OP-46] correctly rounded
primitives, `erf` and `erfc`.

`standard_normal_cdf` is a builtin because no Surf definition can spell `Phi`: its residual
constant `cl` and splitter differ between f32 and f64, f16 and bf16 evaluate the f32
graph, and a definition generic over `p: Float` has no way to select per-dtype
constants. It cannot be named `normal_cdf`: inside the standard library package that name
resolves to the package's own definition, and shells define three-argument
`normal_cdf`s.

### 12.1 Why `erfc` and `erf` are primitives

`Phi(x) = 0.5 * erfc(-x/sqrt(2))`. The left tail, where `Phi` is tiny, is where finance
evaluates it (deep out-of-the-money options) and where `0.5 * (1 + erf(x/sqrt(2)))`
cancels to zero, so `erfc` is the leaf. No graph over the existing primitives reproduces
`erfc` to correct rounding, and the A&S-class polynomial it replaces breaks
monotonicity at f16. `erf` is a primitive on the `tanh` precedent: `1 - erfc(x)` loses
all relative accuracy near zero, so a library `erf` would be wrong exactly where `erf` is
small. ONNX `Erf` (hydronnx) and `Nautilus.Special` need it.

Making `Phi` itself the correctly rounded primitive was rejected. No correctly rounded
binary64 `Phi` kernel exists, in CORE-MATH or elsewhere, so the rule could not be met at
f64; a composition over `erfc` keeps the primitive set to functions with published
kernels at both widths (tenets 2 and 7).

### 12.2 Kernels and vendoring

CORE-MATH provides all four at the commit already pinned in `VENDOR.toml`
(`284b3b0e`), so the change adds files without moving the upstream pin:

| function | binary32 | binary64 | binary64 worst-case corpus shipped |
|---|---|---|---|
| `erf` | `binary32/erf/erff.c` | `binary64/erf/erf.c` | `erf.wc` |
| `erfc` | `binary32/erfc/erfcf.c` | `binary64/erfc/erfc.c` | `erfc.wc` |

Each file includes only `<stdint.h>`, `<errno.h>`, and (binary64 `erfc`) `<fenv.h>`, and
needs nothing outside what §3.3's reduction already handles: the `errno` blocks, the
`FENV_ACCESS` pragma, and binary64 `erfc`'s `roundeven_finite` helper, which §3.3
replaces. None uses `__int128`, `fegetround`, or an
`FE_` constant. `scripts/vendor_core_math.py import` adds the four files, regenerates the
amalgamation, and records their identifiers and hashes; the exhaustive binary32 gate and
the binary64 worst-case corpora of §8 extend to both functions. The lanes gain `Erf` and
`Erfc` wherever `Tanh` is wired (§4), each through the amalgamation's `chelis_cr_erf*`
entries.

### 12.3 The scaled argument and its error

`-x/sqrt(2)` rounds before `erfc` sees it. A relative argument error `d` becomes a
relative error of about `2 t^2 d` in `erfc(t)` for large `t = -x/sqrt(2)`, that is about
`x^2 d` in `Phi`, so the plain graph `0.5 * erfc(mul(neg(x), c))` loses up to `x^2` ulps
in the left tail. Measured against mpmath on 1,500 samples per band, the plain graph
reaches 1,539 ulps at f64 (`x = -36.5`), 159 ulps at f32 (`x = -12.6`), and 13 ulps at
f16.

The §3.3 graph removes that error by composition. A Veltkamp split and Dekker's
two-product give the exact rounding error `tl` of `th = RN(-x*c)`, and the first-order
Taylor term `k * exp(-th^2) * tl` corrects `erfc(th)` toward `erfc(th + tl)`. The
remaining error is the correctly rounded `erfc`, the final subtraction, and a second-order
term of relative size about `(x^2 u)^2`. In units of the result, the `erfc` rounding
costs half a unit except where `Phi` sits just below a power of two: there `erfc(th)`
lies one binade above the result and rounds on a grid twice as coarse, costing up to a
full result unit, and the subtraction adds another half, so the bound is about 1.5 units. At an 8- or 11-bit significand that second-order
term is not small, so f16 and bf16 evaluate the f32 graph and finalize once, which is the
composition [05-OP-46] already uses for its own narrow-width leaves. The range bound
`L = 64` exceeds every width's saturation point (f64 `Phi` underflows below
`x = -38.5`) and keeps `65 * 64`, `4097 * 64`, and `th^2` finite.

Measured with an mpmath model that evaluates the §3.3 text step by step, each
primitive rounded to its width and each leaf correctly rounded:

| width | inputs | max error (normal results) | adjacent-pair monotonicity violations |
|---|---|---|---|
| f64 | 8,006 sampled in `[-38, 8]` plus successors, and every binade boundary of `Phi` | 1.326 ulp (`x = -27.256566083845673`) | 0 |
| f32 | 8,006 sampled in `[-13, 8]` plus successors, and every binade boundary of `Phi` | 1.453 ulp (`x = -4.900964260101318`) | 0 |
| f16 | all 63,490 non-NaN values | 0.500005 ulp | 0 |
| bf16 | all 65,282 non-NaN values | 0.5000026 ulp | 0 |

Random sampling alone reported 0.98 ulp at f32 and f64: it never lands on the binade
boundaries, which a search that solves `Phi(x) = 2^k` for every `k` and scans the
neighbours finds. Every lane returns the same bits there, and the two boundary inputs
are rows of the per-pull-request witness tests.

At f16 and bf16 the f32 result is within about 1.5 f32 units of the truth, which is
`1.5 * 2^-13` f16 units or `1.5 * 2^-16` bf16 units, so the single finalization stays
within half a unit plus that margin; the exhaustive figures above sit far inside it.
`gelu` there is within half a unit of its result plus `|x|` times that `Phi` error,
because its product reads the finalized `Phi`. The manual gate
`half_dtype_phi_and_gelu_error_bounds_hold_on_every_input` checks both bounds and
monotonicity over every f16 and bf16 input.

The spec states the graph, not a bound: the bits are pinned by construction, and these
figures belong to the implementation oracle (§12.6). Monotonicity is measured, not
structural; the plain graph is monotone by construction but fails the accuracy goal.

Computing the f32 graph at f64 and rounding once was rejected. It is not a composition
over f32 primitives, the Metal lane has no f64, and the f32 graph already holds within
about 1.5 ulp. The narrow-width normal_cdf reflection failures recorded in #3116 and in
the regenerated discharge table do not come from the graph's accuracy. Reflection compares
two separately rounded values, `Phi(-x)` and `1 - Phi(x)`; where they lie in `[0.5, 1)`
they share that binade's grid (`2^-24` at f32), so two results each within a unit or so
of the truth differ by a whole unit there, far above the fixed `1e-10` tolerance. The
recorded counterexamples are exactly one such unit: `2^-24` at `x = -3.62` (f32),
`2^-11` at `x = -3.18` (f16), and `2^-8` at `x = -1.95` (bf16). That property needs a
per-width tolerance, not a new graph.

### 12.4 Infinite inputs of the gated activations

`silu`, `gelu`, and `gelu_tanh` multiply `x` by a gate that is `+0` at `-inf`, so the
plain product is `-inf * 0`, a NaN where the limit is zero. The §3.3 multiplicand guard
`m(x)` replaces `-inf` by `-0.0` before the product and changes nothing else. `-0.0`
rather than `+0` because the functions approach zero from below, and every
sufficiently negative finite input already returns `-0.0` (`x * +0`); IEEE 754 likewise
signs a zero limit by its side. The guard is a `where` on `cmplt(x, lowest)`, which
is true only at `-inf`, so the result stays one graph over existing primitives and
the product node never sees `-inf`.
Checked with the same model, old graph against guarded graph: every finite f16 and bf16
value and 4,004 sampled f32 and f64 values are bit-identical, `-inf` gives `-0.0`,
`+inf` gives `+inf`, and NaN stays NaN for all three. `sigmoid` and `normal_cdf` give
`+0` and `1` at the infinities without a guard.

Gradients at an infinite input remain the derivative of the graph and can be NaN, for
example through `exp`'s adjoint `0 * inf` inside `sigmoid` at `-inf`. That is
unchanged by the guard and outside this change.

### 12.5 Gradients

`gelu` and `normal_cdf` differentiate the stated graph (spec/05 §3.3, [05-OP-35]), as
`gelu_tanh` does today. The `erfc` adjoint `-g*k*exp(-(x*x))` supplies the density, so the
gradient approximates `Phi(x) + x*phi(x)`, with relative error growing like `x^2 u`
from the rounded square, as every `exp(-x*x)` density does. Through the split, the
tangents of `ah` and `al` sum to the tangent of `a`, so `tl` contributes a derivative of
order `u` and the correction's gradient stays at rounding level. Outside `|x| < L` the
correction reads `z = 0`, so its gradient is exactly zero.

### 12.6 Implementation surface

`chelis_types::activation` holds the one definition of every §3.3 graph, `Phi`
included; the IR lowering (`chelis_ir::tier2`), the evaluator's scalar and tensor
kernels, and the C host helpers (built from the `tier2` graph) all run it, so `gelu`,
`gelu_tanh`, `silu`, and `standard_normal_cdf` have no hand-written lane copy. The graph's
steps are `neg`, `abs`, `exp`, `erfc`, `recip`, `add`, `sub`, `mul`, `cmplt`, `where`,
and the f16/bf16 `cast`s. `erf` and `erfc` are wired wherever `tanh` is: the
evaluator, IR evaluation and fusion, the adjoint, C kernel and host emission, the wire
schema (version 27), and the device fences. Tests pin the evaluator against an
independent MPFR model of the graphs on the witness set at every width, the IR and the
C host helpers against the evaluator (every finite f16 input and the witnesses at the
other widths), and the f16/bf16 bounds above as a manual gate. The prove discharges in
`crates/chelis-prove/data/standard_contract_discharges.json` and the standard graph digest
regenerate.
