# Backends

This chapter defines how a checked program becomes executable code: the C backend,
the GPU backends, and the invariants every backend preserves.

## 1. Backend Strategy

Chelis lowers typed programs to a RISC DAG and treats backend emission as a separate
concern from parsing, type checking, and lowering.
The backends are layered:

1. the C backend is the reference host backend
2. it is the numerical oracle the other backends are checked against
3. GPU code generation is source-to-source: HIP for AMD GPUs (§3) and Metal for
   Apple GPUs (§4)
4. ecosystem integration backends are additive layers (§5)

Chelis does **not** maintain multiple competing native host code generators.

When host lowering exhausts its native stack budget on a deeply nested checked
expression, it reports a located lowering diagnostic before the process can
abort. A shallow expression with the same operations still lowers normally.

### 1.1 Evaluation output and failure

`chelis eval` preserves output produced before an evaluation failure in source
order. In text mode it writes those transcript lines to stdout before reporting
the failure on stderr and exiting nonzero. It does not publish result roots for
the failed evaluation. Effects following the failure do not execute.

With `--json`, stdout is reserved for the successful result document, including
its transcript. On failure stdout is empty; preceding transcript lines are
written to stderr before the diagnostic. Both ordinary and package-context
evaluation obey the same channel rules. Compiler API evaluation errors retain
the preceding transcript separately from diagnostics so embedders can choose
their own output sink without rerunning the program.

### 1.2 Executable grammar validation

`chelis validate` checks source against the executable grammars before any
backend runs:

- `chelis validate --surf file.ch` validates Surf syntax against the PEG conformance grammar
- `chelis validate --deep file.dp` validates Deep syntax plus the closed tag/metadata/arity rules
- `chelis validate --desugar file.ch` validates compiler-desugared canonical Deep output

The validator lives in the standalone `chelis-validate` crate. Its test suite checks
agreement across the executable examples, the illustrative syntax examples,
`SKILL.md`, curated positive spec fixtures, and curated negative fixtures.

## 2. C Backend

The C backend is the reference implementation.
Its job is to turn the DAG into portable host code that can be compiled with the system
toolchain.

"Reference implementation" here means the first and most complete backend, and the
practical numeric oracle the other backends are checked against. It does not mean the C
backend DEFINES the values: every lane, the evaluator included, owes its results to
`spec/04-type-system.md` §9 - [04-NUM-8]'s arithmetic width, [04-NUM-1..7]'s finalize and
trap rules, and [04-NUM-11]'s exactness guarantee - and where a lane and those atoms
disagree, the lane has the bug. The C backend is the oracle because it is the most
complete conforming lane, not because conformance is defined as agreeing with it.

C compilation derives identifiers from source filenames without treating filesystem
spelling as C syntax. Distinct filename stems SHALL have distinct generated module
symbols, including a stem that literally spells another stem's escaped symbol.
The generated header and source SHALL use the same symbol. Default artifact filenames
retain the source stem; an explicit output filename does not change the module symbol.
Each authored C function export, except source-level `main`, SHALL use the
compiler-reserved `chelis_fn_` prefix followed by the lowercase hexadecimal UTF-8 bytes
of its Chelis name. Source-level `main` uses the module-qualified `<module>__main`
symbol so a downstream C driver can retain its own `main`; Reef/package qualification
does not erase that exception when the decoded source binding is exactly `main`. An
ordinary authored identifier ending in `__main` is not source-level `main` and uses the
universal mapping. For an unqualified source identity exactly equal to `main`, the
symbol SHALL be the exact generated program identity followed by `__main`; accepting
an arbitrary suffix-shaped symbol is not conforming.

The generated header begins with exact artifact version, lowercase-hex UTF-8 program
identity, and lowercase source SHA-256 metadata. Each declaration is preceded by an
exact `chelis-declaration` record carrying lowercase-hex UTF-8 source identity,
canonical C symbol, canonical declaration, and canonical SHA-256 definition
commitment; the commitment covers the exact source function-definition bytes and the
translation unit's canonical source-local preprocessing-directive sequence. The
following declaration bytes SHALL
equal the decoded declaration exactly, including multiline formatting. The generated
source begins with the matching version and program identity. Every public definition
is enclosed by exact `chelis-export-begin` / `chelis-export-end` comment records; the
begin record carries `authored` or `direct`, then the same lowercase-hex UTF-8 source
identity, symbol, declaration, and definition commitment. Validators SHALL parse the C
translation unit structurally, not with substring or line-layout heuristics, and bind
each block to exactly one enclosed function definition. Its declarator and structural
signature SHALL match the record, it SHALL have external linkage, and its exact AST
definition byte range plus source-local preprocessing context SHALL hash to the
recorded commitment. Whitespace, comments,
multiline formatting, and comment-separated storage specifiers are interpreted by
that structural parse.

The declaration records and source export blocks SHALL be exactly bijective in program
identity, source identity, canonical symbol, declaration, external linkage, and exact
definition commitment. The header's source digest additionally binds the complete source
bytes, but resealing that digest SHALL preserve and revalidate the compiler-emitted
per-definition commitments rather than deriving new commitments from arbitrary changed
source. Thus partial headers, unmarked additions, reformatted or type-changed
definitions, and reassociated metadata or bodies SHALL fail before native execution.
Generated source SHALL NOT use direct or indirect preprocessor aliases to reassociate
a public definition. Every externally linked function definition except the generated
process entry `main` SHALL be registered by exactly one public export block.
Translation-unit-private `static` helpers remain outside the public blocks and valid.
Downstream C code SHALL call declarations from the generated header, and tooling SHALL
consume the generated associations rather than reconstructing symbols from Chelis
source spellings.

Design points:

- emit loops for elementwise, reduction, and movement operations
- use OpenMP for elementwise and reduction parallelism
- pattern-match BLAS-friendly subgraphs such as matrix multiplication
- manage temporary buffers with explicit lifetime-aware memory planning
- ship `chelis_runtime.h` plus a Rust static runtime library alongside generated code
- tuple-returning host exports use the stable runtime tuple ABI:
  generated headers surface `chelis_tuple*`, drivers construct tuples with
  `chelis_tuple_from_values(...)`, and typed extraction goes through the
  `chelis_tuple_get_*` helpers documented in `chelis_runtime.h`

This backend is the correctness oracle for the GPU and interoperability backends.

Generated code holds no random state: every draw reads the key it is given
([05-RNG-1]), so reentrant and concurrent public invocations share nothing
random. A scalar `key` parameter or result of a host entry crosses the C ABI
as the published carrier `typedef struct { uint64_t bits; } chelis_key;`,
whose `bits` are the key's 64 bits ([05-RNG-2]), and
`chelis_key chelis_key_from_seed(int64_t seed)` returns [05-OP-69]'s key, and
`chelis_string chelis_string_from_key(chelis_key key)` returns its printed form
([05-OBS-2]), which a compiled program prints for a key root. A key tensor
at a host entry crosses as a `chelis_tensor` of runtime dtype `key` (id 9).
An entry on the four-argument public tensor ABI carries every input and result
as a `chelis_tensor`, a scalar as a rank-0 tensor, so its key inputs and key
results are `chelis_tensor`s of runtime dtype `key`, rank 0 for a scalar key.
Every key tensor crossing an entry is subject to [04-NUM-11]'s entry dtype
check. A key is never a bare integer at the boundary. DLPack exchange (spec/11 §1.3) and NumPy conversion refuse a key
tensor with a typed rejection, because no numeric interchange dtype describes
a key; Python passes keys through spec/10 §3.2's execution values.
Private context transport does not change authored public
function declarations or the four-argument public tensor ABI.

### 2.1 Runtime artifact identity

A compiler build carries its runtime: the static archive and public runtime
headers produced by the runtime compilation unit that the same build links. The
CLI and the Python extension SHALL stage, link and export only those carried
bytes. They SHALL NOT select a runtime archive by searching directories or by
file name, modification time, enumeration order, location relative to the
executable, or environment variable. A set `CHELIS_RUNTIME_DIR` SHALL be rejected
before staging; it is neither honored nor ignored.

The carried archive SHALL be the static-library output of the same runtime
compilation that produced the Rust library the consumer links. A compilation
that emits linkable output and cannot locate that archive SHALL fail. A
compilation that emits no linkable output MAY carry an inert placeholder, and
staging and export SHALL refuse it. A sealed distribution build SHALL carry no
build path.

Staging SHALL replace each staged file atomically, verify the written archive
against the carried SHA-256 digest before publishing it, and write
`chelis_runtime.receipt.json` last, recording that digest, the header digests and
the build mode; a staging without a receipt is incomplete. The receipt records
staging only; it makes no claim about linking or
execution. `chelis build` SHALL name the staged archive and its digest on stdout.
Link commands that Chelis prints or runs SHALL name the staged archive by path,
never through a library search.

A development build SHALL refuse to stage when the runtime's declared source
inputs changed after it was built. The declared inputs are the manifest, any
build script, and every file under `src/` and `include/` of the runtime crate and
of each workspace crate it depends on, and the workspace lockfile; files whose
names begin with `.` are not inputs. The runtime compilation SHALL record a
digest for each declared input; a changed or missing recorded file, or an
unrecorded file in those roots, fails staging with its path. A development build
whose runtime was compiled without its declared inputs SHALL refuse to stage. A
sealed distribution build is declared when it is built and reads no source
checkout. An unavailable checkout SHALL NOT select sealed mode.

A distribution that ships a runtime archive or headers beside a compiler SHALL
take them from that compiler's runtime export and SHALL verify the archive
against the export's digest. The carried runtime adds no C callable and does not
change callable metadata `abi_version: 2` (spec/11 §1.4).

## 3. HIP Backend

The HIP backend uses Futhark-style source-to-source compilation:

- host-side control remains in generated C
- GPU kernels are emitted as HIP source strings
- `hiprtc` performs runtime compilation of those kernels

HIP is the vendor-facing abstraction layer for AMD GPUs.
Chelis does **not** maintain separate CUDA and OpenCL backend implementations.

### 3.1 Kernel code generation

The `chelis-backend-hip` crate generates HIP host source with embedded HIP kernel strings.
It uses the same ABI as the C backend (`chelis_tensor **inputs/outputs`).

- kernel source strings cover elementwise ops, reductions, fill, and cast
- `chelis-ir::grad_then_fuse` differentiates the ordinary DAG first, then fuses the
  resulting gradient DAG
- shapes/strides are passed as individual int kernel parameters (not device pointers)
- debug builds reset/check a per-module `chelis_gpu_failure` flag after every kernel launch
- views preserve backing allocation size so debug index guards validate against real storage
- movement ops (reshape, permute, expand, insert, stride) are host-side metadata operations
- `chelis_gpu_free` releases allocations and `chelis_gpu_free_view` releases views
- `chelis build app.ch --target hip` emits compilable `*_hip.cpp` host output
- `pad` and `shrink` are typed per-output-element kernels
  (`kernel_pad{_dtype}` / `kernel_shrink{_dtype}`)

### 3.2 Kernel fusion

Greedy elementwise fusion merges adjacent single-consumer elementwise ops into
`FusedElem` nodes that emit as single GPU kernels.

- the fusion pass in `chelis-ir/src/fuse.rs` is a DAG-to-DAG rewrite shared by all backends
- `RiscOp::FusedElem` carries `FusedStep`/`FusedStepOp`/`FusedInput` data
- elementwise→reduction fusion: when a `FusedElem`'s sole consumer is a reduction,
  the elementwise chain is inlined into the reduction kernel's inner loop
- the HIP emitter generates fused kernel source strings (register-chained computation)
- the C backend emits fused `#pragma omp parallel for` loops for both targets
- the evaluator decomposes `FusedElem` back to individual ops
- multi-consumer nodes are materialized and serve as external inputs to downstream chains
- `realize()` lowers to a real DAG materialization barrier and blocks fusion across it

### 3.3 Memory planning

- greedy slot reuse for non-overlapping storage lifetimes in `chelis-backend-hip/src/memory.rs`
- unique input copies are transferred once, with repeated `Load(name)` nodes aliasing the first copy
- planner-driven cleanup: every metadata wrapper is freed once, and every backing slot is freed once
- movement ops and `store` remain metadata aliases over the chosen backing slot
- kernel outputs iterate over logical element count (`d_t->size`), while input guard
  checks use backing `storage_size`
- HIP codegen reports a peak-memory formula plus an optional concrete estimate when
  every slot size is statically known
- `chelis build --target hip` prints the formula unconditionally and the concrete
  estimate when available
- the reported formula includes the inline staged-reduction scratch chains of §3.4

### 3.4 Reductions and hipBLAS

- segmented reductions use a single runtime-sized axis-specific kernel for the generic path
- fused elementwise→reduction kernels reuse that same runtime-sized segmented reduction path
- scalar contiguous reductions use a staged scratch-chain reduction with inline
  `hipMalloc`/`hipFree`, outside the slot planner of §3.3
- the peak-memory formula includes the worst single staged scratch chain alongside the
  slot-plan terms
- contiguous `f32` matmul subgraphs with rank ≥ 2 specialize to hipBLAS-backed helpers:
  rank-2 emits `chelis_hipblas_sgemm_row_major(...)`, while batched/symbolic matmul emits
  runtime-sized calls through `chelis_hipblas_sgemm_batched_row_major(...)`
- non-contiguous matmul-shaped DAGs fall back to the generic reduction path
- `chelis build --target hip` surfaces the required `-lhipblas` link flag when hipBLAS
  specialization is emitted

### 3.5 Symbolic dimensions

Symbolic dimensions use the stable tensor ABI on both the C and HIP backends:
generated functions bind symbolic names from input tensor metadata at runtime and
validate repeated occurrences across all participating inputs.

Symbolic shape normalization is name-stable and conservative: memory planners
canonicalize product ordering/associativity and arithmetic identities such as
`n * 1`, but they do not alpha-rename unrelated symbols. Alpha-renaming is valid
only for paths that carry explicit same-property `forall` or binder-equivalent
dimension identity.

### 3.6 Resource-region checks

The effect surface interacts with backend selection in two explicit ways:

- `chelis build --target c` admits only exact `with device("cpu") { ... }`
  host regions; every other device designator, including `cpu:<label>`, is
  rejected before artifact emission as specified by
  `spec/04-type-system.md` [04-EFF-2]
- `chelis build --target hip` rejects incompatible non-GPU resource regions

### 3.7 Coverage and GPU agreement

Operation and dtype coverage differs by target. `spec/04-type-system.md` §1.1.3
controls dtype admission, an operation a target does not support produces a target
diagnostic before emission, and `docs/CHELIS_SURFACE.md` §6 lists each target's
current coverage.

Each HIP GPU correctness case asserts that the HIP GPU result equals the `chelis-ir`
evaluator, including the `g16_pad_*` / `g16_shrink_*` movement cases. Those cases need
an AMD GPU, so they are manual gates; `docs/manual_gates.md` lists their commands.

## 4. Metal Backend

The Metal backend is the macOS-native GPU peer of the HIP backend. It is an
independent backend that mirrors HIP's architecture:

- host-side control remains in generated source, here Objective-C++ in `.mm` files
- GPU kernels are emitted as Metal Shading Language (MSL) source strings
- `[MTLDevice newLibraryWithSource:options:error:]` performs runtime compilation,
  the direct analog of `hiprtc` for the HIP path

Same ABI as the C and HIP backends:
`extern "C" void func_name(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out)`.

The Metal backend is **pure string emission** in Rust. The crate has zero macOS-only
Rust dependencies (no `metal-rs`, no `objc`); it builds, tests, and lints clean on
Linux, macOS, and Windows. Apple-SDK integration happens when the user runs
`clang++ -fobjc-arc -framework Metal -framework Foundation` against the emitted
`.mm`. This keeps `--target metal` available as a cross-compilation target and
mirrors how HIP works (no `hip-rs` dependency on the Rust side; the user runs `hipcc`).
`docs/manual_gates.md` records the dependency-tree check for this property.

Full design in `spec/design/chelis_metal_backend_plan.md`.

### 4.1 Crate structure and target admission

`crates/chelis-backend-metal/` is structurally a peer of `crates/chelis-backend-hip/`:
`src/{lib,emit,kernels,launch,memory,blas}.rs`, `runtime/chelis_metal_runtime.h`,
`tests/{codegen_structure,codegen_adversarial,gpu_correctness}.rs`.

`chelis build --target metal` dispatches alongside `--target c` and `--target hip`.
`pad`/`shrink` are typed MSL movement kernels. Per-dtype admit/reject decisions for
the Metal backend are pinned in `spec/04-type-system.md` §1.1.3 (the
per-backend dtype matrix); the CLI gate, the IR validation pass, and the
codegen entry point each consult that matrix. f64 is **hard-rejected** on
Metal with the FP64-ALU hardware diagnostic per §1.1.3; bf16 admits at
codegen but pipeline creation surfaces the Apple7+ requirement at runtime
on pre-Apple7 devices, also per §1.1.3.

### 4.2 Kernel emission

`MetalEmitter` mirrors `HipEmitter::emit_dag` (same passes: `collect_kernels`,
`emit_kernel_string_decl`, signature emission, `emit_input_shape_preamble`,
plan-driven slot allocation). MSL replaces HIP C/C++ syntax for kernel
declarations, thread indexing, buffer qualifiers, and math built-ins.
Reshape is metadata-only. `docs/CHELIS_SURFACE.md` §6 lists the operations the
Metal target admits.

### 4.3 Reductions

MSL reduction templates (sum/max/min) use threadgroup memory and tree
reduction, with two passes for arrays larger than one threadgroup. Fused
elementwise-into-reduction reuses the `reduction_inlined_fused_elems`
path in `chelis-ir`.

### 4.4 Matmul

A custom 16×16 tiled MSL matmul kernel needs no MPS dependency. The specialization
rule mirrors the HIP hipBLAS detection: rank-2 contiguous f32 matmul subgraphs
(`insert + mul + sum(axis=1)`) route to `chelis_metal_matmul_tiled`. bf16, i8, i16,
i32, and i64 matmul (where admitted by `spec/04-type-system.md` §5.7.2) route through
the parameterized 16×16 tiled MSL kernel. `chelis_metal_runtime.h` exposes MPS
wrapper helpers for f32 and f16 matmul under the ARC ownership model pinned in
`spec/04-type-system.md` §1.1.3 ("Metal runtime header: ARC vs MRC and MPS wrapper
ownership model").

### 4.5 Compile-and-link smoke

A Python smoke harness (`.github/scripts/smoke_macos_metal.py`) drives
`chelis build --target metal` on a fixed-shape elementwise program and then runs
`clang++ -std=c++17 -fobjc-arc -O2 ... -framework Metal -framework Foundation`
on the emitted `.mm`. The smoke is compile-and-link only:
`MTLCreateSystemDefaultDevice` may return null on hosted macOS CI machines, so kernel
dispatch is checked by the device gate of §4.6. Shard 2 of the `macos-workspace-shard` job in
`.github/workflows/macos-nightly.yml` runs it.

### 4.6 GPU correctness and numeric agreement

`tests/gpu_correctness.rs` marks every test `#[ignore]`. Each test invokes
`codegen_metal`, writes the emitted `.mm` to a temporary directory, drives `clang++`
against `-framework Metal -framework Foundation`, runs the resulting binary, and
compares the result with the `chelis-ir` evaluator.

The C backend is the numeric oracle for every GPU backend (§2). The Metal
numeric-agreement contract is [05-OBS-3]'s: for each kernel under test, every
Metal result has the same bits as the `chelis-ir` evaluator (which the C backend
is verified against). There is no Metal-specific absolute or relative tolerance.
A Metal kernel admits an operation only where it can produce those bits:
without fast math or contraction, with round-to-nearest-even and preserved
subnormals, and with every transcendental computed by a Chelis-owned correctly
rounded kernel. An MSL built-in, `precise::` included, is not one, because MSL
bounds its error rather than rounding correctly. MSL also permits a device to
flush f32 subnormals or round f32 arithmetic toward zero, and has no f64, so an
operation or dtype whose bits the device cannot guarantee is rejected under
[05-UNS-1], never approximated.
(The Metal and HIP lanes do not yet meet this contract; see
[chelis#2968](https://github.com/Chelis-Lang/chelis/issues/2968) and
[chelis#2969](https://github.com/Chelis-Lang/chelis/issues/2969).)
The `m6_pad_*` / `m6_shrink_*` cases (1-D and 2-D
pad/shrink, non-zero fill, and a pad→shrink roundtrip) mirror the HIP `g16_*`
cases one-for-one.

The gate needs an Apple Silicon Mac with a usable Metal device, so it is manual and
outside default CI; `docs/manual_gates.md` lists its command and success condition.

### 4.7 Adversarial tests

`tests/codegen_adversarial.rs` mirrors `crates/chelis-backend-hip/tests/codegen_adversarial.rs`.
Cases: zero-element tensors, rank-0 scalars, symbolic dims of 0/1, bool through
`where`, cast f32→bool→f32 round-trip, very large grids (>2^16 threadgroups),
single-element reductions, matmul with degenerate dimensions.

### 4.8 Target constraints

- f64 is **hard-rejected** on Metal because Apple Silicon GPUs have no FP64 ALUs;
  software emulation is out of scope. The diagnostic and rationale are pinned in
  `spec/04-type-system.md` §1.1.3. f64 workloads use `--target c` or
  `--target hip`.
- bf16 on Metal requires the Apple7+ GPU family (M3 or later); the kernel template
  guards `bfloat` on `__METAL_VERSION__ >= 320`, and runtime pipeline creation
  surfaces a clean diagnostic on M1/M2 devices. See `spec/04-type-system.md`
  §1.1.3 for both surfaces.
- `peak_device_bytes_formula` reports peak system RAM for tensor storage on Apple
  Silicon (no separate VRAM); the CLI prefixes the formula with a one-line note so
  users do not double-count

## 5. Integration Backends

Integration backends are additive:

### StableHLO

For TPU/XLA ecosystem access and ML compiler interop.

### FX

For PyTorch ecosystem interop, export, and execution through the FX / TorchInductor
toolchain.

These do not replace the C/HIP/Metal path.

## 6. Interactive Execution

Interactive execution is not a separate backend.
Tide and `chelis eval` use the IR evaluator first.
If interactive latency becomes a problem, the escalation order is:

1. cached C artifacts
2. persistent compiler helper
3. JIT only if measured workloads justify it

Chelis has no Cranelift-based backend.

## 7. Backend Selection

Native build command surface:

- `chelis build app.ch`
- `chelis build app.ch --target hip`
- `chelis build app.ch --target metal`

`chelis build` SHALL invoke the selected target's native compiler and produce an
executable when the checked root manifest requires an observation entry point, or
a static library without a process entry when it does not. Compilation SHALL use
the target's required support sources and compiler/linker flags. Every lane's
arithmetic, host and device alike, SHALL be compiled without implicit
floating-point contraction (`-ffp-contract=off` or the device compiler's
equivalent) and without any value-changing optimization: no fast math,
reassociation, flush-to-zero, or approximate reciprocal, square root, or
transcendental. The compile and link invocations are a function of the declared
target: the compiler SHALL NOT add flags derived from the build host's CPU (such
as `-march=native`), and SHALL run each native tool with an environment cleared
to a fixed allowlist, so that an ambient variable (for example `CFLAGS`,
`CCC_OVERRIDE_OPTIONS`, or `NIX_CFLAGS_COMPILE`) cannot change generated code. A
selected native compiler that does not honor this profile, including a wrapper
that injects flags, SHALL fail the build rather than produce an artifact. A
selected C compiler that compiles against another C library than the one the
carried runtime archive (§2.1) was built for SHALL fail the build before anything
compiles, with a diagnostic that names the archive's C library.
Emitted code computes every transcendental with the compiler-owned correctly
rounded kernels of [05-OP-46], never with the host math library, a vendor vector
library, or a compiler built-in. Executable
linking SHALL name the carried staged runtime archive by path (§2.1).

Every entry point through which other code runs compiled Chelis code, namely an
executable's process entry, an exported static-library function, and a binding
call, SHALL establish round-to-nearest-even with flush-to-zero and
denormals-are-zero disabled before any Chelis arithmetic, and SHALL restore the
caller's floating-point control state when it returns, a trap return included.
The evaluator establishes the same state on every thread on which it computes.
Every lane finalizes a NaN produced by floating arithmetic or conversion to
[04-NUM-2]'s canonical NaN, whatever default NaN or payload propagation its
hardware has.
Static libraries contain the compiled module and support objects; their consumers
link the carried runtime archive and the target's reported native dependencies.

`--emit-c` SHALL emit only the generated C-family sources, headers, runtime support,
and compile guidance, without requiring a native compiler or archiver. It applies
to C, HIP C++, and Metal Objective-C++ output. Both modes retain generated sources.
The output option selects the existing output directory or explicit source filename;
the executable uses that source stem and the library uses `lib<source-stem>.a`.

A native tool failure SHALL fail the build, name the failing tool and stage, and
preserve its diagnostics. Missing-tool diagnostics SHALL explain how to install
it. A failed build SHALL NOT replace a previously published native artifact;
success SHALL be reported only after the new native artifact exists.

Integration backends (§5) add further targets.

## 8. Invariants

All backends must preserve:

- the numeric semantics of `spec/04-type-system.md` §9: finalize at the declared dtype,
  compute at the arithmetic width [04-NUM-8] declares, trap per [04-NUM-9]/[04-NUM-10],
  and carry values without collapse per [04-NUM-11]
- bit-identical values across lanes and hosts: [05-OBS-3] grants no operation a
  nonzero bound, the transcendentals are correctly rounded under [05-OP-46], and no
  host math library, vendor vector library, compiler, flag, or floating-point
  environment is a source of variance (§7)
- named-dimension and precision semantics established before lowering
- agreement with the reference C backend on the shared test suite - a practical oracle
  for the invariants above, not a substitute for them
