# Chelis: Project Plan

## Overview

Build the Chelis programming language from zero to MNIST-on-CPU and beyond.
This plan is written for a small team working with coding agents.
Each phase has a concrete deliverable, verification target, and red-team checkpoint.

**Current status:** Phase 0 complete.
Phases 0a-0i complete.
Phases 1a-1f implemented.
Phase 1 is structurally complete for its shipped fixed-workload deliverable, with known
backend limitations carried forward explicitly rather than treated as hidden blockers.

**Repo:** `chelis-lang/chelis` (Rust workspace)
**Domain:** `chelis.ch`
**License:** MIT in the repo today

---

## Phasing

| Phase | Deliverable | Status |
|---|---|---|
| **0a** | Project scaffold, spec docs, CI, test infra | ✅ Complete |
| **0b** | Deep parser (s-expressions) | ✅ Complete |
| **0c** | Surf parser + Surf→Deep desugaring | ✅ Complete |
| **0d** | Type checker (ADTs, HM inference, precision, named dims) | ✅ Complete |
| **0e** | RISC DAG construction from typed AST | ✅ Complete |
| **0f** | C backend codegen (host + BLAS + OpenMP) | ✅ Complete |
| **0g** | `grad` transformation (reverse-mode AD on DAG) | ✅ Complete |
| **0h** | End-to-end: MNIST on CPU + spec test suite | ✅ Complete |
| **0i** | Tide v0.1 (REPL, `chelis deep`, `chelis surf`, `chelis fmt`, `chelis eval`) | ✅ Complete |
| **1** | Futhark-style GPU backend (HIP) + executable grammar (`chelis validate`) | Structurally complete with known limitations carried forward |
| **2** | Effects, linear types, macros, Tide Agent API + MCP, LSP, TUI (`chelis cove`) |  |
| **3** | Language completeness: pipe-first style pass, package system (Reef), Python FFI, direct execution, scalar/string foundation, collections/iteration, core numeric primitives, data loading/tokenization, `Std.Time`/`Std.Decimal`, SKILL.md v2 |  |
| **4** | ML & AI coding: seed corpus, ICL measurement, ChelisBench, trajectory collection, local model training, model integration |  |
| **5** | Advanced backends + research: StableHLO + JAX DLPack guarantee, FX Graph, Triton, multi-GPU, sparse tensors, complex numbers, research type features, Lean formalization |  |

**Red team checkpoints** after: 0a, 0d, 0h, and each major phase.
A red team round means adversarial review of design decisions, test coverage, spec
compliance, and architectural debt before the next phase proceeds.

---

## Phase 0 Snapshot

### 0a-0e Complete

The project already has:

- core numbered specs written and reviewed
- Deep parsing and canonical printing
- Surf parsing and deterministic desugaring
- type checking with named dimensions, precision tracking, and fitness-oriented errors
- lowering into the RISC DAG

The active question is no longer front-end design.
It is backend completion and end-to-end execution quality.

### 0f: C Backend

**Goal:** Emit correct, testable C from RISC DAGs and validate it numerically.

Current work:

- topological DAG emission into C
- BLAS integration for recognized linear algebra patterns
- OpenMP for elementwise and reduction loops
- memory planning and runtime support
- correctness testing against known numerical results

Acceptance criteria:

- generated C compiles cleanly
- core ops match reference outputs within tolerance
- BLAS and non-BLAS paths agree numerically
- memory behavior is validated with sanitizers or equivalent tooling

### 0g: Automatic Differentiation

**Goal:** Implement `grad` as a DAG-to-DAG rewrite.

Deliverables:

- adjoint rules for every primitive
- backward DAG construction
- finite-difference tests for primitive and composed gradients

Acceptance criteria:

- analytical and numerical gradients agree on the spec corpus
- common model-building blocks produce correct gradients

### 0h: End-to-End MNIST + Spec Test Suite

**Goal:** Prove the whole Phase 0 pipeline works.

Deliverables:

- MNIST training example on CPU
- executable spec test suite organized by language behavior, not crate
- numerical checks for parsing, type checking, codegen, and AD

Acceptance criteria:

- Surf source goes through the full pipeline to execution
- MNIST trains successfully on CPU
- the spec suite becomes the machine-checkable language baseline

Validation note:

- release runner: `cargo run --release -p chelis-e2e --bin train_mnist -- --epochs 5 --min-acc 0.90`
- measured result on the checked-in path: `0.9272` final test accuracy
- `cargo test --workspace` remains the fast regression suite; it does not, by itself,
  run the full long real-data MNIST milestone

### 0i: Tide v0.1

**Goal:** Ship the first interactive workflow.

Deliverables:

- `chelis tide`
- `chelis deep`
- `chelis surf`
- `chelis fmt`
- `chelis eval expr`
- evaluator-agreement tests against the C backend

Interactive execution policy:

1. use the IR evaluator first
2. add cached C artifacts if needed
3. add a persistent compiler helper if needed
4. consider JIT only if the first three fail on measured latency

---

## Phase 1

**Prerequisite:** Phase 0h complete.
**Deliverable:** GPU execution through a single HIP backend plus executable grammar
validation tooling.

**Current shipped boundary:** the HIP backend work through Phase 1e is in `main`.
That includes HIP code generation, fusion, device memory planning, segmented and staged
reduction paths, hipBLAS-backed specialization for contiguous rank ≥ 2 `f32` matmul, and the
fixed-workload benchmark oracle with checked-in results. `chelis validate` from 1f is
now shipped too. The fixed Phase 1e benchmark set (`mnist`, `linreg`,
`transformer_block`) compiles and runs on both backends, so the intended Phase 1
deliverable is met for the shipped models.

**Known carried-forward limitations:** these are real debt and must stay documented, but
they do not block Phase 2 language work.

- HIP codegen still does not implement `pad` / `shrink`; no current Phase 1 benchmark model uses them
- symbolic dimensions are implemented on the stable tensor ABI in both backends, so
  supported Phase 1 models bind batch/sequence-style dims from input metadata at runtime
- `layer_norm` still requires a concrete normalized-axis extent; symbolic leading dims
  are supported, but a symbolic hidden size remains follow-up debt
- the compiler emits dotted Deep module/import paths that `chelis validate` accepts, but the compiler-side Deep parser does not yet fully round-trip that emitted form

These limitations matter for future serving, dynamic batching, and full backend
generality, but they do not block the Phase 2 work on effects, linear types, macros,
and Tide tooling.

### 1a: Kernel Code Generation

- emit HIP kernel strings alongside generated host code
- compile kernels through `hiprtc`
- validate simple elementwise kernels first

### 1b: Fusion

- implement fusion rules as DAG rewrites
- prototype `egg` alongside hand-written heuristics
- adopt `egg` only if the fusion search space proves genuinely combinatorial

### 1c: GPU Memory Planning

- analyze lifetimes on device
- reuse buffers where safe
- minimize host/device transfer boundaries

### 1d: Optimized Reductions + hipBLAS

- ship segmented reductions with tiny/small/large strategy selection
- use staged scratch buffers for safe scalar contiguous reductions
- specialize contiguous rank ≥ 2 `f32` matmul patterns to hipBLAS-backed helpers
- keep irregular flattening/autotuning out of Phase 1d

### 1e: Benchmarks and Real Models

- benchmark fixed executable workloads already supported by the shipped surface
- current benchmark set: MNIST MLP, linear regression, transformer-block-style forward pass
- compare against the reference C backend for correctness
- keep PyTorch as a local/manual comparison dependency through the repo `py/` env, not a CI requirement
- target credibility, not premature parity with PyTorch

Phase 1e is now implemented through `chelis-e2e`'s `bench_phase1e` oracle and the
checked-in `benchmarks/results/latest.json` / `benchmarks/RESULTS.md` artifacts.

### 1f: Executable Grammar

- `chelis validate --surf file.ch`
- `chelis validate --deep file.dp`
- `chelis validate --desugar file.ch`

This remains the one valid `pest`-style parser-generator use case in the plan.
It is not a proposal to rewrite the Surf parser.

---

## Phase 2

**Prerequisite:** the shipped Phase 1 fixed-workload deliverable is in place.
**Deliverable:** language maturity features and interactive tooling.
The detailed implementation plan lives in `spec/design/chelis_phase2_plan.md`.

**Phase 2 deliverable statement:** the language is usable by researchers. Effects,
linear types, macros, the agent API, and tooling make Chelis a credible alternative to
PyTorch for specific workloads. A researcher should be able to write, type-check,
differentiate, compile, train, and debug a model - with AI assistance - using only the
Chelis toolchain.

**Carry-forward fixes before Phase 2 proper:** symbolic dimensions in both backends,
HIP `pad`/`shrink`, and the Deep dotted path round-trip gap. These are explicit debt
from the shipped Phase 1 boundary, not hidden blockers.

**Critical path:** 2a -> 2b -> 2c. The Tide tooling track (2e -> 2f -> 2g) can run in
parallel with the type-system track once 2e has enough compiler API surface.

### Sub-Phase Plans

| Sub-phase | Doc | Summary |
|---|---|---|
| Phase 1 carry-forward fixes | [chelis_phase2_plan.md](chelis_phase2_plan.md) | Symbolic dims, HIP `pad`/`shrink`, dotted Deep round-trip |
| 2a: Algebraic Effects | [chelis_phase2_plan.md](chelis_phase2_plan.md) | shipped subset: `Random` / `Resource(D)` boundary effects, `Diff` as capability, `Accum` internal-only |
| 2b: Linear Types | [chelis_phase2_plan.md](chelis_phase2_plan.md) | Lightweight uniqueness, borrowing, explicit `copy`, safe buffer reuse |
| 2c: Macro System | [chelis_phase2_plan.md](chelis_phase2_plan.md) | Hygienic expansion before all LLM-facing operations, provenance metadata |
| 2d: `vmap` | [chelis_phase2_plan.md](chelis_phase2_plan.md) | DAG rewrite for automatic vectorization with correct `grad` interaction |
| 2e: Tide Agent API + MCP | [chelis_phase2_plan.md](chelis_phase2_plan.md) | HTTP/JSON compiler service, MCP tools, batch interfaces |
| 2f: Tide LSP | [chelis_phase2_plan.md](chelis_phase2_plan.md) | Diagnostics, hover, completion, go-to-definition, Surf/Deep visibility |
| 2g: Tide TUI (`chelis cove`) | [chelis_phase2_plan.md](chelis_phase2_plan.md) | Terminal IDE, live checking, Surf/Deep toggle, agent mode |
| 2s: Seed Corpus | [chelis_phase2_plan.md](chelis_phase2_plan.md) | 50-100 programs collected throughout Phase 2 |

### 2a: Algebraic Effects

- `Diff` is a compiler capability, not a boundary effect
- `Random` and `Resource(Device)` are the user-visible Phase 2a boundary effects
- `Accum` is the design hook for parallelism-preserving gradient accumulation, but it
  remains internal-only in the shipped subset
- implementation split: effect types live in `chelis-types`; inference/checking lives in
  `chelis-effects`

### 2b: Linear Types

- tensor linearity
- borrowing for read-only access
- explicit and compiler-inserted copy/drop points
- compiler-enabled safe buffer reuse

**Phase 2b starting point:** begin with lightweight uniqueness / alias tracking rather
than a full Rust-style ownership-and-lifetimes model. The active `copy-drop` model is
specified in `spec/design/implicit_linearity.md`: source-level consuming fan-out is
made explicit with lowered `RiscOp::Copy` nodes, and unconsumed owners are closed with
lowered `RiscOp::Drop` nodes. Treat heavier borrowing machinery as an escalation only
if the lightweight model proves insufficient.

**GPU AD refinement note:** a later AD refinement can add recomputation-oriented backward
generation for GPU workloads, where replaying cheap forward work is often better than
storing every intermediate. The likely endpoint is selective checkpointing rather than
pure tape-only or pure full-recompute AD.

### 2c: Macro System

- hygienic Deep macros
- type-aware expansion hooks
- explicit phase separation

**LLM representation constraint:** The macro system must produce clean expanded Deep
with provenance metadata in the `{}` slot.
Macro expansion is a compilation step that happens before any LLM-facing operation.
The expanded form uses only the base 61-tag vocabulary.
LLMs never see, generate, or reason about unexpanded macro invocations.
This is a settled design decision, not an open question for Phase 2c.
The Phase 2c design task is: expansion rules, hygiene, phase separation, and the
provenance annotation format — not whether LLMs interact with macros (they don't).

### 2d: `vmap`

- DAG rewrite for automatic vectorization
- correct interaction with `grad`

### 2e: Tide Agent API + MCP

- HTTP / JSON compiler service
- MCP tool surface for coding agents
- streaming and batch interfaces where useful

### 2f: Tide LSP

- diagnostics, hover, completion, go-to-definition
- Surf↔Deep visibility inside editor tooling
- TextMate grammar (`.tmLanguage.json`) for instant Surf/Deep highlighting in VS Code
- LSP semantic tokens override TextMate with compiler-aware highlighting once server is ready

### 2g: Tide TUI (`chelis cove`)

- terminal coding environment
- live type checking
- Surf↔Deep toggling
- tree-sitter grammar (`grammar.js`) for incremental Surf/Deep highlighting in the TUI
- build and run flows without leaving the terminal

### 2s: Seed Corpus

- 50-100 Chelis programs collected throughout Phase 2
- serves as training data, compiler corpus, and documentation material
- stratified by complexity rather than filtered to one difficulty band

---

## Phase 3

**Prerequisite:** Phase 2 complete.
**Deliverable:** language completeness that makes Chelis self-sufficient for real AI
programs rather than only tensor compute kernels.
This phase is about closing the non-tensor gaps after the Phase 2 language surface is
stable: first-class scalar/string values, collections, iteration, core numeric
primitives beyond the initial RISC surface, data loading, tokenization, standard
library time/exact-decimal support, and the teaching material refresh that matches that
fuller language.
The detailed implementation plan lives in `spec/design/chelis_phase3_plan.md`.

**Shipped Phase 3 foundations:** `3a` package system, `3b` Python FFI interop,
`3b-ii` direct Python execution + NumPy guarantee, `3c` scalar/string foundation,
`3d` collections/iteration, and `3e` pipe-first style pass.

**Recommended execution order for remaining work:** `3h` → `3m` → `3g` → `3i` →
`3j-pre` → `3j` ∥ `3k` → `3l` → `3f`. `3j-pre` reserves the `chelis-lang` GitHub
organization, ships a compiler release binary, and expands the standard library surface
(`Std.Nn` attention/GELU/SiLU/RMSNorm/Conv, `Std.Loss`, `Std.Init`) that both `nautilus`
and `coral` depend on. `3j` (nautilus) and `3k` (coral) can then overlap — no mutual
dependency. `3l` (shoals) depends on both. `3f` (SKILL.md v2) is intentionally last in
Phase 3: it needs a real rewrite after the runtime, numeric, tokenization,
standard-library, and domain-shell surfaces stabilize.

### 3e: Style Foundation

Shipped. This remains the stylistic foundation for the remaining Phase 3 work:
pipe-first Surf, short-form block bindings, width-aware multiline layout, and
human-readable examples. All new Phase 3 language-completeness examples should continue
to follow this idiom.

### 3a: Package System (Shells + Reef)

Shipped.

- `reef.toml`
- `reef.lock`
- local-first Reef registry
- `.chb` shell metadata plus source archives for downstream builds
- bounded package-aware `chelis check` / `chelis build`
- dogfood the system by shipping `chelis-std` as a Reef package through the same
  shell/import pipeline users rely on
- keep `Std.Io.Safetensors` as a package/API stub in `3a`; land its runtime
  implementation in `3b`

### 3b: Python FFI

Shipped.

- DLPack tensor exchange with PyTorch as the guaranteed target in this cut
- CPU-only DLPack guarantee in `3b`; GPU tensor ownership/allocator work is deferred to
  `3b-ii`
- PyO3 compiler bindings
- shared `chelis-compiler-api` extraction so Tide and Python share one compiler
  implementation
- GIL release during compiler/evaluator work
- zero-copy tensor handoff as the default interop goal where the runtime permits it
- safetensors interop completing the `3a` API stub at the Python/runtime layer
- install surface: `uv pip install ./bindings/python`
- `chelis.eval(...)` is allowed to copy into the evaluator's internal representation in
  this cut; zero-copy compiled execution belongs to `3b-ii`
- `ChelisError` remains the compiler/build/runtime failure type; Python-side bad inputs
  such as unsupported GPU tensors in `3b` are `ValueError`
- explicit scope boundary: direct Python-callable execution and NumPy guarantee are
  deferred to `3b-ii`; JAX DLPack guarantee is deferred to `5a`
- serve both incremental-ML interop and Python-side compiler/tooling automation

### 3b-ii: Direct Python Execution + NumPy Guarantee

Shipped.

- `chelis.compile_and_load("model.ch")` is the primary path; `chelis.load("model.so")`
  is the advanced path for prebuilt artifacts
- load compiled Chelis artifacts into Python and call them directly, e.g.
  `compiled = chelis.load("model.so")`
- add the runtime loader, calling-convention adapter, and Python-side shape/dtype
  validation needed for direct execution
- emit a sidecar manifest with source path + content hash so `load()` can warn about
  stale artifacts
- add NumPy to the DLPack guarantee surface and document the copy vs zero-copy rules
- release the GIL during native compile/build and compiled host/device execution
- keep compiled execution on fully concrete `f32` tensors in this cut
- reuse `3b`'s PyO3 and DLPack infrastructure rather than broadening `3b` itself

### 3c: Scalar and String Foundation

Shipped.

- add first-class unrestricted `Int`, `Float`, and `Bool` values outside tensors
- add first-class immutable `String` values for file paths, tokens, labels, config
  keys, and logging
- add scalar `if/else` as the natural control surface for boolean decisions
- add practical built-ins around these values: arithmetic, comparison, `%` / `mod`,
  named integer bitwise helpers, formatting, parsing, and tensor shape queries
- make `Option[T]` part of the practical language surface for failure-returning APIs
- add `print` / `debug` as the minimum IO-based debugging surface
- compile scalar/string/`Option`/print programs through a host-value lane in both C and
  HIP builds instead of leaving them evaluator-only
- tag Tide/Python execution values by runtime type on the wire surface
- keep tensor scalars distinct from host-language scalar values; conversions stay
  explicit

### 3d: Collections and Iteration

Shipped.

- add immutable `List[T]` and `Dict[K, V]` as first-class collection types
- add iteration primitives such as `map`, `filter`, `fold`, `zip`, `enumerate`, and
  `range`
- define effect propagation through iteration so list-processing code composes with
  `Random`, IO, and later effects
- add the boundary between variable-length collections and fixed-shape tensors:
  list/tensor conversion, stacking, and `pad_sequences`
- compile collection code through the `3c` host-value lane in both C and HIP builds
  rather than leaving list processing evaluator-only
- current shipped slice covers compiled collection foundations: list literals,
  `List[T]`, `len`, `index`, `append`, `concat`, `take`, `drop`, `chunk`,
  `flatten`, `range`, `zip`, `enumerate`, numeric `to_tensor`, practical rank-1
  `to_list`, `pad_sequences`, and the first immutable `Dict[K, V]` builtins
  (`dict_of`, `dict_get`, `dict_contains`, `dict_remove`, `dict_insert`,
  `dict_merge`, `dict_keys`, `dict_values`, `dict_entries`), plus compiled
  higher-order iteration (`map`, `filter`, `fold`, `scan`, `partition`,
  `flat_map`); callback effects now propagate through iteration under the shipped
  checker, and this compiled helper set now covers the practical collection surface
  needed before `3g` data loading/tokenization work starts
- make preprocessing and dataset plumbing expressible in pure Chelis rather than Python

### 3h: Core Numeric Primitives

- expand the post-RISC practical numeric surface with `einsum`, `concat`, `split`,
  `gather`, `scatter`, `where`, `cumsum`, `sort`, `diagonal`, `trace`, and `clamp`
- treat `einsum` as the highest-leverage single addition because it subsumes matmul,
  batched matmul, transpose, trace, outer products, and common contractions under one
  primitive
- land `concat` / `split` as mandatory model-building tools for skip connections,
  multi-head attention, and tensor packing/unpacking flows
- land `diagonal` / `trace` and `clamp` as the minimum practical linear-algebra and
  training-control additions
- keep `Std.Nn.Embedding` explicit in the standard library even though it is a thin
  wrapper over `gather`, because it is the natural public entrypoint for NLP models
- treat this as the last major tensor-language expansion before the data/token pipeline
  becomes the critical path
- require `chelis check` to reject deterministic literal-driven `3h` value errors
  (such as static `einsum` extent mismatches or duplicate `scatter(..., "replace")`
  indices) before build; when the bad value is only known at runtime, compiled C should
  exit non-zero rather than aborting
- acceptance oracle: `cargo test -p chelis-cli phase3h_numeric_acceptance_oracle -- --nocapture`
  which covers both the compiled tensor-structural example and the Reef
  `Std.Nn.Embedding` package-import build path

### 3m: Rust Runtime Rewrite

- replace the growing `chelis_runtime.c` implementation with a Rust static library in
  `crates/chelis-runtime`
- take the ABI-cleanup-now path: `chelis_tensor` remains layout-visible, while
  `chelis_string`, `chelis_list`, `chelis_tuple`, and `chelis_dict` become opaque
  handles with explicit retain/release and accessor APIs
- update generated host C to stop peeking into host-value fields directly; use runtime
  accessors plus explicit ownership operations instead
- keep tensor kernels and evaluator semantics unchanged; this is a compiled-runtime
  contract cleanup, not a language-semantics phase
- make `chelis build` emit and reference `libchelis_runtime.a` plus `chelis_runtime.h`
  instead of copying `chelis_runtime.c`
- runtime discovery order for `chelis build`: `CHELIS_RUNTIME_DIR`, then path relative
  to `current_exe()`, then a hard actionable error
- block all remaining runtime-heavy Phase 3 work on this rewrite so `3g`/`3i`/shells
  land on Rust infrastructure rather than the old C runtime
- acceptance oracle: `cargo test -p chelis-cli phase3m_rust_runtime_acceptance_oracle -- --nocapture`

### 3g: Data Loading and Tokenization

- add text file I/O as the minimum host-data ingress surface with IO effect
  (`read_file`, `write_file`, `read_lines`, `read_bytes`, `file_exists`, `list_dir`)
- add memory-mapped I/O (`mmap_file`, `mmap_read`, `mmap_len`) backed by Rust runtime's
  memmap2 for large datasets — IO effect on open, pure reads after that, byte-range
  access returned as `List[Int]`
- add CSV loading returning `List[Dict[String, String]]` for tabular data, with
  `try_read_csv` as the recovery API
- add JSON loading returning `Json` for configs and metadata, with `try_load_json` and
  `try_parse_json` for recovery paths
- add a BPE tokenizer that loads HuggingFace `tokenizer.json` format and can
  encode/decode text; tokenizer loading returns `Tokenizer`, with
  `try_load_tokenizer` for recovery paths
- add batch encode + padding flows that bridge `List[List[Int]]` into tensor model
  inputs
- make the tokenizer/data-loader path a first-class Phase 3 deliverable, not a Python
  sidecar
- acceptance oracle: `cargo test -p chelis-cli --test phase3g_io phase3g_text_pipeline_acceptance_oracle -- --ignored --exact --nocapture`

### 3i: Standard Library Expansion

Standard library modules for real model training and inference:

- **`Std.Time`:** Date and duration types. Date arithmetic, comparison,
  formatting/parsing (ISO 8601). UTC only in v1.
- **`Std.Decimal`:** Fixed-point exact arithmetic. Configurable precision, banker's
  rounding. Host-value type, not tensor dtype.
- **`Std.Nn.Generate`:** Autoregressive generation with KV cache management. Greedy and
  sampled generation (temperature, top-k, top-p) via record-config APIs. `KVCache` is
  precision-polymorphic so mixed-precision inference does not force an `f32` cache
  boundary. The shipped pure greedy entrypoint is `generate`; sampled `generate_with`
  currently runs under `with seed(...)`. This is the core inference pattern for
  generative models — eliminates the need for manual fold-based generation loops.
- **`Std.Optim` (expansion):** AdamW (decoupled weight decay, the standard transformer
  optimizer), LAMB (large-batch training).
- **`Std.Schedule`:** Learning rate scheduling — cosine annealing with warmup, linear
  warmup, step decay. Pure `(step, config)` functions using record configs rather than
  positional constructor calls.
- acceptance oracle: `cargo test -p chelis-cli --test phase3i_std -- --ignored --nocapture`

### 3j-pre: Release Infrastructure + Std Surface Expansion

Prerequisite gate for both `nautilus` and `coral`. Not itself a shell.

- confirm and build out the `chelis-lang` GitHub organization at
  <https://github.com/Chelis-Lang> (org already reserved)
- ship a compiler release binary through CI so downstream shell repos can pin a
  toolchain version
- expand `chelis-std` with the additions both domain shells will assume:
  - tensor construction and reductions: `linspace`, `arange`, `stack`, `squeeze`,
    `unsqueeze`, `min`, `prod`, `argmax`, `argmin`
  - `Std.Nn`: `GELU`, `SiLU`, `RMSNorm`, `Conv1d`, `Conv2d`,
    `scaled_dot_product_attention`, multi-head attention, grouped-query attention
  - `Std.Loss`: `KLDivergence`, `BCEWithLogits`, `accuracy`, `perplexity`
  - `Std.Init`: `kaiming_uniform`, `kaiming_normal`, `xavier_uniform`, `xavier_normal`,
    `trunc_normal`
- acceptance oracle: `cargo test -p chelis-cli --test phase3j_pre_std -- --ignored --nocapture`

### 3j: Nautilus — Numerical Methods, Statistics, and Optimization

A reef package. The scipy competitor for Chelis. The name references the chambered
nautilus — nature's logarithmic spiral, mathematical precision. Depends on `chelis-std`
+ 3h primitives + `3j-pre`. Pure Chelis where natural; **nalgebra** as the linear
algebra backend (pure Rust, BLAS/LAPACK when available, no Fortran dependency). Because
nalgebra calls are opaque to the Chelis AD system, `Nautilus.LinAlg` ships with
hand-written adjoint rules for SVD, Cholesky, solve, QR, and eig (same pattern
`torch.linalg` uses). `Nautilus.Signal` ships as a typed stub (blocked by complex
numbers, Phase 5f).

Status update: `Nautilus v0.1.0` is now published as the first downstream shell
release, using `chelis v0.1.7`.

Modules ship in three priority tiers.

**P0 — ship first:**

| Module | Contents |
|---|---|
| `Nautilus.Special` | `erf`, `erfinv`, `log_gamma`, `digamma`, `beta` — required by Distributions |
| `Nautilus.Distributions` | Normal, LogNormal, Uniform, Student-t, Chi-squared, Exponential, Gamma — PDF, CDF, inverse CDF, sampling (`Random` effect). `normal_like` is Box-Muller here. |
| `Nautilus.LinAlg` | SVD, PCA, eigendecomposition, Cholesky, QR, LU, solve, inverse, determinant — nalgebra-backed with hand-written AD adjoints |

**P1:**

| Module | Contents |
|---|---|
| `Nautilus.Stats` | Descriptive statistics, correlation, covariance, shrinkage estimators |
| `Nautilus.Optim` | Convex optimization solvers (QP, SOCP, LP). Differentiable optimization via KKT. NOT `Std.Optim` (neural network optimizers). |
| `Nautilus.Roots` | Root finding (Newton-Raphson, bisection, Brent) |
| `Nautilus.ODE` | ODE solvers (Euler, RK4, adaptive step). Composes with `grad` for neural ODE support. |

**P2:**

| Module | Contents |
|---|---|
| `Nautilus.SDE` | SDE solvers (Euler-Maruyama, Milstein). Uses `Random` effect. |
| `Nautilus.Integrate` | Numerical integration (trapezoidal, Simpson's, Gaussian quadrature) |
| `Nautilus.Interpolation` | Linear, cubic, spline interpolation |
| `Nautilus.Testing` | Hypothesis testing, confidence intervals, p-values |
| `Nautilus.Distance` | Euclidean, cosine, Mahalanobis, Manhattan distances over tensor rows |
| `Nautilus.Signal` | Signal processing (FFT, STFT, filtering). **Stub — blocked by complex numbers (Phase 5f).** |

### 3k: Coral — Typed Dataframes

A reef package. Numeric columns are tensors on the lazy RISC DAG (GPU-accelerable,
fusible — a filter→mutate→aggregate pipeline on numeric columns may compile to one fused
kernel). String columns are host-side lists (eager). AD flows through dataframe
operations (filter → gather, aggregation → reduction) — sensitivity analysis no existing
dataframe library supports. Depends on `chelis-std` + 3h primitives (gather, scatter,
argsort). Single-machine, no query optimizer — does NOT compete with Polars/DuckDB query
planning or Spark distributed processing.

Current Chelis ground truth for Coral startup is now pinned in tests: ADT-based columns,
tensor-valued `Dict`s, bool `gather`, and `Std.Tensor.Construct.arange` all work today.
The standard Phase A index-extraction path is `Std.Tensor.Mask.where_indices`, a host-lane
helper built from `to_list`/`enumerate`/`filter`/`map`/`to_tensor`. Known residual:
zero-match masks still hit the existing empty-`to_tensor([])` limitation and need a
downstream special case until empty tensor construction lands.

| Module | Contents |
|---|---|
| `Coral.Frame` | Core DataFrame type, column selection, row filtering, sorting by column, mutation, `rename`, vertical `concat`, `describe` (via `Nautilus.Stats`), `value_counts`. **NaN handling is built into `Coral.Frame`, not a separate module:** `is_nan`, `fill_nan`, `drop_nan`, `any_nan`, `count_nan`. Float columns use IEEE 754 NaN; integer columns use a companion boolean mask. |
| `Coral.GroupBy` | Group-by via argsort + segmented scatter, aggregation per group |
| `Coral.Join` | Sort-merge and hash joins on typed key columns |
| `Coral.Reshape` | Pivot, melt, stack/unstack |
| `Coral.Window` | Rolling operations over numeric columns: `rolling_mean`, `rolling_sum`, `rolling_std`, `ewm` |
| `Coral.IO` | DataFrame-aware CSV/JSON loading, typed column auto-detection, **Parquet I/O via the Rust `parquet2` crate in the runtime** |

Persistent column dictionary: the internal column map uses a persistent data structure
(HAMT) with structural sharing, so frame operations like `with_column` and
`drop_column` produce new frames sharing column references with the original. Required
for practical AD through frame pipelines — without structural sharing,
`grad(fn_with_many_frame_ops)` copies the entire column map per intermediate frame. Full
rationale and the pure-Chelis-vs-Rust-runtime decision point are covered in
`chelis_coral_design_spec.md`.

### 3l: Shoals — Finance

A reef package. Depends on `chelis-std` (`Std.Time`, `Std.Decimal`) + `nautilus` +
`coral`. Contains only finance-specific logic — nothing a non-finance programmer would
need. Greeks via `grad` for free. Reproducible Monte Carlo via `Random` effect. Typed
market data via named tensor dimensions.

Shoals ships with Chelis-native tests from day one (`tests/*.ch`, run via `chelis test`). Python parity only if comparing against QuantLib or other external pricing references.

**Key design decision: instruments as dicts, not closed ADTs.** Financial instruments
are open-ended — structuring desks invent new payoff formulas continuously. Representing
instruments as `Dict[String, f32]` (or `Dict[String, Column]` for term structures) lets
new instrument types be added as data without modifying the Shoals source or releasing
a new package version. The pricing function dispatches on a key (e.g., `get(instrument,
"type")`), not on a pattern match over a closed enum. This also serves the AI coding
story: an agent generating a new instrument definition writes a dict literal (well
within current LLM capability), not a new ADT variant (which requires understanding the
type system's extension points).

| Module | Contents |
|---|---|
| `Shoals.Pricing` | Black-Scholes, Heston, SABR, Monte Carlo engines. Greeks via `grad`. |
| `Shoals.Risk` | VaR, CVaR, expected shortfall, stress testing |
| `Shoals.Curves` | Yield curve construction, bootstrapping, day count conventions |
| `Shoals.Stochastic` | SDE discretization, path generation (uses `cumsum`), variance reduction |
| `Shoals.Orderbook` | Limit order book representation, matching logic (host-side collections) |

**Reproducibility manifests.** A compiler pass (`chelis manifest`) extracts all
`Random`-effect-annotated operations from the typed AST into a structured JSON report:
which operations introduce randomness, which seed handlers cover them, and whether the
computation is fully reproducible. `chelis manifest --check` exits 0/1 for CI gating.
This is a product feature for model-validation teams ("machine-generated certificate
that your simulation is reproducible"). Now demo-blocking for regulated-finance
prospects: the compile-time guarantee on the Random effect is shipped, the CLI tool
that produces the audit artifact is the missing piece. Full design:
`chelis_manifest_spec.md` (current concrete spec); historical context in
`chelis_reproducibility_manifests.md`. Ships alongside or shortly after Shoals.

**Trust stack integration.** Shoals ships with example `@property` annotations for
standard pricing models (put-call parity, delta bounds, price positivity,
reference-implementation correspondence). These demonstrate the
executable-properties-as-spec pattern to finance customers — properties are the
artifact the customer reviews; `chelis prove` enforces that the optimized
implementation satisfies them. See `chelis_trust_stack.md`.

**Canonical domain properties.** Shoals ships with a `properties/` directory
containing reference `@property` functions for standard finance invariants: put-call
parity, price positivity, delta/gamma bounds, Monte Carlo convergence, no-arbitrage
conditions on spreads. These are onboarding tools (run `chelis prove src/` to see them
pass), credibility artifacts (the shell verifies itself), and templates for
customer-written properties. Convention: every domain shell ships canonical properties
alongside its implementation code. Properties are NOT a separate shell — they are
co-located with and version-controlled alongside the functions they constrain.

**Reference implementations.** Shoals ships `references/` containing simple
textbook-formula implementations of standard models: Black-Scholes call/put with
Greeks, Heston, Vasicek, CIR, vanilla Monte Carlo pricer, standard portfolio risk
measures. These are co-located with the production implementations in `src/`.
Customer pattern: customer writes proprietary models in their own packages, uses
Shoals references for standard models, writes properties that include
`@property matches_reference forall(...)`. Full design:
`chelis_reference_implementations_spec.md`.

### 3n: Octant — LaTeX ↔ Deep Bridge (Part A)

A reef package. The notation bridge between quant-finance LaTeX and Chelis Deep.
Depends on `chelis-std` + `nautilus`; **not** `coral` or `shoals`. Runs **in
parallel with `3l`** because it has the same prerequisite (`3j` green) and needs
nothing from Shoals. Octant is a notation adapter, **not** a CAS — no symbolic
integration, no simplification, no equation solving. The intelligence is in the
coding model (for the LLM-assisted lowering path that lands in `3o`) and the
compiler (for type verification and provenance tracking). Full design in
`chelis_octant_design.md`; executable sub-phase contract in `phase3n_octant.md`.

Octant uses Python (sympy / latex2sympy2) as the external oracle for LaTeX parsing correctness: parse the same LaTeX in both Octant and sympy, compare expression trees. That is a parity test against an external oracle — same pattern as Nautilus vs scipy. Chelis-native tests cover tokenizer correctness, parser crash safety, provenance span accuracy, and parse→pretty-print round-trips.

| Module | Contents |
|---|---|
| `Octant.Parse` | LaTeX subset parser for quant-finance expressions — arithmetic, unary, powers/roots, transcendentals, special functions (`erf`, `\Phi`, `\Gamma`, `B`), derivatives, integrals, sums/products, piecewise, matrix notation. Out-of-scope LaTeX (TikZ, `\begin{theorem}`, prose) produces diagnostic errors, not silent drops. |
| `Octant.Symbolic` | Thin ~30-node `SymExpr` AST mapping directly to Chelis RISC primitives and Nautilus library calls. |
| `Octant.Lower` (deterministic path) | SymExpr → Deep for every form the LaTeX uniquely determines. Special functions → `Nautilus.Special`/`Nautilus.Distributions`; integrals → `Nautilus.Integrate`; matrix ops → `Nautilus.LinAlg`. |
| `Octant.Render` | Deep → LaTeX with type overlays (named tensor dims → subscripts, effect markers, `grad` → partial-derivative notation). |
| `Octant.Provenance` | Source-span annotations on **every** Deep node produced by Octant lowering. Core value proposition — without the audit trail the first round-trip ships a parser, not a product. |

**Canonical properties.** Octant ships with `properties/roundtrip.ch` (parse-render
round-trip: `parse(render(expr))` recovers the original expression and
`lower(parse(latex))` type-checks) and `properties/provenance.ch` (every Deep AST node
has a valid source span pointing inside the original LaTeX string). Same convention as
Shoals: properties co-located with implementation, run via `chelis prove src/`, never
packaged as a separate shell. See `chelis_trust_stack.md`.

### 3o: Octant — Finance Notation + Notebook (Part B)

A reef package extension. Adds finance-notation lowering through `shoals`, Greek
rendering patterns, and the `Octant.Notebook` cell runtime. Depends on
`chelis-std` + `nautilus` + `shoals` + `3n`. Runs **after `3l`**. Provenance
extends to cover the new node kinds using the `3n` contract.

| Module | Contents |
|---|---|
| `Octant.Lower` (LLM-assisted path) | SDE notation → `Shoals.Stochastic` (discretization + time grid + noise), Monte Carlo expectation → `Shoals.Pricing` (variance reduction + `Random` effect), calibration → `Nautilus.Optim`, yield curve → `Shoals.Curves`. |
| `Octant.Render` (finance additions) | Greek pattern matches — `grad(price, wrt=spot) → Δ`, `grad(price, wrt=vol) → 𝒱`, `grad(price, wrt=rate) → ρ`, `grad(price, wrt=T) → Θ`. Configurable variable-name conventions. |
| `Octant.Notebook` | Cell-based environment — formula, parameter, execution, Greek cells. NOT a Jupyter kernel; cells produce Deep, execution runs compiled C, rendering is mathematical notation. |
| `Octant.Provenance` (extension) | Same contract as 3n, applied to the new SDE / MC / calibration / curve node kinds. |

### Post-Phase-3 Shell Stubs

Three further shells (and one Octant sub-scope) are named and reserved but scoped
as stubs beyond Phase 3. They are listed here so the ecosystem story is explicit,
but no Phase 3 sub-phase implements them.

- **`school`** — classical ML (scikit-learn competitor). Regression, decision trees,
  SVMs, clustering, pipelines, cross-validation. Depends on `chelis-std` + `nautilus` +
  `coral`.
- **`darwin`** — evolutionary algorithms. Genetic algorithms, genetic programming over
  the Deep AST, evolution strategies, population-based training, neural architecture
  search. Uniquely natural fit because Deep is homoiconic: program mutation and
  crossover are typed AST operations, and the compiler's 0–1 fitness score is literally
  the fitness function for evolutionary search. Requires `nautilus`; optionally uses
  `coral` for evolving feature-engineering pipelines over tabular data.
- **`hull`** — executable language specification. Depends on `chelis-std` only. A
  self-hosted executable specification shell: the specification of Chelis, written in
  Chelis, checked by Chelis, used to test Chelis. The marine name: the hull defines the
  shape of the vessel; the spec defines the shape of the language.

  Hull tests are entirely Chelis-native. Hull tests ARE Chelis programs testing Chelis — no Python involvement.

  Hull implements the LaCaDiLE typing rules and operational semantics directly as
  Chelis functions over an ADT representation of the Deep AST. The grammar becomes ADTs
  (`type Expr = Var(String) | App(Expr, Expr) | Add(Expr, Expr) | ...`). The typing
  rules become a reference type checker (`type_check(ctx, expr) -> Option[(Type,
  List[Effect])]`) where each pattern-match arm corresponds to one typing rule from the
  paper. The reduction rules become a step function (`step(expr) -> Option[Expr]`).
  Because Deep is homoiconic (programs are data), Hull consumes actual Chelis programs
  parsed from Deep source strings — no translation layer and no bridge to an external
  tool.

  Three capabilities:
  1. **Reference type checker.** Given a Deep program, type-check it according to the
     formal rules and compare the result against `chelis check`.
  2. **Reference evaluator.** Given a well-typed Deep program, evaluate it by iterated
     small-step reduction and compare the result against `chelis eval`.
  3. **Spec-driven test generation.** Generate random well-typed Deep programs using
     the typing rules as constraints, giving conformance tests for free.

  The self-referential property is the key value: the Chelis compiler type-checks
  Hull's reference type checker, and Hull's reference type checker type-checks Chelis
  programs. The compiler validates the spec; the spec validates the compiler. This
  closes the spec-implementation gap without requiring an external tool or a full
  extraction from the Lean mechanization.

  Prerequisites: LaCaDiLE typing rules finalized, a Deep parser in Chelis, and reusable
  property-runner generation infrastructure for language-level random generation. Phase
  4 or Phase 5 item — not actively developed yet. Recorded because it is the natural
  answer to "how do you know the compiler implements the spec?" and because it is a
  strong future-work story for the OOPSLA paper.
- **`octant-docs`** — Octant Phase 4 (from the design doc). Parses full LaTeX model
  documents, extracts `\begin{equation}` environments, and associates formulas with
  surrounding prose so model-validation teams can ingest an entire model document as
  a unit. Parser scope is significantly larger than the quant-finance expression
  subset shipped in `3n`. Depends on the full `3n`/`3o` Octant surface.
- **`beacon`** — automated static analysis (`chelis-lang/beacon`). Depends on
  `chelis-std` and the compiler's tensor DAG IR. A specialized analysis tool that
  computes over-approximations of value ranges at each DAG node. Detects division by
  zero, overflow, NaN propagation, and unbounded output without user annotations. The
  user specifies input ranges; Beacon infers output ranges and flags hazards.
  Computationally expensive (minutes, not milliseconds) — a pre-deployment gate, not
  an inner-loop tool. Inspired by Astree (Airbus A380 flight control verification).
  Future shell, not designed. Marine rationale: a lighthouse warning of hazards. Full
  design pointer: `chelis_trust_stack.md` (Level 3 of the trust stack).

### 3t: Chelis-Native Testing

**Dependencies:** Bug 9 fix (eval hang on reef imports), fast `chelis eval` with package-aware imports.

**Deliverables:**
1. `Std.Test` module in chelis-std — assertion functions (`assert_eq`, `assert_close`, `assert_close_tensor`, `assert_true`, `assert_false`, `fail`), either via a `Test` algebraic effect or runtime builtin.
2. `chelis test` CLI command — discovers `tests/*.ch` files, evaluates each via the evaluator (not build+gcc+link+run), calls every `def test_*()` function, reports pass/fail with structured output, exits 0/1.
3. Test file convention — `tests/*.ch` in every reef package, `def test_*()` naming, documented layout.
4. Nautilus test migration — mathematical identity tests, property tests, edge case tests, smoke tests move from Python to Chelis. Scipy parity tests remain in Python under `parity/`.
5. Coral test migration — same split: structural correctness tests move to Chelis, pandas parity stays in Python.
6. SKILL.md for `Std.Test` — documents assertion API for coding agents.

**Hard rule:** Only code that compares Chelis output against an external oracle (scipy, pandas, QuantLib) uses Python. All other tests are written in Chelis and run via `chelis test`. This applies to all reef packages, current and future.

Full design: `chelis_native_testing_plan.md`

### 3f: SKILL.md v2

Full-surface teaching refresh covering Phase 2 + Phase 3 including domain shells:
effects, linearity, macros, vmap, tuples, pipes, scalars, strings, collections,
iteration, I/O, tokenization, core numeric primitives, package imports, dataframes
(`coral`, including NaN handling and Parquet), numerical methods (`nautilus`, including
the nalgebra-backed LinAlg surface), finance (`shoals` overview), and the expanded
`Std.Nn` surface from `3j-pre` (attention, GELU/SiLU, RMSNorm, Conv1d/2d). Mentions the
`school` (classical ML) and `darwin` (evolutionary algorithms) shells as post-Phase-3
stubs. Goes truly last. Validated via `skill_suite.rs`.

Phase 3 success condition:

- a pure Chelis program can read text, tokenize it, batch and pad it, run a model,
  compute loss and gradients, and print results without dropping to Python
- the package, Python, and style foundations already shipped in `3a`, `3b`, `3b-ii`,
  and `3e` remain valid while the language grows beyond tensor-kernel scope
- domain shells (`nautilus`, `coral`, `shoals`, `octant`) build and import through the Reef
  pipeline, composing correctly on top of `chelis-std`
- `octant` parses the quant-finance LaTeX subset, lowers deterministic expressions
  to Deep, and round-trips Black-Scholes through the compiler with provenance
  annotations intact on every emitted Deep node; SDE / Monte Carlo / Greek
  rendering and the notebook ship in `3o` on top of `shoals`
- `SKILL.md` and examples match the fuller language including domain shells rather than
  the earlier tensor-compute-only subset

---

## Phase 4

**Prerequisite:** Phase 2 complete and the remaining Phase 3 language-completeness work
through `3h`, `3m`, `3g`, `3i`, `3j-pre`, `3j`, `3k`, `3l`, and `3f` complete.
**Deliverable:** Chelis ships with a local coding model as standard tooling and a
reproducible benchmark proving the "designed for LLMs" thesis. The turtle carries its
home.

### Pre-Phase 4 Investments

**Verified error messages with per-property explanations.** When the compiler rejects a
program, the diagnostic should explain which property the rejection protects and
suggest a fix. Not "type mismatch on line 42" but "mul requires dimension-wise
equality: your first operand has dimensions [batch, hidden] but your second has
[hidden, batch]. Did you mean permute(b, [1, 0])?" This is the single highest-leverage
compiler investment for the AI story: when the RLVR training loop generates a program
that fails type checking, the quality of the error message IS the reward signal's
informativeness. "Type error" gives the agent nothing to work with. "Dimension
mismatch: expected [batch, hidden] got [hidden, batch] in mul at position 2" gives the
agent enough information to repair the program. Better error messages = faster RLVR
convergence = better coding model. Implementation: improve diagnostics in
`crates/chelis-types/` to include the specific rule violated, the expected vs actual
types/dims/effects, and a repair suggestion where possible. Does not require Lean — the
existing type checker has the information, it just doesn't format it well.

**Structured fitness score with per-property components.** Break the 0-1 fitness score
into per-property components returned in the fitness JSON: dimension score, effect
score, linearity score, differentiability score, syntax score. An agent generating a
program sees which property failed and can focus its repair on that specific issue
rather than re-generating from scratch. The aggregate fitness score is still computed
(for RLVR reward), but the components are exposed for agent introspection and for the
trajectory-collection step (4c) to record which properties are hardest for the model.
Implementation: extend the fitness JSON output in `crates/chelis-cli/` to include a
`components` object alongside the aggregate `score`. The type checker already computes
these properties independently — the change is formatting, not analysis.

**Fast `chelis eval` with package-aware imports.** The RLVR training pipeline runs
thousands of Chelis programs and scores them via compiler fitness. If every evaluation
requires `chelis build` → gcc → link → run, the training loop is bottlenecked by
compilation (seconds per program). If `chelis eval` resolves reef package imports and
returns results in milliseconds, the RLVR reward signal is fast enough for online
training. Investment: make the evaluator resolve imports from installed reef packages,
optimize startup time, and handle the standard library without filesystem round-trips.
This also serves agent-driven development (sub-second feedback on whether a generated
function produces the right value). Target: `chelis eval myfile.ch --expr "erf(0.5)"`
completes in under 200ms with Nautilus imported. `chelis test` is a thin layer on top of fast eval: discover test files, evaluate each, collect assertion results. The fast eval investment directly enables native testing infrastructure.

**Stability labels on shell APIs.** Mark each exported function in every SKILL.md as
`stable` or `alpha`. The training pipeline's corpus-curation step (4a) uses these
labels: stable functions are safe to include in training data and will not break
between releases; alpha functions are excluded or down-weighted because their
signatures may change. Convention: add a `Stability` column to the API surface tables
in each shell's SKILL.md. Nautilus v0.1.0 candidates for `stable`: all of Special, all
of Distributions (pdf/cdf/inv_cdf). Candidates for `alpha`: CurveFit, SDE (API may
change when autonomous Random sampling lands). Apply the same convention to Coral and
Shoals when they ship.

**`chelis prove` — executable properties as spec.** The earlier pre-Phase 4 framing of
`chelis prove` as a CLI-flag property tester is now subsumed by the broader
executable-properties-as-spec design (first-class `@property` Chelis functions,
type-directed input generation, counterexample minimization). See the expanded
`chelis prove` section below and `chelis_trust_stack.md` for the full design.

### Why This Is a Distinct Phase

Phase 2 planned a seed corpus (2s) that was supposed to accumulate organically during
development. In practice, the programs written during Phase 2 are compiler test fixtures,
not curated training data. Building a real corpus and training a real model requires
dedicated ML effort — data curation, experiment design, training runs, evaluation — that
doesn't fit as a side effect of compiler development.

### 4a: Seed Corpus Collection and Curation

Build a corpus of 50-100 Chelis programs that serve as: training data for the local
model, few-shot examples in the SKILL.md, evaluation anchors for model quality, and
documentation for users.

**Stratification (informed by the Twist et al. complexity research):**
- ~20 single-operation programs (1-3 ops, baseline structural competence)
- ~40 single-layer programs (5-15 ops, one model component)
- ~30 multi-layer programs (15-40 ops, composition of components)
- ~10 full models (40+ ops, end-to-end training pipelines)

**Each program ships as:**
- `corpus/NNN_name.ch` — Surf source (pipe-first style)
- `corpus/NNN_name.dp` — Canonical Deep (pretty-printed)
- `corpus/NNN_name.json` — Fitness score, type info, effect info

**Coverage requirements:** Basic tensor ops, MLP forward/backward, pattern matching on
ADTs, dimension polymorphism, pipe-heavy data flow, effects (Random, Resource),
linearity (copy, borrow), macros, vmap, tuple returns, grad with multiple wrt targets,
PyTorch-equivalent translation pairs.

**Quality gate:** All programs compile, type-check with fitness >= 0.9, use idiomatic
pipe-first style, and exercise the full Phase 2 language surface.

### 4b: ICL Effect Measurement

The prerequisite experiment before any training investment.

**Protocol:** Run the SKILL.md eval on 2-3 target base models (Qwen 35B, Llama 4, etc.)
with and without the spec in context. Measure the delta in compiler fitness scores. If
putting the spec in context measurably improves output quality (fitness score increase
> 0.2), distillation-based methods (SDFT) are viable. If the delta is small, skip
distillation and use standard LoRA.

**Additional measurement (informed by Vera's de Bruijn research):** test generation
accuracy with named-Deep vs a positional-reference Deep variant. If positional references
measurably improve generation accuracy, consider adopting them for the LLM-facing
representation. If not (hypothesis: Deep's closed vocabulary already provides sufficient
structural constraint), document the result and keep named references.

**Cost:** Free. API calls or local inference. But it gates the entire training strategy.

### 4c: Trajectory Collection via Compiler Loop

- collect 2K-4K compiler-in-the-loop trajectories
- include failures, repairs, and complexity metadata
- use the compiler as the scoring/teaching surface rather than free-form human labels

### 4d: ChelisBench

A reproducible benchmark measuring whether LLMs write better ML code in Chelis than in
PyTorch. Inspired by Vera's VeraBench methodology — same tasks, multiple languages,
multiple models, comparable evaluation.

**Structure:** 50 ML programming tasks across 5 difficulty tiers:

| Tier | Examples | Count |
|---|---|---|
| 1: Single ops | "Implement ReLU," "Compute mean along axis 1" | 10 |
| 2: Layer components | "Write a linear layer with bias," "Implement softmax cross-entropy loss" | 15 |
| 3: Model blocks | "Write a transformer attention block," "Implement a residual connection with layer norm" | 12 |
| 4: Full models | "Write a 2-layer MLP for MNIST," "Implement a single transformer encoder layer" | 8 |
| 5: Training pipelines | "Write a training step with gradient computation and parameter update," "Implement per-example gradients with vmap" | 5 |

**Evaluation:** For each task, evaluate on 3-6 models (Qwen, Llama, Claude, GPT, etc.)
in two modes:
- **Chelis mode:** SKILL.md in context, generate Chelis (Surf or Deep), evaluate with
  `chelis check` fitness score + `chelis test` property tests + `chelis eval` numerical
  correctness
- **PyTorch mode:** Standard PyTorch documentation in context, generate Python, evaluate
  with syntax check + execution + numerical correctness

**Metrics per task:** compile/run success rate, fitness score (Chelis only), iterations
to first correct program, final correctness, wall-clock generation time.

**The thesis test:** If Chelis is well-designed for LLM authorship, models should achieve
comparable or better success rates on Chelis tasks than PyTorch tasks, despite zero
Chelis training data. The SKILL.md + compiler feedback loop should compensate for the
training data gap.

**Dual purpose:** ChelisBench is both a measurement tool AND a trajectory source for
Phase 4c. Every benchmark run produces model-generated programs with compiler feedback —
usable as training data.

**Publication target:** Workshop paper or blog post. "ChelisBench: Do LLMs Write Better
ML Code in a Language Designed for Them?"

### 4e: Local Coding Model Training

The training method is determined empirically. The hard constraint is
**anti-forgetting**: the model must preserve its PyTorch/JAX semantic knowledge.

**Decision protocol:**
1. LoRA on trajectories + seed corpus + SKILL.md examples. Measure fitness scores AND
   forgetting on a PyTorch comprehension benchmark.
2. If fitness good AND forgetting minimal → ship.
3. If forgetting measured → replace LoRA with SDFT (same data, on-policy, KL
   regularization prevents drift).
4. If fitness plateau → add RLVR with compiler fitness as continuous reward.

**Target:** >0.95 fitness score on 80%+ of generated programs, including Deep generation
and Deep repair tasks.

**Ship:** GGUF quantization (Q4_K_M) for consumer hardware (8GB VRAM). Published as
`chelis-lang/chelis-coder` on HuggingFace.

### 4f: Coding Model Integration

Wire the local model into the toolchain:
- `chelis model pull` downloads the GGUF weights
- `chelis cove --assist` loads the local model for inline completions
- Tide MCP server's `chelis_generate` tool uses the local model when available, falls
  back to API models when not
- No API key, no internet connection required for the default assist experience

**Distribution summary:**

| Artifact | Track | Phase | Purpose |
|---|---|---|---|
| `chelis-lang/chelis-skill` (SKILL.md v2 + examples + harness) | Track 1 | Phase 3f | Frontier model in-context learning |
| `chelis-lang/chelis-bench` (ChelisBench tasks + evaluation harness) | Measurement | Phase 4d | Benchmark: do LLMs write better ML code in Chelis? |
| `chelis-lang/chelis-trajectories` (trajectory dataset) | Track 2 | Phase 4c | Training data for local model |
| `chelis-lang/chelis-coder` (GGUF quantized model) | Track 2 | Phase 4e | Ships with toolchain |

Phase 4 success condition:

- Chelis ships a first-party local coding model as part of the product, not as an
  optional research extra
- ChelisBench provides reproducible evidence for the "designed for LLMs" thesis

### `chelis prove` — Executable Properties as Spec

Evolution of the originally planned CLI-flag property testing tool into Chelis's core
verification mechanism for AI-generated code. Properties are first-class Chelis
functions with `@property` annotations, living alongside implementation code,
version-controlled, CI-enforced.

Three property categories:

1. **Domain invariants** — things that must always be true: output is non-negative,
   delta is between 0 and 1, portfolio weights sum to 1.
2. **Spec correspondence** — the optimized implementation matches a simple reference
   implementation on random inputs. The reference IS the spec. Differential testing is
   a property, not a separate tool.
3. **Behavioral constraints** — monotonicity, continuity, symmetry, convergence. Domain
   knowledge encoded as executable checks.

`chelis prove src/pricer.ch` discovers `@property NAME forall(...)` declarations,
generates type-directed random inputs, tests each property, and reports the first
deterministic counterexample. CI integration via `chelis prove` as a gate alongside
`chelis test`.

This is the highest-ranking new value prop for consequential computing customers. The
customer writes properties that define correctness. The toolchain verifies the
generated code satisfies them. The customer reviews properties (simple, one-line domain
facts), not generated code (complex, optimized, opaque).

Value for the AI story: sampled properties become part of the RLVR reward signal. A
generated function that passes a deterministic property corpus without NaN is higher
quality than one that only passes fixed golden tests. The reward function can
incorporate `chelis prove` results as a continuous reward component alongside the 0-1
fitness score.

Value for the numerical library story: catches edge cases that golden fixture grids
miss. The Nautilus bessel_y1 drift in (7.5, 8) would have been caught by property
sampling before it became a documented known limitation.

**Convention: domain shells ship canonical properties.** Every shell that targets a
specific domain (Shoals for finance, Octant for LaTeX, future vertical shells)
includes a `properties/` directory with reference `@property` functions, and (where
the domain has standard textbook models) a co-located `references/` directory with
simple, obviously-correct reference implementations. These are the onboarding entry
point for new users (install the shell, run `chelis prove src/`, see the canonical
properties pass), the credibility proof that the shell's implementations satisfy
standard domain invariants, and the spec artifact for the spec-correspondence
property category. The properties and references are co-located with the
implementation, NOT packaged separately — a standalone "properties" shell with no
implementation code is an empty vessel.

**Priority elevation.** `chelis prove` with first-class `@property` annotations is now
demo-blocking for the first commercial CProof prospect. The trust stack pitch lives
or dies on this being demonstrable. Specification is concrete (see
`chelis_property_spec.md`) and the build is a focused project, not a research
investigation. Components: parser changes for `@property NAME forall(...)`,
type-directed input generation, deterministic first-counterexample reporting, Deep
bridge metadata discovery, and the CLI subcommand. Move ahead of secondary-priority
Phase 4 prep work.

**`chelis manifest` priority elevation.** The compile-time guarantee on the Random
effect is shipped. The CLI tool that produces the audit artifact is now demo-blocking
for regulated-finance prospects. Specification is concrete (see
`chelis_manifest_spec.md`). Build the subcommand and JSON output format. CI
integration via `chelis manifest --check` as a gate.

Full design: `chelis_trust_stack.md`, `chelis_property_spec.md`,
`chelis_manifest_spec.md`, `chelis_reference_implementations_spec.md`.

---

## Phase 5: Advanced Backends + Research

**Prerequisite:** Phase 1 HIP backend mature, Phase 2 language stable.
**Deliverable:** Chelis targets additional hardware platforms beyond CPU and AMD GPU.
Research-grade type extensions and mechanized type theory produce publications.

Items ordered by likely demand. All are additive — the HIP backend remains the primary
GPU target.

### 5a: StableHLO Backend

- **Direct emission:** RISC DAG → StableHLO operations. Following the Nx/EXLA pattern
  (not via JAX tracing).
- **TPU access:** The primary motivation. StableHLO is the only serious path to Google
  TPUs.
- **Alternative GPU path:** StableHLO → XLA → GPU code. Useful for comparison against the
  HIP backend.
- **JAX DLPack guarantee:** JAX's `jax.dlpack.from_dlpack()` has device placement
  semantics that interact with JAX's lazy evaluation and XLA compilation — more complex
  than PyTorch or NumPy interop. Slot here alongside StableHLO because at this point
  Chelis and JAX share a compilation target and the interop story is richer than tensor
  exchange alone.

### 5b: FX Graph Backend

- **RISC DAG → FX operator graph:** Map the ~12 primitives to ATen operators.
- **TorchInductor:** FX graphs compile via TorchInductor to Triton kernels (NVIDIA),
  C++/OpenMP (CPU), ROCm (AMD).
- **torch.export → ExecuTorch:** Edge deployment path.
- **Use case:** Interop with PyTorch ecosystem. A Chelis model can be exported as a
  PyTorch module.

### 5c: Triton Backend

- **RISC DAG → Triton IR:** Emit Triton code (or Triton Python via codegen) instead of
  raw HIP.
- **Access Triton's optimization passes:** Memory coalescing, shared memory staging,
  warp-level primitives — without implementing them in the Chelis compiler.
- **Helion as higher-level target:** Optionally emit Helion code and let Helion's
  autotuner handle kernel-level optimization.
- **Use case:** NVIDIA GPU performance without writing CUDA. Cross-vendor via Triton's
  AMD support.

### 5d: Multi-GPU Data Parallelism

- **Data parallelism requires no language changes.** The orchestration layer calls the
  compiled artifact on each GPU with different batch shards, then all-reduces gradients
  via RCCL.
- **Model parallelism** would require a `transfer` primitive, device-aware DAG
  partitioning, and RCCL integration. Scope TBD based on demand.
- **Pipeline parallelism** is an orchestration concern handled externally (DeepSpeed,
  FSDP, etc.).
- **Priority:** Low. Chelis targets single-GPU workloads for Phases 1-4. Multi-GPU
  becomes relevant only for models that don't fit in one GPU's memory.

### 5e: Sparse Tensor Support

Core numerical infrastructure that benefits ML (sparse attention, graph neural networks,
sparse rewards in RL) and numerical computing (large correlation matrices, sparse linear
systems). Not domain-specific.

- **Sparse tensor types:** `sparse_tensor[m, n, f32, CSR]` /
  `sparse_tensor[m, n, f32, COO]`. Named dimensions carry through.
- **Sparse operations:** Sparse matmul (SpMM, SpMV), sparse-dense element-wise ops,
  sparse reduction, format conversion (dense↔CSR↔COO).
- **Backend:** hipSPARSE/rocsparse integration for GPU, reference C implementation for
  CPU.
- **AD:** Sparse adjoints for sparse matmul and element-wise ops. Gradients through
  sparse operations produce sparse gradients.
- **Effort:** large. Significant backend addition.

### 5f: Complex Number Support

Native `complex64` / `complex128` tensor dtypes with correct AD (Wirtinger derivatives).

- **Operations:** Complex arithmetic, conjugate, magnitude, phase, real/imag extraction.
- **FFT:** Built on complex tensors. Forward and inverse FFT as Tier 2 built-ins.
- **Use cases:** Spectral methods, signal processing, Fourier-based pricing methods,
  frequency-domain analysis.
- **Effort:** small. Dtype addition + RISC op implementations + Wirtinger AD rules.

### 5g: Research Type Features

Moved from Phase 3. Publication-grade type system extensions — each should be a paper
before it's an implementation.

- **Rank polymorphism via ILP elaboration:** AUTOMAP-style, insert `expand` operations
  during type inference via integer linear programming.
  No-implicit-broadcasting guarantee preserved.
- **Size-dependent types:** Futhark-style syntactic dimension equality with dynamic
  coercion fallback.
- **Distribution types:** For probabilistic models.
  `Distribution(Normal, {mean: tensor, std: tensor})`. Sampling is `Random` effect.
- **Equivariance constraints:** Track symmetry groups through composition. Most
  novel/publishable.
- **Optimization properties:** `@convex`, `@lipschitz(1.0)`. Trusted annotations
  initially.
- **Inference as a typed effect:** LLM calls as a typed, mockable algebraic effect
  (inspired by Vera). Research direction.

### 5h: Mechanized Type System (Lean 4)

Moved from Phase 3. Formalize Chelis's core type system in Lean 4. Prove type soundness.
Publication target: POPL/ICFP/PLDI. The Lean formalization doubles as an executable
reference type checker — the ultimate conformance oracle.

---

## Differentiable Programming Track (Phases D0–D6)

Parallel committed-scope track for end-to-end differentiable programming. Detailed
design in `spec/design/differentiable_language.md`; roadmap row in
`spec/12-roadmap.md` §Differentiable programming.

Scope is committed; the strategic decision on when to begin Phase D1 is separate.
The phases are sequenced by dependency: D1 → D2 → D3 → D4 → D5, with D6 free to
develop in parallel with D5 once D4 lands.

| Phase | Deliverable | Notes |
|---|---|---|
| **D0** | Spec lock at `spec/design/differentiable_language.md` | ✅ Complete |
| **D1** | AD through `if`, `match`, `while`, `for`, recursion. C/HIP backends gain `RiscOp::Select` and ADT-tagged match lowering; reverse-mode loops emit trajectory storage. | Depends on `IR-SelectOp-F1` and `IR-MatchLowering-F1` (`docs/gap_synthesis.md` §5). Committing reclassifies them from surface-when-forced to required. |
| **D2** | Field-wise gradient ADTs/records and higher-order function AD. | Depends on `IR-FirstClassFn-F1`. |
| **D3** | Effect-aware AD: pure / `state` / `raises` / `capability` / `sample` (reparam, REINFORCE, pathwise). Unlocks Chelis-as-PPL substrate. | Composes with the Phase 2a effect work; does not add new effects, just gradient rules per effect. |
| **D4** | Implicit differentiation via `fix`, `argmin`, `solve` markers (IFT-derived gradients, KKT for `argmin`). | Differentiable simulation / optimization-as-a-layer use cases. |
| **D5** | Type-level differentiability: `Differentiable`, `PartiallyDifferentiable`, `NonDifferentiable` annotations + inference. Composes with the existing property-verification harness. | Connects to `spec/design/chelis_property_spec.md` for gradient-behavior properties (e.g. `Lipschitz(K)`). |
| **D6** | Primer, `examples/differentiable/`, shell library `chelis-diff` (distributions, optimizers, IFT helpers, finite-difference checks). | On-ramp for PyTorch/JAX users. |

Pre-locked decisions (forward + reverse mode parity, AD-as-transformation,
type-level differentiability, effect-driven gradient discipline, marked implicit
differentiation, field-wise ADT gradients, backward compatibility with today's
`grad`) are listed in the canonical document and are not subject to per-dispatch
re-litigation.

What is explicitly *not* in this track: distributed AD, JIT-style per-input
recompilation, mixed-precision AD (lives with the broader dtype build-out),
higher-order AD beyond what falls out of forward∘reverse composition, AD across
FFI boundaries. See `spec/design/differentiable_language.md` §What's not in scope.

---

## Hydronnx Shell Track (Phases H0–H5)

Parallel committed-scope track for Hydronnx, the Chelis shell that consumes ONNX
model files and exposes them as typed, callable Chelis functions. Detailed
design in `spec/design/hydronnx.md`; roadmap row in `spec/12-roadmap.md`
§Hydronnx.

`Hydronnx` (PascalCase) is the module path; `hydronnx` is the crate, package,
file, and prose form. The upstream `ONNX` interchange format is not owned by
Chelis — Hydronnx is the consumer, not the format. The two names are kept
distinct everywhere they appear.

Scope is committed; the strategic decision on when to begin Phase H1 is
separate. Phasing is sequenced by dependency: H1 → H2 → H3 → H4, with H5 free
to develop in parallel with H4.

| Phase | Deliverable | Notes |
|---|---|---|
| **H0** | Spec lock at `spec/design/hydronnx.md` | ✅ Complete |
| **H1** | ONNX protobuf parser, internal IR (`OnnxModel`/`OnnxGraph`/`OnnxNode`/`OnnxTensor`), `chelis-hydronnx-inspect` CLI utility. Inventory matches `onnx.checker.check_model` baseline. | Single-agent dispatch; standalone milestone. |
| **H2** | Operator translator over the v0.1 core subset: tensor manipulation, elementwise arithmetic, comparisons, logical, reductions, matrix, activations, normalization, convolution, decomposed Attention/MultiHeadAttention/RotaryEmbedding, Cast, Constant/ConstantOfShape. | Decomposition recipes for high-level operators are pinned in the spec so future IR-specializer recognizers know the canonical tag tree to match. Per-operator and per-category numerical agreement vs ONNX Runtime. |
| **H3** | Weight loading (ONNX TensorProto → Chelis tensor), layout conversion, dtype conversion, Chelis function emission with provenance metadata. `load_model` / `inspect_model` / `load_model_with_opts` API surface. | Single-agent dispatch; closes the end-to-end loading capability. |
| **H4** | Type-discipline integration: dimension types on loaded signatures, call-site type checking, property attachment, AD composition where operators support it, composition with other Chelis code. | Composes with `spec/design/chelis_property_spec.md` and the existing hull infrastructure. |
| **H5** | Documentation, examples per strong-fit model category, property examples, honest performance framing, ONNX-Runtime → hydronnx migration guide. | Discovery and onboarding milestone. |

Initial dtype scope (`f32`, `f64`, `i32`, `i64`) matches the host-side dtype
build-out. Operator coverage extensions (`bf16`, `f16`, `i8`, `i16`,
quantized formats) ship as the dtype build-out delivers host-side precision.

Dependencies on other Chelis workstreams are deliberately non-blocking where
possible:

- The `IR-FirstClassFn-F1` / `IR-SelectOp-F1` / `IR-MatchLowering-F1` §5 entries
  in `docs/gap_synthesis.md` extend Hydronnx's operator coverage (If, Loop,
  Scan) when they close; v0.1 explicitly excludes those operators.
- The differentiable-language Phase D1 control-flow AD work unlocks AD through
  dynamic-graph operators once those operators load.
- Fusion, kernel authoring, and MLIR-backend work are performance enhancers
  that loaded models pick up automatically — no Hydronnx changes required.

What is explicitly *not* in v1 scope: custom ONNX operators, ONNX Training
graphs, dynamic-graph operators (If, Loop, Scan), quantized formats
(QDQ / Q8 / Q4), multi-device or distributed inference, Chelis → ONNX
round-trip export. See `spec/design/hydronnx.md` §What's not in v1 scope.

---

## Ecosystem Library Decisions

Evaluated via multi-agent review.
These are settled decisions, not open prompts.

### Adopted / Planned

| Library | Phase | Purpose |
|---|---|---|
| `egg` | Phase 1b (prototype) | Equality saturation for kernel fusion. Prototype alongside hand-written heuristics — adopt if fusion space is genuinely combinatorial, skip if greedy heuristics suffice. |
| `salsa` | Phase 2+ (adopt) | Incremental/demand-driven compilation for Tide API, LSP, and AI agent loops. Design for it now (pure function crate boundaries), adopt when interactive use cases materialize. |
| `ariadne` or `miette` | Phase 1+ (evaluate) | Rich diagnostic rendering for fitness reports and error messages. Higher ROI than parser replacement for improving compiler UX. Evaluate when fitness scoring UX is prioritized. |

### Rejected

| Library | Reason |
|---|---|
| `logos` | Lexer works, bugs are fixed, nested block comment handling requires hybrid approach that dilutes the declarative benefit. Revisit only if lexer maintenance becomes a recurring cost. |
| `chumsky` | Parser works, error recovery can be added incrementally to existing Pratt parser. Full rewrite is high-cost, low-marginal-gain. If partial parse recovery is needed for fitness scoring, add recovery points at declaration boundaries in existing code. |
| `petgraph` | Custom DAG is small, specialized, append-only-by-construction, and backed by `verify.rs`. Generic graph API makes compiler-specific structural mutations less ergonomic, and index invalidation on node removal is a footgun. |
| `cranelift` | Speculative second backend with high maintenance cost. The IR evaluator (`eval.rs`) handles interactive execution. If that's too slow, cached C compilation and persistent helper processes are cheaper solutions. No JIT unless measured latency justifies it. |

### Architectural Discipline (Salsa Readiness)

To keep a future `salsa` migration mechanical rather than conceptual:

- each compilation stage is a pure function
- no global symbol tables are mutated across invocations
- crate APIs take inputs and return outputs

---

### Future Compiler Optimizations (recorded, not scheduled)

**Lazy list fusion on the host lane.** The tensor DAG already fuses elementwise operations on tensors (the main performance path). The host lane (lists, strings, dicts) is eager — `map(filter(map(list, g), pred), f)` creates two intermediate lists. A compiler optimization pass could recognize chains of `map`, `filter`, `fold` on lists and fuse them into single-pass traversals, eliminating intermediate allocations. This is the same idea as GHC's `foldr/build` fusion or Clojure's transducers, implemented as a compiler rewrite rather than a user-facing API. Low priority because the performance-critical path is tensors, not lists. Becomes relevant if Coral's string column operations or Hull's AST processing show list allocation as a bottleneck in profiling.

**SIMD codegen improvements.** Four-level plan for improving CPU vectorization in the C backend. Full design: `spec/design/chelis_simd_plan.md`.

- **Level 1 (small, ship with next codegen pass):** `restrict` pointer annotations on all tensor parameters (leverages linearity — the type system proves no aliasing, justifying `restrict`), `const` on input pointers, aligned allocation in the runtime, compiler-appropriate SIMD pragmas (`#pragma omp simd` for gcc, `#pragma clang loop vectorize(enable)` for Apple clang). Unlocks auto-vectorization on loops currently skipped due to aliasing.
- **Level 2 (medium, when profiling shows need):** Hand-written SIMD reduction kernels in the runtime (`sum`, `max`, `min`, `argmax`, `argmin`). AVX2 implementations for x86, NEON for ARM, scalar fallback. Reductions are where auto-vectorization is weakest.
- **Level 3 (medium, before OOPSLA benchmarks):** Vectorized math library integration. Sleef on Linux (both x86 and ARM), Accelerate vForce on macOS (already linked). The emitter maps math ops in fused kernels to SIMD-width library functions (`expf` → `Sleef_expf8_u10` on AVX2, `vvexpf` via Accelerate on Mac). Highest impact: 3-5x additional speedup on math-heavy kernels (erf, normal_cdf) on top of existing fusion wins.
- **Level 4 (large, only if Levels 1-3 leave gaps):** Full SIMD-width-aware codegen as a parallel emit path. Diminishing returns if Level 3 handles math functions. Record as future option.

---

## Design Work Pipeline

Design and implementation run in parallel.
The rule is simple: a spec document must be written and reviewed before the phase that
depends on it.

### Completed Design Work

| Design Task | Document | Status |
|---|---|---|
| Surf formal grammar | `spec/02-surf-syntax.md` — full PEG, keywords, precedence, desugaring table | ✅ Complete (consumed by Phase 0c) |
| Deep formal grammar | `spec/03-deep-syntax.md` — tag vocabulary, 3-tuple node structure, canonical form, PEG | ✅ Complete (consumed by Phase 0b) |
| RISC primitive semantics | `spec/05-risc-primitives.md` — ops, types, AD adjoints, lowerings | ✅ Complete (consumed by IR check) |
| Type system formal rules | `spec/04-type-system.md` — HM inference, tensor algebra, precision rules, fitness scoring | ✅ Complete (consumed by Phase 0d) |
| Standard op lowerings | Included in `spec/05` | ✅ Complete |
| Deep tag vocabulary | explicit `app` / `var` / `lit`, closed structural set | ✅ Settled |
| Deep metadata format | universal `(tag {} children...)` shape | ✅ Settled |
| Surf module system | one module per file, explicit import/export | ✅ Settled |
| Named dimension syntax | module-level `dim`, function-level dimension parameters | ✅ Settled |
| Pipe semantics | first-class `pipe` node in Deep | ✅ Settled |
| Tensor type syntax in Deep | precision last, dimensions explicit | ✅ Settled |
| Block / sequencing syntax | braces, newline and semicolon compatibility | ✅ Settled |
| Pattern matching details | nested patterns, guards, record punning, exhaustiveness | ✅ Settled |
| Type aliases | transparent aliases with dedicated Deep tag | ✅ Settled |
| Record update | syntax settled; implementation later | ✅ Settled |
| Transform syntax | call-like Surf syntax, dedicated Deep tags | ✅ Settled |
| Error message catalog | structured compiler errors with repair direction | ✅ In progress (iterative) |

### Remaining Design Work

| Design Task | Produces | Consumed By | Priority |
|---|---|---|---|
| **Effect system design** | effect typing rules, handler syntax, HM interaction; investigate Dex's Accum effect for parallelism-preserving gradient accumulation, and distinguish parallelism-preserving effects from sequentializing ones | Phase 2a | **HIGH** |
| **Linear type design** | linearity rules, borrowing rules, effect interaction | Phase 2b | **HIGH** |
| **Macro system design** | expansion rules, hygiene, phase separation, provenance annotation format (`{source: ...}` metadata key), interaction with the 61-tag vocabulary constraint (macros cannot introduce new tags) | Phase 2c | **MEDIUM** |
| **Fusion rules** | DAG fusion constraints and correctness conditions | Phase 1b | **MEDIUM** |
| **GPU memory model** | device-memory semantics and ownership model | Phase 1 / 2a | **MEDIUM** |
| **Effect handler syntax** | Surf and Deep syntax for handling effects | Phase 2a | **MEDIUM** |
| **Borrow syntax** | Surf and Deep borrowing forms | Phase 2b | **LOW** |
| **Custom effects** | user-defined effect extensibility model | Phase 2a | **LOW** |
