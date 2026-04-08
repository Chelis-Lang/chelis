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
| **3** | Ecosystem foundations: package system (Reef), Python FFI, research type extensions, Lean formalization, pipe-first style pass |  |
| **4** | ML & AI coding: seed corpus, ICL measurement, trajectory collection, local model training, SKILL.md v2, model integration |  |
| **5** | Advanced backends: StableHLO, FX Graph, Triton, multi-GPU |  |

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
reduction paths, hipBLAS specialization for contiguous rank-2 `f32` matmul, and the
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
- specialize contiguous rank-2 `f32` matmul patterns to hipBLAS
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
- explicit copy points
- compiler-enabled safe buffer reuse

**Phase 2b starting point:** begin with lightweight uniqueness / alias tracking rather
than a full Rust-style ownership-and-lifetimes model. Chelis's fixed tensor primitive
surface and DAG-based execution may allow the simpler rule of thumb that a tensor
consumed by an op is dead unless it is explicitly `copy()`'d. Treat heavier borrowing
machinery as an escalation only if the lightweight model proves insufficient.

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
The expanded form uses only the base 59-tag vocabulary.
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
**Deliverable:** ecosystem foundations that make Chelis usable by external users.
This phase is about packaging, interop, polished examples, and research-facing
extensions after the Phase 2 language surface is stable.

### 3a: Package System (Shells + Reef)

- `reef.toml`
- Shell publishing and dependency resolution
- Reef registry

### 3b: Python FFI

- DLPack tensor exchange
- PyO3 compiler bindings
- GIL release during Chelis execution
- zero-copy tensor handoff as the default interop goal where the runtime permits it

### 3c: Research Type Extensions

- investigate ILP/AUTOMAP-style rank-polymorphism support that inserts explicit `expand`
  operations during inference while preserving Chelis's no-implicit-broadcasting rule
- evaluate size-dependent types and refinement-style constraints as research extensions,
  not baseline language commitments
- treat these as publication-grade extensions layered onto the stable Phase 2 language,
  not prerequisites for the core toolchain

### 3d: Lean Integration

- mechanize the core type system only
- prove soundness for the stable Phase 0 core

### 3e: Pipe-First Style Pass

This is not a cosmetic formatting tweak.
It is a decompiler behavior change plus a full corpus rewrite for idiomatic Surf.

Target outcome:

- the decompiler defaults to pipe-first Surf for linear tensor/dataflow chains
- executable examples are rewritten out of assembly-like `let` ladders and into the
  stable idiomatic Phase 2 style
- `SKILL.md` and related teaching material point at the same idiom the docs and
  examples use

Decompiler rule:

- emit a pipe chain when a binding is used exactly once as the first argument to the
  next call, and that result is again used exactly once as the first argument to the
  next call
- keep explicit named bindings for semantically meaningful intermediates such as `h1`,
  `logits`, and `loss`
- keep explicit named bindings for values used more than once or for steps where naming
  materially improves readability

Examples of the intended direction:

- prefer `matmul(x, w1) |> add(expand(b1, 0, batch)) |> relu`
- avoid decompiled output that expands the same linear flow into `mm1`, `b1_exp`,
  `pre_h1`, `h1` unless those names carry semantic weight

Sequencing rationale:

- do this after Phase 2 language stability so the corpus is not rewritten repeatedly
- do this before Phase 4 corpus collection and model training so the training data uses
  the final idiomatic Surf style

Acceptance criteria:

- decompiler output prefers pipe chains for eligible linear flows
- examples in `examples/` and supporting teaching docs use the same pipe-first style
- `SKILL.md` guidance is updated to match the final idiomatic Surf corpus
- docs state clearly when a named intermediate should remain a `let` instead of a pipe

Phase 3 success condition:

- Chelis has a coherent external-user surface: package story, Python interop,
  research-extension direction, formalization target, and polished pipe-first examples

---

## Phase 4

**Prerequisite:** Phase 3 complete, including the pipe-first corpus/style pass.
**Deliverable:** first-party ML and coding-assistance stack built on the finalized
Phase 2 language and finalized example idioms.

### 4a: Seed Corpus Collection and Curation

- collect and curate 50-100 Chelis programs
- stratify by complexity rather than filtering to one difficulty band
- treat the corpus as shared infrastructure for evaluation, examples, and later training

### 4b: ICL Effect Measurement

- measure SKILL-assisted generation with and without the relevant context in prompt
- stratify results by corpus complexity band
- use this as the prerequisite experiment before choosing a heavier training path

### 4c: Trajectory Collection via Compiler Loop

- collect 2K-4K compiler-in-the-loop trajectories
- include failures, repairs, and complexity metadata
- use the compiler as the scoring/teaching surface rather than free-form human labels

### 4d: Local Coding Model Training

- start with SSD for distributional shaping
- fine-tune with the simplest method that meets the quality bar
- quantize and ship GGUF artifacts for local use
- preserve PyTorch/JAX semantic knowledge as a hard anti-forgetting constraint

### 4e: SKILL.md v2

- update `SKILL.md` for the full stable Phase 2 language surface
- cover effects, linearity, macros, `vmap`, tuples, and the finalized pipe-first Surf
  idiom
- keep teaching examples aligned with the curated corpus, not decompiler-debug style

### 4f: Coding Model Integration

- integrate the local model into `chelis cove --assist`
- expose the same capability through the MCP tool surface with local-model fallback
- baseline coding assistance should work without an API key or internet requirement

Phase 4 success condition:

- Chelis ships a first-party local coding model as part of the product, not as an
  optional research extra

---

## Phase 5

**Prerequisite:** Phase 4 complete or explicit product demand that justifies backend
expansion.
**Deliverable:** specialized backend expansion beyond the primary C and HIP paths.

### 5a: StableHLO Backend

- direct DAG to StableHLO emission
- TPU and XLA-family interop

### 5b: FX Graph Backend

- DAG to PyTorch FX graph export
- interoperability with TorchInductor, Triton kernels, and export flows

### 5c: Triton Backend

- emit Triton IR directly where it is a better fit than FX export
- reuse Triton's optimization passes when they materially help Chelis workloads

### 5d: Multi-GPU Data Parallelism Infrastructure

- add explicit multi-GPU support only if Phase 4 or downstream product needs justify it
- treat this as conditional infrastructure, not a default near-term commitment

---

## Ecosystem Library Decisions

Evaluated via multi-agent review.
These are settled decisions, not open prompts.

### Adopted / Planned

| Library | Phase | Purpose |
|---|---|---|
| `egg` | Phase 1b (prototype) | Equality saturation for fusion if greedy heuristics are not enough |
| `salsa` | Phase 2 | Incremental / demand-driven compilation for Tide, LSP, and agent loops |
| `ariadne` or `miette` | Phase 1+ (evaluate) | Rich diagnostics and fitness-report UX |

### Rejected

| Library | Reason |
|---|---|
| `logos` | Existing lexer works; hybrid handling would dilute the benefit |
| `chumsky` | Existing Pratt parser works; rewrite cost is too high |
| `petgraph` | Custom DAG is small and fits compiler invariants better |
| `cranelift` | Speculative second backend with high maintenance cost; evaluator and cached C paths come first |

### Architectural Discipline (Salsa Readiness)

To keep a future `salsa` migration mechanical rather than conceptual:

- each compilation stage is a pure function
- no global symbol tables are mutated across invocations
- crate APIs take inputs and return outputs

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
| RISC primitive semantics | `spec/05-risc-primitives.md` — ops, types, AD adjoints, lowerings | ✅ Complete (consumed by Phase 0e) |
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
| **Macro system design** | expansion rules, hygiene, phase separation, provenance annotation format (`{source: ...}` metadata key), interaction with the 59-tag vocabulary constraint (macros cannot introduce new tags) | Phase 2c | **MEDIUM** |
| **Fusion rules** | DAG fusion constraints and correctness conditions | Phase 1b | **MEDIUM** |
| **GPU memory model** | device-memory semantics and ownership model | Phase 1 / 2a | **MEDIUM** |
| **Effect handler syntax** | Surf and Deep syntax for handling effects | Phase 2a | **MEDIUM** |
| **Borrow syntax** | Surf and Deep borrowing forms | Phase 2b | **LOW** |
| **Custom effects** | user-defined effect extensibility model | Phase 2a | **LOW** |
