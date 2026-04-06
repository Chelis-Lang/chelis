# Chelis: Project Plan

## Overview

Build the Chelis programming language from zero to MNIST-on-CPU and beyond.
This plan is written for a small team working with coding agents.
Each phase has a concrete deliverable, verification target, and red-team checkpoint.

**Current status:** Phase 0 complete.
Phases 0a-0i complete.
Phase 1 is next.

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
| **1** | Futhark-style GPU backend (HIP) + executable grammar (`chelis validate`) |  |
| **2** | Effects, linear types, macros, Tide Agent API + MCP, LSP, TUI (`chelis cove`) |  |
| **3** | Package ecosystem (Reef), StableHLO/FX backends, Python FFI, research type features, mechanized type system (Lean 4), local coding model (ships with toolchain) |  |

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

### 1d: Flattening and Parallelization

- flatten nested parallel structure
- support segmented reductions
- choose reasonable block sizing heuristics

### 1e: Benchmarks and Real Models

- benchmark MLP, CNN, transformer-block-scale workloads
- compare against the reference C backend for correctness
- target credibility, not premature parity with PyTorch

### 1f: Executable Grammar

- `chelis validate --surf file.ch`
- `chelis validate --deep file.dp`
- `chelis validate --desugar file.ch`

This remains the one valid `pest`-style parser-generator use case in the plan.
It is not a proposal to rewrite the Surf parser.

---

## Phase 2

**Prerequisite:** Phase 1 complete.
**Deliverable:** language maturity features and interactive tooling.

### 2a: Algebraic Effects

- `Diff`, `Random`, and `Resource(Device)`
- inferred rather than manually declared
- handled through explicit effect handlers

### 2b: Linear Types

- tensor linearity
- borrowing for read-only access
- explicit copy points
- compiler-enabled safe buffer reuse

### 2c: Macro System

- hygienic Deep macros
- type-aware expansion hooks
- explicit phase separation

**LLM representation constraint:** The macro system must produce clean expanded Deep
with provenance metadata in the `{}` slot.
Macro expansion is a compilation step that happens before any LLM-facing operation.
The expanded form uses only the base 56-tag vocabulary.
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

### 2g: Tide TUI (`chelis cove`)

- terminal coding environment
- live type checking
- Surf↔Deep toggling
- build and run flows without leaving the terminal

---

## Phase 3

**Prerequisite:** Phase 2 complete.
**Deliverable:** ecosystem, interoperability, and research extensions.

### 3a: Package System (Shells + Reef)

- `reef.toml`
- Shell publishing and dependency resolution
- Reef registry

### 3b: StableHLO Backend

- direct DAG to StableHLO emission
- TPU and XLA-family interop

### 3c: FX Backend

- DAG to PyTorch FX graph export
- interoperability with TorchInductor and export flows

### 3d: Python FFI

- DLPack tensor exchange
- PyO3 compiler bindings
- GIL release during Chelis execution

### 3e: Research Type Features

- distribution types
- equivariance constraints
- optimization-property annotations

### 3f: Mechanized Type System (Lean 4)

- mechanize the core type system only
- prove soundness for the stable Phase 0 core

### 3g: Chelis Coding Assistance

Chelis has two explicit first-party tracks for AI code generation.
Track 1 is the Phase 2 frontier-model workflow.
Track 2 is the Phase 3 shipped local-model deliverable.

**Key product framing:** a language for AIs that does not include an AI is an
incomplete product.

#### Track 1: SKILL.md + Frontier Models (Phase 2)

Ship a first-party `SKILL.md` alongside the Tide MCP server.
This is the compiler-in-the-loop workflow for frontier models such as Claude and GPT-5.

What ships:

- the 56-tag Deep grammar and arity rules
- the Surf→Deep desugaring table
- canonical form rules
- built-in scope and core type signatures
- worked examples over the current compiler surface
- common error patterns and fixes

Validation status:

- current skill validation result: **9/10** tasks passed against the compiler on a local
  Qwen 35B MoE setup
- the remaining failure is a Deep repair execution issue in the local model, not a spec
  or skill-content gap

Critical dependency:

- the Tide MCP server from Phase 2e

Non-negotiable prerequisite:

- 50-100 hand-written or supervised-interaction seed programs covering core Chelis
  patterns
- these serve as few-shot examples, trajectory seeds, and evaluation anchors

Phase 2 success condition:

- frontier models can write useful Chelis through `SKILL.md` + MCP
- the workflow is documented, tested, and repeatable

#### Track 2: Local Model Ships With Toolchain (Phase 3)

Phase 3 requires a local Chelis coding model that ships with the toolchain.
This is not a fallback and not optional.

Target deliverable:

- a 4B-8B-class local model distributed with the Chelis toolchain
- quantized GGUF artifacts for consumer hardware
- integrated into `chelis cove --assist` and related local workflows
- no API key or internet requirement for baseline coding assistance

Training pipeline:

1. **SSD for distributional shaping**
   Use the base model's own outputs to cheaply bias it toward Chelis structure before
   any expensive supervised fine-tuning.
   This is the bridge between "model has never seen Deep" and "model can produce
   something the compiler can score."
   If macros exist by the time the local model is trained, all training data uses
   expanded forms.
   Macro invocations are expanded before inclusion in any training dataset.
   The model never learns to generate macro calls — it generates the expanded pattern
   directly.
2. **Trajectory collection with compiler feedback**
   Collect full attempt traces, including failures, from compiler-in-the-loop runs.
3. **Fine-tune, method chosen empirically**
   Start with LoRA on trajectories, seed programs, spec tests, and curated examples.
   If forgetting is measured, switch to SDFT instead.
   RLVR is available as an optional final polish step if the quality bar still is not
   met.
4. **Quantize and ship as GGUF**
   Package the local model as a standard toolchain artifact.

Why SSD is explicit:

- it is cheap relative to API-driven data generation
- it uses the model's own outputs
- it directly addresses the structural validity gap seen in SKILL validation, where
  small and local models may reason correctly about Deep but fail to execute the
  canonical syntax in their emitted output

Prerequisite measurement:

- before any training, measure the ICL effect by running the SKILL evaluation with and
  without the spec in context
- if the spec meaningfully improves output quality, distillation-style methods are
  viable
- if it does not, stick with the simplest supervised path first

Hard constraint:

- anti-forgetting is mandatory; the trained model must preserve PyTorch/JAX semantic
  knowledge through Chelis training

Decision protocol:

1. Try LoRA first and measure both Chelis fitness and forgetting on a PyTorch/JAX
   comprehension benchmark.
2. If fitness is good and forgetting is minimal, ship it.
3. If forgetting is measured, replace LoRA with SDFT.
4. If quality still plateaus after the chosen fine-tuning method, use RLVR as final
   polish.

Phase 3 success condition:

- Chelis ships with a local coding model as part of the product, not as an optional
  research extra

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
| **Effect system design** | effect typing rules, handler syntax, HM interaction | Phase 2a | **HIGH** |
| **Linear type design** | linearity rules, borrowing rules, effect interaction | Phase 2b | **HIGH** |
| **Macro system design** | expansion rules, hygiene, phase separation, provenance annotation format (`{source: ...}` metadata key), interaction with the 56-tag vocabulary constraint (macros cannot introduce new tags) | Phase 2c | **MEDIUM** |
| **Fusion rules** | DAG fusion constraints and correctness conditions | Phase 1b | **MEDIUM** |
| **GPU memory model** | device-memory semantics and ownership model | Phase 1 / 2a | **MEDIUM** |
| **Effect handler syntax** | Surf and Deep syntax for handling effects | Phase 2a | **MEDIUM** |
| **Borrow syntax** | Surf and Deep borrowing forms | Phase 2b | **LOW** |
| **Custom effects** | user-defined effect extensibility model | Phase 2a | **LOW** |
