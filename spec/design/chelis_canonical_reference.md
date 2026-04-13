# Chelis: Canonical Project Reference

**Status:** Active project-level reference.
Use this document to align README text, roadmap updates, design docs, and agent-authored
documentation.
Language semantics still belong in the numbered spec documents.

---

## 1. What Chelis Is

Chelis is a functional programming language for AI research.
It is designed for a workflow where a coding agent is the primary author and a human is
the supervisor.
Programs written in Chelis are themselves AI systems: models, training loops,
evolutionary search spaces, and learned functions.

Chelis is not a general-purpose language, not a systems language, not a web framework,
and not a Python replacement.
It targets model authoring, training, deployment, and program synthesis for AI
workloads.

**Current status:** Phase 0 is complete.
Phases 0a-0i are complete.
Phases 1a-1f are implemented.
Phase 1 remains in progress at the project level because the phase-level red-team
checkpoint is still the completion bar.

---

## 2. Core Bet

AI development benefits from a language whose type system, representation, and
compilation model are designed around AI primitives instead of being bolted onto Python
or a systems language after the fact.

---

## 3. Design Principles

When tradeoffs appear, apply these in order:

1. **Unambiguity over ergonomics.**
2. **Composition over special cases.**
3. **Inference over annotation.**
4. **Machine generation first.**
5. **Additive sugar only.**
6. **Explicit over implicit.**
7. **Small language, big library.**
8. **Future-proof without over-building.**

Practical consequences:

- no implicit broadcasting
- no implicit precision promotion
- no implicit currying or partial application
- Deep is canonical; Surf is supervision-friendly sugar

---

## 4. Dual Syntax Architecture

### Surf

Surf is the human-facing syntax stored in `.ch` files.
It is the supervisory interface: modern functional syntax with ADTs, pattern matching,
pipes, and type annotations.

### Deep

Deep is the machine-facing syntax stored in `.dp` files.
It is the compiler's canonical textual representation and the generation target for AI
agents.

Every Deep node has the same shape:

```lisp
(tag {} children...)
```

- `tag` comes from the closed Deep tag vocabulary
- `{}` is the metadata map and is always present in canonical form
- `app`, `var`, and `lit` are explicit structural nodes

Deep is the single source of truth.
Surf desugars losslessly to Deep.
Deep decompiles back to Surf on a best-effort basis.
Current Phase 1 caveat: the compiler may emit dotted Deep module/import paths that
`chelis validate` accepts, but the compiler-side Deep parser does not yet fully
reparse that emitted shape. This round-trip gap is tracked as a Phase 2 bug because it
matters for decompilation and LLM repair workflows, but it is not a Phase 1 blocker.

**Macro expansion boundary:** LLMs interact exclusively with expanded Deep.
Macro invocations are expanded before any LLM-facing operation (generation, training,
fitness scoring, error reporting), and before CLI paths such as `chelis deep`,
`check`, `build`, and `eval`.
Provenance metadata in the `{}` slot traces expanded nodes back to their macro source
(e.g., `{source: (relu input)}`).
The 60-tag vocabulary is the complete LLM-facing grammar regardless of how many macros
exist in the ecosystem.
Macros are a human authoring convenience that compiles away before LLMs touch the code.
Compiler-internal pre-expansion forms such as `defmacro` and `macro-invoke` are not
public Deep and are rejected by strict Deep validation.

### Serialization

Chelis currently has three named artifact forms:

- `.ch` for Surf source
- `.dp` for Deep source
- `.chb` for Shell metadata in the Reef package system

`.chb` now exists as an implementation-owned binary package interface.
The name and role are stable; the low-level wire format is intentionally not frozen in
the public spec yet.

---

## 5. Ecosystem Names

All project-level naming follows the turtle/ocean metaphor.

| Concept | Name | Rationale |
|---|---|---|
| Surface syntax | **Surf** | The surface — what you see |
| Canonical s-expression syntax | **Deep** | The depths — what's underneath |
| Packages | **Shells** | Turtles have shells; self-contained |
| Package registry | **Reef** (reef.chelis.ch) | Where shells live |
| Interactive mode | **Tide** (`chelis tide`) | Comes and goes; interactive |
| TUI coding environment | **Cove** (`chelis cove`) | Sheltered workspace where Surf meets shore |
| Project manifest | `reef.toml` | A project's place in the reef |
| Numerical methods shell | **Nautilus** | Mathematical precision — the chambered nautilus is nature's logarithmic spiral |
| Dataframe shell | **Coral** | Structured colonies built from the reef |
| Finance shell | **Shoals** | Where the currents of capital run shallow |
| Classical ML shell | **School** | A school of fish learning together — and the ML sense of *learning* |
| Evolutionary algorithms shell | **Darwin** | Natural selection — survival of the fittest programs, mutated and crossed over the Deep AST |

Chelis is pronounced **CHEL-is**.
The domain is **chelis.ch**.

### Shell Ecosystem

Packages layered on `chelis-std`. `nautilus` and `coral` are independent and can land in
parallel; `shoals` depends on both. `school` (classical ML) and `darwin` (evolutionary
algorithms) are post-Phase-3 stubs.

| Shell | Depends On | Status | Contents |
|---|---|---|---|
| `chelis-std` | (core) | Active | `Std.Nn` (Linear, Embedding, LayerNorm, Generate with KV cache, GELU/SiLU/RMSNorm, Conv1d/2d, attention), `Std.Optim` (SGD, Adam, AdamW, LAMB), `Std.Loss` (including KL, BCEWithLogits, accuracy, perplexity), `Std.Init` (Kaiming, Xavier, trunc_normal), `Std.Schedule`, `Std.IO` (files, mmap, safetensors, CSV, JSON), `Std.Tokenizer`, `Std.Time`, `Std.Decimal` |
| `nautilus` | `chelis-std` | Phase 3j | Numerical methods — stats, distributions, linear algebra (nalgebra-backed with hand-written AD adjoints), convex optimization, ODE/SDE solvers, roots, integration, interpolation, special functions (`erf`, `log_gamma`, …), distances. The scipy competitor. `Nautilus.Signal` stubbed until complex numbers (Phase 5f). |
| `coral` | `chelis-std` | Phase 3k | Typed dataframes — numeric columns are tensors (lazy, GPU-accelerable, fusible via the DAG), string columns are host-side lists (eager). AD through dataframe operations. Column selection, filtering, sort-by, group-by, joins, pivot/melt, rolling windows, NaN handling built into `Coral.Frame`, Parquet I/O via `parquet2`, DataFrame-aware CSV/JSON. The pandas competitor. No query optimizer — numeric optimization comes from the tensor compiler's fusion. |
| `shoals` | `chelis-std` + `nautilus` + `coral` | Phase 3l | Options pricing, risk measures, yield curves, stochastic processes, order books |
| `school` | `chelis-std` + `nautilus` + `coral` | **Stub** (post-3) | Classical ML (scikit-learn competitor). Regression, decision trees, SVMs, clustering, pipelines, cross-validation. |
| `darwin` | `chelis-std` + `nautilus` required, `coral` optional | **Stub** (post-3) | Evolutionary algorithms — GA, genetic programming over the Deep AST, evolution strategies, population-based training, neural architecture search. Uniquely natural fit because Deep is homoiconic: program mutation and crossover are typed AST operations, and the compiler's 0–1 fitness scoring is literally the fitness function for evolutionary search. `coral` is optional for evolving feature-engineering pipelines over tabular data. |

Design rule: `chelis-std` covers what every Chelis program may need (tensors, neural
primitives, time, decimal). `nautilus` owns general numerical methods. `coral` owns
tabular data. `shoals` is finance-only. `school` is classical ML only. `darwin` is
evolutionary search only. Time and decimal stay in `chelis-std` because every domain
needs dates and exact arithmetic.

---

## 6. CLI Surface

The planned user-facing command set is:

```text
chelis build app.ch
chelis build app.ch --target hip
chelis check app.ch
chelis deep app.ch
chelis deep --flat app.ch
chelis surf program.dp
chelis eval expr
chelis tide
chelis tide serve --port 8080
chelis tide mcp
chelis tide lsp
chelis cove
chelis fmt app.ch
chelis fmt app.dp --check
chelis reef init demo --module-prefix Demo
chelis reef build
chelis reef publish
chelis validate --surf app.ch
chelis validate --deep app.dp
chelis validate --desugar app.ch
```

This is the intended stable surface for project-level documentation.
`chelis deep` defaults to canonical pretty Deep; `--flat` is the explicit flat-output
escape hatch.

## 6a. Surf Style

Project-facing Surf should read like human-written model code, not typed Deep debug output.

- prefer `def ... -> T = ...` for typed function definitions
- put input types on parameters instead of top-level load-style bindings
- use symbolic dimensions for runtime-varying axes such as `batch` and `seq`
- keep fixed architecture dimensions concrete
- avoid redundant intermediate type ascriptions when inference already determines the type
- prefer meaningful intermediate names over mechanically naming every primitive step

Planned public-style target for Phase 3:

- use block bindings such as `x = expr`; Surf no longer has a separate `let` surface
- prefer pipe-first composition for eligible linear flows
- use multiline pipes for long or many-stage chains, breaking after `=` and before every
  `|>` when the flat form exceeds the width budget or the chain becomes visually dense
- apply the same flat-first, width-threshold philosophy in Surf that Deep already uses
  for pretty printing

---

## 7. Type System Scope

### In scope for v1 / Phase 0

- ADTs with exhaustive pattern matching
- Hindley-Milner inference
- numeric precision tracking
- named tensor dimensions
- graded fitness scoring with repair suggestions

### Phase 2a shipped subset

- effect annotations on Surf `sig` / `def` and Deep `t-fn` metadata
- checked-program upgrade: downstream passes consume annotated Deep with type metadata
- algebraic-effect boundary handling for `Random` and `Resource(Device)`
- `with seed(...)` for seeded stochastic regions and `with device(...)` for resource regions

### Deferred to later Phase 2 work

- broader effect inference/checking beyond the shipped `Random` / `Resource(Device)` subset
- `Diff` as a fully specified effect surface (it remains a compiler capability today)
- `Accum` as user-visible effect surface (it remains internal-only today)
- linear types for tensors with borrowing rules and explicit `copy`
- lightweight uniqueness / alias tracking before any full heavy ownership-and-lifetimes model
- macro expansion before all LLM-facing operations, with provenance in metadata
- algebraic-effect, linearity, macro, and `vmap` tooling that remains planned in Phase 2

### Planned remaining Phase 3 work

Phase 3 is now the language-completeness phase rather than the research-extension
phase. The remaining practical language work is:

- first-class scalar `Int` / `Float` / `Bool` values outside tensors
- first-class immutable `String` values with practical non-tensor operations
- collection types such as `List[T]` and `Dict[K, V]`
- functional iteration over variable-length host-side data
- core numeric primitives such as `einsum`, `concat` / `split`, `gather` / `scatter`,
  `where`, `cumsum`, `sort`, `diagonal` / `trace`, and `clamp`
- a Rust runtime rewrite that replaces the old C runtime implementation and cleans up
  the compiled host-value ABI before more host/library work lands
- data-loading and tokenization support that removes the mandatory Python
  preprocessing step
- standard-library host modules such as `Std.Time` and `Std.Decimal`

### Deferred to Phase 5+

- sparse tensors
- complex numbers
- distribution types
- equivariance constraints
- optimization-property annotations
- ILP/AUTOMAP-style rank-polymorphism and related research type features
- Lean mechanized formalization of the core type system

---

## 8. Computational Model

Chelis lowers typed programs to a RISC DAG built from a small set of primitive tensor
operations.
High-level operations such as `matmul`, `softmax`, and `relu` are library-facing names
that lower into primitive compositions during compilation.

The remaining core-language expansion planned for Phase `3h` adds the practical tensor
surface real model code expects: `einsum`, `concat` / `split`, `gather` / `scatter`,
`where`, `cumsum`, `sort`, `diagonal` / `trace`, and `clamp`. `Std.Nn.Embedding`
remains the explicit public shell/library surface over `gather`.

Core transforms remain first-class:

- `grad(f)` for reverse-mode AD
- `vmap(f)` for vectorization
- `jit(f)` as a future compilation boundary marker
- the shipped executable 2d subset supports direct `vmap(f)(args...)` and
  `vmap(grad(f))(args...)` applications for named defs and inline lambdas, lowering the
  transform away before ordinary DAG codegen
- the shipped executable `vmap(grad(f))` path now also supports flat tuple-valued
  gradient payloads from multi-parameter `grad(..., wrt=(...))`
- first-class stored/returned transformed function values remain out of the executable
  path for now

Implementation-surface note:

- `min_elem` is part of the specified derived built-in surface.
- `block` is part of the Deep syntax vocabulary and Surf block desugaring.
- `normalize` currently exists in the type checker built-in environment, but it is not
  yet a stable specified/lowered built-in.
  Treat it as provisional implementation surface until `spec/05` and the IR lowering
  are aligned.

The language is built around programs-as-data, but mutation and evolution operators are
left to user space rather than embedded as special language primitives.
Programs-as-data operations (`quote`, `unquote`, `splice`) work on expanded Deep.
Macros have already been resolved — the AST an agent inspects or transforms contains
only base tags.

`vmap` remains a core compiler transform in the same sense as `grad`: a DAG rewrite
whose semantics compose with the rest of the lowering pipeline rather than a library
helper.

Current 2d performance boundary:

- batched `matmul` is correct on both backends but stays in the generic
  `expand -> mul -> sum` decomposition
- the existing HIP rank-2 BLAS fast path does not yet upgrade vmapped rank-3 matmul into
  a batched BLAS call

---

## 8.5. Scope Boundaries

The cut line between core language and library is whether the compiler needs to know
about it.

### Core (ships with the compiler)

Everything the compiler has special knowledge of.
The ~12 Tier 1 RISC primitives (`add`, `mul`, `exp`, etc.) are language-native because
the compiler decomposes them, the AD engine has adjoint rules for them, and the C/HIP
backends emit specialized code for them.
The Tier 2 derived built-ins (`relu`, `sigmoid`, `softmax`, `matmul`, `layer_norm`,
`conv2d`) are in the core because the compiler recognizes them by name and decomposes
them to RISC primitives during IR lowering.
The type checker knows their signatures.
The optimizer can fuse them.
They cannot be defined as user-space library functions because a user-space function
cannot teach the AD engine its adjoint or the GPU backend its kernel fusion strategy.

The core transforms (`grad`, `vmap`, `jit`) are also compiler-intrinsic for the same
reason: they require compiler cooperation to implement.

This is roughly the scope of PyTorch's `torch` namespace — the fundamental tensor
operations, basic neural network layers, loss functions, and optimizers that are
implemented in C++/CUDA underneath.
In Chelis, they are implemented as compiler-recognized patterns that lower to RISC DAG
subgraphs.

### Standard library (`Std.*`)

Ships with Chelis but is implemented in Chelis itself.
The compiler does not know these names.
They ship as Shells (Chelis packages) in the `Std` namespace.

Expected contents:

- common initializers (Xavier, Kaiming, normal, uniform)
- standard neural-network building blocks such as `Std.Nn.Embedding`
- standard optimizers beyond SGD (Adam, AdamW, LAMB — update rules composed from
  primitives)
- learning rate schedulers
- data loading utilities
- tokenizer utilities
- metric computation (accuracy, F1, AUC)
- common loss functions that are compositions of primitives (focal loss, hinge loss)
- basic I/O (tensor serialization, checkpoint save/load)
- time/date helpers (`Std.Time`)
- exact-decimal helpers (`Std.Decimal`)

### External libraries

Anything that expresses an opinion about model architecture, training methodology, or
domain.
These are Chelis programs that depend on the core and standard library but add domain
knowledge the compiler does not need.

Examples by analogy:

- **scikit-learn equivalent** (`chelis-ml`): classical ML algorithms, preprocessing
  pipelines — compositions of tensor ops with specific algorithmic structure
- **HuggingFace Transformers equivalent** (`chelis-transformers`): pre-built
  architectures (GPT, BERT, LLaMA, ViT) and pretrained weight loaders
- **torchvision/torchaudio equivalents**: domain-specific dataset loaders, augmentation
  pipelines, and model architectures
- **Probabilistic modeling** (`chelis-diffusion`): denoising schedules, noise prediction
  architectures, sampling algorithms

### The decision principle

If removing it would make the compiler produce worse code (cannot optimize, cannot
differentiate, cannot fuse), it belongs in the core.
If removing it just means the user has to write it themselves from the primitives, it
belongs in a library.

### Grey area: shipped Phase 2a effects vs later extensibility

The shipped Phase 2a surface is intentionally closed and compiler-known:
`Random`, `Accum`, `IO`, and `Resource(Device)` live in the type layer. `Random` and
`Resource(Device)` are the Phase 2a boundary-checked effects; `IO` is the shipped Phase
3 host-side debugging/logging effect.
This is narrower than the longer-term design space.
User-defined effects remain deferred; the current compiler knows both the effect
mechanism and the concrete built-in effect vocabulary it ships.

### Phase 3 practical note

Everything currently lives in the core repo.
The shipped Phase 3 foundations already cover public Surf style, Reef packaging, and
Python interop. The remaining Phase 3 work is not more ecosystem polish or research
prestige work; it is the language-completeness gap between "tensor programs compile"
and "a full AI workflow can run in pure Chelis."

As part of that practical gap, Phase `3m` rewrites the runtime in Rust and cleans up
the host-value ABI: `chelis_tensor` stays layout-visible for generated numeric code,
while strings, collections, and other host values move to opaque runtime-managed
handles with accessors and explicit ownership.

That means the next practical surfaces are:

- scalar/string programming
- collections and iteration
- core numeric primitives beyond the original minimal tensor surface
- the Rust runtime rewrite that moves host-value runtime work out of
  `chelis_runtime.c`
- file/config/data loading
- tokenization and batching
- standard-library time and exact-decimal support

Broader hosted registry work, research type features, and Lean formalization remain
later work.

---

## 9. Backend Strategy

### Phase 0

Portable C code generation with BLAS and OpenMP.
This is the reference backend and numerical oracle for future backends.
Generated programs include `chelis_runtime.h`; the runtime implementation behind that
header is now expected to ship as a Rust static library rather than a hand-maintained C
implementation file.

### Phase 1

Futhark-style GPU compilation using C host code plus embedded **HIP** kernel strings,
compiled with `hiprtc` at runtime.
There is no separate CUDA backend plan.
HIP is the single GPU code generation path.

### Later

StableHLO, FX, and Triton are additive integration layers for TPU and PyTorch/NVIDIA
ecosystem access. They do not replace the C/HIP story. The Python interop stack now
includes CPU-only
PyTorch DLPack plus PyO3 compiler bindings from `bindings/python`, and `3b-ii` adds
direct execution via `compile_and_load` / `load` plus the NumPy DLPack guarantee. The
JAX DLPack guarantee remains deferred to the StableHLO phase.

### Rejected

- no OpenCL-first backend plan
- no separate JIT backend on the roadmap
- no Cranelift adoption unless measured latency proves that cheaper options fail

---

## 10. Interactive Execution

Tide and `chelis eval` use the IR evaluator in `chelis-ir/src/eval.rs` as the default
interactive execution path.
Interactive execution does **not** compile through C by default.

If latency later becomes a real bottleneck, the escalation order is:

1. IR evaluator
2. cached C artifacts
3. persistent compiler helper process
4. JIT only if the first three fail on measured workloads

This is a performance policy, not an open design question.

---

## 11. Compiler Implementation

Chelis is implemented as a Rust workspace with six primary crates:

- `chelis-deep`
- `chelis-surf`
- `chelis-types`
- `chelis-ir`
- `chelis-backend-c`
- `chelis-cli`

Compiler infrastructure is intentionally hand-written where it matters:

- hand-written lexers
- hand-written Pratt / recursive-descent parsing
- custom DAG
- explicit optimization passes

Settled library decisions:

### Adopted or planned

| Library | Decision |
|---|---|
| `egg` | Prototype for Phase 1 fusion only; adopt only if fusion search is genuinely combinatorial |
| `salsa` | Planned for Phase 2 incremental compilation use cases |
| `ariadne` / `miette` | Evaluate later for diagnostics UX |

### Rejected for now

| Library | Reason |
|---|---|
| `logos` | Existing lexer works; hybrid handling would erase the declarative win |
| `chumsky` | Existing Pratt parser works; rewrite cost is too high for current value |
| `petgraph` | Custom DAG is smaller and better matched to compiler needs |
| `cranelift` | Not planned; IR evaluator and cached C paths are cheaper interactive options |

### Architectural discipline

To keep a future `salsa` migration cheap:

- each compilation stage should remain a pure function
- no global mutable compiler state
- public crate APIs should take inputs and return outputs

---

## 12. AI Coding Assistance

Chelis has two explicit first-party tracks for AI code generation.

### Track 1: SKILL.md + Frontier Models

Chelis ships a first-party `SKILL.md` for frontier models operating through the Tide MCP
server.
This is the Phase 2 coding-assistance story: compiler-in-the-loop generation, no local
training requirement, and immediate usefulness for agents that already have strong
general coding ability.
The full-surface `SKILL.md` v2 refresh belongs to late Phase 3, after the shipped
public Surf idiom and the remaining language-completeness surfaces are stabilized:
pipe-first chains, short-form block bindings, scalar/string code, collections,
iteration, core numeric primitives, tokenization/data-loading workflows, and the later
`Std.Time` / `Std.Decimal` host-program surfaces.

Current validation result:

- the checked-in SKILL workflow validated at **9/10** tasks against the compiler on a
  local Qwen 35B MoE setup
- the remaining miss was a Deep repair execution failure, not a language-design or
  skill-content failure

This means the skill file is real project infrastructure, not aspirational promptware.

### Track 2: Local Model Ships With Toolchain

Chelis also requires a local coding model as a Phase 4 deliverable.
This is not optional and not a speculative fallback.

Target shape:

- a local 4B-8B-class coding model
- quantized GGUF artifacts that run on consumer hardware
- integration with `chelis cove --assist` and related local workflows

Training pipeline:

1. **SSD for distributional shaping**
2. **Trajectory collection with compiler feedback**
   Compute `nesting_depth × operation_count` as a complexity proxy for each generated
   program and log it.
   Stratify by complexity band post-collection; let the ICL prerequisite measurement
   determine the effective band rather than pre-committing.
3. **Fine-tune, method chosen empirically**
4. **Quantize and ship as GGUF**

SSD matters because it is the cheap bridge between "model has never seen Deep" and
"model can emit something the compiler can score."
It uses the model's own outputs and directly targets the structural-validity gap seen
in SKILL evaluation, where smaller or local models may reason correctly about Deep yet
still fail to emit the canonical form.

Step 3 starts with LoRA as the default.
If forgetting is measured, switch to SDFT instead.
RLVR remains available as an optional final polish step if the quality bar still is not
met.
Anti-forgetting is a hard constraint: the model must preserve PyTorch/JAX semantic
knowledge through training.
Before any training, measure the ICL effect by running the SKILL evaluation with and
without the spec in context.
That quick measurement gates whether distillation-style methods are worth trying at
all.

The local model generates and is trained on expanded Deep exclusively.
Macro invocations never appear in training data, generation targets, or compiler
feedback sent to models.

Product framing:

- a language for AIs that does not include an AI is an incomplete product
- Track 1 is the frontier-model path
- Track 2 is the shipped local-model path

**ChelisBench:** A 50-task ML programming benchmark comparing LLM code generation in
Chelis vs PyTorch on equivalent tasks. Measures whether a language designed for LLMs
produces better ML code than the standard approach. Serves double duty as a measurement
tool and a trajectory source for model training.

**Type-driven property testing:** The Tide API provides `chelis test` / `POST /test` —
automatic test generation from function type signatures. The compiler knows tensor
shapes, dtypes, and dimension constraints; it generates random valid inputs, runs the
function, and verifies output shapes, determinism (for pure functions), and gradient
correctness (for differentiable functions). No test code written by anyone — the type
signature is the test specification.

---

## 13. Documentation Hierarchy

Use the docs in this order:

1. this file for project-level truth
2. `README.md` for repository orientation
3. `ARCHITECTURE.md` for crate and pipeline overview
4. numbered `spec/*.md` files for language semantics
5. `spec/design/chelis_project_plan.md` for phased execution

Historical design notes belong under `spec/design/archive/` and must be treated as
rationale, not current guidance.
