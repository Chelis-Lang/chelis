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
Phase 1a is complete.
Phase 1b and later GPU optimization/tooling work remain.

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

**Macro expansion boundary:** LLMs interact exclusively with expanded Deep.
Macro invocations are expanded before any LLM-facing operation (generation, training,
fitness scoring, error reporting).
Provenance metadata in the `{}` slot traces expanded nodes back to their macro source
(e.g., `{source: (relu input)}`).
The 56-tag vocabulary is the complete LLM-facing grammar regardless of how many macros
exist in the ecosystem.
Macros are a human authoring convenience that compiles away before LLMs touch the code.

### Serialization

Chelis currently has three named artifact forms:

- `.ch` for Surf source
- `.dp` for Deep source
- `.chb` for a future binary typed artifact

`.chb` is a planned concept, not a fully frozen on-disk format yet.

---

## 5. Ecosystem Names

All project-level naming follows the turtle/ocean metaphor.

| Concept | Name |
|---|---|
| Surface syntax | Surf |
| Canonical s-expression syntax | Deep |
| Packages | Shells |
| Package registry | Reef |
| Interactive mode | Tide |
| TUI coding environment | Cove |
| Project manifest | `reef.toml` |

Chelis is pronounced **CHEL-is**.
The domain is **chelis.ch**.

---

## 6. CLI Surface

The planned user-facing command set is:

```text
chelis build app.ch
chelis build app.ch --target hip
chelis check app.ch
chelis deep app.ch
chelis surf program.dp
chelis eval expr
chelis tide
chelis cove
chelis fmt app.ch
chelis validate --surf app.ch
```

Not every command is implemented yet.
This list is the intended stable surface for project-level documentation.

---

## 7. Type System Scope

### In scope for v1 / Phase 0

- ADTs with exhaustive pattern matching
- Hindley-Milner inference
- numeric precision tracking
- named tensor dimensions
- graded fitness scoring with repair suggestions

### Deferred to Phase 2

- algebraic effects
- linear types
- borrowing rules
- effect handlers
- lightweight uniqueness / alias tracking should be considered before a full heavy linear-ownership model

### Deferred to Phase 3

- distribution types
- equivariance constraints
- optimization-property annotations
- ILP/AUTOMAP-style rank-polymorphism remains a research idea, not a committed feature

---

## 8. Computational Model

Chelis lowers typed programs to a RISC DAG built from a small set of primitive tensor
operations.
High-level operations such as `matmul`, `softmax`, and `relu` are library-facing names
that lower into primitive compositions during compilation.

Core transforms remain first-class:

- `grad(f)` for reverse-mode AD
- `vmap(f)` for vectorization
- `jit(f)` as a future compilation boundary marker

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
- standard optimizers beyond SGD (Adam, AdamW, LAMB — update rules composed from
  primitives)
- learning rate schedulers
- data loading utilities
- metric computation (accuracy, F1, AUC)
- common loss functions that are compositions of primitives (focal loss, hinge loss)
- basic I/O (tensor serialization, checkpoint save/load)

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

### Grey area: Phase 2 effects

The Phase 2 effect system (`Diff`, `Random`, `Resource(Device)`) will make some
currently library-level constructs interact with the type system.
When `Random` is an effect, a probabilistic sampling library declares the `Random`
effect and the compiler tracks it — but the effect mechanism is designed to be open
(user-defined effects in Phase 3), so the compiler knows about the effect mechanism,
not about specific libraries that use it.
The sampling library declares `Random` effect; the compiler tracks it; neither needs to
know the other's internals.

### Phase 1–2 practical note

Everything currently lives in the core repo.
External libraries do not exist yet because there is no package manager (Reef is Phase
3a).
The standard library grows organically as the MNIST model and subsequent models demand
common utilities.
The first external library will likely be an architecture zoo (`chelis-models`) once
someone wants to ship a pretrained transformer, which requires the package system to
exist.

---

## 9. Backend Strategy

### Phase 0

Portable C code generation with BLAS and OpenMP.
This is the reference backend and numerical oracle for future backends.

### Phase 1

Futhark-style GPU compilation using C host code plus embedded **HIP** kernel strings,
compiled with `hiprtc` at runtime.
There is no separate CUDA backend plan.
HIP is the single GPU code generation path.

### Later

StableHLO and FX are additive integration layers for TPU and PyTorch ecosystem access.
They do not replace the C/HIP story.

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

Current validation result:

- the checked-in SKILL workflow validated at **9/10** tasks against the compiler on a
  local Qwen 35B MoE setup
- the remaining miss was a Deep repair execution failure, not a language-design or
  skill-content failure

This means the skill file is real project infrastructure, not aspirational promptware.

### Track 2: Local Model Ships With Toolchain

Chelis also requires a local coding model as a Phase 3 deliverable.
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
