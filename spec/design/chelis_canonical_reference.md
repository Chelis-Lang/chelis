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

**Current status:** Phase 0f (C backend codegen) is in progress.
Phases 0a-0e are complete.
All core spec documents are written and reviewed.

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

### Deferred to Phase 3

- distribution types
- equivariance constraints
- optimization-property annotations

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

The language is built around programs-as-data, but mutation and evolution operators are
left to user space rather than embedded as special language primitives.

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

## 12. AI Integration

### Coding model strategy

Chelis uses a staged first-party coding-capability strategy rather than assuming a
fine-tuned model from the start.

**Stage A (Phase 2, near-zero cost):** write a `SKILL.md` covering the Deep tag
vocabulary, desugaring rules, canonical form, and worked examples, then use frontier
models with the compiler through the Tide MCP server.
No training.
Pure in-context learning plus compiler feedback.

**Stage B (Phase 2-3):** wrap the compiler as a trajectory-generating evaluator.
Collect model attempts, repair loops, and failures as training data.
The compiler-as-teacher loop is both an inference-time tool and a data-generation
mechanism.

**Stage C (Phase 3, only if needed):** fine-tune on trajectories, spec tests, and
examples only if the earlier stages do not meet the quality bar.

Non-negotiable prerequisite:

- 50-100 seed programs covering core Chelis patterns

Critical dependency:

- Tide MCP server in Phase 2e

The default assumption is **skill file + compiler-in-the-loop first, fine-tune only if
necessary**.

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
