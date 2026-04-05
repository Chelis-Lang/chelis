# Chelis Language Specification: Context and Philosophy

**Version:** 0.2.0-draft
**Status:** Authoritative specification draft

---

## 1. What Chelis Is

Chelis is a functional programming language for AI research.
It is built for a workflow where a coding agent is the primary author and a human is
the supervisor.
The target programs are themselves AI systems: models, training pipelines,
architecture-search programs, and learned functions.

Chelis is designed around three linked claims:

1. AI workloads deserve a type system that understands tensor dimensions, precision,
   and differentiability.
2. AI-generated code benefits from a machine-friendly canonical syntax.
3. Compiler feedback should be useful as training signal, not only as a pass/fail gate.

## 2. What Chelis Is Not

- Not a Python replacement for general scripting
- Not a web or systems language
- Not a framework embedded in another host language
- Not "PyTorch but with different syntax"

Chelis targets the model-definition and compilation layer, not the entire surrounding
application stack.

## 3. Audience

Primary users:

- ML researchers building new architectures and training procedures
- AI systems engineers working on correctness-sensitive tensor programs
- agent frameworks that generate, evaluate, and repair code automatically

The language is intentionally optimized for AI-native authoring rather than for
manually writing large general-purpose applications.

## 4. Core Bet

AI development benefits from a language whose representation, type system, and
compilation strategy are designed around AI primitives from the start rather than
retrofitted onto Python or a systems language later.

## 5. Design Principles

When tradeoffs appear, Chelis applies these in order:

1. **Unambiguity over ergonomics**
2. **Composition over special cases**
3. **Inference over annotation**
4. **Machine generation first**
5. **Additive sugar only**
6. **Explicit over implicit**
7. **Small language, big library**
8. **Future-proof without over-building**

These principles lead directly to several current rules:

- Deep is canonical
- Surf is sugar over Deep
- broadcasting and precision conversion are explicit
- compiler stages should remain mechanically understandable
- the core IR stays small even when the user-facing language grows

## 6. Dual Syntax

Chelis has two syntax layers:

- **Surf** (`.ch`): human-facing syntax for reading, review, and supervision
- **Deep** (`.dp`): machine-facing canonical syntax used by the compiler

Every Deep node has the form:

```lisp
(tag {} children...)
```

This regularity is a design feature.
Agents do not need to infer whether a node head is structural or user-defined.

Surf desugars losslessly to Deep.
Deep decompiles back to Surf on a best-effort basis.
The compiler treats Deep as the source of truth.

## 7. Type System Scope

Phase 0 / v1 scope:

- algebraic data types
- Hindley-Milner inference
- named tensor dimensions
- explicit precision tracking
- graded fitness scoring with repair suggestions

Deferred:

- algebraic effects
- linear types and borrowing
- richer research type features

## 8. Computational Model

Chelis lowers typed programs into a RISC DAG composed from a small set of primitive
tensor operations.
Higher-level operations such as `matmul`, `relu`, `softmax`, and loss functions lower
into primitive compositions during compilation.

Three transforms are first-class in the language design:

- `grad`
- `vmap`
- `jit`

They are compiler transforms, not ordinary library conveniences.

## 9. Backends and Execution

The current reference backend is C with BLAS and OpenMP.
Future GPU compilation is planned around HIP, not separate CUDA and OpenCL backends.
StableHLO and FX are later integration layers rather than replacements.

Interactive execution uses the IR evaluator first.
If latency later becomes a problem, the escalation order is cached C artifacts,
persistent compiler helpers, and only then a possible JIT path.

## 10. Key Influences

Chelis draws important ideas from:

- **Dex** for algebraic effects around differentiation
- **Futhark** for purity-driven compilation and linearity direction
- **tinygrad** for the compact RISC-like primitive set
- **JAX** for composable transforms such as `grad` and `vmap`
- **Lisp / Clojure** for homoiconicity
- **ML-family languages** for HM inference, ADTs, and pattern matching
- **Rust** for explicitness and error-quality ambitions

## 11. Reading Order

Use the spec in this order:

1. `00-context.md`
2. `01-nomenclature.md`
3. `02-surf-syntax.md`
4. `03-deep-syntax.md`
5. `04-type-system.md`
6. `05-risc-primitives.md`
7. `08-backends.md`
8. `09-tide.md`
9. `12-roadmap.md`

For project-level decisions rather than language semantics, read
`spec/design/chelis_canonical_reference.md`.
