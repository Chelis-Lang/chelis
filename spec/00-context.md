# Chelis Language Specification: Context and Philosophy

## 1. What Chelis Is

Chelis is a numerical computing language for code that agents write and people
supervise.
Tensors carry named dimensions and precision in their type, and a proof stack checks
the properties an author states about the code.
Chelis is general purpose within numerical computing: numerical methods, dataframes,
simulation, statistics, and quantitative finance are all in scope.

Chelis is designed around three linked claims:

1. Numerical code deserves a type system that understands tensor dimensions,
   precision, effects, and ownership, and checks them before anything runs.
2. Agent-written code benefits from a regular canonical syntax for machines and a
   readable syntax for the people who review it.
3. Compiler and prover feedback should be structured and deterministic enough for an
   agent to act on, and specific enough for a person to audit.

## 2. What Chelis Is Not

- Not a Python replacement for general scripting
- Not a web or systems language
- Not a framework embedded in another host language
- Not a deep-learning framework

Chelis targets the numerical core of a program, not the entire surrounding
application stack.

## 3. Audience

Primary users:

- coding agents that write, check, and repair numerical code
- engineers who supervise that code in correctness-sensitive domains such as finance
- researchers in programming languages and verification

The language is intentionally optimized for agent authoring and human review rather
than for manually writing large general-purpose applications.

## 4. Core Bet

Numerical code written by agents is easier to trust when its types carry the facts
numerical errors depend on (shape, precision, effects, ownership) and when its
toolchain can check stated properties, than when those facts first surface at run time
in a host language.
Numerical computing defines the scope; `grad`, `vmap`, and `jit` are transforms within
it (§8).

## 5. Design Principles

When tradeoffs appear, Chelis applies these in order:

1. **Unambiguity over ergonomics**
2. **Composition over special cases**
3. **Inference over annotation**
4. **Machine generation first**
5. **Additive sugar only**
6. **Explicit over implicit**
7. **Small language, big library** — the compiler knows only the closed RISC primitive
   set and its derived built-ins; everything else is a library.
   See `spec/design/chelis_canonical_reference.md` §8.5 for the full core/standard-library/external-library taxonomy.
8. **Future-proof without over-building**

These principles determine the following rules:

- Deep is canonical
- Surf is sugar over Deep
- broadcasting and precision conversion are explicit
- compiler stages should remain mechanically understandable
- the core IR stays small even when the user-facing language grows
- for fixed program text, compiler build, target, and declared inputs, every
  check, evaluation, and build result is a function of those inputs; feedback
  that varies between identical runs is a defect, not an implementation freedom

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
Deep decompiles back to formatter-canonical Surf, apart from the fail-closed
resugaring exceptions in `02-surf-syntax.md`; a best-effort verbose form exists for
debugging (`09-tide.md` §4).
The compiler treats Deep as the source of truth.

## 7. Type System Scope

The type system includes:

- algebraic data types
- Hindley-Milner inference
- named tensor dimensions
- explicit precision tracking
- graded fitness scoring with repair suggestions
- algebraic effects
- linear types and borrowing

General dependent types beyond named dimensions are outside the language contract.

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

The reference backend is C with BLAS and OpenMP.
GPU compilation is source-to-source: HIP for AMD GPUs and Metal for Apple GPUs, rather
than separate CUDA and OpenCL backends.
StableHLO and FX are integration layers rather than replacements.

Interactive execution uses the IR evaluator. Cached C artifacts, persistent compiler
helpers, and JIT compilation are alternative implementation strategies; each must
preserve the language semantics defined by the controlling specifications.

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
7. `06-transformations.md`
8. `07-concurrency.md`
9. `08-backends.md`
10. `09-tide.md`
11. `10-serialization.md`
12. `11-ffi.md`
13. `12-roadmap.md`

For project-level decisions rather than language semantics, read
`spec/design/chelis_canonical_reference.md`.
