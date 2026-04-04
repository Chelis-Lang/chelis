# Chelis Language Specification: Context and Philosophy

**Version:** 0.1.0-draft
**Status:** Authoritative specification draft

---

## 1. What Chelis Is

Chelis is a statically typed, compiled programming language for tensor computation. Its thesis can be stated in six words: **AI writing AI writing AI.**

Chelis is designed from the ground up to serve two masters simultaneously:

1. **AI coding agents** that write Chelis code. The compiler emits structured feedback -- fitness scores between 0.0 and 1.0, repair suggestions as diffs, and typed error traces -- so that compiler output functions as a training signal, not just a diagnostic dump. An agent that generates Chelis code can hill-climb toward correctness by treating the compiler as a differentiable loss function over programs.

2. **ML researchers and engineers** who build AI/ML systems in Chelis. The language provides first-class automatic differentiation, vectorized mapping, JIT compilation markers, and a type system that understands tensor shapes and named dimensions. If a program type-checks, its tensor operations are shape-correct and its precision conversions are intentional.

The intersection of these two goals is the core insight: a language whose compiler is a collaborator rather than a gatekeeper is simultaneously better for humans and better for machines.

## 2. Target Audience

**Primary:**
- ML researchers who want type-safe tensor code with first-class `grad`, `vmap`, and `jit`.
- Compiler enthusiasts interested in type-directed tensor compilation and DAG rewriting.
- AI coding agents and agent frameworks that generate, evaluate, and iterate on code.

**Secondary:**
- Educators teaching functional programming, type theory, or compiler construction.
- Teams building reproducible ML pipelines who want compile-time shape checking.

**Explicitly not:**
- General-purpose application developers. Chelis has no standard library for file I/O, networking, or string processing beyond what's needed for its core mission.
- Data scientists who need a REPL-first, dynamically typed workflow. (Chelis has a REPL -- Tide -- but the type checker is always on.)

## 3. What Chelis Is Not

- **Not a Python replacement.** Chelis doesn't compete with Python for scripting, glue code, or general-purpose programming. It targets the inner loop: the tensor computation kernel, the model definition, the loss function.

- **Not a framework.** It's not PyTorch, JAX, or TensorFlow. It's a language. You don't `import chelis` -- you write `.ch` files and compile them.

- **Not a DSL embedded in another language.** Chelis has its own lexer, parser, type checker, IR, optimizer, and code generator. It stands alone.

- **Not dynamically typed.** Every expression has a type known at compile time. Shape mismatches are compile errors, not runtime crashes.

- **Not a research toy.** The goal is production-quality compiled output targeting C, CUDA, and StableHLO backends.

## 4. Key Influences

Chelis draws ideas from five distinct traditions:

### JAX
Functional transformations as the programming model. In JAX, `grad`, `vmap`, and `jit` are higher-order functions that transform functions into new functions. Chelis elevates these from library functions to language-level constructs: they are DAG-to-DAG rewrites with formal semantics and type rules. When you write `grad(f)`, the compiler doesn't call into a tracing runtime -- it rewrites the computation graph using adjoint rules.

### Futhark
A pure functional language that compiles to efficient parallel GPU code. Futhark proved that you don't need imperative mutation to get high-performance parallel tensor code. Chelis adopts the same bet: pure functions in, fast parallel code out. The compiler, not the programmer, decides how to partition work.

### Haskell
The type system is Hindley-Milner with algebraic data types (ADTs), parametric polymorphism, and full type inference. You can write type annotations everywhere, nowhere, or somewhere in between -- the compiler infers the rest. Pattern matching is exhaustive. Types are the specification: if it compiles, the shapes match.

### tinygrad
A minimal set of RISC-like operations (~12 primitives) that every tensor computation reduces to. Chelis's IR is a DAG of these primitives. This tiny instruction set makes it tractable to define adjoint rules (for AD), fusion rules (for optimization), and emission rules (for backends) exhaustively.

### Lean
Type theory as a foundation, not an afterthought. Lean demonstrated that dependent types and proof assistants can be practical tools, not just research artifacts. Chelis doesn't go full dependent types (yet), but it borrows the philosophy: the type system should be expressive enough that "well-typed programs don't go wrong" is a meaningful guarantee for tensor code.

## 5. The Turtle Metaphor

Chelis is named after **Chelonia** -- the order of turtles and tortoises. The ocean metaphor pervades every layer of the system:

| Concept | Ocean term | Meaning |
|---------|-----------|---------|
| Surface syntax | **Surf** | What humans and agents read and write (`.ch` files) |
| Canonical s-expression syntax | **Deep** | What the compiler works with internally (`.dp` files) |
| Interactive mode | **Tide** | The REPL and agent API -- comes and goes, stateful |
| Package ecosystem | **Reef** | Where packages live and grow, interconnected |
| Compiled artifacts | **Shells** | Hard, portable, self-contained output |
| Concurrency model | **Current** | Data flows through the system like water |

**"It's turtles all the way down."** The language is self-referential in its design philosophy:

- The compiler's fitness-score output is itself a differentiable signal -- you could, in principle, train a model to write Chelis by treating the compiler as a loss function.
- The surface syntax desugars to s-expressions, which desugar to a DAG of ~12 primitives, which emit to target code. Each layer is a complete representation of the program. Turtles all the way down.
- The language is designed for AI agents to write code that builds AI systems. The tool builds the toolmaker.

## 6. Design Principles

### Principle 1: Types Are the Specification

If a Chelis program type-checks, its tensor operations are shape-correct, its precision conversions are intentional, and its differentiable functions are actually differentiable. The type system is not a bureaucratic hurdle -- it is the primary mechanism by which the language guarantees correctness.

This means the type system must be expressive enough to capture:
- Named tensor dimensions (`batch`, `hidden`, `seq_len`)
- Precision types (`f32`, `bf16`) with no implicit conversion
- Function linearity (for correct automatic differentiation)
- Algebraic data types for structured model outputs

### Principle 2: No Implicit Anything

Chelis never silently broadcasts a tensor, coerces a precision, or reshapes an array. Every such operation is explicit in the source code. This is a deliberate trade-off: more verbosity in exchange for more predictability. When a shape mismatch occurs, the error message can point to the exact line, because there's no invisible broadcasting rule that might or might not apply.

- No implicit precision widening (`f32` to `f64` requires `cast`)
- No implicit broadcasting (use explicit `vmap` or reshape)
- No implicit tensor creation (shapes are always specified)
- No implicit effects (pure by default)

### Principle 3: The Compiler Is a Collaborator

Traditional compilers are gatekeepers: the program is either correct or it's rejected with an error message. Chelis's compiler is a collaborator:

- **Fitness scores** (0.0 to 1.0): Every program gets a score. A fully correct program scores 1.0. A program with one shape mismatch might score 0.85. An agent can use this as a differentiable signal to improve its output.
- **Repair suggestions**: The compiler doesn't just say "type mismatch on line 7" -- it says "insert `cast(x, f32)` on line 7 to fix precision mismatch" or "change dimension `hidden` to 512 to match the weight matrix."
- **Partial compilation**: Even programs with type errors can be partially lowered, so the compiler can report which parts are correct and which aren't.
- **Structured output**: All compiler output is available as JSON, not just human-readable text. Agents parse JSON; humans read the pretty-printed version.

### Principle 4: Small Core, Big Surface

The internal representation (RISC DAG) has approximately 12 primitive operations:

```
add, mul, reduce_sum, reduce_max, reshape, broadcast, 
slice, concat, matmul, exp, log, compare
```

Every tensor computation in Chelis -- no matter how complex the surface syntax -- desugars through the type checker and lowering passes into a DAG of these primitives. This tiny core makes the system tractable:

- Each primitive has one adjoint rule (for automatic differentiation).
- Each primitive has one fusion rule set (for optimization).
- Each primitive has one emission template per backend (for code generation).
- The entire core can be formally verified.

But the surface syntax is rich: named dimensions, pattern matching, ADTs, pipe operators, comprehension-like `vmap`, and ergonomic type annotations. The surface is for humans; the core is for machines.

### Principle 5: Transformations Are First-Class

`grad`, `vmap`, and `jit` are not library functions. They are language-level constructs with formal type rules and semantics:

- **`grad(f)`** takes a function `f : A -> scalar` and returns `f' : A -> A` (the gradient). Implemented as a DAG rewrite using adjoint rules. The type system ensures `f` actually returns a scalar and that `A` is a differentiable type.

- **`vmap(f, axis=k)`** takes a function `f : A -> B` and returns a function that maps `f` over axis `k` of its input. Implemented as a DAG rewrite that lifts each primitive to operate over an additional dimension. The type system tracks the new dimension.

- **`jit(f)`** marks a function for just-in-time compilation. At the DAG level, this inserts a compilation boundary. The type doesn't change, but the execution strategy does.

These transformations compose: `grad(vmap(f, axis=0))` is a valid expression with a well-defined type and a mechanical DAG rewrite.

## 7. A Note on Versioning

This specification describes Chelis version 0.1.0. The language is in active design. Breaking changes are expected. The stability promise is:

- **Deep syntax**: Stable. Programs written in Deep form will continue to parse.
- **Surf syntax**: Unstable. Surface syntax may change between minor versions.
- **RISC primitives**: Stable set, but semantics may be refined.
- **Type system**: The core (HM + ADTs + tensor types) is stable. Extensions (effects, linearity, dependent dimensions) are experimental.

## 8. How to Read This Specification

The specification is organized as follows:

| Document | Contents |
|----------|----------|
| `00-context.md` | This document. Philosophy, goals, non-goals. |
| `01-nomenclature.md` | Glossary of all terms, metaphors, and conventions. |
| `02-surf-syntax.md` | Complete grammar and semantics of the surface language. |
| `03-deep-syntax.md` | Complete grammar and semantics of the s-expression form. |

Future specification documents will cover the type system, RISC DAG, transformation semantics, backend emission, the Tide REPL, and the Reef package system.

An implementer should read these documents in order. A user of the language needs only `02-surf-syntax.md` and the examples therein to start writing Chelis code.
