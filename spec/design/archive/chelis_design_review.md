# Historical Note

This document is preserved as design-history context only.
It is **not** current guidance.
For active project decisions, use `spec/design/chelis_canonical_reference.md`,
`spec/design/chelis_project_plan.md`, and the numbered spec docs.

---

# Chelis: Design Review and Unity Check

## The Thesis, Restated

Chelis is a language written by AIs, for AIs, to write AI programs. The human is the supervisor, not the author. The coding agent is the primary user. The programs the agent writes are themselves AI systems — neural networks, evolutionary architectures, learned functions, neurosymbolic hybrids. Those AI systems, in turn, write more Chelis programs. Turtles all the way down.

This is NOT "a better PyTorch" or "Mojo but functional." This is a fundamentally new kind of programming language whose primary constituency is machine intelligence.

---

## What the Current Spec Gets Right

### The homoiconic typed AST is the killer feature
Programs are data. An AI agent can read a Chelis program, parse its s-expression AST, understand its types, mutate it, type-check the mutation, and produce a new valid program. This is the foundational enabler for evolutionary program synthesis, NAS, neurosymbolic methods, and AI self-improvement. No existing language offers this with a strong type system attached.

### The RISC primitive set is exactly right
Decomposing all tensor computation to ~12 primitives (from tinygrad's insight) means an AI agent learning to write Chelis only needs to master composition of 12 operations. This is a radically reduced search space compared to PyTorch's thousands of functions. The primitives are the alphabet; programs are sentences; the type system is grammar.

### Algebraic effects for AD, stochasticity, and resources
The Dex-inspired effect system is the right architecture. Effects are structural (inferred by the compiler, not declared by the programmer) which means AI-generated code gets automatic feedback: "this function has unhandled Diff effect at operation X." The AI agent doesn't need to annotate anything — the compiler tells it what's going on.

### Linear types for tensor memory
From Futhark. Tensors consumed exactly once. Borrowing for read-only access. This gives the compiler enough information for automatic memory planning AND gives AI agents immediate feedback on memory errors. No GPU memory leaks from AI-generated code.

### The lazy DAG with explicit realization
Build the computation graph, don't execute until `realize()`. The compiler sees the full graph before codegen, enabling fusion, DCE, memory planning. This is the same architecture that makes JAX, tinygrad, and XLA effective — but embedded in a typed, functional language rather than bolted onto Python.

---

## Where the Current Spec Needs Rethinking

### 1. The Surface Syntax Question

The spec says "modern functional surface syntax" (previously called "Scala 3-like" or "Elixir-influenced"). But if AIs are the primary authors, surface syntax is not the primary interface. The s-expression core IS the primary interface for machine generation. The surface syntax is a rendering layer for human supervision.

**Implication:** Don't over-invest in surface syntax design for v1. Get a clean, unambiguous, parseable surface that desugars predictably to s-expressions. Modern functional is the right family — ADTs, pattern matching, pipes, type annotations, `do` blocks. But the specific choices (braces vs `do/end`, `::` vs `:` for types, `def` vs `fn`) matter less than the property that the desugaring is mechanical and reversible.

**The real design question is the s-expression layer.** What does a Chelis s-expression look like? How are types represented? How do effects appear? How does an AI agent construct, validate, and mutate s-expression trees? This should get MORE design attention than surface syntax.

### 2. The Compiler as Training Signal

The spec treats the type checker as "catching errors." But for an AI-primary language, the type checker is a REWARD SIGNAL. Every compilation attempt produces structured feedback that the AI agent uses to improve its next generation attempt.

Currently missing from the spec:
- **Graded type errors.** Not just valid/invalid, but a measure of "how far from valid." An AI agent evolving programs needs to know that a program with one type error is closer to correct than a program with ten.
- **Repair suggestions in the compiler output.** "Expected tensor(batch, hidden, f32), got tensor(batch, seq, f32). Did you mean to transpose? Or did you mean to use a different dimension name?" These suggestions are training signal for the AI.
- **Partial type inference.** Even if the full program doesn't type-check, infer what you can and report what's broken. Don't just fail — degrade gracefully.

### 3. Program Evolution as a First-Class Concept

The spec identifies evolutionary program synthesis and NAS as the strongest use case, but the language doesn't include primitives for it. If programs-as-data is the thesis, the language should have:

**Post-steering resolution:** Programs-as-data is enabled via `quote`/`unquote` + the type checker as a library call. No built-in `mutate`, `crossover`, `evolve`, or `population` primitives. These are user-space concerns — different teams will want different mutation strategies, selection mechanisms, and fitness functions. The compiler's graded fitness feedback is the key enabler; methodology is yours.

### 4. The Backend Strategy

The spec says "dual primary backends: PyTorch FX and StableHLO." But the earlier design sessions revised this to: Futhark-style own-the-compilation as the primary GPU path, with vendor library dispatch (cuBLAS, cuDNN) for inner-loop operations. FX and StableHLO are still options but the spec hasn't been updated to reflect the shift toward owning compilation.

**Unresolved tension:** If Chelis owns its compilation (Futhark-style), the RISC primitives lower directly to GPU kernels via the Chelis compiler. If it delegates to FX/XLA, the RISC primitives lower to an intermediate representation that those systems compile. These are very different implementation paths.

**Post-steering resolution:** Phase 0 = C + BLAS (CPU, test oracle). Phase 1 = Futhark-style own-the-compilation (C host + embedded CUDA/OpenCL kernel strings). StableHLO and FX are additive Phase 2+ integration layers, not replacements. Compiler language: Rust (settled).

### 5. What's Over-Specified for v1

The following are important for the long-term vision but should not be in v1:
- **Distribution types** — Wait for v2. They're niche (probabilistic ML only) and add type system complexity.
- **Equivariance constraints** — Write the paper first, implement second. Most publishable, least immediately useful.
- **Optimization properties** (`@convex`, `@lipschitz`) — Trusted annotations with no verification are low value. Wait until you can verify them.
- **Concurrency Tier 2 details** (`stream`, `scatter`) — `par` alone covers v1. Stream and scatter are for pipeline parallelism and MoE, which are advanced use cases.

### 6. What's Under-Specified

- **The s-expression concrete syntax.** This is the most important design decision for the AI-generation use case and it's listed as "open."
- **The macro system.** Typed hygienic macros on s-expressions are mentioned but not designed. How do macros interact with effects? With linear types? With the type checker?
- **The REPL / interactive experience.** How does an AI agent interact with the Chelis compiler? A batch compiler? A language server? An API that accepts s-expressions and returns typed results?
- **Program serialization.** How are Chelis programs stored, transmitted, and versioned? S-expression text? A binary AST format? Both?
- **The bridge between surface and core.** Can an AI agent round-trip: surface → s-expression → mutation → s-expression → surface? Is the decompilation lossy?

---

## Influences: What We're Actually Taking

| Source | What We Take | What We DON'T Take |
|---|---|---|
| **Dex** | Algebraic effects for AD, dependent index types for tensor dimensions | The syntax, the Google-research-project-that-stalled execution model |
| **Futhark** | Uniqueness types for memory, purity-driven GPU compilation, the idea that the compiler owns hardware | Futhark's limitation to data-parallel array programs only |
| **tinygrad** | RISC primitive set (~12 ops), lazy DAG evaluation, "complex behavior from simple composition" | The Python implementation, the "small codebase" obsession |
| **JAX** | Composable function transformations (grad, vmap, jit), StableHLO as compilation target | Python as host language, tracing-based dynamism |
| **Lisp/Clojure** | Homoiconicity, s-expression AST, quote/unquote metaprogramming, programs as data | Dynamic typing, REPL-driven development as primary mode |
| **ML family** | Hindley-Milner inference, ADTs, exhaustive matching, strong static types | Haskell's lazy evaluation, OCaml's module system complexity, Scala's type system vastness |
| **Enzyme** | Build-vs-buy decision for AD at the compiler level | Direct LLVM-level AD (Chelis AD operates on the RISC DAG, not LLVM IR) |
| **Myia** | Functional differentiable IR as prior art | The specific implementation (abandoned research project) |

What we're NOT taking from any source:
- No JVM, no .NET, no BEAM as runtime
- No Python interop as a crutch (clean FFI, not "Python but better")
- No OOP, no classes, no inheritance
- No implicit conversions or promotions
- No lazy evaluation (strict by default)

---

## The Design Unity Test

**Does every feature serve the thesis: "language for AI to write AI"?**

| Feature | Serves thesis? | How? |
|---|---|---|
| Homoiconic typed AST | ✅ Core | Programs as evolvable, type-checkable data |
| RISC primitives (~12 ops) | ✅ Core | Minimal alphabet for AI to compose |
| Algebraic effects | ✅ Strong | Structural inference = automatic feedback to AI |
| Linear types | ✅ Strong | Automatic memory safety for AI-generated code |
| Named tensor dims | ✅ Strong | Dimension errors caught = better training signal |
| Numeric precision types | ✅ Strong | Prevents silent degradation in AI-generated models |
| Lazy DAG + realize | ✅ Strong | Full-graph optimization of AI-generated computation |
| grad/vmap/jit transforms | ✅ Strong | Composable transformations AI can reason about |
| Modern functional surface | ✅ Moderate | Human-readable rendering of AI-generated programs |
| Distribution types (v2) | ⚠️ Niche | Serves probabilistic ML, small audience |
| Equivariance (v2) | ⚠️ Novel | Publishable but narrow practical use |
| Concurrency Tier 2 | ⚠️ Future | Pipeline/MoE parallelism, not v1 |

Everything in the core serves the thesis. The v2 features are extensions, not distractions.

---

## Open Questions That Matter Most

Ranked by impact on the AI-writes-AI thesis:

1. **S-expression concrete syntax design.** The primary machine interface. Must be designed for parseability, mutation, type annotation, and round-tripping. This is the most important open question.

2. **Compiler as API.** The AI agent needs to interact with the Chelis compiler programmatically: submit programs, get typed results, get graded error feedback, get repair suggestions. This is a compiler UX question for machine users, not human users.

3. **Program evolution primitives.** Should mutate/crossover/evolve be in the language or in a library? If the language IS a substrate for evolving machine intelligence, these are core language features, not libraries.

4. **Backend strategy.** Own-the-compilation (Futhark-style) vs delegate (FX/XLA). Determines the entire compiler architecture. Needs a spike, not a design discussion.

5. **Compiler implementation language.** Settled: Rust. Workspace crate structure already established.

---

## What "Modern Functional" Means for Chelis

Not "Scala 3 syntax" — that over-indexes on one influence. The surface syntax family is:

- **ADTs and pattern matching** (ML family heritage)
- **Type inference** (write types at boundaries, infer within functions)
- **Pipes** (`|>` for left-to-right composition — natural for tensor pipelines)
- **No imperative escape hatches** (no `var`, no mutation, no loops — recursion and higher-order functions only)
- **Named function arguments** for readability in AI-heavy code (long parameter lists are common)
- **Uniform function call syntax** — `f(x)` and `x |> f` are interchangeable
- **Block expressions** — everything is an expression, blocks return their last value
- **No subtyping, no variance, no path-dependent types** — simpler than Scala 3 by design

The syntax should be clean enough that an AI agent can emit it directly when surface code is needed (documentation, examples, human review) but the s-expression layer is where generation, mutation, and evolution happen.
