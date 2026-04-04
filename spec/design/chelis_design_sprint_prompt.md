# Chelis Language Design Sprint — Agent Brief

You are designing the formal syntax and semantics for **Chelis**, a new functional programming language for AI research. This document contains everything you need to produce sound design decisions. Read it fully before starting.

---

## What Chelis Is

Chelis is a programming language written by AIs, for AIs, to write AI programs. Turtles all the way down.

The human is the supervisor, not the primary author. A coding agent (Claude Code, Cursor, etc.) is the primary user. The programs the agent writes are themselves AI systems — neural networks, evolutionary architectures, learned functions, neurosymbolic hybrids. The language is named for chelys (Greek: turtle). Domain: chelis.ch.

**The core bet:** AI development benefits from a language whose type system, representation, and compilation model are designed around AI primitives — not from bolting AI capabilities onto Python or a systems language like Mojo.

**Target workloads:** Model authoring, training, and deployment. Evolutionary architecture search. Neurosymbolic program synthesis. Learned-function compilation. Not general-purpose systems programming, not web services, not CLI tools.

**Target audience:** ML researchers, AI systems engineers, agent framework developers. A PhD student using Chelis with a coding LLM should produce a better NeurIPS paper faster than they would in PyTorch.

---

## The Dual Syntax Architecture

Chelis has two syntactic layers:

**Surf** — the surface syntax (.ch files). What humans read. Modern functional: ADTs, pipes, pattern matching, type annotations. This is the supervisory interface — a human directing a coding agent reads Surf to understand what the agent produced.

**Deep** — the s-expression syntax (.dp files). What machines write. Homoiconic, parseable, mutable, type-annotated s-expressions. The generation target for AI agents, the substrate for evolutionary methods, the internal representation the compiler operates on. Deep is more constrained than Surf — no sugar, no ambiguity, one canonical form per program.

**The relationship:** Surf desugars losslessly to Deep. Deep decompiles to Surf (cosmetically lossy — multiple Surf forms can produce the same Deep). Round-tripping Surf → Deep → Surf produces valid, readable code. Deep is the single source of truth.

**Serialization:** Three formats — `.ch` (Surf text), `.dp` (Deep text), `.chb` (binary typed AST, like Scala 3's TASTy).

**CLI:**
```
chelis build app.ch        # compile Surf → Deep → IR → target
chelis deep app.ch         # emit Deep form (desugar only)
chelis surf program.dp     # decompile Deep → Surf (best-effort)
chelis check app.ch        # type-check, return graded fitness feedback
chelis tide                # interactive REPL (accepts both Surf and Deep)
```

---

## Settled Design Decisions

These are final. Do not revisit them. Design within these constraints.

### Type System (v1 scope)
- **ADTs** with exhaustive pattern matching (ML family)
- **Hindley-Milner type inference** with explicit annotations at module/function boundaries
- **Numeric precision types:** f32, f16, bf16, f8e4m3, int8, int32, bool. No silent promotion — explicit `cast()` required.
- **Named tensor dimensions:** `tensor[batch, seq, hidden, bf16]`. Nominal checking — `tensor[batch, hidden, f32]` ≠ `tensor[batch, seq, f32]`. Catches transposition and broadcasting errors statically.
- **Algebraic effects** (Dex-inspired, NOT type classes): `Diff` (differentiability), `Random` (stochasticity), `Resource(Device)` (allocation). Effects are structural — inferred by the compiler, not declared by the programmer. `grad` is an effect handler for `Diff`.
- **Linear types for tensors** (Futhark-inspired): Tensors consumed exactly once. Borrowing for read-only access. Explicit `.copy()` for duplication. Compiler uses linearity for in-place mutation and memory planning.

Note: Effects and linear types are Phase 2 implementation, but their interaction with the type system needs to be designed now so Phase 0 doesn't paint itself into a corner.

### Computational Model
- **RISC primitive set** (~12 typed tensor ops, from tinygrad): Elementwise (add, mul, div, neg, exp, log, sin, sqrt, cmplt, max), Reduce (sum, max over axes), Movement (reshape, permute, expand, pad, shrink, stride), Memory (load, store, const). ALL computation decomposes to these.
- **Lazy DAG with explicit realization:** Operations build a typed DAG. Nothing executes until `realize()` or `jit(f)(args)`.
- **Function transforms:** `grad(f)` (reverse-mode AD), `vmap(f)` (vectorization), `jit(f)` (compilation) — all DAG-to-DAG rewrites.

### Backend Strategy
- **Phase 0:** C + BLAS (CPU). Source-to-source transpilation. Test oracle for correctness.
- **Phase 1:** Futhark-style GPU backend — C host code + embedded CUDA/OpenCL/HIP kernel strings. Compiler owns fusion, parallelization, memory planning. Vendor JIT (NVRTC/clBuildProgram) handles last mile.
- **Phase 2+:** StableHLO (TPU) and FX graphs (PyTorch ecosystem) as additive integration layers.

### Compiler as Training Signal
The compiler produces graded fitness feedback (0-1 score), partial type inference, and structured repair suggestions. Not just pass/fail — a reward function for AI agents doing RL or evolution over program space.

### Programs as Data (Enabled, Not Imposed)
Homoiconic typed AST + quote/unquote + type checker as library call enables evolutionary methods, NAS, neurosymbolic search. But NO built-in mutate/crossover/evolve primitives — those are user-space concerns. The language provides the substrate; methodology is yours.

### Concurrency (v1)
DAG-implicit parallelism (automatic) + `par` primitive only. `stream` and `scatter` deferred to later.

### Surf Syntax Decisions Already Made
- **Brace-delimited blocks.** No indentation sensitivity. One block style.
- **No `_` partial application in pipes.** Use explicit lambdas: `x |> fn v -> f(a, v)` not `x |> f(a, _)`.
- **Pipes (`|>`)** as the primary composition mechanism. Left-to-right data flow.
- **No mutation, no loops, no imperative blocks.** Recursion and higher-order functions only.
- **Type annotations with colon:** `x: tensor[batch, seq, f32]` (not `::`)
- **Square brackets for tensor dimensions:** `tensor[batch, seq, dim, bf16]`

### Ecosystem Nomenclature
- Packages = **shells**
- Package registry = **Reef** (reef.chelis.ch)
- Interactive mode = **Tide** (`chelis tide`)
- Project file = `reef.toml`

### Key Influences (What We Take, What We Don't)

| Source | Take | Don't Take |
|---|---|---|
| **Dex** | Algebraic effects for AD, dependent index types for tensor dims | Syntax, stalled execution model |
| **Futhark** | Uniqueness/linear types for memory, purity-driven GPU compilation, source-to-source architecture | Limitation to data-parallel array programs only |
| **tinygrad** | RISC primitive set (~12 ops), lazy DAG evaluation | Python implementation, "small codebase" obsession |
| **JAX** | Composable transforms (grad, vmap, jit), StableHLO target | Python host, tracing |
| **Lisp/Clojure** | Homoiconicity, s-expression AST, quote/unquote | Dynamic typing, REPL-as-primary-mode |
| **ML family (F#, OCaml, Haskell)** | HM inference, ADTs, exhaustive matching, pipes | Haskell's lazy eval, OCaml's module complexity, Scala's type system size |
| **Racket** | Hygienic macros with lexical context, syntax objects with metadata | Full Racket complexity |
| **Elixir** | 3-tuple AST with metadata (quote/unquote model), pipe ergonomics | BEAM runtime, OTP, dynamic types |
| **Rust** | Ownership/borrowing concepts (adapted to linear types), compiler error quality | Borrow checker complexity, lifetimes |
| **Scala 3 TASTy** | Binary typed AST serialization for .chb format | Scala's type system, JVM |

---

## Your Design Tasks

You need to produce complete, precise specifications for the following. Each should be formal enough that a coding agent can implement directly from your output without ambiguity.

### TASK 1: Deep Syntax Specification

The complete s-expression syntax for Chelis Deep. This is the most important deliverable because it's the primary machine interface.

**What you must specify:**

**1a. Tag vocabulary.** The exhaustive list of valid top-level tags in Deep s-expressions. Based on the examples in the settled design, the tags include at least: `module`, `def`, `fn`, `sig`, `let`, `match`, `type`, `variant`, `record`, `field`, `tensor`, `dim`, `pipe`, `quote`, `unquote`, `if`. But this list is incomplete. You need to determine the full set.

Consider: What tags are needed for ADT declarations? Pattern matching arms? Type annotations? Anonymous functions? Function application? Arithmetic and comparison operators? Tensor operations? The ~12 RISC primitives (do they appear in Deep, or only in the IR after lowering?)? Effect annotations? Linear type markers? Import/export?

Design principle: The tag set should be minimal but complete. Every valid Chelis program must be expressible. Redundancy is worse than verbosity.

**1b. Metadata format.** How do types, effects, linearity status, and source location attach to AST nodes?

Options to consider:
- **Clojure-style reader metadata:** `^{:type int32} expr` — metadata prefix on any expression
- **Explicit meta nodes:** `(meta (:type int32) expr)` — metadata wraps expressions
- **Property list in tag position:** `(def {:type (-> int32 int32)} f ...)` — metadata as second element
- **Elixir-style 3-tuple:** Every node is `(tag metadata children)` where metadata is always present (possibly empty)
- **Implicit only:** Metadata is never written in `.dp` text files — it's filled in by compiler passes and only appears in `.chb` binary format

Think about: AI agents constructing Deep programs need to be able to include type annotations when they want (to constrain generation) but omit them when they want the compiler to infer. Which format makes this cleanest?

**1c. Canonical form rules.** Deep has exactly one textual representation per program. Specify:
- Whitespace rules (indentation? one expression per line? compact?)
- Ordering rules (are record fields alphabetized? are match arms in declaration order?)
- Comment handling (are comments preserved in canonical form? probably not)
- Numeric literal normalization (is `1.0` the same as `1.` in canonical form?)

**1d. How function application works.** In Deep, is function application `(f x y)` (Lisp-style) or `(apply f (x y))` (explicit)? What about partial application? Currying? Higher-order functions?

**1e. How operators work.** Are arithmetic operators like `+`, `*`, `-` special forms or regular function application? In Deep, is `a + b` represented as `(add a b)` or `(+ a b)` or `(binop + a b)`?

**1f. Example programs.** Write at least these programs in your specified Deep syntax:
- Hello tensor: create a tensor, add 1, print
- Linear regression: matmul, loss, grad, update loop
- MLP with relu: two layers, pattern matching on activation type
- ADT definition and matching

### TASK 2: Surf Formal Grammar

A complete grammar for the surface syntax. PEG or EBNF format. Must be precise enough to implement a parser from.

**What you must specify:**

**2a. Full grammar.** Every production rule. Covering:
- Module declaration and structure
- Type declarations (ADTs with record-style variants)
- Function definitions with type signatures
- Let bindings
- Pattern matching (match/with)
- Anonymous functions (fn)
- Pipes (|>)
- Brace-delimited blocks
- Tensor type syntax (square brackets)
- Arithmetic and comparison operators with precedence
- Function application
- Record field access
- Import/export (at least a minimal design)

**2b. Operator precedence table.** Complete and unambiguous. Include: arithmetic (+, -, *, /), comparison (==, !=, <, >, <=, >=), logical (&&, ||, !), pipe (|>), function application, function composition, type annotation (:).

**2c. Keyword list.** Every reserved word. No ambiguity about what's a keyword vs. an identifier.

**2d. Desugaring rules.** For each Surf construct, specify exactly what Deep form it desugars to. This is the formal bridge between Surf and Deep.

**2e. Example programs.** Write the same four programs from Task 1 in Surf, and show the Surf → Deep desugaring for each.

### TASK 3: RISC Primitive Semantics

The ~12 operations that all computation decomposes to. For each primitive:

**3a. Precise signature.** Input types, output types, type constraints. Including: what precisions are supported? What ranks? What dimension constraints?

**3b. Semantics.** What does the operation compute? Be precise enough to write a reference implementation.

**3c. AD adjoint rule.** The reverse-mode differentiation rule. For each primitive, what is the backward pass? Express as: given `grad_output`, produce `grad_input(s)`.

**3d. Broadcasting rules (if any).** Does Chelis support implicit broadcasting (NumPy-style)? Or must all dimensions be explicit? This is a critical design decision. Broadcasting is convenient but a major source of silent bugs. Given the "AI-native" thesis and named dimensions, explicit shapes with no broadcasting is probably correct — but this needs to be decided and specified.

**3e. Standard operation lowerings.** How common ML operations decompose into RISC primitives:
- `matmul(A, B)` → ?
- `conv2d(input, kernel, stride, padding)` → ?
- `softmax(x, axis)` → ?
- `layer_norm(x, gamma, beta)` → ?
- `relu(x)` → ?
- `sigmoid(x)` → ?
- `cross_entropy(logits, labels)` → ?
- `embedding(indices, table)` → ?
- `multi_head_attention(q, k, v, mask)` → ?

For each, show the decomposition into the ~12 RISC primitives.

### TASK 4: Type System Formal Rules

Precise enough to implement the type checker from.

**4a. Type representation.** The data types representing types in the compiler:
- Primitive types (f32, bf16, int32, bool, etc.)
- Function types (A → B, with effects)
- Tensor types (with dimensions and precision)
- ADT types (sum types, record types)
- Type variables (for inference)
- Type constructors (for polymorphism)

**4b. Inference rules.** In standard PL notation (judgment form with premises and conclusion). Cover:
- Variable lookup
- Function application
- Let-binding (with let-polymorphism)
- Lambda abstraction
- Pattern matching (exhaustiveness checking)
- Tensor construction
- Dimension checking

**4c. Tensor type algebra.** The rules for how tensor types interact:
- When can a `tensor[batch, seq, f32]` be passed to a function expecting `tensor[batch, seq, f32]`? (exact match)
- When can a `tensor[a, b, f32]` be passed? (dimension variables — polymorphism)
- Can you have `tensor[*, *, f32]` meaning "any 2D f32 tensor"? If so, how does it interact with named dims?
- How does `reshape` change the type? How does `permute` change the type? How does `reduce` change the type?

**4d. Precision type rules.** Complete rules for numeric precision:
- Which operations preserve precision? (add f32 f32 → f32)
- Which require same precision? (add f32 bf16 → ERROR)
- How does `cast` work? (cast(x, bf16) : bf16 if x is any numeric tensor)
- Are there implicit widening rules? (Probably no — explicit only)

**4e. Fitness scoring algorithm.** Given a Deep AST, how is the 0-1 fitness score computed?
- What fraction of subexpressions type-check?
- How are different error severities weighted?
- How does partial inference work — what does the compiler annotate when the full program fails?
- What repair suggestions exist for common error patterns?

### TASK 5: Named Dimension Design

This needs dedicated attention because it's the most novel part of the v1 type system.

**5a. Dimension declaration.** How are dimension names introduced? Options:
- Module-level: `dim batch, seq, hidden` declares dimension names for the module
- Implicit: First use of a lowercase name in tensor position declares it
- Function-level: Dimension parameters on functions: `def f[batch, seq](x: tensor[batch, seq, f32])`
- Some combination

**5b. Dimension polymorphism.** Can a function be generic over dimensions? How?
```
-- Can this be written? What's the Deep form?
def transpose(x: tensor[a, b, f32]): tensor[b, a, f32]
```

**5c. Dimension arithmetic.** Does the type system track dimension relationships?
```
-- After concatenation along seq:
-- if x: tensor[batch, seq1, f32] and y: tensor[batch, seq2, f32]
-- is the result tensor[batch, seq1 + seq2, f32]? Or just tensor[batch, ?, f32]?
```
This gets into dependent types. Where do you draw the line for v1?

**5d. Interaction with RISC primitives.** How does each RISC primitive transform dimension types? `reshape` is the hard case — it can change dimension names and structure arbitrarily.

### TASK 6: Effect System Sketch (Forward-Looking)

Effects are Phase 2 implementation, but the design must not conflict with Phase 0 decisions. Provide a sketch (not full formal rules) covering:

**6a. Effect syntax in Deep.** How do effect annotations appear on functions?
**6b. Effect handler syntax in Deep and Surf.** How does `grad` (the Diff handler) appear?
**6c. Interaction with HM inference.** Do effects change the inference algorithm?
**6d. Interaction with linearity.** How do `Resource` effects interact with linear tensor types?

### TASK 7: Linear Type Sketch (Forward-Looking)

Same as Task 6 — Phase 2 implementation, but Phase 0 must not preclude it.

**7a. Which types are linear?** All tensors? Only GPU tensors? User-annotated?
**7b. Borrowing in Deep and Surf.** Syntax and semantics.
**7c. Interaction with pattern matching.** If you pattern-match on a linear value, which branch consumes it?
**7d. Interaction with closures.** Can a closure capture a linear value?

### TASK 8: Macro System Sketch

**8a. Macro definition syntax in Deep.** How are macros defined?
**8b. Quote/unquote model.** How does code-as-data work? Elixir-style? Lisp-style? Something else?
**8c. Hygiene model.** Automatic renaming? Scope sets (Racket-style)? Something simpler?
**8d. Interaction with types.** Can macros call the type checker? How?
**8e. Phase separation.** When do macros run relative to type checking?

---

## Design Principles to Follow

When facing a tradeoff, apply these in order:

1. **Unambiguity over ergonomics.** If a design choice introduces ambiguity (for a parser OR for an AI agent generating code), reject it even if it's more convenient for humans. Deep especially must have zero ambiguity.

2. **Composition over special cases.** The ~12 RISC primitives work because complex behavior emerges from composing simple things. Apply the same principle to syntax and types.

3. **Inference over annotation.** The compiler should figure things out. But when it can't, the annotation syntax must be clean and non-intrusive.

4. **Machine generation first.** For Deep: optimise for AI agents constructing programs. Parseability, regularity, and predictability matter more than human readability. For Surf: optimise for human reading of AI-generated code. Clarity and consistency matter more than conciseness.

5. **Additive sugar only.** Surf features that don't exist in Deep are sugar. Sugar is added via the Surf → Deep desugaring pass. Sugar is never required — you can always write the desugared form directly. Sugar should be designed so it can be added later without breaking existing programs.

6. **Future-proof without over-building.** Phase 0 doesn't implement effects or linear types, but the syntax and type representation must have room for them. Design explicit extension points where Phase 2 features will plug in.

---

## Output Format

For each task, produce:

1. **The specification** — formal enough to implement from. Use EBNF/PEG for grammars, inference-rule notation for type rules, and precise prose for semantics.
2. **Rationale** — why this design, what alternatives were considered, what was rejected and why.
3. **Example programs** — showing the design in action. Both Surf and Deep versions.
4. **Open questions** — anything you couldn't resolve that needs further discussion. Flag these explicitly rather than making arbitrary choices.

Write the specifications as if they will become the actual `spec/*.md` files in the Chelis repository. They should be complete, precise, and implementable.
