# Chelis: Project Plan

## Overview

Build the Chelis programming language from zero to MNIST-on-CPU, structured for a coding agent working with a small team. Each phase has clear deliverables, test criteria, and red team checkpoints.

**Repo:** `chelis-lang/chelis` (Rust workspace)
**Domain:** chelis.ch
**License:** TBD (likely Apache 2.0 or MIT for compiler, separate for spec)

---

## Phasing

| Phase | Deliverable | Timeline Target |
|---|---|---|
| **0a** | Project scaffold, spec docs, CI, test infra | Week 1 |
| **0b** | Deep parser (s-expressions) | Week 2 |
| **0c** | Surf parser + Surf→Deep desugaring | Week 3-4 |
| **0d** | Type checker (ADTs, HM inference, precision, named dims) | Week 5-8 |
| **0e** | RISC DAG construction from typed AST | Week 9-10 |
| **0f** | C backend codegen (host + BLAS) | Week 11-13 |
| **0g** | `grad` transformation (reverse-mode AD on DAG) | Week 14-16 |
| **0h** | End-to-end: MNIST on CPU | Week 17-18 |
| **0i** | Tide v0.1 (REPL, `chelis deep`, `chelis surf`) | Week 19-20 |
| **1** | Futhark-style GPU backend (CUDA/OpenCL) | Months 6-9 |
| **2** | Effects, linear types, macros, Tide Agent API, MCP | Months 9-14 |
| **3** | StableHLO/FX, package ecosystem (Reef), research type features | Months 14+ |

**Red team checkpoints** after: 0a, 0d, 0h, and each major phase. A red team round means: an adversarial review of design decisions, test coverage, spec compliance, and architectural debt. Document findings, revise, then proceed.

---

## Phase 0a: Project Scaffold

**Goal:** A coding agent can clone the repo and immediately understand: what Chelis is, how it's structured, where to find the spec, how to build, how to test, and what to work on next.

### Directory Structure

```
chelis/
├── README.md                    # Project overview, build instructions, links to spec
├── ARCHITECTURE.md              # High-level architecture guide (the map)
├── CONTRIBUTING.md              # How to contribute, code style, PR process
├── Cargo.toml                   # Workspace root
├── reef.toml.example            # Example Chelis project file (for when the lang works)
│
├── spec/                        # Language specification (the source of truth)
│   ├── 00-context.md            # What Chelis is, why it exists, who it's for
│   ├── 01-nomenclature.md       # Surf/Deep, shells/reef/tide, file extensions
│   ├── 02-surf-syntax.md        # Surface syntax grammar and examples
│   ├── 03-deep-syntax.md        # S-expression syntax grammar and examples
│   ├── 04-type-system.md        # Type system: ADTs, inference, precision, dims, effects, linearity
│   ├── 05-risc-primitives.md    # The ~12 tensor operations and their semantics
│   ├── 06-transformations.md    # grad, vmap, jit — DAG-to-DAG rewrites
│   ├── 07-concurrency.md        # Tier 1 (DAG-implicit) + par
│   ├── 08-backends.md           # C backend (Phase 0), GPU (Phase 1), StableHLO/FX (Phase 2+)
│   ├── 09-tide.md               # Interactive mode: REPL, Agent API, MCP, LSP
│   ├── 10-serialization.md      # .ch, .dp, .chb formats
│   ├── 11-ffi.md                # Python interop (DLPack, PyO3)
│   └── 12-roadmap.md            # Phase plan, what's in/out for each version
│
├── crates/                      # Rust workspace members
│   ├── chelis-deep/             # Deep parser (s-expression lexer + parser)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── lexer.rs         # Tokenizer for s-expressions
│   │   │   ├── parser.rs        # S-expression parser → AST nodes
│   │   │   ├── ast.rs           # Deep AST data types
│   │   │   └── printer.rs       # AST → s-expression text (canonical form)
│   │   └── tests/
│   │       ├── parse_roundtrip.rs
│   │       └── fixtures/        # .dp test files
│   │
│   ├── chelis-surf/             # Surf parser + desugaring
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── lexer.rs         # Tokenizer for surface syntax
│   │   │   ├── parser.rs        # Surface syntax parser → Surf AST
│   │   │   ├── ast.rs           # Surf AST data types
│   │   │   ├── desugar.rs       # Surf AST → Deep AST
│   │   │   └── decompile.rs     # Deep AST → Surf text (best-effort)
│   │   └── tests/
│   │       ├── parse_tests.rs
│   │       ├── desugar_tests.rs
│   │       ├── roundtrip_tests.rs
│   │       └── fixtures/        # .ch test files
│   │
│   ├── chelis-types/            # Type checker
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── types.rs         # Type representation (ADTs, tensors, precision, dims)
│   │   │   ├── infer.rs         # Hindley-Milner inference engine
│   │   │   ├── unify.rs         # Unification algorithm
│   │   │   ├── env.rs           # Type environment (scopes, bindings)
│   │   │   ├── check.rs         # Type checking pass over Deep AST
│   │   │   ├── dims.rs          # Named dimension checking
│   │   │   ├── precision.rs     # Numeric precision type rules
│   │   │   ├── fitness.rs       # Graded type feedback (score, partial inference, suggestions)
│   │   │   └── errors.rs        # Structured error types with repair suggestions
│   │   └── tests/
│   │       ├── inference_tests.rs
│   │       ├── dimension_tests.rs
│   │       ├── precision_tests.rs
│   │       ├── fitness_tests.rs
│   │       └── fixtures/
│   │
│   ├── chelis-ir/               # RISC DAG (intermediate representation)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── dag.rs           # DAG data structure (nodes, edges, types)
│   │   │   ├── ops.rs           # The ~12 RISC primitive operations
│   │   │   ├── lower.rs         # Typed Deep AST → RISC DAG
│   │   │   ├── optimize.rs      # DAG optimizations (fusion, DCE, CSE)
│   │   │   ├── grad.rs          # Reverse-mode AD as DAG-to-DAG rewrite
│   │   │   ├── vmap.rs          # Vectorization as DAG-to-DAG rewrite (Phase 2)
│   │   │   └── verify.rs        # DAG invariant checking
│   │   └── tests/
│   │       ├── lower_tests.rs
│   │       ├── optimize_tests.rs
│   │       ├── grad_tests.rs
│   │       └── fixtures/
│   │
│   ├── chelis-backend-c/        # C code generation backend
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── emit.rs          # RISC DAG → C source code
│   │   │   ├── blas.rs          # BLAS call generation (OpenBLAS/MKL)
│   │   │   ├── memory.rs        # Memory planning and allocation in C
│   │   │   └── runtime.rs       # Minimal C runtime (tensor struct, alloc, free)
│   │   ├── runtime/             # C runtime source files (compiled alongside generated code)
│   │   │   ├── chelis_runtime.h
│   │   │   └── chelis_runtime.c
│   │   └── tests/
│   │       ├── emit_tests.rs
│   │       ├── numerical_tests.rs  # Correctness: known inputs → known outputs
│   │       └── fixtures/
│   │
│   └── chelis-cli/              # CLI binary (chelis command)
│       ├── Cargo.toml
│       └── src/
│           └── main.rs          # Subcommands: build, check, deep, surf, tide
│
├── tests/                       # Integration tests (end-to-end)
│   ├── e2e_mnist.rs             # Phase 0 goal: MNIST on CPU
│   ├── e2e_basic_ops.rs         # Basic tensor operations compile and run
│   ├── e2e_grad.rs              # AD produces correct gradients
│   ├── e2e_roundtrip.rs         # .ch → .dp → .ch roundtrip
│   └── fixtures/
│       ├── mnist.ch             # MNIST in Surf
│       ├── mnist.dp             # MNIST in Deep
│       └── basic_ops.ch
│
├── examples/                    # Example Chelis programs
│   ├── hello_tensor.ch          # Simplest possible program
│   ├── linear_regression.ch     # Minimal ML: grad + update loop
│   ├── mlp.ch                   # Multi-layer perceptron
│   └── mnist.ch                 # The Phase 0 deliverable
│
├── benchmarks/                  # Performance benchmarks (populated in Phase 1)
│   └── README.md
│
└── .github/
    └── workflows/
        ├── ci.yml               # Build + test on every PR
        ├── lint.yml             # Clippy + rustfmt
        └── spec-check.yml      # Verify spec docs are up to date
```

### Task Breakdown for 0a

```
0a-1: Create GitHub repo, initialize Cargo workspace, formatting config
      Create: rustfmt.toml, .clippy.toml, .editorconfig, pre-commit hook
      Test: `cargo build` succeeds with empty crates
      Test: `cargo fmt --check` and `cargo clippy` pass on empty crates
      
0a-2: Write spec/00-context.md
      Content: What Chelis is, the thesis (AI writing AI writing AI),
      target audience, what it's not, key influences, the turtle metaphor
      Test: Human review
      
0a-3: Write spec/01-nomenclature.md
      Content: Surf/Deep, .ch/.dp/.chb, shells/reef/tide,
      CLI subcommands, all terminology defined in one place
      Test: Human review
      
0a-4: Write spec/02-surf-syntax.md
      Content: Grammar sketch (EBNF or PEG), keyword list,
      operator precedence, example programs, design rationale
      Test: Human review; examples should be parseable by 0c
      
0a-5: Write spec/03-deep-syntax.md
      Content: S-expression grammar, tag vocabulary, metadata format,
      type annotation syntax, effect annotation, linearity markers,
      canonical form rules, example programs
      Test: Human review; examples should be parseable by 0b
      
0a-6: Write spec/04-type-system.md
      Content: ADTs, HM inference rules, precision types, named dims,
      tensor type syntax, type error format, fitness scoring algorithm
      Test: Human review; formal enough to implement from
      
0a-7: Write spec/05-risc-primitives.md
      Content: The ~12 ops with precise semantics (inputs, outputs,
      typing rules, AD adjoints), lowering from standard ML ops
      (matmul, conv2d, softmax, etc.) to RISC primitives
      Test: Human review; each op has at least one numerical test case
      
0a-8: Write remaining spec docs (06-12) as stubs with scope notes
      These get filled in as the corresponding phases are built
      
0a-9: Set up CI (GitHub Actions)
      - cargo fmt --check (formatting gate — from day one)
      - cargo clippy -D warnings (lint gate — from day one)
      - cargo build (all crates)
      - cargo test (all crates)
      Test: CI passes on empty crates, rejects unformatted code
      
0a-10: Write README.md and ARCHITECTURE.md
       README: Project intro, build instructions, phase status, links
       ARCHITECTURE: Crate dependency graph, data flow diagram,
       how a .ch file becomes a running program
       Test: A new contributor (or agent) can read these and orient
       
0a-11: Create issue templates and tracking labels
       Labels: phase-0a through phase-3, crate-deep, crate-surf,
       crate-types, crate-ir, crate-backend-c, crate-cli,
       spec, bug, enhancement, red-team
       Test: Issues can be filed and tracked
```

### Deliverable
A coding agent clones the repo, runs `cargo build`, reads `ARCHITECTURE.md`, and knows exactly what to build next (Phase 0b: the Deep parser). All spec documents for Phase 0 are written and reviewable. CI is green.

### Red Team Checkpoint: 0a
- Are the spec docs clear enough to implement from? Could a different agent read them and build the same thing?
- Is the crate dependency graph right? Are there circular dependencies?
- Is anything over-scoped for Phase 0? (Remove it.)
- Is the test strategy clear for each crate?

---

## Phase 0b: Deep Parser

**Goal:** Parse `.dp` files (s-expressions) into a Rust AST. Round-trip: parse → print → parse produces identical AST.

### What the Agent Builds

**`chelis-deep` crate:**

1. **Lexer** (`lexer.rs`): Tokenize s-expression text into: `LParen`, `RParen`, `Symbol`, `Integer`, `Float`, `String`, `Keyword` (`:keyword`), `Comment`. Handle whitespace and line comments (`;; ...`).

2. **AST** (`ast.rs`): Define the Deep AST types. Every node is:
   ```rust
   pub struct SExpr {
       pub tag: Symbol,
       pub meta: Metadata,
       pub children: Vec<SExprChild>,
       pub span: Span,  // source location
   }
   
   pub enum SExprChild {
       Atom(Atom),
       List(SExpr),
   }
   
   pub enum Atom {
       Symbol(String),
       Int(i64),
       Float(f64),
       Str(String),
       Keyword(String),
   }
   
   pub struct Metadata {
       pub ty: Option<Type>,         // filled by type checker
       pub effects: Option<Vec<Effect>>,  // filled by effect inference
       pub linear: Option<Linearity>,     // filled by linearity checker
   }
   ```

3. **Parser** (`parser.rs`): Recursive descent parser. `&[Token] → Result<Vec<SExpr>, ParseError>`. Errors include span information.

4. **Printer** (`printer.rs`): Canonical s-expression printer. `SExpr → String`. Deterministic: same AST always produces same text. This is the reference serialization.

### Test Strategy
- **Unit tests:** Each token type lexes correctly. Each s-expression form parses correctly.
- **Round-trip property tests:** For any valid `.dp` input, `print(parse(input)) == input` (modulo whitespace normalization). Use `proptest` or `quickcheck` for property-based testing.
- **Error tests:** Malformed inputs produce clear error messages with source locations.
- **Fixture files:** `tests/fixtures/` contains representative `.dp` files covering all AST node types.

### Acceptance Criteria
```
cargo test -p chelis-deep     # all pass
```
The Deep parser can parse the MNIST example in `.dp` format and round-trip it perfectly.

---

## Phase 0c: Surf Parser + Desugaring

**Goal:** Parse `.ch` files into Surf AST. Desugar Surf AST to Deep AST. The decompiler (Deep → Surf) is best-effort and comes later (0i).

### What the Agent Builds

**`chelis-surf` crate:**

1. **Lexer:** Tokenize Surf syntax. Keywords (`def`, `let`, `type`, `match`, `with`, `fn`, `module`), operators (`|>`, `->`, `=`, `|`, `:`, `+`, `*`, etc.), delimiters, identifiers, literals.

2. **AST:** Surf-specific AST types (richer than Deep — has pipes, sugar, formatting info).

3. **Parser:** PEG or recursive descent. Produces Surf AST. Consider using `pest` or `chumsky` crate for parser combinators.

4. **Desugarer** (`desugar.rs`): Surf AST → Deep AST. Mechanical, deterministic. Key transformations:
   - Pipes: `x |> f |> g` → `(pipe x f g)` or `(g (f x))`
   - Pattern matching: Surf `match`/`with` → Deep `(match ...)`
   - Type annotations: Surf `x: tensor[...]` → Deep `(: x (tensor ...))`
   - Function defs: Surf `def f(x: T): U = body` → Deep `(def f (sig (-> T U)) (fn (x) body))`
   - Let bindings: Surf `let x = e` → Deep `(let ((x e)) ...)`

### Test Strategy
- **Parse tests:** Each Surf construct parses to the expected Surf AST.
- **Desugar tests:** Each Surf construct desugars to the expected Deep AST. Test by comparing Deep printer output against expected `.dp` fixtures.
- **Round-trip tests:** Parse `.ch` → desugar to Deep → print Deep → parse Deep. The Deep form should be identical.
- **Error tests:** Syntax errors produce clear messages with line/column.

### Acceptance Criteria
The Surf parser can parse all example programs in `examples/`. Desugaring produces valid Deep ASTs that the Deep parser can round-trip.

---

## Phase 0d: Type Checker

**Goal:** Type-check Deep ASTs. Produce typed ASTs with type annotations in metadata. Produce graded fitness feedback for invalid programs.

This is the hardest and most important Phase 0 component. Plan for it to take 3-4 weeks.

### What the Agent Builds

**`chelis-types` crate:**

1. **Type representation** (`types.rs`): ADTs, function types, tensor types with precision and named dimensions, type variables for inference.

2. **Hindley-Milner inference** (`infer.rs` + `unify.rs`): Standard Algorithm W with extensions for tensor types. Unification of type variables. Let-polymorphism.

3. **Named dimension checking** (`dims.rs`): Nominal dimension types. `tensor[batch, hidden, f32]` and `tensor[batch, seq, f32]` are distinct. Dimension variables for generic functions.

4. **Precision type rules** (`precision.rs`): No silent promotion. `f32 + bf16` is a type error. Explicit `cast(x, bf16)` required.

5. **Fitness scoring** (`fitness.rs`): Given a Deep AST, produce a score 0.0-1.0 + structured error list + partial type annotations + repair suggestions. This is the compiler-as-training-signal feature.

6. **Type checking pass** (`check.rs`): Walk the Deep AST, call inference, fill in metadata types. Produce either a fully-typed AST or a fitness report.

### Test Strategy
- **Inference tests:** Known programs produce expected types. Polymorphic functions infer correctly. Let-polymorphism works.
- **Dimension tests:** Matching dimensions pass. Mismatched dimensions produce clear errors with the specific dimension names.
- **Precision tests:** Mixed-precision operations are caught. Explicit casts type-check.
- **Fitness tests:** Programs with 0, 1, N errors produce expected scores. Partial inference annotates what it can.
- **Negative tests:** Every expected error condition has a test case that triggers it.

### Red Team Checkpoint: 0d
This is the most critical checkpoint. Questions:
- Does the type system match the spec? Are there spec ambiguities the implementation had to resolve?
- Are tensor dimension semantics sound? Can you construct a program that type-checks but is dimensionally wrong?
- Is the fitness scoring useful? Does a "closer to correct" program get a higher score?
- Is the error message quality high enough for a PhD student (or an AI agent) to act on?

---

## Phase 0e: RISC DAG

**Goal:** Lower typed Deep ASTs to the RISC DAG (the ~12 primitive operations). Implement basic DAG optimizations.

### What the Agent Builds

**`chelis-ir` crate:**

1. **DAG structure** (`dag.rs`): Nodes are RISC ops with typed inputs/outputs. Edges are data dependencies. The DAG is the central IR — everything downstream operates on it.

2. **RISC ops** (`ops.rs`): The ~12 primitives: `Add`, `Mul`, `Div`, `CmpLt`, `Max`, `Neg`, `Exp`, `Log`, `Sin`, `Sqrt` (elementwise), `ReduceSum`, `ReduceMax` (reductions), `Reshape`, `Permute`, `Expand`, `Pad`, `Shrink`, `Stride` (movement), `Load`, `Store`, `Const` (memory).

3. **Lowering** (`lower.rs`): Typed Deep AST → RISC DAG. Standard operations lower to primitives: `matmul(A, B)` → a sequence of reshape + expand + mul + reduce_sum. `softmax(x)` → exp + reduce_sum + div. `relu(x)` → max(x, 0). Document each lowering.

4. **Optimizations** (`optimize.rs`): For Phase 0, just: constant folding, dead code elimination, common subexpression elimination. Fusion comes in Phase 1.

5. **Verification** (`verify.rs`): Check DAG invariants: no cycles, types consistent at every edge, all inputs connected, no dangling nodes.

### Test Strategy
- **Lowering tests:** Each standard op (matmul, softmax, relu, conv2d, layer_norm, etc.) lowers to the expected RISC primitive sequence.
- **Numerical tests:** Lower a known operation, evaluate the DAG on known inputs, compare against reference output (NumPy or manual calculation).
- **Optimization tests:** DCE removes unused nodes. CSE merges identical subgraphs. Constant folding evaluates static expressions.
- **Verification tests:** Valid DAGs pass verification. Invalid DAGs (cycles, type mismatches) are caught.

---

## Phase 0f: C Backend

**Goal:** Generate C code from RISC DAGs. Compile and run the generated C. Verify numerical correctness.

### What the Agent Builds

**`chelis-backend-c` crate:**

1. **C emission** (`emit.rs`): Walk the RISC DAG in topological order. Emit C code for each node. Elementwise ops become loops. Reductions become loops with accumulators. Movement ops become index transformations.

2. **BLAS integration** (`blas.rs`): Pattern-match RISC DAG subgraphs that correspond to BLAS operations (matmul → `cblas_sgemm`, etc.). Emit BLAS calls instead of naive loops.

3. **Memory planning** (`memory.rs`): Analyze DAG node lifetimes. Allocate buffers. Reuse buffers when lifetimes don't overlap. Emit `malloc`/`free` calls.

4. **C runtime** (`runtime/`): A minimal C header + source providing: tensor struct (pointer + shape + strides + dtype), allocation, deallocation, basic I/O (load/save tensors). This ships alongside generated code.

### Architecture (Following Futhark)
The generated output for a Chelis program `model.ch` is:
```
model.c       # Generated C source (host code + tensor operations)
model.h       # Generated C header (public API)
```
Compiled with: `gcc -O2 -o model model.c chelis_runtime.c -lopenblas -lm`

### Test Strategy
- **Emission tests:** Each RISC op emits correct C code (inspect output).
- **Numerical tests:** The critical tests. Generate C for known operations, compile, run, compare output against reference values. Tolerance: 1e-6 for f32, 1e-12 for f64. Test at LEAST: add, mul, matmul, softmax, relu, reduce_sum, reshape, transpose.
- **BLAS tests:** Verify BLAS path produces identical results to naive path.
- **Memory tests:** Verify no memory leaks (Valgrind or AddressSanitizer on generated C).

---

## Phase 0g: Automatic Differentiation

**Goal:** Implement `grad` as a DAG-to-DAG rewrite. Reverse-mode AD.

### What the Agent Builds

**In `chelis-ir` crate, `grad.rs`:**

1. **Adjoint rules** for each RISC primitive. These are the backward-pass rules:
   - `Add(a, b)` adjoint: `(grad_out, grad_out)`
   - `Mul(a, b)` adjoint: `(grad_out * b, grad_out * a)`
   - `Exp(a)` adjoint: `(grad_out * exp(a))`
   - `ReduceSum(a, axis)` adjoint: `expand(grad_out, axis)`
   - etc. — each of the ~12 ops needs an adjoint rule.

2. **DAG-to-DAG rewrite:** Given a forward DAG (input → output), produce a backward DAG (output_grad → input_grad). The backward DAG reuses forward DAG nodes where needed.

3. **`grad(f)` function transform:** Takes a function (as a DAG), returns a new function that computes both the output and the gradient with respect to specified parameters.

### Test Strategy
- **Adjoint correctness:** For each op, compare AD gradient against numerical finite-difference gradient. This is the gold standard test for AD.
- **Composition tests:** `grad` of a multi-op pipeline (e.g., matmul → relu → sum) produces correct gradients for all parameters.
- **Second-order tests:** `grad(grad(f))` works for twice-differentiable functions.
- **Known-answer tests:** Gradient of `sum(x^2)` is `2x`. Gradient of `softmax_cross_entropy` matches the known closed-form formula.

---

## Phase 0h: MNIST End-to-End

**Goal:** A Chelis program that trains a simple neural network on MNIST, running on CPU via the C backend.

### What This Proves
The full pipeline works: `.ch` source → Surf parse → desugar to Deep → type check → lower to RISC DAG → apply `grad` → emit C → compile → run → learn.

### The Program
A simple MLP (2 layers, ReLU activation, softmax cross-entropy loss, SGD optimizer). Trains for a few epochs. Achieves >90% accuracy (proving the gradients are correct, not just that it compiles).

### Test Criteria
- Program type-checks with no errors
- Compiles to C without errors
- Trains and achieves >90% accuracy on MNIST test set
- Gradients match numerical finite-difference check (sampled)
- No memory leaks in generated C

---

## Phase 0i: Tide v0.1

**Goal:** Interactive REPL that accepts Deep s-expressions, type-checks them, and returns results with fitness scores.

### What the Agent Builds

**In `chelis-cli` crate:**

1. **`chelis tide`** — REPL mode. Reads lines from stdin, parses as Deep, type-checks, prints typed result or fitness report.
2. **`chelis deep app.ch`** — Desugar Surf to Deep, print Deep text.
3. **`chelis surf program.dp`** — Decompile Deep to Surf (best-effort), print Surf text.
4. **`chelis build app.ch`** — Full pipeline: parse → type-check → lower → emit C → compile.
5. **`chelis check app.ch`** — Type-check only, print fitness report (JSON).

### Test Strategy
- **CLI tests:** Each subcommand produces expected output for known inputs. Use `assert_cmd` crate for CLI testing.
- **REPL tests:** Scripted REPL sessions (pipe input, check output).
- **Fitness report tests:** `chelis check` on programs with known errors produces expected JSON output.

---

## Agent Working Conventions

### Formatting and Linting

**Rust code (the compiler):** Enforced from the first commit. No exceptions.

- **`rustfmt`** with a checked-in `rustfmt.toml` at workspace root. Settings: `edition = "2021"`, `max_width = 100`, `use_small_heuristics = "Max"`. Every `.rs` file is formatted before commit. CI fails on unformatted code.
- **`clippy`** with `#![deny(clippy::all)]` in every crate's `lib.rs`/`main.rs`. Specific lints to enable: `clippy::pedantic` (with targeted `#[allow]` where justified and commented). CI fails on clippy warnings.
- **Config files:** `rustfmt.toml` and `.clippy.toml` are created in 0a-1, before any code is written. The first PR that adds code to any crate must pass both.

**Chelis Deep (.dp files):** The canonical printer IS the formatter. By design, Deep has exactly one textual representation per AST. The `printer.rs` in `chelis-deep` (Phase 0b) defines this. Any `.dp` file in the repo must be the output of the canonical printer — enforced by a CI check that round-trips every `.dp` fixture and diffs.

**Chelis Surf (.ch files):** A `chelis fmt` subcommand, but not in Phase 0. The approach: parse Surf → desugar to Deep → decompile back to Surf using the canonical decompiler style. This means the Surf formatter is a byproduct of the round-trip machinery, not a separate tool. Added in Phase 0i when the decompiler is built. Until then, `.ch` fixtures in the repo follow a manually maintained style guide in `spec/02-surf-syntax.md`.

**Spec docs and markdown:** No automated formatter (markdown formatters are opinionated and noisy). Instead, a style convention in `CONTRIBUTING.md`: 80-char soft wrap, ATX headers, fenced code blocks with language tags, one sentence per line in prose.

**CI enforcement order:**
```yaml
# .github/workflows/ci.yml
jobs:
  format:
    - cargo fmt --all -- --check          # Rust formatting
    - cargo clippy --all -- -D warnings   # Rust linting
  
  deep-canonical:                          # Added in Phase 0b
    - cargo run -p chelis-cli -- deep-check tests/fixtures/*.dp
    # Verifies every .dp fixture is in canonical form
  
  test:
    - cargo test --all                    # All crate tests
```

**When each enforcement starts:**

| Check | Starts at | Blocks PR |
|---|---|---|
| `cargo fmt --check` | 0a-1 (first commit) | Yes |
| `cargo clippy -D warnings` | 0a-1 (first commit) | Yes |
| Deep canonical form check | 0b (when printer exists) | Yes |
| `chelis fmt` for .ch files | 0i (when decompiler exists) | Yes |

### Git Hygiene

- **Pre-commit hook** (via `cargo-husky` or a shell script): runs `cargo fmt` and `cargo clippy` locally before allowing commit. Set up in 0a-1. This prevents unformatted code from ever entering the history.
- **`.editorconfig`** at repo root: `indent_style = space`, `indent_size = 4` for Rust, `indent_size = 2` for `.ch`/`.dp`/`.toml`/`.yml` files. Catches editor misconfiguration early.

### For Every Task

1. **Read the relevant spec doc first.** Before writing any code, the agent reads the corresponding `spec/*.md` file. If the spec is ambiguous, file an issue and propose a resolution before coding.

2. **Write tests before or alongside code.** Every function has at least one test. Public APIs have property tests where applicable. The test is the specification.

3. **Document as you go.** Every public type and function has a doc comment. The doc comment explains WHAT (not HOW) and links to the spec section.

4. **Small, reviewable PRs.** One logical change per PR. Each PR includes: code, tests, doc updates, and a description linking to the task/issue.

5. **Track status.** Every task has a GitHub issue. Issues are labeled by phase and crate. The issue is updated with progress notes. When done, the issue is closed by the merging PR.

### Red Team Protocol

After each checkpoint phase, a red team review:

1. **Spec compliance:** Does the implementation match the spec? Where did it diverge? Document the divergences.
2. **Test coverage:** Are there untested code paths? What inputs could break the system?
3. **Adversarial inputs:** Craft pathological programs (huge types, deeply nested expressions, circular type references). Does the compiler handle them gracefully?
4. **Architectural review:** Is the crate boundary in the right place? Are there abstractions that should be refactored before building more on top?
5. **Write a findings document.** List every issue found. Categorize as: must-fix (before next phase), should-fix (soon), nice-to-fix (backlog).

---

## Phase 1: Futhark-Style GPU Backend

**Prerequisite:** Phase 0h complete (MNIST runs on CPU, full pipeline proven).
**Deliverable:** The same MNIST program (and harder models — transformer block, CNN) compiles to GPU via CUDA or OpenCL and produces correct results.
**Timeline target:** Months 6-9.

### 1a: Kernel Code Generation (Weeks 1-3)

Extend the C backend to emit GPU kernel strings alongside host code. The Futhark architecture: generated C file contains host code (CPU) + embedded kernel source as a string literal.

- **Kernel string emission:** RISC DAG nodes that are parallelizable emit CUDA C++ or OpenCL C kernel source. Elementwise ops → one thread per element. Reductions → parallel reduction patterns. Movement ops → index transformations.
- **Host-side dispatch:** Generated C host code calls NVRTC (NVIDIA) or `clBuildProgram` (OpenCL) to JIT-compile the kernel string, then launches it.
- **Two targets from day one:** CUDA and OpenCL. Same RISC DAG, two different kernel string emitters. This is cheap because the kernel shapes are nearly identical — just syntactic differences.

Test: A simple elementwise operation (vector add) compiles, runs on GPU, produces correct output.

### 1b: Fusion (Weeks 4-6)

The critical optimization. Adjacent RISC DAG nodes that can share a GPU kernel should be fused into one kernel launch (fewer memory round-trips).

- **Elementwise fusion:** Chain of elementwise ops (add → relu → mul) becomes one kernel.
- **Reduce-elementwise fusion:** Elementwise ops feeding into a reduction (or following one) fuse.
- **Fusion rules:** Implemented as DAG-to-DAG rewrite rules in `chelis-ir/optimize.rs`. Each rule has a correctness test (fused result == unfused result).
- **Fusion does NOT cross:** Memory barriers, reductions on different axes, or scatter/gather boundaries.

Test: Fused matmul+relu produces identical output to unfused, in fewer kernel launches.

### 1c: Memory Planning for GPU (Weeks 7-8)

GPU VRAM is scarce. The compiler must plan buffer allocation, reuse, and host↔device transfer.

- **Buffer lifetime analysis:** From the DAG, compute when each tensor is first needed and last used on GPU.
- **Buffer reuse:** When a buffer's lifetime ends, its VRAM can be reclaimed for a later tensor of the same size.
- **Host↔device transfer minimization:** Only transfer tensors that are needed on GPU. Keep results on GPU between kernel launches when possible.
- **Memory budget mode:** Given a VRAM budget, insert recomputation points (activation checkpointing) to fit.

### 1d: Flattening and Parallelization (Weeks 9-11)

Futhark's core contribution: flatten nested parallel operations into flat GPU thread grids.

- **Map-of-map flattening:** `map(fn x -> map(fn y -> ..., x), data)` → single kernel over 2D grid.
- **Segmented reductions:** Reduce within segments of an array, parallelized.
- **Thread block sizing:** Heuristic selection of block dimensions based on operation shape.

### 1e: Benchmarking and Real Models (Weeks 12-14)

- **Benchmark suite:** MLP, CNN (LeNet), Transformer block (single layer), Linear regression. Each has a reference PyTorch implementation for numerical comparison.
- **Performance comparison:** Wall-clock time vs. PyTorch on the same hardware. Goal is NOT to beat PyTorch — goal is to be within 2-5x for Phase 1, proving the architecture works. Optimization is ongoing.
- **Correctness gate:** Every benchmark must produce numerically identical results (within tolerance) to the C backend.

### Red Team Checkpoint: Phase 1
- Are kernel outputs bitwise-identical (within tolerance) to CPU backend? On both CUDA and OpenCL?
- Does fusion preserve correctness in all cases? Adversarial fusion tests.
- Memory planning: does it actually prevent OOM on real models? Test with constrained VRAM.
- Performance: where are the bottlenecks? Profile and document.

---

## Phase 2: Language Maturity

**Prerequisite:** Phase 1 complete (GPU backend working, real models running).
**Deliverable:** The language is usable by researchers. Effects, linear types, macros, and the agent API make it a credible alternative to PyTorch for specific workloads.
**Timeline target:** Months 9-14.

### 2a: Algebraic Effects (Weeks 1-4)

Implement the Dex-inspired effect system. This is the second major type system feature (after HM + tensor types from Phase 0d).

- **Three built-in effects:** `Diff` (differentiability), `Random` (stochasticity), `Resource(Device)` (allocation).
- **Effect inference:** Walk the typed Deep AST, infer which effects each function has based on the operations in its body. No annotations needed from the programmer.
- **Effect handlers:** `grad` handles `Diff`. `handleRandom(seed)` handles `Random`. `handleGPU(device)` handles `Resource`.
- **Effect errors:** "Unhandled effect `Random` in function `predict` — this function calls `dropout` at line 42, which introduces the `Random` effect. Handle it with `handleRandom(seed)` for deterministic inference."
- **Spec work required BEFORE building:** Formal effect typing rules. Composition rules (what happens when effects nest). Interaction with existing HM inference.

### 2b: Linear Types for Tensors (Weeks 5-8)

Futhark-style uniqueness types. Tensors consumed exactly once.

- **Linear binding:** A tensor variable can be used exactly once. Using it again is a compile error.
- **Borrowing:** `&tensor` for read-only access without consuming. A borrowed reference cannot be stored or returned.
- **Explicit copy:** `tensor.copy()` to duplicate. Makes memory cost visible.
- **Compiler optimization:** When linearity is satisfied, the compiler can perform in-place mutation (reuse the buffer) safely. This is a significant memory win.
- **Interaction with effects:** `Resource` effect tracks allocation. Linear types track deallocation. Together they prevent GPU memory leaks.
- **Spec work required BEFORE building:** Linearity checking rules. Which types are linear (all tensors? only GPU tensors? configurable?). How borrowing interacts with function calls, closures, pattern matching.

### 2c: Macro System (Weeks 9-12)

Hygienic, type-aware macros operating on Deep s-expressions.

- **Macro definition:** `(defmacro name (params) body)` in Deep. Macros receive Deep AST fragments and return Deep AST fragments.
- **Hygiene:** Variables introduced by macro expansion are automatically renamed to avoid capture (Racket-style). Manual hygiene breaking is possible but discouraged.
- **Type-aware macros:** A macro can call the type checker during expansion. This enables: macros that validate their input types, macros that generate type-correct output, compile-time assertions.
- **Phase separation:** Macros run at compile time, before full type checking. Macro bodies are type-checked independently.
- **Spec work required BEFORE building:** Macro expansion order. Interaction with effects and linearity. What standard macros ship with the language (e.g., `@differentiable` as a macro that checks the Diff effect).

### 2d: vmap Transform (Weeks 13-14)

Automatic vectorization as a DAG-to-DAG rewrite, following JAX's model.

- **`vmap(f)`:** Takes a function over single examples, returns a function over batches. Adds a batch dimension to every operation in the DAG.
- **Interaction with `grad`:** `grad(vmap(f))` and `vmap(grad(f))` must both work and produce correct results (per-example gradients).

### 2e: Tide Agent API + MCP (Weeks 15-18)

The machine-facing interface.

- **HTTP/JSON API:** Endpoints: `/compile`, `/check`, `/desugar`, `/decompile`, `/eval`. Each accepts Surf or Deep, returns typed results + fitness scores.
- **MCP server:** Wraps the HTTP API as MCP tools. A coding agent connects to Chelis via MCP and uses `chelis_compile`, `chelis_check`, etc. as tools.
- **Streaming:** Long-running compilations stream progress. Evaluation streams intermediate results.
- **Batch mode:** Submit multiple programs, get results for all. Useful for evolutionary loops running externally.

### Red Team Checkpoint: Phase 2
- Effects: Can you construct a program with unhandled effects that the compiler doesn't catch?
- Linear types: Can you leak GPU memory despite linearity? Can you trigger a use-after-free?
- Macros: Can a macro break type safety? Can it escape hygiene unintentionally?
- Agent API: Security review — can a malicious program crash the compiler or escape the sandbox?

---

## Phase 3: Ecosystem

**Prerequisite:** Phase 2 complete (the language is feature-complete for v1).
**Deliverable:** Chelis has a package ecosystem, multiple backends, Python interop, and research-grade type system extensions.
**Timeline target:** Months 14+. Open-ended; items are prioritized by demand.

### 3a: Package System (Shells + Reef)

- **`reef.toml`:** Project manifest (name, version, dependencies, build config). Cargo.toml-inspired.
- **Shell format:** A published shell is a `.chb` (binary typed AST) + manifest. Consumers get types without compiling from source.
- **Reef registry:** `reef.chelis.ch` — shell discovery, publishing, version resolution. Can start as a simple static file index (like early crates.io).
- **Dependency resolution:** Semantic versioning. Lock file for reproducibility.

### 3b: StableHLO Backend

- **Direct emission:** RISC DAG → StableHLO operations. Following the Nx/EXLA pattern (not via JAX tracing).
- **TPU access:** The primary motivation. StableHLO is the only serious path to Google TPUs.
- **Alternative GPU path:** StableHLO → XLA → GPU code. Useful for comparison against the Futhark-style backend.

### 3c: FX Graph Backend

- **RISC DAG → FX operator graph:** Map the ~12 primitives to ATen operators.
- **TorchInductor:** FX graphs compile via TorchInductor to Triton kernels (NVIDIA), C++/OpenMP (CPU), ROCm (AMD).
- **torch.export → ExecuTorch:** Edge deployment path.
- **Use case:** Interop with PyTorch ecosystem. A Chelis model can be exported as a PyTorch module.

### 3d: Python FFI

- **DLPack:** Zero-copy tensor exchange between Chelis and Python (NumPy, PyTorch, JAX).
- **PyO3:** Python bindings for the Chelis compiler. `import chelis; chelis.compile("program.ch")` from Python.
- **GIL release:** During Chelis computation, the Python GIL is released. Python callbacks cannot appear inside `jit`/`grad`.
- **Use case:** Incremental adoption. Use Chelis for the model, Python for data loading/visualization/orchestration.

### 3e: Research Type Features

These are publication-grade type system extensions. Each should be a paper before it's an implementation.

- **Distribution types:** For probabilistic models, VAEs, diffusion. `Distribution(Normal, {mean: tensor, std: tensor})`. Sampling is explicitly `Random` effect. KL divergence defined between `Distribution` types.
- **Equivariance constraints:** Track symmetry groups through model composition. `@equivariant(Translation2D)`. Symmetry-breaking must be explicit. Novel type system research — most publishable component.
- **Optimization properties:** `@convex`, `@lipschitz(1.0)`, `@idempotent`. Trusted annotations initially; compiler verification later.

### 3f: Tide LSP

- **Language Server Protocol:** Full IDE integration. Type checking, completion, hover info, go-to-definition, diagnostics.
- **VS Code extension:** First target.
- **JetBrains plugin:** If demand warrants.

---

## Design Work Pipeline

Design and implementation run in parallel. Design work produces spec documents; implementation consumes them. The rule: a spec doc must be written and reviewed BEFORE the corresponding implementation phase begins.

### Design Work Available Now (During Phase 0)

These can be worked on immediately, in parallel with Phase 0 coding. They don't require a working compiler — they're paper/spec work.

| Design Task | Produces | Consumed By | Priority |
|---|---|---|---|
| **Surf formal grammar** | spec/02-surf-syntax.md with full EBNF/PEG | Phase 0c (Surf parser) | **HIGH — blocks 0c** |
| **Deep formal grammar** | spec/03-deep-syntax.md with tag vocabulary, metadata format, canonical form rules | Phase 0b (Deep parser) | **HIGH — blocks 0b** |
| **RISC primitive semantics** | spec/05-risc-primitives.md with precise I/O types, broadcasting rules, AD adjoints | Phase 0e (DAG), 0g (grad) | **HIGH — blocks 0e** |
| **Type system formal rules** | spec/04-type-system.md with inference rules, tensor type algebra, dimension checking rules, fitness scoring algorithm | Phase 0d (type checker) | **HIGH — blocks 0d** |
| **Standard op lowerings** | Appendix to spec/05 mapping matmul, conv2d, softmax, layer_norm, etc. to RISC primitives | Phase 0e (lowering) | **MEDIUM** |
| **Effect system design** | spec/04-type-system.md §effects with formal effect typing rules, composition, interaction with HM | Phase 2a | **MEDIUM — long lead** |
| **Linear type design** | spec/04-type-system.md §linearity with checking rules, borrowing, interaction with effects | Phase 2b | **MEDIUM — long lead** |
| **Macro system design** | New section in spec/03-deep-syntax.md or dedicated doc. Expansion order, hygiene, type-awareness, phase separation | Phase 2c | **LOW — can wait** |
| **Error message catalog** | Reference doc: every error type with code, message template, repair suggestions | Phase 0d onwards | **MEDIUM — iterative** |
| **Surf module system** | spec/02 §modules. Namespacing, imports, exports, visibility. Simple for v1. | Phase 0c | **MEDIUM** |
| **Named dimension declaration syntax** | spec/02 §dimensions. Module-level? Per-function? Both? How do dimensions scope? | Phase 0c, 0d | **MEDIUM** |

### Design Decisions Still Open

These need to be resolved before the corresponding phase. Ranked by urgency.

**Must resolve before Phase 0 coding begins:**

1. **Deep tag vocabulary.** What are the exact tags in the s-expression language? The spec has examples (`def`, `fn`, `sig`, `let`, `match`, `pipe`, `module`, `type`, `variant`, `record`, `field`, `tensor`, `dim`) but no exhaustive list. This is the vocabulary an AI agent generates from — it must be precise and complete.

2. **Deep metadata format.** The spec says "metadata is a property list" but doesn't specify the format. Is it `(def ^{:type (-> int32 int32)} f ...)` (Clojure-style reader metadata)? Is it `(def (meta :type (-> int32 int32)) f ...)` (explicit meta node)? Or is metadata implicit and only filled in by compiler passes, never written by hand? This affects both the parser and the AI generation interface.

3. **Tensor type syntax in Deep.** Currently `(tensor (dim batch) (dim seq) f32)`. Is precision always last? Can dimensions be unnamed? What about rank-only types `(tensor 3 f32)` for "any 3D f32 tensor"? What about wildcard dimensions `(tensor * * f32)`?

4. **How `pipe` desugars.** The spec says `(pipe x f g h)` desugars to `(h (g (f x)))`. But are `f`, `g`, `h` function names? Partial applications? What if `f` takes two arguments and the pipe supplies the first? Need precise semantics for the pipe in Deep.

**Should resolve before Phase 1:**

5. **Fusion rules.** Which RISC DAG patterns can fuse? This is partly an implementation concern but partly a language semantics question (does fusion change observable behavior? It shouldn't, but precision changes might).

6. **GPU memory model.** How does the language model GPU memory? Is it an effect (`Resource(GPU(0))`)? An annotation? Invisible? This affects both the type system (Phase 2a) and the GPU backend (Phase 1).

**Should resolve before Phase 2:**

7. **Effect handler syntax.** How do you handle effects in Surf? `handle Diff with grad { ... }`? `with_grad { ... }`? Something else? How does it look in Deep?

8. **Borrow syntax.** How does borrowing appear in Surf? `&x`? `borrow x`? How in Deep? `(borrow x ...)` or `(& x)`?

9. **Custom effects.** Can users define their own effects? If so, how? If not, is the effect system extensible later?

### Recommended Design Sprint

Before the agent starts coding Phase 0b, do a focused design sprint resolving items 1-4 above. This produces:

- **spec/03-deep-syntax.md** — Complete tag vocabulary, metadata format, canonical form rules. Every valid Deep program can be unambiguously constructed from this document.
- **spec/02-surf-syntax.md** — Full grammar. Every valid Surf program can be parsed from this document.
- **spec/05-risc-primitives.md** — Complete op table with types and AD rules.
- **spec/04-type-system.md** — Inference rules precise enough to implement from.

This sprint is 3-5 days of concentrated design work. It's the highest-leverage pre-coding activity because every implementation phase consumes these documents. Getting them right avoids rework across 0b through 0h.

