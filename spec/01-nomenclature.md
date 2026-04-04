# Chelis Language Specification: Nomenclature

**Version:** 0.1.0-draft
**Status:** Authoritative specification draft

---

This document defines every term used in the Chelis specification. Terms are grouped by category. Within each category, entries are alphabetical. When a term's definition references another defined term, it appears in **bold** on first use within that definition.

---

## 1. Syntax Layers

### Deep

The canonical s-expression syntax of Chelis. File extension: `.dp`. Every Chelis program has exactly one **Deep** representation (up to metadata ordering and whitespace normalization). The compiler operates on Deep form internally. Deep is machine-readable and human-inspectable but not intended as the primary authoring format.

Deep form uses Lisp-style parenthesized prefix notation. The first element of any list is always a **tag** that identifies the form (e.g., `def`, `let`, `fn`, `apply`).

Example:
```
(def square (sig (-> f32 f32)) (fn (x) (apply mul x x)))
```

### Surf

The human-readable surface syntax of Chelis. File extension: `.ch`. Designed for both human programmers and AI coding agents to read and write comfortably. Surf uses infix operators, indentation-sensitive or brace-delimited blocks (author's choice), and familiar ML-family syntax.

Surf desugars mechanically and deterministically to **Deep**. The desugaring is defined in `03-deep-syntax.md`.

Example:
```
def square(x: f32): f32 = x * x
```

---

## 2. File Extensions

### `.ch` -- Surf Source File

A text file containing Chelis source code in **Surf** syntax. This is the primary authoring format. UTF-8 encoded. The `chelis build` command accepts `.ch` files as input.

### `.chb` -- Chelis Binary

A compiled artifact produced by the `chelis build` pipeline. Contains a serialized **RISC DAG**, type metadata, and target-specific compiled code. The format is:

- Magic bytes: `CHEL` (4 bytes)
- Version: semver triple (3 x u16, 6 bytes)
- DAG section: serialized RISC DAG in a binary encoding
- Metadata section: JSON-encoded type signatures, dimension names, and compilation options
- Code section(s): target-specific compiled code (one per backend)

`.chb` files are portable across machines with the same target architecture.

### `.dp` -- Deep Source File

A text file containing Chelis source code in **Deep** (s-expression) syntax. UTF-8 encoded. The compiler accepts `.dp` files anywhere it accepts `.ch` files. Useful for:

- Machine-generated code (agents may find it easier to emit s-expressions)
- Debugging (inspecting the compiler's internal representation)
- Round-trip testing (Surf -> Deep -> Surf should be semantically identical)

---

## 3. The Ocean Metaphor

Chelis is named after Chelonia (turtles). The ocean metaphor provides a consistent naming scheme across the entire system.

### Current

The concurrency and data-flow model. Named after ocean currents: data flows through a Chelis program like water through currents. The Current model is:

- **Dataflow-oriented**: Operations execute when their inputs are ready.
- **Pure by default**: Functions have no side effects unless explicitly annotated.
- **Deterministic**: The same inputs always produce the same outputs, regardless of execution order.

Current is not yet specified. It will be defined in a future specification document.

### Deep

See **Syntax Layers > Deep** above. In the ocean metaphor: the deep ocean, where things happen beneath the surface. The compiler's internal world.

### Reef

The package ecosystem and registry. Named after coral reefs: interconnected, growing, supporting a diverse ecosystem. Reef will provide:

- A package format (a `.chb` file with additional metadata)
- A registry (centralized or federated)
- Dependency resolution (semantic versioning)
- Reproducible builds (lock files)

Reef is not yet specified. It will be defined in a future specification document.

### Shells

Compiled output artifacts. Named after turtle shells: hard, protective, self-contained. A **Shell** is a `.chb` file -- the portable compiled form of a Chelis module. The terminology extends to any compiled output: a Shell can be a `.chb`, a generated `.c` file, a `.cu` CUDA kernel, or a StableHLO module.

### Surf

See **Syntax Layers > Surf** above. In the ocean metaphor: the ocean's surface, where humans look and interact.

### Tide

The interactive mode: REPL and agent API. Named after tides: they come and go, they're stateful (the tide is in or out), and they represent the boundary between the user and the system.

Tide provides:
- An interactive REPL (`chelis tide`) where expressions are evaluated incrementally.
- An agent API (JSON over stdin/stdout) for AI coding agents to interact with the compiler programmatically.
- Stateful sessions: bindings persist across interactions within a session.

Tide is not yet fully specified. The CLI entry point is defined; the agent API protocol will be defined in a future document.

---

## 4. Compiler Pipeline Stages

The Chelis compiler is organized as a sequence of stages. Each stage transforms one representation into the next. The full pipeline is:

```
Source text (.ch or .dp)
    |
    v
  Parse      -- Text -> AST
    |
    v
  Desugar    -- Surf AST -> Deep AST (skipped if input is .dp)
    |
    v
  Check      -- Deep AST -> Typed Deep AST
    |
    v
  Lower      -- Typed Deep AST -> RISC DAG
    |
    v
  Transform  -- RISC DAG -> RISC DAG (rewrites)
    |
    v
  Emit       -- RISC DAG -> target code
    |
    v
  Compile    -- target code -> executable / .chb
```

### Parse

**Input:** Source text (UTF-8 string).
**Output:** Untyped AST.

Parses either **Surf** or **Deep** syntax, producing an abstract syntax tree. The parser auto-detects the syntax from the file extension (`.ch` vs `.dp`). Parse errors include line/column numbers and contribute to the **fitness score** (a file that doesn't parse gets a score near 0.0, proportional to how much of the file parsed successfully).

### Desugar

**Input:** Surf AST.
**Output:** Deep AST.

Mechanically translates the Surf AST into the Deep AST. This is a syntax-to-syntax transformation with no semantic analysis. Every Surf construct has a defined Deep equivalent (see `03-deep-syntax.md` for the complete desugaring table). If the input is already a `.dp` file, this stage is skipped.

### Check

**Input:** Deep AST (untyped).
**Output:** Typed Deep AST (every node annotated with its inferred type).

Performs **Hindley-Milner** type inference and checking. This stage:

1. Infers types for all expressions using Algorithm W (or a modern variant).
2. Checks that tensor dimensions match at every operation.
3. Verifies that precision types are used consistently (no implicit coercion).
4. Checks pattern match exhaustiveness.
5. Produces the **fitness score** and **repair suggestions** for any errors found.

The Check stage is the heart of the Chelis compiler. It is the primary source of the compiler's value as a collaborator.

### Lower

**Input:** Typed Deep AST.
**Output:** **RISC DAG**.

Translates the high-level typed AST into a directed acyclic graph of **RISC primitives**. This stage:

1. Eliminates pattern matching (compiles to conditional branches).
2. Eliminates ADTs (compiles to tagged representations).
3. Translates tensor operations into RISC primitive sequences.
4. Assigns dimension sizes (resolves named dimensions to concrete or symbolic integers).

After lowering, the program is a flat DAG of ~12 primitive operations over tensors.

### Transform

**Input:** RISC DAG.
**Output:** RISC DAG (rewritten).

Applies DAG-to-DAG rewrites. The three language-level transformations are applied here:

- **grad**: Inserts adjoint nodes to compute gradients via reverse-mode automatic differentiation.
- **vmap**: Lifts the DAG to operate over an additional batch dimension.
- **jit**: Marks subgraphs for just-in-time compilation at runtime.

Additionally, optimization passes run in this stage:
- Operator fusion (e.g., fusing element-wise ops into a single kernel)
- Constant folding
- Dead node elimination
- Common subexpression elimination

### Emit

**Input:** RISC DAG (optimized).
**Output:** Target-specific source code.

Translates the RISC DAG into code for a specific backend:

- **C**: Portable CPU code. The default backend.
- **CUDA**: GPU kernel code for NVIDIA hardware.
- **StableHLO**: The standard IR for ML compilers (interop with XLA, IREE, etc.).

Each RISC primitive has an emission template per backend. Emit is a relatively mechanical translation.

### Compile

**Input:** Target-specific source code.
**Output:** Executable binary or `.chb` file.

Invokes the target compiler (e.g., `gcc`, `nvcc`, or an HLO compiler) to produce a runnable artifact. This stage is a thin wrapper around external tools.

---

## 5. CLI Subcommands

The `chelis` command-line tool exposes the compiler pipeline through subcommands. Each subcommand runs the pipeline up to a specific stage and outputs the result.

### `chelis build <file>`

Run the full pipeline: Parse -> Desugar -> Check -> Lower -> Transform -> Emit -> Compile. Produces a compiled artifact (executable or `.chb`).

**Flags:**
- `--backend <c|cuda|stablehlo>`: Choose emission backend. Default: `c`.
- `--output <path>`: Output file path. Default: derived from input filename.
- `--optimize <0|1|2|3>`: Optimization level. Default: `2`.
- `--json`: Emit compiler output (errors, fitness score) as JSON instead of human-readable text.

### `chelis check <file>`

Run Parse -> Desugar -> Check only. Does not lower, transform, or compile. Outputs:

- **Fitness score**: A float in [0.0, 1.0].
- **Type errors**: If any, with line/column, expected vs. actual type, and repair suggestions.
- **Inferred types**: For all top-level definitions.

This is the primary command for AI coding agents. It's fast (no code generation) and produces structured feedback.

**Flags:**
- `--json`: Output as JSON (default for agent use).
- `--verbose`: Include inferred types for all subexpressions, not just top-level.
- `--repair`: Attempt to automatically apply repair suggestions and re-check.

### `chelis deep <file.ch>`

Run Parse -> Desugar only. Outputs the **Deep** (s-expression) form of the input Surf file. Useful for inspecting what the compiler sees after desugaring.

**Flags:**
- `--canonical`: Output in canonical form (deterministic whitespace, sorted metadata).
- `--no-metadata`: Strip metadata annotations from output.

### `chelis emit <file>`

Run Parse -> Desugar -> Check -> Lower -> Transform -> Emit. Outputs the generated target code (e.g., C source) without invoking the target compiler. Useful for inspecting generated code or feeding it into a custom build system.

**Flags:**
- `--backend <c|cuda|stablehlo>`: Choose emission backend. Default: `c`.
- `--output <path>`: Output file path. Default: stdout.

### `chelis surf <file.dp>`

Best-effort decompilation of **Deep** back to **Surf**. This is not guaranteed to be the inverse of `chelis deep` -- the output is a valid Surf program that desugars to the same Deep form, but it may not match the original Surf source. Useful for reading machine-generated Deep code.

**Flags:**
- `--style <compact|verbose>`: Formatting style. Default: `verbose`.

### `chelis tide`

Launch the interactive **Tide** REPL. Provides an incremental, stateful environment for evaluating Chelis expressions.

**Flags:**
- `--agent`: Launch in agent mode (JSON protocol over stdin/stdout instead of human REPL).
- `--backend <c|cuda|stablehlo>`: Backend for JIT compilation of expressions. Default: `c`.

---

## 6. Type System Terms

### ADT (Algebraic Data Type)

A type defined as a sum of product types. Chelis supports ADTs with type parameters.

```
type Option a = Some a | None
type List a = Cons a (List a) | Nil
type Result a e = Ok a | Err e
```

Each variant is a **constructor** that can be used in expressions (to create values) and in patterns (to destructure values). Pattern matching on ADTs is checked for exhaustiveness.

### Constrained Type (Future)

A type variable with a constraint. Not yet implemented. Planned syntax:

```
def sum(x: tensor[n, a]): a where a : Numeric
```

Constraints will include `Numeric`, `Differentiable`, `Comparable`, and user-defined type classes.

### Fitness Score

A floating-point value in the range [0.0, 1.0] that measures how close a program is to being type-correct. The fitness score is computed by the **Check** stage and reported by `chelis check`.

The scoring algorithm:
- Start at 1.0.
- For each type error, subtract a penalty proportional to the severity:
  - Dimension mismatch: -0.1 per mismatched dimension
  - Precision mismatch: -0.05 per mismatched precision
  - Missing definition: -0.2 per undefined name
  - Kind error: -0.3 per kind error
  - Parse error: score is (fraction of file that parsed successfully) * 0.3
- Clamp to [0.0, 1.0].

A score of 1.0 means the program is fully type-correct. A score of 0.0 means it failed to parse or has catastrophic errors.

The fitness score is designed as a training signal for AI coding agents: higher is better, and small code changes produce small score changes (the score is locally smooth).

### HM (Hindley-Milner)

The type inference algorithm used by Chelis. Hindley-Milner inference determines the most general type of every expression without requiring type annotations (though annotations are permitted and encouraged for readability).

Key properties:
- **Principal types**: Every expression has a unique most-general type.
- **Decidable**: Type inference always terminates.
- **Complete**: If a type exists, the algorithm finds it.

Chelis extends standard HM with tensor types, named dimensions, and precision types. These extensions preserve decidability.

### Named Dimensions

Tensor axes identified by name rather than position. In Chelis, a tensor type like `tensor[batch, hidden, f32]` has two named dimensions: `batch` and `hidden`. The precision (`f32`) is always the last element and is not a dimension.

Named dimensions serve two purposes:
1. **Documentation**: The name communicates the semantic role of each axis.
2. **Type safety**: Operations that combine tensors check that dimension names match. You can't accidentally add a `[batch, hidden]` tensor to a `[hidden, batch]` tensor -- the dimension names don't align.

Dimension names can be:
- **Symbolic**: `batch`, `hidden`, `seq_len` -- the name is known but the size is not fixed at compile time.
- **Concrete**: `784`, `10` -- a literal integer size. Written as a bare integer in the type.

### Precision Types

Numeric base types that specify the bit-level representation. Chelis provides:

| Type | Description |
|------|-------------|
| `f16` | IEEE 754 half-precision float (16-bit) |
| `bf16` | Brain float (16-bit, 8-bit exponent) |
| `f32` | IEEE 754 single-precision float (32-bit) |
| `f64` | IEEE 754 double-precision float (64-bit) |
| `i8` | Signed 8-bit integer |
| `i16` | Signed 16-bit integer |
| `i32` | Signed 32-bit integer |
| `i64` | Signed 64-bit integer |
| `u8` | Unsigned 8-bit integer |
| `u16` | Unsigned 16-bit integer |
| `u32` | Unsigned 32-bit integer |
| `u64` | Unsigned 64-bit integer |
| `bool` | Boolean (1-bit logical) |

Precision types **never implicitly convert**. To convert between precisions, use the explicit `cast` construct:

```
let y = cast(x, f64)   -- explicit: f32 -> f64
```

### Repair Suggestion

A compiler-generated fix for a type error. Repair suggestions are emitted alongside error messages by the **Check** stage. Each suggestion is:

- **Actionable**: It specifies an exact code change (insert, delete, or replace) with line/column range.
- **Typed**: The suggestion is itself type-checked -- applying it is guaranteed to fix the specific error it addresses (though it may reveal other errors).
- **Ranked**: When multiple repairs are possible, they are ranked by likelihood of being the programmer's intent.

Example output:
```json
{
  "error": "precision_mismatch",
  "location": {"line": 7, "col": 15},
  "expected": "f32",
  "actual": "f64",
  "repairs": [
    {"action": "wrap", "text": "cast(x, f32)", "confidence": 0.9},
    {"action": "change_annotation", "text": "f64", "confidence": 0.3}
  ]
}
```

---

## 7. IR Terms

### Adjoint

The backward-pass rule for a **RISC primitive**, used in reverse-mode automatic differentiation (AD). Each of the ~12 primitives has a defined adjoint that specifies how to propagate gradients backward through that operation.

For example:
- **Adjoint of `add(a, b)`**: Both inputs receive the upstream gradient unchanged.
- **Adjoint of `mul(a, b)`**: Input `a` receives `upstream * b`; input `b` receives `upstream * a`.
- **Adjoint of `reduce_sum(x, axis)`**: The upstream gradient is broadcast back along the reduced axis.

The `grad` transformation works by: (1) traversing the RISC DAG forward, (2) reversing the edges, and (3) replacing each node with its adjoint. This mechanical process produces a new DAG that computes the gradient.

### RISC DAG

The intermediate representation of a Chelis program after **lowering**. It is a directed acyclic graph where:

- **Nodes** are RISC primitives (one of ~12 operations).
- **Edges** represent data dependencies (the output of one node feeds into the input of another).
- **Leaves** are inputs (function parameters, constants, tensor literals).
- **Roots** are outputs (function return values).

The DAG has no control flow (pattern matching and conditionals are compiled away during lowering), no named variables (everything is an edge), and no types at the surface level (though type metadata is attached for validation).

"RISC" is by analogy with RISC instruction sets: a small number of simple operations, in contrast to CISC-style complex operations. The term is borrowed from tinygrad's philosophy.

### RISC Primitive

One of the fundamental operations in the **RISC DAG**. The current primitive set is:

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `add` | `(T, T) -> T` | Element-wise addition |
| `mul` | `(T, T) -> T` | Element-wise multiplication |
| `reduce_sum` | `(T, axis) -> T'` | Sum reduction along an axis |
| `reduce_max` | `(T, axis) -> T'` | Max reduction along an axis |
| `reshape` | `(T, shape) -> T'` | Reshape tensor (same total elements) |
| `broadcast` | `(T, shape) -> T'` | Broadcast tensor to a larger shape |
| `slice` | `(T, start, end) -> T'` | Extract a contiguous subtensor |
| `concat` | `(T, T, axis) -> T'` | Concatenate along an axis |
| `matmul` | `(T, T) -> T'` | Matrix multiplication |
| `exp` | `(T) -> T` | Element-wise exponential |
| `log` | `(T) -> T` | Element-wise natural logarithm |
| `compare` | `(T, T, op) -> T_bool` | Element-wise comparison (eq, lt, gt, etc.) |

Every tensor computation in Chelis reduces to a composition of these primitives. Operations like subtraction (`a - b`), division (`a / b`), negation (`-a`), and softmax are expressed as combinations:
- `sub(a, b)` = `add(a, mul(b, -1))`
- `div(a, b)` = `mul(a, reciprocal(b))` where `reciprocal(x)` = `exp(mul(log(x), -1))`
- `neg(a)` = `mul(a, -1)`
- `softmax(x)` = `div(exp(x), reduce_sum(exp(x), axis=-1))`

---

## 8. Transformation Terms

### grad

A language-level transformation that computes the gradient of a function via reverse-mode automatic differentiation. Given a function `f : A -> scalar`, `grad(f)` produces a function `f' : A -> A` that computes the gradient of `f`'s output with respect to its input.

At the DAG level, `grad` is implemented by the **Transform** stage as a mechanical rewrite: reverse the edges, apply **adjoint** rules.

Type rule:
```
If   f : tensor[d..., p] -> tensor[p]      (input to scalar)
Then grad(f) : tensor[d..., p] -> tensor[d..., p]   (same shape as input)
```

For multi-argument functions, `grad` returns a tuple of gradients (one per argument).

### jit

A language-level marker that requests just-in-time compilation of a function. `jit(f)` has the same type as `f` -- it doesn't change what the function computes, only how it's executed.

At the DAG level, `jit` inserts a compilation boundary. The subgraph rooted at `jit` is compiled to native code on first invocation and cached for subsequent calls.

### vmap

A language-level transformation that vectorizes a function over an additional batch dimension. Given a function `f : A -> B`, `vmap(f, axis=k)` produces a function that maps `f` over axis `k` of its input.

At the DAG level, `vmap` is implemented by the **Transform** stage as a DAG rewrite: each primitive is lifted to operate over one additional dimension.

Type rule:
```
If   f : tensor[d1, ..., dn, p] -> tensor[e1, ..., em, p]
Then vmap(f, axis=0) : tensor[batch, d1, ..., dn, p] -> tensor[batch, e1, ..., em, p]
```

The `axis` parameter specifies which axis of the input tensor corresponds to the batch dimension that `f` is mapped over.

---

## 9. General Terms

### Backend

A code generation target. Chelis supports multiple backends:
- **C**: Portable CPU code. The default and reference backend.
- **CUDA**: NVIDIA GPU kernels.
- **StableHLO**: Interop with the XLA/IREE ecosystem.

### Canonical Form

The unique textual representation of a **Deep** program. Two Deep programs are semantically equivalent if and only if their canonical forms are identical (byte-for-byte). Canonical form rules are defined in `03-deep-syntax.md`.

### Desugaring

The process of translating **Surf** syntax into **Deep** syntax. Desugaring is mechanical, deterministic, and does not require type information. It is a purely syntactic transformation. The complete desugaring rules are defined in `03-deep-syntax.md`.

### Module

A unit of Chelis source code. A single `.ch` or `.dp` file constitutes one module. Modules can import other modules. A module has:
- A name (derived from the filename or declared with `module`)
- Zero or more imports
- Zero or more type definitions
- Zero or more value definitions (functions and constants)

### Tag

The first element of a list in **Deep** syntax. The tag identifies the form: `def`, `let`, `fn`, `apply`, `if`, `match`, etc. Tags are always symbols (not strings, not numbers).
