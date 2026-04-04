# Chelis: Revised Language Specification

chelis.ch | "Turtles all the way down"

---

## Nomenclature

The language has two syntactic layers. They need names that are terse, clear, and internally consistent.

**Surf** — the surface syntax. What humans read. Modern functional: ADTs, pipes, pattern matching, type annotations. Written in `.ch` files. This is the supervisory interface — a human directing a coding agent reads Surf to understand what the agent produced.

**Deep** — the s-expression syntax. What machines write. Homoiconic, parseable, mutable, type-annotated s-expressions. The generation target for AI agents, the substrate for evolutionary methods, the internal representation the compiler operates on. Deep is more constrained than Surf — no sugar, no ambiguity, one canonical form per program.

**The relationship:** Surf desugars mechanically to Deep. Deep can be decompiled to Surf (with some sugar choices). The desugaring is lossless; the decompilation is lossy (multiple Surf forms can produce the same Deep). Round-tripping Surf → Deep → Surf produces valid, readable code that may differ cosmetically from the original.

**CLI:**
```
chelis build app.ch            # compile Surf → Deep → IR → target
chelis deep app.ch             # emit Deep form (desugar only)
chelis surf program.dp       # decompile Deep → Surf (best-effort)
chelis check app.ch            # type-check, return graded feedback
chelis repl                    # interactive mode (accepts both Surf and Deep)
```

**Ecosystem nomenclature (turtle/ocean themed):**
- Packages are **shells** — self-contained units of code
- The package registry is the **Reef** — reef.chelis.ch
- The interactive REPL/agent interface is the **Tide** — `chelis tide`
- A Chelis project file is `reef.toml` (the project's place in the reef)

---

## 1. Surf Syntax Design

Modern functional. Not "Scala 3 syntax" — draws from the ML family broadly: F#'s clarity, Elixir's pipes and pattern matching, Haskell's type class ergonomics (but via effects, not type classes), Rust's ownership annotations (but via linear types, not a borrow checker).

### Design Principles
- **One obvious way to write anything.** Minimize syntactic ambiguity. AI agents generating Surf should converge on the same form.
- **Types at boundaries, inference within.** Function signatures are explicitly typed. Local bindings are inferred.
- **Left-to-right data flow.** Pipes (`|>`) are the primary composition mechanism. Tensor pipelines read naturally.
- **No escape hatches.** No mutation, no loops, no imperative blocks. Recursion, higher-order functions, and the three concurrency primitives (`par`, `stream`, `scatter`) are the only control flow.

### Surf Syntax Examples

```
-- Module declaration
module MyModel

-- Type declarations
type Optimizer =
  | Adam { lr: f32, betas: (f32, f32), eps: f32 }
  | SGD { lr: f32, momentum: f32 }

-- Function definitions with type signatures
def transformer_block(
  x: tensor[batch, seq, dim, bf16],
  params: BlockParams
): tensor[batch, seq, dim, bf16] =
  let residual = x
  x
  |> layer_norm(params.norm1)
  |> multi_head_attention(params.attn)
  |> add(residual)
  |> feed_forward(params.ff, residual=_)

-- Pattern matching
def step(opt: Optimizer, grads: Gradients): Params =
  match opt with
  | Adam { lr, betas, eps } -> adam_update(lr, betas, eps, grads)
  | SGD { lr, momentum }   -> sgd_update(lr, momentum, grads)

-- Anonymous functions
let scale = fn x, factor -> x * factor

-- Pipes with partial application
data |> map(fn x -> x |> normalize |> augment) |> batch(32)
```

### Type Annotation Style
`name: Type` (colon, not double-colon). Square brackets for tensor dimensions: `tensor[batch, seq, dim, bf16]`. Braces for record-style ADT variants. This is close to F#/Rust convention, not Haskell/Elixir convention.

### Open Surf Decisions (for later)
- Exact keyword set (`def`/`let`/`type`/`match`/`with`/`fn`/`do`)
- Module system (simple namespaces for v1, ML-style signatures/functors possible later)
- Named dimension declaration syntax (module-level? inline? both?)
- Import/export syntax

---

## 2. Deep Syntax Design

The s-expression representation. This is the machine-native interface and arguably the more important syntax to get right.

### Prior Art

**Racket's syntax objects.** The most sophisticated s-expression macro system. Syntax objects carry lexical context (scope information), source location, and type annotations. Racket's `syntax-parse` provides pattern matching on syntax trees. Chelis Deep should learn from Racket's approach to hygienic macros but adapt it for a typed, effects-aware context.

**Elixir's quoted AST.** Elixir represents its AST as nested 3-tuples: `{name, metadata, arguments}`. Metadata carries line numbers, context, and hygiene information. The `quote`/`unquote` system is simple and effective. Chelis Deep should adopt this metadata-carrying approach.

**TASTy (Scala 3).** Typed Abstract Syntax Trees — Scala 3's serialization format for its typed AST. TASTy files are binary-encoded typed ASTs that can be read, analyzed, and transformed by tools. This is the closest prior art for what Chelis Deep serialization needs to be.

**Typed Racket.** Shows how type annotations work within s-expressions: `(: x Integer)` for type ascription, `(define (f [x : Integer]) : Integer ...)` for function signatures.

### Deep Syntax Specification

Every Deep expression is a tagged s-expression: `(tag metadata ...children)`.

Metadata is a property list carrying: source location (for decompilation to Surf), inferred type (after type checking), effects (after effect inference), linearity status. Metadata is optional during construction and filled in by compiler passes.

```lisp
;; Module
(module MyModel
  (type Optimizer
    (variant Adam (record (lr f32) (betas (tuple f32 f32)) (eps f32)))
    (variant SGD  (record (lr f32) (momentum f32))))

  ;; Function with type signature
  (def transformer_block
    (sig (-> (tensor (dim batch) (dim seq) (dim dim_) bf16)
             BlockParams
             (tensor (dim batch) (dim seq) (dim dim_) bf16)))
    (fn (x params)
      (let ((residual x))
        (pipe x
          (layer_norm (field params norm1))
          (multi_head_attention (field params attn))
          (add residual)
          (feed_forward (field params ff))))))

  ;; Pattern match
  (def step
    (sig (-> Optimizer Gradients Params))
    (fn (opt grads)
      (match opt
        ((Adam (record lr betas eps))
         (adam_update lr betas eps grads))
        ((SGD (record lr momentum))
         (sgd_update lr momentum grads))))))
```

### Key Design Choices

**Types in Deep.** Types are explicit s-expressions. `(tensor (dim batch) (dim seq) f32)` not inferred shorthand. The Deep form is fully annotated after type checking — every subexpression carries its type in metadata.

**Effects in Deep.** After effect inference, functions carry effect annotations: `(fn/eff (Diff Random) (x) ...)`. This makes effects visible and manipulable in Deep, which is critical for AI agents reasoning about what a function does.

**Linear types in Deep.** Linear bindings are marked: `(let-linear ((x (alloc (shape 32 64) f32))) ...)`. Borrows are explicit: `(borrow x (fn (x-ref) ...))`.

**Pipe desugaring.** `(pipe x f g h)` desugars to `(h (g (f x)))` in Deep. The pipe is sugar even in Deep — but a canonical form that tools can depend on.

**No ambiguity.** Every Deep program has exactly one parse. No operator precedence, no implicit conversions, no syntactic shortcuts. This is what makes Deep a low-entropy generation target for AI.

### The Macro System

Macros operate on Deep s-expressions and produce Deep s-expressions. They are type-aware: a macro that consumes a typed expression must produce a typed expression.

**Prior art synthesis:** Racket's hygienic macros (preventing accidental variable capture) + Elixir's quote/unquote (simple, practical) + type checking (novel).

```lisp
;; Define a macro
(defmacro with_gradient_checkpoint (body)
  ;; The macro transforms a computation to recompute activations
  ;; instead of storing them, saving memory
  (quote
    (checkpoint
      (fn () (unquote body)))))

;; Typed macro: ensures the output has the same type as the input
(defmacro @differentiable (fn-def)
  ;; Verify all operations in fn-def body handle the Diff effect
  ;; This is a compile-time check implemented as a macro
  (let ((body (fn-body fn-def))
        (violations (check-effect-handling 'Diff body)))
    (if (empty? violations)
      fn-def
      (compile-error "Non-differentiable operations found:" violations))))
```

**Macro phases:** Macros run at compile time, before type checking. But type-aware macros can call into the type checker during expansion. This requires a phased compilation model (Racket-style).

**Macro hygiene:** Automatic. Variables introduced by macro expansion are renamed to avoid capture. Manual hygiene breaking via `(syntax-local name)` for when you intentionally want to capture.

---

## 3. Compiler as Training Signal

The Chelis compiler is not just an error checker — it's a reward function for AI agents.

### Graded Type Feedback

Instead of pass/fail, the compiler produces a **fitness score** for every compilation attempt:

```json
{
  "valid": false,
  "score": 0.73,
  "errors": [
    {
      "kind": "dimension_mismatch",
      "location": {"line": 12, "col": 5},
      "expected": "tensor[batch, hidden, f32]",
      "got": "tensor[batch, seq, f32]",
      "suggestions": [
        "transpose dimensions 1 and 2",
        "rename dimension 'seq' to 'hidden' if this is intentional"
      ],
      "severity": 0.8
    }
  ],
  "partial_types": {
    "transformer_block": "tensor[batch, seq, ?, bf16] -> tensor[batch, seq, ?, bf16]",
    "attention": "fully typed",
    "ffn": "unresolved: dimension conflict at output"
  },
  "effects_inferred": ["Diff", "Random"],
  "linearity_violations": 0,
  "warnings": []
}
```

**Graded scoring:** `score` is a 0-1 measure of "how close to valid." Computed from: fraction of subexpressions that type-check, fraction of effects that are handled, fraction of linear resources that are consumed. An AI agent can use this as a reward signal for RL or as a fitness function for evolution.

**Partial type inference:** Even if the full program fails, infer what you can. Report which functions fully type-check and which don't. This gives the AI agent a map of "what's working and what's broken."

**Repair suggestions:** For common error patterns, suggest fixes. These suggestions are structured data (not just strings) so AI agents can apply them programmatically.

---

## 4. Programs as Data (Enabled, Not Imposed)

The homoiconic typed AST makes programs first-class data. An AI agent can parse, inspect, transform, and regenerate Deep s-expressions using standard language facilities. The type checker validates any transformation. This naturally enables evolutionary methods, NAS, neurosymbolic search, and self-modifying programs — but the language doesn't prescribe HOW these are done.

What the language provides:
- **`quote` / `unquote`** for constructing and deconstructing Deep ASTs at runtime
- **The type checker as a library call** — validate any programmatically constructed AST, get graded feedback
- **Serialization round-tripping** — Deep text is trivially parseable, so any language or tool can construct Chelis programs

What the language does NOT provide:
- No built-in `mutate`, `crossover`, `evolve` primitives. These are user-space concerns. Different teams will want different mutation strategies, selection mechanisms, and fitness functions. Baking one in would constrain rather than enable.
- No population management. That's orchestration, not computation — use Elixir, Python, or whatever manages your experiment loop.

The compiler's graded fitness feedback (Section 3) is the key enabler. An external evolutionary loop submits candidate programs via the Tide API, gets typed results with fitness scores, and uses those scores however it wants. The language is the substrate; the methodology is yours.

---

## 5. Backend Strategy

### Phase 0: C Backend (Bootstrap)

Futhark-style source-to-source compilation. The Chelis compiler outputs:

1. **Host code:** Standard C that handles memory allocation, data loading, orchestration.
2. **Kernel code:** Optimized C with BLAS calls (OpenBLAS/MKL) for tensor operations.

This is the test oracle. Every future backend must produce identical numerical results to the C backend. Phase 0 deliverable: MNIST end-to-end on CPU.

### Phase 1: Futhark-Style GPU Backend (Own the Compilation)

Same architecture as Futhark: source-to-source transpilation.

1. **Chelis compiler** (written in Rust) does: parsing, type checking, effect inference, linearity checking, RISC DAG construction, fusion, flattening, memory planning.
2. **Output:** A C file containing host code + an embedded string of CUDA/OpenCL/HIP kernel code.
3. **Execution:** Host C compiled by GCC/Clang. Kernel string JIT-compiled by NVRTC (NVIDIA) or clBuildProgram (OpenCL) at runtime.

**Why this architecture:**
- Small team can build it (Futhark team is ~3 people)
- Portability: same Chelis program runs on NVIDIA, AMD, Intel, Apple silicon
- No driver maintenance: vendor SDKs handle the last mile
- Easy integration: output is a `.c` + `.h` file, linkable from any language
- The compiler owns fusion, parallelization, and memory planning — the parts where purity + linear types give strictly more information than imperative compilers have

### Phase 2+ (Mid-Roadmap): StableHLO / FX Integration

Once the language has users and the type system is stable:
- **StableHLO emission** for TPU access and as an alternative GPU path. Follow the Nx/EXLA pattern.
- **FX graph emission** for PyTorch ecosystem integration (torch.export, ExecuTorch for edge).

These are additive — the C/GPU backend remains the primary path. StableHLO and FX are integration layers, not replacements.

---

## 6. Agent-Friendly Interactive Mode (Tide)

`chelis tide` — the interactive subsystem. Designed for AI agent interaction from day 0, not retrofitted.

### Interfaces (layered, not mutually exclusive)

**REPL (human).** Accepts both Surf and Deep input. Evaluates expressions, shows types, shows effects. Standard terminal interface.

**Language Server (LSP).** For IDE integration. Type checking, completion, hover info, diagnostics. Works with VS Code, Neovim, etc.

**Agent API (MCP / HTTP).** The primary machine interface. Structured JSON request/response:

```
POST /compile
{
  "source": "(def f (fn (x) (add x 1)))",
  "format": "core",
  "return": ["typed_ast", "fitness_score", "suggestions"]
}

→ {
  "valid": true,
  "score": 1.0,
  "typed_ast": "(def f (sig (-> int32 int32)) (fn (x) (add x 1)))",
  "effects": [],
  "suggestions": []
}
```

Endpoints:
- `/compile` — submit Surf or Deep, get typed result + fitness score
- `/check` — type-check without compiling
- `/desugar` — Surf → Deep conversion
- `/decompile` — Deep → Surf conversion
- `/eval` — compile and execute, return result

**MCP Server.** Chelis exposes itself as an MCP tool server. A coding agent (Claude Code, Cursor, etc.) connects to the Chelis MCP server and gets tools like `chelis_compile`, `chelis_check`, `chelis_desugar`, `chelis_eval`. The agent uses these tools as part of its reasoning loop.

### Project Sequencing for Tide

1. **v0.1:** REPL accepting Deep s-expressions, returning typed results. No LSP, no API. Just a `chelis tide` command that reads stdin, type-checks, evaluates.
2. **v0.2:** Add Surf parsing. REPL accepts both. Add `chelis deep` and `chelis surf` commands.
3. **v0.3:** Agent API (HTTP/JSON). The `/compile`, `/check`, `/desugar` endpoints.
4. **v0.4:** MCP server wrapping the Agent API. Coding agents can use Chelis as a tool.
5. **v0.5:** LSP for IDE integration.
6. **v1.0:** Full Tide with batch compilation, streaming results, project-level operations.

---

## 7. Serialization

### Prior Art

**TASTy (Scala 3).** Binary serialization of typed ASTs. Every `.class` file includes a TASTy section containing the full typed tree. Tools can read TASTy to analyze, transform, or decompile Scala programs. This is the closest prior art for Chelis Deep serialization.

**FASL (Common Lisp).** Compiled Lisp files containing serialized s-expressions + compiled code. Platform-specific but fast to load.

**MLIR text format.** StableHLO and other MLIR dialects have a human-readable text format for IR serialization. Shows how a typed IR can be both human-readable and machine-parseable.

### Chelis Serialization

**Three formats, three purposes:**

1. **`.ch` (Surf text).** Human-readable source files. What you edit, what you commit to git, what you review in PRs. UTF-8 text.

2. **`.dp` (Deep text).** S-expression text files. Machine-readable, human-inspectable. Used for: AI agent I/O, debugging, `chelis deep` output, interoperability with Lisp-family tools. UTF-8 text. A `.dp` file is a valid Chelis program that can be compiled directly.

3. **`.chb` (Chelis Binary).** Binary-serialized typed Deep AST. Like TASTy: contains the full typed tree, effect annotations, linearity info, source locations. Used for: compiled shell (package) distribution, incremental compilation, fast loading. A `.chb` file can be decompiled to `.dp` or `.ch` losslessly (the typed AST preserves all information).

**Shell (package) distribution:** A published shell on the Reef contains `.chb` files (binary ASTs) + a `reef.toml` manifest. Consumers can inspect the types without compiling from source. Analogous to how JAR files contain `.class` + TASTy in the Scala ecosystem.

---

## 8. Round-Tripping

**Surf → Deep (desugaring):** Lossless. Every Surf construct has exactly one Deep equivalent. `chelis deep app.ch` produces a `.dp` file that compiles to the identical program.

**Deep → Surf (decompilation):** Semantically lossless, cosmetically lossy. The decompiler makes choices about: where to insert pipes vs nested calls, how to format pattern matches, whether to use `let` bindings or inline expressions. Multiple Surf programs can produce the same Deep. The decompiler picks one canonical style.

**Deep is the single source of truth.** If there's ever a question about what a program means, Deep is the answer. Surf is a rendering.

**Implications for AI agents:** An agent can: (1) generate Deep directly (lowest effort, no surface sugar to worry about), (2) ask `chelis surf` to render it for human review, (3) receive human edits in Surf, (4) convert back to Deep via `chelis deep`, (5) continue evolving in Deep. The round-trip works because Deep is strictly more constrained than Surf.

---

## 9. Revised Summary: What's Decided

| Aspect | Decision |
|---|---|
| **Name** | Chelis (chelis.ch) |
| **Syntax layers** | Surf (.ch) for humans, Deep (.dp) for machines |
| **Surf family** | Modern functional (ML-family: ADTs, pipes, matching, inference) |
| **Deep family** | Typed s-expressions with metadata (Racket/Elixir-influenced) |
| **Macro system** | Hygienic, type-aware, operating on Deep s-expressions |
| **Type system v1** | ADTs + HM inference + precision types + named dims + effects (Diff, Random, Resource) + linear types for tensors |
| **Compiler role** | Fitness function for AI. Graded feedback, repair suggestions, partial inference. |
| **Programs as data** | Homoiconic Deep AST + quote/unquote + type checker as library. Enables evolution, NAS, neurosymbolic — doesn't prescribe. |
| **RISC primitives** | ~12 tensor ops (tinygrad-style). Composition is the only complexity. |
| **Lazy evaluation model** | Build DAG, realize on demand |
| **Function transforms** | grad, vmap, jit as DAG-to-DAG rewrites |
| **Backend Phase 0** | C + BLAS (CPU). Test oracle. |
| **Backend Phase 1** | Futhark-style: C host + embedded CUDA/OpenCL kernel strings. Own the compilation. |
| **Backend Phase 2+** | StableHLO (TPU access), FX graphs (PyTorch ecosystem). Additive. |
| **Interactive mode** | Tide: REPL → Agent API → MCP server → LSP (sequenced) |
| **Serialization** | .ch (Surf text), .dp (Deep text), .chb (typed binary AST) |
| **Packages** | Shells, distributed via the Reef (reef.chelis.ch) |
| **Compiler language** | Rust |
| **Concurrency v1** | DAG-implicit parallelism + `par` primitive only |
| **Python FFI** | DLPack for tensors, PyO3 for everything else. GIL released during compute. |
