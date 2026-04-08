# Phase 2: Language Maturity - Expanded Implementation Plan

## Context

Phase 1 is structurally complete. The GPU backend works, real models compile and run
correctly, and the benchmark suite proves correctness across CPU/HIP/PyTorch. Known
limitations carried forward: `pad`/`shrink` not implemented in HIP, symbolic
dimensions not implemented in either backend (recompilation required for shape
changes), dotted Deep path round-trip gap.

**Phase 2 deliverable:** The language is usable by researchers. Effects, linear types,
macros, the agent API, and tooling make Chelis a credible alternative to PyTorch for
specific workloads. A researcher should be able to write, type-check, differentiate,
compile, train, and debug a model - with AI assistance - using only the Chelis
toolchain.

**Phase 2 does NOT deliver:** A package manager (Phase 3a), alternative backends like
StableHLO/FX (Phase 3b/3c), Python FFI (Phase 3d), or the local coding model (Phase
3g). Phase 2 is about making the language complete; Phase 3 is about making the
ecosystem complete.

---

## Dependency Graph

```text
Phase 1 carry-forward fixes ----------------------------------------------+
  (symbolic dims, pad/shrink, Deep round-trip)                            |
                                                                          |
2a: Algebraic Effects --> 2b: Linear Types --> 2c: Macros                |
        |                        |                    |                   |
        |                        |                    |                   |
        +-----------+------------+                    |                   |
                    |                                 |                   |
               2d: vmap                              |                   |
                                                      |                   |
2e: Tide Agent API --> 2f: Tide LSP --> 2g: Tide TUI (chelis cove)      |
        ^                                                                 |
        +-----------------------------------------------------------------+

2s: Seed Corpus (runs throughout Phase 2, collecting programs)
```

**Critical path:** 2a -> 2b -> 2c (each depends on the previous for interaction
rules). Tooling (2e -> 2f -> 2g) can run in parallel with the type-system track after
2e has enough compiler API surface.

**Recommended execution order:**
1. Phase 1 carry-forward fixes (symbolic dims - high value, unblocks dynamic batching)
2. 2a: Effects (long lead, spec-heavy, informs everything else)
3. 2e: Agent API (unblocks tooling track and SKILL.md-based workflows)
4. 2b: Linear types (depends on effect system design for interaction rules)
5. 2d: `vmap` (independent of 2b/2c, can slot in anytime after 2a)
6. 2f: LSP (depends on 2e API surface)
7. 2c: Macros (last type-system feature, depends on both effects and linearity for
   interaction rules)
8. 2g: TUI (depends on 2e + 2f, capstone tooling deliverable)
9. 2s: Seed corpus (continuous, not a gated phase)

---

## Phase 1 Carry-Forward Fixes

Before starting Phase 2 proper, address the known limitations that affect Phase 2
work:

### Symbolic Dimensions in Both Backends

**Problem:** Both C and HIP backends require concrete tensor extents at codegen time.
Named dimensions with unresolved sizes (`DimInfo::Named(name, None)`) are rejected.
This means every shape change requires recompilation.

**Solution:**
- IR: `DimInfo::Named(name, None)` stays in the DAG as a symbolic dimension
- Codegen (both backends): emit shape parameters as function arguments (`int batch`,
  `int seq`, ...)
- Kernel launch: grid/block sizes computed from runtime shape values
- Memory planning: buffer sizes computed from runtime shape values (requires a runtime
  `malloc` instead of a compile-time constant)
- Matmul hipBLAS path: `m`/`n`/`k` come from runtime parameters

**Scope:** ~2 weeks. This is an IR-through-codegen change that touches both backends,
the memory planner, and the CLI. Test: same MNIST program works with `batch=32` and
`batch=128` without recompilation.

### Pad/Shrink in HIP Backend

**Problem:** `todo!()` in HIP `emit.rs`. These ops exist for windowing/slicing.

**Solution:** Implement as elementwise kernels with boundary checks. `pad` reads from
source if in bounds, writes fill value otherwise. `shrink` reads from the shrunk
region.

**Scope:** ~2 days. Small, mechanical. Test: `pad`/`shrink` GPU output matches CPU
output.

### Deep Dotted Path Round-Trip

**Problem:** Compiler emits dotted module/import paths that the Deep parser cannot
reparse.

**Solution:** Extend the Deep parser to accept dotted identifiers in `import` and
`module` node positions.

**Scope:** ~1 day. Parser change + test. The validator already accepts them.

---

## 2a: Algebraic Effects

**Goal:** The type system tracks what functions do (differentiation, randomness,
device allocation) and the compiler uses this information for optimization and error
reporting.

### Spec Work (BEFORE implementation)

Produce `spec/04-type-system.md` effects coverage with:

**Effect vocabulary (built-in):**

| Effect | Meaning | Handler | Compiler use |
|---|---|---|---|
| `Diff` | Function body contains differentiable operations | `grad` | AD engine verifies all ops in the body have adjoint rules |
| `Random` | Function body uses stochastic operations | `withSeed(seed, f)` | Ensures reproducibility; can eliminate randomness for deterministic inference |
| `Accum` | Function body uses associative accumulation | implicit in `grad` backward pass | GPU backend knows these loops are parallelizable |
| `Resource(D)` | Function allocates on device `D` | `withDevice(device, f)` | Device placement; prevents cross-device operations without explicit transfer |

**Why `Accum` (from Dex):** Gradient accumulation (`sum` of adjoint contributions from
multiple consumers) is an `Accum` pattern. Typing it as `Accum` rather than general
`State` tells the GPU backend the backward-pass loop is parallelizable. Without this
distinction, the compiler would have to conservatively sequentialize gradient
accumulation.

**No user-defined effects in v1.** Four built-in effects cover the ML domain.
User-defined effects (Phase 3+) add handler semantics, effect polymorphism, and
resumable computations - complexity that is hard for both humans and LLMs.

**Effect inference rules:**
- A function's effect is the union of effects of all operations in its body
- Effect composition is row-polymorphic: `map(f, xs)` has whatever effects `f` has
- Row variables are implicit - the programmer never writes effect annotations unless
  they want documentation
- Effect inference runs after HM type inference on the same typed Deep AST

**Handler rules:**
- A handler eliminates an effect from the type: `grad(f)` takes
  `f: A -> B ! {Diff, e}` and produces `g: A -> (B, A) ! {e}`
- `withSeed(42, f)` takes `f: A -> B ! {Random, e}` and produces `g: A -> B ! {e}`
- `withDevice("gpu:0", f)` takes `f: A -> B ! {Resource(gpu:0), e}` and produces
  `g: A -> B ! {e}`
- Unhandled effects at the program boundary are compile errors with repair suggestions

**Interaction with existing type system:**
- Effects do NOT change HM inference - types are inferred first, effects inferred second
- Effect annotations in Surf are optional:
  `sig f: tensor[n, f32] -> tensor[n, f32] ! {Diff}` is valid but never required
- Effect annotations in Deep: new metadata key `{effects: [diff, random]}` on `fn`
  nodes - informational, not required, validated if present

**Error messages (the LLM-facing concern):**
- "Function `predict` has unhandled effect `Random` - `dropout` at line 42 introduces
  randomness. Wrap the call site with `withSeed(seed, ...)` for deterministic
  inference, or propagate the effect by declaring it in the caller's signature."
- "Cannot differentiate function `step` - `argmax` at line 17 is not differentiable.
  Replace with a differentiable approximation like `softmax` or use
  `stop_grad(argmax(...))` to exclude it from the gradient computation."
- Repair suggestions must reference specific line numbers and concrete fix patterns.
  The fitness score should degrade proportionally to the number of unhandled effects.

### Implementation Plan

**Crate: `chelis-effects` (new)**
- `infer.rs`: walk typed Deep AST, collect effects per function
- `check.rs`: verify handlers eliminate effects, report unhandled effects
- `types.rs`: effect types, row variables, composition rules
- `errors.rs`: structured effect errors with repair suggestions

**Integration points:**
- After `chelis-types::infer()` succeeds, run `chelis-effects::infer()` on the typed AST
- Effect information stored as metadata on function nodes in the typed AST
- `chelis-ir::lower()` uses effect information to validate AD applicability (`Diff`),
  insert seed threading (`Random`), and annotate device placement (`Resource`)
- Fitness scoring extended: unhandled effects reduce fitness proportionally

**Surf syntax additions:**
- `sig f: A -> B ! {Diff, Random}` - optional effect annotation on signatures
- `with seed(42) { ... }` - handler block syntax
- `with device("gpu:0") { ... }` - handler block syntax
- `grad(f)` - already exists, now formally an effect handler

**Deep syntax additions:**
- `(handle-effect {} (lit {} diff) (fn {} ...body...))` - handler node
- `(effect-ann {} (eff {} diff) (eff {} random))` - optional annotation on `fn` nodes
- New tags: `handle-effect`, `effect-ann`, `eff`

**Test strategy (~25 tests):**
- Pure function has empty effect set
- Function calling `dropout` has `Random` effect
- Function calling differentiable ops has `Diff` effect
- `grad(f)` eliminates `Diff`, produces gradient function
- `withSeed(42, f)` eliminates `Random`
- Unhandled `Random` at program boundary -> error with repair suggestion
- Non-differentiable op inside `grad` -> error citing the specific op
- Effect propagation through higher-order functions: `map(f, xs)` inherits `f`'s effects
- Effect composition: function with both `Diff` and `Random` needs both handlers
- Fitness score degrades with unhandled effects
- Deep metadata validation: `{effects: [diff]}` on `fn` node is accepted
- `Accum` effect inferred for reduction operations in backward pass

### Acceptance Gate

`cargo test -p chelis-effects --test effect_inference` - all pass.

Manual: write a Surf program with dropout in training, `grad` for differentiation, and
explicit seed for deterministic inference. Compile. Verify:
- effect errors appear when handlers are missing
- effect errors disappear when handlers are added
- the compiled program produces deterministic output with a fixed seed

---

## 2b: Linear Types for Tensors

**Goal:** The type system prevents GPU memory leaks and enables safe in-place buffer
reuse.

### Spec Work (BEFORE implementation)

Produce `spec/04-type-system.md` linearity coverage with:

**Design choice: lightweight uniqueness (Futhark), not full linear types (Rust).**

The Futhark research shows that simple intra-procedural alias analysis with uniqueness
tracking is sufficient for the data-parallel case. Full Rust-style ownership with
borrowing, lifetimes, move semantics, and the borrow checker is over-engineering for
Chelis's use case. The RISC DAG structure, where tensors flow through a fixed set of
primitives, allows an even simpler model.

**The rule:** A tensor consumed by a RISC op is dead unless explicitly `copy()`'d.
Consumption is static - the compiler tracks which variables have been consumed and
rejects programs that use a consumed variable.

```text
let a = matmul(x, w)
let b = relu(a)
let c = add(a, b)        -- ERROR: a was consumed by relu on the previous line
```

Fix: `let c = add(copy(a), b)` - explicit copy makes the cost visible.

**Borrowing:** `&tensor` for read-only access. A borrow does not consume the tensor.
Borrows cannot be stored in data structures, returned from functions, or captured by
closures. This is deliberately restrictive - the simple rule is "borrows are temporary
views that exist for one function call."

```text
let a = matmul(x, w)
let shape = dims(&a)
let b = relu(a)
```

**Which types are linear?** All tensors. Scalars, booleans, integers, and ADT
constructors are not linear - they are freely copyable.

**Interaction with effects:**
- `Resource(D)` effect tracks where a tensor is allocated
- Linearity tracks when a tensor is deallocated (consumed)
- Together: the compiler knows every tensor's full lifecycle - allocated on device `D`,
  used by these ops, consumed (freed) at this point
- In-place optimization: when a tensor is consumed by an op that produces a same-shaped
  output, the compiler can reuse the buffer

**Interaction with patterns:**
- Pattern matching on a linear value consumes it
- The matched sub-components become the new linear bindings
- Exhaustive matching remains required

**Interaction with closures:**
- A closure that captures a linear variable consumes it
- The variable cannot be used after the closure is created

### Implementation Plan

**Extend `chelis-types` (not a new crate):**
- `linearity.rs`: walk typed Deep AST, track consumed variables, report
  use-after-consume errors
- Runs after effect inference (needs to know which ops consume vs borrow)
- Linearity checking is intra-procedural - no cross-function lifetime analysis
- Linearity errors carry repair suggestions: "Variable `a` was consumed by `relu` at
  line 5. To use it again, insert `copy(a)` before the first consumption."

**Surf syntax:**
- `copy(x)` - explicit copy (desugars to a `(copy {} x)` Deep node)
- `&x` - borrow syntax in function call arguments
- No ownership annotations on function signatures in v1

**Deep syntax:**
- `(copy {} x)` - explicit copy node
- `(borrow {} x)` - borrow node

**Compiler optimization (Phase 2b+):**
- After linearity checking passes, the IR lowering pass can mark consumed tensors as
  reusable buffers
- The memory planner from Phase 1c already does buffer reuse based on lifetime analysis;
  linearity provides a guarantee that the buffer is dead, not just an estimate
- For the HIP backend: in-place mutation when a consumed tensor feeds an op with same
  shape/dtype output

**Test strategy (~20 tests):**
- Use-after-consume detected and reported
- `copy(x)` allows reuse
- Borrowing does not consume
- Borrow cannot be stored/returned
- Pattern match consumes the scrutinee
- Closure capturing a linear variable consumes it
- Scalars are not linear
- Linearity error includes line number and repair suggestion
- In-place buffer reuse triggered when linearity is satisfied
- Interaction with effects: `Resource` + linearity tracks full tensor lifecycle

### Acceptance Gate

`cargo test -p chelis-types --test linearity` - all pass.

Manual: write a program that allocates two large tensors, consumes one to produce an
output, and verify the memory planner reuses the consumed tensor's buffer for a
subsequent allocation.

---

## 2c: Macro System

**Goal:** User-defined syntactic abstractions that expand to standard Deep before any
LLM-facing operation.

### Spec Work (BEFORE implementation)

Produce `spec/03-deep-syntax.md` macro coverage with:

**Settled design constraint:** LLMs interact exclusively with expanded Deep. Macros are
a human authoring layer that compiles away completely before any LLM-facing operation.
Provenance metadata in the `{}` slot traces expanded nodes back to their macro source.

**Macro definition (Deep):**

```text
(defmacro {} relu (params {} x)
  (app {source: (relu x)} (var {} max_elem) (var {} x)
    (lit {type: (t-prim {} f32)} 0)))
```

**Macro definition (Surf):**

```text
macro relu(x) = max_elem(x, 0.0)
```

**Expansion rules:**
- Macros expand top-down, outside-in
- Expansion is iterative until no macro invocations remain (fixed point)
- Expansion limit (default: 100 iterations) prevents infinite recursion
- After expansion, the result contains only base tags; unexpanded macro tags never
  appear in LLM-facing output

**Hygiene:**
- Racket-style scope sets are the ideal model
- Simple implementation start: rename macro-introduced bindings with a unique suffix
  (`_macro_N`)
- Manual hygiene breaking via `(unhygiene {} ...)` is possible for power users but
  discouraged in `SKILL.md`

**Provenance:**
- Every node produced by macro expansion carries
  `{source: (macro-name original-args...)}` in its metadata
- Error messages use provenance to report at both levels:
  "type error at node N (in expansion of `relu` at line 5)"
- Fitness scoring reports at both levels: per-node and per-macro-invocation aggregate

**Phase separation:**
- Macros run after parsing, before type checking
- Macro bodies are NOT type-checked during definition
- Macro outputs are type-checked normally after expansion
- A macro can produce ill-typed code; the error shows up at the expansion site with
  provenance

**Type-aware macros (stretch goal):**
- A macro can call the type checker on its arguments during expansion:
  `(typecheck-in-macro {} arg)`
- This enables compile-time type assertions and type-directed code generation
- Defer to Phase 2c+ if the basic macro system is already complex enough

**Interaction with effects:** Macros expand before effect inference. The expanded code
has effects inferred normally. No special interaction.

**Interaction with linearity:** Macros expand before linearity checking. The expanded
code is checked normally. A macro that duplicates a linear variable will produce a
linearity error at the expansion site; provenance explains which macro caused the
duplication.

### Implementation Plan

**Crate: `chelis-macros` (new)**
- `expand.rs`: macro expansion engine (iterative fixed point, hygiene renaming,
  provenance annotation)
- `parse.rs`: parse `defmacro` / `macro` definitions
- `lib.rs`: public API: `expand_macros(ast, macro_defs) -> ast`

**Integration into pipeline:**

```text
parse_surf -> desugar -> expand_macros -> type_check -> effect_infer
-> linearity_check -> lower -> ...
```

**Deep syntax additions:**
- `(defmacro {} name (params ...) body)` - macro definition
- `(macro-invoke {} name args...)` - unexpanded macro invocation used only before
  expansion
- Provenance: `{source: (relu x)}` metadata on expanded nodes

**Surf syntax additions:**
- `macro name(params) = body` - macro definition
- Macro invocations look like function calls

**Standard macros shipped with the language:**
- `relu(x)` -> `max_elem(x, 0.0)`
- `gelu(x)` -> expanded GELU approximation
- `dropout(x, rate)` -> conditional zeroing with `Random` effect
- `@differentiable` -> annotation macro that asserts the `Diff` effect

**Test strategy (~15 tests):**
- Simple macro expands correctly
- Hygiene: macro-introduced variable does not shadow outer scope
- Provenance metadata present on expanded nodes
- Nested macro expansion terminates
- Expansion limit prevents infinite recursion
- Type error in expanded code reports provenance
- Linearity error in expanded code reports provenance
- Standard macros (`relu`, `gelu`, `dropout`) expand and type-check
- LLM-facing output contains no macro tags

### Acceptance Gate

`cargo test -p chelis-macros --test expansion` - all pass.

Manual: define a custom `attention(q, k, v)` macro in Surf, expand it, verify the Deep
output is macro-free, type-checks, and compiles to both backends.

---

## 2d: vmap Transform

**Goal:** Automatic vectorization - write a function over single examples, apply it to
batches.

### Design

`vmap(f, axis=0)` takes a function
`f: tensor[a, b, f32] -> tensor[c, f32]` and produces
`g: tensor[batch, a, b, f32] -> tensor[batch, c, f32]` by adding a batch dimension to
every operation in the DAG.

**Implementation strategy:** DAG-to-DAG rewrite. Walk the function's DAG, and for every
node:
- Elementwise ops: add the batch dimension to input/output shapes
- Reductions: reduction axis shifts by 1
- Matmul: becomes batched matmul (`bmm`) - the `expand` + `mul` + `sum` pattern with an
  additional outer dimension
- Movement ops: dimension indices shift by 1

**Interaction with `grad`:** `grad(vmap(f))` and `vmap(grad(f))` must produce the same
result. Both are DAG-to-DAG rewrites, so they compose naturally; the order affects the
intermediate DAG shape but the final result is mathematically equivalent.

**Interaction with effects:** `vmap` preserves effects. If `f` has `Random`, `vmap(f)`
has `Random` (each batch element samples independently).

### Implementation Plan

**Extend `chelis-ir`:**
- `vmap.rs`: DAG-to-DAG rewrite that adds a batch dimension
- Operates on the RISC DAG (after lowering, before codegen)

**Surf syntax:** `vmap(f, axis=0)` - call-like syntax

**Deep syntax:** `(vmap {} fn axis)`

**Test strategy (~10 tests):**
- `vmap(elementwise_fn)` adds a batch dimension
- `vmap(reduction_fn)` shifts the reduction axis
- `grad(vmap(f))` matches `vmap(grad(f))` numerically
- `vmap` with explicit axis parameter
- Nested `vmap` (batch + sequence dimensions)

### Acceptance Gate

`cargo test -p chelis-ir --test vmap` - all pass, including `grad`/`vmap` composition
tests.

---

## 2e: Tide Agent API + MCP

**Goal:** A machine-facing interface to the Chelis compiler for AI coding agents.

### Design

HTTP/JSON API that wraps every compiler pass as an endpoint. An AI agent connects via
MCP and uses the compiler as a tool. The shipped contract uses explicit stable wire
models rather than serializing compiler AST/DAG structs directly.

**Endpoints:**

| Endpoint | Input | Output |
|---|---|---|
| `POST /parse` | Surf or Deep source | AST (success) or parse errors |
| `POST /desugar` | Surf source | Canonical Deep |
| `POST /check` | Surf or Deep source | Fitness report + structured diagnostics |
| `POST /lower` | Surf or Deep source | RISC DAG (JSON-serialized) |
| `POST /compile` | Surf or Deep source + target | Generated C/HIP files + flags |
| `POST /eval` | Surf or Deep source + named bindings | Evaluated roots |
| `POST /grad` | Surf or Deep source + output/wrt names | Differentiated DAG JSON |
| `POST /validate` | Surf or Deep source + mode | Conformance result |
| `POST /decompile` | Deep source | Surf source |

**MCP tools (wrapping the HTTP endpoints):**
- `chelis_check`
- `chelis_compile`
- `chelis_desugar`
- `chelis_decompile`
- `chelis_eval`
- `chelis_grad`
- `chelis_validate`

**Batch mode:** `POST /batch` accepts an array of requests, returns results for all.
Essential for evolutionary loops and trajectory collection (Phase 3g training
pipeline).

**Implementation:** Rust HTTP server (`axum`) wrapping the existing compiler crates.
The MCP server is a thin stdio adapter over the same compiler adapter layer. The
server runs as `chelis tide serve` (HTTP) or `chelis tide mcp` (stdio MCP).

### Implementation Plan

**Crate: `chelis-tide` (new)**
- `schema.rs`: stable wire models and request/response types
- `compiler.rs`: shared compiler adapter and name-based eval binding bridge
- `http.rs`: HTTP server with endpoints
- `mcp.rs`: MCP protocol adapter
- `lib.rs`: shared service exports

**CLI:**
- `chelis tide serve --port 8080` starts the HTTP server
- `chelis tide mcp` starts the MCP server on stdio

**Test strategy (~15 tests):**
- Each endpoint returns correct results for valid input
- Each endpoint returns structured errors for invalid input
- Fitness score endpoint matches CLI `chelis check` output
- Batch endpoint handles mixed success/failure
- MCP tool calls map correctly to HTTP endpoints
- Server handles concurrent requests without data races
- `/eval` resolves request bindings by input name rather than position
- `/grad` returns DAG JSON and rejects unsupported non-scalar outputs cleanly

### Acceptance Gate

`cargo test -p chelis-tide --test api` - all pass.

Manual: connect Claude (or another MCP-capable agent) to `chelis tide mcp`, have it
write a program, check fitness, fix errors based on feedback, compile, and run.
For a reproducible external-process validation harness, run
`python scripts/redteam_tide_phase2e.py`.

---

## 2f: Tide LSP

**Goal:** Language Server Protocol support for VS Code / Cursor / Windsurf.

### Design

Built on the same compiler infrastructure as the Agent API (2e). The LSP is a protocol
adapter - it translates LSP messages to compiler API calls and formats the results as
LSP responses.

**Features:**
- **Diagnostics:** real-time compiler diagnostics and fitness score in the status bar
- **Completion:** built-in scope plus standard-library surface
- **Hover:** type information and Deep form of the current Surf selection where
  available
- **Go-to-definition:** navigate to function definitions, type aliases, module sources
- **TextMate grammar:** a `.tmLanguage.json` regex grammar for Surf and Deep, bundled
  with the VS Code extension. Provides instant keyword, string, comment, and literal
  highlighting before the LSP server is ready.
- **Surf <-> Deep toggle:** command to show/toggle the Deep representation of the
  current selection

**Implementation note:** v1 ships without `salsa`.
The server recomputes from the full current document and keeps the compiler boundary
clean so a later `salsa` migration remains mechanical.

### Implementation Plan

**Crate: `chelis-lsp` (new)**
- `server.rs`: LSP server (`tower-lsp`)
- `analysis.rs`: document parsing, symbol indexing, diagnostics mapping, and Deep-view
  preparation
- `commands.rs` equivalent inside the server layer for read-only Deep view and fitness
  status

**VS Code extension:** minimal JS extension that bundles the TextMate grammar for
immediate syntax highlighting, starts the LSP server, and provides Chelis-specific UI
(Deep toggle button, fitness score in status bar).

**Test strategy (~10 tests):**
- Diagnostics match CLI `chelis check` output
- Completion includes built-in scope
- Hover shows correct type information
- Deep toggle produces valid Deep
- Edit -> re-check cycle completes in <500ms for MNIST-sized programs on the manual
  editor gate

### Acceptance Gate

Open a `.ch` file in VS Code with the extension installed. Syntax highlighting appears
immediately on open. Type errors appear in
real time. Hover shows types. Deep toggle works. Fitness score is visible.

---

## 2g: Tide TUI (`chelis cove`)

**Goal:** A terminal-based coding environment that ships as the default Chelis
development experience.

### Design

Built on Ratatui. The TUI is a frontend to the Agent API (2e) and uses the same
compiler services as the LSP (2f).

**Panels:**
- Editor pane: Surf code with syntax highlighting
- Deep pane: live canonical Deep of the current function
- Type/effect pane: inferred types, effects, linearity annotations
- Output pane: evaluation results, training loss, compilation output
- Agent pane (agent mode): AI writes Deep, human reviews in Surf

**Flagship feature: Surf <-> Deep live toggle.** The programmer writes in Surf and sees
the AI's representation in Deep in real time. No other language has this. It makes the
"written by AIs, for AIs" thesis concrete and visible.

**Agent mode:** The TUI hosts an AI agent session - the agent proposes Deep programs,
the human reviews them in Surf, and edits are round-tripped through the compiler. This
requires the MCP server (2e) running in the background.

### Implementation Plan

**Crate: `chelis-cove` (new)**
- `app.rs`: application state, event loop
- `ui.rs`: panel layout, rendering (Ratatui)
- `editor.rs`: text editing with tree-sitter-based Surf syntax highlighting
- `agent.rs`: agent mode integration (connects to `chelis tide mcp`)
- `live.rs`: live desugar/check/decompile pipeline (debounced, async)

**Tree-sitter grammar (`grammar.js`):** A tree-sitter grammar for Surf and Deep generates
a C parser used by the editor pane for incremental reparsing on each keystroke.
Tree-sitter is required here because Ratatui TUIs cannot use TextMate grammars or LSP
semantic tokens — terminal editors need their own parser for responsive highlighting. The
same grammar also benefits Neovim, Emacs, and Helix users who support tree-sitter
natively.

**CLI:**
- `chelis cove` launches the TUI
- `chelis cove --file examples/mnist.ch` opens a specific file

**Test strategy:** TUI testing is primarily manual. Automated tests cover non-UI logic
(live pipeline, agent mode protocol). The acceptance gate is a human completing the
MNIST tutorial entirely within `chelis cove`.

### Acceptance Gate

A user (not the developer) can: open `chelis cove`, load `examples/mnist.ch`, see the
Deep form, edit the Surf code, see type errors update live, compile, and run training
- without leaving the TUI.

---

## 2s: Seed Corpus (Continuous Throughout Phase 2)

**Goal:** Build a corpus of 50-100 Chelis programs that serve as training data for the
local model (Phase 3g), test cases for the compiler, and documentation for users.

**Strategy:** Every time a new compiler feature is implemented, write 2-3 programs that
exercise it. By the end of Phase 2, the corpus grows organically to 50-100 programs
covering the full language surface.

**Stratification (informed by the complexity research):**
- ~20 single-operation programs (1-3 ops, baseline structural competence)
- ~40 single-layer programs (5-15 ops, one model component)
- ~30 multi-layer programs (15-40 ops, composition of components)
- ~10 full models (40+ ops, end-to-end training pipelines)

**Each program ships as:**
- `examples/corpus/NNN_name.ch` - Surf source
- `examples/corpus/NNN_name.dp` - canonical Deep (generated by the compiler)
- `examples/corpus/NNN_name.json` - fitness score, type info, effect info

**Acceptance:** The corpus directory exists, contains >=50 programs, all compile and
type-check with fitness >=0.9, and the complexity distribution roughly matches the
stratification targets.

---

## Red Team Checkpoint: Phase 2

After all sub-phases, before declaring Phase 2 complete:

**Effects:**
- [ ] Construct a program with unhandled effects that the compiler does not catch
- [ ] Verify `Random` effect is eliminated by `withSeed` (deterministic output)
- [ ] Verify `Diff` effect prevents non-differentiable ops in `grad` body
- [ ] `Accum`-typed loops in backward pass are parallelized (not sequentialized)

**Linear types:**
- [ ] Attempt to use a consumed tensor - compiler rejects
- [ ] Attempt to leak GPU memory despite linearity - compiler prevents
- [ ] In-place buffer reuse triggered for consumed-then-same-shape-produced pattern
- [ ] Borrowing prevents closure capture

**Macros:**
- [ ] Macro expansion produces only base tags (no macro tags in LLM-facing output)
- [ ] Hygiene prevents variable capture
- [ ] Type error in expanded code includes provenance trace to macro invocation
- [ ] Linearity error in expanded code includes provenance trace

**`vmap`:**
- [ ] `grad(vmap(f)) == vmap(grad(f))` numerically
- [ ] Nested `vmap` works for batch + sequence dimensions

**Agent API:**
- [ ] MCP agent can write, check, fix, compile, and run a program via tools
- [ ] Batch mode handles 100+ programs without timeout
- [ ] Invalid input does not crash the server

**LSP:**
- [ ] VS Code extension works in Cursor and Windsurf
- [ ] Real-time diagnostics appear in <500ms
- [ ] Deep toggle shows correct canonical Deep

**TUI:**
- [ ] User completes MNIST tutorial entirely within `chelis cove`
- [ ] Agent mode: AI proposes, human reviews, edits round-trip

**Seed corpus:**
- [ ] >=50 programs, all fitness >=0.9
- [ ] Complexity distribution matches stratification targets
- [ ] Every Phase 2 feature exercised by at least 2 corpus programs
