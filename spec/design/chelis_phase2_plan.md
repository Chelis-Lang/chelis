# Phase 2: Language Maturity - Expanded Implementation Plan

## Scope

**Phase 2 deliverable:** The language is usable by researchers. Effects, linear types,
macros, the agent API, and tooling make Chelis a credible alternative to PyTorch for
specific workloads. A researcher should be able to write, type-check, differentiate,
compile, train, and debug a model - with AI assistance - using only the Chelis
toolchain.

**Phase 2 does NOT deliver:** A package manager (Phase 3a), alternative backends like
StableHLO/FX/Triton (Phase 5), Python FFI (Phase 3b), or the local coding model
(Phase 4). Phase 2 is about making the language complete; Phase 3 is about making the
ecosystem polished and externally usable. Full cross-function specialization for
user-defined library helpers is also outside the Phase 2 completion claim; it is tracked
as a separate compiler/codegen workstream in
[`cross_function_specialization.md`](cross_function_specialization.md).

---

## Dependency Graph

```text
Phase 1 carry-forward fixes ----------------------------------------------+
  (layer_norm symbolic-axis follow-up, pad/shrink, Deep round-trip)       |
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

Seed corpus (originally 2s) is deferred to Phase 4a.
```

**Critical path:** 2a -> 2b -> 2c (each depends on the previous for interaction
rules). Tooling (2e -> 2f -> 2g) can run in parallel with the type-system track after
2e has enough compiler API surface.

**Recommended execution order:**
1. Phase 1 carry-forward fixes (remaining backend follow-ups)
2. 2a: Effects (long lead, spec-heavy, informs everything else)
3. 2e: Agent API (unblocks tooling track and SKILL.md-based workflows)
4. 2b: Linear types (depends on effect system design for interaction rules)
5. 2d: `vmap` (independent of 2b/2c, can slot in anytime after 2a)
6. 2f: LSP (depends on 2e API surface)
7. 2c: Macros (last type-system feature, depends on both effects and linearity for
   interaction rules)
8. 2g: TUI (depends on 2e + 2f, capstone tooling deliverable)

---

## Phase 1 Carry-Forward Fixes

Before starting Phase 2 proper, address the known limitations that affect Phase 2
work:

### Symbolic Runtime-Axis Follow-Up

**Current state:** symbolic dimensions now ship in both backends through the stable
tensor ABI, with bindings inferred from input tensor metadata at runtime. Batch and
sequence-style dims no longer require recompilation on the supported Phase 1 surface.

**Remaining gap:** `mean`/`layer_norm` still require a concrete normalized-axis extent,
so a symbolic hidden size is not yet supported.

**Follow-up direction:**
- carry a runtime divisor for axis-size-dependent normalization paths
- keep the stable tensor ABI: derive bindings from input metadata rather than adding
  scalar function parameters
- preserve the existing repeated-occurrence validation across inputs

**Scope:** This is an IR-through-codegen change that touches both backends,
the memory planner, and the CLI. Test: same MNIST program works with `batch=32` and
`batch=128` without recompilation.

### Pad/Shrink in HIP Backend

**Problem:** `todo!()` in HIP `emit.rs`. These ops exist for windowing/slicing.

**Solution:** Implement as elementwise kernels with boundary checks. `pad` reads from
source if in bounds, writes fill value otherwise. `shrink` reads from the shrunk
region.

**Scope:** ~2 days. Small, mechanical. Test: `pad`/`shrink` GPU output matches CPU
output.

### Cross-Function Specialization Follow-Up

**Current state:** Inline `matmul`, inline hand-written `expand -> mul -> sum`
patterns, and simple C user-defined helper wrappers now hit the C backend BLAS path.
The compiler derives narrow `blas_matmul` summaries from pure tensor helpers and simple
wrappers, consumes those summaries at C callsites, and still emits the helper body for
debugging and non-specialized callers.

**Phase boundary:** This is a known limitation, not a Phase 2 language-maturity
deliverable. It should continue in the dedicated cross-function specialization
workstream, using the path described in
[`cross_function_specialization.md`](cross_function_specialization.md): auto-derived
function summaries plus verified callsite emission rules. Remaining scope includes HIP
summary consumption, gather/scatter summaries, explicit negative diagnostics for
summary-derived-but-callsite-rejected cases, and broader helper shapes.

**Workaround status:** clang LTO may inline generated helper functions and recover some
ordinary C optimization, but it cannot be the compiler's backend-dispatch story. The
decision to emit `cblas_sgemm`, hipBLAS, or a generic fused loop must be made by Chelis
before the native compiler sees the generated C/HIP. The first C summary slice no
longer relies on clang LTO for the simple helper cases covered by
`crates/chelis-cli/tests/cross_library_semantic_gap.rs`.

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

**Settled design decisions:**

| Design point | Decision |
|---|---|
| `Diff` | compiler capability, not a user-visible boundary effect |
| `Accum` | internal-only in v1; the user does not handle it directly |
| Boundary effects | `Random` and `Resource(D)` |
| Crate split | effect types in `chelis-types`; inference/checking in `chelis-effects` |
| Compiler pipeline | upgrade `CheckedProgram` first so downstream passes consume annotated Deep |

**Why `Accum` (from Dex):** Gradient accumulation (`sum` of adjoint contributions from
multiple consumers) is an `Accum` pattern. Typing it as `Accum` rather than general
`State` tells the GPU backend the backward-pass loop is parallelizable. Without this
distinction, the compiler would have to conservatively sequentialize gradient
accumulation.

**No user-defined effects in v1.** The shipped Phase 2a surface is intentionally
closed and compiler-known.
User-defined effects (Phase 3+) add handler semantics, effect polymorphism, and
resumable computations - complexity that is hard for both humans and LLMs.

**Current shipped Phase 2a subset:**
- effect inference runs after HM type inference on the same annotated Deep tree
- `CheckedProgram` carries annotated Deep with `type` metadata written onto the returned
  tree, and downstream passes consume that upgraded representation
- `dropout(x, rate)` is the minimum concrete `Random` source
- `with seed(42) { ... }` handles `Random`
- `with device("...") { ... }` marks a resource region validated against
  `chelis build --target ...`; host C admits only exact `cpu` under
  `spec/04-type-system.md` [04-EFF-2]
- unhandled top-level `Random` is a check error with repair guidance
- seeded `dropout` is implemented in lowering/eval/AD, but not yet in emitted C/HIP
  codegen
- later Phase 3j-pre closure adds C host preservation for `with seed(...)` across
  direct `uniform_like` and nested stdlib/user calls; seeded `dropout` codegen remains
  deferred

**What this plan does NOT yet claim as shipped:**
- full row-polymorphic higher-order effect inference
- user-visible `Diff` effect checking
- user-visible `Accum` inference/handling
- backend codegen for seeded `dropout`

**Interaction with existing type system:**
- Effects do NOT change HM inference - types are inferred first, effects inferred second
- Effect annotations in Surf are optional:
  `sig f[n]: tensor[n, f32] -> tensor[n, f32] ! {Random}` is valid but never required
- Effect annotations in Deep:
  - `t-fn` type expressions may carry `eff: (effects {} ...)`
  - checked `fn` nodes may carry inferred `effects: (effects {} ...)` for the effect
    information the checker synthesizes today
  - `Resource(Device)` is currently enforced at the handler/build boundary rather than
    being synthesized back onto checked `fn` metadata

**Error messages (the LLM-facing concern):**
- Current shipped shape:
  "Function `predict` has unhandled effect `Random`; `dropout` requires
  `with seed(...)`."
- Current shipped repair suggestions are concrete but not yet source-located.
- Line-numbered effect diagnostics remain a follow-up rather than a shipped guarantee.
- The fitness score should degrade proportionally to the number of unhandled effects.

### Implementation Plan

**Crate split**
- `chelis-types`: effect vocabulary / shared type-layer representation
- `chelis-effects`: effect inference, handler validation, build-boundary validation,
  structured effect errors

**Integration points:**
- `chelis-types::check_ir_program()` returns the upgraded `CheckedProgram`
- after type checking succeeds, run `chelis-effects::check_program()` on the annotated AST
- inferred effect information is stored as metadata on checked `fn` nodes where the
  checker synthesizes it today
- `chelis-ir::lower()` threads handler-provided seeds into `dropout`
- `chelis-cli build` validates `with device(...)` regions against the requested target
- Fitness scoring extended: unhandled effects reduce fitness proportionally

**Surf syntax additions:**
- `sig f: A -> B ! {Random}` - optional effect annotation on signatures
- `with seed(42) { ... }` - handler block syntax
- `with device("gpu:0") { ... }` - handler block syntax
- `grad(f)` remains a compiler transform, not an effect handler

**Deep syntax additions:**
- `(handle-effect {effect: random} seed-expr body)` - seeded stochastic region
- `(handle-effect {effect: resource} device-expr body)` - resource region
- `(effects {} random (resource {} "gpu:0"))` - effect-set helper form used in metadata
- New Deep tags in the shipped subset: `handle-effect`, `effects`, `resource`

**Test strategy (shipped subset):**
- Pure function has empty effect set
- Function calling `dropout` has `Random` effect
- `with seed(42) { ... }` eliminates `Random`
- Unhandled `Random` at program boundary -> error with repair suggestion
- Fitness score degrades with unhandled effects
- Deep metadata validation: `(effects {} random)` on `fn` metadata is accepted
- seeded lowering/eval is deterministic for same seed and observably different for
  different seeds
- build-target mismatch is reported for incompatible `with device(...)` regions;
  host C admits only exact `cpu`, and rejects every other selector before
  artifact emission

### Acceptance Gate

Current shipped-subset oracle:

```sh
cargo test -p chelis-effects
cargo test -p chelis-types --test linearity
cargo test -p chelis-cli --test cli check_reports_a_keyless_dropout_as_an_arity_error
cargo test -p chelis-compiler-api --test resource_target_admission
cargo test -p chelis-cli --test issue_735_device_fence
```

Supporting manual gates for the local HIP-capable validation path:

```sh
cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
```

Expected success condition: all ignored HIP correctness tests pass on a machine with a
working ROCm + `hipcc` environment.

Concrete Phase 2a effect-surface manual check:

```sh
tmpdir="$(mktemp -d)"
cat > "$tmpdir/unhandled_random.ch" <<'EOF'
x: tensor[32, f32] = x
y: tensor[32, f32] = dropout(x, 0.5)
EOF
cat > "$tmpdir/handled_random.ch" <<'EOF'
x: tensor[32, f32] = x
y: tensor[32, f32] = with seed(42) { dropout(x, 0.5) }
EOF
cargo run -q -p chelis-cli -- check "$tmpdir/unhandled_random.ch"
cargo run -q -p chelis-cli -- check "$tmpdir/handled_random.ch"
```

Expected success condition:
- the first `check` output reports `UnhandledEffect` / `Random`
- the second `check` output reports no effect errors

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

**Superseded rule:** Early Phase 2b required a tensor consumed by a RISC op to be
explicitly `copy()`'d before reuse. The active `copy-drop` contract in
`spec/design/implicit_linearity.md` keeps explicit `copy()` valid, but ordinary
source-level consuming fan-out is handled by inserted `RiscOp::Copy` nodes.

```text
a = matmul(x, w)
b = relu(a)
c = add(a, b)        -- old Phase 2b: ERROR; copy-drop: compiler inserts Copy
```

Explicit `c = add(copy(a), b)` remains valid and lowers to the same `RiscOp::Copy`.

**Borrowing:** `&tensor` for read-only access. A borrow does not consume the tensor.
Borrows cannot be stored in data structures, returned from functions, or captured by
closures. This is deliberately restrictive - the simple rule is "borrows are temporary
views that exist for one function call."

```text
a = matmul(x, w)
shape = dims(&a)
b = relu(a)
```

**Which values are tracked linearly?** Tensors, plus tuple/composite values that carry
tensor payloads. Scalars, booleans, integers, and ordinary non-tensor payloads remain
freely copyable.

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
- Linearity errors carry repair suggestions and source offsets: "Variable `a` was
  consumed by `relu` at offset 42. To use it again, insert `copy(a)` before the first
  consumption."

**Surf syntax:**
- `copy(x)` - explicit copy (desugars to a `(copy {} x)` Deep node)
- `&x` - borrow syntax in function call arguments
- No ownership annotations on function signatures in v1

**Deep syntax:**
- `(copy {} x)` - explicit copy node
- `(borrow {} x)` - borrow node

**Compiler optimization (Phase 2b+):**
- After linearity checking passes, the IR lowering pass can mark consumed tensors as
  reusable-input candidates on same-shape/same-dtype paths
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
- Linearity error includes source location and repair suggestion
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
    (lit {type: (t-prim {} f32)} 0.0)))
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
- `lib.rs`: macro expansion engine, binder-only hygiene, provenance annotation, and
  the shipped standard macro prelude
- internal compiler forms are parsed/desugared as `defmacro` and expanded away before
  any public Deep surface is emitted

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

Current shipped public-boundary rule:

- `defmacro` / `macro-invoke` are compiler-internal only
- `chelis deep`, `check`, `build`, `eval`, the e2e pipeline, and validation of
  desugared output all run after expansion
- strict public Deep validation still accepts only the ordinary Deep tag vocabulary

**Surf syntax additions:**
- `macro name(params) = body` - macro definition
- Macro invocations look like function calls

Current shipped resolution / hygiene behavior:

- lexical bindings block macro expansion
- then user-defined top-level macros are considered
- then the standard prelude macros are considered
- otherwise the form remains an ordinary call
- hygiene renames only binders introduced by the expansion; free references in the
  macro body remain free and resolve in the caller's scope

**Standard macros shipped with the language:**
- `linear_layer(x, w, b)` -> `add(matmul(x, w), expand(b, 0, batch))`
- `residual(x, f)` -> `add(x, f(x))`
- `cross_entropy(logits, labels)` -> the standard `softmax` / `log` / `sum` / `mean`
  composition used by the current executable corpus

**Test strategy (~15 tests):**
- Simple macro expands correctly
- Hygiene: macro-introduced variable does not shadow outer scope
- Provenance metadata present on expanded nodes
- Nested macro expansion terminates
- Expansion limit prevents infinite recursion
- Type error in expanded code reports provenance
- Linearity error in expanded code reports provenance
- Standard macros (`linear_layer`, `residual`, `cross_entropy`) expand and type-check
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

`vmap(f)` takes a function at the default zero axis
`f: tensor[a, b, f32] -> tensor[c, f32]` and produces
`g: tensor[batch, a, b, f32] -> tensor[batch, c, f32]` by adding a batch dimension to
every operation in the DAG.

**Implementation strategy:** DAG-to-DAG rewrite. Walk the function's DAG, and for every
node:
- Elementwise ops: add the batch dimension to input/output shapes
- Reductions: reduction axis shifts by 1
- Matmul: keep the existing `expand` + `mul` + `sum` decomposition on the already-batched
  shapes; do not add a new batched RISC op in 2d
- Movement ops: dimension indices shift by 1

**Interaction with `grad`:** `vmap(grad(f))` means per-example gradients. Direct
`grad(vmap(f))` remains rejected by the shipped source-level `grad` path because `vmap`
turns a scalar-returning function into a batched tensor-returning function unless the
caller explicitly reduces it back to a scalar first.

The direct executable lowering path for `vmap(grad(f))` composes the existing rewrites in
order: lower `f` to a single-example DAG, run `grad_dag`, then run the `vmap` rewrite on
that gradient DAG. Phase 2db extends this path to flat tuple-valued gradient payloads
from multi-parameter `grad(..., wrt=(...))` without introducing tuple nodes into the
RISC DAG.

**Interaction with effects:** `vmap` preserves effects. If `f` has `Random`, `vmap(f)`
has `Random` (each batch element samples independently).

### Implementation Plan

**Extend `chelis-ir`:**
- `vmap.rs`: DAG-to-DAG rewrite that adds a leading batch dimension in canonical axis-0
  form
- public nonzero axes canonicalize to axis-0 via `permute`, run the rewrite, then
  `permute` outputs back
- the executable subset lowers direct `vmap(f)(args...)` and `vmap(grad(f))(args...)`
  applications away before ordinary DAG codegen

**Surf syntax:** `vmap(f)` - call-like syntax at the default zero axis, and the resulting function can be
applied as `vmap(f)(xs)`

**Deep syntax:** `(vmap {} fn axis)`

**Test strategy (~10 tests):**
- `vmap(elementwise_fn)` adds a batch dimension
- `vmap(reduction_fn)` shifts the reduction axis
- `vmap(grad(f))` matches a per-example loop baseline
- `vmap` with explicit axis parameter
- Nested `vmap` (batch + sequence dimensions)
- batched matmul stays correct through the generic decomposition and now specializes to
  runtime-sized BLAS when the generated matrix slices are contiguous

### Acceptance Gate

```sh
cargo test -p chelis-ir --test vmap
cargo test -p chelis-e2e --test spec_suite
cargo test -p chelis-e2e --test pipeline
```

Expected success condition: the structural rewrite suite, the numerical `vmap(grad(...))`
baselines, and the tuple-root lowering path all pass.

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
| `POST /test` | Surf or Deep source + function name + num_cases | Property test results (shapes, determinism, gradient check) |

**MCP tools (wrapping the HTTP endpoints):**
- `chelis_check`
- `chelis_compile`
- `chelis_desugar`
- `chelis_decompile`
- `chelis_eval`
- `chelis_grad`
- `chelis_validate`
- `chelis_test` — type-driven property testing

**Batch mode:** `POST /batch` accepts an array of requests, returns results for all.
Essential for evolutionary loops and trajectory collection (Phase 4c training
pipeline).

**Type-driven property testing (`/test`):** Generate test inputs automatically from
function type signatures. A function
`def f(x: tensor[batch, 784, f32]) -> tensor[batch, 128, f32]` has enough information
to generate random valid inputs — the compiler knows the shapes, dtypes, and dimension
constraints. The endpoint generates `num_cases` random inputs of the correct shapes,
calls the function via the evaluator, and verifies: output shape matches the declared
return type, pure functions are deterministic (same inputs → same output),
`Diff`-annotated functions have finite-difference-verified gradients. No test code
written by anyone — the type signature IS the test specification. Also available as
`chelis test <file> --fn <name>` on the CLI.

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

**Proof boundary:** automated coverage today is library-level via `cargo test -p chelis-lsp`.
That proves analysis behavior and command preparation, not full editor-host protocol
integration. VS Code / Cursor / Windsurf behavior remains a documented manual gate.

### Acceptance Gate

Open a `.ch` file in VS Code with the extension installed. Syntax highlighting appears
immediately on open. Type errors appear in
real time. Hover shows types. Deep toggle works. Fitness score is visible.

---

## 2g: Tide TUI (`chelis cove`)

**Goal:** A terminal-based coding environment that ships as the default Chelis
development experience.

### Design

Built on Ratatui. The shipped v1 uses the same direct compiler services as the LSP (2f)
without introducing a background daemon or `salsa`.

**Panels:**
- Editor pane: Surf code with syntax highlighting
- Deep pane: live canonical Deep of the current buffer, rendered with the canonical
  pretty Deep layout
- Diagnostics pane: fitness and structured compiler diagnostics
- Output pane: compile preview and evaluator output

**Flagship feature: Surf <-> Deep live toggle.** The programmer writes in Surf and sees
the AI's representation in Deep in real time. No other language has this. It makes the
"written by AIs, for AIs" thesis concrete and visible.

### Implementation Plan

**Crate: `chelis-cove` (new)**
- `app.rs`: application state, event loop
- `ui.rs`: panel layout, rendering (Ratatui)
- `editor.rs`: text editing plus tree-sitter-based Surf/Deep highlighting
- `live.rs`: in-process desugar/check/eval/compile helpers

**Tree-sitter grammar (`grammar.js`):** A tree-sitter grammar for Surf and Deep generates
a C parser used by the editor pane for incremental reparsing on each keystroke.
Tree-sitter is required here because Ratatui TUIs cannot use TextMate grammars or LSP
semantic tokens — terminal editors need their own parser for responsive highlighting. The
same grammar also benefits Neovim, Emacs, and Helix users who support tree-sitter
natively.

**CLI:**
- `chelis cove` launches the TUI
- `chelis cove --file examples/mnist.ch` opens a specific file
- canonical Deep elsewhere in the toolchain now defaults to pretty-printed `.dp`; use
  `chelis deep --flat` for flat machine-oriented output and `chelis fmt --check` to
  verify canonical Surf/Deep formatting without rewriting files

**Test strategy:** TUI testing is primarily manual. Automated tests cover non-UI logic
(live pipeline, zero-binding eval, file loading, CLI surface). The acceptance gate is a
human running the documented manual oracle and confirming live Deep/diagnostic updates
plus compile/eval output from inside `chelis cove`.

**Proof boundary:** automated tests cover the compiler-facing and CLI-facing pieces of
`chelis cove`, not the interactive terminal event loop itself. The interactive UI claim
remains manual-gate only.

### Acceptance Gate

A user (not the developer) can run `cargo run -p chelis-cli -- cove --file
examples/mnist.ch`, see the Deep form, edit the Surf code, see fitness/diagnostics
update live, save, and trigger compile/eval output without leaving the TUI.

---

## Deferred: Seed Corpus

Seed corpus work was originally tracked as 2s. It is now deferred to Phase 4a and is
not part of the Phase 2 completion gate.

---

## Red Team Checkpoint: Phase 2

After all sub-phases, before declaring Phase 2 complete:

**Effects:**
- [ ] Construct a program with unhandled effects that the compiler does not catch
- [ ] Verify `Random` effect is eliminated by `with seed(...)` (deterministic output)
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
- [ ] `vmap(grad(f))` matches a per-example loop baseline numerically
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
