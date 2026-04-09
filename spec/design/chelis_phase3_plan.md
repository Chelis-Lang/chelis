# Phase 3: Ecosystem Foundations - Expanded Implementation Plan

## Context

Phase 2 makes the language usable by researchers.
Phase 3 makes Chelis usable by people other than its creator.
That means: the public Surf style is stabilized, packages can be shared, Python can
interoperate with the compiler/runtime, and the research-facing extensions are strong
enough to support publication.

**Phase 3 deliverable:** A researcher can write idiomatic pipe-first Chelis with
short-form block bindings, consume libraries through Reef, exchange tensors with Python
through DLPack, script the compiler from Python through PyO3, and point to a credible
research roadmap for both type-system extensions and mechanized conformance.

**Phase 3 does NOT deliver:** The local coding model (Phase 4), alternative backends
like StableHLO / FX / Triton (Phase 5), or multi-GPU support (Phase 5d).

This phase has two explicit tracks across six numbered sub-phases:

- engineering track: `3e -> 3f -> 3a -> 3b`
- research track: `3c` and `3d` in parallel

`3f` (SKILL.md v2) is the public-teaching/documentation refresh that lands after the
style foundation is settled and before the Phase 4 model-training track begins.

---

## Dependency Graph

```text
Engineering track

3e: Style Foundation -----------------------------------------------+
  (pipes, optional block let, width-aware formatting, examples)     |
         |                                                          |
         v                                                          |
3f: SKILL.md v2 Refresh                                             |
  (teaching surface locked to 3e idiom, before Phase 4)             |
         |                                                          |
         v                                                          |
3a: Package System (Shells + Reef) ---------------------------------+--> 3b: Python FFI
  (dogfood with chelis-std)                                              (DLPack + PyO3)

Research track

3c: Research Type Extensions (paper-first, corpus-validated)
3d: Lean Formalization (independent conformance-oracle track)
```

**Recommended execution order:**

1. `3e` style foundation
2. `3f` SKILL.md v2 refresh (locks the teaching idiom before corpus collection)
3. `3a` package system
4. `3b` Python FFI
5. `3d` Lean formalization in parallel
6. `3c` research type extensions

`3e` goes first because every later deliverable publishes examples, package code, or
teaching material that should already be in the final public idiom. `3f` follows
immediately so the SKILL.md is locked to the finalized style before Phase 4 corpus
collection and model training.

---

## 3e: Style Foundation

**Goal:** Pipe-first, short-binding Surf becomes the default public Chelis style across
decompiler output, formatting, examples, docs, and teaching material.

This is not only a formatter tweak.
It is a coordinated rendering and corpus cleanup covering:

- pipe-chain detection in decompilation
- short-form block bindings (`name = expr`)
- width-aware multiline rendering for pipe chains
- example/spec/tutorial refresh

### Decompiler Behavior

Given a sequence of sequential bindings in a block, scan for linear chains:

```text
a = f(x)      -- used once, as first arg to g
b = g(a, y)   -- used once, as first arg to h
c = h(b)      -- end of chain
```

Collapse to:

```text
c = f(x) |> g(y) |> h
```

Pipe-eligibility rule:

1. the binding RHS is a function call
2. the bound value is used exactly once in the block
3. that single use is as the first positional argument to another function call
4. the use appears in the immediately following binding

When a chain is detected, emit a single pipe expression.
The first stage is the full initial call; later stages are rendered as the function plus
its non-piped arguments.

### Named Binding Preservation

Do not pipe through semantically meaningful names.
Keep explicit bindings for names such as `h1`, `logits`, `probs`, `loss`,
`predictions`, and `gradients`.
Also keep explicit bindings for values used more than once.

The intended result is a small number of meaningful bindings connected by pipes rather
than a flat sequence of mechanically named temporary values.

### Optional `let` in Block Bindings

Accept both forms in block statement position:

```chelis
x = expr
let x = expr
(a, b) = pair
let (a, b) = pair
```

Both forms desugar to the same Deep `(let {} (bind {} ...) ...)` structure.

Rules:

- short form is preferred for block-level sequential bindings
- explicit `let` remains accepted for backward compatibility and user preference
- `let ... in` expressions remain unchanged and still require `let`
- function-call left sides such as `f(x) = expr` are not valid binding patterns

### Width-Aware Pipe Formatting

Both `chelis surf` and `chelis fmt` should follow the same flat-first, break-if-over-
budget philosophy as the Deep pretty printer.

Width budget:

- 80 characters at the current indentation level

Rendering rules:

- keep a pipe chain on one line if the full binding/expression fits within 80 chars
- 2-3 short stages may stay flat
- break before every `|>` when the chain exceeds 80 chars
- break before every `|>` for chains with 4+ stages even if they technically fit
- once a chain is broken, break all stages consistently; do not keep multiple stages on
  the same continuation line
- continuation lines are indented 2 spaces from the binding

Binding rule:

- keep `name = ...` on one line only if the whole flat binding fits
- if the chain is broken and `name = first_stage` still fits for a short chain, keep the
  first stage on that first line
- otherwise break after `=`

Target layout:

```chelis
loss =
  softmax(logits, 1)
  |> log
  |> mul(labels)
  |> sum(1)
  |> neg
  |> mean(0)
```

Short layout:

```chelis
h1 = matmul(x, w1) |> relu
pred = matmul(x, w) |> add(expand(b, 0, batch))
```

`chelis fmt` responsibilities:

- reflow existing pipe chains according to width and indentation rules
- preserve whether the user wrote long-form or short-form block bindings

`chelis fmt` must not:

- invent pipes from let-chains
- remove pipes back into let-chains
- rewrite `let x = expr` into `x = expr`, or vice versa

### CLI Surface

`chelis surf` default vs verbose:

- default: pipe-first output, short-form block bindings, width-aware multiline layout
- `--verbose`: explicit `let`, no pipe-chain compression, and the more expanded debug
  rendering choices already associated with verbose mode

### Example / Doc Refresh

Rewrite the public Surf corpus to the finalized style:

- examples in `examples/`
- README hero/sample programs
- user-facing spec/tutorial examples where the sample is intended as idiomatic Surf
- supporting teaching material used by agents and users

Phase 3 public examples should consistently use:

- short-form block bindings
- pipe-first composition
- multiline pipes for long or many-stage chains
- symbolic runtime-varying dimensions such as `batch` and `seq`
- `->` return syntax

### Test Plan

- parser tests: `x = 5` and `let x = 5` parse identically in block position
- parser tests: tuple/destructuring short form works
- parser tests: `f(x) = 5` is not accepted as a binding pattern
- parser tests: `let ... in` still requires `let`
- decompiler tests: linear chains emit pipes
- decompiler tests: multi-use values do not pipe
- decompiler tests: meaningful names stay as bindings
- decompiler tests: default output uses short-form block bindings
- decompiler tests: verbose output restores explicit `let`
- formatter tests: flat chains stay flat under budget
- formatter tests: 4+ stage chains break at every `|>`
- formatter tests: over-budget chains break at every `|>`
- formatter tests: long names break after `=`
- formatter tests: once broken, each continuation line holds one stage
- validator tests: both binding forms are accepted
- round-trip tests: parse -> decompile -> parse preserves AST shape
- example corpus tests: rewritten examples still parse, check, and compile

### Acceptance Oracle

Authoritative oracle:

```sh
cargo test -p chelis-cli phase3e_pipe_first_acceptance_oracle -- --exact
```

Expected result:

- the test passes
- decompiled Surf uses short-form block bindings by default
- the `loss` chain is rendered pipe-first and broken at every `|>`
- no unnecessary intermediate body ascriptions are reintroduced

Supporting manual probe:

```sh
tmp=$(mktemp)
chelis deep examples/mnist.ch > "$tmp"
chelis surf "$tmp"
rm -f "$tmp"
```

---

## 3a: Package System (Shells + Reef)

**Goal:** Chelis libraries can be published, discovered, and consumed.
The first real package is the standard library itself.

### Core pieces

- `reef.toml` manifest
- `reef.lock` lockfile
- `.chb` shell artifact containing public metadata owned by `chelis-shell`
- local registry index and artifact publishing
- dependency resolution and import loading
- source archive consumption during downstream builds

### Dogfooding rule

`chelis-std` must ship as a Reef package using the package system itself.
If the package system cannot build, export, and re-import the standard library through
its own shell format, it is not done.

This makes `chelis-std` the real acceptance gate for:

- shell compilation
- `.chb` public type/effect metadata
- import resolution
- compiler-version compatibility handling
- bundled standard-library resolution
- bounded manifest discovery for package-aware `check` / `build`

### Standard Library Surface

The standard library is library code, not language magic.
Shipped `3a` dogfood surface:

- `Std.Nn`
- `Std.Init`
- `Std.Loss`
- `Std.IO`

All Phase 3 standard-library code should already use the finalized `3e` Surf style.

**Model serialization strategy (`Std.IO.Safetensors`):**

Safetensors is the explicit format choice for Chelis tensor serialization. No custom
format, no HDF5, no pickle.

Why safetensors: memory-mapped (fast loading, no deserialization overhead), stores tensor
metadata (shapes, dtypes, names) alongside data, universally supported (PyTorch, JAX,
HuggingFace), simple spec (JSON header + flat tensor data), and safe (no arbitrary code
execution unlike pickle).

In `3a`, `Std.IO.Safetensors` ships as a package/API stub only.
It proves that the package system can carry I/O-shaped modules and exported signatures
through `.chb`, import resolution, and shell consumption.
The real runtime implementation and cross-framework loading gate move to `3b`.

Surface in `Std.IO`:

```chelis
import Std.IO.Safetensors

-- Save trained parameters
save_tensors("checkpoint.safetensors", {
  "w1": w1, "b1": b1, "w2": w2, "b2": b2
})

-- Load parameters
params = load_tensors("checkpoint.safetensors")
```

Implementation split:

- `3a`: package stub with exported typed signatures only
- `3b`: runtime implementation via the host-callable boundary and safetensors library

Interop with Python (Phase 3b): a model trained in Chelis and saved as safetensors can
be loaded by PyTorch with `safetensors.torch.load_file("checkpoint.safetensors")`. A
model trained in PyTorch and saved as safetensors can be loaded by Chelis. No conversion
step — the format is the interop layer.

The full serialization story for a Chelis model in production:

| Artifact | Format | Purpose |
|---|---|---|
| Model definition | `.ch` (Surf source) | Human-readable, version-controlled |
| Compiled artifact | `.c` / `.hip` (generated code) | Compiled by gcc/hipcc, deployed |
| Trained weights | `.safetensors` | Parameter values, portable across frameworks |
| Coding model | `.gguf` (Phase 4) | The AI that wrote the model, ships with toolchain |

Formats NOT supported (and why):
- HDF5: legacy, being replaced by safetensors across the ML ecosystem
- NPZ: NumPy-specific, not framework-portable
- Pickle: security hazard (arbitrary code execution), no new system should support it
- ONNX/TorchScript: whole-model export formats (computation graph + weights), not
  weight-only serialization; relevant for Phase 5 FX backend export, not for
  Chelis-native checkpointing

### Test Plan

- manifest parse/write round-trip
- deterministic lockfile generation
- shell `.chb` round-trip for public metadata
- shell import/type-check consumption
- dependency-resolution success and conflict cases
- `chelis reef` CLI scaffolding and build flows
- bundled `chelis-std` build/import success
- bounded package-root discovery for `check` / `build`
- `module_prefix` enforcement on both module declaration and `src/` path shape
- safetensors shell import/type-check success from the `3a` stub

### Acceptance Oracle

Authoritative oracle:

```sh
cargo test -p chelis-cli phase3a_reef_std_acceptance_oracle -- --exact
```

Expected result:

- a temp `chelis-std` package builds and publishes into an isolated local Reef registry
- a temp consumer package resolves `chelis-std` by exact version
- `chelis check` and `chelis build` succeed on the consumer through the package system,
  not through ad hoc compiler special cases

Supporting manual probe:

```sh
chelis reef build packages/chelis-std
chelis reef publish packages/chelis-std
```

---

## 3b: Python FFI

**Goal:** Chelis fits into incremental Python-based adoption paths.

This has two explicit layers, aimed at different users:

### DLPack Layer

- zero-copy tensor exchange where framework/runtime constraints permit it
- PyTorch/JAX-style tensor handoff for embedding Chelis compute inside existing ML loops

### PyO3 Layer

- Python bindings for compiler access such as `chelis.check()` and related APIs
- tool-builder story for scripting the compiler from Python

### Product story

The intended adoption path is:

- Chelis for the model/compiler surface
- Python for the surrounding workflow, orchestration, and experimentation

### Test Plan

- DLPack tensor round-trip without silent copy where zero-copy is expected
- framework interop examples
- PyO3 compiler-call smoke tests
- Python bindings that mirror key CLI/compiler operations

### Acceptance Oracle

Manual:

```python
import chelis
```

and both of the following work:

- exchanging tensors with a Python ML framework through DLPack
- invoking compiler checks/build-facing APIs from Python

---

## 3c: Research Type Extensions

**Goal:** pursue research-grade extensions that are paper-worthy, practical, and
validated against real Chelis programs.

This track is paper-first, not code-first.
Each candidate feature follows this order:

1. write the paper draft
2. implement a prototype
3. validate it on the Chelis corpus
4. revise based on results
5. submit
6. merge into the mainline only after it proves out

**Tier 1 (high priority, core Phase 3 deliverables):**

1. ILP/AUTOMAP-style rank polymorphism
2. size-dependent types

**Tier 2 (medium priority, start in Phase 3, may extend into Phase 4+):**

3. distribution types
4. equivariance constraints

**Tier 3 (low priority, Phase 4+ or opportunistic):**

5. optimization-property annotations
6. **Inference as a typed effect** — LLM calls as a typed, mockable algebraic effect
   (`Inference.complete`). Relevant if Chelis programs ever delegate to LLMs
   (meta-learning, reward model queries, LLM-as-judge). Composes with existing effects —
   `effects(<Random, Inference>)` means the function uses both randomness and LLM calls.
   Inspired by Vera's `Inference` effect. Phase 3+ research direction, not a v1 feature.

### Research Notes

**De Bruijn references in Deep (study, don't adopt).** Vera's elimination of variable
names in favor of typed positional indices (`@T.n`) achieves 100% LLM generation
accuracy on its benchmark. The question for Chelis: would replacing `(var {} x)` with
positional references `(ref {} 0)` in Deep improve LLM generation accuracy for the
s-expression representation? The hypothesis: Deep's closed 60-tag vocabulary and 3-tuple
structure may already provide enough structural constraint that de Bruijn indices add
complexity without measurable benefit. Additionally, named references carry semantic
meaning for ML code (`logits` vs `hidden`) that positional indices destroy. Study this
as an empirical question during Phase 4b (ICL effect measurement): test generation
accuracy with named-Deep vs positional-Deep variants of the SKILL.md and compare. Don't
adopt unless the measurement shows a clear advantage.

### Test / Validation Plan

- corpus validation against real examples rather than toy-only proofs
- negative cases showing explicit failure modes
- paper-quality writeup of tradeoffs and limitations

### Acceptance Oracle

A feature is not complete because a prototype exists.
It is complete when the prototype is validated on the corpus and the paper submission
artifact exists.

---

## 3d: Lean Formalization

**Goal:** create a mechanized reference for the core Chelis type system and use it as
the ultimate conformance oracle.

This is an explicit parallel research track.
It does not block package work, Python interop, or public-style cleanup.

Target outcomes:

- a POPL/ICFP-style paper
- an executable reference type checker
- a conformance-oracle rule: if Lean and Rust disagree on whether a program type-checks,
  Lean is right

### Test / Validation Plan

- shared corpus checked by both Rust and Lean implementations
- disagreement cases captured as bugs against the Rust checker unless Lean is shown wrong
- explicit documentation of the supported formalized core versus out-of-scope language
  features

### Acceptance Oracle

The Lean checker runs on the agreed core corpus and serves as the authoritative
reference for conformance disputes in that subset.

---

## 3f: SKILL.md v2 Refresh

**Goal:** refresh `SKILL.md` for the complete stable Phase 2 language surface and the
finalized Phase 3 Surf idiom.

This stays in Phase 3 because the skill file is part of the public face of the
language, not merely a Phase 4 training input.

The updated skill should cover:

- effects
- linearity
- macros
- `vmap`
- tuples
- short-form block bindings
- pipe-first Surf
- multiline pipe formatting for long chains

All Surf examples in `SKILL.md` should align with the `3e` style foundation.
Deep examples remain canonical Deep.

### Test Plan

- update worked examples to the finalized idiom
- ensure guidance mentions short-form block bindings while keeping `let` documented as
  valid
- re-run the skill validation suite against the current compiler

### Acceptance Oracle

The checked-in skill validation suite passes against the compiler with the refreshed
examples and guidance.

---

## Phase 3 Completion Check

Before calling Phase 3 complete:

- `3e` style rules are documented and reflected across examples/teaching material
- `3a` package system dogfoods `chelis-std`
- `3b` covers both DLPack and PyO3
- `3c` has at least one paper-quality, corpus-validated extension in flight
- `3d` has a running Lean checker on the agreed core subset
- `3f` SKILL.md v2 matches the final public Surf idiom
