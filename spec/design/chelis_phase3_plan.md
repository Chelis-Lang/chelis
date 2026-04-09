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

This phase has two explicit tracks across five numbered sub-phases:

- engineering track: `3e -> 3a -> 3b`
- research track: `3c` and `3d` in parallel

`SKILL.md` v2 is the public-teaching/documentation refresh that lands after the style
foundation is settled and before the Phase 4 model-training track begins.
It stays near `3e`, but it is not counted as a sixth numbered sub-phase.

---

## Dependency Graph

```text
Engineering track

3e: Style Foundation -----------------------------------------------+
  (pipes, optional block let, width-aware formatting, examples)     |
         |                                                          |
         v                                                          |
3a: Package System (Shells + Reef) ---------------------------------+--> 3b: Python FFI
  (dogfood with chelis-std)                                              (DLPack + PyO3)
         |
         +---------------------------------------------> SKILL.md v2 refresh
                                                        (after 3e, before 4a)

Research track

3c: Research Type Extensions (paper-first, corpus-validated)
3d: Lean Formalization (independent conformance-oracle track)
```

**Recommended execution order:**

1. `3e` style foundation
2. `3a` package system
3. `3b` Python FFI
4. `3d` Lean formalization in parallel
5. `3c` research type extensions
6. `SKILL.md` v2 refresh after the idiom is stable and before Phase 4 corpus/training work

`3e` goes first because every later deliverable publishes examples, package code, or
teaching material that should already be in the final public idiom.

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
- `.chb` shell artifact containing public typed metadata
- registry index and artifact publishing
- dependency resolution and import loading

### Dogfooding rule

`chelis-std` must ship as a Reef package using the package system itself.
If the package system cannot build, export, and re-import the standard library through
its own shell format, it is not done.

This makes `chelis-std` the real acceptance gate for:

- shell compilation
- `.chb` public type/effect/linearity metadata
- import resolution
- compiler-version compatibility handling
- bundled standard-library resolution

### Standard Library Surface

The standard library is library code, not language magic.
Representative module areas:

- `Std.Nn`
- `Std.Optim`
- `Std.Init`
- `Std.Loss`
- `Std.Metrics`
- `Std.IO`
- `Std.Schedule`

All Phase 3 standard-library code should already use the finalized `3e` Surf style.

### Test Plan

- manifest parse/write round-trip
- deterministic lockfile generation
- shell `.chb` round-trip for public metadata
- shell import/type-check consumption
- dependency-resolution success and conflict cases
- `chelis reef` CLI scaffolding and build flows
- bundled `chelis-std` build/import success

### Acceptance Oracle

Manual:

```sh
chelis reef build chelis-std
chelis check examples/using_std.ch
```

Expected result: the standard library is built as a Reef package and consumed through
the package system, not through ad hoc compiler special cases.

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

Priority order:

1. ILP/AUTOMAP-style rank polymorphism
2. size-dependent types
3. distribution types
4. equivariance constraints

The first two are the main Phase 3 focus.
The latter two remain more speculative.

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

## SKILL.md v2 Refresh

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
- `SKILL.md` v2 matches the final public Surf idiom
