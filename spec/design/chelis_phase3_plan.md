# Phase 3: Language Completeness - Expanded Implementation Plan

## Context

Phase 2 shipped a feature-rich tensor computation language: effects, linearity, macros,
`vmap`, tuples, Tide APIs, LSP, and Cove. Phase `3a` shipped the package system.
Phases `3b` and `3b-ii` shipped Python interop, safetensors, and direct execution.
Phase `3e` shipped the pipe-first public Surf idiom.

But Chelis still depends on Python for every non-tensor part of a real AI program. It
can define a transformer forward pass and compute gradients, but it still cannot:

- represent first-class scalar integers/floats/bools outside tensors
- process strings as ordinary language values
- store variable-length sequences and maps
- iterate over non-tensor data
- read text/config/data files directly
- tokenize text and batch it into model inputs

That means Chelis is still a sophisticated tensor-compute DSL rather than a
self-sufficient AI programming language.

**Phase 3 deliverable:** a researcher can write, in pure Chelis, a program that reads
text, tokenizes it, batches and pads it, runs a model, computes loss and gradients, and
prints results, without dropping to Python for preprocessing or orchestration.

**Phase 3 does NOT deliver:** sparse tensors, complex numbers, research type
extensions, or Lean formalization. Those move to Phase `5e`, `5f`, `5g`, and `5h`.
Phase 3 is the pragmatic language-completeness phase.

---

## Dependency Graph

```text
Shipped foundations

3e: Pipe-First Style Pass
3a: Package System
3b: Python FFI
3b-ii: Direct Python Execution + NumPy
3c: Scalar & String Foundation
3d: Collections & Iteration

Remaining work

3h: Core Numeric Primitives -> 3g: Data Loading & Tokenization -> 3i: Std.Time & Std.Decimal -> 3f: SKILL.md v2 Redo
```

**Recommended execution order:**

1. `3h`: Core numeric primitives
2. `3g`: Data loading and tokenization
3. `3i`: `Std.Time` and `Std.Decimal`
4. `3f`: SKILL.md v2 redo

Shipped Phase 3 foundations stay in place and continue to constrain the remaining work:

- `3e` defines the public Surf idiom new examples must follow
- `3a` defines the package/distribution story new libraries should use
- `3b` / `3b-ii` define the Python interop boundary the fuller language must still fit

`3h` follows the shipped `3c`/`3d` foundations because real AI model code still needs
practical tensor-language primitives such as `einsum`, `concat`, `gather`, and
`clamp`. `3g` depends on that fuller host-and-tensor surface because tokenization,
batching, and model ingress should not force awkward library workarounds. `3i` comes
after `3g` because time/exact-decimal support rounds out the standard library rather
than blocking the pure-AI workflow milestone. `3f` goes last because the teaching
surface should describe the real full Phase 3 language, not a partially complete
midpoint.

---

## Shipped Foundations

### 3e: Pipe-First Style Pass

**Status:** shipped.

This remains the public style foundation for all remaining Phase 3 work:

- pipe-first decompiler output
- short-form block bindings
- width-aware multiline pipe layout
- examples and docs that read like human-written Surf rather than typed Deep debug text

All new examples introduced in `3h`, `3g`, `3i`, and `3f` should continue to follow
this style.

### 3a: Package System (Shells + Reef)

**Status:** shipped.

The package system remains in scope as infrastructure, not as remaining work. New
Phase 3 library surfaces such as tokenizer helpers or text/data modules should be
documented as package-friendly APIs that fit the existing Reef / `.chb` model.

### 3b: Python FFI

**Status:** shipped.

Python interop remains a supporting boundary, not the solution to Chelis's remaining
language gaps. Phase 3 is successful only when preprocessing and tokenization no
longer require Python for ordinary use.

### 3b-ii: Direct Python Execution + NumPy Guarantee

**Status:** shipped.

Direct execution remains the bridge for embedding Chelis in Python workflows, but the
remaining Phase 3 work is about making that embedding optional for end-to-end AI
program authoring.

---

## 3c: Scalar and String Foundation

**Status:** shipped.

**Goal:** make Chelis a real programming language for AI workflows by adding first-class
scalar values and strings outside the tensor-only world.

### Why This Matters

Without first-class scalars and strings, Chelis cannot naturally express:

- loop counters and vocabulary indices
- loss-threshold checks and training decisions
- file paths and config keys
- labels, tokens, and log messages
- tokenizer state and text-derived metadata

### Required Surface

**Scalar types as ordinary values:**

- unrestricted `Int`, `Float`, and `Bool`
- arithmetic, comparison, `%` / `mod`, and basic integer bitwise helpers such as
  `bitand`, `bitor`, `bitxor`, `shl`, and `shr`
- scalar values are distinct from rank-0 tensors
- explicit scalar/tensor conversions where needed

**String values as ordinary values:**

- immutable UTF-8 `String`
- length, concat, slice, contains, starts/ends-with, trim
- string length is defined in characters, not bytes
- parse/format helpers such as `to_int`, `to_float`, and `to_string`
- string-producing/logging use cases must work without Python

**Control and observability:**

- scalar `if cond then a else b`
- `print(x)` and `debug(x)` as minimal IO-backed debugging tools with explicit `IO`
  effect tracking
- tensor shape queries such as `shape`, `rank`, and `numel`
- `Option[T]` promoted as the practical failure-returning surface for parse/lookups

### Implementation Shape

- extend Surf/Deep/type docs and implementation for scalar/string literals and scalar
  conditionals
- extend type inference/checking for non-tensor scalar and string operations
- extend the evaluator for scalar/string execution
- lower scalar/string/`Option`/print logic through a compiled host-value lane so
  `chelis build` does not fall back to evaluator-only behavior for mixed programs
- keep that host-value lane on CPU for both C and HIP targets; tensor kernels still use
  the existing tensor DAG/device paths
- use tagged execution values on the Tide/Python wire surface
- keep tensor computation semantics unchanged: no implicit scalar/tensor blending

### Acceptance Oracle

A pure Chelis training-step-style program can:

- compute scalar stopping criteria
- build or format a checkpoint/log path as a string
- print progress without Python
- compile with `chelis build --target c`, run as a native binary, and match `chelis eval`

---

## 3d: Collections and Iteration

**Status:** shipped.

**Goal:** add the variable-length data structures and functional iteration primitives
required for preprocessing and dataset plumbing.

### Why This Matters

Real AI programs need to represent:

- lists of token ids
- lists of sentences with different lengths
- dictionaries for vocabularies and configs
- dataset rows and intermediate preprocessing results

Tensors alone cannot express that variable-length host-side structure.

### Required Surface

**Collections:**

- immutable `List[T]`
- immutable `Dict[K, V]` with practical key types such as `String` and `Int`

**Current executable slice:**

- shipped compiled collection slice: list literals, `List[T]`, `len`, `index`,
  `append`, `concat`, `take`, `drop`, `chunk`, `flatten`, `range`, `zip`,
  `enumerate`, numeric `to_tensor`, practical rank-1 `to_list`, `pad_sequences`, and
  the first immutable `Dict[K, V]` builtins: `dict_of`, `dict_get`,
  `dict_contains`, `dict_remove`, `dict_insert`, `dict_merge`, `dict_keys`,
  `dict_values`, `dict_entries`, plus compiled higher-order iteration (`map`,
  `filter`, `fold`, `scan`, `partition`, `flat_map`)
- this now covers the practical compiled collection surface needed to hand off cleanly
  to `3g` tokenization/data-loading work rather than leaving obvious batching or
  nested-list gaps behind

**Iteration primitives:**

- `map`, `filter`, `fold`, `scan`, `partition`, `flat_map`, `zip`, `enumerate`,
  `range`
- collection length (lists and dicts), indexing, append/concat, sequence truncation
  (`take`, `drop`), batching (`chunk`), nested-list flattening, key lookup, immutable
  dict remove/update/overlay, key/value/entry enumeration, and dataset-friendly
  helpers such as cumulative scans, stable boolean partitioning, and callback-driven
  list expansion
- effect propagation through iteration
  this is now covered by checker tests for callback-driven `IO` and `Random`

**Collection/tensor bridge:**

- list-to-tensor conversion for numeric lists
- practical rank-1 tensor-to-list conversion
- stacking and padding helpers
- `pad_sequences` as the critical bridge from variable-length token lists to batched
  tensor inputs

### Implementation Shape

- add collection types and type inference rules
- add collection literals and built-in iteration APIs
- add evaluator/runtime support for immutable collections
- document linearly typed tensor elements inside collections without making collections
  themselves linear by default
- compile collections through the same host-value lane used by `3c` in both C and HIP
  builds; collections stay host-side as CPU data structures, but they are not
  evaluator-only
- keep tensors as the fixed-shape compute substrate and make the list/tensor bridge
  explicit

### Acceptance Oracle

A pure Chelis preprocessing program can:

- build a vocabulary/config map
- transform a list of examples with functional iteration
- truncate a variable-length sequence before tensorization
- immutably extend or overlay a config/vocabulary dictionary
- pad variable-length integer sequences into a batched tensor input

Current shipped oracles for the executable collection slices:

- `examples/list_foundation.ch` survives `chelis fmt`, `chelis check`, `chelis eval`,
  and `chelis build --target c`, and the compiled binary matches eval output
- `examples/dict_foundation.ch` survives `chelis fmt`, `chelis check`, `chelis eval`,
  and `chelis build --target c`, and the compiled binary matches eval output
- `examples/iter_foundation.ch` survives `chelis fmt`, `chelis check`, `chelis eval`,
  and `chelis build --target c`, and the compiled binary matches eval output

---

## 3h: Core Numeric Primitives

**Goal:** close the most important tensor-language expressiveness gaps between the
initial RISC + derived surface and the operations real models and numerical libraries
expect to use directly.

### Why This Matters

The shipped scalar/string/collection/data foundations make Chelis a host language, but
serious AI/model code still needs a richer tensor core than the original minimal set.
Without this cut, users still hit avoidable walls around:

- Einstein-style contractions and batched linear algebra
- skip connections and multi-head packing/unpacking flows
- embedding lookup and index-driven tensor access
- masked/conditional tensor updates
- cumulative and order-sensitive tensor statistics
- diagonal/trace style linear-algebra utilities
- basic value clipping and training-control helpers

### Required Surface

**Core primitives added in this phase:**

- `einsum`
- `concat` / `split`
- `gather` / `scatter`
- `where`
- `cumsum`
- `sort`
- `diagonal` / `trace`
- `clamp`

**Priority notes:**

- `einsum` is the single highest-impact addition because it covers matmul, batched
  matmul, transpose, trace, outer products, and common contraction patterns in one
  primitive
- `concat` / `split` are the minimal structural operations needed for modern
  transformer-style model blocks
- `diagonal` / `trace` and `clamp` are small but broadly useful numerical surfaces

**Standard-library addition unlocked here:**

- `Std.Nn.Embedding` as the explicit named surface over `gather`

### Implementation Shape

- extend the type/checking/lowering/backend docs and implementation for the new tensor
  primitives
- keep `Std.Nn.Embedding` in the standard library rather than the compiler core, but
  make the underlying `gather` primitive part of this phase
- preserve the Phase 0/1 invariants: explicit shapes, explicit broadcasting, and
  backend agreement with the reference C path
- reject deterministic literal-driven `3h` value errors during `chelis check` when the
  offending extents/indices are statically concrete; runtime-only bad values still fail
  during execution, but compiled C exits non-zero rather than aborting

### Acceptance Oracle

A pure Chelis model program can:

- express common contraction-heavy blocks using `einsum`
- concatenate and split tensor features without falling back to Python
- perform embedding lookup through `Std.Nn.Embedding`
- clamp or trace intermediate tensors in compiled programs on the C path

Current shipped oracle for the executable `3h` slice:

- `cargo test -p chelis-cli phase3h_numeric_acceptance_oracle -- --nocapture`
  which proves both the compiled tensor-structural example path
  (`examples/tensor_structural_ops.ch`, including `einsum`) and the Reef package path
  (`Std.Nn.Embedding` imported from `chelis-std`) build to valid C artifacts

---

## 3g: Data Loading and Tokenization

**Goal:** make Chelis self-sufficient for the preprocessing path every serious AI
program needs.

### Why This Matters

Without file I/O and tokenization, Chelis can only consume already-prepared tensors.
That leaves the most basic LLM/data workflow steps in Python:

- read text
- parse configs/data
- tokenize text
- batch and pad tokens

Phase 3 is not complete until those steps are expressible in pure Chelis.

### Required Surface

**Data loading:**

- text file I/O: read/write text files
- CSV loading for small/medium structured datasets
- JSON loading for configs, vocabularies, and metadata

**Tokenization:**

- a BPE tokenizer as the minimum practical tokenizer surface
- ability to load an external vocabulary/merge artifact format used in real workflows
- encode/decode text to and from integer token sequences

**Batching bridge:**

- batch encode helpers
- padding to fixed sequence length
- clean handoff from `List[List[Int]]` to model-ready tensors

### Implementation Shape

- model file I/O as explicit IO-effect operations
- document tokenizer/state formats and library placement without inventing unnecessary
  registry or remote-service machinery
- keep the first tokenizer target pragmatic: enough to ingest ordinary HuggingFace-style
  BPE assets rather than inventing a novel Chelis-native tokenizer format

### Acceptance Oracle

A pure Chelis program can:

- read a text file
- tokenize its contents into integer sequences
- batch and pad those sequences into tensors
- feed them into a model without Python preprocessing

---

## 3i: `Std.Time` and `Std.Decimal`

**Goal:** round out the standard-library host-language surface with time and
exact-arithmetic utilities that real workflows need but the compiler core should not own.

### Why This Matters

Pure Chelis workflows still need ordinary application scaffolding around the model:

- dates and durations in schedules, checkpoints, and reporting
- exact decimal arithmetic for money/config/reporting cases where binary floats are the
  wrong surface

These do not justify new compiler intrinsics, but they do belong in the Phase 3
language-completeness story rather than an indefinite backlog.

### Required Surface

- `Std.Time` for dates, timestamps, durations, and basic time arithmetic
- `Std.Decimal` for exact decimal values and arithmetic

### Implementation Shape

- ship both as standard-library modules, not new core-language primitives
- keep the APIs package-friendly under the existing Reef / `chelis-std` model
- sequence this after `3g` so data/tokenization remains the critical practical
  milestone and before `3f` so the teaching surface can cover the complete host-side
  standard stack

### Acceptance Oracle

A pure Chelis workflow can:

- represent and format a timestamp or duration without Python
- represent exact decimal configuration/reporting values without binary-float drift
- use those values in package-friendly standard-library code

---

## 3f: SKILL.md v2 Redo

**Goal:** rewrite the teaching surface after the language-completeness work lands so
that frontier-model prompting and future training both target the real usable language.

### Why This Must Move Last

The earlier SKILL refresh is no longer enough. If `SKILL.md` is refreshed before
scalar/string/collection/tokenization support lands, it will encode the tensor-only
subset and then need another rewrite.

### Required Surface

The refreshed skill should teach:

- the shipped pipe-first Surf idiom from `3e`
- effects, linearity, macros, `vmap`, and tuples
- scalar/string programming
- collections and iteration
- core numeric primitives such as `einsum`, `gather`, and `concat`
- tokenization and data-loading idioms
- `Std.Time` / `Std.Decimal` host-program idioms
- the boundary between host-side preprocessing and tensor compute inside Chelis itself

### Acceptance Oracle

The checked-in skill validation suite passes with examples and guidance that reflect the
post-`3i` language surface.

---

## Post-Phase-3 Shell Stack

The ecosystem layering this phase is setting up is:

- `chelis-std` as the first package-layer target, including `Std.Nn.Embedding` and the
  later `Std.Time` / `Std.Decimal` host-program surface
- `school` on top of `chelis-std` for general numerical methods: stats,
  distributions, optimization solvers including differentiable optimization,
  interpolation, SVD/PCA, ODE/SDE solvers, integration, root finding, and later signal
  processing once complex numbers land
- `coral` on top of `chelis-std` as the typed dataframe shell in the reef/shells/tide/
  cove/school/coral/treasure marine lineup: GPU-accelerated numeric columns, host-side
  string columns, and AD through dataframe operations where `filter` lowers to gather
  and aggregation lowers to reduction; positioned for pandas/Polars-style tabular work
  with typed correctness and differentiable composition
- `treasure` on top of `chelis-std` + `school`, with optional `coral` integration for
  finance-specific pricing, risk, curves, stochastic processes, and order-book
  workloads

---

## Phase 3 Red-Team Checkpoint

Before calling Phase 3 healthy enough to continue, red-team these concrete surfaces:

**Scalar/string foundation (`3c`):**

- scalar arithmetic and conditionals work without silently becoming tensor operations
- strings are usable for paths, labels, and logs
- `print` / `debug` are honest about their IO/effect behavior

**Collections/iteration (`3d`):**

- collection APIs handle variable-length data without hidden mutation
- effect propagation through `map` / `fold` is correct
- list/tensor bridging rejects malformed shape cases clearly

**Core numeric primitives (`3h`):**

- `einsum` agrees with the reference backend on the supported contraction corpus
- `concat` / `split`, `diagonal` / `trace`, and `clamp` behave consistently across
  evaluator and compiled paths
- `Std.Nn.Embedding` exercises the real `gather` path rather than a fake host-side stub

**Data loading/tokenization (`3g`):**

- text/CSV/JSON loading returns the documented structures
- tokenizer encode/decode is deterministic against the documented assets
- batching/padding produces the expected tensor shapes and values

**Standard library host types (`3i`):**

- `Std.Time` and `Std.Decimal` stay standard-library scoped rather than leaking
  compiler-intrinsic assumptions
- examples/docs do not overclaim backend or tensor-kernel relevance for these modules

**Teaching surface (`3f`):**

- `SKILL.md` teaches the real executable language, not a stale tensor-only subset
- examples align with the package/style/python foundations already shipped

---

## Phase 3 Estimated Timeline

| Sub-phase | Effort | Dependencies | Nature |
|---|---|---|---|
| `3e`: Pipe-first style pass | shipped | none | Engineering |
| `3a`: Package system | shipped | `3e` | Engineering |
| `3b`: Python FFI interop core | shipped | `3a` | Engineering |
| `3b-ii`: Direct execution + NumPy | shipped | `3b` | Engineering |
| `3c`: Scalar and string foundation | shipped | `3e` shipped | Engineering |
| `3d`: Collections and iteration | shipped | `3c` | Engineering |
| `3h`: Core numeric primitives | ~4-6 weeks | `3d` | Engineering |
| `3g`: Data loading and tokenization | ~4-6 weeks | `3h` | Engineering |
| `3i`: `Std.Time` and `Std.Decimal` | ~2-3 weeks | `3g` | Engineering / standard library |
| `3f`: SKILL.md v2 redo | ~2-3 weeks | `3h`, `3g`, `3i` | Engineering / teaching surface |

This phase is now intentionally sequential and pragmatic. The remaining work is about
making Chelis usable, not publishable. Remaining estimated effort is roughly
`12-17 weeks`, and the expanded Phase 3 stack now totals roughly `25-35 weeks` end to
end across shipped and planned sub-phases.

---

## Phase 3 Completion Check

Before calling Phase 3 complete:

- `3a`, `3b`, `3b-ii`, and `3e` remain honest shipped foundations
- `3c` provides practical scalar/string programming without Python fallback
- `3d` provides collections and iteration for variable-length host-side data
- `3h` provides the expanded tensor-language surface needed for real model code
- `3g` provides text/config/data loading plus tokenizer and batching support
- `3i` provides `Std.Time` and `Std.Decimal` as practical standard-library host types
- `3f` reflects the real post-`3i` language in `SKILL.md` and examples
- a pure Chelis program can read text, tokenize it, batch/pad it, run a model, compute
  loss and gradients, and print results without Python
