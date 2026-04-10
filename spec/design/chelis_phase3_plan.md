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

**Phase 3 does NOT deliver:** research type extensions or Lean formalization. Those move
to Phase `5e` and `5f`. Phase 3 is the pragmatic language-completeness phase.

---

## Dependency Graph

```text
Shipped foundations

3e: Pipe-First Style Pass
3a: Package System
3b: Python FFI
3b-ii: Direct Python Execution + NumPy

Remaining work

3c: Scalar & String Foundation -> 3d: Collections & Iteration -> 3g: Data Loading & Tokenization -> 3f: SKILL.md v2 Redo
```

**Recommended execution order:**

1. `3c`: Scalar and string foundation
2. `3d`: Collections and iteration
3. `3g`: Data loading and tokenization
4. `3f`: SKILL.md v2 redo

Shipped Phase 3 foundations stay in place and continue to constrain the remaining work:

- `3e` defines the public Surf idiom new examples must follow
- `3a` defines the package/distribution story new libraries should use
- `3b` / `3b-ii` define the Python interop boundary the fuller language must still fit

`3c` must precede `3d` because collections need scalar and string element types.
`3d` must precede `3g` because tokenization and file/config processing produce lists,
dicts, and variable-length sequences. `3f` goes last because the teaching surface
should describe the real full Phase 3 language, not a partially complete midpoint.

---

## Shipped Foundations

### 3e: Pipe-First Style Pass

**Status:** shipped.

This remains the public style foundation for all remaining Phase 3 work:

- pipe-first decompiler output
- short-form block bindings
- width-aware multiline pipe layout
- examples and docs that read like human-written Surf rather than typed Deep debug text

All new examples introduced in `3c`, `3d`, `3g`, and `3f` should continue to follow
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
- extend C/HIP codegen only where these values must survive through executable programs;
  host-side runtime support is acceptable where GPU execution is not the point
- keep tensor computation semantics unchanged: no implicit scalar/tensor blending

### Acceptance Oracle

A pure Chelis training-step-style program can:

- compute scalar stopping criteria
- build or format a checkpoint/log path as a string
- print progress without Python

---

## 3d: Collections and Iteration

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

**Iteration primitives:**

- `map`, `filter`, `fold`, `zip`, `enumerate`, `range`
- collection length, indexing, append/concat, key lookup, key/value/entry enumeration
- effect propagation through iteration

**Collection/tensor bridge:**

- list-to-tensor conversion for numeric lists
- tensor-to-list conversion where practical
- stacking and padding helpers
- `pad_sequences` as the critical bridge from variable-length token lists to batched
  tensor inputs

### Implementation Shape

- add collection types and type inference rules
- add collection literals and built-in iteration APIs
- add evaluator/runtime support for immutable collections
- document linearly typed tensor elements inside collections without making collections
  themselves linear by default
- keep collection data host-side; tensors remain the compiled compute substrate

### Acceptance Oracle

A pure Chelis preprocessing program can:

- build a vocabulary/config map
- transform a list of examples with functional iteration
- pad variable-length integer sequences into a batched tensor input

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
- tokenization and data-loading idioms
- the boundary between host-side preprocessing and tensor compute inside Chelis itself

### Acceptance Oracle

The checked-in skill validation suite passes with examples and guidance that reflect the
post-`3g` language surface.

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

**Data loading/tokenization (`3g`):**

- text/CSV/JSON loading returns the documented structures
- tokenizer encode/decode is deterministic against the documented assets
- batching/padding produces the expected tensor shapes and values

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
| `3c`: Scalar and string foundation | ~4-6 weeks | `3e` shipped | Engineering |
| `3d`: Collections and iteration | ~4-6 weeks | `3c` | Engineering |
| `3g`: Data loading and tokenization | ~4-6 weeks | `3d` | Engineering |
| `3f`: SKILL.md v2 redo | ~2-3 weeks | `3c`, `3d`, `3g` | Engineering / teaching surface |

This phase is now intentionally sequential and pragmatic. The remaining work is about
making Chelis usable, not publishable.

---

## Phase 3 Completion Check

Before calling Phase 3 complete:

- `3a`, `3b`, `3b-ii`, and `3e` remain honest shipped foundations
- `3c` provides practical scalar/string programming without Python fallback
- `3d` provides collections and iteration for variable-length host-side data
- `3g` provides text/config/data loading plus tokenizer and batching support
- `3f` reflects the real post-`3g` language in `SKILL.md` and examples
- a pure Chelis program can read text, tokenize it, batch/pad it, run a model, compute
  loss and gradients, and print results without Python
