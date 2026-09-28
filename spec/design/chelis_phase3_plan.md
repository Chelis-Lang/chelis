# Phase 3: Language Completeness - Current Contract and Remaining Plan

## Context

Phase 2 shipped a feature-rich tensor computation language: effects, linearity, macros,
`vmap`, tuples, Tide APIs, LSP, and Cove. Phase `3a` shipped the package system.
Phases `3b` and `3b-ii` shipped Python interop, safetensors, and direct execution.
Phase `3e` shipped the pipe-first public Surf idiom.

At the start of Phase 3, Chelis still depended on Python for every non-tensor part of a
real AI program. It could define a transformer forward pass and compute gradients, but
it still could not:

- represent first-class scalar integers/floats/bools outside tensors
- process strings as ordinary language values
- store variable-length sequences and maps
- iterate over non-tensor data
- read text/config/data files directly
- tokenize text and batch it into model inputs

Those gaps drove the compiler/runtime and standard-library foundations that have since
shipped through `3j`. The remaining Phase 3 work is now the shell ecosystem, native
testing surface, and final teaching refresh.

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
3h: Core Numeric Primitives
3m: Rust Runtime Rewrite
3g: Data Loading & Tokenization
3i: Standard Library Expansion
3j-pre: Release Infra + Std Surface Expansion
3j: Nautilus

Remaining work

3k: Coral
  -> 3l: Shoals
3n: Octant (Part A)
  -> 3o: Octant (Part B)               (also depends on 3l)
3t: Chelis-Native Testing              (depends on fast package-aware eval)
  -> 3f: SKILL.md v2 Redo
```

`3j` (nautilus) has shipped in the downstream shell repo and is now a dependency for
the remaining shell work. `3k` (coral) is the next unshipped domain shell. `3l`
(shoals) depends on both `3j` and `3k`. `3n` (octant Part A) can proceed against
`3j`; `3o` (octant Part B) is sequential after both `3l` and `3n` because its
finance-notation lowering paths require `Shoals.Stochastic`, `Shoals.Pricing`, and
`Shoals.Curves`. `3t` is remaining infrastructure for Chelis-native shell tests.

**Recommended execution order:**

1. `3k`: Coral (typed dataframes)
2. `3l` and `3n`: Shoals (finance) and Octant Part A (LaTeX ↔ Deep parser,
   deterministic lowering, render, provenance), parallel where staffing allows
3. `3o`: Octant Part B (finance-notation lowering through `shoals`, Greek rendering,
   notebook), depends on both `3l` and `3n`
4. `3t`: Chelis-native testing and package test migrations
5. `3f`: SKILL.md v2 redo

Shipped Phase 3 foundations stay in place and continue to constrain the remaining work:

- `3e` defines the public Surf idiom new examples must follow
- `3a` defines the package/distribution story new libraries should use
- `3b` / `3b-ii` define the Python interop boundary the fuller language must still fit
- `3c` / `3d` define the host-value and collection lane
- `3h`, `3m`, `3g`, `3i`, `3j-pre`, and `3j` define the shipped compiler/runtime,
  standard-library, release, and Nautilus surfaces that Coral/Shoals/Octant must build
  against

The earlier dependency chain (`3h` → `3m` → `3g` → `3i` → `3j-pre` → `3j`) is now
historical context, not remaining work. `3f` still goes last because the teaching
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
- effect propagation through iteration, including callback-driven `IO`

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

**Status:** shipped.

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

- `Std.Nn.Embedding` as the explicit named surface over `gather` (the `Std.Nn.Embedding`
  shell module references in this 3h section since moved to `School.Nn.Embedding` in
  chelis-std 0.4.0; the `gather` primitive itself stayed in the compiler core)

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

## 3m: Rust Runtime Rewrite

**Status:** shipped.

**Goal:** replace the growing C runtime implementation with a Rust static library and
clean up the host-value ABI before more runtime-heavy language/library work lands.

### Why This Matters

Phases `3c` and `3d` made the compiled host-value lane real: mixed programs now compile
scalars, strings, lists, dicts, tuples, `Option`, and print/debug through the generated
host program path. The old `chelis_runtime.c` implementation is now the wrong substrate
for what comes next:

- `3g` adds file I/O, CSV/JSON, tokenizer loading, and batching helpers
- `3i` adds time/date and exact-decimal runtime support
- later shells depend on a stable host runtime rather than ad hoc C helpers

If Chelis keeps expanding the C runtime, it takes on more manual memory management and
more C-side ownership risk exactly where the language is getting broader.

### Required Surface

**Runtime packaging and ABI:**

- new crate: `crates/chelis-runtime`
- `libchelis_runtime.a` becomes the shipped runtime artifact
- `chelis_runtime.h` remains the C ABI contract but is now owned by the runtime crate

**ABI cleanup decision:**

- The shipped Phase 3 runtime kept `chelis_tensor` layout-visible because generated tensor code
  dereferences tensor fields directly
- chelis#1286 supersedes that historical choice: the next ownership cut makes the tensor
  opaque and moves generated code to tagged read/unique-write access plus a shared
  unique-storage proof; see `compiled_value_ownership.md`
- the chelis#1286 target keeps `chelis_string`, `chelis_list`, `chelis_tuple`,
  `chelis_dict`, `chelis_adt`, `chelis_option`, and `chelis_mapped_file` behind
  opaque handles and deletes the by-value option-carrier split
- generated host code must use runtime accessors and retain/release APIs rather than
  peeking into `.data`, `->len`, `->items`, or `->entries`

**Build surface:**

- `chelis build` emits generated source/header plus `chelis_runtime.h` and
  `libchelis_runtime.a`
- `chelis_runtime.c` stops being an emitted build artifact
- `chelis build` stages the runtime the CLI carries and rejects a set
  `CHELIS_RUNTIME_DIR` (`spec/08-backends.md` §2.1; chelis#1354 replaced the
  original directory discovery order)

### Implementation Shape

- implement the runtime in Rust modules (`tensor`, `blas`, `string`, `list`, `tuple`,
  `dict`, `value`, `print`, `io`) behind a C ABI
- preserve current language-level behavior: same string character-count semantics, same
  insertion-order dict behavior, same compiled-vs-eval outputs
- update host C emission to use accessors and explicit ownership operations for host
  temporaries, branches, loops, container operations, and returns
- keep tensor kernels and evaluator semantics unchanged; this is a runtime contract
  rewrite, not a language-semantics phase

### Acceptance Oracle

A mixed tensor + host-value Chelis program can:

- build through `chelis build --target c`
- emit `chelis_runtime.h` and `libchelis_runtime.a` but not `chelis_runtime.c`
- compile and link against the staged `libchelis_runtime.a`
- run as a native binary and match `chelis eval`

Authoritative oracle:

- `cargo test -p chelis-cli phase3m_rust_runtime_acceptance_oracle -- --nocapture`

Manual HIP mirror gate:

- `cargo test -p chelis-cli phase3m_rust_runtime_hip_manual_gate -- --ignored --nocapture`

---

## 3g: Data Loading and Tokenization

**Status:** shipped.

**Goal:** Chelis can load data from files and tokenize text, making it self-sufficient
for the complete AI training pipeline.

### Why This Matters

This is the capstone of Phase 3. With scalars (3c), strings (3c), and collections (3d),
Chelis has the data types needed for preprocessing. But without file I/O and
tokenization, the data still has to come from Python. This phase makes Chelis's
preprocessing story complete.

The test: can you write a complete training pipeline — from raw text file to trained
model — in pure Chelis? After 3g, the answer is yes.

### Design

**File I/O:**

Pragmatic surface. IO effect on everything.

| Operation | Signature | Effect | Notes |
|---|---|---|---|
| `read_file` | `String -> String` | IO | Read entire file as UTF-8 string |
| `write_file` | `(String, String) -> unit` | IO | Write string to file |
| `read_lines` | `String -> List[String]` | IO | Read file, split by newline |
| `read_bytes` | `String -> List[Int]` | IO | Read raw bytes as integer list |
| `file_exists` | `String -> Bool` | IO | Check file existence |
| `list_dir` | `String -> List[String]` | IO | List directory entry names in [05-HOST-4] byte order, with strict UTF-8 conversion and whole-call failure on an invalid name |
| `mmap_file` | `String -> MappedFile` | IO | Memory-map a file for zero-copy random access |
| `mmap_read` | `(MappedFile, Int, Int) -> List[Int]` | Pure | Read bytes from offset+length after open |
| `mmap_len` | `MappedFile -> Int` | Pure | File size in bytes |

The #1479 row-7 directory conversion acceptance command is:

```sh
cargo nextest run -p chelis-runtime -p chelis-compiler-api -p chelis-cli --lib --test issue_1479_list_dir_order --test issue_1479_list_dir_lane_parity -E 'test(list_dir_conversion) | binary(~issue_1479_list_dir)' --no-fail-fast --retries 0
```

Acceptance requires every selected case to execute and pass, including the
Linux filesystem cases and the compiled C executions (a C compiler is required).
The suites also run in the hosted workspace jobs. macOS executes the in-memory
invalid-name conversion controls and valid-name filesystem/parity controls; Linux
additionally creates invalid filenames and verifies whole-call failure through
the evaluator, runtime FFI, and generated C. macOS alone does not prove those
filesystem failure paths. The executable example is
`examples/io/list_directory.ch`, with an absolute fixture path substituted by
the parity suite before formatting, checking, evaluating, and compiling it.
This is the conversion row's acceptance surface, not completion of #1479's
other carriers or this phase's broader I/O work.

**Memory-mapped I/O:** For datasets that don't fit in memory (billions of trade events,
large token corpora), `mmap_file` provides zero-copy random access to file contents. The
file is memory-mapped via the Rust runtime (`memmap2` crate behind the C ABI).
`mmap_read` returns the requested byte range as a `List[Int]`. In the compiled runtime,
the file handle stays memory-mapped after `mmap_file`; in the evaluator, the mapped file
value is represented as an in-memory byte buffer. `MappedFile` is an opaque runtime
handle (like strings and collections after 3m). The IO effect is on `mmap_file`
(opening the file); subsequent reads are pure because they operate on the already-open
mapping/buffer.

**CSV parsing:**

```chelis
import Std.Io.Csv

data = read_csv("train.csv")

first_label =
  match dict_get(index(data, 0), "label") with {
    | Some(text) => to_int(text)
    | None => None
  }
```

Implementation: a simple CSV parser in pure Chelis (using string split, not a C library).
Handles quoted fields and escaped commas. `read_csv` is the fail-loud public API;
`try_read_csv` remains available for callers that want recovery as `Option`. Ships as a
`Std.Io.Csv` module in chelis-std.

**JSON parsing:**

```chelis
import Std.Io.Json

config = load_json("config.json")

lr = json_float(json_get(config, "learning_rate"))
```

Implementation: a JSON parser in pure Chelis. Returns a `Json` ADT:
```chelis
type Json =
  | JsonNull
  | JsonBool(Bool)
  | JsonInt(Int)
  | JsonFloat(Float)
  | JsonString(String)
  | JsonArray(List[Json])
  | JsonObject(Dict[String, Json])
```

**Tokenizer:**

The critical capability. A BPE tokenizer that can load HuggingFace tokenizer
vocabularies.

```chelis
import Std.Tokenizer

tok = load_tokenizer("tokenizer.json")

tokens = encode(tok, "Hello, world!")

inputs = batch_encode(tok, ["Hello, world!"], cast(512, i64), cast(0, i64))
```

Implementation: the tokenizer is a Chelis program, not a C library wrapper. BPE merge
rules are stored as a `Dict[String, Int]` (merge priority). Encoding walks the input
string, applies merges greedily, and returns a `List[Int]`. This is slower than
HuggingFace's Rust tokenizer but it's pure Chelis — the model can learn to write and
modify tokenizer code.

`load_tokenizer` reads the HuggingFace `tokenizer.json` format (using the JSON parser
from this phase) and constructs the internal merge table and vocabulary dict.
`try_load_tokenizer` remains available when the caller wants recovery instead of a
fail-loud load.

`batch_encode` is the bridge function: encode N strings, pad to an exact width, and call
`pad_sequences_to` from 3d to produce a model-ready tensor.

**Data loading pipeline:**

Compose the above into a training data loader:

```chelis
import Std.Tokenizer

def load_training_data(data_path: string, tok_path: string,
                       max_len: i64, batch_size: i64) -> List[tensor[batch, max_len, i64]] = {
  tok = load_tokenizer(tok_path)
  lines = read_lines(data_path)
  encoded = map(fn (line: string) -> encode(tok, line), lines)
  batches = chunk(encoded, batch_size)
  map(
    fn (batch: List[List[i64]]) -> pad_sequences_to(batch, max_len, cast(0, i64)),
    batches
  )
}
```

### Implementation Plan

1. Add file I/O built-ins with IO effect (read_file, write_file, read_lines, read_bytes,
   file_exists, list_dir)
2. Add memory-mapped I/O (mmap_file, mmap_read, mmap_len) backed by Rust runtime's
   memmap2
3. Implement `Std.Io.Csv` as a pure Chelis package module
4. Add Json ADT and implement `Std.Io.Json` as a pure Chelis parser
5. Implement `Std.Tokenizer` with BPE encode/decode in pure Chelis
6. Add `load_tokenizer` for HuggingFace tokenizer.json format
7. Implement `batch_encode` bridging to pad_sequences from 3d
8. Add `chunk` utility for batching lists
9. Write a complete training data loader example using mmap for large datasets

### Test Plan

- File I/O: read_file/write_file round-trip, read_lines splits correctly, read_bytes
  returns correct values, file_exists works
- Memory-mapped I/O: mmap_file opens successfully, mmap_read returns correct bytes at
  offset, mmap_len matches file size, IO effect on mmap_file only (reads are pure)
- CSV: parse a simple CSV, handle quoted fields, handle headers
- JSON: parse all JSON value types, nested objects/arrays, malformed JSON returns error
- Tokenizer: BPE encode matches HuggingFace reference output for a small vocabulary
- Tokenizer: decode(encode(text)) round-trips for ASCII text
- Tokenizer: batch_encode produces correctly padded tensor
- Tokenizer: load_tokenizer parses HuggingFace tokenizer.json correctly
- Data loader: `examples/illustrative/phase3g_text_pipeline/` checks, evaluates, builds,
  and compiled output matches `chelis eval`

### Acceptance Oracle

Authoritative oracle:

```sh
cargo test -p chelis-cli --test std_io_pipeline phase3g_text_pipeline_acceptance_oracle -- --ignored --exact --nocapture
```

The checked-in illustrative Reef package at
`examples/illustrative/phase3g_text_pipeline/` must:

- read CSV and JSON files
- load a HuggingFace-format tokenizer
- encode text into integer sequences
- batch and pad those sequences into tensors
- produce matching `chelis eval` and compiled C output without Python preprocessing

---

## 3i: Standard Library Expansion

**Status:** shipped.

**Goal:** Fill out the standard library modules needed for real model training and
inference. Includes date/time types, exact decimal arithmetic, autoregressive generation
with KV caching, optimizer variants, and learning rate scheduling.

### Std.Time

Pure Chelis standard library module for dates and durations.

- `Date` type: year, month, day. Constructed via
  `date(cast(2024, i64), cast(1, i64), cast(15, i64))`.
- `Duration` type: days, hours, minutes, seconds. Constructed via
  `duration(cast(1, i64), cast(2, i64), cast(3, i64), cast(4, i64))`.
- Arithmetic: `add_days(date, n)`, `sub_days(date, n)`, `days_between(date1, date2)`.
- Comparison and ordering on dates.
- Formatting: `date_to_string(date)` → ISO 8601 (`"2024-01-15"`).
- Parsing: `parse_date(string)` → `Option[Date]`.
- Queries: `day_of_week(date)`, `day_of_year(date)`, `is_leap_year(year)`.
- No timezone handling in v1 — UTC only. Timezone support deferred.

### Std.Decimal

Pure Chelis standard library module for fixed-point exact arithmetic.

- `Decimal` type: exact representation with configurable scale.
- Construction: `decimal("0.1")`, `decimal_from_int(cast(42, i64))`.
- Arithmetic: `decimal_add`, `decimal_sub`, `decimal_mul`, `decimal_div` with explicit
  rounding mode.
- Rounding modes: `round_half_up`, `round_half_even` (banker's rounding), `round_down`,
  `round_up`.
- Comparison and ordering.
- Conversion: `decimal_to_float(d)` → `Float`, `decimal_to_string(d)` → `String`.
- Property: `0.1 + 0.2 == 0.3` is true with Decimal arithmetic.
- Not a tensor dtype — host-value type only. Exact, not fast.

### Std.Nn.Generate

(Module since moved to `School.Nn.Generate` in chelis-std 0.4.0.)

Autoregressive generation with KV cache management — the core inference pattern for
generative models. Exposed as the TradeFM analysis showed: generate-one-token-then-
feed-back is the fundamental inference loop, and doing it manually via `fold` is correct
but verbose.

```chelis
import School.Nn.Generate

-- Simple greedy generation
generated = generate(model_fn, context, cast(512, i64))

-- With sampling controls
generated = generate_with(key_from_seed(42i64), model_fn, context, GenerateConfig {
  max_tokens: cast(512, i64),
  temperature: 0.8,
  top_k: cast(50, i64),
  top_p: 0.95
})
```

Internals:
- `KVCache` type: stores past key/value tensors per layer to avoid recomputing attention
  over the full sequence at each step. In the shipped library this is precision
  polymorphic: `KVCache[a]`.
- `generate` runs the autoregressive loop: call the model with cached KV, get logits for
  the next position, sample or argmax, append to the sequence, update the cache.
- Sampling: `sample_next_tokens(k, logits, config)` applies temperature
  scaling, top-k filtering (via `sort`), top-p (nucleus) filtering (via suffix-mass on
  sorted probabilities), and categorical sampling with an explicit key.
- The model function signature:
  `(input_ids: tensor[batch, seq, i64], cache: Option[KVCache[p]]) -> (logits: tensor[batch, vocab, f32], new_cache: KVCache[p])`.

The greedy `generate` helper needs no key. At the explicit-key School release,
sampled `generate_with` receives a key and splits it for each sampling step;
the key is consumed even when a runtime config chooses greedy output. The
sampled-noise helper accepts exact supplied uniform values so tests can check
token selection independently of the generator.

### Std.Optim (expansion)

(Module since moved to `School.Optim` in chelis-std 0.4.0.)

Add optimizer variants beyond the existing SGD and Adam:

- **AdamW:** Adam with decoupled weight decay. The standard optimizer for modern
  transformer training. Difference from Adam: weight decay is applied directly to
  parameters, not through the gradient. One additional parameter
  (`weight_decay: Float`).
- **LAMB:** Layer-wise Adaptive Moments optimizer for large-batch training. Scales
  learning rates per-layer based on parameter and gradient norms.

Both are pure Chelis library functions composed from existing tensor operations. AdamW is
a small modification of the existing Adam implementation.

### Std.Schedule

(Module since moved to `School.Schedule` in chelis-std 0.4.0.)

Learning rate scheduling — adjust the learning rate over the course of training.

```chelis
import Std.Schedule

-- Cosine annealing with warmup
lr = cosine_with_warmup(
  step,
  CosineWarmupConfig {
    warmup_steps: cast(1000, i64),
    total_steps: cast(100000, i64),
    min_lr: 1e-6,
    max_lr: 3e-4
  }
)

-- Linear warmup then constant
lr = linear_warmup(
  step,
  LinearWarmupConfig {
    warmup_steps: cast(1000, i64),
    target_lr: 3e-4
  }
)

-- Step decay
lr = step_decay(
  step,
  StepDecayConfig {
    initial_lr: 3e-4,
    decay_factor: 0.1,
    decay_steps: [cast(30000, i64), cast(60000, i64)]
  }
)
```

All schedulers are pure functions from `(step, config)` to a scalar learning rate. They
compute the learning rate from the step number and record configuration. No state, no
mutation — pass the step counter explicitly. This composes cleanly with the training
loop (`fold` where the accumulator carries step count + model parameters).

### Implementation

All modules are pure Chelis code shipped as part of `chelis-std` via the reef package
system. `Date` is internally an ADT with integer fields. `Decimal` is internally a
scaled integer representation. `KVCache` is a `List` of layer caches. Optimizers and
schedulers are pure functions over tensors and scalars. No C runtime additions needed —
these are host-value computations (Time, Decimal, Schedule) and tensor computations
(Generate, Optim) using existing primitives.

### Test Plan

- Date arithmetic: add/subtract days, month boundaries, leap years
- Date parsing: ISO 8601 round-trip
- Date comparison: ordering works correctly across year boundaries
- Date utilities: `Duration`, `day_of_year`, and `is_leap_year` evaluate with expected values
- Decimal: `decimal("0.1") + decimal("0.2") == decimal("0.3")`
- Decimal: banker's rounding matches expected behavior
- Decimal: division with explicit rounding mode
- Generate: greedy generation produces correct tokens for a trivial model
- Generate: temperature sampling with the same key produces reproducible output;
  a second use of that key is rejected
- Generate: positional record-constructor misuse is rejected in package mode
- AdamW: zero-gradient decay still updates parameters via decoupled weight decay
- LAMB: trust-ratio update produces the expected tensor output on a fixed probe case
- Schedule: cosine_with_warmup hits the expected edge points (`step=0`, `step=total`)
- Schedule: step_decay triggers at correct steps
- All modules build and import through reef package system

### Acceptance Oracle

`cargo test -p chelis-cli --test std_package_acceptance -- --ignored --nocapture`

This is the owning executable oracle for the 3i-shipped `Std.Time`,
`Std.Decimal`, `Std.Schedule`, `Std.Optim`, and `Std.Nn.Generate` surface (the ML
modules — `Schedule`, `Optim`, `Nn.Generate` — since moved to `School.*` in chelis-std
0.4.0). A later
phase-completion claim still requires a fresh-context red team and any documented manual
gates.

---

## 3j-pre: Release Infrastructure + Std Surface Expansion

**Status:** shipped.

**Goal:** clear the infrastructure and `chelis-std` surface that both `nautilus` and
`coral` depend on before either can start. This is the last pure-compiler + pure-`chelis-std`
sub-phase before the parallel shell work.

**Prerequisite:** 3i (`Std.Time`, `Std.Decimal`, `Std.Nn.Generate`, AdamW/LAMB, Schedule).

### Infrastructure

- confirm and build out the `chelis-lang` GitHub organization at
  <https://github.com/Chelis-Lang> (already reserved); no repos moved yet
- ship a compiler release binary via CI (e.g. `cargo dist`), so downstream shell repos
  can pin a specific `chelis` toolchain instead of building from source

### Std Surface Additions

Pure compiled additions to `chelis-std`. Nothing here requires a new shell.

(Historical record of the 3j-pre surface. The ML modules listed below — `Std.Nn.*`,
`Std.Loss.*` — since moved to `School.Nn.*` / `School.Loss.*` in chelis-std 0.4.0; the
tensor/reduction additions stayed in `chelis-std`.)

- **Tensor shape/construction:** `linspace`, `arange`, `stack`, `squeeze`, `unsqueeze`
- **Reductions:** `min`, `prod`, `argmax`, `argmin`
- **`Std.Nn`:** `GELU`, `SiLU`, `RMSNorm`, `Conv1d`, `Conv2d`,
  `scaled_dot_product_attention`, multi-head attention, grouped-query attention
  - **Acknowledged limitations (Batch 3 shipped 3j-pre):**
    - `GELU` ships as the tanh approximation
      (`0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`) because `erf` is
      not a Chelis primitive. This is the OpenAI/BERT/GPT-2 form, not the
      exact `erf`-based GELU. Switching to exact GELU is deferred until
      `erf` lands as a primitive.
    - `SiLU`/`GELU`/`RMSNorm` ship as **rank-1 variants** (`tensor[n, f32]`)
      because the current `tensor_binop` signature (single tvar shared
      between both operands) does not permit scalar-tensor broadcast, and
      the type system does not support rank-polymorphic defs. Callers
      with transformer-shaped hidden states (`[batch, seq, dim]`) are
      expected to flatten to rank-1 before applying the activation. The
      `sigmoid` tensor primitive is also unsupported by the host runtime
      lowering, so `SiLU` computes sigmoid element-wise via a scalar
      helper rather than the tensor primitive.
    - `Conv1d`/`Conv2d` wrappers and the attention modules
      (`scaled_dot_product_attention`, multi-head, grouped-query) are
      shipped in **Batch 3b** (see `crates/chelis-cli/tests/std_nn_conv_attention_acceptance.rs`)
      as **literal concrete-shape** defs — not polymorphic wrappers —
      for the following reasons, each tracked as a non-silent deferral:
      - `conv2d` requires concrete d-lit dims at IR check lowering time
        (`validate_ir_builtin_symbolic_requirements` in
        `crates/chelis-types/src/infer.rs`); a polymorphic
        `conv2d_forward[batch, in_c, out_c, ...]` wrapper is rejected.
        Batch 3b therefore ships `Std.Nn.Conv.conv1d` (`tensor[1, 4, 1, 16, f32]
        → tensor[1, 8, 1, 14, f32]`) and `Std.Nn.Conv.conv2d_small`
        (`tensor[1, 3, 8, 8, f32] → tensor[1, 8, 6, 6, f32]`). Making
        these polymorphic remains future work once IR check's concrete-
        dim requirement is relaxed for builtin conv2d.
      - Batch 3b additionally relaxed an internal IR check lowering
        assertion in `crates/chelis-ir/src/tier2.rs::lower_conv2d`. The
        bottom-up type-annotation writeback leaves the inner `conv2d`
        app's result-type metadata as non-concrete d-vars even when the
        enclosing def's return ascription already pins the spatial dims.
        The lowering now falls back to the `(strided_h, strided_w)`
        values computed from input/kernel/stride/padding and rebinds
        `h_out`/`w_out` as `DimInfo::Lit`. The relaxation is strictly
        concrete — if both the inferred and computed extents disagree
        the pre-existing shape-mismatch panic still fires.
      - `scaled_dot_product_attention` ships as concrete rank-2
        `tensor[4, 4, f32]` (seq=4, d_head=4). The `/sqrt(d_k)` scaling
        cannot be expressed as a scalar-tensor multiply because
        `tensor_binop` is still monomorphic `(T,T)→T` with no scalar
        broadcast primitive in the host runtime (`expand` is
        type-checkable but unsupported by the runtime). The wrapper
        therefore takes `scale: tensor[4, 4, f32]` as an explicit
        pre-built full-shape scaling tensor; callers are expected to
        fill it with `1/sqrt(d_k)` before the call. `multi_head_attention`
        is a direct per-head delegation to SDPA (MHA's reshape/permute
        orchestration is the caller's responsibility until rank-
        changing reshape is accepted by the package-mode defsig
        enforcer). `grouped_query_attention` is likewise per-head;
        the separate `gqa_broadcast_kv` helper wraps
        `gather(kv_pool, group_map, 0)` on the head axis, and a
        positive `gather` eval test in `phase3j_pre_std_batch3b` pins
        the exact broadcasted layout the GQA pattern relies on.
      - Numeric end-to-end evaluation of the attention/conv wrappers
        through `chelis eval` is not reachable because the host runtime
        lowering does not implement `matmul`, `softmax`, `permute`, or
        `expand`, and rank-changing `reshape` is rejected by the
        package-mode enforce-defsig pass whenever a `def` is present
        in the same source unit. The Batch 3b acceptance surface is
        therefore publish + import-touch + check-level shape negatives
        (SDPA q/k seqlen, MHA head dim, GQA group-map rank, Conv2d
        channel, Conv1d kernel length) plus the GQA gather positive,
        matching the `phase3j_pre_std_batch2` precedent for
        stack/squeeze/unsqueeze. Full build+gcc numeric coverage for
        the attention + conv wrappers is deferred to the Phase 3j-pre
        oracle test once rank-changing reshape / scalar-tensor
        broadcast / `matmul`+`softmax` host runtime support lands.
- **`Std.Loss`:** `KLDivergence`, `BCEWithLogits`, `accuracy`, `perplexity`
- **`Std.Init`:** `kaiming_uniform`, `kaiming_normal`, `xavier_uniform`, `xavier_normal`,
  `trunc_normal`

### Test Plan

- Each new primitive has a positive numerical test against a numpy/torch reference
  (within tolerance) and at least one shape/type negative test
- Attention has explicit shape-checking negative tests: mismatched head dims,
  wrong mask shape, q/k seqlen disagreement for grouped-query attention
- Init functions have statistical tests against target mean/variance

### Acceptance Oracle

`cargo test -p chelis-cli --test production_stdlib_typechecks` +
`cargo test -p chelis-cli --test std_package_acceptance` — exercise the in-repo std
surface through `chelis check`/`chelis eval` with hand-computed exact reference values.
(The original `std_nn_build_acceptance` suite, which exercised RMSNorm + GELU + Kaiming
init + SDPA import, was removed when the `Std.Nn`/`Std.Loss`/`Std.Optim` ML surface moved
to the downstream School library in chelis-std 0.4.0, #331. RMSNorm/GELU/SDPA now live in
School; Kaiming init (`Std.Init`) stayed and is covered by the surviving suites.)

Release infrastructure is also in place as of 3j-pre Batch 6: the
hand-rolled `.github/workflows/release.yml` builds and publishes a
Linux x86_64 `chelis` tarball to the GitHub Releases page on any `v*`
tag push. Release history:

- `v0.1.0` — marked prerelease; cut before the Batch 5b/7b red-team
  findings were resolved, so the C backend silent-seed-drop and the
  weak oracle assertions shipped in that tarball. Left in place for
  forensics; do NOT pin.
- `v0.1.1` — shipped the Batch 7b fixes but the workspace `Cargo.toml`
  was still pinned at `0.1.0`, so the published binary self-reports
  `chelis 0.1.0`. Do NOT pin. Caught by the post-fix red team.
- `v0.1.2` — first release whose `chelis --version` matches its tag.
  Downstream shells (Nautilus, Coral) should pin via
  `compiler = "=0.1.2"` in their `reef.toml`. Also ships the Batch 7c
  KL divergence `0·log(0)` guard. See
  `spec/design/phase3j_pre_release.md` for the full release contract.

  - **Acknowledged limitations (3j-pre, current state after Batch 7b):**
    > Historical note: the `crates/chelis-cli/tests/std_nn_build_acceptance.rs`
    > file referenced below was removed when the `Std.Nn`/`Std.Loss`/`Std.Optim`
    > ML surface moved to the downstream School library in chelis-std 0.4.0
    > (#331). The `Std.Nn.*` paths named here now live under `School.Nn.*`; the
    > in-repo successor oracles are `production_stdlib_typechecks` and
    > `std_package_acceptance`. The prose is kept as a Batch 5b/7b work record.
    - **Fixed in Batches 5b and 7b** (now exercised end-to-end through
      `chelis build --target c` + gcc-link + run with byte-exact stdout
      assertions in `crates/chelis-cli/tests/std_nn_build_acceptance.rs`, none
      of which are `#[ignore]`d):
      - `Std.Nn.RmsNorm.forward` (rank-1 wrapper) — Batch 5b restored
        the dim-variable emission and the host-lane wrapper pathway.
      - `Std.Nn.Gelu.forward` (rank-1 wrapper) — Batch 5b fixed the
        scalar-fn lowering through `to_tensor(map(scalar_fn,
        to_list(...)))`.
      - `Std.Tensor.Reduce.{min, prod, argmax, argmin}` wrappers — the
        host-lane lowerer force-routes pure-tensor wrapper function
        bodies through the tensor-helper path using the declared return
        type, so the four reduction wrappers (and `Std.Nn.Linear.forward`)
        emit real C definitions and link cleanly.
        (Superseded — the four `Std.Tensor.Reduce` wrappers were later
        REMOVED in chelis#333: they were bodyless sigs taking a runtime
        `i32` axis but the `*_reduce` builtins require a const axis, so
        they were unimplementable as declared and never had a runtime
        function. Consumers call `min_reduce`/`prod_reduce`/`argmax_reduce`/
        `argmin_reduce` directly with a const axis. `Std.Nn.Linear.forward`
        is unaffected.)
      - `Std.Nn.Attention.scaled_dot_product_attention` import — Batch
        5b removed the duplicate `chelis_uniform_sample_f32` emission
        so a downstream package can import the symbol and build a
        program that touches it through the C backend. The numeric
        attention math itself is still deferred (see below).
    - **Closed in Bucket 5** (closure of Batch 7b's deferral): the
      C backend now preserves `with seed(...)` through direct
      `uniform_like` calls and through generated host functions. Direct
      DAG-lowered `uniform_like` still binds the seed at IR-lowering
      time (`chelis_ir::lower::lower_handle_effect`) and bakes it into
      `RiscOp::UniformLike { seed }`, which the C and HIP emitters pass
      as the first argument to `chelis_uniform_sample_f32(seed, index,
      low, high)`. The C host fallback path also represents
      `with seed(...)` explicitly and emits a generated
      `chelis_rng_current` handler scope; nested stdlib/user functions
      such as `kaiming_uniform` and `normal_like` draw random values
      from that active handler instead of silently using baked seed `0`.
      The pre-Bucket-5 project-wide rejection gate at
      `crates/chelis-cli/src/main.rs::reject_with_seed_for_build_target`
      is removed; the function survives as a no-op forward-compat
      hook. The Bucket-5 oracle tests in
      `crates/chelis-cli/tests/std_nn_build_acceptance.rs` and
      `crates/chelis-cli/tests/cli.rs` pin: (1) `with seed` builds and
      runs end-to-end through `chelis build --target c` + gcc + run,
      (2) the same seed produces identical bytes across runs
      (determinism), (3) different seeds produce different bytes
      (no silent drop), (4) a sibling file that does not use
      `with seed` is no longer blocked, and (5) cross-function stdlib
      random initializers use the enclosing handler seed. The C runtime
      reduces in f32 so its byte-exact output differs from the f64
      `chelis eval` reference in `phase3j_pre_oracle_integrated_eval`
      in trailing bits; that drift is intrinsic to f32 vs f64 reduction
      precision, not a seed-plumbing bug.
    - **Still deferred to Phase 3j (Nautilus):**
      - Numeric verification of `Std.Nn.Attention.scaled_dot_product_attention`,
        `multi_head_attention`, `grouped_query_attention`, and
        `gqa_broadcast_kv` — still blocked by the host-runtime
        `matmul`/`softmax`/`permute`/`expand` gap on the attention
        path. The Batch 3b wrappers ship as type-check + importability
        only at the std layer; consumers cannot yet run them
        numerically.
      - Numeric verification of `Std.Nn.Conv.conv1d` and `conv2d_small`.
        Batch 3b ships these as type-check-only wrappers; the executable
        numeric oracle is owed to Phase 3j.
      - `argmax`/`argmin` index precision above 2^24: the F32 index
        path silently saturates above 2^24 elements per axis. Tracked
        for a Phase 3j fix; until then, callers should keep reduction
        axes well below 16M elements.
    - The `grad` non-differentiability negative test is enforced at
      the IR layer (`crates/chelis-ir/src/grad.rs::adv_argmax_on_grad_path_errors_cleanly`,
      `adv_argmin_on_grad_path_errors_cleanly`). The package-mode
      `chelis eval` lowering panics with `` `grad` is not
      representable in the IR check RISC DAG `` before the gradient
      pass runs, so the CLI-level negative pins the typecheck-time
      refusal that `grad` requires a scalar floating output (which
      is what `argmax`/`argmin` violate). Promoting this to the
      IR-level "non-differentiable" message is gated on package-mode
      grad lowering, which is out of scope for 3j-pre.

**Effort:** medium. Each primitive is small, but the surface is wide and every item
needs both a positive numerical test and a negative shape/type test.

---

## 3j: Nautilus — Numerical Methods, Statistics, and Optimization

**Status:** shipped in the downstream Nautilus repo. `Nautilus v0.5.0` is the current
released shell referenced by the canonical ecosystem table. `chelis v0.1.7` cleared the
last documented core blockers for the first Nautilus shell release.

**Goal:** A reef package providing the numerical methods that sit between raw tensor
primitives and domain applications. The scipy competitor for Chelis — `scipy.stats` +
`scipy.optimize` + `scipy.integrate` + `scipy.linalg` + `scipy.special` under one shell.

**Prerequisite:** 3h (core numeric primitives), 3i (`Std.Time` for time-series stats),
3j-pre (compiler release binary, expanded `Std.Nn`/`Std.Loss`/`Std.Init` surface).

### Implementation Strategy

- **Pure Chelis** where it's natural: stats, roots, ODE, integration, distributions,
  interpolation. No C FFI, no compiler special-casing. Functional composition over
  existing tensor primitives, scalar math, and collections.
- **nalgebra** as the linear algebra backend. The Rust runtime exposes
  `chelis_svd`, `chelis_cholesky`, `chelis_qr`, `chelis_lu`, `chelis_solve`, `chelis_eig`,
  `chelis_inv`, `chelis_det` backed by the nalgebra crate. Same pattern as BLAS-for-matmul.
  nalgebra is pure Rust, links against BLAS/LAPACK when present, and falls back to a
  pure-Rust implementation otherwise — no Fortran dependency.
- **AD through LinAlg:** nalgebra calls are opaque to the Chelis AD system, so we ship
  hand-written adjoint rules for `svd`, `cholesky`, `solve`, `qr`, `eig`, registered
  alongside the RISC primitive adjoints. Same pattern PyTorch uses for `torch.linalg`.
  References: Giles 2008 (*An extended collection of matrix derivative results for
  forward and reverse mode AD*), Townsend 2016 (*Differentiating the Singular Value
  Decomposition*).
- `Nautilus.Signal` ships as a typed API stub (like `Std.Io.Safetensors` was in 3a) —
  correct signatures and documentation, implementation blocked by complex number support
  in Phase 5f.

`grad` through ODE solvers is the highest-value composition test for the pure-Chelis
tier: `grad(solve_ode(f, x0, t), wrt=x0)` must work for neural ODE research. `grad`
through `svd` via finite differences is the highest-value test for the hand-written
adjoint tier.

### Modules (Priority Tiers)

#### P0 — ship first

| Module | Contents | Key Dependencies |
|---|---|---|
| `Nautilus.Special` | `erf`, `erfinv`, `log_gamma`, `digamma`, `beta`, `lbeta`. Required by Distributions. | scalar math |
| `Nautilus.Distributions` | Normal, LogNormal, Uniform, Student-t, Chi-squared, Exponential, Gamma — PDF, CDF, inverse CDF, sampling. `normal_like` is Box-Muller inside this module. | `Nautilus.Special`, explicit keys for sampling |
| `Nautilus.LinAlg` | SVD, PCA, eigendecomposition, Cholesky, QR, LU, solve, inverse, determinant. nalgebra-backed with hand-written adjoint rules for AD. | runtime nalgebra FFI, adjoint registry |

#### P1

| Module | Contents | Key Dependencies |
|---|---|---|
| `Nautilus.Stats` | Descriptive statistics (variance, skew, kurtosis, median), correlation, covariance, shrinkage estimators | 3h: sort, quantile, einsum |
| `Nautilus.Optim` | Convex optimization solvers (QP, SOCP, LP). Differentiable optimization via implicit differentiation through KKT conditions. NOT neural network optimizers (those are `School.Optim`). | `Nautilus.LinAlg`, einsum |
| `Nautilus.Roots` | Root finding (Newton-Raphson, bisection, Brent) | scalar math, host control flow |
| `Nautilus.ODE` | ODE solvers (Euler, RK4, adaptive step). Composes with `grad` for neural ODE support. | cumsum, host control flow |

#### P2

| Module | Contents | Key Dependencies |
|---|---|---|
| `Nautilus.SDE` | SDE solvers (Euler-Maruyama, Milstein). Stochastic steps consume explicit keys. | `Nautilus.ODE`, keyed draws, cumsum |
| `Nautilus.Integrate` | Numerical integration (trapezoidal, Simpson's, Gaussian quadrature) | fold, scalar math |
| `Nautilus.Interpolation` | Linear, cubic, spline interpolation | sort, gather |
| `Nautilus.Testing` | Hypothesis testing, confidence intervals, p-values | `Nautilus.Distributions`, `Nautilus.Stats` |
| `Nautilus.Distance` | Euclidean, cosine, Mahalanobis, Manhattan distances over tensor rows | einsum, `Nautilus.LinAlg` |
| `Nautilus.Signal` | Signal processing (FFT, STFT, filtering). **Blocked by complex numbers (Phase 5f) — stub in 3j.** | Phase 5f complex tensors |

### Test Plan

- Each module has at least 3 positive tests comparing against scipy/numpy reference
  values within tolerance
- Negative tests: wrong input shapes, unsupported types, singular matrices for solve,
  non-PSD matrices for Cholesky
- **nalgebra parity:** every `Nautilus.LinAlg` operation is tested against scipy on a
  battery of random + structured matrices
- **AD through LinAlg:** finite-difference check of hand-written adjoints for `svd`,
  `cholesky`, `solve`, `qr`, `eig` on non-degenerate inputs
- **AD composition:** `grad` through `Nautilus.ODE.rk4`, `Nautilus.Optim.solve_qp`,
  `Nautilus.Interpolation.cubic`
- **Key discipline:** each sampler consumes its key once, splits keys for independent
  draws, and has a supplied-noise helper for deterministic value tests;
  `Nautilus.Signal` stub propagates its actual effect annotations
- **Package gate:** `chelis reef build` produces a valid `.chb`, consumer imports and
  type-checks

### Acceptance Oracle

The authoritative ship signal for `3j` is the downstream Nautilus release gate:
the green Nautilus `main` CI that produced the published release
(tagged at commit `20c5553`). Chelis no longer treats an unimplemented
monorepo `phase3j_nautilus_oracle` placeholder as the completion oracle for
this phase.

### Cross-Repo CI

Nautilus now validates in its own repository and release process. Chelis core
changes that affect shell-facing compiler behavior should be revalidated against
that downstream gate before a compiler release, but Nautilus is no longer a
hypothetical future dependency here: it is a shipped consumer.

**Effort:** large. `Nautilus.LinAlg` (nalgebra bridge + AD adjoints) and `Nautilus.Optim`
(QP solver) are the bulk; stats, distributions, and special functions are straightforward.

---

## 3k: Coral — Typed Dataframes

**Goal:** A reef package for structured tabular data where numeric columns are
GPU-accelerable tensors and string columns are host-side lists. The unique feature: AD
flows through dataframe operations, enabling sensitivity analysis no existing dataframe
library supports.

**Prerequisite:** 3h (gather, scatter, argsort for sort-by/group-by), 3d (collections
for string columns), 3g (Std.Io.Csv/Json for data loading).

**Verified compiler/std ground truth before Coral starts:**

- ADTs + exhaustive pattern matching already support a `Column` sum type.
- `Dict[string, tensor[...]]` already type-checks, so tensor-valued maps are available.
- `gather` already accepts `tensor[..., bool]` and builds through the C backend.
- `Std.Tensor.Construct.arange` already ships.
- `Std.Tensor.Mask.where_indices` is the sanctioned Phase A helper for
  boolean-mask-to-index conversion. It is intentionally host-lane
  (`to_list` → `enumerate` → `filter` → `map` → `to_tensor`), not a new tensor
  primitive.
- Known residual: an all-false mask still hits the existing empty-`to_tensor([])`
  limitation, so zero-match filtering needs a downstream special case until empty
  tensor construction lands.

### Core Design

A DataFrame is `Dict[String, Column]` where:

```chelis
type Column =
  | IntCol(tensor[n, i64])
  | FloatCol(tensor[n, f32])
  | StringCol(List[String])
  | BoolCol(tensor[n, bool])
```

Numeric columns are tensors on the lazy RISC DAG — they go through the tensor lane, get
GPU-accelerated, support AD, and benefit from the compiler's operation fusion. A
`filter → mutate → aggregate` pipeline on numeric columns may compile to a single fused
kernel. String columns are host-side lists — they go through the host lane and execute
eagerly. The type system tracks which columns are which. No query optimizer — numeric
optimization comes from the tensor compiler's existing fusion passes, not a
dataframe-specific planner.

**Persistent column dictionary (HAMT).** The internal `Dict[String, Column]` backing a
Frame uses a persistent data structure (hash array mapped trie) so that `with_column`,
`drop_column`, and `rename` produce new frames that share column references with the
original via structural sharing. This is a performance requirement for AD through frame
pipelines: `grad(fn_with_10_frame_ops)` produces intermediate frames on the backward
pass, and without structural sharing each intermediate copies the entire column
dictionary — making AD memory cost O(num_columns * num_operations) instead of
O(num_operations). The HAMT stores column references (`Arc` handles to immutable
tensors), not column data, so the tree is small at real portfolio sizes (50-100
columns). Pure-Chelis HAMT is strongly preferred over Rust-side HAMT so the persistent
dict composes transparently with `grad`; decide at implementation start which path
actually composes (see coral spec §13 open question #7).

### Modules

| Module | Contents | Key Primitives Used |
|---|---|---|
| `Coral.Frame` | DataFrame construction, column selection, row filtering (`Std.Tensor.Mask.where_indices` on the host lane, then `gather`), sorting by column (`argsort` → `gather` all columns), mutation (add computed column), column type queries, `rename`, vertical `concat`, `describe` (calls `Nautilus.Stats`), `value_counts` (groupby shorthand). **NaN handling lives here, not in a separate module:** `is_nan`, `fill_nan`, `drop_nan`, `any_nan`, `count_nan`. Float columns use IEEE 754 NaN propagation (GPU kernels handle NaN correctly); integer columns use a companion boolean mask for missingness. | gather, argsort, where, `Nautilus.Stats` |
| `Coral.GroupBy` | Group-by via `argsort` + run-length detection, aggregation (sum, mean, count, min, max per group) via segmented `scatter(..., "add")` | argsort, scatter, cumsum |
| `Coral.Join` | Sort-merge join on typed key columns, left/inner/outer join variants | argsort, gather, concat |
| `Coral.Reshape` | Pivot (long → wide), melt (wide → long), stack/unstack | Dict manipulation, tensor reshape |
| `Coral.Window` | Rolling operations over numeric columns: `rolling_mean`, `rolling_sum`, `rolling_std`, `ewm` (exponentially weighted moving average). Expressible via `cumsum` tricks but worth naming. | cumsum, einsum |
| `Coral.IO` | DataFrame-aware CSV loading (wraps `Std.Io.Csv`, auto-detects column types, returns typed DataFrame), JSON loading, DataFrame → CSV export, **`read_parquet` / `write_parquet` backed by the Rust `parquet2` crate in the runtime** (same integration pattern as `mmap_file` via `memmap2` in 3g). Parquet is the standard columnar format for ML datasets and the largest functional gap vs pandas. | `Std.Io.Csv`, `Std.Io.Json`, runtime `parquet2` FFI |

### AD Through Dataframes

The key differentiator. Because filter is `gather` and aggregation is
`scatter(..., "add")` + `sum`/`mean`, the entire filter → aggregate pipeline is
differentiable:

```chelis
def portfolio_risk(prices: Coral.Frame, threshold: f32) -> f32 = {
  -- filter: gather (differentiable)
  high_vol = Coral.filter(prices, \row -> get_float(row, "volatility") > threshold)
  -- aggregate: mean over tensor column (differentiable)
  Coral.mean_col(high_vol, "return")
}

-- Sensitivity of risk measure to threshold
grad(portfolio_risk, wrt=threshold)  -- works because filter → gather → AD
```

No existing dataframe library supports this.

### Competitive Position vs pandas / Polars

**Where Coral wins:**
- **GPU-accelerated numeric columns for free.** A filter → mutate → aggregate pipeline on
  numeric columns compiles through the tensor DAG and can fuse into a single GPU kernel.
  pandas is CPU-only even with the Arrow backend. RAPIDS cuDF has GPU dataframes but a
  different API; Coral's numeric columns are tensors, so they get GPU acceleration
  without a separate code path.
- **AD through dataframe operations.** `grad(portfolio_risk)` where `portfolio_risk`
  filters a frame and aggregates a column flows through `gather` and `sum`. pandas,
  Polars, and RAPIDS cannot do this.
- **Statically typed columns once retrieved.** `get_float_col(df, "price")` returns
  `tensor[n, f32]`. Wrong column *type* is caught at the type-system level. Column
  *existence* is still dynamic because column names are strings.
- **Effect-tracked provenance.** Loading a CSV/Parquet has `IO` effect; a frame derived
  purely from computation is pure. The type system tracks what each frame depends on.

**Where pandas/Polars win:**
- **Query optimization.** Polars has lazy query plans with predicate pushdown,
  projection pushdown, and join reordering. Coral is eager for string columns and
  lazy-via-tensor-DAG for numeric columns — no cross-operation query planner. For
  complex multi-join analytical queries, Polars will be faster.
- **Missing-data maturity.** pandas has decades of NaN handling baked into every
  operation. Coral's NaN story starts with IEEE 754 propagation plus mask columns for
  integers — functional but newer.
- **Ecosystem.** pandas has thousands of integrations. Coral has none. Cold-start
  problem only adoption solves.

### Test Plan

- `Coral.Frame`: construct from columns, select, filter, sort_by, mutate — all produce
  correct results
- `Coral.Frame` NaN: `fill_nan`, `drop_nan`, `is_nan` match pandas reference on mixed
  float/int test data; GPU kernels propagate NaN correctly
- `Coral.GroupBy`: group_by + sum/mean/count matches pandas.groupby reference on test
  data
- `Coral.Join`: inner join matches pandas.merge on test data, key type enforcement works
- `Coral.Window`: `rolling_mean` / `rolling_std` / `ewm` match pandas reference within
  tolerance
- `Coral.IO`: CSV round-trip (load → export → reload) preserves data and column types;
  **Parquet round-trip** via `parquet2` preserves schema and typed columns
- AD: `grad` through filter + aggregate pipeline produces correct gradients
- GPU: numeric column operations compile to HIP and produce correct results (manual gate)
- Negative: wrong column name errors, type mismatch errors, join key type mismatch errors

### Acceptance Oracle

Current in-repo authoritative oracle:

```sh
cargo test -p chelis-cli --test coral_prerequisites -- --ignored --nocapture
```

This gate covers the compiler and `chelis-std` prerequisites Coral currently depends on:
where-indices behavior, bool-list tensor conversion, tensor-scalar comparison broadcasting,
and negative parity for invalid mixed precision/list inputs. The older placeholder name
`phase3k_coral_oracle` is not wired; do not use it as evidence for Phase 3k completion.

Future downstream Coral completion will replace this prerequisite gate with a Coral-owned
oracle that loads data into a DataFrame from both CSV and Parquet, filters rows
(including NaN handling), applies a rolling window, groups by a column, aggregates, and
verifies results match expected values. It should also include AD coverage through a
filter-aggregate pipeline.

**Effort:** medium. The core Frame/GroupBy/IO modules are the priority; Join, Reshape,
and Window can ship with minimal implementations and grow. Parquet via `parquet2` is a
runtime FFI addition following the existing `memmap2` pattern.

---

## 3l: Shoals — Finance

**Goal:** A reef package for quantitative finance. Pricing models, risk measures, yield
curves, stochastic processes, order books. Built entirely on `chelis-std` + `nautilus` +
`coral`. Contains only finance-specific logic.

**Prerequisite:** 3j (nautilus — distributions, optimization, SDE solvers), 3k (coral —
for loading/manipulating financial data), 3i (Std.Time for dates, Std.Decimal for cash
amounts).

### Key Design Decision: Instruments as Dicts, Not Closed ADTs

Financial instruments are open-ended — structuring desks invent new payoff formulas
continuously. Representing instruments as `Dict[String, f32]` (or `Dict[String, Column]`
for term structures) lets new instrument types be added as data without modifying the
Shoals source or releasing a new package version. The pricing function dispatches on a
key (e.g., `get(instrument, "type")`), not on a pattern match over a closed enum. This
also serves the AI coding story: an agent generating a new instrument definition writes
a dict literal (well within current LLM capability), not a new ADT variant (which
requires understanding the type system's extension points).

### Modules

| Module | Contents | Key Dependencies |
|---|---|---|
| `Shoals.Pricing` | Black-Scholes analytical, Heston semi-analytical, SABR calibration, Monte Carlo engines with variance reduction. Greeks via `grad` for free — write the pricing function, `grad(price, wrt=(spot, vol, rate))` gives delta/vega/rho automatically. | `Nautilus.Distributions`, `Nautilus.SDE`, explicit keys, cumsum |
| `Shoals.Risk` | VaR (parametric, historical, Monte Carlo), CVaR/expected shortfall, stress testing, scenario generation | `Nautilus.Stats`, sort/quantile, explicit keys |
| `Shoals.Curves` | Yield curve construction (bootstrap from market instruments), interpolation (linear, cubic, Nelson-Siegel), day count conventions (ACT/360, ACT/365, 30/360) | `Nautilus.Interpolation`, `Nautilus.Roots`, `Std.Time` |
| `Shoals.Stochastic` | SDE models: GBM, Heston, SABR, jump-diffusion. Path generation using cumsum + `Nautilus.SDE`. Variance reduction (antithetic, control variates). | `Nautilus.SDE`, explicit keys, cumsum, einsum |
| `Shoals.Orderbook` | Limit order book representation (price-priority sorted collections), matching logic, bid/ask spread computation, VWAP | Host-side collections, sort, `Std.Decimal` |

### What Makes This Work in Chelis

- **Greeks for free:** `grad(black_scholes_price, wrt=(spot, vol, rate, T))` gives all
  four first-order Greeks in one backward pass. `vmap(grad(...))` gives per-instrument
  Greeks for a portfolio. No bump-and-reprice, no finite differences.
- **Reproducible Monte Carlo:** `key_from_seed` and key-first draws make a
  simulation's random inputs explicit. Two runs with the same keys and declared
  inputs produce identical paths under [05-RNG-1]. Split child keys for separate
  paths and draws; this reproducibility is required for audit.
- **Typed market data:** Named tensor dimensions like `tensor[instrument, scenario, f32]`
  prevent accidentally multiplying a `[portfolio, maturity]` matrix by a
  `[maturity, scenario]` matrix when the dimensions don't match.
- **Data provenance:** A function that reads from a market data feed has an
  `IO` effect. A Monte Carlo function receives its key as an ordinary typed
  input, so its random dependency is visible in its signature.

### Test Plan

- `Shoals.Pricing`: Black-Scholes price matches analytical formula (< 1e-10 error)
- `Shoals.Pricing`: Monte Carlo price converges to Black-Scholes analytical for
  vanilla European call (< 1% error with 100K paths)
- `Shoals.Pricing`: Greeks via `grad` match analytical Black-Scholes Greeks
  (< 1e-6 error). Current Shoals status is source-level/typechecked only; the
  focused runtime smoke skips with a warning until the full pricing body is
  IR-lowerable under host-runtime `grad`.
- `Shoals.Risk`: Parametric VaR matches `Nautilus.Distributions.Normal.ppf` at standard
  confidence levels
- `Shoals.Curves`: Bootstrap reproduces known market instrument prices (< 1bp error)
- `Shoals.Stochastic`: GBM paths satisfy known statistical properties
  (mean = spot * exp(mu*T), variance matches theory)
- `Shoals.Orderbook`: matching logic satisfies price-time priority invariant
- Key discipline: MC pricing consumes its key once, and curve construction
  propagates `IO` for market data
- Reproducibility: identical keys and declared inputs produce identical prices
- Manifest: once `chelis manifest` is implemented, it records random operations
  and their explicit key roots for Shoals examples

**Reproducibility manifests.** `chelis_manifest_spec.md` specifies a planned
`chelis manifest` command that follows random operations' explicit key derivations
to a seed or entry argument and emits a structured report. It is not an existing
Shoals CI gate. Historical context is in
`archive/chelis_reproducibility_manifests.md`.

**Canonical finance properties.** Shoals ships with a `properties/` directory of
reference `@property` functions:

- `properties/pricing.ch` — put-call parity, price positivity, call bounded by spot,
  delta in [0,1], gamma positive for vanilla Europeans
- `properties/greeks.ch` — source-level grad-derived Greeks match textbook
  Black-Scholes Greeks within tolerance; executable tests retain finite-difference
  checks until the full pricing body is IR-lowerable under host-runtime `grad`
- `properties/monte_carlo.ch` — Monte Carlo price converges to analytic price as path
  count increases, variance decreases with path count
- `properties/no_arbitrage.ch` — bull spread payoff non-negative, butterfly spread
  payoff non-negative

Convention (cross-cutting, applies to every domain shell): properties are co-located
with the implementation code they constrain — same repo, same package, version-
controlled together. Properties are NOT a separate shell. `chelis prove src/` runs them
all against the shipped exports. Status of the underlying tool: `chelis prove` with
first-class `@property` annotations is **demo-blocking, scoped, ready to build** for
the first commercial CProof prospect. Full conventions: `chelis_trust_stack.md`,
`chelis_property_spec.md`.

**Reference implementations.** Shoals' delivery scope now includes a `references/`
directory alongside `properties/`. Each standard model in `Shoals.Pricing`,
`Shoals.Stochastic`, `Shoals.Curves`, and `Shoals.Risk` ships a simple
textbook-formula reference (Black-Scholes call/put + Greeks, Heston, Vasicek, CIR,
vanilla Monte Carlo, VaR/CVaR via historical simulation). The optimized `src/`
implementation is verified against the reference by
`@property matches_textbook_reference forall(...)` in `properties/pricing.ch`. Customers
write their own references only for proprietary models. Full design:
`chelis_reference_implementations_spec.md`.

### Acceptance Oracle

`cargo test -p chelis-cli --test shoals_oracle phase3l_shoals_oracle -- --ignored --exact --nocapture`
— prices a European call option via Black-Scholes and Monte Carlo, verifies
convergence, and checks same-seed reproducibility. A separate focused manual smoke,
`cargo test -p chelis-cli --test shoals_oracle phase3l_shoals_oracle_grad_greeks_match_analytic -- --ignored --exact --nocapture`,
checks that Shoals's grad-derived Greek properties are lower/type-check clean
without paying the Monte Carlo oracle runtime, then runtime-skips with a
warning because the downstream pricing body still contains constructs the IR
DAG path cannot execute.

**Manual gate.** Wall-clock ~5 minutes on AMD Ryzen AI Max+ 395 (driven by the
host evaluator's 20K MC sample loop on Shoals). The test is marked `#[ignore]`
to keep `cargo test --workspace` under the 60-second inner-loop budget defined
in `AGENTS.md` / `CLAUDE.md`. Run before any release tag whose pitch includes Shoals
end-to-end pricing. Skips with a clear message if a Shoals checkout is not
present (set `CHELIS_SHOALS_PATH` or place `shoals/` as a sibling of the
chelis monorepo root).

**Effort:** medium. Black-Scholes + Monte Carlo + basic risk is the core; curves and
order book are smaller. The bulk of the work is composing existing primitives (`nautilus`
solvers, `coral` dataframes, tensor ops), not implementing new infrastructure.

---

## 3n: Octant — LaTeX ↔ Deep Bridge (Part A)

**Goal:** A reef package providing a notation bridge between quant-finance LaTeX and
Chelis Deep. Part A ships the parser, the thin `SymExpr` AST, deterministic lowering
for every expression form that maps mechanically to Chelis, Deep → LaTeX rendering
with type overlays, and — critically — provenance spans on every Deep node produced
by Octant lowering.

**Prerequisite:** `3j` (nautilus — special functions, distributions, linear algebra,
integrals). Does **not** depend on `3k` or `3l`, so `3n` runs **in parallel with `3l`**.

Full architectural spec: `chelis_octant_design.md`. Executable sub-phase contract
(test plans, acceptance oracles, non-silent deferrals, infrastructure decisions):
`phase3n_octant.md`.

### Modules (Part A)

| Module | Contents | Key Dependencies |
|---|---|---|
| `Octant.Parse` | LaTeX subset parser — arithmetic, unary functions, powers/roots, transcendentals, special functions (`erf`, `\Phi`, `\Gamma`, `B`), derivatives (`\partial`), integrals, sums/products, piecewise, matrix notation, subscript/superscript conventions. Out-of-scope LaTeX produces clean diagnostic errors, never silent drops. | Rust LaTeX parser crate via runtime FFI (infrastructure decision owned by `phase3n_octant.md`) |
| `Octant.Symbolic` | The ~30-node `SymExpr` AST. | `chelis-std` |
| `Octant.Lower` (deterministic path) | SymExpr → Deep for every form where the LaTeX uniquely determines the computation. Special functions route through `Nautilus.Special` / `Nautilus.Distributions`; integrals through `Nautilus.Integrate`; matrix ops through `Nautilus.LinAlg`. | `Nautilus.Special`, `Nautilus.Distributions`, `Nautilus.LinAlg`, `Nautilus.Integrate` |
| `Octant.Render` | Deep → LaTeX with type overlays (named tensor dims → subscripts, effect markers, `grad` → partial-derivative notation). Excludes the Greek pattern matches that need Shoals context (deferred to 3o). | typed Deep from the compiler |
| `Octant.Provenance` | Source-span annotations on every Deep node produced by Octant lowering. Contract: every lowered Deep node's metadata map carries `provenance` (raw LaTeX fragment) and `source_span` (line, column, length). This is the core value proposition — the audit trail that proves compiled code implements the formula. | nothing new — Deep nodes already carry a metadata slot |

### Test Plan

- Parser corpus: at least one positive test per in-scope LaTeX construct plus
  negative tests for out-of-scope constructs (`\begin{theorem}`, TikZ, paragraph
  text, symbolic integration requests) that must produce diagnostics naming the
  offending token.
- **Black-Scholes `d_1` round-trip acceptance test:** parse
  `d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) T}{\sigma \sqrt{T}}` → lower → compile
  through `chelis check`/`chelis eval` → render back to LaTeX → strip type
  overlays (named-dim subscripts, effect markers, linearity markers are
  rendering decoration, not part of the in-scope parser grammar) → **re-parse
  the stripped output and assert the re-parsed `SymExpr` is structurally
  equal to the original modulo whitespace, bracket normalization, and
  floating-point formatting** (the `phase3n_octant.md §3.4` determinism invariant, not a
  brittle byte-for-byte LaTeX comparison) and walk the lowered Deep asserting
  every node carries a `provenance` span whose fragment is a substring of the
  original LaTeX.
- **Provenance error-localization test:** a deliberately ill-typed LaTeX
  fragment (for example a shape-mismatched `\sigma \sqrt{T}`) produces a
  `chelis check` error whose message surfaces the originating LaTeX source
  span, not just the Deep node id. This pins the audit-trail semantics — the
  presence-only provenance check is not enough by itself to prove the core
  value proposition.
- **Out-of-scope LaTeX invariant test (Cross-Sub-Phase Invariant §3.2):** fed
  `\begin{theorem}`, a TikZ block, and "please integrate `\int e^{-x^2}`", the
  parser returns diagnostics naming the offending token — it must never
  silently drop to an empty `SymExpr`. This test lives inside the acceptance
  oracle, not only in loose unit tests.
- Type-overlay rendering: named tensor dims appear as subscripts, `IO`
  effects produce correct markers, explicit key parameters render as ordinary
  inputs, and `grad(f, wrt=x)` renders as
  `\frac{\partial f}{\partial x}`.
- Provenance completeness: a fuzz-style test generates ten varied in-scope
  expressions, lowers each, and asserts zero Deep nodes have missing or empty
  provenance metadata. This pins the core invariant.
- Special function lowering positives: `erf`, `\Phi`, `\Gamma`, `\log\Gamma`, `B`
  each route to the correct `Nautilus` call.
- Integral lowering: `\int_0^T f(t)\,dt` → `Nautilus.Integrate.adaptive_simpson`.
- Matrix lowering: `\mathbf{A}^{-1}\mathbf{b}` → `Nautilus.LinAlg.solve(A, b)`.
- Package gate: `chelis reef build` produces a valid `.chb`, a consumer crate
  imports `octant` and type-checks.

### Acceptance Oracle

`cargo test -p chelis-cli phase3n_octant_oracle -- --exact` — exercises the
Black-Scholes `d_1` round-trip, the provenance-completeness invariant, a
special-function lowering through Nautilus, and the reef-package gate. Named here
but deliberately not implemented by the planning change set that introduced
`3n`; creating it is owned by the agent who picks up 3n coding work. A fresh-
context red team is still required before any `3n` completion claim.

### Cross-Repo CI

The `octant` shell builds and tests both in its own repo (on the `chelis-lang`
org) and in the Chelis monorepo integration run. Green octant CI is a
prerequisite for green Chelis CI once `3n` is shipping.

**Effort:** medium. The LaTeX parser + deterministic lowering + rendering is
mechanical once the `SymExpr` grammar is pinned; the provenance plumbing is
small but spans every lowering path.

---

## 3o: Octant — Finance Notation + Notebook (Part B)

**Goal:** Extend Octant with finance-notation lowering (SDE, Monte Carlo
expectation, calibration, yield curves) through `shoals` and `nautilus`, Greek
rendering pattern matches, and the `Octant.Notebook` cell runtime. Provenance
extends to cover the new node kinds using the `3n` contract.

**Prerequisite:** `3l` (shoals — `Shoals.Stochastic`, `Shoals.Pricing`,
`Shoals.Curves`) green, `3i` green (`Std.Time` is a direct dependency of the
yield curve / day count lowering path, not only a transitive dep through
`shoals`), **and** `3n` (octant Part A) green.

Full design: `chelis_octant_design.md`. Sub-phase contract: `phase3n_octant.md`.

### Modules (Part B)

| Module | Contents | Key Dependencies |
|---|---|---|
| `Octant.Lower` (LLM-assisted path) | SDE notation → `Shoals.Stochastic` (discretization, time grid, noise strategy), Monte Carlo expectation → `Shoals.Pricing` (variance reduction, explicit keys), calibration → `Nautilus.Optim`, yield curve → `Shoals.Curves`. Boundary rule: if LaTeX specifies the *what* but not the *how*, the coding model fills in the *how*. | `Shoals.Stochastic`, `Shoals.Pricing`, `Shoals.Curves`, `Nautilus.Optim`, `Std.Time` |
| `Octant.Render` (finance additions) | Greek pattern matches — `grad(price, wrt=spot) → \Delta`, `grad(price, wrt=vol) → \mathcal{V}`, `grad(price, wrt=rate) → \rho`, `grad(price, wrt=T) → \Theta`. Configurable variable-name conventions. | 3n render surface |
| `Octant.Notebook` | Cell runtime — formula, parameter, execution, Greek cells. Not a Jupyter kernel. Cells produce Deep, execution runs compiled C, rendering is mathematical notation. UI layer (web / VS Code / Cove extension / standalone) is a separate implementation decision. | full 3n Octant surface |
| `Octant.Provenance` (extension) | Same contract as 3n, applied to the new SDE / MC / calibration / curve node kinds. No Deep node produced by Octant lowering may be missing a span. | 3n provenance surface |

### Test Plan

- Black-Scholes full pricer round-trip: parse the full call pricing formula →
  lower → compile → evaluate → match analytical Black-Scholes within `1e-10`.
- Greeks: `grad(price, wrt=spot)` lowers and renders as `\Delta`, numerical
  value matches analytical delta within `1e-6`.
- GBM SDE lowering: `dS = \mu S\,dt + \sigma S\,dW_t` lowers through
  `Shoals.Stochastic`, generated paths satisfy statistical properties
  (mean `S_0 exp(\mu T)`, variance within tolerance).
- Monte Carlo expectation: `\mathbb{E}[\max(S_T - K, 0)]` converges to
  analytical Black-Scholes price within `1%` at 100k paths via
  `Shoals.Pricing`.
- Notebook cell contracts: formula edit triggers parse → lower → compile →
  render in one transaction; parameter cell bindings propagate to downstream
  execution cells; Greek cells show formula + simplification + numerical
  value.
- Provenance completeness (3o extension): every Deep node produced by SDE /
  MC / calibration lowering carries a valid span — regression of the 3n
  invariant on the new node kinds.

### Acceptance Oracle

`cargo test -p chelis-cli phase3o_octant_oracle -- --exact` — Black-Scholes full
pricer round-trip (including Greeks via `grad` and `Shoals.Pricing` Monte
Carlo), the extended provenance-completeness invariant, a notebook cell-kind
contract test, and the reef-package gate. Named here but not implemented by
this planning change. Fresh-context red team still required before any `3o`
completion claim.

### Non-Silent Deferrals

- **Octant Phase 4 — full-document LaTeX ingestion** (parsing full LaTeX
  papers, `\begin{equation}` extraction, formula↔prose association) is parked
  as a post-Phase-3 shell stub (`octant-docs`), tracked alongside `school`
  and `darwin` in `chelis_project_plan.md`. No Phase 3 sub-phase implements
  it.
- **`Octant.Signal`** (FFT, STFT, filter lowering) remains stubbed — blocked
  indefinitely by complex-number support (Phase 5f).

**Effort:** medium to large. The LLM-assisted lowering path is the novel piece
and depends on the SSD → SDFT → RLVR coding-model pipeline being mature enough
to produce correct Deep fragments for SDE / MC / calibration notation. Greek
rendering, notebook cell runtime, and provenance extension are mechanical by
comparison.

---

## 3t: Chelis-Native Testing

**Goal:** Add first-class testing to Chelis: a `Std.Test` module, a `chelis test` CLI command, and a convention for reef packages to include Chelis-language tests.

**Prerequisites:** Bug 9 fix (eval hang on reef imports), fast `chelis eval` with package-aware imports.

### Why This Matters

Python test harnesses remain ONLY for cross-language parity verification (Nautilus vs scipy, Coral vs pandas). Everything else — unit tests, property tests, integration tests, smoke tests — is written in Chelis and run via `chelis test`. This is a hard rule for all reef packages, current and future.

`chelis test` runs tests via the evaluator, not the build→gcc→link→run path. This means tests execute without a C compiler, without linking, without the runtime library. The evaluator already handles the full language surface. This is what makes it fast and what makes it usable as the default testing path for every reef package.

### Deliverables

1. **`Std.Test` module in chelis-std** — assertion functions (`assert_eq`, `assert_close`, `assert_close_tensor`, `assert_true`, `assert_false`, `fail`), either via a `Test` algebraic effect or runtime builtin.
2. **`chelis test` CLI command** — discovers `tests/*.ch` files, evaluates each via the evaluator (not build+gcc+link+run), calls every `def test_*()` function, reports pass/fail with structured output, exits 0/1.
3. **Test file convention** — `tests/*.ch` in every reef package, `def test_*()` naming, documented layout; `parity/` for Python-only parity scripts.
4. **Nautilus test migration** — mathematical identity tests, property tests, edge case tests, smoke tests move from Python to Chelis. Scipy parity tests remain in Python under `parity/`. Migration scope: ~60-70% of assertions move to Chelis.
5. **Coral test migration** — same split: structural correctness tests (frame construction, HAMT, filter/sort/groupby on known data, NaN handling, reshape round-trips) move to Chelis. Pandas parity stays in Python.
6. **SKILL.md for `Std.Test`** — documents assertion API for coding agents.

### Hard Rule

Only code that compares Chelis output against an external oracle (scipy, pandas, QuantLib) uses Python. All other tests are written in Chelis and run via `chelis test`.

### Reef Package Layout Convention

```
shell-name/
├── src/           # Chelis source
├── tests/         # Chelis-native test files (def test_*())
├── parity/        # Python-only parity scripts (scipy/pandas/QuantLib comparison)
└── reef.toml
```

### Acceptance Oracle

`chelis test tests/` exits 0 on the migrated Nautilus and Coral test suites. `parity/run_parity.py` continues to pass for the scipy/pandas comparison subset.

Full design: `chelis_native_testing_plan.md`

---

## 3f: SKILL.md v2

**Goal:** Update the teaching surface for the full Phase 2 + Phase 3 language, including
domain shells.

**Prerequisite:** All other Phase 3 sub-phases complete.

### Required Surface

The refreshed skill should teach:

- the shipped pipe-first Surf idiom from `3e`
- effects, linearity, macros, `vmap`, and tuples
- supported effect handlers and explicit key inputs
- scalar types (`Int`, `Float`, `Bool`) and operations
- strings and string operations
- collections (`List`, `Dict`) and functional iteration (`map`, `filter`, `fold`)
- `Option` type and pattern matching
- core numeric primitives (`einsum`, `concat`, `gather`, `cumsum`, `sort`, etc.)
- list/tensor bridge (`pad_sequences`, `stack`, `to_tensor`)
- file I/O and `IO` effect
- CSV/JSON parsing
- tokenizer usage
- `Std.Time` and `Std.Decimal` host-program idioms
- package imports (`Std.*`, `Nautilus.*`, `Coral.*`, `Shoals.*`)
- dataframe operations (`coral`), including NaN handling and Parquet I/O
- numerical methods (`nautilus`), including the nalgebra-backed LinAlg surface
- financial models (`shoals` overview, not exhaustive)
- expanded neural-network surface from `3j-pre` (GELU, SiLU, RMSNorm, Conv1d/2d,
  attention, GQA), now shipped as `School.Nn.*` in the `school` library (moved out of
  `chelis-std` in 0.4.0)
- `school` (classical ML), `darwin` (evolutionary algorithms), and `hull`
  (executable language specification) shells are named but scoped as stubs; SKILL.md
  mentions them as post-Phase-3 targets only
- the boundary between host-side preprocessing and tensor compute inside Chelis itself

### API Stability Labels

Every API surface table in the refreshed SKILL.md (`Std.*`, `Nautilus.*`, `Coral.*`,
`Shoals.*`) carries a `Stability` column whose value is `stable` (signature will not
change between releases — safe for training-corpus inclusion) or `alpha` (signature may
change — excluded or down-weighted for training). This is the labeling convention the
Phase 4a corpus-curation step relies on. Nautilus candidates for `stable`: all
of `Nautilus.Special`, all of `Nautilus.Distributions` (pdf/cdf/inv_cdf). Candidates
for `alpha`: `Nautilus.CurveFit`, `Nautilus.SDE` (APIs may shift when autonomous
keyed sampling lands). Apply the same convention to `Coral` and `Shoals` when they
ship.

### Supplied-Value Random Tests And Effect Handlers

Document and standardize the two distinct patterns for test doubles:

- A sampled function receives a key and delegates value-dependent work to a pure
  helper that accepts sampled noise. Tests pass exact noise to that helper, while
  same-key replay and split-child tests check the random boundary separately.
- `with_mock_io(recorded_trace) { ... }` — a test handler for `IO` effect that returns
  recorded data instead of reading files.
- `with_cpu_fallback { ... }` — a test handler for `Resource(GPU)` that routes all GPU
  allocations to CPU.

The supplied-noise helper is ordinary pure function composition. Effect-handler
substitutions remain governed by the actual effects they handle.

Implementation: standard library functions in `Std.Test` (or documented patterns in the
SKILL.md if the functions are trivial). Not a compiler change — library code plus
documentation.

### Post-POPL: Typing Rules in Documentation

After the LaCaDiLE POPL paper is submitted and the typing rules are finalized, publish
them as a reference appendix in the Chelis mdBook. Users can look up the precise rule
for any construct. This falls out naturally from the POPL paper — the typing-rule
figures are already typeset in LaTeX and can be rendered in the book. Not a separate
work item; just a "copy the figures into the docs" step after submission.

### Acceptance Oracle

All SKILL.md examples validated via `skill_suite.rs` against the current compiler. Eval
on target base models to measure improvement over SKILL.md v1.

---

## Shell Ecosystem (Phase 3 Sub-Phases)

The domain shells are now proper Phase 3 sub-phases (3j, 3k, 3l) with full
specifications above, not deferred post-phase work. The tier structure is:

- `chelis-std` (core) — standard library shipped as the first Reef package
- `nautilus` (3j) — general numerical methods on top of `chelis-std`
- `coral` (3k) — typed dataframes on top of `chelis-std`
- `shoals` (3l) — finance on top of `chelis-std` + `nautilus` + `coral`

Three further shells are named and reserved but scoped as stubs beyond Phase 3:

- `school` — classical ML (scikit-learn competitor). Depends on `chelis-std` + `nautilus`
  + `coral`.
- `darwin` — evolutionary algorithms (GA, genetic programming over the Deep AST, ES,
  PBT, NAS). Depends on `chelis-std` + `nautilus`; optionally uses `coral` for evolving
  feature-engineering pipelines over tabular data.
- `hull` — executable language specification. Depends on `chelis-std` only. Provides a
  self-hosted reference type checker, reference evaluator, and spec-driven random
  program generation over Deep AST ADTs so the compiler can validate the spec and the
  spec can validate the compiler. Blocked on LaCaDiLE rule finalization, a Deep parser
  in Chelis, and `chelis prove` infrastructure. Phase 4/5 item, not a Phase 3 sub-phase.
- `hydrostatic` — automated static analysis on the tensor DAG. Depends on `chelis-std`
  + the compiler's DAG IR. Computes over-approximations of value ranges at each node
  to detect division by zero, overflow, NaN propagation, and unbounded outputs without
  user annotations. Pre-deployment gate (minutes, not milliseconds). Trust stack
  Level 3. Future stub, not designed.
- `beacon` — IR-native bound-propagation verification engine. Depends on `chelis-std`
  + the compiler's DAG IR. Proves sound output bounds by forward propagation through the
  RISC DAG; the same engine bounds finance pricing graphs and neural networks (via
  Hydronnx). Plugs into the verification orchestrator through the discharge-engine
  interface. Future stub; design in `verification_stack_master_plan.md` and `beacon_plan.md`.

`chelis prove` scope has expanded from a CLI-flag property testing tool to first-class
executable properties with `@property` annotations. See `chelis_trust_stack.md` for the
full design. Implementation remains deferred until after the RLVR pipeline (Phase 4)
but the design is locked.

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
  (the embedding wrapper since moved to `School.Nn.Embedding` in chelis-std 0.4.0)

**Rust runtime rewrite (`3m`):**

- no active docs or build/test paths still rely on `chelis_runtime.c`
- generated host C no longer peeks into non-tensor runtime struct fields
- `chelis build` stages only the runtime the CLI carries and rejects a set
  `CHELIS_RUNTIME_DIR` (`spec/08-backends.md` §2.1)
- mixed-program compiled execution matches `chelis eval` on both C and HIP paths

**Data loading/tokenization (`3g`):**

- text/CSV/JSON loading returns the documented structures
- read_bytes returns correct values for binary files
- memory-mapped I/O: mmap_file opens, mmap_read returns correct bytes at offset,
  mmap_len matches file size, IO effect on mmap_file only (reads are pure)
- tokenizer encode/decode is deterministic against the documented assets
- batching/padding produces the expected tensor shapes and values

**Standard library expansion (`3i`):**

(The ML modules in this block — `Std.Nn.Generate`, AdamW/LAMB optimizers, schedulers —
since moved to `School.*` in chelis-std 0.4.0; `Std.Time` / `Std.Decimal` stayed.)

- `Std.Time` and `Std.Decimal` stay standard-library scoped rather than leaking
  compiler-intrinsic assumptions
- generation keeps the pure greedy path (`generate`) distinct from the keyed
  sampled path (`generate_with`)
- temperature + top-k + top-p sampling replays across runs with the same initial key
- decoupled AdamW weight decay remains observable even with zero gradients
- cosine_with_warmup matches expected edge points
- at least one pure package-mode 3i program builds to C, links, runs, and matches
  `chelis eval`
- examples/docs do not overclaim backend or tensor-kernel relevance for host-value
  modules

**Release infrastructure + Std surface expansion (`3j-pre`):**

(The `Std.Nn.*` / `Std.Loss.*` surface validated in this block since moved to
`School.Nn.*` / `School.Loss.*` in chelis-std 0.4.0; `Std.Init` stayed.)

- `chelis-lang` GitHub organization exists and is reserved
- compiler release binary builds and downloads cleanly in CI
- `scaled_dot_product_attention`, MHA, and GQA produce correct forward/backward
  numerics against a PyTorch reference within tolerance
- attention rejects mismatched head/seqlen/mask shapes at type-check time where the
  shapes are static, and at runtime otherwise
- `Std.Init` initializers hit target mean/variance on statistical tests
- `GELU` / `SiLU` / `RMSNorm` / `Conv1d` / `Conv2d` match the torch reference
- `Std.Loss` `KLDivergence` / `BCEWithLogits` / `accuracy` / `perplexity` match reference
  implementations on documented test cases

**Numerical methods (`3j` Nautilus):**

- `Nautilus.Stats`, `Nautilus.Distributions`, and `Nautilus.Special` agree with scipy
  reference values within tolerance (P0 tier)
- `Nautilus.LinAlg` matches scipy on SVD, Cholesky, QR, LU, solve, eig for a battery of
  random + structured matrices; nalgebra bridge is covered end-to-end
- hand-written adjoints for `svd`, `cholesky`, `solve`, `qr`, `eig` agree with finite
  differences on non-degenerate inputs
- `grad` through ODE solvers produces correct gradients for neural ODE composition
- `Nautilus.Signal` stub has correct type signatures but clearly errors at runtime
- `Nautilus.Optim` solvers converge on well-conditioned problems and reject ill-conditioned
  inputs
- `Nautilus.Distance` (Euclidean, cosine, Mahalanobis, Manhattan) matches scipy
- `normal_like` Box-Muller sampling passes a Kolmogorov-Smirnov test against Normal(0,1)

**Dataframes (`3k` Coral):**

- filter/group-by/join produce correct results against pandas reference
- NaN operations (`is_nan`, `fill_nan`, `drop_nan`) match pandas reference on mixed
  float/int data; GPU kernels propagate NaN correctly through fused pipelines
- rolling window operations (`rolling_mean`, `rolling_std`, `ewm`) match pandas reference
- Parquet round-trip via `parquet2` preserves schema and typed column contents
- AD flows through filter → aggregate pipelines
- GPU compilation of numeric column operations works (manual gate)
- wrong column names and type mismatches produce clear errors

**Finance (`3l` Shoals):**

- Black-Scholes price matches analytical formula
- Greeks via `grad` match analytical Greeks
- Monte Carlo converges to analytical for vanilla options
- same seed produces identical prices across runs

**Teaching surface (`3f`):**

- `SKILL.md` teaches the real executable language, not a stale tensor-only subset
- examples align with the package/style/python foundations already shipped
- coverage includes domain shells (Nautilus, Coral, Shoals) and mentions the `school`
  (classical ML) and `darwin` (evolutionary algorithms) stubs

---

## Phase 3 Dependency and Size Summary

| Sub-phase | Size | Dependencies | Nature |
|---|---|---|---|
| `3e`: Pipe-first style pass | shipped | none | Engineering |
| `3a`: Package system | shipped | `3e` | Engineering |
| `3b`: Python FFI interop core | shipped | `3a` | Engineering |
| `3b-ii`: Direct execution + NumPy | shipped | `3b` | Engineering |
| `3c`: Scalar and string foundation | shipped | `3e` | Engineering |
| `3d`: Collections and iteration | shipped | `3c` | Engineering |
| `3h`: Core numeric primitives | shipped | `3d` | Engineering (RISC ops + AD + backends) |
| `3m`: Rust runtime rewrite | shipped | `3h`, `3d` | Engineering (runtime ABI + codegen + CLI/build) |
| `3g`: Data loading and tokenization | shipped | `3h`, `3m` | Engineering (I/O + pure Chelis libraries) |
| `3i`: Std library expansion | shipped | `3h`, `3g` | Pure Chelis library (Time, Decimal, Generate, AdamW, Schedule) |
| `3j-pre`: Release infra + Std surface expansion | shipped | `3i` | Engineering (CI release, GitHub org, Std.Nn/Std.Loss/Std.Init additions) |
| `3j`: Nautilus | shipped | `3h`, `3i`, `3j-pre` | Downstream Reef shell with nalgebra-backed LinAlg and numerical methods |
| `3k`: Coral | medium | `3h`, `3d`, `3g`, `3j-pre` | Pure Chelis library (dataframes, NaN handling, rolling windows, Parquet via `parquet2` FFI) |
| `3l`: Shoals | medium | `3j`, `3k`, `3i` | Pure Chelis library (finance) |
| `3t`: Native Testing | medium | Bug 9 fix, fast eval | Chelis library (`Std.Test`) + CLI (`chelis test` command) + test migrations for Nautilus and Coral |
| `3f`: SKILL.md v2 | small | all above | Documentation |
| Hydrostatic (future) | TBD | `chelis-std` + DAG IR | Automated static analysis shell (value range inference, hazard detection). **Future**, not in Phase 3. |
| Beacon (future) | TBD | `chelis-std` + DAG IR | IR-native bound-propagation verification engine (output bounds, bounded Greeks, NN verification). **Future**, not in Phase 3. |

This phase is intentionally pragmatic. The remaining work is now shell ecosystem and
testing work rather than the earlier compiler/runtime foundations. `3k` is the next
unshipped shell. `3l` depends on `3j` and `3k`. `3n` can proceed against `3j`, while
`3o` waits for both `3n` and `3l`. `3t` gives reef packages a native test surface.
`3f` goes truly last because it must cover the complete ecosystem including the domain
shells.

`school` (classical ML, sklearn competitor), `darwin` (evolutionary algorithms), `hull`
(executable language specification), `hydrostatic` (automated static analysis on the
tensor DAG), and `beacon` (IR-native bound-propagation verification) are post-Phase-3
shell stubs and do not appear as Phase 3 sub-phases.

---

## Phase 3 Completion Check

Before calling Phase 3 complete:

- `3a`, `3b`, `3b-ii`, and `3e` remain honest shipped foundations
- `3c` provides practical scalar/string programming without Python fallback
- `3d` provides collections and iteration for variable-length host-side data
- `3h` provides the expanded tensor-language surface needed for real model code
- `3m` provides a Rust-owned compiled runtime so later host-language/library work does
  not keep expanding the old C runtime
- `3g` provides text/config/data loading plus tokenizer and batching support
- `3i` provides `Std.Time`, `Std.Decimal`, `Std.Nn.Generate` (KV cache), AdamW/LAMB
  optimizers, and `Std.Schedule` as practical standard-library modules
- `3j-pre` provides the `chelis-lang` GitHub org, a compiler release binary, and the
  expanded `Std.Nn`/`Std.Loss`/`Std.Init` surface (GELU, SiLU, RMSNorm, Conv1d/2d,
  attention, GQA, KL, BCEWithLogits, Kaiming/Xavier/trunc_normal)
- `3j` provides numerical methods (special functions, distributions, nalgebra-backed
  linear algebra with AD, convex optimization, ODE/SDE, distances) as the `nautilus`
  Reef package
- `3k` provides typed dataframes with AD through tabular operations, NaN handling,
  rolling windows, and Parquet I/O as the `coral` Reef package
- `3l` provides finance-specific pricing, risk, and stochastic process tools as the
  `shoals` Reef package
- `3t` provides `Std.Test` and the `chelis test` CLI command; Nautilus and Coral test
  suites are migrated so that mathematical identity / property / smoke tests run via
  `chelis test` and only scipy/pandas parity checks remain in Python under `parity/`
- `3f` reflects the full post-shell language in `SKILL.md` and examples, including
  `Nautilus.*`, `Coral.*`, and `Shoals.*` package imports and mentions of the `school`
  (classical ML) and `darwin` (evolutionary algorithms) stubs
- a pure Chelis program can read text, tokenize it, batch/pad it, run a model, compute
  loss and gradients, and print results without Python
- domain shells compose correctly: `shoals` depends on `nautilus` + `coral`, all build
  and import through the Reef pipeline
