# Foreign Function Interface

**Status:** Outline for later phases.
Chelis does not depend on FFI work for Phase 0 completion, but the expected direction is
already clear enough to record.

## 1. Python Interop

The Phase 3 Python path is split into two cuts:

### Phase 3b: Interop Core

- PyO3 bindings for compiler-facing entry points through a shared
  `chelis-compiler-api` crate
- install surface: `uv pip install ./bindings/python`
- CPU-only DLPack interop with PyTorch as the tested guarantee
- safetensors read/write helpers that round-trip with PyTorch
- GIL release during compiler/evaluator work
- `ChelisError` is reserved for compiler/build/runtime failures
- Python-side data mismatches such as unsupported GPU tensors in this cut are surfaced as
  `ValueError`
- `chelis.eval(...)` may copy Python tensor inputs into the evaluator's internal
  `Vec<f64>` representation in this cut
- GPU tensors are rejected as Python-side `ValueError`s and deferred to `3b-ii`

### Phase 3b-ii: Direct Execution + NumPy Guarantee

- `chelis.compile_and_load("model.ch")` as the product path for compiled execution
- `chelis.load("model.so")` as the advanced loader for an existing artifact
- sidecar manifest (`model.json`) recording source path + content hash, with a warning on
  `load()` if the source has changed since compilation
- direct loading/calling of compiled Chelis artifacts from Python
- CPU direct execution verified for PyTorch CPU tensors and NumPy arrays
- HIP device ABI emitted for direct GPU execution, with the Python GPU bridge layered on
  that ABI
- GIL release during native compile/build and compiled host/device execution
- NumPy DLPack guarantee as a documented/tested promise
- compiled execution currently limited to fully concrete `f32` / `f64` tensors on the C
  target, and to fully concrete `f32` tensors on the HIP target (chelis#919, chelis#920).
  The per-target admit-list is `supported_execution_dtypes` in `chelis-python`; the
  marshalling layer derives the NumPy dtype and the DLPack element width from it rather
  than assuming f32, and rejects any dtype it cannot describe
- reef dependency resolution via `project_root=` on `compile_and_load` and `eval`
  (chelis#816): with a reef package root, imports of reef-declared dependencies resolve
  against the package's linked library context instead of failing with `unbound
  variable`. `compile_and_load` auto-discovers the root by walking up from the source
  file, but only when the (Surf) source contains an `import` declaration; an import-free
  or non-Surf source, or `project_root=False`, takes the bare self-contained path. An
  explicit `project_root=` path forces in-context resolution regardless. `eval` (raw
  text) requires an explicit `project_root=`. With no applicable root the bare
  self-contained behavior is unchanged. Default in-context entry selection prefers a
  tensor def named `main`. A scalar-signature entry has no callable tensor kernel and is
  rejected with tensor-wrap guidance; `eval` runs it. Roots sourced from the reef
  library graph keep their linker-mangled names; entries from the package's own source
  keep their bare names.

### Phase 5a

- JAX DLPack guarantee alongside the StableHLO backend

This is a Phase 3 interoperability feature, not a Phase 0 requirement.

## 2. C Interop

Generated C headers and runtime support should make it possible to call compiled Chelis
artifacts from C or C++.
That interoperability follows naturally from the reference backend and does not require a
separate host-language embedding model first.
After Phase `3m`, that C-facing surface is expected to come from `chelis_runtime.h`
plus the shipped Rust static runtime library rather than a generated `chelis_runtime.c`
implementation file.
The compiler-api pipeline behind this surface serves two products with different
entry contracts:

- **The callable surface** (`compile_for_execution`, backing Python's
  `compile_and_load`) treats `entry_name` STRICTLY as a def selector. An unknown
  `entry_name`, an ambiguous default (multiple tensor defs, none named `main`),
  a program with top-level (non-`def`) value bindings, or a `grad`/`vmap`
  transform entry that cannot be entry-scoped is a loud error; this surface
  never returns metadata merged from every def (the chelis#817 defect class)
  and never silently ignores the requested entry.
- **The C-source surface** (`compile`, backing tide's `/compile`, cove's live
  pane, and Python's `chelis.compile()`) keeps the legacy whole-program
  contract: with no unambiguous entry it emits the whole program, and an
  `entry_name` naming no def is the sanitized OUTPUT SYMBOL, not a selector
  error. When the entry lane does claim a program (an unambiguous tensor
  entry), both surfaces emit the same entry-scoped kernel.

When compiled-execution emission scopes an artifact to a selected `def`, the
emitted C symbol is decoupled from the def name: the artifact always emits the
fixed symbol `chelis_main`. Because each artifact is scoped to exactly one entry
def there is exactly one emitted entry per translation unit, so a single fixed
symbol suffices and is collision-free by construction — a def literally named
`main` no longer redefines the reserved process entry
`int main(int, char**, char**)`, a def named after a libc symbol (`free`,
`malloc`) no longer collides at link time, and neither does a def named after a
runtime symbol in the `chelis_*` namespace (`chelis_runtime.h` declares
`chelis_free`, `chelis_tuple_get`, …). The artifact manifest's `host_entry_name`
carries `chelis_main` so the loader (`dlsym`) and generated header stay consistent.

Outside the entry-scoped lane — the host-program lane that owns top-level globals
and scalar/`grad` entries, the free-form pure-DAG path taken by a program that
lowers no host program (such as a single fully-DAG-lowerable `def`), and the
C-source surface's whole-program fallback when the entry lane declines —
`entry_name` becomes the output symbol after sanitization only: `main` maps to
`chelis_main`, non-identifier characters map to `_`, and a digit-leading or
empty name gains a `chelis_` prefix. Any other name passes through unchanged, so
these paths guard neither libc nor the runtime's own `chelis_*` namespace: a
single-def pure program whose def is named `free` still emits `void free(...)`,
and an `entry_name` of `chelis_free` is emitted verbatim. Those are pre-existing
gaps of the legacy symbol mapping, accepted on the C-source surface where the
caller owns the symbol choice; the strict callable surface is immune because it
always emits `chelis_main`.

The `chelis build` object-mode lane is a separate emitter with its own symbol
rule, unchanged by the above: when object-mode host emission exports a
source-level `def main(...)`, the generated C symbol is renamed to the
file-stem-derived `<program>__main` so downstream C or C++ drivers can still
define their own process entry `main(void)`; other def names route through the
chelis#840 `c_ident` mapping (`emitted_function_name` in
`chelis-backend-c/src/host_emit.rs`).

## 3. Embedding the Compiler

Longer term, the Rust crates should remain usable as libraries so Tide, editor tooling,
and external integrations can embed compiler functionality directly rather than shelling
out to the CLI.
