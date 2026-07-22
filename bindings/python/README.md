# Chelis Python Bindings

Phase `3b` / `3b-ii` Python bindings for the shared Chelis compiler API, DLPack tensor
interop, and direct compiled execution via `chelis.compile_and_load(...)` /
`chelis.load(...)`.

Current guarantees:

- `chelis.check(...)`, `compile(...)`, `desugar(...)`, `decompile(...)`, `validate(...)`
  share the same compiler implementation as Tide through `chelis-compiler-api`
- `chelis.from_dlpack(...)` is CPU-only; unsupported GPU tensors fail as `ValueError`
- `chelis.eval(...)` may copy inputs into the evaluator's internal `Vec<f64>`
  representation in `3b`
- `chelis.compile_and_load(...)` is the product path for compiled execution;
  `chelis.load(...)` is the advanced path for existing artifacts
- compiled execution currently supports fully concrete `float32` tensors
- `ChelisError` reports compiler/build/runtime failures; `ValueError` reports bad Python
  arguments such as wrong dtype, wrong shape, or unsupported device placement
- native compile/build and compiled host/device execution release the Python GIL

Entry selection for `compile_and_load` (chelis#817 / chelis#818):

- `entry_name=` selects the `def` to compile and expose as the callable model. The
  loaded `CompiledModel`'s `input_names` / `output_names` are scoped to that def, not
  the union of every def in the file.
- Default selection when `entry_name` is omitted: a tensor-signature `def main` is
  preferred; with no `main` and exactly one tensor-signature def, that def is used;
  with two or more tensor-signature defs and no `main`, `compile_and_load` errors and
  lists the candidates so you can pass `entry_name`. An `entry_name` that names no def
  in a clean tensor program is likewise a loud error listing the available defs.
- Emitted symbol scheme: the artifact's C entry symbol is always the fixed
  `chelis_main` (each artifact is scoped to one entry def, so one symbol suffices and
  is collision-free), so a def named `main`, `free`, `malloc`, or a runtime
  `chelis_*` name still compiles and links. The manifest's `host_entry_name` carries
  this symbol; the loader resolves it via `dlsym`.
- Unused parameters (pre-existing behavior, not changed here): a declared parameter
  that never appears in the def's body is dead-code-eliminated and does NOT appear in
  the manifest `input_names`. For example `def solve(a, b) = f(b)` exposes
  `input_names == ("b",)`. Positional callers must therefore bind by consulting
  `input_names` rather than assuming declaration order, or a positional argument can
  misbind. Tracking the ergonomics of this is out of scope for the entry-scoping fix.
- Some entries genuinely require the host-program lane and cannot be exposed as a
  callable tensor kernel — programs built from top-level bindings/globals, and defs
  that use `grad`/`vmap` or string/record/effect operations. `compile_and_load` fails
  loudly for these; select a tensor-in/tensor-out `def` with `entry_name=` instead.

Reef dependency resolution (chelis#816):

- `compile_and_load(..., project_root=)` and `eval(..., project_root=)` resolve
  imports of reef-declared dependencies (`import Shoals.Pricing (bs_call_scalar)`) by
  compiling the source against the reef package's linked library context, instead of
  failing with `unbound variable`. Without a root the source is compiled/evaluated
  self-contained, as before — with the single exception in the **Rank-0 scalars** bullet
  below (a rank-0 tensor entry is now rejected with wrap guidance on every path).
- Root selection differs by entry point:
  - `compile_and_load` **auto-discovers** the enclosing reef package by walking up from
    `source_path` (looking for `reef.toml`) — but **only when the source actually
    contains an `import` declaration**. An import-free (self-contained) source, or any
    non-Surf source, takes the bare path as before (save the rank-0 rejection below),
    so a self-contained file
    that happens to sit inside a reef project neither pays the project's
    context-compile cost nor is coupled to a broken sibling file. Pass a
    `project_root=` path to force in-context resolution regardless of imports, or
    `project_root=False` to force the bare path even for an importing source. A
    `project_root` with no `reef.toml` is a loud error naming `project_root=`.
  - `eval` takes raw text with no file to walk from, so it does **not** auto-discover:
    pass `project_root=` explicitly, or omit it (or pass `False`) for the
    self-contained path.
  - Reef imports are a Surf-only construct: `source_kind="deep"` never routes
    in-context (an explicit `project_root=` with a deep source is rejected).
  - Default in-context entry selection prefers a tensor def named `main` (matching the
    monolithic path), so a multi-def file behaves the same inside and outside a project.
- `CHELIS_REEF_HOME` keys the on-disk context cache the same way the CLI uses it; the
  first build of a package's library context is slow (tens of seconds to minutes),
  subsequent calls hit the cache.
- **Scalar entries.** A def with a scalar signature (`def main(s: f32, ...) -> f32`)
  has no callable tensor kernel; `compile_and_load` rejects it with guidance to wrap
  scalars as rank-1 tensors (`tensor[1, f32]`). The same program runs through `eval`,
  which supports scalar and host-only entries.
- **Mangled names.** Roots that come from the reef **library** graph carry
  linker-mangled names (`pkg__<pkg>__<Module>__<name>`), matching the CLI (no
  demangling); entries from the package's **own** source keep their bare names. A
  compiled entry's `input_names` are its own parameter names; consult them (do not
  assume order).
- **Rank-0 scalars.** A tensor-in / scalar-out entry (e.g. a reduce to a `tensor[f32]`)
  is rejected on every path — including the bare self-contained path — with the same
  wrap-as-`tensor[1, f32]` guidance, rather than emitting an unbuildable scalar kernel.
- **C-target only.** Reef-context resolution is supported only for `target="c"`. A
  `target="hip"` compile with a `project_root=` (or auto-discovered root) is rejected as
  unsupported, because the HIP backend does not yet apply the entry-scoped DAG selection
  the C path uses and would otherwise merge every reef-linked def into one kernel. Use
  `target="c"`, or run through `eval`. Tracked as chelis#829.

Compiler selection:

- native C compilation resolves through Chelis's shared platform toolchain:
  `clang` + Accelerate on macOS, `gcc` + OpenBLAS on Linux; override with `CHELIS_CC`
- native HIP compilation prefers `/usr/bin/hipcc` when present; override with
  `CHELIS_HIPCC`
