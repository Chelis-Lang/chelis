# Chelis Python Bindings

Phase `3b` / `3b-ii` Python bindings for the shared Chelis compiler API, DLPack tensor
interop, and direct compiled execution via `chelis.compile_and_load(...)` /
`chelis.load(...)`.

Current guarantees:

- `chelis.check(...)`, `compile(...)`, `desugar(...)`, `decompile(...)`, `validate(...)`
  share the same compiler implementation as Tide through `chelis-compiler-api`
- `chelis.from_dlpack(...)` is CPU-only; unsupported GPU tensors fail as `ValueError`
- `chelis.eval(...)` copies inputs through execution wire v4's exact dtype
  carriers. Float payloads use fixed-width lowercase IEEE bit strings, preserving
  signed zero and every NaN payload. Scalars return their declared NumPy integer
  or float type; `bf16` uses `ml_dtypes.bfloat16`. Tensor results carry the dtype
  in `TensorValue.dtype`, with exact integer values or own-width float scalars in
  `TensorValue.data`. Calling `float(value)` explicitly converts a result.
- NumPy `uint8`, `uint16`, and `uint32` inputs widen exactly to `int16`, `int32`,
  and `int64`; unsupported widths such as `uint64` require an explicit caller
  conversion. Native and nonnative byte order, strided arrays, and rank-zero
  arrays preserve their stored bits.
- Execution results require schema version 3 before any value is decoded.
  Missing or other versions, malformed dtype carriers, invalid shapes, and
  mismatched element counts raise `ValueError`.
- `chelis.compile_and_load(...)` is the product path for compiled execution;
  `chelis.load(...)` is the advanced path for existing artifacts
- compiled execution currently supports fully concrete `float32` tensors
- `ChelisError` reports compiler/build/runtime failures; `ValueError` reports bad Python
  arguments such as wrong dtype, wrong shape, or unsupported device placement
- native compile/build and compiled host/device execution release the Python GIL
- the JSON entry points (`check`, `desugar`, `decompile`, `compile`, `eval`,
  `validate`) run on a worker thread and stay responsive to Ctrl-C: SIGINT
  raises `KeyboardInterrupt` promptly instead of waiting for the call to
  finish (chelis#914, chelis#930). The worker is always joined, never
  detached, so a cancelled call leaves no work running behind it.

  The **front end** is cancellable too (chelis#930): parse, desugar,
  type-check and lowering poll the same token at every phase boundary, and
  at every top-level declaration inside the passes whose cost scales with
  declaration count. A SIGINT arriving during compilation therefore
  abandons the compile instead of waiting it out. Measured on a 70 KB
  source with a ~19 s front end: interrupting 2 s in used to return after
  17.3 s (the *remaining compile time*) and now returns in tens of
  milliseconds.

  In both phases, latency is bounded by the longest single uninterruptible
  step, not by how much work remains. Two shapes make that step large:

  - during evaluation, a program holding one very large intermediate
    value, where allocating or freeing it is itself one step —
    interrupting a fold over a 40M-element list takes a few seconds,
    because that is how long the list takes to tear down;
  - during compilation, one very large top-level declaration, since a
    declaration is the grain at which the front end polls. Many
    declarations are cheap regardless of how many there are; one
    pathological declaration is not.

  Not everything on the path polls. The Reef graph resolution that precedes
  a package compile, and lowering's whole-program walk, run to completion
  once entered.

  Acceptance probe: `bindings/python/tests/manual_eval_interrupt.py`.

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
    `project_root` with no `reef.toml` is a loud error naming `project_root=`, and an
    empty/whitespace `project_root=""` is rejected outright (it would otherwise probe
    for `reef.toml` relative to the process CWD). When auto-discovery finds no root
    and the bare compile of an importing source then fails, the error carries a hint
    that discovery came up empty and that `project_root=` names the remedy.
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
- **Top-level globals (entry-scoped semantics).** In-context compilation is
  entry-scoped: a top-level (non-`def`) binding in the compiled source (e.g.
  `glb = 2.0` next to `def main`) does not block compilation, and the compiled
  artifact runs only the selected entry — an unreferenced sibling global's
  computation is not part of it. This is deliberately more permissive than the
  bare/monolithic path, where top-level bindings decline the entry lane and keep
  whole-program host-lane routing. Use `eval` when the sibling globals'
  computations matter.
- **Selectable entries.** Only the defs in the compiled source itself are
  selectable as entries, by their bare names; imported library defs are callable
  from the entry's body but are not themselves selectable via `entry_name=`. A
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

## Editable development installs

From the repository root, `uv pip install -e bindings/python` builds an editable,
checkout-backed extension. It stages the development runtime and checks the
declared runtime sources before each stage; changing a declared source without
rebuilding the extension raises `ChelisError`. `maturin develop` uses the same
unsealed development configuration.

## Build and smoke a distribution wheel

The local PEP 517 backend seals standard wheel builds with `sealed-runtime`.
The default Maturin feature remains only `extension-module`, so editable builds
do not inherit the distribution-only feature.

To build a wheel from the repository root:

```sh
CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0 \
CARGO_TARGET_DIR="$PWD/target/python-wheel" \
uv build --wheel --out-dir target/python-wheel/wheels bindings/python
```

For the bounded end-to-end smoke, run the smoke directly instead of first
building a wheel. It performs one standard wheel build from a disposable source
copy, removes that copy, installs the wheel outside the checkout, and exercises
compiled execution and persisted reload in separate Python processes:

```sh
CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0 \
CARGO_TARGET_DIR="$PWD/target/python-wheel-smoke/cargo" \
.venv/bin/python bindings/python/tests/python_wheel_smoke.py \
  --receipt target/python-wheel-smoke/cargo/wheel-smoke.json
```

If a wheel was already built, pass `--wheel <path>` to consume it without
building again. That mode verifies installed consumer behavior but does not
prove the wheel was built without its source copy. The optional
`--crossed-bundle` builds a synthetic second sealed wheel after changing the
exported key-seed operation in its disposable runtime source. A native C program
links each wheel's staged archive and requires seed 7 to yield key bits 7 in A
and 8 in B; the first wheel must reject B's compiled artifact. The smoke records
runtime and library digests, exact values, process results,
and negative controls; on failure it retains command logs and receipts under
the Cargo target directory. This is a bounded Python distribution check, not
the aggregate runtime-artifact oracle.

On macOS, installed-wheel subprocesses also run under an OS sandbox that
denies reads from the original checkout. The receipt records a denied read
of its `Cargo.toml` for native compile/call, persisted reload, and crossed
bundle rejection. On Linux, the disposable source copy is removed and the
developer target withheld, but the original checkout remains readable;
`checkout_read_denied: false` does not claim filesystem isolation.
`source_free_build_proven` is true only when this driver built the wheel and
every installed consumer verified that checkout read denial.
