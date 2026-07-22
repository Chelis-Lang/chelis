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
- Emitted symbol scheme: the artifact's C entry symbol is always `chelis_`-prefixed —
  `chelis_main` for the default entry, `chelis_<def>` for a selected def — so a def
  named `main`, `free`, or `malloc` still compiles and links. The manifest's
  `host_entry_name` carries this symbol; the loader resolves it via `dlsym`.
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

Compiler selection:

- native C compilation resolves through Chelis's shared platform toolchain:
  `clang` + Accelerate on macOS, `gcc` + OpenBLAS on Linux; override with `CHELIS_CC`
- native HIP compilation prefers `/usr/bin/hipcc` when present; override with
  `CHELIS_HIPCC`
