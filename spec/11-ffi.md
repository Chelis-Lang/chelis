# Foreign Function Interface

## 1. Python Compiler API

Python bindings expose compiler-facing entry points through the shared
`chelis-compiler-api` boundary. They:

- install as the `chelis` Python package;
- accept CPU tensor interchange through DLPack;
- provide safetensors round-trip helpers;
- release the GIL during compiler, evaluator, build, and native-execution work; and
- use explicit Python wire models rather than exposing Rust implementation types.

`ChelisError` represents compiler, build, and Chelis runtime failures. Invalid Python
arguments, incompatible tensor devices, unsupported Python dtypes, and malformed
interchange objects raise `ValueError`.

The evaluator may copy a Python tensor into its internal representation, but that copy
must preserve the source dtype's value semantics. It may not funnel integer or
wide-floating data through an untagged `f64` or `f32` carrier.

## 2. Compiled Python Execution

`chelis.compile_and_load("model.ch")` is the product path for compilation and direct
execution. `chelis.load("model.so")` is the advanced artifact loader.

A sidecar manifest records the source path and content hash. Loading an artifact whose
source hash has changed emits a stale-source warning.

NumPy and CPU PyTorch arrays use DLPack-compatible typed marshalling. The accepted dtype
set is derived from the selected backend's declared capability; the binding must derive
element width and NumPy/DLPack dtype from the tagged Chelis dtype. It must never assume
that compiled tensors are `f32`.

Before entering native code, supplied shapes are validated against the function's typed
shape contract, including symbolic and runtime shape expressions. A rejected shape,
dtype, or device produces a Python-side error rather than truncation, implicit copy to
an incompatible device, or an ABI mismatch.

JAX interchange uses the same typed DLPack principles when paired with the StableHLO
integration target.

## 3. C And C++ Interop

Generated headers and the Chelis static runtime make compiled artifacts callable from C
and C++. The public runtime surface comes from `chelis_runtime.h` plus the static
runtime library; generated programs do not publish a separate handwritten runtime
implementation.

The exported function ABI is the backend ABI in `spec/08-backends.md`. Public
declarations are configuration-invariant and every numeric value crosses the boundary
through an attributable tagged representation.

When object-mode host emission exports a source-level `def main(...)`, its generated C
symbol is renamed to a file-stem-derived helper such as `<program>__main`. This allows
the C or C++ driver to define its own process entry `main(void)`.

## 4. Rust Embedding

Compiler crates remain usable as libraries. Tide, editor tooling, Python bindings, and
external integrations embed these crates directly rather than shelling out to the CLI.

Every embedding surface observes the same parse, check, lower, compile, diagnostic,
numeric, and unsupported-case contracts as the CLI. An embedding API must not bypass a
validation or rejection boundary merely because it avoids command-line dispatch.
