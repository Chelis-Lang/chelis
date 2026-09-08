# ffi

## Purpose

Define the Chelis foreign-function-interface direction: the phased Python interop (interop core,
direct execution with the NumPy guarantee, JAX), C interop via generated headers plus the Rust
static runtime and the exported-`main` symbol-rename rule, and compiler-crate embedding. This is
the current and directional truth of how Chelis interoperates with host languages.

**Source:** captured from [`spec/11-ffi.md`](../../../spec/11-ffi.md).

## Requirements

### Requirement: Python interop core (Phase 3b)

The Phase 3b Python interop core SHALL expose compiler-facing entry points via PyO3 bindings
through a shared `chelis-compiler-api` crate, install via `uv pip install ./bindings/python`,
provide CPU-only DLPack interop with PyTorch and safetensors round-trip helpers, and release the
GIL during compiler/evaluator work. `ChelisError` SHALL be reserved for compiler/build/runtime
failures while Python-side data mismatches SHALL surface as `ValueError`.

#### Scenario: CPU DLPack interop with PyTorch

- **WHEN** a PyTorch CPU tensor is passed through the 3b interop
- **THEN** it interops via DLPack as the tested guarantee, releasing the GIL during compiler/evaluator work

#### Scenario: GPU tensor is a Python-side ValueError

- **WHEN** an unsupported GPU tensor is passed in the 3b cut
- **THEN** it is rejected as a Python-side `ValueError`, deferred to 3b-ii, rather than raising `ChelisError`

### Requirement: Direct execution and NumPy guarantee (Phase 3b-ii)

Phase 3b-ii SHALL provide `chelis.compile_and_load("model.ch")` as the compiled-execution product
path and `chelis.load("model.so")` as the advanced loader, with a sidecar manifest recording the
source path and content hash. `load()` SHALL warn when the source has changed since compilation.
The NumPy DLPack guarantee SHALL be documented and tested; compiled execution SHALL currently be
limited to fully concrete `f32` tensors.

#### Scenario: compile_and_load runs a compiled artifact

- **WHEN** `chelis.compile_and_load("model.ch")` is called
- **THEN** it compiles and loads the artifact for direct execution, verified for PyTorch CPU tensors and NumPy arrays

#### Scenario: load warns on changed source

- **WHEN** `chelis.load("model.so")` is called and the sidecar manifest's content hash no longer matches the source
- **THEN** it emits a warning that the source has changed since compilation

### Requirement: C interop surface

Generated C headers and runtime support SHALL make it possible to call compiled Chelis artifacts
from C or C++. After Phase 3m the C-facing surface SHALL come from `chelis_runtime.h` plus the
shipped Rust static runtime library rather than a generated `chelis_runtime.c`. When object-mode
host emission exports a source-level `def main(...)`, the generated C symbol SHALL be renamed to a
file-stem-derived helper (e.g. `<program>__main`) so a C/C++ driver can define its own
`main(void)`.

#### Scenario: Compiled artifact callable from C

- **WHEN** a compiled Chelis artifact is linked into a C program
- **THEN** it is callable via the generated headers plus the Rust static runtime library

#### Scenario: Exported main is renamed to avoid collision

- **WHEN** object-mode host emission exports a source-level `def main(...)`
- **THEN** the generated C symbol is renamed to a file-stem-derived helper so the driver can define its own process entry `main(void)`

### Requirement: Compiler embedding

The Rust crates SHALL remain usable as libraries so Tide, editor tooling, and external
integrations can embed compiler functionality directly rather than shelling out to the CLI.

#### Scenario: Tide embeds the compiler crates

- **WHEN** Tide or editor tooling needs compiler functionality
- **THEN** it embeds the Rust crates directly rather than shelling out to the CLI

#### Scenario: Embedding avoids CLI subprocessing

- **WHEN** an external integration uses compiler functionality
- **THEN** it can link the crates as libraries instead of spawning the CLI as a subprocess
