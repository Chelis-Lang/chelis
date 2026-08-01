# ffi

## Purpose

Define Python typed interchange and compiled execution, C/C++ interop through generated headers
and the static runtime, the exported-`main` symbol-rename rule, and compiler-crate embedding.

**Source:** captured from [`spec/11-ffi.md`](../../../spec/11-ffi.md).

## Requirements

### Requirement: Python interop core

Python interop SHALL expose compiler-facing entry points via PyO3 bindings
through a shared `chelis-compiler-api` crate, provide typed CPU DLPack interop with PyTorch and
safetensors round-trip helpers, and release the
GIL during compiler/evaluator work. `ChelisError` SHALL be reserved for compiler/build/runtime
failures while Python-side data mismatches SHALL surface as `ValueError`.

#### Scenario: CPU DLPack interop with PyTorch

- **WHEN** a PyTorch CPU tensor is passed through Python interop
- **THEN** it interops via typed DLPack, releasing the GIL during compiler/evaluator work

#### Scenario: Incompatible device is a Python-side ValueError

- **WHEN** a tensor on a device incompatible with the selected backend is passed in
- **THEN** it is rejected as a Python-side `ValueError` rather than copied or reinterpreted silently

### Requirement: Direct execution and NumPy guarantee

Python SHALL provide `chelis.compile_and_load("model.ch")` as the compiled-execution product
path and `chelis.load("model.so")` as the advanced loader, with a sidecar manifest recording the
source path and content hash. `load()` SHALL warn when the source has changed since compilation.
The NumPy DLPack contract SHALL derive element width and dtype from the tagged Chelis dtype and
the selected backend capability. It SHALL never assume compiled tensors are `f32`.

#### Scenario: compile_and_load runs a compiled artifact

- **WHEN** `chelis.compile_and_load("model.ch")` is called
- **THEN** it compiles and loads the artifact for direct execution, verified for PyTorch CPU tensors and NumPy arrays

#### Scenario: load warns on changed source

- **WHEN** `chelis.load("model.so")` is called and the sidecar manifest's content hash no longer matches the source
- **THEN** it emits a warning that the source has changed since compilation

### Requirement: C interop surface

Generated C headers and runtime support SHALL make it possible to call compiled Chelis artifacts
from C or C++. The C-facing surface SHALL come from `chelis_runtime.h` plus the
Rust static runtime library rather than a generated `chelis_runtime.c`. When object-mode
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
