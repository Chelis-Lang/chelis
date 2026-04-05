# Foreign Function Interface

**Status:** Outline for later phases.
Chelis does not depend on FFI work for Phase 0 completion, but the expected direction is
already clear enough to record.

## 1. Python Interop

The planned Python path is:

- PyO3 bindings for compiler and runtime entry points
- DLPack for zero-copy tensor exchange with NumPy, PyTorch, and JAX
- GIL release during Chelis execution

This is a Phase 3 interoperability feature, not a Phase 0 requirement.

## 2. C Interop

Generated C headers and runtime support should make it possible to call compiled Chelis
artifacts from C or C++.
That interoperability follows naturally from the reference backend and does not require a
separate host-language embedding model first.

## 3. Embedding the Compiler

Longer term, the Rust crates should remain usable as libraries so Tide, editor tooling,
and external integrations can embed compiler functionality directly rather than shelling
out to the CLI.
