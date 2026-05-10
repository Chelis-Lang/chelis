# Serialization

**Status:** Partial.
Text formats are settled enough to reference.
Binary formats remain intentionally under-specified until there is an implemented owner.

## 1. Text Forms

- `.ch` is Surf source text in UTF-8
- `.dp` is Deep source text in UTF-8

Deep canonical printing is defined by `spec/03-deep-syntax.md`.

## 2. Binary Shell Form

`.chb` is the binary Shell metadata artifact used by the Reef package system.
Its role is stable, but the exact wire format is still owned by the implementation.

Current `.chb` expectations:

- public package metadata
- exported symbol metadata
- compiler compatibility metadata
- room for future cached products without freezing the public wire layout

The project should not publish fake low-level `.chb` layout guarantees while the
implementation is still expected to evolve.

## 3. Compiler API Wire Compatibility

The compiler API wire models in `crates/chelis-compiler-api/src/schema.rs` are
the current machine-facing JSON surface. During this pre-release period, adding
new tagged variants such as `WireRiscOp::Gather` or
`WireRiscOp::ScatterAdd` is an additive schema change. Producers may emit the
new variant after the owning compiler behavior lands.

Consumers should tolerate unknown additive variants where possible and report a
clear unsupported-variant diagnostic rather than failing only because the enum
grew. Consumers that intentionally pattern-match exhaustively must treat the
wire schema as version-coupled to the compiler crate they were built with.

## 4. Related Serialization Work

Future serialization work may include:

- binary program artifacts for Shell distribution
- DAG serialization
- tensor interop formats such as DLPack

These belong to the phases that implement them rather than to speculative prose here.
