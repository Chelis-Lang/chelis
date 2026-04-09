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

## 3. Related Serialization Work

Future serialization work may include:

- binary program artifacts for Shell distribution
- DAG serialization
- tensor interop formats such as DLPack

These belong to the phases that implement them rather than to speculative prose here.
