# Serialization

**Status:** Partial.
Text formats are settled enough to reference.
Binary formats remain intentionally under-specified until there is an implemented owner.

## 1. Text Forms

- `.ch` is Surf source text in UTF-8
- `.dp` is Deep source text in UTF-8

Deep canonical printing is defined by `spec/03-deep-syntax.md`.

## 2. Planned Binary Form

`.chb` is the name reserved for a future binary Chelis artifact.
The term is stable, but the exact wire format is not yet frozen.

Current expectations for a future `.chb` artifact:

- typed program representation
- compiler metadata
- room for backend-specific cached products

The project should not publish fake low-level `.chb` layout guarantees until a real
serialization implementation exists.

## 3. Related Serialization Work

Future serialization work may include:

- binary program artifacts for Shell distribution
- DAG serialization
- tensor interop formats such as DLPack

These belong to the phases that implement them rather than to speculative prose here.
