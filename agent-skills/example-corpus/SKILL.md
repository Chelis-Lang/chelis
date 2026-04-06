---
name: example-corpus
description: Use when adding, moving, or validating Chelis example programs. Preserves the executable-vs-illustrative split and keeps example-related tests honest.
---

# Example Corpus

Use this skill whenever touching `examples/` or tests/docs that reference examples.

## Policy

- `examples/` is the executable corpus.
- `examples/illustrative/` is for syntax/design examples that are not on the executable
  phase path.

## Rules

- If an example is used as phase proof, it belongs in `examples/`.
- If it intentionally uses unsupported or future-phase constructs, move it to
  `examples/illustrative/`.
- Update tests when moving examples so executable-corpus tests do not silently become parse-only.

## Verification

For executable examples:

- parse succeeds
- `chelis fmt --inplace` preserves validity
- `chelis check` returns score `1.0` with no errors

For illustrative examples:

- parsing/round-trip expectations may still apply
- docs and tests must not imply they are executable phase proof

