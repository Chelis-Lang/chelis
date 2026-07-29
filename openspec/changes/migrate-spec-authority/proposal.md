## Why

`spec/` is to become legacy and `openspec/` the specification tree going forward. Capturing the numbered chapters as capabilities begins that move, but capture alone does not transfer authority — it produces two trees carrying normative language about the same subject, with no statement of which one controls.

That ambiguity is not stable. A reader cannot answer "which tree governs the type system today?", an author does not know which tree to edit, and a captured capability can drift from its chapter with nothing to detect it. `spec/design/spec_provenance.md` currently forbids OpenSpec from being an authority at all, so the migration presently contradicts a document still in force.

This change defines how authority moves: one chapter at a time, recorded, with the legacy chapter controlling until its transfer lands.

## What Changes

- Add a `spec-authority-migration` capability defining per-chapter authority transfer, the provenance a capture must carry, the supersession marking a transferred chapter receives, and the recording of known divergence.
- Do not transfer any chapter's authority in this change. This establishes the contract; transfers are separate changes.
- Do not modify `spec/**`, the captured capabilities, or any tooling here.

## Capabilities

### New Capabilities

- `spec-authority-migration`: How normative authority moves from the numbered `spec/` chapters to captured OpenSpec capabilities — per-chapter, recorded, and reversible in review.

### Modified Capabilities

None.

## Impact

- Adds `openspec/changes/migrate-spec-authority/` and, on sync, `openspec/specs/spec-authority-migration/`.
- Changes no language, compiler, CLI, backend, numerical, package, or generated-code behavior.
- Blocks nothing: until a transfer change lands, every `spec/` chapter stays controlling exactly as it is today.
