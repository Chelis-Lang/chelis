## Why

Chelis has no canonical OpenSpec planning tree. `spec/design/spec_provenance.md` describes an eventual required-governance regime (Phase 0), but activating that regime now is premature. An initial, non-blocking adoption gives contributors an OpenSpec planning root and catches malformed artifacts early, without gating merges or coupling to `spec/**`.

## What Changes

- Add the canonical `openspec/` planning tree (built-in spec-driven schema) with one baseline capability, `openspec-validation`.
- Add a dedicated, advisory CI workflow that runs `openspec validate --all --strict` on changes under `openspec/` with `contents: read`. It is not a required status check and does not gate merges.
- Do not require OpenSpec lifecycles for changes, do not govern `spec/**`, and do not treat OpenSpec or provider metadata as Chelis product authority.

## Capabilities

### New Capabilities

- `openspec-validation`: Initial non-blocking OpenSpec adoption — structural validation of `openspec/` in CI and locally, with no governed-change enforcement and no `spec/**` coupling.

### Modified Capabilities

None.

## Impact

- Adds `openspec/` and `.github/workflows/openspec-validate.yml`.
- Changes no language, compiler, CLI, backend, numerical, package, or generated-code behavior.
- Leaves the `spec/design/spec_provenance.md` Phase 0 governance regime inactive; a future change may activate it.
