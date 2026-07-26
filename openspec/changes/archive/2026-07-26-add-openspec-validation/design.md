## Context

`spec/design/spec_provenance.md` describes a future required-governance phase (Phase 0) for OpenSpec, but Chelis has no OpenSpec planning tree yet and is not ready to require lifecycles for every governed change or every `spec/**` edit. This change adopts OpenSpec in the smallest useful form: a valid planning tree plus advisory structural validation.

## Goals / Non-Goals

**Goals:**

- Establish the canonical `openspec/` root with the built-in spec-driven schema.
- Validate the tree structurally in CI and locally, non-blocking.

**Non-Goals:**

- Require OpenSpec lifecycles, citations, planning-before-code ordering, or branch-scope rules.
- Govern `spec/**` or make OpenSpec a documentation-only gate.
- Make OpenSpec or provider metadata canonical Chelis product, atom, coverage, or implementation-correctness authority.
- Activate `spec/design/spec_provenance.md` Phase 0.

## Decisions

### 1. Advisory validation, not governance

CI runs `openspec validate --all --strict` on `openspec/` changes with `contents: read`. It is not a required status check and does not gate merges. A malformed OpenSpec artifact is caught early; nothing else is enforced.

**Alternative rejected:** A merge-bound governance checker enforcing lifecycle ordering, PR citations, branch scope, and `spec/**` coupling. That is the future Phase 0 regime, deferred until the repository is ready to require it.

### 2. OpenSpec installed from npm at an exact version

The workflow installs `@fission-ai/openspec@1.6.0` and runs `openspec validate`. No private shared action, no full-history diffing, and no secrets are required.

**Alternative rejected:** A pinned private composite action. It is only needed for merge-bound governance, which this change does not perform.

## Risks / Trade-offs

- **[Validation is advisory, so a malformed tree could merge]** → Acceptable for an initial adoption; the check still surfaces the failure on the pull request, and a future change can make it required.
- **[The OpenSpec tree could drift from `spec/design/spec_provenance.md`]** → The `openspec-validation` capability states explicitly that it does not activate Phase 0 and that `spec/**` remains controlling.

## Open Questions

None.
