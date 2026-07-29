## Context

Main already contains the stock OpenSpec tree from #839. That tree has template configuration and no hosted structural validation.

`spec/design/spec_provenance.md` describes a future required OpenSpec phase. Chelis is not ready to require a lifecycle for each governed change or `spec/**` edit.

This change adds authored configuration and advisory structural validation. It does not activate Phase 0.

## Goals / Non-Goals

**Goals:**

- Add authored project context and artifact rules to the canonical `openspec/` root.
- Run structural validation in CI and locally.
- Keep schema findings advisory and keep operational failures nonzero.

**Non-Goals:**

- Require OpenSpec lifecycles, citations, planning order, or branch-scope rules.
- Govern `spec/**` or make OpenSpec a documentation-only gate.
- Make OpenSpec or provider metadata a Chelis product authority.
- Activate `spec/design/spec_provenance.md` Phase 0.

## Decisions

### 1. Advisory structural validation, not Phase 0 governance

CI calls `Chelis-Lang/ci/actions/openspec-governance` at an exact SHA in advisory mode. The consumer checker runs `openspec validate --all --strict --no-interactive` over the complete active tree.

The fixed action interface supplies a comparison base. The checker parses that base but applies no diff policy.

The checker reserves exit `1` for schema findings. The central action emits a warning and returns zero for this result.

Exit `2` identifies an operational failure. The central action keeps this result nonzero in both modes.

The checker applies no lifecycle, citation, planning-order, branch-scope, or `spec/**` policy.

**Alternative rejected:** Restore the Phase 0 merge-bound checker. That checker enforces policy that this initial adoption does not activate.

### 2. Use the locked central toolchain

The central action supplies Node 24.18.0 and installs its locked OpenSpec 1.6.0 dependency graph. It also validates the OpenSpec version before it calls the consumer checker.

The workflow uses a full-history checkout because the central action requires a comparison base. The exact central SHA permits source review and rollback.

**Alternative rejected:** Install OpenSpec globally in the workflow. That command resolves transitive dependencies again during each run.

## Risks / Trade-offs

- **[A malformed tree can merge]** → Advisory mode emits a warning and returns zero for schema findings.
- **[The central action name implies broader governance]** → The consumer checker defines the policy and runs structural validation only.
- **[An operational failure creates a failed job]** → This result identifies a broken validation path instead of a schema finding.
- **[The OpenSpec tree can conflict with `spec/**`]** → The authority boundary in `spec/design/spec_provenance.md` remains controlling.

## Open Questions

None.
