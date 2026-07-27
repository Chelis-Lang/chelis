## Context

`spec/08-backends.md` records the backend strategy (C-first, then a single HIP GPU path, then
Metal as an independent macOS GPU peer), the shared `chelis_tensor **` ABI, the source-to-source
GPU model, the phased HIP (1a–1f) and Metal (M0–M7) delivery with per-phase oracles, the
Phase-2a backend-boundary effect checks, and the all-backend invariants. This change records
that content as a `backends` capability.

## Goals / Non-Goals

**Goals:**
- Capture the backend strategy, the shared ABI and source-to-source model, per-backend dtype
  gating, backend-boundary effect checks, backend selection/escalation, and the all-backend
  invariants as SHALL requirements with positive and rejection scenarios.

**Non-Goals:**
- Reproducing every HIP/Metal sub-phase and its manual oracle command; the requirement captures
  the delivered contract (shared ABI, source-to-source, dtype gating) rather than the phase
  ledger.
- Restating the runtime ARC/MPS ownership invariants (owned by `type-system` §1.1.3); they are
  cross-referenced there.

## Decisions

- Fold the per-dtype matrix enforcement into one "per-backend dtype gating" requirement that
  cites the `type-system` §1.1.3 matrix, avoiding duplication of the full table.
- Keep backend selection and interactive-execution escalation in one requirement since the
  source frames interactive execution as "not a separate backend."

## Risks / Trade-offs

- [Phase-ledger detail omitted] → The source is heavy on per-phase completion status and manual
  oracle commands. The requirements capture the stable contract; the phase ledger and manual
  gates are recorded in the owning phase plans, not duplicated as requirements.

## Open Questions

- Later integration backends (StableHLO, FX) are additive and not yet shipped; the requirement
  records the C/HIP/Metal path and treats those as roadmap additions, not current behavior.
