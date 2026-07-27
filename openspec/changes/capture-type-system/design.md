## Context

`spec/04-type-system.md` (v0.3) is the largest language chapter: it covers the checked-Deep
contract, the active dtype set and per-backend matrix, ADTs and opaque-type module identity,
Hindley-Milner inference, the named-tensor-dimension algebra (no broadcasting), dimension and
rank polymorphism, precision rules and accumulators, runtime shape semantics, fitness scoring,
the Phase-2a effect subset, linearity, and two decided-but-not-yet-implemented sections
(numeric value semantics §9, checker totality §10).

## Goals / Non-Goals

**Goals:**
- Capture the normative typing rules as SHALL requirements, each with a positive scenario and a
  matching type-error/rejection scenario.
- Faithfully record §9 and §10 as decided normative requirements while marking their
  implementation status honestly.

**Non-Goals:**
- Reproducing every issue-linked micro-rule of §4.5.2/§4.7.2 (list-join head bias, per-form
  expand-size materialization); the requirement captures the governing rule and its failure
  mode, deferring the exact enumeration to the source and the checker.
- Restating the full inference-rule calculus of §3.2; the requirement captures the algorithm
  and the load-bearing branch/let/annotation rules.

## Decisions

- Fold the many dimension-algebra sub-rules (§4.1–§4.7) into four requirements (named-dimension
  matching/no-broadcasting, dim-polymorphism rigidity, wildcard/rank-uniform lists, rank
  polymorphism, runtime shape) rather than one per sub-section, since they share the
  named-dimension safety rationale and their failures are all `DimensionMismatch`.
- Keep §9 (numeric value semantics) and §10 (checker totality) as their own requirements and
  label them "(decided)" in the requirement name, matching the source's own status banners.

## Risks / Trade-offs

- [Decided-vs-implemented gap] → §9 and §10 are DECIDED normative but diverge from current
  behavior (issue-linked `#[ignore]`d tests). Capturing them as SHALL requirements is faithful
  to the source's normative intent; the divergence is recorded in Open Questions rather than
  softened into non-normative language.

## Open Questions

- §9 numeric value semantics and parts of §10 checker totality are explicitly not yet
  implemented (the source lists per-atom issue references and locks the divergences as ignored
  tests). Verified against code: the §9 [04-NUM-7] `wrap_*` builtins do not exist anywhere in the
  workspace, and every integer-overflow-trap test is `#[ignore]`d with notes that eval wraps and
  compiled C saturates/escapes the width (`crates/chelis-cli/tests/issue_680_int_exactness.rs`,
  `int_width_lane_matrix.rs`). The §10 [04-TOT-2] half the source calls "structurally honored"
  is real: `Type::Error` carries a private `ErrorWitness` minted through an append-only
  `DiagnosticSink` (`crates/chelis-types/src/errors.rs`, `session.rs`). This is a known
  implementation-vs-spec gap tracked upstream (chelis#729, #731, #733), not a spec ambiguity;
  the requirements record the decided semantics as written.
- The Metal MPS runtime ARC ownership invariants (§1.1.3) are pinned for a not-yet-added
  implementation phase; they are captured under the per-backend matrix requirement's rationale
  rather than as a separate typing requirement, since they are a codegen-runtime contract, not
  a type judgment.
