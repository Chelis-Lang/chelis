## Context

`spec/03-deep-syntax.md` (v0.2) is the machine interface spec: the universal 3-tuple node,
the closed 62-tag vocabulary, the metadata contract (including the span trust boundary), the
canonical form, and the validation rules split across the parser, post-parse arity checks, and
the type checker. This change records that content as a `deep-syntax` capability.

## Goals / Non-Goals

**Goals:**
- Capture the node/metadata/vocabulary/validation contract as SHALL requirements with
  positive and failure scenarios.
- Preserve the trust-boundary rules for producer-supplied strings (span, load/store names) as
  first-class security invariants, since the source frames them as a real injection class.

**Non-Goals:**
- Reproducing the full 62-tag table row-by-row as separate requirements; the closed-vocabulary
  requirement plus representative tags is sufficient and matches the source's own framing.
- Restating the canonical-form byte layout in full; the requirement captures the invariants
  (single representation, ordering, literal normalization) that tests can check.

## Decisions

- Keep span validation, synthesized-marker reservation, and the general producer-string
  trust-boundary rule as three separate requirements because they have distinct enforcement
  points (parser vs marker minting vs IR newtype construction).
- Split type-expression resolution (§2.5.1) and dimension-expression shape (§2.6) into two
  requirements since they are owned by the type checker with different failure vocabularies.

## Risks / Trade-offs

- [62-tag count is a moving target] → The source states the count is authoritative only until
  the doc is revised for Phase 3+ collection/scalar/string tags. The requirement captures the
  closed-vocabulary invariant and the strict-vs-fitness handling rather than hardcoding future
  additions.

## Open Questions

- None material. The source is explicit that pre-expansion macro tags and Phase 3+ tags are
  out of the current public vocabulary; the requirement records the shipped closed vocabulary
  and the revision obligation, which is not a spec ambiguity.
