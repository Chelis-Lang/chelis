## Context

`spec/01-nomenclature.md` is the canonical source of truth for identifier and filename
conventions across the chelis monorepo and downstream shells. It splits its content into
hard language constraints (§1), settled conventions, and lint enforcement rules (§§2–12).
The existing `lint-traversal-policy` capability owns the traversal contract in §12.2. This
change records the remaining content as a `naming-and-style` capability spec.

## Goals / Non-Goals

**Goals:**
- Faithfully record the normative naming and style rules as SHALL requirements with
  positive and negative (failure) scenarios.
- Capture the enforcement contract (blocking vs advisory severity, escape hatches, built-in
  style gate) as machine-checkable invariants.

**Non-Goals:**
- Enumerating every worked example or every model/algorithm prefix from §7; the requirement
  captures the rule and one representative pair, not the full lookup tables.
- Restating the historical resolved-decision rationale (§11) except where it changes a
  current, testable rule.
- The change does not duplicate the lint traversal requirements from §12.2. The existing
  `lint-traversal-policy` capability owns those requirements.

## Decisions

- Group the source sections into 17 coherent requirements instead of one requirement per
  subsection. Each requirement is testable and has a negative scenario.
- The capture links the existing `lint-traversal-policy` capability for §12.2. It does not
  add weaker duplicate requirements to `naming-and-style`.
- Treat the closed Deep tag vocabulary (§1.4) and its hyphen exception as one requirement
  plus a separate user-symbol-charset requirement (§1.5), because the two invariants have
  distinct enforcement points (validator vs lint).

## Risks / Trade-offs

- [Requirement granularity hides some specific prefixes] → The full
  `MODEL_NAMESPACE_PREFIXES` / `MATH_ML_WELL_KNOWN_PREFIXES` tables live in the lint source;
  the spec captures the rule and defers the exact table to the lint, matching the source's
  own "adding a prefix requires updating that table" framing.
- [§12.2 has a separate capability] → `naming-and-style` links
  `lint-traversal-policy`. That capability cites §12.2 as its source.

## Open Questions

- None material. The source chapter marks §11 items as closed/resolved, so there are no
  outstanding spec ambiguities for this capability. The exact contents of the model/ML
  prefix tables are intentionally deferred to the lint implementation by the source itself.
