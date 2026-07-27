## Context

`spec/01-nomenclature.md` is the canonical source of truth for identifier and filename
conventions across the chelis monorepo and downstream shells. It splits its content into
hard language constraints (§1, parser/resolver/backend enforced), settled conventions
(§§2–10, `chelis lint` enforced), and resolved decisions (§11). This change records that
content as a `naming-and-style` capability spec without altering any behavior.

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

## Decisions

- Group the 40+ sub-sections of the source into ~17 coherent requirements rather than one
  requirement per sub-section, so each requirement is independently testable and carries its
  own negative-parity scenario. Rationale: OpenSpec requirements are test units; over-fine
  granularity would duplicate the lexer/lint boundary in noise.
- Treat the closed Deep tag vocabulary (§1.4) and its hyphen exception as one requirement
  plus a separate user-symbol-charset requirement (§1.5), because the two invariants have
  distinct enforcement points (validator vs lint).

## Risks / Trade-offs

- [Requirement granularity hides some specific prefixes] → The full
  `MODEL_NAMESPACE_PREFIXES` / `MATH_ML_WELL_KNOWN_PREFIXES` tables live in the lint source;
  the spec captures the rule and defers the exact table to the lint, matching the source's
  own "adding a prefix requires updating that table" framing.

## Open Questions

- None material. The source chapter marks §11 items as closed/resolved, so there are no
  outstanding spec ambiguities for this capability. The exact contents of the model/ML
  prefix tables are intentionally deferred to the lint implementation by the source itself.
