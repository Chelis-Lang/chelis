## Context

`spec/07-concurrency.md` is a short, settled outline: implicit DAG parallelism is the default,
`par` is reserved for explicit fork/join, backends map concurrency without changing semantics,
and several models are explicit v1 non-goals. This change records that outline as a
`concurrency` capability.

## Goals / Non-Goals

**Goals:**
- Capture the default implicit-parallelism story, the `par` reservation, backend mapping, and
  the v1 non-goals as SHALL requirements with scenarios.

**Non-Goals:**
- Specifying detailed `par` runtime scheduling semantics; the source explicitly leaves that to
  a later phase.

## Decisions

- Encode the v1 non-goals as a requirement whose failure scenario asserts the out-of-scope
  models are rejected as out of scope, so the boundary is testable rather than prose-only.

## Risks / Trade-offs

- [Outline-level detail] → The source is an outline with settled direction but limited detail.
  The requirements capture the settled invariants (semantics-preserving backend mapping,
  annotation-free default) and avoid over-specifying `par` runtime behavior.

## Open Questions

- Detailed `par` runtime semantics and scheduling are deliberately left to a later phase by the
  source; this is a roadmap matter, not a spec ambiguity.
