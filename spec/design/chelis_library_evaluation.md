# Chelis Compiler: Library Decisions

**Status:** Settled project guidance.
This file records the current outcome of the library evaluation work in a form that is
safe to cite from active docs.

## Summary

Chelis should keep its existing hand-written lexer, parser, and DAG infrastructure
through the remainder of Phase 0 and Phase 1.
The only library with a planned prototype slot is `egg`, and only for Phase 1 fusion.
`salsa` remains the planned incremental-compilation direction for Phase 2.

## Adopted or Planned

| Library | Decision | Why |
|---|---|---|
| `egg` | Prototype in Phase 1b only | Useful if kernel-fusion search becomes genuinely combinatorial |
| `salsa` | Plan for Phase 2 | Fits Tide, LSP, and agent-loop incrementality once those use cases exist |
| `ariadne` / `miette` | Evaluate later | Diagnostic UX is a better near-term leverage point than parser replacement |

## Rejected for the Current Architecture

| Library | Decision | Why |
|---|---|---|
| `logos` | Reject | Existing lexer works; nested comments and language-specific behavior erase most of the declarative benefit |
| `chumsky` | Reject | Existing Pratt / recursive-descent parser works; rewrite cost is high and recovery value is not yet central |
| `petgraph` | Reject | Custom DAG is smaller, append-only by construction, and easier to shape around compiler invariants |
| `cranelift` | Reject | The IR evaluator and cached C paths are cheaper interactive solutions; no second backend is planned |

## Guidance by Problem Area

### Front End

Do not replace the lexer or parser to chase abstraction.
Improve diagnostics and recovery incrementally on the existing implementation.

### IR and Optimization

Keep the custom DAG.
Use direct passes for Phase 0 optimizations.
Only prototype `egg` once Phase 1 fusion rules are concrete enough to justify rewrite
search.

### Interactive Performance

Do not introduce a JIT backend preemptively.
Measure the IR evaluator first, then cached C artifacts, then a persistent helper
process.
Only revisit JIT options if those three fail on real Tide workloads.

## Architectural Discipline

To keep future `salsa` adoption straightforward:

- each compilation stage should remain a pure function
- no global mutable state should accumulate across compiler invocations
- public crate APIs should take inputs and return outputs rather than mutating a shared
  compiler object

## Relationship to Historical Notes

Older evaluation writeups may still exist in the archive as rationale.
If an archived note disagrees with this file, this file wins.
