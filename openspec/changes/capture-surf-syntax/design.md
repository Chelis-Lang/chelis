## Context

`spec/02-surf-syntax.md` (v0.3, authoritative) records the Surf language surface as PEG
grammar plus a punchlist of decisions (P1–P16a) with Deep desugaring shapes. It explicitly
notes that "parser support or desugaring shape does not, by itself, mean a feature is already
part of the practical executable language" (Phase 3 honesty rule). This change records the
surface and its normative parse/type rules as a `surf-syntax` capability.

## Goals / Non-Goals

**Goals:**
- Capture each major surface form as a SHALL requirement with a positive scenario and its
  matching parse-error or type-error scenario.
- Preserve the desugaring contract (Surf ⟹ Deep) where it is normative for a requirement.

**Non-Goals:**
- Reproducing the full PEG grammar verbatim; the grammar lives in the source and the crate.
- Distinguishing "parser-accepts" from "executable-today"; the source's Phase-3 honesty note
  is captured in Open Questions rather than split across every requirement.

## Decisions

- Group the punchlist into ~24 requirements aligned to surface concerns (modules, imports,
  dimensions, signatures, blocks, records, matching, tuples, transforms, literals, aliases,
  opaque types) rather than one requirement per P-item, so related parse/type rules share a
  requirement and its negative parity.
- Keep the literal-defaults, literal-suffix, and contextual-tensor-literal rules as three
  separate requirements because they have distinct failure modes (default binding, adjacency,
  suffix/context disagreement).

## Risks / Trade-offs

- [Surface vs shipped-subset gap] → The source flags that some surface forms are not yet in
  the executable language. Requirements are written against the surface contract as the
  source states it; where a form is explicitly forward-compatible-only (e.g. `Diff`/`Accum`
  effects), the requirement records that status rather than asserting execution.

## Open Questions

- The Phase-3 honesty rule means the parser may accept forms (collections, iteration,
  file/data loading, tokenization) that are not yet the full executable story. This capability
  records the surface as specified; the executable subset is tracked by the roadmap chapter
  (out of scope here) and is not a spec ambiguity.
- The keyword count differs by table between §1 (26 + 7 Phase-2) and the PEG `Keyword` rule
  (which also lists `tensor`). Resolved against the shipped lexer (`crates/chelis-surf/src/lexer.rs`
  `classify_ident`): exactly 24 identifiers are promoted to keywords, and the set includes
  `tensor` but excludes `where`, `property`, `forall`, and every Phase-2 word (`effect`,
  `handler`, `perform`, `resume`, `borrow`, `do`), which lex as ordinary identifiers today. So
  the shipped reserved set matches neither §1's 26+7 nor the PEG superset. The requirement
  captures the spec's "reserved words cannot be identifiers" plus the Phase-2 reservation as the
  spec states them; the shipped lexer is narrower than the spec's Phase-2 reservation, a
  spec-vs-code gap (the Phase-2 words are not yet lexed as keywords), not a spec ambiguity.
