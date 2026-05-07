# Agent Editing Surface (Exploratory)

**Status:** Exploratory. One bounded proof-of-concept tool, followed by a
comparative benchmark. Expansion to a full toolset is gated on those two
items demonstrating value.

**Owning surface:** Tide (MCP tool layer). Operates on Deep AST in
`chelis-deep`. Validates via the existing compiler API.

**Cross-references:**

- `spec/12-roadmap.md` — exploratory section pointer
- `spec/design/chelis_trust_stack.md` — Trust Stack Implications for Editing Tools
- `spec/design/chelis_canonical_reference.md` §4 (Deep), §12 (AI Coding Assistance) — architectural framing
- `spec/03-deep-syntax.md` §1.3 — Deep as an editing target
- `spec/design/chelis_span_survival.md` — provenance dependency for diff display

---

## Background

Agent coding harnesses (Claude Code, Cursor, etc.) experience predictable
failure modes when editing files: string-match edit failures, patch format
brittleness, whole-file rewrite regressions, stale-context bugs, cross-file
invariant violations. These failures are substrate-level — the harnesses
operate on plain text files where the substrate gives the agent no traction
beyond character matching.

Chelis's architectural properties (small Deep vocabulary, fast compiler with
structured fitness output, dual Surf/Deep syntax, agent-first design) suggest
structural editing primitives could be more reliable than text-based editing.
This direction explores whether that hypothesis holds empirically with one
bounded proof-of-concept tool.

The architectural foundation is real: Deep is a 60-tag closed vocabulary
with 3-tuple uniformity and a metadata slot for provenance. Structural
operations on that substrate (replace a function body, rename a symbol,
change a signature) are well-defined in a way they are not on plain text.
Whether that translates to measurably better edit success rates for AI
agents is the empirical question this direction sets out to answer.

---

## Item 1: `chelis_replace_body` proof-of-concept

The simplest workhorse structural edit. Operates on Deep AST: take a function
name and a new body, produce a new file state with that function's body
replaced. Compiler validates the result and returns fitness + diagnostics.

**Scope:**

- New Tide MCP tool: `chelis_replace_body(file, name, new_body)` returning
  `{file_state, fitness, diagnostics}`
- Operates on Deep AST in `chelis-deep`, not on text
- Uses existing compiler API for validation (already exposed via Tide)
- Surf rendering of the result for human-readable diff display

**Effort:** ~1 week of focused work after the dependencies (compiler API
surface for body-replacement, Surf rendering of partial Deep changes) are
exposed. The dependencies themselves may add scope.

**Acceptance oracle:** Tool replaces a function body in a Chelis file,
returns the new state, fitness and diagnostics match what `chelis check`
would produce on the new file.

**Value:** Demonstrates the structural-editing-tool category. If it works
and shows measurably better edit success rates than text-based editing on
equivalent tasks, it justifies expanding to the full toolset.

---

## Item 2: Comparative benchmark

To validate the structural-editing hypothesis empirically, run a benchmark:

- Corpus of editing tasks (function body replacement, signature change,
  rename, etc.)
- Compare success rates across:
  - Generic `str_replace` on Python
  - Generic `str_replace` on Chelis
  - Structural Chelis tools (after Item 1 ships)

The expected result is structural tools winning decisively on success rate
even though models have less Chelis familiarity. This anchors the
structural-tool advantage against an industrial baseline.

**Effort:** Benchmark design and execution: ~1 week.

**Acceptance oracle:** Benchmark numbers published with methodology;
structural-tool advantage measured.

**Value:** Provides empirical grounding for the structural-editing claim.
Either confirms the direction is worth investing in further, or surfaces
the actual magnitude of the advantage.

---

## Item 3 onward: full toolset (gated on Items 1-2 succeeding)

If Items 1 and 2 demonstrate the structural-editing advantage, the full
toolset becomes worth building:

- `chelis_define(file, definition_block)` — adds a function
- `chelis_change_signature(name, new_sig)` — modifies signature, cascades
  to callers automatically using the compiler's call graph
- `chelis_rename(symbol, new_name)` — updates every reference, fails on
  collision
- `chelis_property_check(file, function)` — runs `chelis fuzz` on the
  function, requires `@property` upstream first
- `chelis_view(file, mode="surf"|"deep")` — rendering for human or agent
- Transactional grouping (`begin/commit/abort`) for multi-step refactors

**Effort:** Each tool is ~1-2 weeks of focused work. The transactional
layer is ~2 weeks on top of individual tools.

**Status:** Gated on Items 1-2 succeeding. Don't start building speculatively.

---

## Item 4: Deferred — comparative paper / external positioning

If the empirical benchmark from Item 2 produces strong results, this
direction has external positioning value (research paper on structural
editing reliability, technical blog post, demo for evaluations). External
positioning work is not part of OSS technical priorities; it's a separate
non-repo channel.

---

## Dependencies

- **Tide MCP surface** — shipped (per `spec/09-tide.md`).
- **Compiler call graph as a queryable surface** — probably exists internally,
  not yet exposed. Required for `chelis_change_signature` and `chelis_rename`
  cascade behavior.
- **Surf rendering of arbitrary Deep changes** — high-quality enough for human
  diff review; separate work, may need refinement.
- **`@property` upstream** — required for `chelis_property_check`.
- **Span survival** (`spec/design/chelis_span_survival.md`) — required for
  cross-tool provenance in diff display once edits propagate through the
  pipeline.

---

## Non-goals

- **Generic file editing for non-Chelis files.** Markdown, manifests,
  configuration files continue to use text-based editing. The structural
  advantage is for Chelis source specifically.
- **Cross-language edits.** Chelis projects with Python glue, Rust FFI,
  or build config fall back to text editing for those parts. Standard
  problem.
- **Tool hallucination mitigations at the language level.** Tool-name and
  parameter hallucination are model problems addressed by harness-level
  schema validation, not by language-level changes.
- **Replacing existing IDE refactoring tools.** This direction doesn't
  compete with IntelliJ, Roslyn, etc. for industrial languages; it builds
  Chelis-specific tooling that wouldn't exist otherwise.

---

## Status framing discipline

This direction is exploratory. Until Item 1 ships and Item 2 produces
empirical results:

- No doc anywhere claims structural editing tools as a current capability.
- No doc claims category-defining novelty for the structural-editing
  approach. The substrate-vs-model framing is correct internally but
  remains overclaim until empirical evidence backs it.
- Skill files (`AGENTS.md` / `CLAUDE.md`, `.cursorrules`, `agent-skills/`) and Tide MCP
  tool docs are NOT updated to mention structural tools — those updates
  land when tools ship, not before.
- The full toolset (Items 3+) is named for design completeness, not
  committed for build. Adding `chelis_define`, `chelis_rename`, etc. to
  the canonical reference as planned features before Item 1 ships
  overcommits to a direction that hasn't been validated.

If the proof-of-concept succeeds, the docs already have the framing in
place to expand to the full toolset. If it doesn't, the docs cleanly
contain the exploration without misleading future readers about what was
actually built.
