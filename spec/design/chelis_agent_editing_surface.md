# Agent Editing Surface

**Status:** L0 shipped in v0.11.1 and hardening; the handover substrate adds
the second structural edit tool. The shipped Tide MCP and HTTP tools are
`chelis_replace_function_body` and `chelis_add_function`. Expansion to the
remaining toolset remains future work.

**Owning surface:** Tide (MCP tool layer). Operates on Deep AST in
`chelis-deep`. Validates via the existing compiler API.

**Cross-references:**

- `spec/12-roadmap.md` — agent editing surface pointer
- `spec/design/chelis_trust_stack.md` — Trust Stack Implications for Editing Tools
- `spec/design/chelis_canonical_reference.md` §4 (Deep), §12 (AI Coding Assistance) — architectural framing
- `spec/03-deep-syntax.md` §1.3 — Deep as an editing target
- `spec/design/chelis_span_survival.md` — provenance dependency for diff display
- `spec/design/chelis_deep_authoring_handover.md` — handover state and extension seams

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
The L0 replacement tool establishes the first trust boundary: an agent may ask
for a Deep function body splice, but success is reported only after the
rewritten whole module validates through the compiler-owned checks.

The architectural foundation is real: Deep is a 62-tag closed vocabulary
with 3-tuple uniformity and a metadata slot for provenance. Structural
operations on that substrate (replace a function body, rename a symbol,
change a signature) are well-defined in a way they are not on plain text.
Whether that translates to measurably better edit success rates for AI
agents is the reason this surface exists. The shipped measurement path is
downstream usage by authoring campaigns that consume the tool; this spec does
not require a text-vs-Deep benchmark before further hardening work.

**Public contract judgment:** the authoring API remains text-first at the
wire boundary: callers send Deep text, and successful tools return canonical
Deep text. The implementation uses parsed Deep AST operations internally.
That is sufficient for body replacement and function insertion. Rename and
signature-change are not blocked on JSON AST input; they are blocked on a
public reference/call-graph query surface so cascades can be computed
honestly instead of by textual search.

---

## L0: `chelis_replace_function_body`

The first workhorse structural edit. It operates on Deep AST strings: take a
module, a function name, and a new Deep body expression; return the rewritten
module only if the post-splice whole-module check is clean.

**Scope:**

- Tide MCP tool: `chelis_replace_function_body`
- Tide HTTP endpoint: `/replace_function_body`
- Request shape: `{module, function_name, new_body}`, all strings
- `module` and `new_body` are Deep; `new_body` must parse to exactly one Deep
  expression
- Success shape: `{changed_def_deep, module_deep}`, both canonical Deep
- Failure shape: the normal Tide structured error envelope with `ok:false`,
  `stage`, and typed diagnostics
- No file I/O and no persisted state; callers own applying returned text
- No Surf edit surface; Surf rendering remains separate CLI/API behavior

**Soundness rule:** `ok:true` is written only when the post-splice
whole-module check is clean. Internal parse, splice, type, effect, or
linearity failures are hard rejections: `ok:false`, no replacement result, and
the diagnostic for the rejecting stage is surfaced.

**Acceptance oracle:** `cargo test -p chelis-tide --test mcp
replace_function_body` and `cargo test -p chelis-tide --test api
replace_function_body` exercise success, malformed-body rejection, type
rejection, effect rejection, linearity rejection, name-resolution rejection,
and the no-result-on-failure envelope over the shipped wire surfaces.

**Value:** Gives authoring agents one compiler-owned edit primitive whose
success verdict is tied to the same whole-module checks users trust elsewhere
in the CLI/API.

---

## L1: `chelis_add_function`

The second structural edit proves the extension pattern beyond a body splice.
It inserts a Deep function declaration bundle into an existing Deep module,
then reports success only if the whole rewritten module validates.

**Scope:**

- Tide MCP tool: `chelis_add_function`
- Tide HTTP endpoint: `/add_function`
- Request shape: `{module, new_decls, insert_after_function?}`
- `module` and `new_decls` are Deep strings
- `new_decls` must contain exactly one `(def ...)` and may contain one
  matching `(defsig ...)`; the bundle is inserted in the authored order
- No export editing: existing `(export ...)` declarations are preserved
  unchanged, and duplicate-export policy is not reimplemented in this tool
- Default insertion appends to the module declaration list
- `insert_after_function` inserts immediately after the target function's
  existing declaration bundle (`def` plus any same-name `defsig`)
- Success shape: `{added_def_deep, added_defsig_deep?, module_deep}`, all
  canonical Deep
- Failure shape: the normal Tide structured error envelope with `ok:false`,
  `stage`, and typed diagnostics
- No file I/O and no persisted state; callers own applying returned text

**Soundness rule:** `ok:true` is written only when the post-insertion
whole-module check is clean. Request-shape and insertion-target failures are
hard rejections before validation; duplicate definitions/signatures, type
errors, effect errors, and linearity errors are surfaced from the
whole-module pipeline.

**Insertion faithfulness:** the returned module is the canonical input module
with exactly the authored declaration bundle inserted at the requested
position, and no other declaration changed under canonical Deep printing.
Whole-module validation catches invalid programs; insertion-faithfulness tests
catch misplaced edits.

**Acceptance oracle:** `cargo test -p chelis-compiler-api --test
deep_authoring`, `cargo test -p chelis-tide --test mcp add_function`, and
`cargo test -p chelis-tide --test api add_function` exercise success,
schema availability, structured failure, no-result-on-failure, and insertion
faithfulness.

---

## Substrate hardening

The L0 tool sits on language and compiler invariants that must fail closed:

- Phase 0 of the handover milestone found the trust-boundary and decompiler
  canonicalization fixes already landed on current `origin/main` by the
  hardening work in PR #548. They remain part of the oracle; they were not
  skipped.
- Duplicate `defsig` declarations for the same function are rejected. Chelis
  does not dispatch user functions by arity, type, or rank; the valid same-name
  pair is exactly one `defsig` plus one `def`.
- The Tide MCP and HTTP envelopes are locked for parse/type/effect/linearity
  failures so model-facing clients can branch on `stage` and diagnostic
  `kind`.
- Default `chelis surf` output is formatter-canonical before it is returned.
  If idiomatic decompile output cannot parse or format, that is a
  decompiler-vs-formatter divergence to fix, not a documentation fallback.

**Acceptance oracle:** see `docs/phase_oracles.md` for the Deep substrate
hardening campaign row.

---

## Next capability frontier

After L0 hardening, the next structural-edit increment is a small set of
high-leverage authoring tools:

- replace a whole function
- add a property
- rename a symbol
- change a signature

`chelis_add_function` is now the worked extension reference. The remaining
tools are not shipped by this milestone. They must each get an owning oracle
before implementation claims land in active docs.

---

## Dependencies

- **Tide MCP surface** — shipped (per `spec/09-tide.md`).
- **Compiler reference/call graph as a queryable surface** — graph-like helper
  code exists internally in checker/backend passes, but no public authoring API
  exposes it. Required for `chelis_change_signature` and `chelis_rename`
  cascade behavior.
- **Surf rendering of arbitrary Deep changes** — default `chelis surf` output
  is canonical for the supported round-trip path; richer diff display remains
  separate work.
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

- Docs may claim only `chelis_replace_function_body` and
  `chelis_add_function` as shipped.
- Docs may claim whole-module validation only for builds that enforce the
  soundness rule and pass the wire-surface oracle above.
- Future tools must remain explicitly future until their implementation and
  oracle land in the same change set.
- No doc should imply that downstream authoring campaigns, skeleton design,
  ABI reconciliation, or model-field design are owned by this surface.
