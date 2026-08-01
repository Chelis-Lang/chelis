# Agent Editing Surface

**Status:** L0/L1 shipped and L2 query/cascade tools added. The shipped Tide
MCP and HTTP tools are `chelis_replace_function_body`,
`chelis_add_function`, `chelis_deep_outline`, `chelis_deep_references`,
`chelis_deep_call_graph`, `chelis_replace_function`,
`chelis_add_property`, `chelis_rename`, and `chelis_change_signature`.

**Owning surface:** Tide (MCP tool layer). Operates on Deep AST in
`chelis-deep`. Validates via the existing compiler API.

**Cross-references:**

- `spec/design/chelis_project_plan.md` — agent editing surface delivery sequence
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
That is sufficient for body replacement, function insertion, replacement of a
whole function, property insertion, rename, and signature change. Rename and
signature-change use the public reference/call-graph query surface so
cascades are computed structurally instead of by textual search.

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

## L2: Query and cascade tools

The L2 increment adds the query surface needed for auditable cascades and
then ships the remaining high-leverage Deep edit tools:

- `chelis_deep_outline`: returns module name, exports, function outline,
  canonical def/defsig Deep, and the `preimage_sha256` for each function.
- `chelis_deep_references`: returns scope-aware references to a top-level
  symbol. Function parameters and local bindings shadow same-name top-level
  functions; shadowed vars are not reported as references.
- `chelis_deep_call_graph`: returns scope-aware direct-call edges.
- `chelis_replace_function`: replaces one whole `(def ...)` and optional
  matching `(defsig ...)`, preserving declaration position, then requires full
  whole-module validation.
- `chelis_add_property`: inserts one Deep property declaration bundle, then
  requires full whole-module validation.
- `chelis_rename`: renames one function, matching defsig/export entries, and
  every unshadowed reference. It asserts no residual old references before
  validation.
- `chelis_change_signature`: replaces a function defsig and parameter list,
  rewrites every direct call from the pre-edit call graph according to
  `argument_order`, and fails if any direct call remains stale.

The optional `preimage_sha256` on cascade/whole-function edit requests is an
optimistic-concurrency guard over the canonical Deep of the addressed `(def
...)` node. A mismatch fails the whole request at `stage:"preimage"` with no
result payload; callers should re-query and retry.

**Acceptance oracle:** `cargo test -p chelis-deep --test authoring`, `cargo
test -p chelis-compiler-api --test deep_authoring`, `cargo test -p chelis-tide
--test mcp deep_query_and_rename_tools_are_model_facing_contracts`, and `cargo
test -p chelis-tide --test api
deep_query_and_rename_http_endpoints_lock_preimage_contract`.

---

## Dependencies

- **Tide MCP surface** — shipped (per `spec/09-tide.md`).
- **Compiler reference/call graph as a queryable surface** — shipped for the
  Deep authoring surface as `chelis_deep_references` and
  `chelis_deep_call_graph`. It is syntactic and scope-aware for top-level
  functions; richer semantic reference classes remain future work.
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

- Docs may claim only the tools listed in this document's Status paragraph as
  shipped.
- Docs may claim whole-module validation only for builds that enforce the
  soundness rule and pass the wire-surface oracle above.
- Future tools must remain explicitly future until their implementation and
  oracle land in the same change set.
- No doc should imply that downstream authoring campaigns, skeleton design,
  ABI reconciliation, or model-field design are owned by this surface.
