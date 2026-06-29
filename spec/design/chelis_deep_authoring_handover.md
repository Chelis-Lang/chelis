# Deep Authoring Substrate Handover

**Status:** handover milestone for the Chelis-side Deep authoring substrate.
The owning behavior spec is `spec/design/chelis_agent_editing_surface.md`.

## Verified Baseline

- Current branch starts from `origin/main` at `996fdb31` (`fix(deep): harden
  replacement authoring substrate (#548)`).
- Phase 0 found the trust-boundary blocker already fixed on `origin/main`:
  malformed replacement bodies, including `70.0(as)(f32)`, reject with
  `ok:false` and no replacement result over MCP/HTTP.
- Phase 0 found duplicate `defsig` rejection already fixed in both
  `chelis-types` and `chelis-validate`. Chelis does not currently dispatch
  same-name user-function overloads; issue #258 tracks that design gap.
- Phase 0 found default `chelis surf` canonicalization already fixed and
  covered by `cargo test -p chelis-cli --test surf_round_trip`.
- Duplicate export entries are not a language error today:
  `validate --deep` accepts duplicate names inside `(export ...)`, and checker
  / prove metadata stores exports in set-like structures. Edit tools preserve
  exports unless a future export-edit tool owns a different policy.

## Stable Authoring API

The public contract is text-first: tools accept Deep text and return canonical
Deep text. They parse to structured Deep ASTs internally, perform structural
edits on the AST, then run the whole-module validation pipeline before
reporting success.

Shared success rule for every edit tool:

- `ok:true` only after the edited whole module passes fitness, type, effects,
  and linearity.
- Any parse, request-shape, name-resolution, type, effect, or linearity failure
  returns `ok:false` with no `result` payload.
- Tools are pure: no file writes, no persisted state, no transaction log.
- Surf edit input is not shipped; callers may use `chelis surf` separately for
  display.

Shipped tools:

- `chelis_replace_function_body`
  - request: `{module, function_name, new_body}`
  - success: `{changed_def_deep, module_deep}`
- `chelis_add_function`
  - request: `{module, new_decls, insert_after_function?}`
  - `new_decls`: exactly one `(def ...)` plus optional matching `(defsig ...)`
  - default insertion: end of module declaration list
  - targeted insertion: immediately after the target function's def/defsig
    declaration bundle
  - success: `{added_def_deep, added_defsig_deep?, module_deep}`

The compiler-api seam for additional tools is:

1. Parse request Deep with strict parser.
2. Check only request shape and structural target resolution locally.
3. Apply the AST edit through `chelis-deep`.
4. Run `check_whole_module_edit`.
5. Return canonical Deep only from the accepted module.

## Extension Pattern

Use `chelis_add_function` as the worked reference.

- Put structural AST manipulation in `chelis-deep`, with tests that prove the
  edit is faithful under canonical printing.
- Put request/response schemas in `chelis-compiler-api::schema`; derive
  `JsonSchema` for machine-facing contracts.
- Put the compiler entry point in `chelis-compiler-api::compiler`; do not
  duplicate the validation pipeline.
- Wire Tide MCP and HTTP to the compiler-api entry point.
- Add compiler-api, MCP, and HTTP tests for success, structured failure, and
  no-result-on-failure.
- Update this spec family and the phase oracle in the same change set.

Next edit tools:

- `chelis_replace_function`: replace a whole `(def ...)` and optional matching
  `(defsig ...)`; preserve declaration position and require full validation.
- `chelis_add_property`: add one property declaration once the property Deep
  shape is stable enough for a tool-specific oracle.
- `chelis_rename` and `chelis_change_signature`: do not implement by text
  search. They need a public reference/call-graph query seam so cascades are
  explicit and auditable.

## Known Limitations

- `.dp` prove parity is not complete. Chelis #507 tracks the headline gap:
  Deep `.dp` properties have no Tier-B SMT lowering path.
- Related prove/lowering gaps remain adjacent, not substrate blockers:
  Chelis #463, #434, #506, and raw evidence cleanup #496.
- Shoals #19 is downstream work to expose a tensor Black-Scholes entry as a
  WireDag root for Beacon; Beacon currently has no open issue/PR that blocks
  this substrate.
- Deep-path diagnostics are not populated. Span survival exists for audit
  trails, but the edit diagnostic `deep_path` field remains a future L2 seam.
- The rest of the edit toolset is intentionally unbuilt in this milestone.
- No runtime sandbox, effects distribution, FlukeBall harness, Beacon proof
  path, or `.dp` prove completeness is claimed here.

## Handover Oracle

The named milestone oracle is the `Deep substrate handover` row in
`docs/phase_oracles.md`, plus the full `scripts/gate.py` run before merge.
The focused suite covers:

- replacement trust-boundary regressions over MCP and HTTP
- `add_function` compiler-api, MCP, and HTTP behavior
- duplicate `defsig` rejection in type-check and validation
- canonical `chelis surf` round-trip behavior

Before merge, a fresh-context red team must execute the malformed-body,
duplicate-`defsig`, and `chelis_add_function` failure cases.
