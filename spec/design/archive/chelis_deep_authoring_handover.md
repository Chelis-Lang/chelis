# Deep Authoring Substrate Handover

**Status:** handover milestone for the Chelis-side Deep authoring substrate.
The owning behavior spec is `spec/design/chelis_agent_editing_surface.md`.

## Verified Baseline

- Current handover polish starts from `origin/main` at `a2e894e7`
  (`release: v0.14.0 -- error localization + ConstTensor`).
- Phase 0 found the trust-boundary blocker already fixed on `origin/main`:
  malformed replacement bodies, including `70.0(as)(f32)`, reject with
  `ok:false` and no replacement result over MCP/HTTP.
- Phase 0 found duplicate `defsig` rejection already fixed in both
  `chelis-types` and `chelis-validate`. Chelis does not currently dispatch
  same-name user-function overloads; issue #258 is closed.
- Phase 0 found default `chelis surf` canonicalization already fixed and
  covered by `cargo test -p chelis-cli --test surf_round_trip`.
- Duplicate export entries are not a language error today:
  `validate --deep` accepts duplicate names inside `(export ...)`, and checker
  / prove metadata stores exports in set-like structures. Edit tools preserve
  exports unless a future export-edit tool owns a different policy.
- WIP audit on 2026-07-06 found the old `deep-substrate-handover` worktree
  patch-equivalent to main and the old `deep-authoring-l2` branch superseded by
  the squashed mainline work. The next handover work should branch from current
  `origin/main`, not from either stale worktree.
- Remote audit on 2026-07-06 found open Chelis PRs #617 and #619 on AD/shape
  machinery; neither changes the Deep authoring API contract documented here.

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
- `chelis_deep_outline`
  - request: `{module}`
  - success: module/function outline with canonical Deep and `preimage_sha256`
    over each addressed `(def ...)` node
- `chelis_deep_references` and `chelis_deep_call_graph`
  - request: `{module, symbol}` for references and `{module}` for call graph
  - success: scope-aware structural references/call edges; shadowed parameter
    and local-binding names are not reported as top-level function references
- `chelis_replace_function`
  - request: `{module, function_name, new_decls, preimage_sha256?}`
  - success: `{replaced_def_deep, replaced_defsig_deep?, module_deep}`
- `chelis_add_property`
  - request: `{module, new_decls, insert_after_function?}`
  - success: `{added_property_def_deep, added_defsig_deep?, module_deep}`
- `chelis_rename`
  - request: `{module, function_name, new_name, preimage_sha256?}`
  - success: `{renamed_def_deep, renamed_defsig_deep?, module_deep,
    renamed_references}`
- `chelis_change_signature`
  - request: `{module, function_name, new_defsig, new_params,
    argument_order, param_renames?, preimage_sha256?}`
  - success: `{changed_def_deep, changed_defsig_deep, module_deep,
    rewritten_calls}`

The compiler-api seam for additional tools is:

1. Parse request Deep with strict parser.
2. Check only request shape and structural target resolution locally.
3. Apply the AST edit through `chelis-deep`.
4. Run `check_whole_module_edit`.
5. Return canonical Deep only from the accepted module.

Cascade-tool rule: whole-module validation is necessary but not sufficient for
structural correctness. `chelis_rename` asserts no residual unshadowed old-name
references after the cascade. `chelis_change_signature` records the pre-edit
direct call graph and asserts every direct callsite was rewritten according to
`argument_order`; stale direct calls fail before success is reported.

Optimistic concurrency rule: `preimage_sha256` is SHA-256 over the canonical
Deep of the addressed `(def ...)` node returned by `chelis_deep_outline`. A
mismatch fails closed at `stage:"preimage"` with no result payload.

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

Next edit tools are now incremental refinements, not substrate blockers:
rename-by-module-qualified disambiguation in multi-module payloads, richer
semantic references beyond direct function calls, and higher-level property
authoring helpers over the raw Deep property bundle.

The next substantial colleague pickup is prove parity, not more edit substrate:
broaden `.dp` property lowering beyond the scalar SMT-amenable subset while
preserving the same per-edit oracle shape for Deep-authored properties.

## Known Limitations

- `.dp` prove parity is partially complete. Chelis #507 remains open even
  though the direct Deep property-to-SMT path is implemented for scalar
  SMT-amenable properties and call-form boolean connectives; unsupported
  property shapes still fall back or report unsupported per the existing prove
  tier policy.
- Related live prove/lowering gaps remain adjacent, not substrate blockers:
  Chelis #434 (transcendental finance SMT), #506 (scalar WireDag root), #513
  (symbolic-dim-aware grad machinery), and #423 (standalone eval package
  imports). Chelis #463 and #496 are closed as of the 2026-07-06 remote audit.
- Shoals #19 is downstream work to expose a tensor Black-Scholes entry as a
  WireDag root for Beacon; Beacon currently has no open issue/PR that blocks
  this substrate.
- Deep-path diagnostics are not populated for type/effect/linearity errors.
  Query/edit-owned failures can identify addressed functions; compiler pass
  diagnostics still carry `deep_path: None`.
- No runtime sandbox, effects distribution, FlukeBall harness, Beacon proof
  path, or `.dp` prove completeness is claimed here.

## Handover Oracle

The named milestone oracle is the `Deep authoring L2 query/cascade + .dp SMT
parity` row in `docs/phase_oracles.md`, plus the full `scripts/gate.py` run
before merge.
The focused suite covers:

- replacement trust-boundary regressions over MCP and HTTP
- `add_function` compiler-api, MCP, and HTTP behavior
- query/cascade authoring behavior in `chelis-deep`, compiler-api, MCP, and HTTP
- Deep `.dp` SMT property proof for the supported scalar subset
- duplicate `defsig` rejection in type-check and validation
- canonical `chelis surf` round-trip behavior

Before merge, a fresh-context red team must execute the malformed-body,
duplicate-`defsig`, `chelis_add_function`, preimage, cascade-completeness, and
Deep `.dp` SMT proof cases.
