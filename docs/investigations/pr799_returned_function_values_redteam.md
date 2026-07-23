# PR #799 Red Team: Returned Function Values and the Phase 2 Loud-Unsupported Contract

- **Date:** 2026-07-23
- **Reviewed head:** `5a4e755ebf88dab511baed9607831b120aaed4c8` (detached from `origin/agent/730-phase2-ratchets`)
- **Reviewer:** fresh-context local subagent (Fable 5) in an isolated git worktree, per the
  repository red-team protocol (`redteam-exec`). Execution-based: oracle, targeted suites,
  and 24 self-authored adversarial `.ch` probes; no main-thread validation counted.

## 1. Verdict

**PASS WITH FINDINGS** for Phase 2 acceptance.

The safety half of the contract survived every attack mounted: no rejection path ever
emitted a C artifact, no placeholder value, `void *`, zero, or unit substitution was
reachable, all positive cells produced exact expected outputs from compiled-and-executed
C, and the authoritative oracle passes. All findings are diagnostic-quality,
false-positive-rejection, or documentation-honesty issues; none is a silent-substitution
or unsound-emission hole. Findings 1, 3, 4 deserve tracking issues (finding 1 arguably
belongs in the Phase 3 gate work it exposes).

## 2. Findings

### Major

**F1 - CLI/compiler-API diagnostic drift on the PR's named focus (returned/dynamically-selected function values).**
When a returned named function or dynamically selected callback is bound and *used*,
`chelis build --target c` rejects via the legacy grad/vmap gate at
`crates/chelis-cli/src/main.rs:2591` (`host_program_unresolved_call_sites`), producing:
`` `chelis build --target c` can't lower these defs. Their body applies/binds `grad` (or `vmap`) ... ``
for programs containing neither `grad` nor `vmap`. The public compiler API for the *same
source* produces the frozen `unsupported: ... function value ... (codegen:c) ... chelis#730`
diagnostic. The PR's own test (`assert_named_function_value_has_no_c_abi` in
`crates/chelis-compiler-api/tests/pr799_host_resolution_result.rs`) asserts "callable
rejection must not be misclassified as an AD transform failure" - but only on the API
surface; the CLI violates that exact principle for the identical fixtures.

- Repro (CLI, misleading): write the exact source of
  `used_returned_named_function_rejects_before_an_unresolved_c_call_is_emitted`
  (`def choose() -> int8 -> int8 = increment; chosen = choose(); out = print(chosen(cast(6, int8)))`)
  to `p.ch`, then `CHELIS_STYLE_GATE_DISABLE=1 chelis build p.ch --target c --output out`
  -> exit 1, grad/vmap message, no artifacts.
- Repro (API, frozen diag): the passing pr799 test itself; re-proved with a temporary API
  test (removed after).
- Also affected: conditional selection (`selected = if true then increment else decrement`),
  two-hop returns, `apply(make(), 5)`, ADT-field function retrieved-and-called.
  NOT affected: *unused* returned named function (frozen diag on CLI too), inline lambdas
  (frozen lowering diag on CLI).
- Safety is intact (exit 1, zero artifacts, before codegen). Phase 3 owns gate resolution,
  but this is the acceptance focus of this PR and the drift is live today.

**F2 - `chelis build` succeeds but emits non-compiling C for a def named `double` (C keyword). Likely pre-existing.**
`def double(x: int32) -> int32 = mul(x, 2)` + `map(double, [1, 2, 3])` builds with exit 0
(style gate active or not), and the emitted `static inline int32_t double(int32_t x)`
fails at clang with `redefinition of 'int32_t' as different kind of symbol`. No silent
wrong value (the user's compiler fails loudly), but the build's success report is
dishonest about producing a usable artifact, and `EmittedExpr`/`CIdentifier` validate
lexical shape only - no C-reserved-word mangling anywhere on the host emission path.
Almost certainly predates this PR (verbatim name emission is old); it sits inside the
emission surface this PR hardened, so it should be filed, not fixed silently here.

### Minor

**F3 - In-band name sentinels reject valid programs with wrong-reason diagnostics.**
`crates/chelis-backend-c/src/host_abi.rs` `project_expr` rejects any call where
`function == "call"` or `function.starts_with("__unresolved_")`, and the host lowerer
uses the same in-band strings. Consequences, both verified: a user def legitimately named
`call` evals fine (`6`) but cannot be built for C - rejected with F1's grad/vmap message;
a def named `__unresolved_x` is rejected with the branded diagnostic but its prose
falsely claims "the host lowerer did not resolve this application". Fail-closed, so not
unsafe, but a legal snake_case identifier makes a supported program shape uncompilable
(violates the B2.4 "supported neighbor still works" discipline). The string sentinels in
the new projection layer freeze this collision.

**F4 - spec §C6.2 claims an "added-dtype mutation oracle" that does not exist.**
`spec/design/loud_unsupported.md:502` states the added-dtype mutation oracle "requires a
compile error at every exhaustive consumer". No script or test performs a RuntimeDType
mutation; the authoritative oracle mutates `EffectKind` only (and the spec's own oracle
description at lines 764-781 says so). The missing mutation was performed manually during
this review (temporary fully-decodable `RedTeamProbe = 9` variant, owner restored
byte-for-byte): the *property* holds - 13+ `E0004` non-exhaustive errors across
`crates/chelis-runtime/src/lib.rs` - but the doc claims an executable oracle stronger
than what ships. Fix the sentence or add the mutation to the oracle.

### Nit

**F5 - `UnsupportedKind::HostType` renders "unresolved host type" for resolved-but-unrepresentable values.**
Function-value rejections read
`` unsupported: unresolved host type `function value `Function([Scalar(Int8)], Scalar(Int8))`` ... ``
- the type *did* resolve; the C ABI rejected it. The structured kind distinction (§C6.3:
"known logical type without target representation") is blurred in the rendering. (Dtype
rejections correctly use `UnsupportedKind::Dtype`.)

**F6 - `process_run` build rejection wording drifts between surfaces.**
CLI gate: unbranded "process_run is an eval/test-only builtin; not available in compiled
targets...". Compiler API (no gate): branded
`` unsupported: builtin `process_run` on `chelis build` host emission (codegen:c); ... `` -
proven loud via a temporary API test, so the emitter channel (the safety boundary) is
fail-closed either way. Gate-vs-emitter wording alignment is Phase 3's contract;
recording for that inventory.

**Environment note (not a code finding):** the workstation's Data volume hit ENOSPC
mid-run (it was at ~99% before the review started; cold builds tipped it). One suite
invocation (`host_type_failure_states`, first attempt) failed spuriously as a result and
passed 11/11 on retry after deleting the oracle's temp `CARGO_TARGET_DIR` cache
(`$TMPDIR/chelis-loud-unsupported-phase2-target`, ~5 GiB). The review worktree's
`target/` (~6 GiB) was removed after finishing. The oracle's use of a shared tempdir
cache also means its first "cold" run on this machine reused a pre-existing warm cache.

## 3. Commands run

Setup: `git fetch origin agent/730-phase2-ratchets && git checkout --detach FETCH_HEAD`
(head `5a4e755e`); `uv venv --python 3.11`; `.venv/bin/python scripts/reap_orphans.py`
(no orphans, before and after).

Oracles and script tests:

- `.venv/bin/python scripts/loud_unsupported_phase2_oracle.py` -> exit 0, final line
  `PHASE 2 ORACLE: PASS`; mutation stage reported "mutation produced the expected
  downstream compile failures"; internal summaries 7, 5, 11, 9(+375 skipped), 16,
  12(+7 skipped) all passed.
- `.venv/bin/python scripts/test_loud_unsupported_phase2_oracle.py` -> 6/6 OK.
- `.venv/bin/python scripts/test_release_runtime_header_manifest.py` -> 2/2 OK.

Targeted suites (all `cargo nextest run` in the review worktree's own target):

- `chelis-vocab --test closed_vocabulary` -> 7 passed, 0 skipped
- `chelis-runtime --test runtime_dtype_generated_header --test runtime_dtype_invalid_ffi` -> 5 passed
- `chelis-ir --test host_type_failure_states` -> 11 passed (first attempt exit 1 = ENOSPC, rerun green)
- `chelis-ir --test span_threading_through_host_lowering` -> 14 passed
- `chelis-compiler-api --test pr799_host_resolution_result` -> 16 passed
- `chelis-cli --test closed_vocabulary_architecture` -> 3 passed
- `chelis-cli --test loud_unsupported_tripwire` -> 6 passed
- `chelis-cli --test loud_unsupported_phase1` -> 6 passed
- `chelis-cli --test issue_734_tostring_placeholder` -> 2 passed, 2 skipped;
  `--run-ignored all` -> the 2 ignored fail as designed (they assert future #732
  tensor/list stringification support)
- `chelis-cli --test int_width_lane_matrix` -> 7 passed (incl. all 3 callback-execution
  and both generic-record-execution cells with a real `cc`), 5 skipped (ignored
  overflow-trap contract cells)
- `chelis-cli --test issue_687_rejected_cells_corpus` -> 3 passed
- `chelis-cli --test rt791_redteam_probes` -> 13 passed
- `chelis-backend-c -E "test(emitted_expr::tests) | test(host_abi_tests)"` -> 9 passed
- `chelis-backend-c --test host_emit_dtype_dispatch` -> 11 passed
- `cargo test -p chelis-backend-c --doc` -> 2 passed (both privacy compile-fail doctests)

Adversarial probes (all via `target/debug/chelis` with `CHELIS_STYLE_GATE_DISABLE=1`,
each through `build --target c`, `check`, `eval --file`; positive builds also
clang-compiled with the emitted flags and executed): 24 probe programs - r_ret_named,
r_ret_lambda, r_ret_cond, r_ret_twohop, r_ret_capture, r_ret_apply_direct,
x_api_fixture_named_used/unused/selected, d_adt_fn_field, d_list_fn, d_tuple_fn,
d_generic_adt_fn, dy_dynamic_arg, f_f16_scalar, f_bf16_scalar, f_bf16_callback,
a_def_named_call, a_def_named_unresolved, s_process_run, s_tostring_tensor,
s_tensor_scan, i_empty_list, h_never_fail,
p_pos_{i8_named,i16_named,forward,inline,map_named,map_named2}. One temporary Rust API
test (2/2, then deleted) and one temporary RuntimeDType mutation script (owner restored
byte-for-byte, verified via `git status`).

## 4. Coverage map

| Claim | Validation |
|---|---|
| `chelis-vocab` dependency-free, no Unknown/Default, Result-only decoders | Source read (crate has zero deps, no `Default`, decode returns `Result` preserving raw ID/symbol) + suite 7/7 + oracle |
| Exhaustive EffectKind dispatch at every consumer | Oracle's controlled EffectKind mutation (required non-exhaustive evidence in `chelis-surf/decompile.rs` + `chelis-types/infer.rs`, the two independently-checkable roots) + `closed_vocabulary_architecture` 3/3 |
| RuntimeDType nine IDs/spellings/macros/widths; byte-locked header | Suites 5/5; `chelis_runtime_dtype.h` shipped in every build output and byte-identical to the checked-in artifact (diff -> IDENTICAL); generated C size switch has aborting `default` (source) |
| Runtime FFI decodes before sizing/allocation/access; invalid IDs (-1, 9, extrema) abort | `runtime_dtype_invalid_ffi` executed (subprocess tests) |
| Added-dtype exhaustiveness | Executed manual mutation probe: 13+ E0004 errors in chelis-runtime; property holds, claimed oracle missing (F4) |
| HostTypeTerm -> ConcreteHostType -> HostAbiType staging; errors cannot become terms/types | `host_type_failure_states` 11/11; `pr799` 16/16; empty-list CLI probe -> typed "unresolved host inference variable" error, no panic, no artifact |
| Fallible host-expression lowering; no Unit placeholder | Frozen `unsupported: anonymous function value` lowering diagnostics on lambda probes; oracle endpoint scan; tripwire 6/6 |
| Generic ADT substitution before boundary | Lane-matrix direct+nested generic int8 cells executed exactly (7); pr799 bf16-preserving rejection; generic-ADT-with-function-field probe rejected |
| Callback ABIs contextual-only; returned/dynamic/stored callables reject pre-codegen | 12+ executed rejection probes: every one exited nonzero with ZERO artifacts; inline/named/forwarded/int8/int16 positives compiled and ran with exact outputs 7/302/7/7; emitted C carries exact `int8_t (*f)(int8_t)` / `int16_t (*f)(int16_t)` declarators, no `void *` (grep of all positive artifacts). Diagnostic drift = F1 |
| f16/bf16 typed `UnsupportedKind::Dtype` rejection; exact elsewhere | Scalar + callback probes: exact `` unsupported: dtype `f16|bf16` on C host ABI selection (codegen:c); ... chelis#714 ... ``; eval computes 0.75 for both |
| EmittedExpr crate-private, no raw constructor | Source read + 2/2 compile-fail doctests + unit tests (identifier-laundering panic test) + oracle scan |
| Shared checked-Deep host-only-builtin scan (no surface drift) | tensor_scan CLI probe -> branded diagnostic; both surfaces call the same `chelis_ir::host::find_direct_builtin_call` + `Unsupported::compiled_host_only_builtin` (source identity); process_run API probe loud (F6 wording note) |
| Release manifests ship dtype header; installed-layout preprocess test | `test_release_runtime_header_manifest.py` 2/2 |
| No silent placeholder smuggling | to_string(tensor) -> frozen diag; process_run -> loud both surfaces; artifact greps clean; rt791 13/13; rejected-cells corpus 3/3 |
| Negative-test parity | Ignored tests fail for documented right reasons (issue_734 future-support cells); gap: no CLI-lane locks for used-returned-callable diagnostics - the review's probes filled it and exposed F1 |
| Docs honesty | `agent_quality_architecture.md` diff honest (lint retraction documented); spec Phase 2 status honest ("ACCEPTANCE PENDING"); one overstatement = F4 |

## 5. Unvalidated

- **Full workspace nextest** - excluded per environment constraints (chelis#356);
  GitHub CI on this head is the workspace oracle.
- **PR body's "790/790 across chelis-backend-c and chelis-compiler-api"** - focused
  subsets of those crates were run, not the full pair.
- **HIP and Metal lanes** - not exercised beyond source inspection of
  `dtype_macro`/`msl_type`; no GPU or hipcc runs.
- **`.dp` (Deep) build-lane callable diagnostics** - the twin gate at `main.rs:2852`
  was not probed; F1 may have a `--deep` sibling.
- **Pre-existence of F2/F3 on origin/main** - confirming would need a second full
  toolchain build; disk pressure made that imprudent. Both are consistent with pre-PR
  behavior (verbatim name emission, the old `__unresolved_grad` gate) but not proven.
- **chelis-python vocab consumption** - source-inspected only (`RuntimeDType::F32.id()`
  replaces the local constant); not executed.

## 6. Disposition (2026-07-23, post-review)

- **F1 - fixed in this PR.** The CLI gates (both lanes) now use
  `host_program_unresolved_transform_sites`, which matches only
  `__unresolved_grad`/`__unresolved_vmap` marker builtins; plain unresolved
  callable values fall through to ABI projection's frozen `unsupported:`
  function-value diagnostic, matching the compiler API. Locked by
  `crates/chelis-cli/tests/pr799_callable_diagnostic_surface.rs` and the
  classifier unit tests in `crates/chelis-ir/src/host.rs`.
- **F2 - filed as [chelis#840](https://github.com/Chelis-Lang/chelis/issues/840)**
  (standalone emission fix at the `CIdentifier` chokepoint; ledgered in
  `spec/design/remediation_roadmap.md`).
- **F3 - filed as [chelis#841](https://github.com/Chelis-Lang/chelis/issues/841)**
  (typed marker vocabulary, sequenced after PR #800's `ErrorWitness`;
  ledgered). The not-an-AD-transform half is regression-locked in this PR.
- **F4 - fixed in this PR.** The Phase 2 oracle's controlled mutation now
  adds a fully-decodable variant to BOTH vocabularies (`EffectKind` and
  `RuntimeDType`) in one workspace check and requires non-exhaustive-match
  evidence in `crates/chelis-runtime/src/lib.rs`; the spec sentences in
  `spec/design/loud_unsupported.md` now describe exactly what ships.
- **F5 - fixed in this PR.** Resolved-but-unrepresentable function values
  carry the new `UnsupportedKind::HostAbi` ("with no target ABI
  representation"); `UnsupportedKind::HostType`'s "unresolved" rendering is
  reserved for terms that never resolved.
- **F6 and the `.dp` build lane** - recorded on
  [chelis#730](https://github.com/Chelis-Lang/chelis/issues/730) for the
  Phase 3 gate-dedupe inventory.

## 7. Amendment (2026-07-23, chelis#841 / PR #843)

The F1 disposition above describes the mechanism as of PR #799's head:
`host_program_unresolved_transform_sites` keyed on `__unresolved_*` marker
builtins plus a whole-program grad/vmap tag detector. PR #843 (closing
chelis#841) replaced both in-band string sentinels with unspellable
markers (`#chelis-unresolved-callable` / `#chelis-unresolved-transform`)
chosen per call site by the callee's tag; the transform scan is re-keyed
to the transform marker, the tag detector is gone, and the F3 collision
class is closed structurally. This section records the drift so the
disposition text stays readable as history.
