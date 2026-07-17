# Loud Unsupported: the failure-channel contract

**Status:** Design proposal, pre-implementation. Tracking issue: [#730].
**Owning specs:** `spec/05-risc-primitives.md` (op support statements;
its §7 carries this plan's decided contract as provisional atoms
[05-UNS-1..4], seeded ahead of Phase 1, which ratifies them),
`spec/04-type-system.md` §1.1.1 (deferred dtypes precedent), the repo
Contract Invariants ("if a command reports perfect success, its error list
must be empty"), and the audit record in
`docs/investigations/numeric_audit_next_sweeps.md` /
`docs/investigations/numeric_audit_structural_prevention.md` (items 3, 8).
**Class fixed:** [#703] (unsupported cases silently substitute a value
instead of failing). Sibling plans: `spec/design/dtype_semantics.md`
([#729]) owns what SUPPORTED cells compute; this plan owns what every
UNSUPPORTED encounter does. §I1 pins the interlock. [#709]'s checker
analogue is item 5 of the prevention doc and is NOT this plan (see
Non-goals).

## Summary

When Chelis hits a case it does not support, it substitutes a plausible
value - a literal `0`, a zero constant with the operand dropped, an f32
kernel over an int64 buffer, an empty window list, a discarded effect
handler - and compiles a working program. The user gets a binary that runs,
returns numbers, and is wrong. The 2026-07 audit confirmed ten live
substitution sites across four crates and measured the failure mode's
signature property: **the output is always plausible** (`add(abs(x),
[1,1,1,1])` returning `[1,1,1,1]` computed `0 + 1` perfectly correctly on a
zeroed operand).

The class persists for a structural reason, not a discipline reason: **most
substitution sites live inside functions that have no way to fail.** The
C emitter's expression builders return bare `String`; the host-type parser
returns bare `HostType`; extraction helpers return bare `Vec`. A function
that cannot return an error will always invent a value. Meanwhile the
codebase contains all three possible responses to the same situation -
silent substitution (the bug), internal panic (loud but user-hostile,
[#692]), and clean diagnostic (`lower_unsupported`,
`chelis_int_div_guard`, HIP's narrow-float rejection) - because each site
chose independently.

This plan: (1) gives every stage a **failure channel** (Result-typed
emission; lowering already has `raise_lowering_error`), (2) converts every
censused site to the diagnostic row, (3) makes new silent fallbacks
**unwritable or undetectable-free** via three ratchets (a chelis-lint rule
against wildcard arms on closed enums, an `Unsupported`-only fallback type
for open-set dispatch, and a token tripwire), and (4) demotes the
pre-codegen gates from safety mechanism to early-UX, because a gate that is
the only line of defense rots (the one-entry `HOST_ONLY_BUILTINS` allowlist
let five builtins walk past it, [#682]/[#705]).

**Verdict on scope** (the question this document answers first): not a
rewrite. No storage, no wire format, no semantics change. One moderate,
mechanical plumbing refactor (the emitter Result channel), a bounded
remediation sweep over a censused site list, and permanent guardrails. The
long-tail guarantee for op x dtype cells arrives via [#729] Phase 4's
capability table and is deliberately not duplicated here.

## Why not a pile of patches, in one table

The audit's recurrence signature, now at four sightings: the correct
mechanism existed next door and was not called.

| correct mechanism | built for | was not applied to | result |
|---|---|---|---|
| `checked_int_binop` | div/mod ([#387]) | add/sub/mul/... | [#680] |
| `raise_lowering_error` | `lower_unsupported` | `lower_transcendental` | [#699] |
| `fold_static_size`'s `checked_i64` | extent folding | `fold_static_cond` | [#711] |
| `raise_lowering_error` (again) | same file | `reduce_window` extraction | [#725] |

Optionality is the root cause. A patch adds another optional mechanism;
this plan removes the option: after Phase 1 the fallible signature is the
only signature, and after Phase 2 the silent arm is a lint/build failure.

## Non-goals

- **Not [#709].** The checker's `Type::Error`-without-diagnostic hole is the
  same disease in a different organ with a different fix - now its own
  plan (`spec/design/checker_totality.md`, [#731]). One shared piece
  lands here: the `EffectKind` enum (§C4.4), because its catch-all is a
  lowering-side substitution proven live via `.dp`.
- **Not [#727]/[#729].** This plan never decides what an op computes or which
  cells are supported. Where a site is both silently-substituting AND
  semantically-wrong-when-fixed (f16 scalars, [#714]), this plan delivers
  the loud rejection and [#729] delivers the support (§I1).
- **Not [#728].** The print helper's `default:` arm is censused here (it is
  a substitution) but its fix ships with the faithful-observation plan
  (`spec/design/faithful_observation.md`, [#732]) Phase 2's generated
  formatter; this plan only requires that until then the arm aborts with
  the dtype id rather than misreading (§C1 rule 4 applies to it).
- **Not a diagnostics-UX project.** §C2 fixes the *shape* of unsupported
  diagnostics; wording polish beyond the contract is out of scope.

## Vocabulary

- **Substitution site** - a code location that, on an unsupported input,
  produces a value/type/emission instead of an error. The unit of the
  census (§C5).
- **Live / dead site** - reachable from user input today / provably
  guarded upstream. The audit measured that dead-by-inspection claims are
  wrong often enough that liveness is only ever established by execution
  (five source-derived claims refuted; [#725] predicted dead, was live).
- **Failure channel** - the typed path by which a stage reports
  "unsupported" to its caller and ultimately to the user.
- **Ratchet** - a mechanism that makes the count of silent fallbacks
  monotonically non-increasing (lint, tripwire, type privacy).

---

# Part I - the normative contracts (§C1-§C5)

## C1. The response contract

Normative, per encounter with an unsupported case, in every stage:

1. **Substitution is forbidden.** No stage may produce a value, type,
   dtype, kernel, emission, or silently-narrower behavior as its response
   to an unsupported input. This includes "temporary" placeholders; the
   audit's placeholders shipped.
2. **The response is a diagnostic** through the stage's failure channel
   (§C3), at the **earliest competent stage**:
   - the **checker**, when the question is answerable from types alone
     (this routing is [#709]/[#729]-territory; this plan does not move checks
     earlier, it only guarantees no later stage is silent);
   - the **build/lowering stage**, when the target or lowering context
     decides it (unsupported builtin for target C, unsupported dtype for a
     kernel family, non-literal window);
   - the **runtime**, ONLY for genuinely dynamic conditions (a dtype or
     value knowable only at run time), as a branded abort - the
     `to_tensor` narrow-float abort is the accepted shape.
3. **Panics are for broken compiler invariants, never for reachable
   input.** A `panic!`/`assert!` is legitimate only when the front end
   already guarantees the condition and that guarantee is named in the
   panic message. Every panic reachable from `.ch`/`.dp` source is a bug
   ([#692], [#725]'s assertion at `emit.rs:4509`). Rule of thumb: if a
   test can trigger it through the CLI, it must be a diagnostic.
4. **Dead placeholders raise anyway.** For sites believed unreachable
   (the ~25 guarded `RiscOp::Const { value: 0.0 }` sites in `lower.rs`),
   the policy is *raise-or-prove*: replace the placeholder with a raise
   (costs nothing on a dead path), or keep it only alongside an executable
   canary that drives the guard and a comment citing it. Belief without a
   test is not an option; the audit's one "expect these to be refuted too"
   prediction that was wrong ([#725]) is why.
5. **Success reports stay honest**: any command that emitted an
   unsupported diagnostic exits nonzero with a non-empty error list (the
   repo Contract Invariant, violated today by check on [#709]'s cases -
   tracked there).

## C2. The diagnostic shape contract

One error kind, one shape, every stage. Frozen at Phase 1 exit.

```rust
pub struct Unsupported {
    /// What was encountered: an op, builtin name, dtype, tag, effect
    /// kind, or construct. Closed enum + payload, not a bare string.
    pub what: UnsupportedKind,
    /// Which stage refused (checker | lowering | codegen(target) | runtime).
    pub stage: Stage,
    /// Source span when one exists (lowering/codegen must thread it;
    /// `raise_lowering_error` already takes span + span_id).
    pub span: Option<SpanRef>,
    /// The supported alternative, when one exists. Not optional prose:
    /// sites without an alternative say why (deferred per spec §X, file
    /// an issue, etc.).
    pub hint: &'static str,
}
```

**Message format (frozen):**
`unsupported: <what> on <context> (<stage>); <hint>` - branded with the
literal prefix `unsupported:` so tests and shells can match it. The three
existing exemplary messages are the calibration set and must remain
conformant when migrated:

- HIP: `` `chelis build --target hip` admits `f16` only on tensor
  load/store nodes ... See spec/04-type-system.md §5.7.1`` (names the
  construct, the boundary, and the spec);
- Metal: `` `chelis build --target metal` rejects f64 ... `` ;
- runtime: `to_tensor: unsupported destination dtype ...`.

**Surfacing per surface:** `chelis check` -> JSON error entry, score < 1,
carrying the machine-readable kind (`unsupported`) and the `what` payload
as structured fields - the branded string is the RENDERING of the
contract, not the contract; agents match the structured kind, never
regex over prose (2026-07 review integration);
`chelis build`/`eval` -> `error:` line + nonzero exit; compiled binary
(dynamic-only cases) -> stderr + nonzero exit. The [#687] oracle corpus
gains a rejected-cells section asserting these strings byte-for-byte per
lane (a rejection emitted differently per lane is lane skew, [#712]'s
shape).

**Authored vs accidental (2026-07 review):** a deliberately-unsupported
case cites the spec atom that decides it (the [#733] linkage); a
not-yet-implemented case cites its issue. The capability table makes
the distinction structural - `Rejected(atom)` vs `Unimplemented
{ issue }` - and this contract's `hint` carries the same citation at
the diagnostic surface, so "unsupported by decision" and "unsupported
because nobody built it yet" are never conflated again.

## C3. The failure channel

Per stage, what exists and what this plan builds:

| stage | today | after Phase 1 |
|---|---|---|
| lowering (`chelis-ir/src/lower.rs`) | `raise_lowering_error` exists and works (used by `lower_unsupported`) | unchanged mechanism; every censused lowering site calls it |
| host-type resolution (`chelis-ir/src/host.rs`) | infallible `-> HostType`, silent `Unknown`/`Int64` defaults | numeric-prim resolution returns `Result<HostType, Unsupported>`; `Unknown` remains legal ONLY for genuinely polymorphic signatures (the documented rank/precision-poly cases), never as a numeric fallback |
| C host emitter (`chelis-backend-c/src/host_emit.rs`) | **no channel**: expression builders return `String`, statement emitters return `()` | expression builders return `Result<EmittedExpr, Unsupported>`; statement emitters `Result<(), Unsupported>`; the compile entry surfaces the first error as a §C2 diagnostic. This is THE plumbing refactor - mechanical (`?` all the way up), moderate in size, and the enabling move for everything else |
| DAG C emitter (`chelis-backend-c/src/emit.rs`) | panics ([#692]) | same Result channel; the reduce-family panic arms become diagnostics |
| HIP emitter | gate rejects much; `elem_kind` substitutes F32 ([#689]) | `elem_kind` returns `Result`; its `_` arm deleted (§C4.1) |
| runtime (`chelis-runtime`) | `runtime_fail!` aborts exist and are the right shape | unchanged; used only for dynamic-only cases per §C1.2 |

**The `EmittedExpr` rule:** the C expression type is a newtype over
`String` whose constructors are the typed builders. There is deliberately
no `EmittedExpr::raw(String)` on the unsupported path - the only way to
respond to an unmatched builtin is to construct `Unsupported`. The audit's
`format!("/* unsupported builtin {other} */ 0")` becomes unwritable, not
just unfashionable. (A `raw` constructor may exist `pub(crate)` for the
legitimate template snippets; the lint in §C4.2 patrols its use sites.)

## C4. The ratchets (making reintroduction near-impossible)

Four mechanisms, ordered by strength. §C3 removes the *need* to
substitute; these remove the *ability*:

1. **Closed enums match exhaustively.** No `_` arm over `Prim`,
   `RiscOp`, `ElemKind`, `UnsupportedKind`, or the CHELIS dtype ids in the
   numeric/backends crates. Adding a variant then produces a compile-error
   work-list at every dispatch site. Where an arm is genuinely N/A it says
   so per-variant (`Prim::String => unreachable-per-<cited guarantee>` or
   a §C1.4 raise).
2. **The lint rule** (new, in `chelis-lint`, which already lints Rust
   source for §8.6): `rust-no-wildcard-dispatch` - a blocking rule that
   forbids `_ =>` / `default:`-emitting arms in matches over a configured
   list of enum types within configured crates, with a per-site allowlist
   file that requires a written justification string. The rule also flags
   `unwrap_or_default()` / `unwrap_or(Prim::` / `unwrap_or(HostType::` in
   the same crates. This is the ratchet's enforcement: rustc cannot forbid
   wildcards; the repo gate can.
3. **The token tripwire** (test, land-first, cheapest): a unit test in the
   workspace that greps the source tree for the recidivist tokens -
   `*/ 0"`, `<value>`, `unwrap_or_default()` and `unwrap_or(Prim::` in
   lowering/emission paths, `=> ElemKind::` wildcards - against the §C5
   census. Any NEW site fails with a message pointing at this document.
   Exists purely to bridge until 1+2 land and to catch generated-string
   contexts the lint cannot see.
4. **`EffectKind` becomes an enum** (shared deliverable with [#709]'s plan):
   `lower_handle_effect`'s `_ if elems.len() >= 4 => lower body, drop
   handler` catch-all is deleted; unknown effect kinds in `.dp` input get
   a §C2 diagnostic. (Proven live today: `effect: teleport` builds and
   runs.)

## C5. The census (normative appendix; Phase 0 re-verifies by execution)

The remediation work-list, from the audit record as of 2026-07-16.
**Live** = confirmed by execution.

| # | site | substitutes | issue | status |
|---|---|---|---|---|
| 1 | `lower.rs` `lower_transcendental` non-float arm | `Const 0.0`, operand dropped | [#699] (+[#722] via grad) | live (test `cos_on_integer_tensor_is_not_silently_zeroed`) |
| 2 | `host_emit.rs:2300` builtin fallback | literal `0` | [#682] [#704] [#705] [#715] | live (test `no_build_ever_emits_a_silent_unsupported_builtin_stub`) |
| 3 | `host_emit.rs:2236` string fallback | `chelis_string_from_cstr("<value>")` | [#734] | **live** (test `to_string_of_a_tensor_stringifies_in_the_compiled_lane`; probed 2026-07-16, status carried by P0 - scalar-arm controls green in `issue_734_tostring_placeholder.rs`) |
| 4 | `host_emit.rs:4305/4390` print of unclassifiable value | literal `<value>` text | [#714] symptom | live (test `c_f16_floor_prints_the_value_placeholder_today`) |
| 5 | HIP `emit.rs` `elem_kind` `_` arm | `ElemKind::F32` | [#689] | live (test `hip_int64_neg_emits_the_f32_fallback_kernel_today`; emission-proven) |
| 6 | `host.rs:7572-7583` `parse_host_type` `_` arm | `HostType::Unknown` -> downstream `int64_t`/`void*` | [#714] | live (tests `f16_scalar_abs_compiles_and_runs`, `f16_scalar_fraction_survives_compilation`) |
| 7 | `host.rs:7861-7882` arithmetic type default | `HostType::Int64` | [#714] [#718] | live (tests `f16_scalar_fraction_survives_compilation`, `c_scalar_overflow_traps_at_every_width`) |
| 8 | `lower.rs:7522-7537` reduce_window extraction | `vec![]` windows -> silent no-op | [#725] | live (test `c_nonliteral_window_and_strides_pool_or_reject`) |
| 9 | `lower.rs:9518-9519` handle-effect catch-all | drops handler, lowers body | [#709]-adjacent | live via `.dp` (test `unknown_effect_kind_is_rejected`) |
| 10 | emitted print helper `default:` arm | reads buffer as f32 | [#716] ([#728] owns fix) | live (test `c_print_of_f16_tensor_prints_f16_values`) |
| 11 | `emit.rs:4238/4406/4777` reduce panics | (row-two: panic, not substitution) | [#692] | live (test `int64_max_reduce_does_not_panic_the_compiler`) |
| 12 | `emit.rs:4509` window-length assert | (row-two) | [#725] half | live (test `c_nonliteral_window_does_not_panic_the_compiler`) |
| 13 | `lower.rs:9833`, `:10235` `unwrap_or(Prim::F32)` | F32 dtype | [#710]-adjacent, [#744] | **live via `.dp` build lane** for `:10235` (test `dp_bogus_cast_target_must_not_build_silently`; P0 re-execution refuted the dead-by-probe claim - the guard is eval-only, [#744]); eval guard locked green (canary `canary_dp_cast_bogus_dtype_is_guarded`); `:9833` needs an internal desync; §C1.4 applies |
| 14 | `named_axis.rs:430` `unwrap_or(Prim::F32)` | F32 dtype | audit item 7 | dead (reachable-surface clearance: `canary_vmap_int64_roots_keep_integer_precision`; the arm is internal-desync-only, undrivable from input); §C1.4 applies |
| 15 | `host_emit.rs` `assign_partition` non-tuple arm | emits a C comment, no assignment | audit item 7 | dead (reachable-surface clearance: `partition_agrees_across_lanes`; the arm is internal-desync-only, undrivable from input); §C1.4 applies |
| 16 | ~25 guarded `Const { 0.0 }` sites in `lower.rs` | zero values | backlog §pattern | dead (canaries `canary_unknown_deep_tag_is_rejected`, `canary_bare_keyword_atom_fails_cleanly`, `canary_dynamic_fail_aborts_loudly`); §C1.4 applies |
| 17 | `HOST_ONLY_BUILTINS` one-entry allowlist | (gate, not site - lets sites 1-2 fire) | [#682] [#705] | live (test `tensor_scan_does_not_silently_compile_to_a_stub`) |
| 18 | duplicated/drifted gates | (gate skew) | [#697] [#698] | live (tests `hip_int64_neg_emits_the_f32_fallback_kernel_today` for the [#698] half, `int64_max_reduce_does_not_panic_the_compiler` for the [#697] half) |
| 19 | Metal `emit.rs:1419` `host_scalar_literal` pad-fill catch-all | `/* unsupported pad fill dtype */ 0` | [#745] (P0 token-sweep discovery, B2.5) | dead (canaries `metal_rejects_f64_with_a_specific_diagnostic`, `f8e4m3_is_rejected_in_both_lanes` - the gate/checker are the only defense); §C1.4 applies |

**Status backing convention** (Phase 0 verification, executed 2026-07-17):
a `test <name>` backing a live row is either a green evidence lock that
asserts today's substituting behavior directly, or an `#[ignore]`d
red-by-design test whose failure under `--ignored` is the liveness proof
(each was re-executed at P0, except row 3, whose 2026-07-16 probe is
carried per the contract). A `canary <name>` is a green test that drives
the guard keeping a dead row dead - except rows 14 and 15, whose guarded
arms are internal-desync-only and cannot be driven from any input
surface: their canaries are reachable-surface clearance (the reachable
path behaves correctly), not deadness proofs, and §C1.4's raise-or-prove
converts both arms at Phase 1 regardless. New named executables live in
`crates/chelis-cli/tests/loud_unsupported_census_canaries.rs`; the token
baseline is frozen in
`crates/chelis-cli/tests/loud_unsupported_tripwire.rs` (which also hosts
the [#732] plan's `%.16g`/`%.1f`/`{value:.1}` format-token row per its
Phase 0 item 3 and the roadmap's Wave 0 handshake; that row's new-site
message points at `faithful_observation.md` §B2.4).

Phase 0 freezes this table into the tripwire; additions after that are
either new work (filed + censused) or regressions (red gate).

---

# Part II - process rules at every boundary

## B1. Freeze points

| contract | frozen at end of | may change after only by |
|---|---|---|
| §C2 `Unsupported` shape + message format | Phase 1 | this doc + [#687] rejected-corpus update, same PR |
| §C3 channel signatures (`Result` plumbing shape) | Phase 1 | this doc |
| §C5 census (as tripwire baseline) | Phase 0 | append-only via filed issue; removals only with the site's fix |
| §C4.2 lint rule config (enum list, crate list, allowlist) | Phase 2 | this doc + lint-rule spec (`spec/01-nomenclature.md` registration) |
| gate inventory (§C5 rows 17-18 resolution) | Phase 3 | this doc |

## B2. Invariants that hold across every boundary

1. **Controls never move.** Every green control in the audit files -
   including the loud-failure locks (HIP's rejection strings, Metal's
   abort stub, the runtime aborts) - keeps its expected text until a
   freeze-point change says otherwise. Diagnostic-wording migration to
   §C2's format is a single dedicated change, not a drip.
2. **Red-to-green only by un-ignoring** (same rule as [#729]): the
   `#[ignore]`d tests asserting rejection-or-correctness flip by removing
   the attribute, never by weakening the assertion.
3. **No behavior change without both lanes.** A site converted to raise
   must raise identically via eval and build paths where both reach it.
4. **Every conversion carries negative parity**: the test that the
   supported neighbor STILL WORKS lands with the raise (the audit's
   control discipline - [#715]'s fix must not reject `sqrt`).
5. **Discoveries fork.** New substitution sites found mid-phase are filed,
   censused (append), tripwired, and scheduled - not silently fixed
   in-passing (unreviewed conversions are how gates drifted, [#698]).
6. **Interlock edits are bidirectional.** Any change to §I1's boundary
   with [#729] updates both documents in the same change set.

## B3. How to pick up a phase

1. Read Part I, your phase, and the previous phase's frozen-at-exit list.
2. Run your oracle suite first; the red set is your work-list. The probe
   corpus (`docs/investigations/probes/`) re-derives any census row's
   live behavior from scratch; believe execution, not this table.
3. Conversions are per-site PR-reviewable units; the plumbing refactor
   (Phase 1 item 1) is one PR that changes signatures with zero behavior
   change, so review is mechanical.
4. Gate with `scripts/gate.py --local`; macOS Smoke is the workspace
   oracle.

---

# Part III - the phases

## Phase 0 - census verification + tripwire (land-first; small)

**You inherit:** the audit record (§C5 as written) and the PR [#696] test
surface.

**You deliver:**

1. **Executed re-verification of every §C5 row** using the probe corpus:
   each row's status column becomes `live (test <name>)` or
   `dead (canary <name>)`. Row 3 (`<value>` string fallback) is already
   settled: probed 2026-07-16, live, filed as [#734]
   (`issue_734_tostring_placeholder.rs`); P0 carries that status into the
   verified census. Discrepancies edit the census (B2.5 protocol).
2. **The token tripwire test** (§C4.3) with the verified census as its
   allowlist, wired into the default workspace run (it is a fast grep).
3. **The [#687] rejected-cells corpus stub**: the existing loud-failure
   locks (HIP/Metal/runtime strings) collected into one table-driven test
   so Phase 1's message migration has a single file to update.

**Frozen at your exit:** the census baseline (append-only); the tripwire
is live in CI.

**Explicitly not yours:** fixing anything; the lint rule; any plumbing.

**Oracle:** the tripwire test green on the current tree and demonstrably
red on a planted `*/ 0` fallback (the test's own negative test); every
census row's status backed by a named executable.

## Phase 1 - the failure channel + the live-site sweep

**You inherit:** the verified census, the tripwire (your regression
guard), and lowering's existing `raise_lowering_error`.

**You deliver:**

1. **The plumbing PR**: `Unsupported` (§C2) and the Result-typed emitter
   channel (§C3) threaded through `host_emit.rs`/`emit.rs`, behavior
   unchanged (existing fallbacks temporarily map to their current strings
   behind the new signatures). Zero test movement; purely mechanical.
2. **The live-site sweep**, one reviewable PR per row or tight group:
   census rows 1, 2, 3, 5, 6, 7, 8, 9 convert to §C2 diagnostics (row 3's
   `to_string` placeholder becomes a diagnostic here, [#734]; real
   tensor/list rendering arrives with [#732]'s formatter); rows 11,
   12 (panics) convert to diagnostics through the same channel; row 4's
   `<value>` prints become diagnostics at emit time (an unclassifiable
   value is a compiler bug surfaced, not a placeholder printed). Row 10
   interim-hardens per Non-goals (abort-with-dtype-id) pending [#729]
   Phase 3.
3. **Message migration**: the three exemplary diagnostics and all new
   ones conform to §C2's format; the rejected-cells corpus updated in the
   same PR (B2.1).
4. **§C1.4 for the dead rows** (13-16): default to raise; keep-with-canary
   only where a raise is impossible (document which, expected: none).

**Frozen at your exit:** §C2 shape + strings; §C3 signatures. Downstream
([#729] Phase 3, shell repos) may match `unsupported:` diagnostics without
re-checking.

**Explicitly not yours:** making any unsupported thing SUPPORTED (that is
[#729]'s or an op-owner's work; see §I1 for what your rejections do to
existing red tests); the lint rule; gate deletion.

**Oracle:** `issue_703_silent_placeholders.rs` fully green and
un-ignored; `scalar_stub_matrix.rs`'s stub-marker assertions green
(value-correctness rows go red-for-a-better-reason per §I1 and stay
ignored with updated notes); `reduce_window_nonliteral_matrix.rs` green
(both rows accept reject-with-diagnostic); the [#722] eval rows green via
row 1's raise (a loud lowering error, not zero gradients); the tripwire
census shrinks to rows 10, 17, 18.

## Phase 2 - un-writability (the lint ratchet)

**You inherit:** a tree with no live silent fallbacks (Phase 1) and the
tripwire proving it.

**You deliver:**

1. **`rust-no-wildcard-dispatch`** in chelis-lint (§C4.2): blocking, with
   the enum list (`Prim`, `RiscOp`, `ElemKind`, dtype-id constants), the
   crate list (chelis-ir, chelis-backend-c, chelis-backend-hip,
   chelis-compiler-api numeric modules, chelis-runtime; candidate
   addition per Jeff's reflexive finding on [#738]/[#739]:
   chelis-conformance, whose `check_row` `other =>` arm returns a silent
   `Verdict::Manual` for un-dispatched MANIFEST rows - the class inside
   the soundness auditor itself; coordinate with the conform
   workstream), the
   justification-string allowlist mechanism, and registration in the lint
   rule spec per repo convention. Plus its own positive/negative rule
   tests (the lint crate's standard).
2. **Wildcard removal** across the listed crates so the rule lands green
   with a minimal allowlist (target: single digits, each with a written
   justification).
3. **The `EmittedExpr` newtype** (§C3) if not already forced by Phase 1's
   plumbing; the `pub(crate) raw` constructor's use sites enumerated in
   the allowlist.
4. `EffectKind` enum (§C4.4), closing census row 9's future-kinds hole
   permanently.

**Frozen at your exit:** the lint config (B1). From here, a new silent
fallback requires editing an allowlist file with a justification -
reviewable by construction.

**Explicitly not yours:** [#709]'s DeepTag enum (same pattern, checker
crates, tracked there - coordinate the lint's enum list so it can adopt
DeepTag later without a config freeze exception).

**Oracle:** `chelis lint --check .` green with the new rule enabled and
red on a planted wildcard arm in a listed crate (the rule's negative
test); the tripwire baseline unchanged.

## Phase 3 - gates become UX, not safety

**You inherit:** emitters that cannot silently substitute (so gates are
no longer load-carrying for correctness) and the census rows 17-18.

**You deliver:**

1. **Gate inventory resolution**: the duplicated `reject_*` gates ([#698]'s
   drifted pair, [#705]'s missing CLI twin) are deduplicated to single
   definitions consumed by both the CLI and compiler-api paths;
   `HOST_ONLY_BUILTINS` and kin either derive from the [#729] Phase 4
   capability table (if landed) or are replaced by "let the emitter's
   channel speak, gate only to move the diagnostic earlier and add span
   context".
2. **The gate contract**, recorded in this doc: a gate may only ever make
   a diagnostic EARLIER or MORE SPECIFIC; it may never be the sole
   defense, and a gate/emitter disagreement is a bug in the gate. The
   enforcement ladder (2026-07 review): compile-error > lint > tripwire
   > gate > prose - a gate that is load-bearing for correctness is on
   the wrong rung.
3. Deletion of gates that now only duplicate emitter rejections, with the
   cross-lane rejected-cells corpus proving the diagnostic surface
   unchanged or improved (earlier stage, same `unsupported:` content).

**Frozen at your exit:** the gate inventory and contract.

**Explicitly not yours:** the capability table itself ([#729] Phase 4).

**Oracle:** the rejected-cells corpus byte-stable (or improved-with-
updated-expectations in the same PR); [#697]/[#698]'s tests green and
un-ignored; zero gates existing in two copies (a new tripwire assertion,
mirroring the conformance MANIFEST pattern).

---

# Part IV - bookkeeping

## I1. The interlock with [#729] (dtype semantics)

The two plans touch the same sites from different directions. The
boundary, pinned:

- **This plan makes the unsupported loud; [#729] makes the supported
  correct.** For cells that are both (f16 scalars [#714]; int-tensor
  abs/floor [#699]/[#722]; int64 reduces [#692]): after this plan's Phase 1
  they are *cleanly rejected*; after [#729]'s Phases 2-3 they are
  *computed*. The intermediate rejected state is an improvement and is
  expected to persist for a while.
- **Test semantics across the interlock:** audit tests asserting final
  correct VALUES (e.g. `f16_scalar_fraction_survives_compilation`) stay
  `#[ignore]`d through this plan - their ignore notes gain one line
  ("now rejected loudly per [#703] plan; value support tracked by [#729]").
  Tests asserting NO-SILENT-STUB or reject-or-correct (the stub-marker
  and `pool_or_reject` forms) go green here and are un-ignored here.
- **Ordering freedom:** the plans are independently landable in any
  interleaving EXCEPT [#729] Phase 3's `parse_host_type` work, which
  builds on this plan's Phase 1 conversion of census rows 6-7 (support
  replaces a raise, never a silent default). If [#729] Phase 3 arrives
  first, it absorbs rows 6-7's conversion and says so in both docs.
- The capability table ([#729] Phase 4) eventually *generates* what this
  plan's Phase 3 gate contract governs by hand; at that point rejected
  cells' diagnostics come from the table's `Rejected(reason)` and this
  plan's §C2 format governs the rendering.

## Issue map

| phase | goes green / becomes unwritable |
|---|---|
| 0 | census verified; tripwire live; regressions detectable |
| 1 | [#699] (+[#722]'s eval half via the raise), [#682], [#704], [#705], [#715] (stub half), [#689], [#692], [#725], [#734], the [#709]-adjacent effect catch-all; [#714]/[#718] downgraded from silent-wrong to cleanly-rejected |
| 2 | the entire FUTURE supply of the class (lint + newtype + enums) |
| 3 | [#697], [#698], [#705]'s gate half; gate rot as a class |

## Open questions and where they get decided

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | `Unsupported`: one shared type vs per-crate mirrors | DECIDED 2026-07-17: ONE shared type, defined in `chelis-types` beside the dtype_semantics module and the future capability table (whose `Rejected` cells both the checker and the backends render into it). No cycles: chelis-types sits at the workspace bottom and every producer (chelis-ir, all backends, compiler-api) already depends on it (verified against the Cargo graph). Lowering's existing error type absorbs it via `From<Unsupported>`, not replacement. Per-crate mirrors would be the scatter pattern in miniature | §C2/§C3 of this doc |
| 2 | whether `check` should pre-report target-independent unsupported constructs | **RESOLVED** by `capability_table.md`: yes for semantic-table (A) rejections - they are target-independent type facts; target-level (B) rejections surface at build | `capability_table.md` §Derivations |
| 3 | lint allowlist mechanics (inline justification comment vs allowlist file) - follow whatever §8.6's rule already does for exceptions | Phase 2 | lint rule spec |
| 4 | which gates survive Phase 3 as early-UX vs die | Phase 3 | §C5 rows 17-18 + gate contract |

## The one-sentence summary for a reviewer

Give every stage a way to say no (Result-typed emission), convert the ten
live yes-anyway sites to §C2 diagnostics, then take the pen away: no
wildcard arms over closed enums (lint), no raw strings on the unsupported
path (newtype), no unwrap_or in lowering (lint + tripwire), and gates
demoted to UX so the type system, not a one-entry allowlist, is the thing
standing between an unsupported case and a plausible wrong number.

[#387]: https://github.com/Chelis-Lang/chelis/issues/387
[#680]: https://github.com/Chelis-Lang/chelis/issues/680
[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#687]: https://github.com/Chelis-Lang/chelis/issues/687
[#689]: https://github.com/Chelis-Lang/chelis/issues/689
[#692]: https://github.com/Chelis-Lang/chelis/issues/692
[#696]: https://github.com/Chelis-Lang/chelis/pull/696
[#697]: https://github.com/Chelis-Lang/chelis/issues/697
[#698]: https://github.com/Chelis-Lang/chelis/issues/698
[#699]: https://github.com/Chelis-Lang/chelis/issues/699
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#704]: https://github.com/Chelis-Lang/chelis/issues/704
[#705]: https://github.com/Chelis-Lang/chelis/issues/705
[#709]: https://github.com/Chelis-Lang/chelis/issues/709
[#710]: https://github.com/Chelis-Lang/chelis/issues/710
[#711]: https://github.com/Chelis-Lang/chelis/issues/711
[#712]: https://github.com/Chelis-Lang/chelis/issues/712
[#714]: https://github.com/Chelis-Lang/chelis/issues/714
[#715]: https://github.com/Chelis-Lang/chelis/issues/715
[#716]: https://github.com/Chelis-Lang/chelis/issues/716
[#718]: https://github.com/Chelis-Lang/chelis/issues/718
[#722]: https://github.com/Chelis-Lang/chelis/issues/722
[#725]: https://github.com/Chelis-Lang/chelis/issues/725
[#727]: https://github.com/Chelis-Lang/chelis/issues/727
[#728]: https://github.com/Chelis-Lang/chelis/issues/728
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
[#733]: https://github.com/Chelis-Lang/chelis/issues/733
[#734]: https://github.com/Chelis-Lang/chelis/issues/734
[#738]: https://github.com/Chelis-Lang/chelis/issues/738
[#739]: https://github.com/Chelis-Lang/chelis/issues/739
[#744]: https://github.com/Chelis-Lang/chelis/issues/744
[#745]: https://github.com/Chelis-Lang/chelis/issues/745
