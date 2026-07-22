# Loud Unsupported: the failure-channel contract

**Status:** Phases 0-1 implemented (Phase 0: PR [#746]; Phase 1: PR
[#791] - the failure channel, the live-site sweep, and the section C2/C3
freeze). Phase 2 is IN PROGRESS. The dependency-bottom `EffectKind` and
`RuntimeDType` slice is now implemented: Result-only boundary decoders,
exhaustive semantic consumers, Rust-generated C dtype agreement, immediate
runtime ABI decoding, and executed added-variant mutation oracles. PR [#799]
also supplied useful defense evidence (the lexical wildcard scanner, count
baselines, and a first `EmittedExpr` newtype), but its red team showed that
those mechanisms alone do not make substitution structurally unrealizable.
Phase 2 is not complete: the HostType failure-channel split, capability-cell
proofs, and private structured C-expression AST remain. The completion
boundary is §C4/§C6, not a green lexical scan. Phase 3 (gate demotion) remains
pending. Tracking issue: [#730].
**Owning specs:** `spec/05-risc-primitives.md` (op support statements;
its §7 carries this plan's ratified contract as current blockquote authorities
[05-UNS-1..4], independently of their later chelis#733 migration through the
pinned Buoy shell-side integration),
`spec/04-type-system.md` §1.1.1 (deferred dtypes precedent), the repo
Contract Invariants ("if a command reports perfect success, its error list
must be empty"), and the audit record in
`docs/investigations/numeric_audit_next_sweeps.md` /
`docs/investigations/numeric_audit_structural_prevention.md` (items 3, 8).
**Class:** [#703] (unsupported cases silently substitute a value instead of
failing). Phase 1 removed the censused live instances; Phase 2's recurrence
proof is still in progress. Sibling plans: `spec/design/dtype_semantics.md`
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
**structurally unrealizable** through dependency-bottom closed vocabularies,
fallible boundary decoding, and exhaustive typed consumers, with lexical
lint/tripwires retained only as defense-in-depth evidence, and (4) demotes the
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
only signature, and after Phase 2 raw input is decoded once into a closed
type and every semantic consumer must exhaust that type. Adding a variant
then creates a rustc work-list; no source scanner is part of that proof.

## Non-goals

- **Not [#709].** The checker's `Type::Error`-without-diagnostic hole is the
  same disease in a different organ with a different fix - now its own
  plan (`spec/design/checker_totality.md`, [#731]). One shared piece
  lands here: the `EffectKind` enum (§C4.1-2), because its catch-all is a
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

# Part I - the normative contracts (§C1-§C6)

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
    /// The context of the encounter (the op family, target lane, or
    /// call position) - the `on <context>` clause of the rendering.
    /// (Added at Phase 1 ratification: the frozen message format always
    /// carried a context clause; the struct now carries it explicitly.)
    pub context: String,
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

Implemented as `chelis_types::unsupported::Unsupported` (per open
question 1: one shared type at the workspace bottom; the span field is
boxed in the implementation so the Err variant stays small on the
Result-typed emission paths - a representation detail, not a shape
change).

**Message format (frozen):**
`unsupported: <what> on <context> (<stage>); <hint>` - branded with the
literal prefix `unsupported:` so tests and shells can match it. The three
existing exemplary messages are the calibration set and must remain
conformant when migrated:

- HIP: `` `chelis build --target hip` admits `f16` only on tensor
  load/store nodes ... See spec/04-type-system.md §5.7.1`` (names the
  construct, the boundary, and the spec);
- Metal: `` `chelis build --target metal` rejects f64 ... `` ;
- runtime: `unsupported: destination dtype ... on to_tensor host-lane
  literal storage (runtime); ...` (migrated to the branded shape at
  Phase 1; the pre-migration spelling was
  `to_tensor: unsupported destination dtype ...`).

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

| stage | today | target state |
|---|---|---|
| lowering (`chelis-ir/src/lower.rs`) | `raise_lowering_error` exists and works (used by `lower_unsupported`) | unchanged mechanism; every censused lowering site calls it |
| host-type resolution (`chelis-ir/src/host.rs`) | infallible `-> HostType`, with one `Unknown` state shared by malformed/missing metadata, polymorphism, inference, bottom, and unimplemented logical dtypes | syntax decoding returns `Result<HostTypeTerm, HostTypeDecodeError>`; named type/precision/rank variables, inference variables, and `Never` are distinct terms; resolution returns `Result<ConcreteHostType, HostTypeResolutionError>` and never chooses a default |
| host ABI selection (`chelis-ir` -> C host emitter) | infallible `HostType -> &'static str`; `Unknown -> void*`, plus zero defaults in boxing/unboxing | `ConcreteHostType -> Result<HostAbiType, Unsupported>`; the emitter accepts only `HostAbiType`; a known logical type with an unimplemented target representation is a Table-B `Unimplemented` diagnostic, never a type-erasure or emitted value |
| C host emitter (`chelis-backend-c/src/host_emit.rs`) | **no channel**: expression builders return `String`, statement emitters return `()` | expression builders return `Result<EmittedExpr, Unsupported>`; statement emitters `Result<(), Unsupported>`; the compile entry surfaces the first error as a §C2 diagnostic. This is THE plumbing refactor - mechanical (`?` all the way up), moderate in size, and the enabling move for everything else |
| DAG C emitter (`chelis-backend-c/src/emit.rs`) | panics ([#692]) | same Result channel; the reduce-family panic arms become diagnostics |
| HIP emitter | gate rejects much; `elem_kind` substitutes F32 ([#689]) | `elem_kind` returns `Result`; its `_` arm deleted (§C4.2) |
| runtime (`chelis-runtime`) | `runtime_fail!` aborts exist and are the right shape | unchanged; used only for dynamic-only cases per §C1.2 |

**The speculative-sub-lowering laundering rule (Phase 1 addition; the
[#776]/[#782] finding):** the host-emit backend speculatively sub-lowers
def bodies through the tensor-DAG path and, on a NON-FATAL lowering
error, recovers by falling back to host emission. Before Phase 1 that
recovery path could terminate in the silent
`/* unsupported builtin */ 0` stub - laundering a raised error back
into a compiled zero, which is why [#782] had to mark its lowering
errors fatal. The Result channel closes this structurally: the
recovery terminal is now `Err(Unsupported)`, so a swallowed non-fatal
error either host-emits CORRECTLY or fails the build loudly - it can
no longer end silent. Rule for new lowering-side rejections: use the
FATAL raise when the host fallback would mis-emit the same construct
(a garbage host call over a tensor pointer); a non-fatal raise is
acceptable only where the host fallback legitimately owns the form,
because its own unsupported terminal is now loud.

**The `EmittedExpr` rule (Phase 2 groundwork, not yet the boundary):** PR
[#799] introduced `chelis_backend_c::emitted_expr::EmittedExpr`, a newtype
over `String` whose `raw` constructor is crate-private. That usefully closes
out-of-crate construction and the unmatched-builtin arm is correctly an
`Err(Unsupported)`. It does not make a plausible zero stub unwritable inside
`chelis-backend-c`: `EmittedExpr::raw("0".to_string())` remains legal and the
historical token tripwire cannot recognize equivalent spellings. Phase 2
therefore treats this newtype as migration scaffolding. Its completion form
is a private structured C-expression AST whose public builders cannot encode
an unsupported placeholder; string rendering occurs only after a typed
expression has been constructed.

## C4. The ratchets (making reintroduction structurally unrealizable)

The primary mechanisms are typed and exhaustive. A lexical rule may detect a
known spelling, but it SHALL NOT be cited as proof that a substitution cannot
be represented.

1. **One dependency-bottom vocabulary owner.** `chelis-vocab` has no Chelis
   dependencies and owns `EffectKind` and `RuntimeDType`. Each vocabulary is
   declared once with its canonical external spelling or integer ID. It has no
   `Unknown` variant, implements no `Default`, and exposes only `Result`
   decoders. Effect metadata decoding distinguishes `Missing`, `Malformed`,
   and `Unknown { symbol }`; runtime dtype decoding preserves the invalid raw
   ID. `chelis-types` is not the owner: it depends on `chelis-deep` and
   `chelis-pred`, while the runtime must consume the same dtype vocabulary
   without depending on the checker. This bottom placement also leaves a
   cycle-free migration path for `Prim` and `BuiltinId`.
2. **Decode once, then exhaust.** Raw strings and raw dtype integers exist
   only at serialization/FFI boundaries. `chelis-deep` adapts metadata shape
   into the bottom crate's effect decoder; every checker, effects, lowering,
   evaluator, and decompiler decision receives `EffectKind`. The C runtime
   decodes `chelis_tensor.dtype` and dtype arguments immediately; sizing and
   reading helpers accept `RuntimeDType`, never `c_int`. Matches over these
   enums have no wildcard arm. Adding a variant is therefore a compile-error
   work-list at every semantic consumer.
3. **Generated Rust/C dtype agreement.** The `RuntimeDType` declaration is
   the sole source for Rust IDs, canonical names, C macro names, and byte
   widths. It generates a checked-in C header fragment consumed by the host,
   HIP, and Metal runtime headers. A byte-for-byte regeneration test and a
   Rust round-trip table test lock agreement. Every generated C size switch
   has an aborting `default` that prints the raw ID; no default may select f32.
4. **Structured emission.** Open-set builtin dispatch returns
   `Result<EmittedExpr, Unsupported>`. The final `EmittedExpr` representation
   is a private C AST with typed builders; a crate-private raw-string escape
   is migration-only and cannot remain on the Phase 2 completion surface.
5. **Lexical defenses are temporary evidence.** The current
   `rust-no-wildcard-dispatch` scanner, count allowlist, and token tripwire stay
   blocking while the typed migration is incomplete. They are neither a site
   identity nor an exhaustiveness proof: aliases, bindings, indirection,
   equivalent numeric-default spellings, count relocation, and in-crate raw
   emission can evade them. Phase 2 may remove or narrow them only after the
   corresponding typed mutation oracle is green.
6. **Host types are a staged typed pipeline.** `HostTypeTerm` preserves exact
   logical scalar precision and gives named type, precision, rank, inference,
   and bottom states distinct variants. Missing and malformed metadata are
   `HostTypeDecodeError`, not terms. Only `HostTypeTerm::into_concrete` may
   produce `ConcreteHostType`, and it returns `HostTypeResolutionError` while
   any variable or `Never` remains. Target ABI selection consumes only
   `ConcreteHostType` and returns `Result<HostAbiType, Unsupported>`; backend
   emitters consume only `HostAbiType`. Thus `f16`, `bf16`, `int8`, and `int16`
   remain known logical types even while the C-host capability cell rejects
   their representation. There is no `Unknown` variant, no `Default`, and no
   conversion from a decode/resolution failure to a concrete or ABI type.

## C5. The census (normative appendix; Phase 0 re-verifies by execution)

The remediation work-list, from the audit record as of 2026-07-16.
**Live** = confirmed by execution.

**Phase 1 dispositions (PR [#791], 2026-07-20).** Every live row below
converted to a section C2 diagnostic; every dead row got its section
C1.4 raise except five documented structural keeps in `lower.rs` (the
defsig/deftype/typealias inert declaration node, the
unknown-tag-with-children sequence seed, `zero_tensor_node`'s
deliberate ADT zero adjoint, the empty-`drop` sequencing zero, and the
`fail`-in-if mask placeholder - each annotated at the site). Row status
notes below are left as the P0 record; the per-row conversion evidence
is the un-ignored acceptance tests named in PR [#791] plus
`loud_unsupported_phase1.rs`. Appended rows:

| # | site | substitutes | issue | status |
|---|---|---|---|---|
| 20 | `host.rs` handle-effect arm (host-lane sibling of row 9) | drops handler, lowers body for non-`random` kinds | [#709]-adjacent (P1 discovery, B2.5) | CONVERTED with row 9 in the same change set (test `unknown_effect_kind_is_rejected`) |
| 21 | `lower.rs` `expand` positional-axis `unwrap_or(0)` | axis 0 | flagged by [#782] | CONVERTED: fatal raise (section C1.4; a computed axis previously expanded axis 0 silently) |
| 22 | `lower.rs` `tuple-get` index `unwrap_or(0)` | field 0 | flagged by [#782] | CONVERTED: raise (section C1.4; compile-time index by construction) |
| 23 | `lower.rs` conv2d present-but-non-literal stride/padding `unwrap_or(1)`/`unwrap_or(0)` | stride 1 / padding 0 | P1 discovery (the [#776] shape); TO FILE | flagged and left per B2.5 (absent-arg defaults are the documented optional-arg semantics; the present-but-non-literal case needs its own probe + issue); baselined in the tripwire's numeric-unwrap class |

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

## C6. Phase 2 dependency and migration inventory

This inventory is part of the Phase 2 contract. A consumer may be removed
only when the underlying behavior is removed; a newly discovered consumer is
appended before it is migrated (B2.5).

### C6.1 Dependency placement and effect consumers

The static Cargo audit at PR [#799] head `42d61a2` found that
`chelis-types` depends on `chelis-deep` and `chelis-pred`, while
`chelis-runtime` has no Chelis dependency. Making either one the shared owner
would put a front-end or runtime implementation above the other. The new
`chelis-vocab` crate instead has no dependencies. These are the permitted new
edges:

| crate | edge/purpose | cycle audit |
|---|---|---|
| `chelis-deep` | `chelis-vocab`; adapt Deep metadata into `EffectKindInput` | vocab has no reverse edge |
| `chelis-surf` | `chelis-vocab`; canonical effect serialization/decompilation | already depends on Deep, never on types |
| `chelis-types` | `chelis-vocab`; checker dispatch; temporarily re-export `EffectKind` for source compatibility | vocab does not depend on Deep/Pred/types |
| `chelis-effects` | `chelis-vocab`; direct semantic dispatch | its existing types/Deep edges remain above vocab |
| `chelis-ir` | `chelis-vocab`; IR and host lowering dispatch | its existing types/Deep edges remain above vocab |
| `chelis-compiler-api` | `chelis-vocab`; evaluator dispatch and wire adapters | already sits above effects/IR/types |
| `chelis-runtime` | `chelis-vocab`; immediate FFI dtype decoding | vocab adds no libc/runtime edge |
| C/HIP/Metal backends | `chelis-vocab`; `Prim -> RuntimeDType -> C macro` mapping | all already sit above IR/types |
| `chelis-python` | `chelis-vocab`; remove the local `CHELIS_F32 = 0` copy | already sits above compiler-api/backend C |

Future ownership remains cycle-free: move `Prim` and the future `BuiltinId`
declaration into `chelis-vocab`, then re-export them from `chelis-types` while
callers migrate. No Deep/runtime/backend edge points back upward from vocab.

The effect migration work-list is exhaustive for the current tree. Every row
is implemented in the typed slice; the middle column records the replaced
behavior at `42d61a2`:

| boundary/consumer | replaced raw behavior | delivered typed behavior |
|---|---|---|
| `chelis-vocab::{EffectKind, EffectKindInput, EffectKindDecodeError}` | `chelis-types::EffectKind::from_symbol -> Option` | one declaration; `decode -> Result`; distinct missing/malformed/unknown errors; canonical `symbol()` inverse |
| `chelis-deep` effect-metadata adapter (new) | every caller re-walks `{effect: ...}` | one adapter maps absent key, non-symbol value, and unknown symbol without collapsing them |
| `chelis-types::infer::infer_handle_effect` | partial enum use over `Option` | consume the Deep adapter result and exhaust `EffectKind`; map each decode error to the checker diagnostic |
| `chelis-effects::infer_handle_effects` | `Some("random")` / `Some("resource")` / silent `_` | exhaustive typed effect removal; decode failures enter `EffectError` |
| `chelis-effects::validate_handler_expr` | raw-string guarded arms and `_ => {}` | exhaustive per-kind handler validation; decode failures are errors |
| `chelis-effects::validate_build_target_expr` | raw equality with `"resource"` | exhaustive typed target policy |
| `chelis-ir::lower::lower_handle_effect` | enum reached through `Option`, with missing/unknown collapsed | consume `Result`; exhaustive `Random`/`Resource`; preserve distinct diagnostic payloads |
| `chelis-ir::host::lower_host_expr` handle-effect arm | empty-string default before partial enum decode | consume `Result` at the boundary; no empty sentinel and no raw comparison |
| `chelis-compiler-api::runtime::eval::HostEvaluator::eval_expr` | `if effect == "random"`, else evaluate body | exhaustive typed `Random` and `Resource`; every decode error is `Err`; no unknown/non-random passthrough |
| `chelis-surf::desugar` `WithSeed`/`WithDevice` | hand-written `"random"`/`"resource"` | emit `EffectKind::symbol()` |
| both `chelis-surf::decompile_handle_effect` implementations | raw-string match with generic fallback | decode once and exhaust known kinds; preserve malformed/unknown Deep only through an explicit decode-error observation path, never semantic execution |

The checked-in source inventory keeps every row inside the mutation oracle.
The executed oracle added a fully decoded temporary variant and produced
Rust `E0004` at an exhaustive consumer; the variant was then removed. This
compile failure, not the inventory's string scan, is the structural proof.

The added-kind mutation oracle inserts one temporary variant in the single
vocab declaration and builds `chelis-deep`, `chelis-surf`, `chelis-types`,
`chelis-effects`, `chelis-ir`, and `chelis-compiler-api`. Every semantic row
above must produce a non-exhaustive-match error until it makes an explicit
decision. The source-inventory test only proves there is no untyped consumer
outside that rustc oracle; it is not itself the exhaustiveness mechanism.

### C6.2 Runtime dtype boundaries and consumers

`RuntimeDType` owns the stable ABI IDs `F32=0`, `F64=1`, `I32=2`, `Bool=3`,
`I64=4`, `Bf16=5`, `F16=6`, `I8=7`, and `I16=8`, together with canonical
language spellings, C macro spellings, and byte widths. The numeric IDs do
not change in this migration. `chelis_tensor.dtype` and the C ABI arguments
remain `int`; that is the wire representation, not the internal type.

| boundary/consumer | delivered migration |
|---|---|
| `chelis-runtime/include/chelis_runtime.h` and HIP/Metal runtime headers | include a generated dtype fragment from the vocab declaration; remove handwritten ID/size copies |
| `chelis-runtime::{CHELIS_*}` | compatibility constants derive from `RuntimeDType::id()`, never literal integers |
| `TensorElement::DTYPE` / `DtypeMismatch` | carry `RuntimeDType`; decode the tensor field before comparing or accessing |
| `chelis_alloc`, `chelis_alloc_view`, `chelis_dtype_size`, `chelis_tensor_from_value_list_typed` | decode the inbound `c_int` immediately; invalid IDs abort with the raw ID before allocation, sizing, or element access |
| `tensor_elem_size` | signature is `fn(RuntimeDType) -> usize`; exhaustive, no fallback |
| `read_index_slot`, `chelis_tensor_to_f64`, list-from-tensor, comparison/where/cumsum/sort/trace/clamp/einsum, and tensor formatting | decode once, pass `RuntimeDType` into typed read helpers, and match exhaustively |
| clone/concat/split/gather/scatter/diagonal/contiguous byte-copy paths | decode before byte-width calculation; pass `RuntimeDType` to sizing; raw integers may be copied back only into the ABI field via `id()` |
| C/HIP/Metal `dtype_macro` and sparse/dtype-arm helpers | return `RuntimeDType` first and obtain the C spelling from the vocab declaration; no repeated `Prim -> "CHELIS_*"` tables |
| `chelis-python` tensor construction | use `RuntimeDType::F32.id()`; remove the local numeric constant |

The negative suite covers `-1`, the first unused ID (`9` for this frozen
table), `i32::MIN`, and `i32::MAX`. Each must fail at the decoder. Runtime
subprocess tests then pass those IDs to each public FFI dtype boundary and
assert nonzero exit before any allocation-size result or buffer read can be
observed. Positive coverage round-trips every ID and compares the generated C
fragment byte-for-byte with its checked-in artifact.

The runtime slice is implemented. Its subprocess suite drives every raw dtype
argument boundary plus a tensor-field read with ID `9` and requires nonzero
exit carrying the raw ID. The executed added-dtype oracle likewise produced
Rust `E0004` at an exhaustive consumer before the temporary variant was
removed. These results freeze the vocabulary slice, but do not complete Phase
2's unrelated HostType and structured-emission work.

### C6.3 Host-type state and ABI migration

The HostType inventory at PR [#799] base `42d61a2` found 167 textual
`HostType::Unknown` occurrences in `chelis-ir/src/host.rs` and 22 in
`chelis-backend-c/src/host_emit.rs`, including comments. Those numbers are
**migration telemetry only**. They are not an allowlist, a safety gate, or the
completion oracle: an exact-count scanner repeats the lexical-design failure
described in C4.5 and can be evaded by aliases or relocation. The authoritative
endpoint is zero direct legacy constructions plus compile-time boundaries that
make an unresolved term impossible to pass to codegen.

Every production occurrence belongs to one of the following semantic classes.
The ranges are navigation aids at that base, not stable identities:

| semantic class | producer/consumer inventory at `42d61a2` | required migration |
|---|---|---|
| missing or malformed metadata/type syntax | `expr_type`/`expr_fn_type` and `parse_host_type[_with_subst]` (`host.rs:7614-7776`): non-list input, absent metadata, unknown primitive, malformed `t-ref`/`t-adt`/`t-fn`, unknown tag | boundary returns `HostTypeDecodeError::{MissingTypeMetadata, MalformedTypeSyntax, UnknownPrimitive}`; no type term is manufactured |
| legitimate type polymorphism | precision/rank-poly filters and standalone-emission skips (`host.rs:1068-1398`, `2035-2064`, `7691-7725`), plus signature/inlining substitutions (`8427-8506`, `8615-8617`) | preserve names in `TypeVariable`, `HostPrecisionTerm::Variable`, and ordered `HostShapeSlot::RankVariable` entries (including Tier-3 prefix/suffix spreads); specialize or return a resolution error before codegen |
| underconstrained inference | function/parameter defaults (`2161-2274`), expression inference/refinement (`3428-4618`, `4796-6007`), inlining/call metadata (`6469-7253`) | allocate stable `HostInferenceVar` identities; empty collections retain an element inference variable; joins/refinement unify terms and unresolved holes return `HostTypeResolutionError` |
| builtin result inference failure | `infer_builtin_host_type[_from_arg_tys]` (`7924-8348`): invalid argument shapes, tuple indices, collection element extraction, and absent callback/result facts | distinguish an invalid builtin application (typed error) from a valid underconstrained result (inference variable); no `Some(Unknown)` success value |
| bottom/divergence | `fail` result (`8223`) and branch joins that currently use `Unknown` as the missing side | represent `fail` as `HostTypeTerm::Never`; a join may eliminate bottom in favor of the other branch, but `Never` cannot resolve to a value or ABI representation |
| known logical type, target representation absent | `t-prim` fallback (`7684`), scalar arithmetic propagation (`7981-7988`), tensor/scalar/list conversions (`8271-8348`), with C baking guard (`host_emit.rs:2388-2422`) | retain exact `Prim` in `ConcreteHostType::Scalar`; C-host selection returns `Unsupported` for an unimplemented Table-B cell such as f16/bf16/int8/int16 |
| emitter expectation/refinement sentinel | C list/tuple/call/callback paths (`host_emit.rs:1471-1812`, `3902-4466`, `5192-5212`) choose expected vs inferred types and synthesize unknown container fields | resolve before entering the emitter; collection/callback helpers receive concrete element/parameter ABI types, not a fallback marker |
| emitted representation/default | `c_type` maps `Unknown` to `void*` (`4995`); boxing and unboxing catch-alls return boxed/bare zero (`4760-4810` vicinity) | delete the conversions; `c_type`, box, and unbox consume `HostAbiType` exhaustively and return the existing `Unsupported` channel where an operation-specific capability is absent |

The non-production carriers are also inventoried: three span-survival fixtures
construct placeholder host expressions
(`span_threading_through_host_lowering.rs:629,646,781`); the Phase-1 and
narrow-dtype CLI tests describe the old failure chain
(`loud_unsupported_phase1.rs:284`, `narrow_dtype_matrix.rs:12`); and the
temporary wildcard-dispatch lint's docs/self-tests contain six exemption
examples (`rust_no_wildcard_dispatch.rs:20,89,575,804,809,903`). Fixtures move
to the explicit term appropriate to their role, historical comments are
rewritten as history, and the lint exemption/self-tests are deleted. None is a
semantic producer, and none is allowed to keep the legacy enum alive.

The staged source contract is:

1. `decode_host_type(...) -> Result<HostTypeTerm, HostTypeDecodeError>` owns raw
   Deep syntax and metadata.
2. Inference and specialization operate on `HostTypeTerm`. A successful
   `into_concrete` is the only path to `ConcreteHostType`.
3. Target selection consumes `ConcreteHostType` and the capability Table-B
   cell, returning `Result<HostAbiType, Unsupported>`.
4. Host codegen accepts `HostAbiType` only. Its declaration, boxing, unboxing,
   call, and callback matches are exhaustive and cannot observe a term or
   resolution error.

During migration, any unavoidable compatibility bridge is private and named
`LegacyUnresolvedHostType`; it may support legacy inference/refinement only. It
has no `From`/`TryFrom` path to `ConcreteHostType` or `HostAbiType`, cannot be
passed to emission, and is deleted at this subsection's exit. Direct
`HostType::Unknown` construction outside that one bridge is not an accepted
transitional architecture. A source endpoint test requires zero direct legacy
constructions; compile-time function signatures, not that source scan, prove
that only resolved types reach codegen.

Positive/negative parity covers: every concrete primitive; each named variable
kind; missing versus malformed syntax; empty-list inference; `Never`
value-boundary rejection; supported ABI representations; and known logical
f16/bf16/int8/int16 values rejected by an unimplemented C-host ABI cell. The
negative ABI tests assert the structured `UnsupportedKind::Dtype` and
`Stage::Codegen("c")`, not diagnostic prose alone.

---

# Part II - process rules at every boundary

## B1. Freeze points

| contract | frozen at end of | may change after only by |
|---|---|---|
| §C2 `Unsupported` shape + message format | Phase 1 | this doc + [#687] rejected-corpus update, same PR |
| §C3 channel signatures (`Result` plumbing shape) | Phase 1 | this doc |
| §C5 census (as tripwire baseline) | Phase 0 | append-only via filed issue; removals only with the site's fix |
| §C4 vocabulary declarations and typed consumer inventory | Phase 2 | this doc + owning active spec, with an added-variant mutation oracle in the same change |
| §C4.5 lexical defense config | temporary during Phase 2 | this doc + lint-rule spec (`spec/01-nomenclature.md` registration); never a completion oracle |
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
2. **The token tripwire test** (§C4.5) with the verified census as its
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

**Delivered** (PR [#791], 2026-07-20), with three recorded
deviations: (1) `issue_703_silent_placeholders.rs`'s [#712]
lane-agreement rows stay `#[ignore]`d - they assert checker/eval
agreement on scalar activations, which no emission conversion can move
(that contract is [#712]/[#731] territory; the oracle's "fully
un-ignored" overshot). (2) The [#722] eval rows stay `#[ignore]`d as
value tests; the loud-not-zero contract they were listed for is locked
by the new green `grad_through_int_abs_fails_loudly_not_zero`.
(3) Deliverable 3's message migration landed in full for the runtime
`to_tensor` exemplar (re-rendered to the frozen 4-clause shape); the
HIP admit-gate and Metal f64-gate exemplars carry the `unsupported:`
brand with their original message bodies - their full-shape
conformance rides Phase 3's gate work, per B2.1's
single-dedicated-change rule for diagnostic-wording migration.

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

## Phase 2 - un-writability (typed closed vocabularies)

**Status: IN PROGRESS.** The `EffectKind`/`RuntimeDType` items 1-3 below are
implemented and their mutation/negative oracles have been executed. PR
[#799]'s lexical scanner, count allowlist, newtype, and widened tripwire remain
useful defense evidence. They do not satisfy the remaining exit boundary:
the HostType failure-channel split and capability proofs remain, and the
backend crate can still construct arbitrary raw `EmittedExpr` strings instead
of a private structured C-expression AST.

**You inherit:** a tree with no live silent fallbacks (Phase 1) and the
tripwire proving it.

**You deliver:**

1. The dependency-free `chelis-vocab` crate and its single declarations for
   `EffectKind` and `RuntimeDType`, with Result-only decoders and the positive
   and negative tests in §C4.1.
2. The end-to-end `EffectKind` migration in §C6.1. No semantic consumer may
   compare the metadata string. The added-variant mutation must fail every
   named consumer until it explicitly handles the new kind.
3. The runtime dtype migration in §C6.2: generated Rust/C agreement,
   immediate FFI decoding, typed sizing/reading helpers, and invalid-ID tests
   for `-1`, the first unused ID, and both `i32` extrema.
4. The HostType state/ABI split in §C4.6 and §C6.3: Result-only syntax
   decoding, named polymorphic and inference states, `Never`, resolution to
   `ConcreteHostType`, and fallible target conversion to `HostAbiType`. Host
   codegen accepts only the resolved ABI vocabulary; the legacy `Unknown`
   sentinel and every emitted/default-value conversion from it are deleted.
5. The private structured C-expression AST described in §C4.4, replacing
   `EmittedExpr::raw` as a general construction path.
6. Keep the PR [#799] lint and tripwire green as defense evidence during the
   migration. Their allowlists do not freeze as the safety authority.

**Frozen at your exit:** the two vocabulary declarations, their external
spellings/IDs, the typed consumer inventories, and the generated-header
contract. From here, a new kind or dtype forces a compile-error work-list in
every semantic consumer.

**Explicitly not yours:** [#709]'s DeepTag enum (same pattern, checker
crates, tracked there - coordinate the lint's enum list so it can adopt
DeepTag later without a config freeze exception).

**Oracle:** the Phase 2 closed-vocabulary suite is green; adding one temporary
`EffectKind` variant fails compilation in every §C6.1 semantic consumer;
invalid runtime dtype IDs fail before any §C6.2 sizing or access helper;
regenerating the C dtype header is byte-identical; every §C6.3 term-state parity
test is green; and the C emitter's codegen boundary is typed in `HostAbiType`
so a compile-fail test cannot pass `HostTypeTerm`, `Never`, a decode/resolution
error, or the deleted legacy sentinel. The lint, tripwire, and zero-occurrence
endpoint scan are supporting evidence, not the oracle.

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
| 2 | the future supply of the class (bottom vocabularies + Result decoding + exhaustive consumers + structured emission) |
| 3 | [#697], [#698], [#705]'s gate half; gate rot as a class |

## Open questions and where they get decided

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | shared error/vocabulary ownership | `Unsupported` remains in `chelis-types` for Phase 1 compatibility, but the Phase 2 dependency audit corrected the earlier claim that types is the workspace bottom: it depends on Deep and Pred and cannot be the runtime vocabulary owner. Closed identity moves to dependency-free `chelis-vocab`; error rendering may continue to wrap it above that boundary. Per-crate mirrors remain forbidden | §C2, §C4.1, §C6 of this doc |
| 2 | whether `check` should pre-report target-independent unsupported constructs | **RESOLVED** by `capability_table.md`: yes for semantic-table (A) rejections - they are target-independent type facts; target-level (B) rejections surface at build | `capability_table.md` §Derivations |
| 3 | lint allowlist mechanics | DECIDED for the temporary defense: the count registry remains blocking only during typed migration and is never a completion oracle | lint rule spec + §C4.5 |
| 4 | which gates survive Phase 3 as early-UX vs die | Phase 3 | §C5 rows 17-18 + gate contract |

## The one-sentence summary for a reviewer

Give every stage a way to say no (Result-typed emission), convert the ten
live yes-anyway sites to §C2 diagnostics, then take the pen away: decode raw
symbols and dtype IDs once into dependency-bottom closed types, exhaust those
types at every semantic consumer, build C expressions structurally, and keep
gates/lexical scans as UX and evidence rather than the thing standing between
an unsupported case and a plausible wrong number.

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
[#746]: https://github.com/Chelis-Lang/chelis/pull/746
[#776]: https://github.com/Chelis-Lang/chelis/issues/776
[#782]: https://github.com/Chelis-Lang/chelis/pull/782
[#791]: https://github.com/Chelis-Lang/chelis/pull/791
[#799]: https://github.com/Chelis-Lang/chelis/pull/799
