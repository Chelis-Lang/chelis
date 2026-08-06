# Loud Unsupported: the failure-channel contract

**Status:** Phases 0-2 are complete and accepted (Phase 0: PR [#746];
Phase 1: PR [#791]; Phase 2: PR [#799], merged 2026-07-24 with the
authoritative oracle green - `PHASE 2 ORACLE: PASS` - and a fresh-context
execution red team, PASS WITH FINDINGS, every finding dispositioned:
`docs/investigations/pr799_returned_function_values_redteam.md`). Count
baselines and the token tripwire remain supporting checks, not completion
evidence. Phase 3 is in progress: its typed rejection-authority (§C2.1)
and sealed diagnostic-kind (§C2.2) slices landed on main in PR [#1037]
(merged 2026-08-04). The first gate-contract slice then made the typed
compiler-api definitions the single policy consumed by both public build
paths, removed the stale C precision preflights, and froze the exact
syntactic `reject_*` inventory (source path, name, and visibility for free
functions and `impl` methods across both crate source trees) in
`crates/chelis-compiler-api/tests/phase3_gate_inventory.rs`. Shared gates take
the closed `BuildTarget` enum, so an unknown target cannot silently skip a
gate. The final gate-contract slice moved the Metal and effect policies to
compiler-api as typed shared definitions, applied the effect policy to host
tensor-helper DAGs as well as pure DAGs, and scoped pure-DAG rejection to the
entry actually emitted. It prevents a host tensor-helper `dropout` from
reaching the C emitter's panic boundary, cites the compiled-kernel owner
[#1192], and replaces stale closed-#616 Metal/HIP diagnostic prose with the
deciding [05-MOV-1] atom. Census rows 17-18 and row 26 are converted. The
authoritative Phase 3 runner now exists, but
its final root-realizability leg deliberately remains red until independently
owned [#912] removes the ignored/stubbed integration cases. Phase 3 is
therefore not complete. Phase 3 was amended 2026-07-30 to absorb
the typed slices. Phase 4
(ratchet totality, §C7) was added 2026-07-30 after the nine-PR
class-coverage review recorded on [#730] found four holes in this plan's
own detection mechanisms; §C7's opening paragraph states the finding.
Tracking issue: [#730].

**Implementation record (re-planned 2026-07-22).** The initial Phase 2 draft
treated a large source lint as a recurrence proof. Execution demonstrated
both laundering paths and false positives, so that mechanism was extracted to
PR [#815] and removed from this phase's acceptance argument. The final design
is the typed boundary specified here: failures are not type terms, unresolved
terms cannot enter codegen, target capability selection is fallible, and an
unknown builtin name has no structured expression identity.
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
failing). Phase 1 removed the censused live instances; Phase 2 implemented
the typed recurrence proof and was accepted 2026-07-24. Sibling
plans: `spec/design/dtype_semantics.md`
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
fallible boundary decoding, and exhaustive typed consumers, with narrow token
and count inventories retained only as defense-in-depth evidence, and (4) demotes the
pre-codegen gates from safety mechanism to early-UX, because a gate that is
the only line of defense rots (the one-entry `HOST_ONLY_BUILTINS` allowlist
let five builtins walk past it, [#682]/[#705]).

**Scope:** this contract changes failure representation and compiler-state
boundaries. It does not change numeric storage, wire-format IDs, dtype
semantics, or the set of supported operation cells. The long-tail op x dtype
guarantee belongs to [#729] Phase 4's capability table and is not duplicated
here.

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

## Class ownership

The remediation plans share types and call sites but own different decisions:

| decision | authority |
|---|---|
| closed compiler/runtime representation identity and stable ABI spelling | this contract: `EffectKind`, `RuntimeDType`, and `Repr` in `chelis-vocab` |
| dtype rounding, overflow, storage, casts, and operation semantics | `dtype_semantics.md` ([#729]) |
| target-independent operation acceptance | capability Table A ([#729] Phase 4) |
| per-backend implementation status | capability Table B ([#729] Phase 4) |
| failure channel and `unsupported:` rendering | this contract |
| root identity, manifest order, and artifact obligation | [#912] / [05-OBS-6] |
| checker error totality | `checker_totality.md` ([#731]) |
| value observation and formatting | `faithful_observation.md` ([#732]) |
| atom/issue authority for capability decisions | `spec_provenance.md` ([#733]) |

`RuntimeDType` and `Repr` own representation identity only. `RuntimeDType`
owns stable numeric IDs, canonical names, generated C macro names, and the
mapping to `Repr`. `Repr` names each active physical encoding and owns its byte
width. These declarations describe the current encodings. They do not approve
or select a storage format. `dtype_semantics.md` §C3 owns storage decisions.
They SHALL NOT own numeric finalization, value domains, cast behavior,
operation legality, or kernel availability. `HostTypeTerm` and
`ConcreteHostType` preserve checked logical identity; they SHALL NOT form a
second checker or dtype-semantics layer.

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
  monotonically non-increasing (typed boundaries, mutation oracles, and the
  source tripwire).

---

# Part I - the normative contracts (§C1-§C7)

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

One error kind, one shape, every stage. Phase 1 froze the brand, subject,
context, and stage clauses; [05-UNS-5] later added the visible typed-authority
clause because hiding the validated citation behind free-form hint prose did
not satisfy the normative surface contract.

```rust
pub struct Unsupported {
    /// What was encountered: an op, builtin name, dtype, tag, effect
    /// kind, or construct. Closed enum + payload, not a bare string.
    pub what: UnsupportedKind,
    /// The context of the encounter (the op family, target lane, or
    /// call position) - the `on <context>` clause of the rendering.
    /// (Added at Phase 1 ratification: the branded message format always
    /// carried a context clause; the struct now carries it explicitly.)
    pub context: String,
    /// Which stage refused (checker | lowering | codegen(target) | runtime).
    pub stage: Stage,
    /// Source span when one exists (lowering/codegen must thread it;
    /// `raise_lowering_error` already takes span + span_id).
    pub span: Option<SpanRef>,
    /// Opaque typed authority plus the supported alternative. Construction
    /// distinguishes a numbered-spec decision from tracked implementation
    /// work and rejects empty or unregistered citations.
    pub authority: RejectionAuthority,
}
```

Implemented as `chelis_types::unsupported::Unsupported`. The span field is
boxed so the `Err` variant stays small on Result-typed emission paths; this is
a representation detail, not a contract change.

**Message format:**
`unsupported: <what> on <context> (<stage>); <authority-kind> <citation>: <hint>` - branded with the
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
as structured fields - the branded string is the RENDERING of the contract,
not the contract; machine consumers match the structured kind, never prose;
`chelis build`/`eval` -> `error:` line + nonzero exit; compiled binary
(dynamic-only cases) -> stderr + nonzero exit. The [#687] oracle corpus
gains a rejected-cells section asserting these strings byte-for-byte per
lane (a rejection emitted differently per lane is lane skew, [#712]'s
shape).

**STATUS (updated 2026-08-01): the structured `chelis check` surface above is
still the TARGET, not current behavior.** The `Unsupported` object now carries
the §C2.1 opaque typed authority, but no stage constructs `Stage::Checker`
(`chelis check` never reaches lowering or codegen, so no `Unsupported` can
arrive there), no type on the `Unsupported` path derives `Serialize`, and
the build surface renders the branded string into a flat
`kind: "unsupported_feature"` envelope
(`crates/chelis-compiler-api/src/compiler.rs`, `unsupported_stage_error`).
Consumers - including this plan's own rejected-cells corpus - match prose
today. Two prerequisites, in order: [#729] Phase 4 supplies capability
Table A, without which `check` has no target-independent rejections to
report (`capability_table.md` §Derivations); then the structured payload
must be plumbed onto `schema::Diagnostic`. Until both land, the
`unsupported:` brand is the machine surface and tests may match it. The
byte-for-byte per-lane corpus assertions likewise arrive with [#732]
Phase 3; today's corpus is deliberately substring-level and says so.

**This is a normative relaxation, recorded as a decision.** Naming it
rather than leaving a later reader to discover it: the surfacing bullet
above says machine consumers match the structured kind, *never* prose, and
for as long as the two prerequisites are outstanding that prohibition is a
permission - matching the `unsupported:` brand is the sanctioned machine
surface, and the corpus doing so is correct rather than tolerated. The
permission is given an expiry that fires on its own: the `compile_fail`
doctest oracles in `crates/chelis-types/src/unsupported.rs`, one per type
on the `Unsupported` path, go red the day any of the four gains
`Serialize`, which is exactly the moment this note stops being true. They
run in CI through the
`cargo test -p chelis-types -p chelis-compiler-api --doc` gate stage
([#875]). When both prerequisites land, this STATUS block and those
oracles come out together and the bullet above stands as written ([#871]).

**Authored versus unimplemented:** a deliberately unsupported case cites the
spec atom that decides it (the [#733] linkage); a
not-yet-implemented case cites its issue. The capability table makes
the same distinction at the capability layer - `Rejected(atom)` vs
`Unimplemented { issue }` - and `RejectionAuthority` already makes it
structural on this diagnostic surface. Rendering prints `deliberate [atom]`
or `unimplemented chelis#N` before the hint, so free-form hint prose cannot
hide or contradict the validated identity.

### C2.1 Typed rejection authority (added 2026-07-30; lands at Phase 3)

**Deciding atom: [05-UNS-5]** (spec/05 §7, authored in the same change
as this section - the numbered spec decides the user-visible rule;
this section implements it). **Delivery status (2026-08-02): the typed
authority slice is implemented.** `Unsupported` no longer accepts a bare
hint; every current production constructor supplies an opaque validated
authority. The windowed-reduction dtype gap cites its implementation owner
[#729], and runtime-symbolic window extents cite the dynamic-shape owner
[#600]. Runtime-valued window/stride lists cite their compiled-lowering owner
[#1058], compiled tensor/list `to_string` cites [#1059], and host-runtime-only
builders cite the deliberate [05-HOST-1] contract; remediation instances
[#705], [#725], [#734], and [#959] are not capability owners.
Compiled `dropout` rejection cites its actual kernel owner [#1192]; the shared
host-helper traversal closes the former C-emitter panic path, while
entry-scoped pure-DAG compilation ignores unsupported effects in siblings it
does not emit. The remaining Phase 3 work is the independently owned [#912]
root-realizability interlock. The Metal/effect gate migration and the full
phase runner are implemented, but those focused slices do not claim Phase 3
completion while the runner's final leg is red.

The amended Phase 3 replaces the bare `hint` field on `Unsupported`
with an OPAQUE `RejectionAuthority` type. Two things are NOT the
mechanism, named so nobody re-derives them: required fields still
admit `Deliberate { atom: "", hint: "" }`, and a grammar check still
admits `[05-ZZZ-999]` - a well-shaped citation of an atom that exists
nowhere, or of chelis#1 (a merged, unrelated PR), is arbitrary prose
wearing a token. The mechanism is validated construction against
REGISTRIES of what actually exists:

- Fields and scalar constructors are private. Exported macros cross one
  public implementation edge each, taking the raw literal and performing
  validation inside `chelis-types`; downstream crates cannot construct a
  `SpecAtomRef`/`IssueRef` and then compose it with a second generic authority
  constructor. `scripts/check_rejection_authority_boundary.py` locks the
  complete public-function inventory of the owning module (free functions as
  well as methods), rejects extra module/re-export/include edges, rejects
  direct production calls to the hidden builders, and mutation-tests those
  bypasses. Construction itself refuses [05-UNS-1..6], so a macro alias or a
  direct validating-builder call cannot turn the response contract into a
  semantic authority.
- `SpecAtomRef` validates in two gates: the `[NN-AAA-N]` grammar,
  then MEMBERSHIP in the derived atom registry - a generated artifact
  parsed from the numbered specs' blockquote atoms and locked by a
  byte-agreement test (the §C4.3 generated-header pattern), so the
  registry cannot drift from `spec/` silently. Citing an atom that no
  numbered spec declares is a construction error.
- `IssueRef` wraps `NonZeroU32` AND validates membership in a
  checked-in issue manifest recording, per number: it is an ISSUE
  (not a PR), and it is OPEN. **A checked-in manifest is
  self-authorizing unless its edits are gated** - the same PR that
  cites a bogus number can add the manifest row that blesses it (the
  2026-07-30 addendum's countermodel: after a same-PR edit, a closed
  issue, a PR number, and a 404 all "validate"). So manifest
  ADDITIONS receive blocking LIVE validation: every construction,
  registry, validator, imported liveness-helper, detector, and workflow input
  is in the `Rejection Authority Liveness` change set, and that job verifies
  every added or modified row against the live tracker (exists, is an
  issue, is open) before the PR can merge. Phase 4's §C7.5 scheduled job
  will re-verify the STANDING manifest for drift, so a cited issue closing
  later makes the stale citation red - the shell contract's "probe flips
  green, remove the citation" rule, pointed inward. Until that Phase 4 job
  lands, standing-state drift remains a named pending control rather than an
  implied continuous guarantee. Membership answers the compile-time question;
  the change-gated job answers the truth question when authority inputs change.
- Hints are validated non-empty; direct struct-literal construction and the
  former scalar constructor composition from outside the owning module are
  privacy errors, locked by `compile_fail` doctests. The required CI job also
  runs the boundary checker before the network-backed manifest validation.
- **These registries are the named pre-table authority source.**
  Phase 3 does not wait for [#729]'s capability table; when Table A/B
  lands, its `Rejected(atom)` / `Unimplemented { issue }` cells
  populate this type through the same validation, and the registries
  become derivation inputs rather than the standalone source.

**Calibrated claim:** construction proves only citation IDENTITY and
last-verified tracker STATE - an existing atom, or a manifest member
that was live-verified as an open ISSUE when its row was added and at
the last scheduled re-verification since. Between schedules the issue
manifest means "open at last verification", never "open this instant".
Neither registry proves RELEVANCE: whether an atom semantically
decides this rejection, or an issue actually tracks implementing this
rejected site/capability, remains an explicit review obligation.
`IssueRef` therefore MUST NOT be described as proving "the tracking
issue"; it proves only "a real open issue at last verification" until
review establishes the relationship. The authority-bearing rendering is
`;<space>deliberate [atom]: <hint>` or
`;<space>unimplemented chelis#N: <hint>`. Negative controls land with the change, one per
admission route: the empty atom, the malformed atom, the WELL-SHAPED
NONEXISTENT atom, issue zero, a real-but-closed issue, a real number
that is a PR rather than an issue, a nonexistent nonzero issue, and
the out-of-module struct literal each fail to compile, construct, or
validate; a same-PR manifest addition of a closed issue, a PR number,
or a nonexistent number fails the change-gated live validation. An
open but unrelated issue (chelis#879) is the supported structural
control: it passes construction/live validation and MUST fail review
when attached to an unrelated rejection. The [#687] corpus update
rides the same PR per B1. Before Table A/B exists, each production site is
adjudicated against an already-controlling numbered-spec rule or the actual
open issue that implements that capability. Table A/B later derives those
decisions through the same type; this design document never authors a
capability decision (§I1).

### C2.2 The closed kind vocabulary (added 2026-07-30; lands at Phase 3)

Before this slice, the machine-facing diagnostic `kind` was a bare `String`
chosen three unpoliced ways: hand-typed literals (~23 sites spelled
`"unsupported_feature"` themselves; `unsupported_stage_error`, the only
typed `Unsupported -> Diagnostic` path, has 4 call sites), a
`format!("{:?}", error.kind)` of a foreign enum (an enum rename
silently rewrites the wire), and message-substring dispatch. A
capability rejection could be mislabeled `"compile_error"` and every test
stayed green. Inventory and instances: [#959].

**Deciding atom: [05-UNS-6]** (spec/05 §7, authored in the same change
as this section; the stable kind spellings are a user-visible machine
contract and live in the numbered spec, not here). The amended Phase 3
implements it with the plan's own §C4.1 move plus a sealed producer -
a public closed enum alone is NOT the seal, because a public variant
remains directly constructible:

**Delivery status (2026-08-02): implemented as a focused Phase 3 slice.**
`DiagnosticKind`, the sealed producer/wire split, the typed general-kind
projection, the compile-fail controls, the consumer-to-producer conversion
mutation, the two same-crate planted privacy mutations, and the added-kind
exhaustiveness mutation now execute through
`scripts/diagnostic_kind_oracle.py`. The required
`Diagnostic Kind Mutation Oracle` job runs that actual mutation runner when
any owner, oracle, detector, or workflow input changes; its path detector is
self-triggering and fails safe on an unreadable change set. That runner is the
C2.2 component oracle, not a claim that Phase 3's gate work is complete.
The old Reef message classifier had no typed cause to justify its
`package_not_found` / `lockfile_error` guesses, so those substring-invented
spellings collapse to the honest `reef_error` kind rather than entering the
closed vocabulary as false precision.

- **Identity**: `DiagnosticKind` joins `chelis-vocab` (stable wire
  spellings via `as_str()`, `Result`-only `decode`, no serde
  dependency - kind identity is exactly the crate's charter, like
  `EffectKind` and `RuntimeDType`). A wire-spelling lock test freezes
  `as_str()` output against the [05-UNS-6] spellings (the
  generated-header agreement pattern, §C4.3).
- **The seal**: producer construction is a crate-private chokepoint in
  compiler-api. `stage_error`'s kind parameter is a type that
  structurally EXCLUDES the unsupported kind (a `GeneralKind`
  projection of the vocabulary); the `unsupported_feature` spelling is
  reachable only through `unsupported_stage_error(Unsupported)`, whose
  internal envelope constructor is private to its module. Handing
  `DiagnosticKind::UnsupportedFeature` to any general envelope API is
  therefore a type error, not a reviewed convention. The chokepoint's
  privacy is locked by `compile_fail` doctests.
- **The literal and mutation routes close by privacy, not by scan**:
  `Diagnostic.kind` becomes a PRIVATE field, with construction and the
  read accessor living in the sealed schema module.
  `#[non_exhaustive]` alone is NOT the seal - it binds only
  other crates, and any module inside compiler-api could still write
  `Diagnostic { kind: "unsupported_feature".to_owned(), .. }` or
  mutate a public field after construction. With the field private,
  the in-crate literal and the post-construction mutation are ordinary
  privacy errors everywhere outside the sealed module - which is the
  chokepoint itself. The existing hand-rolled `Diagnostic` literals
  (the `reef_error` substring-dispatch family and kin) converge onto
  the constructors as part of this change; no architecture scan
  carries the seal.
- **Deserialization is not allowed to be a back door.** A private
  field does not seal a type that derives `Deserialize`: serde IS a
  public constructor, and
  `serde_json::from_str(r#"{"kind":"unsupported_feature",..}"#)`
  builds the forged value without touching the chokepoint (executed in
  the 2026-07-30 addendum). The producer type therefore derives
  `Serialize` ONLY. Wire READING moves to a separate consumer-side
  graph (`WireDiagnostic`, `WireApiSuccess`/`WireApiFailure`,
  `WireApiEnvelope`, `WireCheckResult`, `WireBatchResult`, and
  `WireBatchResultEnvelope`) that derives `Deserialize` for tests, tooling,
  and shells. The complete Tide response and batch shapes therefore remain
  decodable without placing `Deserialize` on a producer carrier. The
  envelope-assembly APIs accept only producer types, so a deserialized value
  cannot re-enter the production pipeline: there is no consumer-to-producer
  conversion. **Threat-model
  calibration, stated:** no type system stops a process from printing
  arbitrary bytes to stdout; what this seals is the PRODUCTION
  pipeline - every diagnostic that reaches the wire through the
  envelope APIs was built at the chokepoint. Forgery outside that
  pipeline is out of scope here and is what [#733]'s provenance layer
  exists for.

Negative controls land with the change, one per bypass route, not only
the old free-string route: a free string literal, a direct
`DiagnosticKind::UnsupportedFeature` argument to a general envelope
API, and an out-of-crate `Diagnostic` literal each fail to compile
(`compile_fail` doctests); a serde-deserialized `WireDiagnostic`
handed to any envelope-assembly API fails to compile (the
no-conversion rule above); the producer type gaining `Deserialize` is
expiry-locked with a `compile_fail` doctest in the [#871] style; an
IN-CRATE literal outside the sealed module and a post-construction
`kind` mutation are proven unwritable by planted-mutation controls in
the Phase 3 oracle - a temporary planted bypass in a non-chokepoint
compiler-api module must fail the workspace check, byte-restored
after (the Phase 2 oracle's planted style, which is how a same-crate
privacy violation can be continuously proven at all - a
`compile_fail` doctest compiles as a foreign crate and cannot see
in-crate privacy). A planted `From<WireDiagnostic> for Diagnostic` conversion
must make the same doctest component oracle go red, proving that the negative
control tests the conversion route rather than only a direct vector type
mismatch. The change-path CI job executes these mutations themselves,
not merely the Python runner's unit tests; the `format!("{:?}")` and message-substring
dispatch sites are gone.
The wire field itself remains a `String` - the enum governs producers,
the string is the rendering - and nothing here adds `Serialize` or
otherwise lifts the §C2 STATUS relaxation above; the [#871] expiry
doctests stay untouched.
Boundary: this plan owns the kind vocabulary and the
`unsupported_feature`-only-from-`Unsupported` rule; span threading and
the check-JSON serialization mechanics stay with [#883]/[#886] (§I2).

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

**The `EmittedExpr` rule:** `EmittedExpr` is a private structured
C-expression representation with typed builders. Open-set dispatch returns
`Result<EmittedExpr, Unsupported>`; an unmatched case has no expression
constructor. Rendering to `String` occurs only after a structured expression
has been constructed. A general raw-string constructor is not part of the
Phase 2 completion surface.

## C4. The ratchets (making reintroduction structurally unrealizable)

The primary mechanisms are typed and exhaustive. A lexical rule may detect a
known spelling, but it SHALL NOT be cited as proof that a substitution cannot
be represented.

1. **One dependency-bottom vocabulary owner.** `chelis-vocab` has no
   dependencies, standard library, allocation, or unsafe code. It owns
   `EffectKind`, `RuntimeDType`, and `Repr`. These identity declarations are
   closed and have no unknown, custom, or fallback variant. `EffectKind` and `RuntimeDType`
   implement no `Default` and expose only `Result` decoders. Effect metadata
   decoding distinguishes `Missing`, `Malformed`, and
   `Unknown { symbol: &'a str }`. The unknown case borrows the input symbol.
   Runtime dtype decoding preserves the invalid raw ID. `chelis-types` is not
   the owner: it depends on `chelis-deep` and `chelis-pred`, while the runtime
   must consume the same dtype vocabulary without depending on the checker.
   This bottom placement also leaves a cycle-free placement for the `Prim` and
   `BuiltinId` identities required by the capability table. Semantic behavior
   and storage decisions remain in `dtype_semantics.md`.
2. **Decode once, then exhaust.** Raw strings and raw dtype integers exist
   only at serialization/FFI boundaries. `chelis-deep` adapts metadata shape
   into the bottom crate's effect decoder; every checker, effects, lowering,
   evaluator, and decompiler decision receives `EffectKind`. The C runtime
   decodes `chelis_tensor.dtype` and dtype arguments immediately; sizing and
   reading helpers accept `RuntimeDType`, never `c_int`. Matches over these
   enums have no wildcard arm. Adding a variant is therefore a compile-error
   work-list at every semantic consumer.
3. **Generated Rust/C dtype agreement.** The `RuntimeDType` declaration and
   its `Repr` mapping are the sole source for Rust IDs, canonical names, C macro
   names, and physical encodings. Byte widths derive from `Repr`.
   `chelis-runtime` owns the C-header renderer and the checked-in runtime
   artifact. The renderer reads the vocabulary declarations, and the host,
   HIP, and Metal runtime headers consume the checked-in fragment. A
   byte-for-byte regeneration test and a Rust round-trip table test lock
   agreement. Every generated C size switch has an aborting `default` that
   prints the raw ID; no default may select f32.
4. **Structured emission.** Open-set builtin dispatch returns
   `Result<EmittedExpr, Unsupported>`. `EmittedExpr` is a private C AST with
   typed builders and no general raw-string construction path.
5. **Source inventories are supporting evidence.** The token/count tripwire
   stays blocking while the typed migration is incomplete. It is neither a
   site identity nor an exhaustiveness proof: aliases, bindings, indirection,
   equivalent numeric-default spellings, count relocation, and in-crate raw
   emission can evade it. The corresponding typed mutation oracle is the
   authority. For the hosted [#732] no-third-formatter classes the same
   textual limits apply and the residue is DECLARED at the owning rule
   (`faithful_observation.md` §B2.4, each piece with its owner):
   derived-Debug containers embedding floats, bare `{}` Display /
   `.to_string()` of numeric payloads, and exits born outside the
   declared `OBSERVATION_EXIT_SURFACES`.
6. **Host types are a staged typed pipeline.** The host-type layer consumes
   checked type metadata; it does not re-infer source types. `HostTypeTerm`
   preserves exact
   logical scalar precision and gives named type, precision, rank, inference,
   and bottom states distinct variants. Missing and malformed metadata are
   `HostTypeDecodeError`, not terms. Only `HostTypeTerm::into_concrete` may
   produce `ConcreteHostType`, and it returns `HostTypeResolutionError` while
   any variable or `Never` remains. Target ABI selection consumes only
   `ConcreteHostType` plus the target capability decision and returns
   `Result<HostAbiType, Unsupported>`; backend emitters consume only
   `HostAbiType`. `int8` and `int16` select their exact existing
   `int8_t`/`int16_t` C representations; `f16` and `bf16` remain known
   logical types while the C-host capability cell rejects their scalar
   representation pending [#729]'s grounded storage and rounding. There is
   no `Unknown` variant, no `Default`, and no
   conversion from a decode/resolution failure to a concrete or ABI type.
   Before Table B is generated, the target selector is a private exhaustive
   adapter whose rejection arms cite a spec atom or implementation issue.
   [#729] Phase 4 replaces that adapter's decisions without changing this
   typed boundary.

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
`loud_unsupported_phase1.rs`. Chelis#729 Phase 3 subsequently turns the
supported scalar-math and narrow-scalar rows 2/4/6/7 into exact positive
controls while leaving genuinely unsupported siblings branded. Appended rows:

| # | site | substitutes | issue | status |
|---|---|---|---|---|
| 20 | `host.rs` handle-effect arm (host-lane sibling of row 9) | drops handler, lowers body for non-`random` kinds | [#709]-adjacent (P1 discovery, B2.5) | CONVERTED with row 9 in the same change set (test `unknown_effect_kind_is_rejected`) |
| 21 | `lower.rs` `expand` positional-axis `unwrap_or(0)` | axis 0 | flagged by [#782] | CONVERTED: fatal raise (section C1.4; a computed axis previously expanded axis 0 silently) |
| 22 | `lower.rs` `tuple-get` index `unwrap_or(0)` | field 0 | flagged by [#782] | CONVERTED: raise (section C1.4; compile-time index by construction) |
| 23 | `lower.rs` conv2d present-but-non-literal stride/padding `unwrap_or(1)`/`unwrap_or(0)` | stride 1 / padding 0 | P1 discovery (the [#776] shape); FILED as [#795] | flagged and left per B2.5 (absent-arg defaults are the documented optional-arg semantics; the present-but-non-literal case needs its own probe + issue); baselined in the tripwire's numeric-unwrap class. **MIGRATION DISPOSITION (recorded 2026-08-04):** this is a true Phase 1 migration miss, not a new mechanism. The shared optional-static-argument resolver distinguishes `Absent`, `Resolved(T)`, and `PresentButUnresolvable(Unsupported)`; callers apply their documented default only to `Absent`, propagate the branded §C2 diagnostic for `PresentButUnresolvable`, and consume `Resolved(T)`. The conversion audits every optional compile-time argument in `lower.rs`, not only conv2d, and routes each through that resolver or records a typed, tested reason it cannot. [#795]'s liveness probe and absent/present negative parity land with the migration. The tripwire's `unwrap-or-numeric-literal` count shrinks by the two conv2d sites and by every equivalent site the audit migrates, per B1; a replacement local `match` or `unwrap_or` is red. This work does not wait for Phase 3 or Phase 4 because it is completion of the already-frozen Phase 1 construction path |
| 24 | `emit.rs` `emit_fused_elem` and `emit_fused_reduce` non-f32 arms | `panic!` (row-two: panic, not substitution) | [#919] (P1 discovery, B2.5); [#951] owns the surviving `emit_fused_reduce` half | CONVERTED in [#932]: both arms now return a section C2 `Unsupported` instead of panicking (tests `f64_activations_compute_in_double_not_float`, `f64_exp_returns_full_double_precision`, and both `*_is_a_diagnostic_not_a_panic`). `emit_fused_elem` is additionally **widened** to f64, so its arm now fires only for the reduced-float and integer dtypes ([#691] - whose direct DAG integer path is repaired by [#729] Phase 3; PR #1164 rehomed the emitter citations off that issue as authority and it CLOSED 2026-08-04, so this arm's remaining scope is the reduced-float and integer dtypes on its own terms rather than under a live [#691] rejection); `emit_fused_reduce` stays a rejection ([#951]) - a shipped divergence from `spec/04-type-system.md` §1.1.3's admitted f64 cell, recorded here so [#729] Phase 4 inherits a known hole rather than discovering one |

**2026-07-30 ratchet-review appends (rows 24-27).** Filed per B2.5 from
the nine-PR class-coverage review recorded on [#730]. Each row lands
censused-not-fixed; rows 24-25 are Phase 4 work, rows 26-27 split
between the amended Phase 3 (row 26) and Phase 4 scope entry plus
[#893] (row 27):

| # | site | substitutes | issue | status |
|---|---|---|---|---|
| 24 | the lowering/emission panic family: `emit_fused_elem`/`emit_fused_reduce` f32-hardcoded panics in `chelis-backend-c/src/emit.rs` plus ~150 production `panic!`/`unreachable!`/`todo!` across `chelis-ir` and the three backends, ~44 of them §C2 rejections wearing panics (row-two: panic, not substitution - the rows 11/12 class, uncensused growth) | - | [#919] (the fused pair) + [#957] (the family) | live (population measured 2026-07-30; the fused pair execution-confirmed per [#919]); backing tests land with the Phase 4 §C7.2 sweep |
| 25 | `chelis-ir/src/host.rs` einsum output-precision inference: the `.unwrap_or(chelis_types::types::Prim::F32)` site (line 8843 as of 2026-08-04, re-measured; the token is the durable anchor - #975 already moved it once, and the site has drifted again since) - path-qualified, so the `unwrap_or(Prim::` token misses it; in-scope for the class and absent from BASELINE | F32 precision | [#958] | live-suspect (liveness not execution-confirmed - the empty-operand path may be checker-guarded); §C1.4 raise-or-prove applies regardless; converted at Phase 4 |
| 26 | the rejection-inventory class, anchored to expression identities rather than line numbers: the CLI-local `reject_*` family and hand-typed `"unsupported: "` literals, plus compiler-api free-literal kinds, debug/substr kind dispatch, and uncited window-gate hints | (gate/kind skew, not a value) | [#959] | CONVERTED (2026-08-05). PR [#1037] sealed the diagnostic-kind and authority channels. The first gate-contract slice removed the duplicate HIP, host-builtin, eval-only, and windowed-reduction definitions from `chelis-cli`; the final slice moved the Metal/effect definitions into compiler-api, routes Surf and Deep plus host tensor-helper DAGs through those typed policies, and replaced Metal's hand-typed brand and closed-#616 prose with [04-TGT-1], [05-MOV-1], or the open [#729] implementation owner as appropriate. Both public build paths consume the typed compiler-api definitions, the stale C precision pair is deleted, shared gate targets are closed by `BuildTarget`, and `phase3_gate_inventory.rs` recursively freezes every syntactic `reject_*` definition under both crate source trees by source path, name, and visibility. That inventory deliberately does not claim semantic detection of a reimplementation hidden under an unrelated name; the typed shared-policy boundary and review remain the controls for that broader property. The full Phase 3 runner exists; its independent [#912] interlock, not row 26, remains red |
| 27 | `chelis-python` raw-i32 dtype surface: `dtype: i32` struct fields, no decode-on-entry anywhere, string-gated dtype checks, hardcoded DLPack `code: 2, bits: 32`, 4 production `unwrap_or_default()` - the crate is outside every `UnwrapOr*` tripwire scope and the Phase 2 oracle's evidence set | dtype identity by convention | [#960] (requirement half; representation fix shape is [#893]'s - the runtime-representation tracker, corrected 2026-07-30 from the earlier [#909] misassignment) | live ([#900] is the executed value-corrupting instance); scope entry + §C6.2 row at Phase 4; FFI/representation redesign at [#893] |

`chelis-runtime` is deliberately not a row: it is the reference
implementation of the §C6.2 decode-on-entry contract
(`require_runtime_dtype`, `tensor_elem_size(RuntimeDType)`, zero
fallbacks) and enters the derived scan scope at Phase 4 as a clean
baseline, not a finding.

**2026-08-04 disposition-sweep appends (rows 28-30).** Filed per B2.5
from the 2026-08-04 cross-plan disposition pass over the open [#730]
surface. Each row lands censused-not-fixed and is claimed by either a
named class-maintenance work item below or the dtype plan's checked-cast
work item. A row found between phases therefore extends a construction
chokepoint and its oracle; it never receives a site-local side patch merely
because the phase that introduced the mechanism has exited. The Issue map
records these maintenance items explicitly.

None of the three carries a lexical-tripwire baseline edit because no
spelling below matches a declared §C4.5 token class. That lexical limit is
not permission to make the row its own guard. The maintenance oracles below
bind the semantic families that lexical counting cannot recognize: recursive
type traversal, host cast planning, and indexed trap selection. Rows 29 and
30 also enter the probe corpus with their implementation changes rather than
waiting for §C7.4 to discover them.

| # | site | substitutes | issue | status |
|---|---|---|---|---|
| 28 | `chelis-prove/src/obligations.rs` `type_from_deep_depth`'s depth-32 fuse (`if depth > 32 { return Some(Type::Unit) }`, `:230-232` re-measured 2026-08-04 - [#872] filed it at `:238-241`, so the fuse expression is the durable anchor) | `Type::Unit` returned as a SUCCESS: the consumer `type_contains_depth` ends in `_ => false`, so the stand-in reads as "this type does not contain the opaque type" and a producer nested deeper than 32 generates NO proof obligation | [#872] | live-suspect (source-confirmed on `origin/main` @ `2b80b474`, not execution; no upstream depth bound was located in `DeepTypeResolver` or the parser, so the site is neither raised nor proven dead). This row is one half of maintenance item LU2: the same region also has `type_contains_depth`'s depth-16 `false` fuse. Both are deleted in favor of the same cycle-aware worklist and explicit exhaustion error; converting only this `Some(Type::Unit)` site would leave the adjacent false-negative mechanism intact. The five sibling semantic placeholders named in [#872] remain separate §C1.4 canary-and-comment debt because they are not traversal-exhaustion fuses |
| 29 | `chelis-backend-c/src/host_emit.rs` checked-`cast` host arm identity fallthrough (`_ => arg_vars[0].0.clone()` closing the `cast` match, `:2566` re-measured 2026-08-04 - [#1150] filed it at `:2572`, so the arm expression is the durable anchor) | no conversion at all on a host-built tensor: the f32 buffer is reinterpreted at the target dtype, with no `chelis_checked_float_to_int` call, no cast loop, no `Overflow` trap for an out-of-range element and no `Domain` trap for a non-finite one | [#1150] | live (read-confirmed on the generated C during the PR [#1144] fold; PRE-EXISTING, not introduced there). Silently violates [04-NUM-11] and [04-NUM-14]. Maintenance item LU6 owns the failure-channel half: every checked host cast is produced from an exhaustive `(source Prim, target Prim)` plan whose only identity cells are exact same-type pairs; every other pair emits the checked conversion selected by [#729]'s semantics or a cited typed rejection. The dtype plan's checked-cast work item owns those conversion semantics and the full source x target x surface x backend conformance product. A local replacement for this wildcard is insufficient because it leaves the next source/target pair outside the plan |
| 30 | checked-`cast` multi-offender trap identity, two sites: the eval whole-buffer int-width pre-pass (the `int_wide` collection short-circuits on domain BEFORE the per-width range loop runs) and the `#pragma omp parallel for` on the emitted conversion loop (`chelis-backend-c/src/emit.rs`) | (trap-kind skew, not a substituted value - the rows 17/18/26 shape) for a tensor carrying BOTH an out-of-range and a non-finite element: which trap kind fires is thread-scheduling-dependent under OpenMP, and eval is domain-biased rather than first-offender-in-order | [#1152] | live (execution-observed as the PR [#1144] Linux CI failure, where `cast_trunc`'s mixed-offender case exposed the class; macOS clang ignores the pragma, which is why local runs look deterministic - PRE-EXISTING for the checked `cast`, unobserved because untested). [04-NUM-12] and [04-NUM-15] fully decide this behavior, so this is numeric implementation work rather than an unsupported-channel patch. The [#729] checked-cast item owns a shared `IndexedTrapCandidate { flat_index, trap }` reduction in eval and compiled lanes; the lowest flat index wins independently of scheduling. This plan owns only rendering a candidate or implementation rejection through §C2. No separate [#730] implementation is permitted, and [#1152] is tracked under [#729] with an `Also part of #730` provenance note |

| # | site | substitutes | issue | status |
|---|---|---|---|---|
| 1 | `lower.rs` `lower_transcendental` non-float arm | `Const 0.0`, operand dropped | [#699] (+[#722] via grad) | live (test `cos_on_integer_tensor_is_not_silently_zeroed`) |
| 2 | `host_emit.rs:2300` builtin fallback | literal `0` | [#704] [#705] [#715] | live at P0 (test `no_build_ever_emits_a_silent_unsupported_builtin_stub`); [#682]'s bitwise/shift instances are CONVERTED and no longer reach this fallback - `bitand`/`bitor`/`bitxor`/`shl`/`shr` are exhaustive `SCALAR_NUMERIC_BUILTINS` arms emitting real declared-width C (landed 2026-07-31 with the [#935] generic-ADT specialization change, shipped in 0.17.4), and [#682] closed 2026-08-04 |
| 3 | `host_emit.rs:2236` string fallback | `chelis_string_from_cstr("<value>")` | [#734] | live at P0 (test `to_string_of_a_tensor_stringifies_in_the_compiled_lane`; probed 2026-07-16, status carried by P0 - scalar-arm controls green in `issue_734_tostring_placeholder.rs`); CONVERTED in PR [#1037] - the catch-all is now a section C2 `Unsupported` carrying `unimplemented_rejection!(1059, ...)`, so compiled tensor/list `to_string` rejects instead of substituting and the support half is owned by [#1059]; [#734] closed 2026-08-04 |
| 4 | `host_emit.rs:4305/4390` print of unclassifiable value | literal `<value>` text | [#714] symptom | live (test `c_f16_floor_prints_the_value_placeholder_today`) |
| 5 | HIP `emit.rs` `elem_kind` `_` arm | `ElemKind::F32` | [#689] | live (test `hip_int64_neg_emits_the_f32_fallback_kernel_today`; emission-proven) |
| 6 | `host.rs:7572-7583` `parse_host_type` `_` arm | `HostType::Unknown` -> downstream `int64_t`/`void*` | [#714] | live (tests `f16_scalar_abs_compiles_and_runs`, `f16_scalar_fraction_survives_compilation`) |
| 7 | `host.rs:7861-7882` arithmetic type default | `HostType::Int64` | [#714] [#718] | live (tests `f16_scalar_fraction_survives_compilation`, `c_scalar_overflow_traps_at_every_width`) |
| 8 | `lower.rs:7522-7537` reduce_window extraction | `vec![]` windows -> silent no-op | [#725] | live (test `c_nonliteral_window_and_strides_pool_or_reject`) |
| 9 | `lower.rs:9518-9519` handle-effect catch-all | drops handler, lowers body | [#709]-adjacent | live via `.dp` (test `unknown_effect_kind_is_rejected`) |
| 10 | emitted print helper `default:` arm | reads buffer as f32 | [#716] ([#728] owns fix) | live (test `c_print_of_f16_tensor_prints_f16_values`) |
| 11 | `emit.rs:4238/4406/4777` reduce panics | (row-two: panic, not substitution) | [#692] | live (test `int64_max_reduce_does_not_panic_the_compiler`) |
| 12 | `emit.rs:4509` window-length assert | (row-two) | [#725] half | live (test `c_nonliteral_window_does_not_panic_the_compiler`) |
| 13 | `lower.rs:9833`, `:10235` `unwrap_or(Prim::F32)` | F32 dtype | [#710]-adjacent, [#744] | **live via `.dp` build lane** for `:10235` (test `dp_bogus_cast_target_must_not_build_silently`; P0 re-execution refuted the dead-by-probe claim - the guard is eval-only, [#744]); eval guard locked green (canary `canary_dp_cast_bogus_dtype_is_guarded`); `:9833` needs an internal desync; §C1.4 applies. CONVERTED in PR [#791]: re-measured 2026-08-04, `lower.rs` contains no `unwrap_or(Prim::F32)` at all - a bogus `.dp` cast target now hits the `raise_bogus_target` fatal raise instead of building an f32 binary; [#744] closed 2026-08-04 |
| 14 | `named_axis.rs:430` `unwrap_or(Prim::F32)` | F32 dtype | audit item 7 | dead (reachable-surface clearance: `canary_vmap_int64_roots_keep_integer_precision`; the arm is internal-desync-only, undrivable from input); §C1.4 applies |
| 15 | `host_emit.rs` `assign_partition` non-tuple arm | emits a C comment, no assignment | audit item 7 | dead (reachable-surface clearance: `partition_agrees_across_lanes`; the arm is internal-desync-only, undrivable from input); §C1.4 applies |
| 16 | ~25 guarded `Const { 0.0 }` sites in `lower.rs` | zero values | backlog §pattern | dead (bare keywords are parse-guarded per `spec/03-deep-syntax.md` §8.1; canaries `canary_unknown_deep_tag_is_rejected`, `canary_bare_keyword_atom_fails_cleanly`, `canary_dynamic_fail_aborts_loudly`); §C1.4 applies |
| 17 | `HOST_ONLY_BUILTINS` one-entry allowlist | (gate, not site - lets sites 1-2 fire) | [#705] | CONVERTED in the first Phase 3 gate slice: `HOST_ONLY_BUILTINS` now has one definition in compiler-api, and both the CLI and compiler-api build paths consume its typed checked-program and concrete-host-program gates. Those shared APIs accept only `BuildTarget`, so no unknown string target can turn the pre-lowering walk into a no-op. The fallible host emitter remains the independent defense. The shared `tensor_scan` rejection is pinned by `phase3_reject_function_inventory_matches_the_reviewed_manifest` plus the existing CLI/compiler-api `tensor_scan` suites; [#682] no longer depends on this gate |
| 18 | duplicated/drifted gates | (gate skew) | [#697] [#698] | CONVERTED in the first Phase 3 gate slice. For [#697], both stale CLI C precision preflights were deleted: active-dtype admission no longer changes when an unrelated host declaration changes the lowering path, while the checker still rejects deferred `f8e4m3`. For [#698], the CLI consumes compiler-api's typed HIP gate, preserving supported f64/integer cells and direct-load f16/bf16 `BlasMatmul`, preserving the `ScatterElements` payload/index validation that the former CLI copy lacked, and rejecting real f16/bf16 compute in a matmul operand before emission. `phase3_gate_contract.rs` locks those positive and negative behaviors. `phase3_gate_inventory.rs` recursively freezes the exact syntactic `reject_*` manifest across both crate source trees; it does not pretend that name-based source discovery proves semantic uniqueness under arbitrary names |
| 19 | Metal `emit.rs:1419` `host_scalar_literal` pad-fill catch-all | `/* unsupported pad fill dtype */ 0` | [#745] (P0 token-sweep discovery, B2.5) | dead at P0 (canaries `metal_rejects_f64_with_a_specific_diagnostic`, `f8e4m3_is_rejected_in_both_lanes` - the gate/checker were the only defense); CONVERTED - §C1.4's raise-or-prove was applied in PR [#791] and re-typed in PR [#1037]: `host_scalar_literal` is now exhaustive over `Prim` with no catch-all, and f64/f8e4m3/string each return a section C2 `Unsupported` carrying its own authority, so the gate is no longer the only defense; [#745] closed 2026-08-04 |

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
the [#732] plan's three no-third-formatter classes per its Phase 0 item
3, the roadmap's Wave 0 handshake, and `faithful_observation.md`
§B2.4/§B2.8's instrument list:
`c-format-narrowing` (the original `%.16g`/`%.1f` row),
`rust-format-narrowing` (Rust precision-spec forms, subsuming the old
`{value:.1}` token), and `rust-debug-numeric-format` (Debug tokens at
the declared observation exit surfaces). The Rust format parser uses
Unicode XID rules at BOTH identifier positions - the capture argument
and the named dynamic precision `.ident$` - so valid non-ASCII
identifiers are in scope in either position, and the bare exponential
selectors `{v:e}`/`{v:E}` count as narrowing (round-4 F1/F5).
Pattern IDs are unique and every pattern must directly count its
own compiler-forced sample; global evidence from another pattern cannot
satisfy applicability. Every hosted class's new-site
message points at `faithful_observation.md` §B2.4, and that plan's
Phase 2 oracle cross-checks each class's baseline paths against its own
permitted sets. Hosted-class scopes - such as
`OBSERVATION_EXIT_SURFACES` - are declared by the OWNING plan's §B2
rule with residue declared per `faithful_observation.md` §B2.8, not by
this doc's §B2.8 derived-universe requirement, which governs this
plan's own classes; the hosted scopes are revisited when Phase 4's
§C7.1 derived-universe rewrite lands - an interlock recorded in both
docs).

Phase 0 freezes this table into the tripwire; additions after that are
either new work (filed + censused) or regressions (red gate). Once
Phase 4's §C7.4 census binding lands, the backing convention above is
machine-checked: every `test <name>` / `canary <name>` token in a
status cell must name an existing test function and every row must
carry an issue link, or the binding test is red.

## C6. Phase 2 ownership and consumer contract

This inventory is normative. A consumer may be removed only when the
underlying behavior is removed. A newly discovered consumer is added here
before implementation proceeds (B2.5).

### C6.1 Dependency placement and effect consumers

`chelis-vocab` is the dependency-bottom owner because `chelis-types` depends
on front-end crates while `chelis-runtime` must remain independent of the
checker. These are the permitted edges:

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

The capability-table identity types `Prim` and `BuiltinId` also belong in
`chelis-vocab`, with compatibility re-exports from `chelis-types` during the
move. Their dtype semantics remain in `chelis-types::dtype_semantics`. No
Deep/runtime/backend edge points upward from vocab.

The effect consumer set is exhaustive for the current tree:

| boundary/consumer | required typed behavior |
|---|---|
| `chelis-vocab::{EffectKind, EffectKindInput, EffectKindDecodeError<'a>}` | single declaration; `decode -> Result`; distinct missing/malformed/unknown errors; `Unknown { symbol: &'a str }` borrows the decoder input; canonical `symbol()` inverse |
| `chelis-deep` effect-metadata adapter | map absent key, non-symbol value, and unknown symbol without collapsing them |
| `chelis-types::infer::infer_handle_effect` | consume the adapter result; exhaust `EffectKind`; map each decode error to a checker diagnostic |
| `chelis-effects::infer_handle_effects` | exhaustively remove the handled kind; decode failures enter `EffectError` |
| `chelis-effects::validate_handler_expr` | exhaustive per-kind validation; decode failures are errors |
| `chelis-effects::validate_build_target_expr` | exhaustive typed target policy |
| `chelis-ir::lower::lower_handle_effect` | consume `Result`; exhaust `Random`/`Resource`; preserve distinct diagnostic payloads |
| `chelis-ir::host::lower_host_expr` | consume `Result`; no empty sentinel or raw comparison |
| `chelis-compiler-api::runtime::eval::HostEvaluator::eval_expr` | exhaustive typed evaluation; every decode error is `Err` |
| `chelis-surf::desugar` `WithSeed`/`WithDevice` | emit `EffectKind::symbol()` |
| both `chelis-surf::decompile_handle_effect` implementations | decode once; exhaust known kinds; preserve malformed/unknown Deep only through an explicit observation path |

The added-kind mutation oracle inserts a temporary variant in the single
vocab declaration and builds `chelis-deep`, `chelis-surf`, `chelis-types`,
`chelis-effects`, `chelis-ir`, and `chelis-compiler-api`. Every semantic row
above must produce a non-exhaustive-match error until it makes an explicit
decision. The source-inventory test only proves there is no untyped consumer
outside that rustc oracle; it is not itself the exhaustiveness mechanism.

### C6.2 Runtime dtype boundaries and consumers

`RuntimeDType` owns the stable ABI IDs `F32=0`, `F64=1`, `I32=2`, `Bool=3`,
`I64=4`, `Bf16=5`, `F16=6`, `I8=7`, and `I16=8`. It also owns the canonical
language spellings, C macro spellings, and mapping to `Repr`. `Repr` names the
active physical encodings and owns their byte widths. `RuntimeDType` derives
its byte width from `Repr`. These are current representation facts, not dtype
semantics or storage decisions. `dtype_semantics.md` §C3 owns the storage
decision. `chelis_tensor.dtype` and the C ABI arguments remain `int`; that is
the wire representation, not the internal type. Each element consumer must
use a pointer or value type that is compatible with `RuntimeDType::repr()`.
Equal byte widths do not permit one shared element view.

| boundary/consumer | required typed behavior |
|---|---|
| `chelis_runtime::dtype_header` and the runtime headers | render the checked-in C fragment from the vocab declarations; include it in the C, HIP, and Metal headers; remove handwritten ID/size copies |
| `chelis-runtime::{CHELIS_*}` | compatibility constants derive from `RuntimeDType::id()`, never literal integers |
| `TensorElement::DTYPE` / `DtypeMismatch` | carry `RuntimeDType`; decode the tensor field before comparing or accessing |
| `chelis_alloc`, `chelis_alloc_view`, `chelis_dtype_size`, `chelis_tensor_from_value_list_typed` | decode the inbound `c_int` immediately; invalid IDs abort with the raw ID before allocation, sizing, or element access |
| `tensor_elem_size` | signature is `fn(RuntimeDType) -> usize`; exhaustive, no fallback |
| `read_index_slot`, `chelis_tensor_to_f64`, list-from-tensor, comparison/where/cumsum/sort/trace/clamp/einsum, and tensor formatting | decode once, pass `RuntimeDType` into typed helpers, and select an element type that matches `repr()` |
| `data_as_f32` and `data_as_f32_const` | accept F32 and the current `BoolInBinary32` payload. Reject I32 with a debug assertion before access |
| clone/concat/split/gather/scatter/diagonal/contiguous byte-copy paths | decode before byte-width calculation; pass `RuntimeDType` to sizing; raw integers may be copied back only into the ABI field via `id()` |
| C/HIP/Metal `dtype_macro` and sparse/dtype-arm helpers | return `RuntimeDType` first and obtain the C spelling from the vocab declaration; no repeated `Prim -> "CHELIS_*"` tables |
| generated C element access | use `int32_t` for `TwosComplement32`. Keep `float` for `Ieee754Binary32` and the current `BoolInBinary32` payload |
| `chelis-python` tensor construction | use `RuntimeDType::F32.id()`; remove the local numeric constant |

The negative suite covers `-1`, the first unused ID (`9` for this frozen
table), `i32::MIN`, and `i32::MAX`. Each must fail at the decoder. Runtime
subprocess tests then pass those IDs to each public FFI dtype boundary and
assert nonzero exit before any allocation-size result or buffer read can be
observed. Positive coverage round-trips every ID and compares the generated C
fragment byte-for-byte with its checked-in artifact.

The runtime subprocess suite drives every raw dtype argument boundary plus a
tensor-field read with ID `9` and requires nonzero exit carrying the raw ID.
The added-dtype mutation oracle requires a compile error at every exhaustive
consumer until the new representation identity is handled explicitly. It is
executed by the Phase 2 oracle's controlled vocabulary mutation, which adds a
fully-decodable tenth `RuntimeDType` variant alongside the `EffectKind`
mutation in one workspace check and requires non-exhaustive-match errors in
`crates/chelis-runtime/src/lib.rs`, the runtime FFI's exhaustive dtype
boundary.

### C6.3 Host-type state and ABI boundary

The host-type boundary must represent every pre-codegen state explicitly.
Source counts are not an allowlist or an oracle; the completion proof is the
typed boundary that prevents unresolved state from entering codegen.

| semantic class | required representation |
|---|---|---|
| missing or malformed checked metadata/type syntax | `HostTypeDecodeError::{MissingTypeMetadata, MalformedTypeSyntax, UnknownPrimitive}`; no term is manufactured |
| legitimate type polymorphism | named `TypeVariable`, `HostPrecisionTerm::Variable`, and ordered `HostShapeSlot::RankVariable` entries; specialization or a resolution error before codegen |
| underconstrained inference | stable `HostInferenceVar` identity; unresolved variables return `HostTypeResolutionError` |
| invalid builtin result inference | typed error, distinct from a valid underconstrained inference variable |
| bottom/divergence | `HostTypeTerm::Never`; joins may eliminate bottom, but `Never` has no value or ABI representation |
| known logical type without target representation | exact `Prim` in `ConcreteHostType`; target selection returns `Unsupported` from the capability decision |
| emitter expectation/refinement | resolution before emission; helpers receive concrete element/parameter ABI types |
| declaration, boxing, unboxing, call, and callback representation | exhaustive `HostAbiType`; no `void *`, numeric zero, or other default for an unsupported state |
| reduced-float element inside a heap LIST (the `to_list` exit, chelis#732 Phase 2) | the NAMED boxed-only state `HostAbiType::ReducedFloatBoxed(prim)`, constructed in list-element position ONLY: the list is an ordinary `chelis_list *` whose elements live behind `chelis_value` boxes as exact f64 images; the state has no standalone C spelling (`c_type_name` is `None`, like the callback declarator) and every declaration/boxing/unboxing path that would materialize an f16/bf16 C scalar rejects with the chelis#714 diagnostic; scalar, tuple, dict, and function positions keep the wholesale rejection. Not an erasure: the anti-erasure lock asserts the two-outcome contract. Owned jointly with `faithful_observation.md` (its Phase 2 Delivered note records the same state; interlock edits are bidirectional per B2.6) |

The staged source contract is:

1. `decode_host_type(...) -> Result<HostTypeTerm, HostTypeDecodeError>` owns raw
   Deep syntax and metadata.
2. Inference and specialization operate on `HostTypeTerm`. A successful
   `into_concrete` is the only path to `ConcreteHostType`.
3. Target selection consumes `ConcreteHostType` and an authoritative target
   capability decision, returning `Result<HostAbiType, Unsupported>`. Before
   Table B exists, the decision comes from a private exhaustive adapter whose
   negative arms cite a spec atom or implementation issue.
4. Host codegen accepts `HostAbiType` only. Its declaration, boxing, unboxing,
   call, and callback matches are exhaustive and cannot observe a term or
   resolution error.

Any compatibility bridge used while implementing this boundary is private,
cannot convert to `ConcreteHostType` or `HostAbiType`, and cannot be passed to
emission. The Phase 2 endpoint contains no legacy `HostType::Unknown` and no
such bridge. Compile-time function signatures, not a source count, prove that
only resolved types reach codegen.

Every production compiler entry point returns host decode, inference,
resolution, and target-selection failures through its declared `Result`.
There is no public infallible host-lowering wrapper and no production caller
may turn one of those failures into a panic. Host-expression lowering is also
`Result`-typed: an unrecognized or malformed Deep expression returns a lowering
diagnostic instead of constructing a placeholder expression. `Unit` is created
only by the language's empty-tuple form and cannot be used as a temporary
replacement during refinement or conformance.

A generic ADT declaration is not a concrete field layout. Constructor,
pattern, and direct or nested field-access lowering must supply an applied ADT
type to one substitution operation, which returns either fully instantiated
fields or a typed arity/name/unresolved-term error. No lookup API exposes the
declaration's field terms as concrete fields. Direct anonymous callback
applications are specialized before host emission; an unresolved callable is
never represented by a null function pointer or numeric value. An expression
node may inhabit only its resolved ABI type: in particular, `Unit` emission
cannot satisfy a function, callback, or numeric ABI expectation.

A nullary generic constructor is the one layout-free exception to immediate
field substitution: it may carry its declaration's named ADT term until the
checked enclosing expectation supplies the applied arguments
([chelis#935](https://github.com/Chelis-Lang/chelis/issues/935)).
Zero-argument generic functions whose body is exactly that constructor have no
standalone C ABI and are specialized at concrete call sites; the call's checked
result type is materialized onto the inlined body before host-type resolution.
A missing call-site application still fails at the same resolved-type boundary,
and wrong ADT names or arities still reject before emission. This exception
does not permit an unresolved constructor term to cross into
`ConcreteHostType` or target ABI selection.

The same checked-application rule owns generic ADT match patterns
([chelis#936](https://github.com/Chelis-Lang/chelis/issues/936)):
nonrecursive ordinary-generic functions are specialized before helper-summary
probing, so each constructor arm sees the concrete scrutinee application.
Dimension parameters hidden behind ADTs are classified by recursively following
checked constructor-field types rather than by source spelling
([chelis#940](https://github.com/Chelis-Lang/chelis/issues/940)); parameters
that reach only tensor dimensions stay on the existing rank-specialization
path, while stored value parameters remain ordinary type polymorphism.
Invoked recursive ordinary-generic functions are **outlined**, not inlined
([chelis#1158](https://github.com/Chelis-Lang/chelis/issues/1158), successor to
[chelis#941](https://github.com/Chelis-Lang/chelis/issues/941)). A generic call
site resolves the call's fully concrete signature, interns
`(callee, canonical signature) -> mangled symbol`, and emits an ordinary call;
host lowering drains the interner and emits one standalone definition per
interned pair. The two prohibitions that made this a boundary still bind:
lowering emits no reference to an omitted generic symbol, because every symbol
it names is one the drain defines, and it expands no recursive source without a
bound, because the interner's memo is what terminates the worklist rather than
a depth budget. Only fully concrete signatures are interned; a call site
carrying a free type, precision, or rank variable falls through to the existing
inline and rejection paths unchanged.

A free tensor **dimension name** is not in that list and does not fall through.
Concreteness there is a question about type TERMS: `tensor[n, f32]` is a
resolved term however its dims are spelled, so a signature mixing a type
variable with a symbolic dim — `f[a](x: a, t: tensor[n, f32])` — reaches the
interner looking concrete, monomorphizes `a`, and leaves `n` untouched, because
dims are not type-term slots and nothing substitutes them. The disagreement
with the caller's `tensor[2, f32]` would otherwise survive to C ABI projection
and abort the build as an internal error rather than a diagnosis, so the call
site compares the instantiated parameter and result types against the types the
call actually supplies and rejects a dimension disagreement under this same
brand. Dimension equality is what keeps that narrow rather than a ban on
tensors in generic signatures: a literal-dim signature agrees with its caller
and still monomorphizes.

Three residues keep the `[05-UNS-1]` brand, and the diagnostic distinguishes
them. A call site whose type arguments do not resolve to a concrete signature
names the term that still carries a free variable — this is the ordinary
chelis#730 unresolved-term shape reached through a generic call rather than a
new class. The dimension disagreement above is that residue's tensor-shaped
sibling: the terms did resolve, so the diagnostic names the instantiated type
and the type the call passes rather than a free variable.
Polymorphic recursion — a cycle that mints a NEW instantiation at
every level, as in `f[a]` calling `f[(a, a)]` — has no finite family of
monomorphic symbols at all, so it is rejected under two named bounds: a
per-callee specialization cap and a per-signature type-node cap. Both are
required, not one: the multiplicative shape reaches the count cap only after
constructing a signature larger than memory, so the size cap is what makes the
rejection prompt. The diagnostic prints the instantiation chain rather than
only the bound, because a rejection an author cannot localize is a boundary
that fails loudly at the compiler and silently at the desk. Neither residue may
be converted into a default instantiation: an unannotated argument is typed
from its own structure where the checker recorded enough to do so, and only a
bare name reference in a self-recursive call may take the enclosing
instantiation, which is the checker's own solution for it rather than a
substituted value.

For `fold`, the checked callback's first parameter is the authoritative
accumulator type. Host lowering materializes that type onto an unresolved
initializer such as `[]` before concrete host resolution
([chelis#939](https://github.com/Chelis-Lang/chelis/issues/939)). If the
callback accumulator is itself unresolved, no default is invented and the
existing resolved-type boundary remains loud.

The C ABI vocabulary distinguishes a typed callback declarator from a
first-class function value. General `ConcreteHostType -> HostAbiType`
conversion rejects `Function`; the callback constructor is private and is
available only for declared callback parameters and direct statically-known
callback arguments. It recursively requires ordinary value ABIs for every
parameter and result. A callback therefore has no standalone C type spelling,
and the former `Function -> void *` arm does not exist. Function results,
bindings, ADT fields, collection elements, dynamic callback selection, and
indirect values cannot acquire the callback variant. Both public compiler APIs
cross this same projection and return its structured `Unsupported` before any
C source or header is produced.

Positive/negative parity covers: every concrete primitive; each named variable
kind; missing versus malformed syntax; empty-list inference; `Never`
value-boundary rejection; supported ABI representations; and known logical
f16/bf16 values rejected by an unimplemented C-host ABI cell. Exact int8/int16
ABI selection is positive parity because those representations already exist.
The negative ABI tests assert the structured `UnsupportedKind::Dtype` and
`Stage::Codegen("c")`, not diagnostic prose alone. Resolved non-scalar value
classes the target cannot represent (function values) carry the distinct
`UnsupportedKind::HostAbi`; the unresolved-term `UnsupportedKind::HostType`
state is never used for a type that did resolve.

## C7. Ratchet totality (the coverage contract; added 2026-07-30)

**Why this section exists.** The 2026-07-30 nine-PR class-coverage
review (recorded on [#730]) found four holes in this plan's own
ratchets: the tripwire has no panic class and no early-return-default
class, its scope map is a hand-maintained default-allow crate list that
excludes the two user-closest crates (`chelis-python`, `chelis-runtime`),
unbranded and mislabeled rejections are unpoliced, and the §C6 consumer
inventory is append-by-hand. Every instance walked through those holes
was found by a person reading code, not by CI. That is this plan's own
disease one level up: §"Why not a pile of patches" diagnoses the class
as "optionality is the root cause" - each site chose independently - and
each RATCHET here had likewise chosen its own universe independently,
so each drifted independently. This section removes the option once.

**The contract, normative:** a ratchet's universe is DERIVED, never
enumerated. Every mechanism claiming recurrence prevention declares
where its universe comes from - the §C7.1 product-source manifest, the
§C7.3 dual-source consumer derivation, or the §C5 table itself. A
hand-maintained crate or file list is mis-rung on the
enforcement ladder (compile-error > lint > tripwire > gate > prose) for
the same reason a gate must not carry correctness: it rots invisibly.
Scope narrowing is a typed exclusion row with exactly two variants,
and every exclusion for every mechanism lives in ONE canonical
exclusion registry - a single checked-in artifact the §C7.5 filter
can name; an exclusion recorded anywhere else is itself a red
discovery test. An absent entry is never an exclusion. The variants
encode the one distinction that decides whether an issue belongs:

- `Exclusion::Structural { class, path, reason }` - a PERMANENT,
  correct classification: upstream-generated code, non-product prose,
  a language a Rust-only class cannot apply to. No issue field, by
  design: a classification no future work will change must not
  manufacture a permanently-open tracker entry (the work queue is
  `-label:tracking`, and an issue that can never close is noise
  there). The review that admits a `Structural` row is the registry
  edit itself, which the §C7.5 filter gates.
- `Exclusion::Deferred { class, path, reason, issue: IssueRef }` -
  temporary narrowing that parks real coverage work. The issue is
  mandatory and consumes the SAME §C2.1 issue manifest and validator
  as a rejection - not a second regex or bare-number channel. The
  §C7.5 standing re-validation applies, and the cited issue CLOSING
  makes the row red, forcing the exclusion's removal or its honest
  reclassification - the deferral ratchet working as intended.

The variant choice is enforced, not advisory: an exclusion that
silences a DISCOVERED consumer candidate (a cargo dependency edge, an
adapter hit) is always `Deferred` - deferring discovered coverage is
work, and work has an issue; the discovery tests reject a
`Structural` row over a discovered candidate. Validation proves
existence, kind, and open state at the specified verification times,
not that the issue is relevant to the exclusion; relevance remains a
named review obligation. The registry only shrinks, or grows through
one of these two typed forms. A new product-source
root - Cargo member or not - is born inside every ratchet or visibly
excluded from it, never silently outside it.

### C7.1 Derived scopes over the product-source manifest

**The universe is the product-source manifest, not the Cargo
workspace.** The Cargo member list is one derivation input, not the
whole universe: the shipped product includes non-Cargo roots -
`bindings/python/` (where the executed [#900] corruption path lives,
outside every Rust crate), the `py/` tooling sources, and
`packages/` - and a universe defined as "workspace members" leaves
every one of them structurally invisible. The manifest is therefore:

1. the Cargo workspace members, derived from the root `Cargo.toml` at
   test time; joined with
2. a checked-in registry of non-Cargo product roots, guarded by a
   totality check: every top-level repository directory containing
   shipped source must be classified - in-universe, or excluded with a
   typed §C7 exclusion row (`Structural` for permanent
   classifications like generated or upstream-owned trees; `Deferred`
   with its `IssueRef` where coverage work is parked) - and an
   unclassified root is
   a red test (the conformance MANIFEST/REGISTRY pattern). Adding a
   new product root of any language without classifying it is
   impossible to do quietly.

**"Shipped source", defined** (the totality guard's predicate, so it
cannot be a reading exercise): source whose content reaches a product
artifact or executes on the product/toolchain path - Rust sources and
`build.rs`, C/HIP/Metal headers and kernel templates, Python under
the registered binding/tooling roots, Chelis `.ch` sources, and
generated-code templates. Prose (`.md`), manifests and lockfiles,
editor and asset files are not source; binary artifacts (`.chb`,
`.zst`) are classified as ARTIFACTS, not source. `.ch` files are
Chelis-language product source whose ratchet surface is the
language's own gate (`chelis fmt`/`chelis lint`, the repo Style
Gate) - that classification IS their declared adapter, so a root like
`packages/` (46 shipped `.ch` files) is handled, not red, and the
Rust token classes do not pretend to scan a language they cannot
parse.

Each in-universe root is walked recursively (`src/**`, `build.rs`,
`include/**`, and for non-Cargo roots their own source globs - not
only `src/*.rs`, which is how the generated dtype headers escaped
scanning). Every non-Cargo root declares a language adapter and the
walker requires every shipped source extension beneath it to be
handled - by a scanning adapter or by an explicit classification (the
`.ch` and artifact classifications above); an unknown extension is
red, not silently skipped.

**An adapter is derived from the boundary the root ACTUALLY uses, not
from a countermodel's shape.** The Python boundary in
`bindings/python/` is numpy and DLPack flowing into the pyo3
`_native` extension module - `grep ctypes` over every registered root
returns nothing, and an adapter whose floor is
`ctypes.Structure`/`_fields_` flags zero real files while the [#900]
corruption line (`np.asarray(array, dtype=np.float64)` at
`bindings/python/chelis/__init__.py:393`) walks past it. The Python
adapter therefore works the way §C7.3 already works for Rust - edge
first, tokens within edges:

- **candidacy by import edge**: a Python file that imports `chelis`
  or the `chelis._native` extension module is a consumer candidate
  (the import graph is Python's cargo-dependency-edge analogue);
  numpy usage in a file with no such edge is not a candidate, which
  is what keeps the adapter from firing on every scientific-Python
  file and drowning the registry in exclusions;
- **within candidates**, the flagged forms are the real ingress/
  egress shapes: `np.asarray`/`np.array` calls carrying a `dtype=`
  argument that feed `_native` calls (the [#900] shape is the
  REQUIRED positive case), `np.from_dlpack` and DLPack capsule
  handling, `dtype` attribute plumbing on `_native`-owned objects,
  and any literal `CHELIS_*` constant or integer dtype id. The
  `ctypes` form stays recognized when it appears, but it is not the
  floor - the observed boundary is.

The C-family adapter recognizes integral `dtype` struct members and
`CHELIS_*` uses/definitions (that shape DOES match the real headers).
Language-independent boundary names are generated from the closed
vocabulary declaration - canonical names, C macro spellings, decoder
names, ABI field names - and combined with these language forms. Raw
ABI declarations outside generated bindings are forbidden. Rust-only
classes use a `Structural` exclusion row for genuinely inapplicable
non-Rust files. The point the registry establishes is that both the
ROOT and its language-specific coverage state are visible and
deliberate.

Each token class declares an explicit `TestsPolicy` (production-only
via `#[cfg(test)]` region splitting, or tests-included) so that
decision is per-class and reviewed rather than an accident of the
walker; the substitution-token classes are expected to go
production-only, which also removes the in-test noise the current
baseline carries. Class additions and widenings landing with this
contract:

- line-conjunction matching for `.unwrap_or(` / `.unwrap_or_else(` /
  `.map_or(` x `Prim::` on the same line, closing the path-qualified
  spelling evasion ([#958] is the live instance);
- an `.unwrap(` / `.expect(` count class over the lowering/emission
  scope (both spellings - a class that counts `.expect(` but not
  `.unwrap(` polices the polite spelling of the same panic);
- an early-return-default class: `else`-branch and `return`-position
  selection of `F32` (`Prim::F32`, `RuntimeDType::F32`,
  `ElemKind::F32`, `.c_macro()` on a default) outside a match arm -
  §C4.3 already says "no default may select f32" with nothing detecting
  the non-match spellings;
- the §C7.2 panic-token class (including the assertion macros).

`tree-sitter-chelis` is the worked exclusion example:
`Exclusion::Structural { class, path: "tree-sitter-chelis", reason:
"generated parser C, upstream-owned grammar" }` per class - a
permanent classification carrying no issue, recorded in the registry
rather than implied by sitting outside a walked directory. A
`Deferred` worked example is a registered root whose adapter is not
yet written: that row cites the issue tracking the adapter work and
goes red when it closes.

### C7.2 Panic terminals: sealed adapters, one validated invariant, spelling ratchets

**The claim, calibrated first.** [05-UNS-3] is a BEHAVIOR contract: any
panic or assertion failure reachable from `.ch`/`.dp` source is a
defect, whatever its spelling - `panic!`, `.unwrap()`, `assert!`, an
index out of bounds, an arithmetic overflow, or a panic inside a
function whose signature returns `Result` (a fallible signature does
not make its body panic-free). No signature rule or token scan proves
that behavior claim; its instruments are execution - the probe corpus,
the census canaries, and CLI-driven [05-UNS-3] tests. What this
section makes structural is narrower and stated exactly: the known
adapter routes become unconstructible, the sanctioned invariant
terminal cannot be written without real metadata, and every
ENUMERABLE panic spelling is ratcheted. §C1.3 says panics are for
broken compiler invariants with the guarantee named in the message;
nothing enforces either half today, and the measured population (~150
production panic-family sites across `chelis-ir` and the three
backends, ~44 of them §C2 rejections wearing panics, ~65 naming
neither an invariant nor a citation) says the prose rule does not
hold ([#957], census row 24). Three layers, strongest first; per
§C4.5's own rule the lexical layer is evidence and the structural
layers are the authority for the routes they cover:

1. **No infallible production entry point into lowering or emission.**
   §C6.3's "no public infallible host-lowering wrapper and no
   production caller may turn one of those failures into a panic"
   extends to every lowering and emission stage. The existing
   Result-to-panic adapters (the `unwrap_or_else(|d| panic!(..))`
   wrappers in `lower.rs`, the boundary unwraps in `host.rs` and Metal
   `dtype.rs`) become test-only - `pub(crate)` plus test-gating - and
   the privatization is locked by `compile_fail` doctests exactly as
   `host_abi.rs` locks its crate privacy. A production caller reaching
   for a panicking adapter is a compile error, not a review comment.
2. **One sanctioned invariant terminal, with validated metadata.** A
   `compiler_invariant!` macro (beside `Unsupported` in `chelis-types`)
   is the only blessed spelling for a §C1.3-legitimate panic. Its
   signature REQUIRES the named guarantee and the upstream guard
   citation (`compiler_invariant!(guarantee: "...", guard: "<canary/
   test or spec cite>", ...)`), validated in two tiers because
   non-emptiness alone accepts `"TODO"`: the macro const-asserts SHAPE
   at compile time (non-empty guarantee; guard non-empty and either a
   test/canary identifier or a `[NN-AAA-N]` atom token), and the
   Phase 4 oracle verifies the REFERENT - a guard naming a test or
   canary must resolve in the §C7.4 nextest-selected set, and a guard
   citing an atom must be a member of the §C2.1 atom registry.
   `guard: "TODO"` and `guard: "not-a-canary-or-spec-citation"` pass
   neither tier. "The guarantee is named" stops being prose and
   becomes an argument that cannot be empty or fabricated quietly,
   greppable as a single token. **Calibration, same as §C2.1's:**
   referent resolution proves the guard names a real, currently
   selected instrument or existing atom; whether that instrument
   actually GUARDS this site (a resolvable but unrelated test name
   passes both tiers) is not mechanizable and remains review's job -
   the tiers exist to make the fabricated and the stale citation
   unwritable, not to certify relevance.
3. **Spelling ratchets, explicitly the supporting rung.** Raw
   `panic!` / `unreachable!` / `todo!` / `unimplemented!` and the
   assertion macros (`assert!` / `assert_eq!` / `assert_ne!` /
   `debug_assert*`) in the derived production universe become tripwire
   token classes with frozen, shrink-only baselines, alongside §C7.1's
   `.unwrap(` / `.expect(` class; `[workspace.lints]` denies
   `clippy::todo` / `clippy::unimplemented` workspace-wide and
   `clippy::panic` / `clippy::unreachable` / `clippy::unwrap_used` /
   `clippy::expect_used` per emission crate as that crate's triage
   completes, with lint adoption itself checked against the derived
   member list. These detect the enumerable spellings; layers 1-2 are
   the structure; the non-enumerable routes (indexing, arithmetic,
   internal panics behind `Result`) remain owned by the behavior
   instruments named above, and no part of this plan may cite the
   ratchets as proof for them.

Negative controls, one per route: a planted raw `panic!`, a planted
`.unwrap()`, and a planted bare `assert!` in a derived-universe member
each go red; a `compiler_invariant!` with empty metadata fails to
compile and one with a fabricated non-empty guard (`"TODO"`) fails
the oracle's referent tier; the sealed adapters' privacy is
`compile_fail`-locked. For the non-enumerable routes, the control is a
CONTROLLED SOURCE MUTATION, not a scratch member or a product Cargo
feature - a disconnected planted function proves only that an
explicitly invoked panic panics, while a selectable feature remains
available to optimized and `--all-features` product builds. The
Phase 4 oracle refuses a dirty owner, temporarily replaces one REAL
lowering/emission site's behind-`Result` body with a
`compiler_invariant!` call - the sanctioned terminal, not a bare
`panic!` - builds the CLI into an oracle-only target directory,
drives that real site with a named `.ch` probe, then restores the
source byte-for-byte before reporting success. **What it asserts is
the invariant FRAME, not the planted prose.** The existing surfacing
machinery already converts a lowering panic payload verbatim into an
ordinary diagnostic (`catch_lowering` and
`panic_payload_to_lower_diagnostic`, `crates/chelis-ir/src/lower.rs`;
hook installed by the CLI), so a control that asserts its own planted
message cannot distinguish "the surfacing contract holds" from
"`catch_unwind` returned my string". `compiler_invariant!` therefore
emits a structured payload frame - the literal
`compiler invariant violated:` marker plus the guarantee and guard -
which only the sanctioned macro produces (a Phase 4 deliverable-4
item), and the oracle asserts nonzero exit, THAT frame with its guard
token in the rendering, and absence of any emitted artifact.
Calibrated: this proves panic-to-diagnostic conversion and frame
preservation for the sanctioned terminal at one named real site per
run.

**The no-shipped-seam guard is a NAMED-ARTIFACT check, stated on the
supporting rung.** It asserts the specific artifacts this plan
defines - the `internal-panic-seam` feature name and the oracle's
mutation sentinel - appear in no product manifest, product source, or
release/package workflow; its negative control temporarily adds that
named feature to a product crate and must go red. It is deliberately
NOT presented as a derived guard over "any runtime switch": the
repository legitimately carries a sanctioned test-selector
convention (`CHELIS_TEST_*`, `CHELIS_STYLE_GATE_DISABLE` - the
latter documented in the repo Style Gate section) at dozens of
product-source sites, and product crates carry eight legitimate
features; no derivation distinguishes those from a forbidden switch,
so a guard claiming to derive the class would be either red on day
one or a literal grep wearing B2.8's language. This plan polices the
one artifact it invented and does not adjudicate the pre-existing
selector convention.

The shipped-neighbor control is a STAGEABLE, credential-free subset
of the release path, not the release workflow itself: executing
`.github/workflows/release.yml` is a publish action (it holds
`contents: write` and creates releases), and an `--all-features`
build enables the z3/clarabel/carcara/arb native-dependency graph no
CI job routinely builds - neither is an oracle step. The control
builds `cargo build --release -p chelis-cli --features smt` (the
release configuration's own product build) from the clean restored
tree and runs `scripts/verify_release_smt.py` against it, drives the
same ordinary input without the mutation, and verifies the sentinel,
the frame marker, and the named feature are absent from the artifact
and its feature graph. Sequencing is one checkout, in order:
clean-tree refusal, in-place mutation with byte-for-byte restore
(the Phase 2 oracle's pattern), then the neighbor build from the
restored clean tree; the oracle-only target directory is never an
input to any package step. Stated plainly: this validates the
panic-SURFACING path. DISCOVERY of an unknown internal panic is what
the probe corpus and fuzzing are for; no planted control can certify
it, and this plan does not claim one does.

The ~44 rejection-shaped panics are census work, not annotation work:
they convert through the existing `Unsupported` channel in Phase 4's
sweep (triage buckets: convert-to-`Unsupported`,
migrate-to-`compiler_invariant!`, delete-dead).

### C7.3 Derived consumer discovery (dual-source, not circular)

The §C6 consumer inventories remain the normative content - the
hand-ratified floor the mutation oracle proves against. What changes
is that membership stops being a fixed file list, and the derivation
is deliberately NOT a single token scan: a scan that derives both the
coverage set and the oracle's required evidence proves only that the
two agree with each other, and a consumer spelled outside the token
set is invisible to both (the live [#960] form is `dtype: i32`, not
`dtype: c_int` - a one-token spelling gap inside the Rust workspace).
Two independent derivations, both feeding the same obligation:

1. **Dependency-graph candidates**: any manifest member with a cargo
   dependency edge to `chelis-vocab` or `chelis-runtime` is a consumer
   candidate, derived from `cargo metadata` - no spelling involved. An
   aliased import (`use chelis_vocab::RuntimeDType as R`) cannot hide
   the edge.
2. **Language-adapter candidates** over the §C7.1 universe. Rust
   carries the vocabulary type names, decoder names, effect symbol
   literals, and raw-boundary signatures in BOTH spellings
   (`dtype: c_int` and `dtype: i32`, plus `#[repr(C)]` structs carrying
   integer dtype fields). Non-Cargo files are not searched only for
   Rust-shaped tokens: the declared Python/C adapters consume the
   generated language-independent boundary names and recognize the
   real boundary forms in §C7.1. A Python file importing
   `chelis._native` with an `np.asarray(.., dtype=..)` coercion
   feeding a `_native` call (the [#900] shape), and a C struct with
   `int dtype` or a `CHELIS_F32` use, are candidates even when
   planted in an already registered root; a `ctypes` ABI mirror is
   recognized too, but it is not the adapter's floor (§C7.1). A root
   language with no adapter or classification, and a shipped source
   extension not covered by either, are red.

Every candidate from either source must appear in the §C6 inventory or
in an annotated exclusion; a candidate in neither is a red test. The
mutation oracle's required-evidence set derives from the RATIFIED §C6
inventory, not from either scan - scan finds, a human ratifies into
§C6, the oracle proves §C6 - so the compile-failure proof and the
discovery mechanism cannot silently agree on a shrunken universe.
Adversarial controls land with the change: an aliased import, a
`dtype: i32` field in a new Rust file, a scratch crate with a vocab
edge and no inventory row, a raw Python consumer and a raw C consumer
inside existing registered non-Cargo roots, and an unregistered
non-Cargo root must each go red. Benign Python/C files in those same
roots are the supported-neighbor controls and do not become consumers.
**Discovery is never path-filtered**: both derivations and the
manifest-totality guard run as ordinary workspace tests on every PR
(exactly as today's architecture test does). §C7.5's filter question
applies only to the expensive mutation legs; the cheap detection of a
new consumer or root must not depend on anyone predicting its path.
**Exclusion is the third authorization route and is gated like the
other two** (the 2026-07-30 addendum's countermodel: a candidate can
be silenced by an annotated exclusion without the mutation job ever
running). Three closures: every exclusion lives in the ONE canonical
exclusion registry (§C7's contract - an exclusion in any other
source is itself red, so there is no unnamed source the filter could
miss); that registry is in the §C7.5 mutation filter, so an exclusion
edit runs the legs pre-merge; and the discovery tests validate
exclusion annotations through the SAME `IssueRef` type/manifest/live
validator as §C2.1 - a discovered consumer candidate (dependency edge
or adapter hit) may be excluded only as `Exclusion::Deferred` with a
typed reference to a real open issue at last verification, and a
`Structural` row over a discovered candidate is red. A bare number or
unchecked issue token is not a valid schema value. Control: a planted exclusion for a vocab-edge candidate
must trigger the mutation job and fail until its `IssueRef` passes the
manifest and live checks; chelis#944 (closed issue), chelis#1 (PR),
chelis#999999999 (missing), and same-PR additions for those values are
negative controls. Chelis#879 is the live/open structural positive
control; using it for an unrelated exclusion MUST still fail review,
because the validator does not certify relevance. A planted exclusion
OUTSIDE the registry is red regardless of its annotation.

### C7.4 The census is doc-bound and its backings are selected

A test parses the §C5 tables out of this document (`include_str!`; the
`traversal_policy.rs` / `builtins.rs` precedent), extracts the
`test <name>` / `canary <name>` / `[#NNN]` tokens, and binds each row
to the SELECTED executable record, not to token presence - a function
that exists but is `#[ignore]`d or filtered out of every invoked
suite proves nothing:

- every named function must exist AND appear in the selected set of
  an invoked suite (verified against `cargo nextest list` output);
- an `#[ignore]`d backing is legal only when the row declares it, with
  the owning issue - the [#732] Phase 2 ignore-ledger pattern: the
  declared set and the actual ignore inventory must be EQUAL, so an
  undeclared skip and a stale declaration are both red;
- every row carries an issue link;
- a row whose status cell claims LIVE-by-execution names its
  probe-corpus record (`docs/investigations/probes/`), and the binder
  checks the named record exists - per B3.2 the corpus, not this
  table, is already the substantive-truth instrument.

Negative controls: a planted row naming a nonexistent function, a
planted backing that exists but is ignored without a row declaration,
a planted backing filtered out of the invoked suites, and a planted
live-row probe reference naming a nonexistent record each go red.

**Calibrated claim, stated at the binder's actual strength:** the
binder proves SELECTED, NON-IGNORED REGISTRATION plus record
presence - no stale references, no unselected or silently ignored
backings, no phantom probe citations. Selection cannot prove
SUBSTANCE: a no-op body with the right name is listed, selected, and
green, and `nextest list` cannot see inside it. Substantive liveness
and status truth remain the probe corpus's job (B3.2: "believe
execution, not this table"), and any future claim stronger than this
paragraph requires per-row expected-outcome records plus a
replace-with-no-op mutation control - deliberately not specified
here. The binder also cannot discover an UNCENSUSED site: an omitted
row has no token to parse. Discovery of the [#919] shape (sites that
"survived the Phase 1 sweep uncensused") belongs to the §C7.1/§C7.2
derived classes - which would have counted those panics - and to the
probe corpus, not to this binder.

### C7.5 Change-gated and scheduled authority

A named structural authority must actually execute: pre-merge when its
inputs change, and on a schedule as the drift canary for everything
else. A point-in-time run is phase-completion evidence, never
class-maintenance evidence. The Phase 2 oracle's two
controlled-mutation legs - the plan's own named authority - currently
run in no CI job; only their self-tests are continuous. A nightly
alone would be honest as a canary but is NOT recurrence prevention: it
detects a hole only after main has moved. Phase 4 therefore delivers
both halves:

1. **Change-gated, pre-merge, blocking - with the trigger loop
   closed.** A filter derived only from the already-ratified §C6
   inventory has a hole exactly where it matters: a NEW consumer is by
   definition not in the inventory yet, so its path would never fire
   the filter, and the nightly would find it only after merge. The
   loop closes in two moves. First, per §C7.3, discovery and the
   manifest-totality guard are unconditional per-PR workspace tests -
   a PR adding a consumer, member, or root goes red pre-merge with no
   filter involved, and can only go green by editing the ratified
   inventory or the registry. Second, the mutation job's path filter
   covers the files that edit therefore touches: the vocabulary owner
   sources, `host_abi.rs`, the RATIFIED INVENTORY FILE itself, the
   CANONICAL EXCLUSION REGISTRY (the third authorization route, per
   §C7.3 - one artifact, so the filter names it exhaustively), the
   §C2.1 issue manifest, the root `Cargo.toml`, the non-Cargo
   product-root registry, the oracle scripts, and the workflows. A new
   consumer thus cannot reach main without the discovery test forcing
   an inventory, exclusion-registry, or root-registry edit, and none
   of those edits
   can merge without the mutation legs running. The same job carries
   the §C2.1/§C7 exclusion live validation: added or modified issue-
   manifest rows AND added or modified canonical-exclusion rows are
   resolved through the same validator against the tracker (exists,
   is an issue rather than a PR, is open) before merge. It also runs
   the sibling [#729] §C6 census-liveness command
   (`scripts/capacity_census_liveness.py`) when the capacity census or
   its citations change - converged 2026-07-31 from that plan's
   original manual-only gate; the script, pass line, and census
   contract remain [#729]'s, the scheduling and gating are this
   section's (the [#732] oracle split), and the census artifacts join
   this filter. An exclusion
   row cannot bypass the check by citing a standing manifest entry:
   the changed exclusion's `IssueRef` is re-queried too. **The live
   check's operational consequences, named rather than implied:** the
   validation step requires `issues: read` on its job (`ci.yml`'s
   restrictive permissions block grants nothing by default, and no
   blocking job queries the issue API today - the open/close pattern
   this doc cites lives in separate `report` jobs with
   `issues: write`); tracker unavailability fails CLOSED as a
   retryable job failure, never a merge-through; and the live half is
   CI-owned - `scripts/gate.py --local` runs the membership and shape
   half offline, exactly as the workspace nextest stage is CI-owned
   in the existing gate split. The
   crate-scoped `HostAbiType` leg runs on every
   filter match; the workspace vocabulary leg runs when the vocab
   owner, the inventory, the exclusion registry, the root manifest, or
   the product-root registry changes. The job is classified in
   `scripts/test_gate.py`'s job tables in the same change.
   Trigger controls land with the workflow: a planted new workspace
   member, a planted vocab dependency edge, a planted file in a
   previously uninventoried consumer, and a planted non-Cargo root
   must each (a) fail the per-PR discovery tests and (b) have their
   forced inventory/registry edit matched by the filter - asserted
   against the workflow's path list, not assumed.
2. **Scheduled full matrix, the drift canary.**
   `loud-unsupported-nightly.yml`: a cron workflow running
   `scripts/loud_unsupported_phase2_oracle.py` in full,
   `scripts/faithful_observation_phase2_oracle.py` (whose contents remain
   [#732]'s, per §I2, and whose existing per-PR blocking job is prevention
   rather than a substitute for this scheduled drift canary), and the Phase 4
   oracle, with full standing re-validation of
   both the issue manifest and every `Deferred` exclusion's
   `IssueRef` (`Structural` rows carry no issue and are outside the
   liveness sweep by design), plus the [#729] §C6 census-liveness
   sweep (`scripts/capacity_census_liveness.py`, converged here
   2026-07-31 so a cited census issue closing goes red within a day),
   plus the open/close tracking-issue
   report pattern the repo's other nightlies use (the `heavy-e2e`
   precedent - "scheduled-run failures do not show a red status on
   main" is exactly the failure mode to avoid). This half catches what
   no path filter can name: toolchain drift, movement the filter did
   not anticipate, and the standing proof that the full matrix still
   completes. The workflow file is added to `NON_GATE_WORKFLOWS` in
   `scripts/test_gate.py` in the same change.

Neither half alone satisfies this section: the gated job without the
nightly leaves unfiltered drift invisible; the nightly without the
gated job is a canary mislabeled as prevention.

---

# Part II - process rules at every boundary

## B1. Freeze points

| contract | frozen at end of | may change after only by |
|---|---|---|
| §C2 `Unsupported` shape + message format | Phase 1 | this doc + [#687] rejected-corpus update, same PR |
| §C3 channel signatures (`Result` plumbing shape) | Phase 1 | this doc |
| §C5 census (as tripwire baseline) | Phase 0 | append-only via filed issue; removals only with the site's fix |
| §C4 vocabulary declarations and typed consumer inventory | Phase 2 | this doc + owning active spec, with an added-variant mutation oracle and the §C7.3 discovery scan updated in the same change |
| §C4.5 source-inventory tripwire | temporary during Phase 2 | this doc + the tripwire test; never a completion oracle |
| gate inventory (§C5 rows 17-18 resolution) | Phase 3 | this doc |
| §C2.2 `DiagnosticKind` vocabulary + wire spellings | Phase 3 | the deciding atom [05-UNS-6] (spec/05 §7) + this doc + the wire-spelling lock test, same PR |
| §C2.1 `RejectionAuthority` shape | Phase 3 | the deciding atom [05-UNS-5] (spec/05 §7) + this doc + [#687] corpus, same PR |
| §C7 universe derivations + the canonical exclusion registry | Phase 4 | exclusions shrink-only, in the one registry only; additions are typed rows - `Structural` with a reviewed reason, or `Deferred` carrying an `IssueRef` that passes changed-row live validation and relevance review |
| §C7.5 execution jobs (the change-gated `ci.yml` job + the nightly workflow + `NON_GATE_WORKFLOWS` entry) | Phase 4 | this doc + `scripts/test_gate.py`, same PR |

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
   with [#729] updates both documents in the same change set. The same
   rule governs the §I2 interlocks.
7. **A named structural authority executes pre-merge when its inputs
   change, and on a schedule otherwise** (§C7.5). A point-in-time
   oracle run is phase-completion evidence only; class maintenance
   requires the change-gated blocking job plus the scheduled full
   matrix with its open/close tracking-issue report. A scheduled-only
   authority is a canary, not prevention; an authority with neither is
   itself a hole of this class and gets censused.
8. **Ratchet universes are derived, never enumerated** (§C7). A
   recurrence mechanism whose scope is a hand-maintained list is
   mis-rung; scope narrowing is an annotated, issue-backed exclusion.
   Coverage drift is a red test, not a reading exercise.

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

**Explicitly not yours:** fixing anything or adding plumbing.

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
`to_tensor` exemplar (re-rendered to the branded section C2 shape); the
HIP admit-gate and Metal f64-gate exemplars carry the `unsupported:`
brand with their original message bodies - their full-shape
conformance rides Phase 3's gate work, per B2.1's
single-dedicated-change rule for diagnostic-wording migration.

**Explicitly not yours:** making any unsupported thing SUPPORTED (that is
[#729]'s or an op-owner's work; see §I1 for what your rejections do to
existing red tests); gate deletion.

**Oracle:** `issue_703_silent_placeholders.rs` fully green and
un-ignored; `scalar_stub_matrix.rs`'s stub-marker assertions green
(value-correctness rows go red-for-a-better-reason per §I1 and stay
ignored with updated notes); `reduce_window_nonliteral_matrix.rs` green
(both rows accept reject-with-diagnostic); the [#722] eval rows green via
row 1's raise (a loud lowering error, not zero gradients); the tripwire
census shrinks to rows 10, 17, 18.

## Phase 2 - un-writability (typed closed vocabularies)

**Status: COMPLETE AND ACCEPTED (PR [#799], merged 2026-07-24).** Items 1-5
below are implemented; the authoritative oracle ran green
(`PHASE 2 ORACLE: PASS`, including the combined vocabulary mutation), and
the fresh adversarial review confirmed that unresolved host types cannot
enter codegen and unsupported open-set dispatch cannot construct an emitted
expression (PASS WITH FINDINGS, dispositioned:
`docs/investigations/pr799_returned_function_values_redteam.md`). The
token/count tripwire remains a supporting check.

**You inherit:** a tree with no live silent fallbacks (Phase 1) and the
tripwire proving it.

**You deliver:**

1. The dependency-free `chelis-vocab` crate and its single declarations for
   `EffectKind`, `RuntimeDType`, and `Repr`, with Result-only decoders and the
   positive and negative tests in §C4.1.
2. The end-to-end `EffectKind` migration in §C6.1. No semantic consumer may
   compare the metadata string. The added-variant mutation must fail every
   named consumer until it explicitly handles the new kind.
3. The runtime dtype migration in §C6.2: generated Rust/C agreement,
   immediate FFI decoding, typed sizing/reading helpers, and invalid-ID tests
   for `-1`, the first unused ID, and both `i32` extrema.
4. The HostType state/ABI split in §C4.6 and §C6.3: Result-only syntax
   decoding, named polymorphic and inference states, `Never`, resolution to
   `ConcreteHostType`, and fallible target conversion to `HostAbiType`
   authorized by the target capability decision. Before Table B exists, the
   private exhaustive adapter cites a spec atom or implementation issue for
   every negative decision. Host codegen accepts only the resolved ABI
   vocabulary; host-expression lowering rejects unhandled forms through its
   own `Result`; generic ADT fields cross the boundary only after applied-type
   substitution; and the legacy `Unknown`, placeholder-`Unit`, and emitted
   default-value escape hatches are deleted.
5. The private structured C-expression AST described in §C4.4, replacing
   `EmittedExpr::raw` as a general construction path.
6. Keep the token/count tripwire green as supporting evidence. Its baseline
   does not freeze as the safety authority.

**Frozen at your exit:** the two vocabulary declarations, their external
spellings/IDs, the typed consumer inventories, and the generated-header
contract. From here, a new kind or dtype forces a compile-error work-list in
every semantic consumer.

**Explicitly not yours:** dtype finalization/storage/kernel semantics or the
Table A/B policy ([#729]); [#709]'s `DeepTag` enum ([#731]); observation
formatting ([#732]); capability atom authorship ([#733]).

**Authoritative oracle (owner: Phase 2):**

```sh
.venv/bin/python scripts/loud_unsupported_phase2_oracle.py
```

Success means exit 0 with the final line `PHASE 2 ORACLE: PASS`. This single
runner executes the closed-vocabulary suite, generated-header and invalid-ID
runtime checks, §C6.3 term-state and public-API parity (including generic ADT
specialization, direct/nested field access, callable-value rejection, and
empty-list failure propagation), exact inline and named callback and
reduced-float ABI cells, structured-emission and private-ABI compile-fail
checks, the endpoint scan for unresolved, placeholder-`Unit`, and raw generic
field escape hatches, and a controlled temporary vocabulary mutation that adds
one fully-decodable variant to each closed vocabulary (`EffectKind` and
`RuntimeDType`) in a single workspace check. The mutation must produce
non-exhaustive-match errors in each vocabulary's independent semantic
consumers and the runner must restore the owner source byte-for-byte.
Since chelis#732 Phase 2 the runner carries a third controlled-mutation
leg for the backend's crate-private `HostAbiType` (the section C6.3
typed-state boundary, which gained `ReducedFloatBoxed`): an added ABI
variant with no consumer arms must produce non-exhaustive-match errors at
the ABI owner's matches (`host_abi.rs`) and the emitter's
boxing/unboxing/print consumers (`host_emit.rs`), with the same
clean-owner refusal and byte-for-byte restore.
The tripwire remains supporting evidence executed by the normal gate; it is
not a second completion oracle.

## Phase 3 - gates become UX, and rejections become typed

**Amended 2026-07-30**: this phase absorbs §C2.1-C2.2 because gate
dedup and the typed constructors are the same code motion - the two
`reject_*` families cannot be deduplicated without deciding what the
shared constructors return, and that decision IS the typed kind
channel. Not gated on [#729] Phase 4: deliverable 1 below stands on
"let the emitter's channel speak" whether or not the capability table
has landed (the roadmap's Wave 4 placement is advice; where the two
disagreed, this plan wins and the roadmap was corrected 2026-07-30).

**You inherit:** emitters that cannot silently substitute (so gates no
longer carry correctness), the census rows 17-18 and 26, and two live
demonstrations that the unpoliced shape is actively widening (PR
[#891]'s 1 -> 21 eval-only gate names; PR [#822]'s `"compile_error"`
mislabel).

**Delivery status (2026-08-05): the gate, kind, and authority work is
implemented, but Phase 3 is not complete.** The CLI consumes compiler-api's
typed host-builtin, eval-only, effect, HIP, Metal, and windowed-reduction gate
definitions; its obsolete C precision pair is deleted; all shared gate target
arguments use the closed `BuildTarget` enum; direct-load f16/bf16
`BlasMatmul` remains admitted while actual narrow-float operand compute is
rejected; and the exact syntactic `reject_*` inventories (path, name, and
visibility across every Rust file, nested free function, and `impl` method in
both crate source trees) are locked by a parsed-source tripwire. The tripwire
does not claim semantic uniqueness for unrelated function names. The only
remaining Phase 3 deliverable is the independently owned [#912]
root-realizability integration.

**You deliver:**

1. **Gate inventory resolution**: the duplicated `reject_*` gates ([#698]'s
   drifted pair, [#705]'s missing CLI twin) are deduplicated to single
   typed definitions consumed by both the CLI and compiler-api paths -
   the `Unsupported::compiled_host_only_builtin` shared-constructor
   precedent, extended across the inventory in census row 26, including
   the `process_run` / eval-only family; `HOST_ONLY_BUILTINS` and kin
   either derive from the [#729] Phase 4 capability table (if landed)
   or are replaced by "let the emitter's channel speak, gate only to
   move the diagnostic earlier and add span context".
2. **The gate contract**, recorded in this doc: a gate may only ever make
   a diagnostic EARLIER or MORE SPECIFIC; it may never be the sole
   defense, and a gate/emitter disagreement is a bug in the gate. The
   enforcement ladder: compile-error > lint > tripwire
   > gate > prose - a gate that carries correctness is on
   the wrong rung.
3. **The closed kind vocabulary** (§C2.2): `DiagnosticKind` in
   `chelis-vocab`, the `stage_error` signature change, the
   `unsupported_feature`-only-via-`unsupported_stage_error` rule, the
   producer/wire type split (`Serialize`-only producer with the
   private `kind`; a complete `Deserialize`-only consumer envelope/check/batch
   graph that no envelope-assembly API accepts), the wire-spelling lock test,
   the required change-path mutation job, and
   removal of the `format!("{:?}")` and message-substring kind paths.
4. **The typed rejection authority** (§C2.1): the 33-site `hint ->
   RejectionAuthority` migration, with the [#687] rejected-cells corpus
   updated in the same PR per B1 (each site is adjudicated against the
   deciding spec atom or actual implementation owner; the two uncited
   `reduce_window` hints cite [#729] and [#600], not [#959]) -
   plus the two validation registries the constructors consume: the
   derived atom registry (generated from the numbered specs,
   byte-agreement-tested) and the checked-in issue manifest (whose
   liveness re-verification is §C7.5 scheduled-job work).
5. Deletion of gates that now only duplicate emitter rejections, with the
   cross-lane rejected-cells corpus proving the diagnostic surface
   unchanged or improved (earlier stage, same `unsupported:` content).
6. **Root-realizability integration ([#912])**: a `HostReason` or equivalent
   manifest reason supplies context, not authority. When an owed root cannot
   be produced, the boundary constructs the same typed `DiagnosticKind` and
   `RejectionAuthority` required everywhere else, naming root, lane, and
   reason per [05-OBS-6]/[05-UNS-1]. Before Tables A/B land, the authority is
   the validated pre-table atom/issue source; afterward it derives from the
   deciding table cell. No free-form reason string, silent missing root, or
   manifest-only error vocabulary becomes a parallel failure channel.

**Frozen at your exit:** the gate inventory and contract; the
`DiagnosticKind` vocabulary and wire spellings; the
`RejectionAuthority` shape.

**Explicitly not yours:** the capability table itself ([#729] Phase 4);
span threading and the check-JSON serialization mechanics
([#883]/[#886], §I2); lifting the §C2 STATUS relaxation (nothing gains
`Serialize`); deciding which roots exist, their order, or whether an artifact
is owed ([#912]).

**Oracle:** `.venv/bin/python scripts/loud_unsupported_phase3_oracle.py` is
the single authoritative Phase 3 command. Success is exit 0 with final line
`LOUD UNSUPPORTED PHASE 3 ORACLE: PASS`. It runs the rejected-cells corpus
and requires it to remain stable (or improved-with-updated-
expectations in the same PR) - substring-level today by design, with
byte-exact per-lane assertions arriving via the [#732] Phase 3
handshake, a named cross-plan residual this phase does not wait for;
[#697]/[#698]'s tests green and un-ignored (note: [#698]'s
census-named test was renamed after the census froze - the Phase 4
census binding re-links it; use the current suite name); zero gates
existing in two copies (a new tripwire assertion, mirroring the
conformance MANIFEST pattern); the wire-spelling lock test green; and
the full §C2.1-C2.2 negative-control set, one control per bypass or
admission route: a planted free-literal kind string, a direct
`DiagnosticKind::UnsupportedFeature` argument to a general envelope
API, an out-of-crate `Diagnostic` literal, and a serde-deserialized
`WireDiagnostic` handed to an envelope-assembly API each failing to
compile, with the producer type's `Deserialize` absence
expiry-locked; an in-crate `Diagnostic` literal outside the sealed
module and a post-construction `kind` mutation proven unwritable by
planted-mutation controls (planted bypass in a non-chokepoint
compiler-api module fails the workspace check, byte-restored after);
and the authority admission set - empty atom, malformed atom,
well-shaped nonexistent atom, issue zero, closed issue, PR-number
citation, nonexistent nonzero issue, and the same-PR manifest
addition of any such number - each rejected at construction,
validation, or the change-gated live check. The supported-neighbor
control is an open ISSUE manifest row (chelis#879), which passes those
structural checks; deliberately attaching that unrelated issue to a
rejection is the paired review-negative control and MUST be rejected
in review. The machine check does not pretend to prove that semantic
relationship. Its final leg runs `issue_912_root_boundary` including ignored
tests; that leg is intentionally red as of 2026-08-05, so the runner cannot
print PASS and Phase 3 cannot be called complete until [#912] lands.

## Phase 4 - ratchet totality (added 2026-07-30)

Independently landable in any interleaving with Phase 3 (one ordering
note: deliverable 2's census binding lands before either phase edits
further §C5 rows, so row edits are machine-checked from the start).

**You inherit:** the frozen Phase 2 vocabularies and oracle, the
tripwire, §C7, and census rows 24-25 and 27.

**You deliver:**

1. **The derived-universe tripwire** (§C7.1): the product-source
   manifest (Cargo members joined with the registered non-Cargo roots,
   under the top-level totality guard), recursive per-root walk,
   required Python/C language adapters over vocabulary-derived boundary
   names and raw ABI AST forms,
   per-class `TestsPolicy`, the new and widened token classes
   (line-conjunction `unwrap_or` x `Prim::`, `.unwrap(`/`.expect(`,
   early-return defaults, the panic + assertion classes), annotated
   exclusions, and a regenerated baseline in which every production
   hit carries an issue or a conversion. This produces the honest
   backlog and is the first PR.
2. **The census binding** (§C7.4): the `include_str!` parse of §C5,
   binding every row to a selected backing (nextest-selected or
   row-declared ignored per the ignore-ledger rule), an issue link,
   and - for live-by-execution rows - an existing probe-corpus record.
3. **Consumer discovery** (§C7.3): `closed_vocabulary_architecture.rs`
   inverted from a fixed file list to the dual-source derivation
   (Cargo dependency graph + Rust and non-Cargo language adapters),
   with the mutation oracle's
   evidence set staying on the ratified §C6 inventory.
4. **The structural panic work** (§C7.2): adapter privatization with
   `compile_fail` locks; `compiler_invariant!` including its
   structured payload frame (the `compiler invariant violated:`
   marker + guarantee + guard, which only the sanctioned macro
   produces and the surfacing control keys on); then the [#957] triage
   sweep in crate-scoped PRs (metal -> hip -> backend-c -> ir), each
   crate's `clippy::panic`/`clippy::unreachable` deny flipping as its
   triage completes, `clippy::todo`/`clippy::unimplemented` denied
   workspace-wide up front, lint adoption checked against the derived
   member list.
5. **Census row 25's conversion** ([#958], the einsum default), the
   row shrinking with the fix per B1.
6. **Both §C7.5 execution halves**: the change-gated blocking `ci.yml`
   job for the mutation legs and the shared manifest/exclusion live
   validation (path
   filter per §C7.5: the vocab owners, `host_abi.rs`, the ratified
   inventory, the canonical exclusion registry, the issue manifest,
   the root `Cargo.toml`, the non-Cargo registry, the oracle scripts,
   the workflows; classified in `scripts/test_gate.py`), and
   `loud-unsupported-nightly.yml` + the `NON_GATE_WORKFLOWS` entry +
   the open/close report job, running both plans' Phase 2 oracle
   runners and this phase's oracle as the full-matrix drift canary.

**Frozen at your exit:** the §C7 universe derivations and the
canonical exclusion registry (shrink-only thereafter); the workspace
lints table; the nightly authority job.

**Explicitly not yours:** making any rejected cell supported ([#729]);
`chelis-python`'s FFI/representation redesign ([#893] - this phase only
brings its dtype boundary into scan scope and requires §C6.2
decode-on-entry); [#732]'s oracle contents (this phase schedules it;
they own it); checker-side panics ([#731]'s organ).

**Authoritative oracle (owner: Phase 4):**

```sh
.venv/bin/python scripts/loud_unsupported_phase4_oracle.py
```

Success means exit 0 with the final line `PHASE 4 ORACLE: PASS`. The
runner executes: the derivation checks (the tripwire's scope set is
provably computed from the product-source manifest - Cargo members
plus the non-Cargo-root registry under its top-level totality guard -
not from a literal list); the lints-adoption check over the derived
members; the §C7.4 census binding against the nextest-selected set
with the ignore-ledger equality and the live-row probe-record
existence check; the §C7.3 dual-source discovery scan; the
`compiler_invariant!` guard-referent tier (every guard resolves in the
selected test set or the atom registry); the execution-wiring
assertions (the change-gated `ci.yml` job exists, is classified in
`scripts/test_gate.py`, and its path filter covers the ratified
inventory file, the canonical exclusion registry, the §C2.1 issue
manifest, the root `Cargo.toml`, and the non-Cargo registry -
the §C7.5 trigger-loop closure, asserted against the workflow's
actual path list; the nightly file exists, names both oracle runners,
and appears in `NON_GATE_WORKFLOWS`; the mutation legs, changed-row
manifest/exclusion live validation, and scheduled full standing
manifest/exclusion validation are executed by those jobs, not by this
assertion); and negative controls
in the Phase 2 oracle's planted style - a planted production
`panic!`, a planted `.unwrap()`, and a planted bare `assert!` in a
derived-universe member each go red; a planted census row naming a
nonexistent test, an ignored-undeclared backing, a phantom
probe-record citation, and an unregistered non-Cargo product root
each go red; a planted new consumer (vocab dependency edge), a
planted uninventoried Rust consumer file, a planted Python file that
imports `chelis._native` and carries an
`np.asarray(x, dtype=np.float64)` coercion feeding a `_native` call
(the [#900] shape, the adapter's REQUIRED positive case), and a
planted C ABI struct carrying `int dtype` plus `CHELIS_F32` inside
ALREADY REGISTERED non-Cargo roots each fail discovery, while a
numpy-using Python file with no `chelis` import is the
supported-neighbor control and does not become a candidate. A planted
`Deferred` exclusion
for a vocab-edge candidate fails until its `IssueRef` names a
live open issue: chelis#944 (closed issue), chelis#1 (PR),
chelis#999999999 (missing), and same-PR manifest/exclusion additions
for each are negative controls, while chelis#879 is the structurally
valid open-issue control; a planted `Structural` exclusion over a
discovered candidate is red regardless of its reason. A same-PR
exclusion citing chelis#879 but
unrelated to the excluded candidate is the paired review-negative
control: structural validation passes and review MUST reject it.
The §C7.2 surfacing control applies a temporary mutation at a REAL
lowering/emission site behind its `Result` signature - planting the
sanctioned `compiler_invariant!`, not a bare panic - builds an
isolated non-product CLI from that dirty oracle checkout, and drives
the site through a named `.ch` probe, asserting nonzero exit, the
invariant FRAME (marker + guard token, which no ordinary diagnostic
produces), and no artifact. The runner refuses a
pre-dirty owner and restores every planted file byte-for-byte before
success. Architecture negatives plant the NAMED `internal-panic-seam`
Cargo feature in a product crate and require the named-artifact guard
to go red; the shipped-neighbor positive builds the stageable release
subset (`cargo build --release -p chelis-cli --features smt` +
`scripts/verify_release_smt.py`) from the restored clean tree and
proves no selector, sentinel, or frame marker is present in the
artifact or its feature graph. Its
self-tests ride the
existing per-PR `unittest discover` CI step automatically, which
keeps the oracle's anchors continuously verified between scheduled
runs.

## C8. Class-maintenance work items

These items extend the class after the numbered phases established its typed
failure channel and construction rules. They are not a fifth phase, do not
reopen a freeze point, and do not permit a local patch. Each item names the
smallest reusable mechanism whose oracle makes the reported instance and its
same-region siblings unrepresentable.

### LU1 - one supervised external-process degradation path ([#870])

All Beacon child-process waiting, timeout notification, signal/self-pipe
handling, and teardown route through one private supervisor that returns a
closed `Completed | TimedOut | Degraded(Unsupported)` outcome. A denied OS
mechanism, including the reported `sendto` denial, becomes `Degraded` and is
rendered through §C2; no signal handler, destructor, or callback may panic or
abort. Direct `wait-timeout` use outside the supervisor is a structural test
failure. The oracle runs both current prover launch paths with each supervisor
mechanism denied in turn, asserts a branded nonzero result and no abort, and
includes a normally completed child plus an actual timeout as positive
controls. Catching `sendto` at the reported site does not satisfy this item.

### LU2 - total proof-type traversal ([#872])

Projection and containment over proof types share one iterative, cycle-aware
worklist keyed by the traversed node identity. A genuine cycle terminates by
visited-set convergence; an implementation resource limit returns a branded
exhaustion error and never a semantic type or boolean. This deletes both
same-region depth fuses: `type_from_deep_depth`'s depth-32 `Type::Unit` and
`type_contains_depth`'s depth-16 `false`. The oracle drives acyclic types
through and beyond both former limits, a cyclic type, and a planted low
resource budget; it asserts the complete obligation set or the explicit
exhaustion error, never a smaller successful proof problem.

### LU3 - canonical iterative compiler list spines ([#906])

Compiler-authored traversal of canonical `Cons`/`Nil` data uses one iterative
spine iterator/folder shared by literal baking and every equivalent evaluator,
lowering, and observation walk. It reports an improper tail through the typed
channel instead of recursing or fabricating `Nil`. The oracle exercises flat
lists below, at, and well beyond the reported 2k-5k range, plus an improper
tail, and forbids a second recursive canonical-spine walker by structural
inventory. This item does not own user-authored recursive programs ([#257]) or
general lowering/cost-model recursion ([#409]); those are different recursion
classes with different oracles.

### LU4 - compositional host-ABI capability ([#955])

Host lowering resolves a checked host-type term by recursively composing the
closed constructor dispositions in `capability_table.md`; it does not match a
hand-picked inventory of concrete nestings. If `List<T>`, `Option<T>`, and the
inner `T` are represented for a backend, `List<Option<T>>` and
`Option<List<T>>` are represented by construction. If any constructor cell is
unimplemented or rejected, the composed result carries that exact typed
authority. The oracle covers every constructor pair in both nesting orders,
positive and negative, and a newly added constructor makes the disposition
match non-exhaustive. A special arm for the reported `Option`/`List` shape is
not a fix.

### LU5 - derived compiled-stdlib acceptance corpus ([#955])

The build-lane stdlib corpus is derived from the exported stdlib manifest,
not maintained as a sample list. Every exported module contributes at least
one minimal importing program that reaches build/link/run when its Table-B
cells are implemented, or asserts its exact typed capability rejection when
they are not. The derivation fails if an exported module has neither result.
The oracle mutates the manifest with a synthetic export and proves that the
missing compiled cell is red, then runs the standing corpus. Eval-only stdlib
self-tests and one hand-picked `Std.Io` build are supporting evidence, not
this acceptance surface.

### LU6 - exhaustive checked host-cast planning ([#1150])

Host `cast` emission consumes one exhaustive plan over
`(source Prim, target Prim)`. Only exact same-type pairs select identity;
every other pair selects the checked conversion defined by [04-NUM-14] and
the dtype plan or returns a cited `Unsupported`. The emitted code cannot copy
the source expression through a wildcard. The oracle is generated from the
full source x target x scalar/tensor x eval/c-host parameter product in
`capability_table.md`, with in-range and trapping parity for each applicable
cell and a mutation that replaces one non-identity plan with identity. Trap
selection for multi-offender tensors belongs to [#729]'s indexed-trap item,
not a second mechanism here.

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
- **Identity is not semantics.** `RuntimeDType` supplies stable ABI identity,
  spelling, and the mapping to `Repr`. `Repr` describes the current physical
  encoding and supplies its byte width. [#729] supplies the storage decision,
  finalization, operation semantics, and kernel behavior. Neither layer may
  duplicate the other.
- **Existing exact integer ABIs are not reclassified as unsupported.** C
  already has `int8_t` and `int16_t`, and the checked in-range controls use
  them without type erasure. Phase 2 therefore selects those exact
  `HostAbiType` variants. [#729] still owns overflow traps and per-operation
  semantics. Reduced-float scalars are different: no exact host storage and
  rounding path exists yet, so their former widened-double green cases are
  rejected rather than preserved as proof artifacts.
- **Resolution is not capability policy.** `HostTypeTerm -> ConcreteHostType`
  preserves checked logical identity without consulting a backend.
  `ConcreteHostType -> HostAbiType` consumes the target decision. The private
  pre-table adapter is replaced by Table B without changing the boundary.
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
- **`RejectionAuthority` (§C2.1) types the citation channel; the table
  owns the citations.** Which cells are `Deliberate` versus
  `Unimplemented` is Table A/B's decision; when Table B lands, its
  cells populate the enum through the same boundary the pre-table
  adapter uses today. This plan never authors a capability decision.
- **Finite parameter products prevent dtype-valued omissions.** The
  capability schema expands `cast` over source dtype x target dtype x surface
  x backend. LU6 consumes that closed product to plan host emission; [#729]
  owns the conversion and trap semantics. Adding one source or target dtype
  makes both the plan and conformance derivation incomplete at build time.
- **Host containers compose from constructor dispositions.** LU4 consumes the
  companion host-ABI constructor table recursively. This plan owns the typed
  rejection path and forbids default host representations; [#729]'s table
  owns each backend disposition. A concrete nesting never becomes a new
  policy site.
- **[#1152] is numeric semantics with a shared failure value, not a second
  unsupported mechanism.** [#729] owns `IndexedTrapCandidate` construction
  and lowest-index selection in every lane. This plan owns only the §C2
  rendering of a selected trap or capability rejection.
- **[#912] consumes this failure channel; it does not fork it.** Root identity,
  manifest order, and artifact routing remain [#912]'s decisions. A root-level
  `HostReason` is diagnostic context. The deciding atom/issue enters through
  `RejectionAuthority`, and `DiagnosticKind` supplies the stable machine
  spelling; after Tables A/B land, their cell supplies that authority.

## I2. Interlocks added 2026-07-30

Same discipline as §I1; edits are bidirectional per B2.6.

- **[#883]/[#886] (the diagnostic surface).** This plan owns the closed
  `DiagnosticKind` vocabulary, the frozen wire spellings, and the rule
  that an `unsupported_feature` envelope is constructible only from a
  typed `Unsupported`. [#883]/[#886] own span threading and the
  check-JSON serialization mechanics that carry the payload. A new kind
  changes this doc; a new wire field or serializer changes theirs.
- **[#732] (faithful observation).** B2.7 is stated here and
  generalizes: the §C7.5 nightly workflow executes both plans' Phase 2
  oracle runners. `faithful_observation.md` carries the reciprocal
  continuity note; its oracle's contents, stages, and pass criteria
  remain [#732]'s alone.
- **[#893] (runtime representation).** `chelis-python`'s dtype/FFI
  representation - the `dtype: i32` struct fields, the DLPack triple,
  the tensor payload shape - is the runtime-representation family and
  is [#893]'s to redesign (corrected 2026-07-30: an earlier draft
  pointed this at [#909], whose scope is the callable ABI, not a
  generic ABI bucket). This plan requires only that the dtype boundary
  decodes through `chelis-vocab` on entry and that its rejections
  carry the §C2 brand; census row 27 records the current state
  ([#960]) and points at [#893] for the fix.
- **[#909] (host function values).** Owns the callable ABI only. Its
  contact with this plan is the callable-boundary substitution
  instances (the null-function-pointer emission shape recorded in the
  2026-07-30 review) and the §C6.3 callback-declarator boundary, both
  unchanged by this amendment.

## Issue map

| phase | goes green / becomes unwritable |
|---|---|
| 0 | census verified; tripwire live; regressions detectable |
| 1 | [#699] (+[#722]'s eval half via the raise), [#682], [#704], [#705], [#715] (stub half), [#689], [#692], [#725], [#734], the [#709]-adjacent effect catch-all; [#714]/[#718] downgraded from silent-wrong to cleanly-rejected |
| 2 | the future supply of the class (bottom vocabularies + Result decoding + exhaustive consumers + structured emission) |
| 3 | [#697], [#698], [#705]'s gate half, [#959] (kind/brand skew); gate rot and kind mislabeling as classes |
| 4 | [#957] (the panic family, incl. [#919]'s pair), [#958], [#960]'s decode-on-entry half; ratchet-coverage drift as a class - a new product root, vocabulary consumer, or enumerable panic/default spelling outside the classified universe is a red test, and the non-enumerable panic routes are named §C7.2 limits owned by the behavior instruments, not silently out of scope |
| maintenance | [#795] completes the optional-static-argument migration; LU1 [#870], LU2 [#872], LU3 [#906], LU4/LU5 [#955], and LU6 [#1150] extend the class mechanisms without reopening a phase; [#1152] is transferred to [#729]'s checked-cast work item with this plan retaining diagnostic provenance only |

## Settled ownership and remaining phase decisions

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | shared error/vocabulary ownership | `Unsupported` remains in `chelis-types`; closed cross-layer identity - including `DiagnosticKind` (2026-07-30) - lives in dependency-free `chelis-vocab`; per-crate mirrors are forbidden | §C2, §C2.2, §C4.1, §C6 |
| 2 | whether `check` reports target-independent unsupported constructs | yes: Table A rejections are type-level facts; Table B rejections surface at build where the target is known | `capability_table.md` §Derivations |
| 3 | source-inventory mechanics | the token/count baseline remains blocking only while typed boundaries are incomplete and is never a completion oracle | tripwire test + §C4.5 |
| 4 | which gates survive Phase 3 as early-UX vs die | Phase 3 | §C5 rows 17-18 + gate contract |
| 5 | first-error-only vs accumulating `Unsupported` reporting (the checker accumulates ~30 independent errors per pass; the §C3 channel short-circuits at one - the asymmetry named in the 2026-07-24 post-merge review, item 5) | Phase 3, before the gate contract freezes; §C3 specified first-error deliberately, so this is affirm-or-re-open, not drift | §C3 + this row |

## Contract summary

Give every stage a way to say no (Result-typed emission), convert the ten
live yes-anyway sites to §C2 diagnostics, then take the pen away: decode raw
symbols and dtype IDs once into dependency-bottom closed types, exhaust those
types at every semantic consumer, build C expressions structurally, and keep
gates/lexical scans as UX and evidence rather than the thing standing between
an unsupported case and a plausible wrong number. Then take the pen away from
the ratchets themselves (§C7): derive every guard's universe from the
product-source manifest instead of enumerating it, seal the kind behind a
private producer field and the citation behind registry-validated
constructors so a mislabeled or uncited rejection cannot be built and a
closed/PR/missing issue identity cannot masquerade as live. Citation
RELEVANCE remains explicitly review-owned; the registries do not claim to
prove it. Sanction exactly one invariant terminal whose metadata is
shape-checked at compile time and referent-checked by the oracle, and run
discovery unconditionally on every PR with the mutation authority
change-gated pre-merge and nightly otherwise, so the proof never goes stale
and never depends on predicting a path.

[#257]: https://github.com/Chelis-Lang/chelis/issues/257
[#387]: https://github.com/Chelis-Lang/chelis/issues/387
[#409]: https://github.com/Chelis-Lang/chelis/issues/409
[#680]: https://github.com/Chelis-Lang/chelis/issues/680
[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#687]: https://github.com/Chelis-Lang/chelis/issues/687
[#689]: https://github.com/Chelis-Lang/chelis/issues/689
[#691]: https://github.com/Chelis-Lang/chelis/issues/691
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
[#795]: https://github.com/Chelis-Lang/chelis/issues/795
[#799]: https://github.com/Chelis-Lang/chelis/pull/799
[#815]: https://github.com/Chelis-Lang/chelis/pull/815
[#822]: https://github.com/Chelis-Lang/chelis/pull/822
[#870]: https://github.com/Chelis-Lang/chelis/issues/870
[#871]: https://github.com/Chelis-Lang/chelis/pull/871
[#872]: https://github.com/Chelis-Lang/chelis/issues/872
[#875]: https://github.com/Chelis-Lang/chelis/issues/875
[#883]: https://github.com/Chelis-Lang/chelis/issues/883
[#886]: https://github.com/Chelis-Lang/chelis/issues/886
[#891]: https://github.com/Chelis-Lang/chelis/pull/891
[#893]: https://github.com/Chelis-Lang/chelis/issues/893
[#900]: https://github.com/Chelis-Lang/chelis/issues/900
[#906]: https://github.com/Chelis-Lang/chelis/issues/906
[#909]: https://github.com/Chelis-Lang/chelis/issues/909
[#919]: https://github.com/Chelis-Lang/chelis/issues/919
[#932]: https://github.com/Chelis-Lang/chelis/pull/932
[#935]: https://github.com/Chelis-Lang/chelis/issues/935
[#951]: https://github.com/Chelis-Lang/chelis/issues/951
[#955]: https://github.com/Chelis-Lang/chelis/issues/955
[#957]: https://github.com/Chelis-Lang/chelis/issues/957
[#958]: https://github.com/Chelis-Lang/chelis/issues/958
[#959]: https://github.com/Chelis-Lang/chelis/issues/959
[#1037]: https://github.com/Chelis-Lang/chelis/pull/1037
[#1058]: https://github.com/Chelis-Lang/chelis/issues/1058
[#1059]: https://github.com/Chelis-Lang/chelis/issues/1059
[#1144]: https://github.com/Chelis-Lang/chelis/pull/1144
[#1150]: https://github.com/Chelis-Lang/chelis/issues/1150
[#1152]: https://github.com/Chelis-Lang/chelis/issues/1152
[#1192]: https://github.com/Chelis-Lang/chelis/issues/1192
[#960]: https://github.com/Chelis-Lang/chelis/issues/960
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
