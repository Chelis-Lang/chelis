# Numeric Remediation Roadmap: sequencing, ownership, and the ledgers

**Status:** Living coordination document for the plan set that came out of
the 2026-07 numeric audit. This doc owns three things nothing else owns:
the **global sequencing** across the five plans, the **unclaimed-issue
ledger** (every filed issue that no plan's kill table claims, with its
assigned home), and the **deferred-evidence ledger** (claims still resting
on inspection, each with its verification task). It contains NO contracts
of its own - contracts live in the five plans; when this doc and a plan
disagree, the plan wins and this doc has a bug.

The plan set: `dtype_semantics.md` (#729), `loud_unsupported.md` (#730),
`checker_totality.md` (#731), `faithful_observation.md` (#732),
`spec_provenance.md` (#733), plus the `capability_table.md` schema (rides
#729 Phase 4) and the audit record under `docs/investigations/`.

## The class map

| meta (the class) | method (design spec) | tracker |
|---|---|---|
| #727 no dtype's semantics enforced at any single point (#695 = its integer instance) | `dtype_semantics.md` - per-dtype finalizer behind private constructors, int/float kernel split, one storage decision, generated backend dispatch | #729 |
| #703 unsupported cases substitute values instead of failing | `loud_unsupported.md` - Result-typed failure channel, the 18-row census sweep, un-writability ratchets (lint, newtype, tripwire), gates demoted to UX | #730 |
| #709 unrecognized constructs silently exempt from checking (+#710's silent half) | `checker_totality.md` - loud wildcard + handle-effect case, ErrorWitness token (silent Type::Error unconstructible), totality invariant, DeepTag exhaustiveness | #731 |
| #728 the observation channel is not dtype-faithful | `faithful_observation.md` - one Rust formatter, generated C print helper, round-trip invariant, tolerance table; landable before #729; unblocks #687 | #732 |
| spec silence + stale claims (#694; the unauthored cells) | `spec_provenance.md` - hash-addressed spec atoms, lint-checked @spec claims, test-carrier coverage, the PR authority gate | #733 |

Supporting: `capability_table.md` (schema; rides #729 Phase 4),
`docs/agent_quality_architecture.md` (#740), the seeded atoms
(spec/04 §9-§10, spec/05 §7-§8), and PR #696 (the acceptance surface).

## Global sequencing

The plans are deliberately independently landable - every pairwise
interlock is pinned in both landing orders (each doc's §I1). The
*recommended* order optimizes for detectors-before-fixes and for
small-wins-early:

**Wave 0 - all five Phase 0s, in any order, immediately.** Each is
afternoon-scale, none touches production code, and together they make
every class regression-visible before anyone fixes anything: the
domain-validity invariant + #687 oracle lanes (#729 P0), the substitution
census verification + token tripwire (#730 P0 - note census row 3 is
already settled: live, chelis#734), the `Type::Error` census + red
totality invariant (#731 P0), the round-trip harness + exit census
(#732 P0), the PR spec gate + `spec/**` signoff (#733 P0).

**Wave 1 - the small loud fixes.** #731 Phase 1 (the checker holes -
days) and #730 Phase 1 (the failure channel + live-site sweep). These
convert every silent-wrong-answer into either a correct answer or a clean
rejection, which shrinks the danger surface before the big refactor and
makes the remaining reds honest.

**Wave 2 - the independently-landable value work.** #732 Phases 1-2 (the
formatter + generated C side: fixes #716/#723 outright and gives the
refactor its byte-exact instrument) in parallel with #731 Phases 2-3 (the
witness token + DeepTag) and #730 Phase 2 (the lint ratchets). #733
Phase 1 rides alongside, atomizing whatever spec text Waves 1-2 author.

**Wave 3 - the semantics refactor.** #729 Phases 1-3 in order (the
module + storage decision; the kernel split + prove; backend adoption),
validated by everything Waves 0-2 built.

**Wave 4 - the permanent guards.** #729 Phase 4 delivers the capability
table per `capability_table.md`; #730 Phase 3 (gates become UX) and #733
Phase 3 (citations) ship inside it; #732 Phase 3 (tolerance table +
the #687 handshake) closes the oracle; #733 Phase 2's coverage ratchet
turns on for the atomized specs.

Standing exception: any Wave may be entered early for a cell that becomes
urgent - the interlock sections make that safe; this ordering is advice,
not law.

## The unclaimed-issue ledger

Filed issues no plan's kill table claimed, each now with an owner. Rule:
an entry leaves this ledger only by appearing in a plan's issue map, in
the capability table's seed decisions, or by being closed.

| issue | what | assigned home |
|---|---|---|
| #681 | Std.Decimal negative `result_scale` leaks an unbranded error | standalone small fix; diagnostic text should conform to #730 §C2 when touched. No plan dependency. |
| #683 | `i64::MIN` not writable as a literal | standalone front-end fix; natural moment is #729 Phase 2 (the exact int lane makes the round-trip testable), but nothing blocks doing it sooner |
| #689 | HIP int64 ops emit F32 kernels | two-part: the SILENT half dies at #730 Phase 1 (`elem_kind` raises); the SUPPORT half is owned by capability-table B-cells `Unimplemented { issue: #689 }` until int64 kernel templates land (see `capability_table.md` seed decisions) |
| #690 | HIP has no integer div-by-zero guard | rides the same HIP B-cell work as #689; the guard is part of `Implemented` for HIP int division cells |
| #691 | C DAG lane emits fmaxf/fabsf for int64 | owned by capability-table seed decision: B-cells `Unimplemented { issue: #691 }` - the substitution becomes a rejection at #730 Phase 1 / table landing, correct kernels later |
| #693 | Metal int64 `abs` zero emission | root cause is #699 (confirmed by emission); Metal B-cell `Unimplemented { issue: #693 }` until the MSL integer path is wired post-#699-fix |
| #705 | `reject_host_only_builtins` has no CLI twin | #730 Phase 3 (gate dedupe) - was in its map; listed here because only the gate half was claimed. The builtin-coverage half dies with #730 Phase 1's typed emitter |
| #713 | `pad_sequences` allocates int32 output for int64 input | standalone lowering fix; natural moment is #729 Phase 3 (C host dtype parity), tracked here until claimed there |
| #719 | vvsqrtf not correctly rounded; layout-dependent results | THE FIX (sqrtf or intrinsic on the contiguous path, one implementation per op) is standalone and must land BEFORE #732 Phase 3 writes `sqrt = 0` into the tolerance table |
| #721 | eval cannot ingest the canonical Deep of a nullary fn | standalone eval-ingestion fix; explicitly non-goaled by #731; no plan dependency |
| #734 | `to_string` on tensors/lists compiles to the literal `<value>` | #730 census row 3 (now live); dies at #730 Phase 1, unwritable after Phase 2; rendering via #732's formatter |

Also tracked to closure but already claimed (listed for completeness):
#680/#684/#685/#686/#688 -> #729; #682/#692/#697/#698/#699/#704/#725 ->
#730; #709/#710 -> #731; #716/#723 -> #732; #694 -> #733; #711/#720 ->
#729 Phase 2 / #720's own note; #712/#715/#724/#726 -> capability table
seed decisions; #722 -> #730 Phase 1 (loud) then #729/table (computed).

## The deferred-evidence ledger

Claims resting on inspection or partial execution, per the audit's own
standard ("execute everything"). Each has a tracked task.

| item | current evidence | task |
|---|---|---|
| HIP runtime behavior (#689/#690 symptoms at runtime) | emission-proven only; the audit machine (arm64 macOS) has no hipcc | chelis#736: run the archived probes on the gfx1151 box via `scripts/hip_test.py` per `docs/local_hip_environment.md`; attach outputs; update both issues' evidence lines |
| Metal runtime execution (typed kernels actually computing) | emission-locked (`metal_dtype_emission_and_bool_add.rs`); never executed | chelis#737: a small driver harness (main.mm + chelis runtime link) on an arm64 Mac; promote the emission locks to run locks |
| #688's opaque produced-value chokepoint (`flatten_field_value`) | the CLASS is executed (spurious fuzz-tier counterexample); the cited site is not | tracked on #688 itself: needs `--features smt` + an `@opaque` int64-field type; exact repro sketch is in the issue comments |
| `with seed` / `with device` semantics (cross-lane seeding reproducibility) | never swept; both #730 and #731 explicitly non-goal it | chelis#735: spec-gap issue - author the semantics (atoms, per #733), then sweep both lanes |
| shell repos' compiled-lane numerics (school validates eval-only) | audit note, unexecuted downstream | chelis#738: conform-contract amendment proposal - shells gain a compiled-lane numerical row |
| census rows 13-16 of #730 (dead-by-probe placeholder sites) | probed dead or dead-by-inspection | re-verified mechanically at #730 Phase 0; §C1.4 raise-or-prove applies regardless |

(Filed: #735 effects, #736 HIP runtime, #737 Metal harness, #738 shell
lanes.)

## Effects: the one semantic area with no owner

Called out beyond the ledger because it is a spec-silence case (the
exact #733 shape) and not merely missing evidence: nothing anywhere
states what `with seed(n)` guarantees (determinism? cross-lane
reproducibility? scope of the seed?), what `with device` selects, or
what either means under `grad`/`vmap`. #731 will make their BODIES
type-checked and #730 will make unknown KINDS loud, but the meaning
stays unauthored - the same undecided-cell condition that produced
#724/#726, one construct over. Owned by chelis#735; its
output is atoms plus a sweep, not code.
