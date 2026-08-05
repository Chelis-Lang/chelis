# Chelis Numeric Remediation - Status (2026-08-04, end of day)

State basis: `main` after the merges of PR #1154 (capacity-disposition
ratification), PR #1149 (int64 dim carrier), PR #1164 (stale rejection-citation
rehoming, merge commit `5abb7ccd`), and PR #1163 (plan-doc status sync);
**v0.18.3 released** the same day. Sources: the five plan docs under
`spec/design/`, the class trackers and their sub-issue graphs, the 2026-08
sprint audit (`docs/investigations/sprint_audit_2026_08_908_912_729.md`), and
a five-agent execution-adjacent validation pass over every cited child issue,
followed by an executed cleanup (closures, parent links, doc corrections).

Post-basis update: the #729 follow-through after PR #1167 repairs #878,
#901, #937, #942, #980, #1009, and #1113 in the same change set that updates
this report. The defect entries below retain the exact audit evidence at the
recorded state basis; rows marked repaired are no longer current-state claims.
The live delivery ledger and recurrence guards are in
[`spec/design/remediation_roadmap.md`](../../spec/design/remediation_roadmap.md).

## TL;DR

Waves 0-3 of the remediation roadmap are complete, with one exception: #733's
Phase 0 (OpenSpec activation) - the lone outstanding Wave-0 item - has never
landed. Of the four Wave-4 slots, #732 Phase 3 is delivered, #729 Phase 4 is
opened (its census-permanence slice is merged; the capability table proper
remains), #730 Phase 3 is partially delivered with Phase 4 unbuilt, and #733
has nothing at all. The correctness *repairs* of Waves 1-3 held up under
validation: the issue graph was carrying ~29 already-fixed issues as open,
and those are now closed. What remains splits cleanly: a short list of
state-basis verified defects (minus the post-basis repairs above), two real
engineering campaigns (the capability table
and the #912 root-boundary chain), one unowned keystone (#1088), and an
institutional enforcement layer that is specified but largely not running.

## Scoreboard

| plan | landed | remaining |
|---|---|---|
| **#729 dtype semantics** | P0-P3 + §C6 census, all CI-wired (PRs #758, #956, #1033, #1049, #1054, #1065, #1118); Phase 4 opened by the merged PR #1154 (permanent dispositions for all 301 census rows, descriptor-manifest bound, red-teamed  x 2) | Capability table proper: Table A/B, derived checker acceptance, macro-generated dispatch (Supported cell without kernel = compile error), generated conformance suite, #912 projections. Plus: the P0 detector-half adjudication (unowned), the P3 oracle adjudication + stale tracker checkbox, and the #1160 seam-relocation decision |
| **#730 loud unsupported** | P0-P2; P3's typed `RejectionAuthority` + sealed 48-kind `DiagnosticKind` slices (PR #1037) | P3's gate half - verified undelivered despite earlier plan-doc claims (now corrected): dedupe the drifted `reject_unsupported_hip_ops` pair and the `COMPILED_HOST_ONLY_BUILTINS`/`HOST_ONLY_BUILTINS` duplication, record the gate contract, delete emitter-duplicating gates, brand the chelis-cli surface (zero `unimplemented_rejection!` uses there); deliverable 6 rides Jeff's #912 chain; the first-error-vs-accumulate decision must be affirmed before the contract freezes; Phase 4 (ratchet totality, #990) is entirely unbuilt - no phase-4 oracle script, no nightly workflow |
| **#731 checker totality** | P0-P3 implemented, incl. the construction gate (PR #1101) and authoring-scope fixes (PR #1136) | The decode-once rework's fresh-context red team (named pending in the doc header); the §C4.2 ingress half (#1088 -> open PR #1036); carrier deletion last (#1029, precondition currently false: ~1,082 `Expr::List` occurrences, production producers in 8+ crates) |
| **#732 faithful observation** | All four phases (P3 via PRs #1099/#1115/#1118); both known-red ledgers empty; tolerance table + #687 handshake shipped | Nothing in-plan. Open residue: #997 (21 derived-Debug tokens), #1059 (C-host to_string capability). Oracle facts: the Phase 2 script is manual-only (invoked by nothing); the Phase 3 oracle runs continuously via the Dtype Phase 0-3 Oracle job on every non-docs-only PR, nesting enforced by `test_nextest_profile_partition.py` |
| **#733 spec provenance** | Nothing - Phase 0 is defined but explicitly not activated; no oracle suite, no PR template, no CODEOWNERS | Everything. P0 is Chelis-owned and self-contained; P1-P3 additionally wait on three Buoy prerequisites (the pinned revision's `devenv test` oracle, the `buoy.adapter-sdk/v1` pin, the approved host parsed-item schema). v0.20 is formally blocked on P3 |

## Sibling classes (what the five plans structurally cannot deliver)

- **#908** (Deep AST successor; the `Atom`/list-head domain #731 can't reach;
  children #885, #887 Tier 2): the headline invariant is enforced (a bare
  `Name` at a RuntimeExpr slot is unrepresentable), but the class is not
  closed - the legacy carrier still has production producers, and the Aug 3
  audit explicitly retracted an earlier "functionally closed" claim.
  **#1088** - the compiler-api generic ingress (`parse_str_strict`) skipping
  the declaration-role stamp - is its keystone and is **unowned**.
- **#912** (root boundary; children #1079-#1084, #862, #947, #1102): Face 2
  (the `name = ` prefix) shipped in v0.18.1; Face 1 (existence) is partial  - 
  manifest walkers are `Expr::Node`-blind (#1082), anonymous tuple/ADT roots
  drop (#1083); Face 4 (artifact) is open - `requires_main` reaches no
  production caller and `c_source.contains("int main(")` still decides, which
  spec/05 says cannot satisfy [05-UNS-1]. #1023 §C is the live acceptance
  checklist; #730's deliverable 6 lands inside this chain.
- **#909** (typed host function values / C-host callable ABI; children #866,
  #867, #879): dormant, zero comments ever; PR #880 (the spec lock) sits
  open.
- **#883** (diagnostic spans; #868/#886 keep their #730 parent): decision
  made (fold with #880, `report_at` as the primitive) but not executed;
  `Unsupported::with_span` and `CheckError::with_span_id` still have zero
  call sites.
- **#893** (runtime representation: seal the `pub *mut u8` tensor data, make
  `Repr` the ABI primitive; children #899, #889; #892's bool storage rides
  v0.19): the last issue-bound capacity-census row; its delivery PR #894 was
  closed unmerged, superseded by #975/#964/#907. No August activity.
- **#740** (agent quality architecture; child #895): the entire owned backlog
  is unchecked - per-crate CLAUDE.md guardrails, mechanism index,
  change-shape skills, sibling check, contract-first docs-PRs, CI ratchet
  metrics, duplicates-registry tripwire. #895's worked instance: a plan's
  own inventory row was silently skipped by a phase PR, and the #893 cmplt
  defect (`-1 < 0` -> false through an f32-reinterpreted read) rode the gap.
- **#788** (conform audit surface; children #814, #825, #845): open parent by
  design ("a parent, not a plan"); draft PR #838 is in its family.

## Release axis

- **v0.17.x** - the first source-migration wave (loud checking, seed suffix,
  dtype-faithful eval rendering). Shipped.
- **v0.18.0/v0.18.1** - checker totality, host-type/ABI boundary, compiled
  rendering; then the [05-OBS-6] `name = ` prefix. Shipped.
- **v0.18.2** (2026-08-03) - additive (CSV/JSON serializers, eval JSON/CSV
  builtins, `--timeout`, Nix/Devenv, #729 Phase 2 typed integers, exact
  trapping integer `abs` in C), but it carried one recorded divergence: eval
  `shape()` returned int32 against [05-DIM-2]'s int64 SHALL, with an explicit
  do-not-migrate-onto note (#1120).
- **v0.18.3** (2026-08-04) - **a real source + exact-output migration cut,
  not a pin-only patch** (its release PR #1153 says so directly): [05-DIM-1/2]
  extent dtypes (`shape()` -> int64, int64 movement bounds; bare int32 extent
  sites become check errors; axis params stay int32 - this also retires
  0.18.2's #1120 divergence), compiled `round` now half-ties to even, and
  `cast_trunc` shipped as [05-OP-6] plus a reserved word. It is the migration
  target for the #1091 shell breakage.
- **v0.19** (planned; source migration) - remaining payload: the manifested
  root completion (#912/#1023: complete root set/order, unavailable-root
  diagnostics, artifact routing), the residual capability decisions
  (#712/#715's unsettled scalar contract, #753's wrap ops, #759's
  `cast_saturate` rung, the #689/#693 GPU cells, #965), the budgeted ABI
  removals (the pre-widening dim-carrier rows from #1149; #894's bool
  storage rename with #729 §C3's storage break; #892), and the GPU-lane
  extent widening (#1112's remaining half: `chelis_gpu_tensor` still carries
  32-bit extents).
- **v0.20** (planned; mechanical) - #729 P4 table + #730 P3 gates-as-UX +
  #733 P3 blocking provenance ratchet. Guaranteed behavior-preserving; #733's
  inactivity is its critical path.

## The issue graph after cleanup

### Closed 2026-08-04 (29, each with evidence comments on the issue)

- **#729 family (15):** #680, #684, #685, #686, #688, #711, #717, #720,
  #724, #726, #856, #860, #897, #1120 - verified fixed on main by the
  Phase 1-3 sealed storage, typed kernels, per-dtype wire/Python boundaries,
  and decided rejections (#1120 verified empirically with a live `chelis
  eval` run before closing) - plus **#691**, closed after PR #1164 rehomed
  its stale fused-chain citations (the direct DAG integer path was repaired
  in Phase 3).
- **#730 family (8):** #682, #692, #725, #734, #744, #745, #900, and
  **#714** (closed after #1164; `HostTypeTerm` has no `Unknown` state and
  every #714 matrix row runs un-ignored). #725 and #734 were consolidated
  into their open support owners #1058 and #1059 per the one-tracker rule.
- **#731 family (1):** #1132 (disposition A decided per [04-NUM-10], locked
  by PR #1123's test).
- **#732 family (5):** #716, #723, #748, #749, #775 - all fixed by Phase 2
  (PR #863) / Phase 3; the plan doc's own "close on the PR #863 merge" step
  had simply never been performed.

Deliberate holdouts: **#796** (spec/05 §3.6.1 now authors the eval-only
`test_*` contract the issue asked for as an alternative - needs a maintainer
call, not engineering) and **#862** (its fix is on main; it closes when the
named unit-valued-root regression test lands - the cheapest close on the
board).

### The CI-pinned set (now 10, all legitimately live)

`spec/design/loud_unsupported_issue_manifest.json` requires these OPEN while
emitter `unimplemented_rejection!` sites cite them - closing one reddens the
Rejection Authority Liveness job: **#600** (value-derived output dims),
**#689** (HIP typed kernels beyond f32/f64), **#729** (the tracker), **#759**
(the HIP cast half - `cast_trunc` landed on C, but HIP still emits an
unguarded device conversion and the issue demands every-lane parity),
**#829**, **#879** (C-host function-value ABI), **#951** (`emit_fused_reduce`
f32-hardcoded), **#1058**, **#1059**, **#1138**. Each closes by landing its
support, not by cleanup. (#691 and #714 left this set via PR #1164; their
one-time closure by #1151 and un-closure by the #1159 revert is the worked
example of why closing a pinned issue from the roadmap top-down breaks the
build.)

### Parent links (27 added and verified; one-tracker convention)

- -> **#730**: #935, #936, #939, #940, #941, #948, #1158 (the
  nullary-generic-ADT loud-rejection family), #1138, #1143.
- -> **#729**: #1110, #1112, #1113, #1116, #1120, #1140, #1160 (the extent
  cluster placed flat under #729; #1112 is not framed as a tracker - re-point
  #1120 under it if a sub-hub is preferred).
- -> **#731**: #1109, #1124, #1125, #1134. -> **#912**: #947, #1102.
- -> **#908**: #1029, #1107. -> **#732**: #1104. -> **#990**: #1089, #1090.

Pending maintainer decisions, best guesses recorded on the audit: **#916**
(-> #883?), **#1148** (-> #731? - the masking-test half is checker honesty),
**#1156** (no class; PR #1161 fixes it), **#1157** (-> #730's #957 panic
family, though it is a deliberate invariant assertion), **#1137** (no class),
**#985/#995** (closed dupes of #965 - leave unlinked), **#895** (kept at
#733 though its body opens "Part of #740"; an "Also part of #740" comment
would honor the convention). Structural note: #1023 and #1029 are
`tracking`-labelled hubs that themselves carry parent #908 - legal, but
worth confirming intended.

## State-basis verified defects (historical inventory)

Each entry is tagged with its owning tracker class and describes the repository
at the recorded state basis. All but the last two
bullets are parented sub-issues of a class this doc covers; the caveat is
that **parented != scheduled** - a parent gives ownership and the oracle that
proves the fix, but several of these appear in no remaining phase
deliverable of their owning plan (see gap 5 below).

- **#690** (owner: #729) - HIP integer division by zero: kernels emit bare `a/b`; exit-0
  garbage on GPU where eval traps branded. Fix is a kernel-template guard,
  validatable locally via `scripts/hip_test.py`.
- **#901** (owner: #729; **repaired post-basis**) - mixed int/float equality used
  signed absolute value and could panic on `i64::MIN`. The repair uses a shared,
  total exact-representability predicate keyed by the actual float dtype.
- **#937** (owner: #729; **repaired post-basis**) - `emit_uniform_like` writes f32 samples through `float *data`
  with no dtype dispatch: f64 `uniform_like` silently returns near-zero
  garbage in compiled C. The exact silent-wrong-value class the program
  exists to kill. The repair dispatches sampling and storage by float dtype and
  locks the f64 raw-bit result in the compiled C and HIP gates.
- **#942** (owner: #729; **repaired post-basis**) - cast rejected tensors whose
  element type came from inference (`expand`, `uniform_like` results). The
  repair makes the cast match exhaustive and carries a monomorphic deferred
  shape obligation from positional `expand` through its binding. Declared
  results or later tensor consumers select insertion versus same-rank
  replacement; a shape-neutral cast materializes the documented default. The
  final checked-program freeze point materializes the same default for an
  otherwise unconsumed result and refreshes earlier owner stamps before
  annotation, while reusable library contexts serialize the unresolved choice
  for downstream selection.
- **#878** (owner: #729; **repaired post-basis**) - `RiscOp::Pad { fill: f64 }`
  was the last raw constant-carrier seam. `Pad` now carries `ScalarValue`
  through lowering, IR, wire schema v5, eval, and typed backend emission.
- **#795** (owner: #730) - conv2d's present-but-non-literal stride/padding fall to
  `unwrap_or(1)`/`unwrap_or(0)` verbatim (census row 23); §C1.4
  raise-or-prove applies.
- **#906** (owner: #730) - eval stack-overflow abort on ~2-4k-element flat list literals;
  the WI-1 depth-budget pattern exists to apply.
- **#1150 / #1152** (owner: #730) - the newest filings: checked casts on host-built
  tensors emit no conversion (f32 buffer read as int32,
  `host_emit.rs:2572`); multi-offender trap kinds are race-dependent under
  OpenMP.
- **#1147** (owner: #731) - `scatter_elements` has no inference arm: string axes, f32
  indices, and string operands all check clean. The `scatter` arm is the
  in-tree template.
- **#851** (owner: #731) - match-arm pattern binders leak into cycle detection, producing
  false "binding cycle" rejections. Now cheap: port PR #1136's
  `collect_pattern_binders` scoping into `collect_eager_refs`.
- **#731's standing checker children** - #780 and #847 (their stated
  Phase 1-2 sequencing blocker has cleared without a fix landing), #783 (the
  durable never-degrade-an-annotation invariant), #850 (both halves: a `sig`
  with no `def` checks at 1.0 and builds to an undeclared C call), #874
  (partial: role totality landed; the tag-keyed exemption remains), #1131
  (a `lit` whose atom kind contradicts its prim family scores 1.0).
- **#730 partials** - #699 (integer floor/ceil/round support), #705 (the
  gate-dedup half), #722 (compiled unary grad rows), #794 (the
  `extract_f64_value` par catch-all), #960 (the
  chelis-python input side: raw `dtype: i32` fields, hardcoded DLPack code).
- **#729 follow-through** - #704's scalar dispatch gap remains live under
  #729. The rows #980 (op/profile/lane-dependent integer overflow), #1009
  (`pad_sequences` normative gap), and #1113 (axis-argument dtype
  inconsistency) are **repaired post-basis** with the dev/release overflow
  oracle, [05-OP-9]/[05-OP-10] semantic registrations, and required
  `BuiltinDecl` axis-layout metadata respectively.
- **Structural/other** - #888 (compiler size arithmetic saturates ->
  memory-planning slot reuse at wrong capacity; **deliberately standalone**,
  held in the roadmap's unclaimed ledger with its disposition prose) and
  #889 (runtime under-allocates >=2^32 elements; owner: #893); #886
  (hand-assembled check JSON) and #868 (zero `with_span` call sites) - both
  owner #730, dual-claimed in prose by #883; #870 (prove SIGABRT instead of
  degrading), #872 (depth-32 fuse projects to `Type::Unit`), #955, #957
  (~224 production panic sites), #958 (the einsum `unwrap_or`,
  token-anchored at `host.rs:8843`), #959 - all owner #730.
- **Evidence gaps that could hide wrong answers** (standalone
  deferred-evidence items, not class children) - #737 (Metal typed
  kernels have never executed), the #735 RNG confirmation re-sweep
  (unblocked since 2026-07-20, never run), #738/#754/#763 (no shell has ever
  run a compiled binary; the cross-lane gate is speced, Linux x86-64 first,
  and unbuilt).

## Prevention map (failure mode -> mechanism -> status)

| failure mode it prevents | mechanism | status |
|---|---|---|
| A new bare-float/raw-dtype numeric surface | §C6 census + tripwire; PR #1154's permanent dispositions (baseline regeneration cannot silently bless a change) | landed; #1160 (seam-identity relocation vocabulary) is the open design question, and it recurs on every dtype-migration rung |
| Op x dtype drift, hand-mirrored lists | #729 Phase 4 capability table | opened; table proper unbuilt |
| The next silent value substitution | #730's typed channels (landed) + Phase 4 ratchet totality (§C7) | Phase 4 unstarted (#990) |
| Gate drift / duplicate gates | #730 P3 gate contract + dedup + deletion | unstarted |
| Unhandleable AST states reappearing | #731/#908: stamped-only ingress (#1088), then carrier deletion (#1029) | in flight (PR #1036, red CI); #1088 unowned |
| Spec silence and stale claims | #733 end-to-end | nothing active - the weakest link, with the #891/#904 twenty-builtins-no-spec instance as its measured cost |
| Guards existing but not running | #1089 oracle wiring (four oracles run in no CI job; empirically confirmed - a deleted assertion produced no CI failure while the unwired oracle caught it instantly) | partial; #1090 closed as refuted - the canary already auto-files, into the shell repos (coral#23, school#189, hull#14, hello-chelis#19, octant#42, hydronnx#64) |
| Silent lane divergence | #754/#763 cross-lane gate; #738 shell compiled lanes | unblocked by #732 P2, undelivered |
| Agent-driven recurrence | #740's enforcement-ladder backlog; #895 executable plan inventories | entirely unchecked, dormant |
| Wrong issue closures | the liveness manifest (detects after the fact); a keyword-auto-close guard (prevention) | the guard is missing - three incidents (#716, #729, #912), one reverted overreach (#1151/#1159) |

## The six biggest remaining gaps, elaborated

### 1. #730's gate half - the largest doc-vs-main gap found, now honestly recorded

The CLI carries twelve hand-rolled `reject_*` gates that pre-screen programs
before the emitter runs. The proven duplications: `reject_unsupported_hip_ops`
exists in two drifted copies (the chelis-cli copy omits `ScatterElements`
from the sparse-index set and returns bare strings; the compiler-api copy is
branded), and `COMPILED_HOST_ONLY_BUILTINS` duplicates `HOST_ONLY_BUILTINS`.
Because chelis-cli has zero `unimplemented_rejection!` uses and hand-typed
`"unsupported: "` literals, the sealed 48-kind DiagnosticKind vocabulary
PR #1037 built does not govern the CLI surface at all - the same program
gets different diagnostics depending on entry path, and drift between copies
is the mechanism that produced #697/#698 originally. The work: one typed
definition per gate shared by both surfaces, the recorded gate contract (a
gate may only make a diagnostic earlier or more specific), deletion of gates
that duplicate emitter rejections (proven by the rejected-cells corpus), the
no-duplicate-gates tripwire, and the first-error decision affirmed before
the contract freezes. Phase 4 (#990) is separate and entirely unbuilt: the
derived-universe tripwire over the real product-source manifest (Python/C
adapters so a substitution cannot hide in a script), the ~150-site panic
triage (#957), #958, and the two CI legs. The #1151 same-day revert shows
the delivery must ship narrow.

### 2. #729's capability table - the mechanism that retires hand-mirrored lists

Today op acceptance is hand-maintained lists, backend dispatch hand-written
match arms, conformance hand-curated files - three places every new op must
agree. The table inverts all three: checker acceptance derived from Table A
(op x dtype x surface, atom citation mandatory), dispatch skeletons
macro-generated so a `Supported` cell with no kernel is a compile error and
a kernel with no cell is dead code, conformance generated. Part of v0.19's
decision payload already shipped ahead of it as check-time rejections (#724
and #726 closed on that basis). Still to decide/ship: #712/#715 (five tests
still ignored, scalar contract unsettled), #753 (wrap ops exist only as spec
text + Unimplemented cells), #759's `cast_saturate` rung, the #689/#693 GPU
cells, #965. Entry conditions: schema freeze plus five recorded open
questions (the surface axis for container callables is the one that bites).
The merged census permanence is the floor; #1160 needs deciding so the next
dtype migration can relocate seam identities without a hand-carried
maintainer override. Known inherited holes: #951, #957, #958, #960, and the
capability table's own stale seed-row statuses (#724/#726 "not yet landed",
left for this phase to refresh).

### 3. #1088 - unowned, and everything downstream of it is blocked

The compiler-api generic Deep ingress (`parse_str_strict`) skips the
declaration-role stamp - a weaker second front door through which unstamped
trees reach consumers today. Three things queue behind it: #731's §C4.2
successor acceptance (forbidden while a second ingress exists), open PR
#1036's ingress half, and #1029's deletion of `Expr::List`/`Atom::Tag` - the
make-illegal-states-unrepresentable payoff of the entire #908 arc, still
~1,082 occurrences deep. The enabling work is done (PR #1126's 31-reader
`stamped_parts` sweep; a re-salvage baseline exists in
`phase3_stamped_ingress.rs` needing the `.kind().as_str()` adaptation). It
needs an owner; nothing else.

### 4. #912 Faces 1 and 4 - the other half of v0.19

Face 1 (existence): the manifest walkers are `Expr::Node`-blind (#1082), so
a stamped program can produce a silently empty manifest; anonymous tuple/ADT
roots drop (#1083). Face 4 (artifact): `requires_main` reaches no production
caller - all three `cmd_build_c_result` sites pass `None`, leaving
`c_source.contains("int main(")` as the deciding authority, which spec/05
says cannot satisfy [05-UNS-1]. The dependency order is written: #1082 ->
#1079 (consume `ManifestedProgram`, delete the source_arch guard) -> thread
`requires_main` incl. HIP -> #1083 -> #1084's acceptance cells -> the C1 f64
build-routing. Jeff's chain; #1097 landed its fail-closed disposition piece;
#730's deliverable 6 lands inside it.

### 5. The state-basis defects exposed missing delivery slots

The list above is the audit inventory at its recorded basis. The post-basis
#729 follow-through now resolves #878, #901, #937, #942, #980, #1009, and
#1113 and records each repair's recurrence guard in the roadmap. The remaining
high-value quick wins include #851 (port an existing helper) and #1147 (copy an
existing inference arm). The lane-level items - #690
and the pinned support cells (#689 HIP int64 kernels, #951 fused reduce,
#759's HIP cast trap) - close only by landing kernels and validating through
the local HIP gate.

**The structural problem at the state basis was that these defects were owned
but not scheduled.** Every entry had a parent class, but almost none appeared
in a remaining phase deliverable. The #729 follow-through now supplies a
delivery disposition and recurrence guard for #878, #901, #937, #942, #980,
#1009, and #1113. The scheduling gap remains for #780, #783, #847, #850, and
#851, whose "behind #731 Phase 1-2" sequencing point has passed, and for #795,
#906, #870, #872, and the #957 panic family under #730. No owning plan oracle
turns red while those defects stay open. Their class plans and the roadmap
ledger still need an explicit disposition - a
named phase/kill-table row in its owning plan, an assignment to a v0.19
capability decision, or a recorded "standalone fix, no phase dependency"
line like the ledger already does for #681 and #888 - so the delivery story
is inspectable rather than implied by parentage.

### 6. The institutional layer - specified, largely not running

#733's Phase 0 is self-contained and Chelis-owned, so its dormancy has no
external excuse; the measured cost is already visible (twenty builtins and a
prelude ADT with zero spec entries at the state basis; the #1009 instance is
now repaired). #740/#895 stay untouched even
though #895's worked instance showed a plan's own inventory row silently
skipped. #1089's four unwired oracles are empirically proven gaps. The
keyword-auto-close guard is owed and unowned after three incidents and one
reverted overreach - the liveness manifest only detects a wrong closure
after the fact, on an unrelated PR's CI. And #754/#763/#737/#738 leave whole
lanes evidence-free: Metal has never executed a typed kernel, and no shell
has ever run a compiled binary. Ranked by leverage: give #1088 an owner,
wire the oracles, then the gate dedup - all three are small relative to what
they unlock, while the capability table and the #912 chain are the two real
campaigns left before v0.19/v0.20.

## Open PRs

| PR | what | state |
|---|---|---|
| #1036 | #731 P3 successor/authoring ingress hardening (tracks the #1088-gated half) | open, **red CI** (Integration Linux + macOS Smoke) |
| #1031 | Canonical Surf grammar Phases 3-5 (#1024 track, v0.19 syntax) | open, red macOS Smoke |
| #1034 | Timeless normative specs | open, red macOS Smoke |
| #880 | Typed host function values + spans spec plan (#909/#883) | open |
| #1155 | Extract functional compiler pipeline core | draft |
| #1135 | CI dedup / merge-queue workflow policy (wants the merge queue configured first) | draft |
| #838 | Conform central workflow wrappers (#788 family, Chelis-Lang/ci#5 prereq) | draft |

## Pending maintainer decisions

1. **#796** - accept spec/05 §3.6.1's authored eval-only `test_*` contract as
   its resolution (close by decision), or keep it open for compiled-lane
   support.
2. **#862** - land the named unit-valued-root regression test, then close.
3. The **eight parent-link calls** listed above, plus the #1023/#1029
   hub-with-a-parent shape.
4. **#1160** - the census seam-relocation mechanism; the issue itself asks
   for a dedicated red-team pass because it loosens a ratchet.
5. **spec/04 [04-NUM-4]/[04-NUM-5]** - their "not honored" parentheticals
   were removed on a negative search (no open bool-arithmetic or
   `fold_static_cond` issue found); if #892's Metal bool storage is judged
   inside [04-NUM-4]'s scope, restore that one parenthetical citing it.
6. The **§C5 census row 26 site-column anchors** in `loud_unsupported.md`
   are stale but frozen under B1; re-anchoring them is a deliberate
   baseline change, recorded in the PR #1163 body.
7. **The owned-but-unscheduled disposition pass** (gap 5): amend the five
   plan docs / the roadmap ledger so every live defect with a class parent
   also has an explicit delivery disposition - a phase or kill-table row, a
   v0.19 capability-decision assignment, or a recorded standalone-fix line.

## Process findings worth keeping in view

1. **Keyword auto-close is a live hazard.** #716, #729, and #912 were all
   closed by squash-body phrases and reopened; the mitigation attempt
   (#1151) was reverted as overreach the same day it merged. Nothing guards
   this today beyond the liveness manifest's after-the-fact detection.
2. **Review debt is now structurally addressed.** 22 merges landed 8/1-8/2
   with zero approving reviews, several with self-reported verification
   later found false; the #1096 governance rollout (no-bypass, squash-only,
   green up-to-date checks) now governs every merge. The audit's
   verify-by-execution discipline refuted two claimed bugs before code was
   written and caught a plan-doc over-claim (#730 Phase 3) that inspection
   had accepted.
3. **CI masking.** #1095 was deterministic in both lanes and invisible to
   required CI under the filtered profile (fixed by #1103, but the fail-fast
   + filtered-profile pattern that hid it is unchanged). Related: one #749
   lock cell (`c_nested_tensor_truncates_at_32_with_marker`) silently skips
   without a C toolchain while every sibling panics - the only cell in its
   group that can vanish without turning anything red.
4. **Issue hygiene decays without a closing discipline.** The validation
   pass found ~29 fixed-but-open issues, several with threads describing
   their own fix as still in flight; the plan docs even contained an
   explicit unperformed instruction ("close the issues on the PR #863
   merge"). Closures now carry evidence comments; the standing gap is
   nothing enforces the close step when a fix lands.
