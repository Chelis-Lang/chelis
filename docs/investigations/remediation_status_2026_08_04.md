# Chelis Numeric Remediation - Status (2026-08-21)

This is the state of the numeric-remediation board. `main` is at `77a74ea1`.
The shipped release is **v0.18.4**, and it is breaking on four boundaries.

**`main` is ahead of the release.** The `v0.18.4` tag is `c0138c82`, fourteen
commits below this basis, so every post-release repair is true of `main` and
not of the release a shell can install: the scalar alignment (#1188), the gate
and reshape fixes (#1195, #1196), the pipeline core (#1155), the
recursive-generic build capability (#1204, #1215, #1218 - closing #1158,
#1201, #1216 and coral#26's build lane), the #1200 linearity repair (#1208),
the CSV ingestion repair (#1213, part of #943), the #912 test wave
(#1230-#1232), and the rustc 1.98.0 toolchain pin (#1236). Where that
distinction changes what a reader should do, the text says which side of the
tag a repair sits on.

Every count and code claim here was measured on `841b6ddb` for the 2026-08-05
revision and re-measured on `77a74ea1` wherever this 2026-08-21 revision
changes it, and the command is
named where the number matters. Three figures are *not* independent
measurements and are attributed in place: the changelog drift numbers are PR
#1199's own, #847's non-reproducing disposition is PR #1178's, and the 8/1-8/2
review-debt count comes from the sprint audit
(`docs/investigations/sprint_audit_2026_08_908_912_729.md`). The other
standing sources are the five plan docs under `spec/design/` and the class
trackers' live GitHub sub-issue graphs.

The file keeps its 2026-08-04 path because
[`spec/design/remediation_roadmap.md`](../../spec/design/remediation_roadmap.md)
cites it there. The roadmap is the delivery ledger and the home of the
per-defect recurrence guards; this document is the board.

## TL;DR

Waves 0-3 of the remediation roadmap are complete, with one exception: #733's
Phase 0 (OpenSpec activation), the lone outstanding Wave-0 item. Of the four
Wave-4 slots, #732 Phase 3 is delivered, #729 Phase 4 is open at its
census-permanence slice with the capability table proper unbuilt, #730 Phase 3
has its gates but not its contract and Phase 4 is unbuilt, and #733 has
nothing.

The five classes carry **40 open children** between them, down from 63.
Everything still open falls into four groups: a defect inventory concentrated
in #730's support cells and #729's GPU lanes, two engineering campaigns (the
capability table and the #912 root-boundary chain), one unowned keystone
(#1088), and an institutional enforcement layer that is specified and largely
not running.

The weak point is downstream. **v0.18.4 is breaking on four boundaries in a
patch cut** - canonical Surf, the int64 C ABI, the declaration contracts, and
WireDag schema 5 - and two of them land on the roadmap's anti-churn
invariants: the ABI widening overrides invariant 7's default, and the schema
step raises a scope question invariant 1 does not answer. Neither is a clean
violation and neither is clean compliance (pending decision 1). Of the two
post-release defects, **#1200** - the linearity regression that took coral's
suite from 74 passing to 17 - is repaired on `main` (PR #1208) but live in the
release a shell installs, and **#1197**, the `chelis migrate surf` batch
abort, is still open and unowned. The same main-side wave also retired coral's
other build-lane wall: recursive generic host calls compile via bounded
memoized monomorphization (#1204/#1215/#1218), and `Coral.Frame` now builds,
links, and runs against the eval lane. Nothing in CI could see any of these;
all were found downstream.

## Scoreboard

| plan | landed | remaining |
|---|---|---|
| **#729 dtype semantics** | P0-P3 + §C6 census, all CI-wired (PRs #758, #956, #1033, #1049, #1054, #1065, #1118); Phase 4 opened by PR #1154 (permanent dispositions for all 301 census rows, descriptor-manifest bound, red-teamed  x 2); the dtype ledger delivered as code by PR #1181 (sealed `ScalarValue` `Pad` carrier, total exact-representability predicate, per-float-width `uniform_like` emission, exhaustive cast result-state match, checked runtime arithmetic with the dev/release overflow oracle, [05-OP-9]/[05-OP-10], required `BuiltinDecl` axis-layout metadata); [04-NUM-15] authored by PR #1169 and enforced by PR #1189 (one `CheckedCastPlan` over all 81 active dtype pairs, identity only on the same-`Prim` diagonal, lowest row-major flat index selected under OpenMP - closing #1150 and #1152); the scalar numeric family aligned across checker, eval, and C by PR #1188 (closing #704, #712, #715), which merged after the v0.18.4 tag | Capability table proper: Table A/B, derived checker acceptance, macro-generated dispatch (Supported cell without kernel = compile error), generated conformance suite, #912 projections. Plus: the P0 detector-half adjudication (unowned), the P3 oracle adjudication + stale tracker checkbox, the #1160 seam-relocation decision, #1112's GPU half (`chelis_gpu_tensor` still carries 32-bit extents after the host ABI widened), and the GPU support cells, none of which has moved (#689, #690, #693, #951) |
| **#730 loud unsupported** | P0-P2; P3's typed `RejectionAuthority` + sealed 48-kind `DiagnosticKind` slices (PR #1037); P3's gate contract in two slices - PR #1175 (one typed compiler-api policy behind both public build paths, the `reject_unsupported_hip_ops` pair deduped, `COMPILED_HOST_ONLY_BUILTINS` deleted, the C precision preflights removed, a syntactic `reject_*` source manifest over both crates) and PR #1186 (Metal and seeded-effect policy moved into those same typed definitions, one shared host tensor-helper traversal, entry-scoped pure-DAG effect rejection, the single Phase 3 runner). Closed #697, #698, #705, #959, and - via LU6 in PR #1189 - #1150, whose `_ => input` host-emission fallback is deleted. Closed #1158 via PR #1204 (after the tag): recursive generic host calls compile through bounded memoized monomorphization under the new [04-INF-2]/[04-INF-3] uniform-recursive-instantiation atoms (spec/04 §3.1.1) - a BREAKING check-time rejection of polymorphic recursion, lane-uniform - with PRs #1215/#1218 extending the same specialization to non-recursive generics (#1201) and recursive erased-dim generics (#1216), and the surviving fail-closed residue re-cited to the open tracker #1226 | **Phase 3 is not complete**: `scripts/loud_unsupported_phase3_oracle.py` passes five legs and is red on the sixth, the independently owned #912 root-realizability interlock. Also owed: the recorded gate contract as a written rule, deletion of gates that duplicate emitter rejections, a no-duplicate-gates tripwire beyond the syntactic manifest, and the first-error-vs-accumulate decision affirmed before the contract freezes; deliverable 6 rides the #912 chain; Phase 4 (ratchet totality, #990) is entirely unbuilt - no phase-4 oracle script, no nightly workflow |
| **#731 checker totality** | P0-P3 implemented, incl. the construction gate (PR #1101) and authoring-scope fixes (PR #1136); **PP1 + PP2** (PR #1178): the function-parameter typing channel under [04-INF-1], the writeback information-ordering gate, the atom/primitive agreement matrix under [04-LIT-1] and `spec/03` §6.4, the orphan-`defsig` rule in `spec/03` §2.2, and the registered-builtin arm tripwire - closing #780, #783, #847, #851, #1131, #1147 | The decode-once rework's fresh-context red team is named pending in `checker_totality.md`'s header, but the round ran on PR #855 (an exact-head architectural pass and an explicitly fresh local-subagent pass, both FAIL, findings folded before merge), so that line is stale text rather than unmet work; the §C4.2 ingress half (#1088 - unowned, with #1129 holding its landing inventory); #850's build half; #1125's stamped-Node ingress-parity sweep; #1134's ingress-divergence decision; #874's tag-keyed exemption with #887 under it; carrier deletion last (#1029, whose stated precondition - no producer exists - is false: 1,014 `Expr::List` and 77 `Atom::Tag` occurrences under `crates/`, and the `Node::to_list` bridges plus `chelis-surf`'s desugar output are live producers) |
| **#732 faithful observation** | All four phases (P3 via PRs #1099/#1115/#1118, shipped in v0.18.3); both known-red ledgers empty (`KNOWN_RED_CELLS` is `()` at `faithful_observation_phase2_oracle.py:159`, and `C_LANE_EXCLUDED` has no production occurrences left); tolerance table + #687 handshake shipped; Phase 2's empty-ledger acceptance repaired and continuously wired by PR #1185 | Direct class debt: #997's one structural `FO-DIAG` diagnostic-rendering migration (36 attributed tripwire tokens, one of them cfg-test-only), which is what keeps #732 open. Separate capability: #1059's C-host `to_string`. Oracle facts: Phase 2 has its own blocking `Faithful Observation Phase 2 Oracle` job and Phase 3 runs nested inside the `Dtype Phase 0-3 Oracle` job, both on every non-docs-only PR; #990's scheduled/change-gated package is separate |
| **#733 spec provenance** | Nothing - Phase 0 is defined but explicitly not activated: no oracle suite, no PR template, no `governance` gate stage, and a `.github/CODEOWNERS` that covers 16 #729/#730 guard artifacts but not `spec/**` (its own header says Phase 0 owns the broader signoff) | Everything. P0 is Chelis-owned and self-contained; P1-P3 additionally wait on three Buoy prerequisites (the pinned revision's `devenv test` oracle, the `buoy.adapter-sdk/v1` pin, the approved host parsed-item schema). Its four children are #735, #797, #895, #898. v0.20 is formally blocked on P3 |

## Sibling classes (what the five plans structurally cannot deliver)

- **#908** (Deep AST successor; the `Atom`/list-head domain #731 can't reach;
  children #885, #1023, #1029, #1087, #1088, #1093; design doc
  `spec/design/unrepresentable_ast_domain.md`, whose status line still reads
  "Plan approved, implementation pending" and which names none of the newer
  children - their delivery slots live in #1023 §B instead): the headline
  invariant is enforced (a bare `Name` at a RuntimeExpr slot is
  unrepresentable), but the class is not closed. The legacy carrier has
  producers in ten crates, and the decisive one is
  `chelis-surf/src/desugar.rs`'s `normalize_to_lists`, which converts every
  validated `Node` back to `Expr::List` at the desugar output boundary - so
  **every `.ch` compile emits the legacy carrier**. `chelis-deep/src/ast.rs`
  still documents that variant as "DEPRECATED: No producer creates this
  variant anymore", which its own crate contradicts. **#1088** - the
  compiler-api generic ingress (`parse_str_strict`) skipping the
  declaration-role stamp - is the keystone and is **unowned**; `parse_deep`
  at `compiler.rs:2903` is a third, non-strict door that #1088's body does
  not yet name.
- **#912** (root boundary; open children #1079, #1080, #1082, #1083, #1102,
  #1148, plus closed #820/#862/#947/#1081/#1084/#1095): Face 2
  (the `name = ` prefix) shipped in v0.18.1; Face 1 (existence) is partial -
  manifest walkers are `Expr::Node`-blind (#1082), and anonymous tuple/ADT
  roots still drop from the manifest: the dotted-root expansion is a live
  `TODO` at `chelis-effects/src/realizability.rs:506`. #1083 owns that half;
  it was briefly closed on 2026-08-21, two minutes after PR #1231 merged and
  against that PR's own "this does not close #1083" record, and reopened the
  same day with the conflict recorded on the thread. The manifest is non-empty today only because of the
  transitional `normalize_nodes_to_lists` bridge, so **#1029's carrier
  deletion silently empties manifests unless #1082 lands first** - the two
  sibling classes are coupled in that order. Face 4 (artifact) is open, and
  worse than the checklist wording suggests: `requires_main()` has exactly one
  production caller, `compute_manifest_json` at `main.rs:1799`, which
  decorates `eval --json` and decides nothing. All three `cmd_build_c_result`
  sites pass `None`, so `c_source.contains("int main(")` at `main.rs:8556`
  decides, and `cmd_build_hip_host` never takes the parameter at all - it runs
  the same text search at `main.rs:8633`. spec/05 says a source-text scanner
  cannot satisfy [05-UNS-1]. #1023 §C is the live
  acceptance checklist. The chain still gates a second plan's oracle as well
  as its own: #730's Phase 3 runner ends at a leg that runs
  `issue_912_root_boundary` with `--run-ignored all`. PR #1231 re-ran that
  target's six ignored cells against `main`, found four quietly green
  (un-ignored, after a review round caught three of them passing vacuously),
  and declared the remaining two - both `todo!` manifest-completeness cells -
  in an enforced `IGNORE_LEDGER` citing #1079, so the leg is red on exactly
  those two until #1079 lands. #1084's surviving ask (the ledger itself) is
  what #1231 delivered; its first two asks had already landed unrecorded.
- **#909** (typed host function values / C-host callable ABI; children #866,
  #867, #879): no longer paperless - `spec/design/host_function_values.md`
  is written and re-authored against current `main` (PR #1173), and the
  earlier spec lock PR #880 closed as superseded. Nothing is implemented, and
  #879 stays in the CI-pinned set.
- **#883** (diagnostic spans; open children #868, #886, #1172; #916 closed
  2026-08-17 as fixed-at-main by code audit, its purpose-built reshape
  slot-mismatch diagnostic having shipped in what is now
  `infer/app_shape.rs`): the decision
  is made - `Unsupported` folds into the #909 plan as its FV1 slice,
  `CheckError` stays here with `report_at` as the primitive - and none of it
  is executed. `Unsupported::with_span` has zero call sites while 49
  production sites construct through `Unsupported::new`, which hardcodes
  `span: None`, and `Display` would not render the field anyway. On the
  checker side 7 of 334 `CheckError::new` production sites carry a span, so
  this is a producer gap rather than a rendering gap. #730 §C2 owns the span
  *contract* while the fix lands here, which is why the roadmap's sibling
  table names #883. **#1172 is worse than uncovered**:
  `spec/design/chelis_span_survival.md` §1 explicitly sanctions the
  metadata-only carrier it rejects, so repairing it means amending that doc,
  not only writing code.
- **#893** (runtime representation: seal the `pub *mut u8` tensor data, make
  `Repr` the ABI primitive; children #899, #889; #892's bool storage rides
  #729's v0.19 cut): 39 of the 199 capacity-census rows cite it, and they are
  the last live pre-ratchet seam rows, so `capacity_census_liveness.py`
  requires it OPEN until they unwind. Its delivery PR #894 closed unmerged;
  the real successors are #975 and #964 (#907 is the Devenv/Nix PR and carries
  none of this work). `Repr` now exists as the vocabulary primitive but
  nothing in the tensor struct or header is keyed on it, and
  `chelis_runtime.h` still types storage as `float *data` for every dtype. No
  August activity, and the class is paperless by design.
- **#740** (agent quality architecture; children #803, #808, #823, #824,
  #852, all CI measurement work; #895 sits under #733 with an "Also part of
  #740" comment recording the second home): two of the eight backlog items
  exist. `crates/chelis-lint/AGENTS.md` is the one per-crate guardrail of
  four, and the mechanism index carries 13 rows - but nothing binds an index
  row to landing a mechanism. Change-shape skills, the sibling check,
  contract-first docs-PRs (gated on #733 Phase 0), CI ratchet metrics, the
  duplicates registry, and scheduled audit agents do not exist. #895's worked
  instance is the cost: a plan's own inventory row was silently skipped by a
  phase PR, and the #893 cmplt defect (`-1 < 0` -> false through an
  f32-reinterpreted read) rode the gap.
- **#788** (conform audit surface; children #814, #825, #845): open parent by
  design ("a parent, not a plan"); draft PR #838 is in its family.

## Release axis

What each cut obliges a shell to do, not what each cut contained.

- **v0.17.x** (shipped) - the first source migration: loud checking, the seed
  suffix, dtype-faithful eval rendering.
- **v0.18.0/v0.18.1** (shipped) - checker totality, the host-type/ABI
  boundary, compiled rendering, then [05-OBS-6]'s `name = ` prefix as an
  exact-output migration.
- **v0.18.2** (shipped) - additive (CSV/JSON serializers, eval JSON/CSV
  builtins, `--timeout`, Nix/Devenv, #729 Phase 2 typed integers, exact
  trapping integer `abs` in C). Its one source-visible delta is a **recorded
  divergence, not a migration target**: eval `shape()` returns int32 against
  [05-DIM-2]'s int64 SHALL, and #1120's note says explicitly not to migrate
  onto it.
- **v0.18.3** (shipped) - a **source + exact-output migration**, not a pin-only
  patch. Extents move to int64 under [05-DIM-1/2] (`shape()` returns int64,
  movement bounds take int64, bare int32 extent sites are check errors, axis
  params stay int32), which also retires 0.18.2's #1120 divergence so a shell
  that skipped 0.18.2 migrates once. Compiled `round` half-ties to even.
  `cast_trunc` is [05-OP-6] and a reserved word. This is the migration target
  for the #1091 shell breakage. It also carries #732 Phase 3's tolerance
  table, shared comparator, and oracle - behavior-preserving, and the reason
  Phase 3 is not v0.20 payload.
- **v0.18.4** (shipped, current) - **a source migration, an ABI break, and a
  wire break in one patch cut.** Four boundaries move: the published C ABI
  carries extents as `int64_t` while `chelis_tensor_shape`'s `axis` narrows to
  `int32_t` (chelis#1149, part of #1112 - rebuild anything linking
  `chelis_runtime.h` or consuming emitted C); Surf's canonical form becomes
  v0.19 and `chelis fmt --check` runs ahead of the front end, so previously
  canonical source fails the gate (PR #1031 - migrate with
  `chelis migrate surf --from 0.18 --inplace`, and author the renames it
  refuses to guess); an orphan `defsig` no longer checks and unresolved
  deferred inference rejects at its declaration boundary ([04-INF-1], PR
  #1178 - authored annotations); and WireDag payloads move to schema 5,
  migrating on read one way only (PR #1181). Riding along without a migration:
  release-build integer overflow traps in `scatter`/`cumsum`/`trace`/`einsum`,
  the exhaustive checked cast (PR #1189), and the dependency typecheck cache
  (PR #1176, which invalidates 0.18.3 on-disk caches by design and reorders
  emitted C for multi-package reef builds). **The scalar
  `/* unsupported builtin */ 0` stubs behind #704/#712/#715 are still present
  here** - PR #1188 repairs them on `main`, after the tag - and so are the
  #1200 linearity regression (repaired main-side by PR #1208), the branded
  recursive-generic rejection (retired main-side by #1204/#1215/#1218), and
  the quadratic compiled-lane CSV ingestion (repaired main-side by #1213).
  `[Unreleased]` now also carries a BREAKING checker change - polymorphic
  recursion rejects at check time under [04-INF-2]/[04-INF-3] - so the next
  cut is migration-bearing regardless of what else it takes (pending
  decision 9). Two of the four
  boundaries land on the roadmap's anti-churn invariants; see pending
  decision 1 for what that does and does not settle.
- **v0.19** (planned; source migration) - carries what 0.18.4 did not take:
  the manifested root completion (#912/#1023: complete root set and order,
  unavailable-root diagnostics, artifact routing), the residual capability
  decisions (#753's wrap ops, #759's `cast_saturate` rung, the #689/#693 GPU
  cells, #965), #729 §C3's per-dtype storage break with #894's bool storage
  rename and #892, and the GPU-lane extent widening. That last one is
  currently a live inconsistency rather than a plan item: `chelis_gpu_tensor`
  carries 32-bit extents while the host ABI carries 64, so the two disagree on
  `main` today (#1112's remaining half).
- **v0.20** (planned; mechanical) - #729 P4's table, #730 P3's gates-as-UX,
  and #733 P3's blocking provenance ratchet. Behavior-preserving by
  construction; #733's inactivity is the critical path to it.

## The issue graph

### What the five classes hold

Of the 119 issues parented to the five trackers, 40 are open. #733 is the only
class with nothing closed under it.

| tracker | open / total | open children |
|---|---|---|
| #729 | 12 / 50 | #689, #690, #693, #753, #759, #892, #951, #965, #1091, #1092, #1112, #1160 |
| #730 | 17 / 36 | #699, #722, #794, #795, #870, #872, #906, #955, #957, #958, #960, #990, #1058, #1137, #1138, #1143, #1157 |
| #731 | 5 / 20 | #850, #874, #1125, #1129, #1134 |
| #732 | 2 / 9 | #997, #1059 |
| #733 | 4 / 4 | #735, #797, #895, #898 |

What holds the closed side down matters more than the count, because it is
what says whether a class can recur. Grouped by mechanism (not exhaustive - each
class also carries older closures from its own phases):

- **#729.** Phases 1-3's sealed storage, typed kernels, and per-dtype
  wire/Python boundaries close the bulk (#680, #684, #685, #686, #688, #711,
  #713, #717, #720, #724, #726, #856, #860, #897, #1120). PR #1181's dtype
  ledger closes #878, #901, #937, #942, #980, #1009, #1113. PR #1189's
  exhaustive `CheckedCastPlan` closes #1152. PR #1188's scalar alignment
  closes #704, #712, #715 - `main`-side of the v0.18.4 tag, so a shell on the
  release still meets those stubs. **#691** and **#714** needed only PR
  #1164's rehoming of their stale rejection citations; both sit under #729
  even though #714's diagnosis is host-type/ABI work #730 P2 delivered.
- **#730.** Phases 1-2 make the substitutions unwritable (#682, #692, #725,
  #734, #744, #745, #900, and the nullary-generic-ADT family #935, #936,
  #939, #940, #941, #948). PR #1175's first gate-contract slice closes #697,
  #698, #705; PR #1186's second closes #959; PR #1189's deletion of the
  `_ => input` host-emission fallback closes #1150. PR #1204 closes #1158 by
  landing the capability itself - bounded memoized monomorphization with
  recursive edges preserved as calls - rather than by re-branding the
  rejection; what still cannot resolve moved to the new residue tracker #1226
  with recursion-neutral wording (PR #1215). #725 and #734 are
  consolidated into their open support owners #1058 and #1059 per the
  one-tracker rule, so each still owes a *support* half.
- **#731.** PP1/PP2 (PR #1178) closes #780, #783, #847, #851, #1131, #1147,
  on top of #1132's [04-NUM-10] disposition and the earlier phase work. Two
  entries do not mean what a closure usually means: **#847** is recorded
  non-reproducing at the PP1 baseline on PR #1178's word (its own thread
  records no such verification), and **#850** stays open because only its
  check half shipped.
- **#732.** All seven (#716, #723, #748, #749, #775, #1078, #1104) come from
  Phase 2/3 and the oracle hardening around them.

Two issues outside the classes belong to the same story. **#796** is closed on
the maintainer call its row asked for, citing spec/05 §3.6.1's eval-only
`test_*` contract; the compiled-assertion request it carried lives at #1170.
**#862** closed on 2026-08-21 - the cheapest close on the board got made: PR
#1230 landed the named unit-valued-root regression test over three
sole-print-root programs, avoiding the three vacuity traps its disposition
notes documented.

**Two caveats on that 40.** All 22 of the 2026-08-05 class closures rest on a
PR-body keyword, the same mechanism that mis-closes #716, #729, and #912, and
the guard named in the prevention map is still owed (#1158's 2026-08-07
closure rode the same path, legitimately - PR #1204's oracle is green). And a
small number of open children is not the same as a small amount of open work:
**twenty-seven** open issues now sit outside the five classes, parented to
#1024, #883, #1170, or nothing. Thirteen predate the 2026-08-05 revision
(#1168, #1170, #1171, #1172, #1177, #1179, #1180, #1182, #1183, #1184, #1192,
#1197, #1198 - #1190/#1191/#1193/#1194 closed, and #1200/#1201 left the list
by repair rather than by gaining a class). Fourteen were filed since, all
parentless: a performance family (#1205 front-end superlinearity, #1206
recursive-def tensor retention, #1207 superlinear typechecking), the #1208
linearity residue (#1209 name-vs-generation alias resolution, #1211 carrier
identity, #1212 authored-carrier collision), two `soundness` finds (#1214 HIP
fused-in-place missing chelis#933's caller-storage check, #1224 the SMT
runner dropping a where-clause assumption), #1222 (compiled-binary abort at
exit), the #1213 excavations (#1217 stale-stdlib false green, #1225
per-character CSV recursion wall), the fail-closed residue tracker #1226, and
two CI items (#1221, #1234). The class-shaped-work-with-no-class observation
that #1200/#1201 exposed now applies verbatim to the linearity residue and
the performance family.

### The CI-pinned set (11, all legitimately live)

`spec/design/loud_unsupported_issue_manifest.json` requires these OPEN while
emitter `unimplemented_rejection!` sites cite them - closing one reddens the
Rejection Authority Liveness job: **#600** (value-derived output dims),
**#689** (HIP typed kernels beyond f32/f64), **#729** (the tracker), **#759**
(the HIP cast half - `cast_trunc` landed on C, but HIP still emits an
unguarded device conversion and the issue demands every-lane parity),
**#829**, **#879** (C-host function-value ABI), **#951** (`emit_fused_reduce`
f32-hardcoded), **#1058**, **#1059**, **#1138**, and **#1192** (compiled
seeded-dropout kernels, added by PR #1186 when the host-helper `dropout`
emitter panic became a branded rejection). Each closes by landing its support,
not by cleanup. (#691 and #714 left this set via PR #1164; their one-time
closure by #1151 and un-closure by the #1159 revert is the worked example of
why closing a pinned issue from the roadmap top-down breaks the build.)

### Where the graph puts things

The graph is the assignment record, and it agrees with these two documents.
Every issue either one assigns to a class resolves to that class's
`/issues/<n>/sub_issues` list, and the extent cluster (#1110, #1112, #1113,
#1116, #1120, #1140, #1160) sits flat under #729 rather than under a sub-hub -
#1112 is not framed as a tracker, so re-point #1120 under it if one is wanted.
The one link the walk found missing is now in place: **#713** (`pad_sequences`
allocates int32 output for int64 input), which the roadmap assigns to #729
Phase 3 and PR #1118 retired, is a sub-issue of **#729**.

Three placements are deliberate choices rather than missing links, and read
wrongly if you take the prose in either document as the authority:

- **#868, #886, #916, and #1172** are the four children of **#883**, not of
  #730. #730 §C2 still owns the span contract; the link records where the fix
  lands, and the roadmap's sibling table names #883 for that reason.
- **#887** is a child of **#874**, which is itself a child of #731 - so it
  sits in the #731 subtree, matching "Tier 1 stays with #731". The roadmap's
  sibling table lists it under #908; that is the Tier 2 claim, not the link.
- The five class **METAs** (#727 with its #695, #703, #709, #728, #694) are
  deliberately outside the sub-issue graph. Per `AGENTS.md` the META/tracker
  pairing is historical and not the pattern for a new class, so they are not
  filed as children of their own trackers.

Every parent question the plan docs leave open has an answer on the graph:
**#916** sits at #883, **#1148** at #912, **#1157** and **#1137** at #730,
**#1156** is parentless because it belongs to no class (PR #1161 fixes the
defect), **#985/#995** are unlinked as closed dupes of #965, and **#895** is
at #733 while its body opens "Part of #740", and the "Also part of #740"
comment the one-parent convention requires is posted, so that pairing is
recorded rather than implied.

Two structural notes stand. #1023 and #1029 are both hubs that carry parent
#908, which is legal but worth confirming intended; of the two, only #1029
carries the `tracking` label, so under `AGENTS.md`'s "`-label:tracking` is the
work queue" rule #1023 sits *in* the queue as an ordinary work item
(`documentation`, `soundness`, `type-system`, three `area:*`). And #1170, the
C-lane capability tracker, is correctly its own root with one child: #1192,
filed by PR #1186 when the host-helper `dropout` panic became a branded
rejection needing a support owner.

Five open children appear in no plan doc, and all five are legitimate:
**#1091** (the [04-NUM-14] cast trap breaking five shells) and **#1092**
(hydronnx's privatized `TensorValue.data`) under #729; **#1129** (the landing
inventory for the #1036/#1038 and #1037/#1042 branch work) under #731, which
carries more weight now that #1036 is closed unmerged; and **#797** (central
OpenSpec governance gate) and **#898** (reference-only `min_reduce`/
`prod_reduce`/`argmax_reduce`/`argmin_reduce`) under #733, which are the only
two concrete work items that tracker carries beyond #735 and #895.

## Live defect inventory

Each entry is tagged with its owning tracker class and describes `main` at
`77a74ea1`. All but the last two groups are parented sub-issues of a class this
doc covers; the caveat is that **parented != scheduled** - a parent gives
ownership and the oracle that proves the fix, but several of these appear in
no remaining phase deliverable of their owning plan (see gap 5 below). No open
PR carries any of them: the in-flight board holds no plan-set work at all
(#1227 documents #1207's investigation but fixes nothing).

- **#690** (owner: #729) - HIP integer division by zero: kernels emit bare
  `a/b`; exit-0 garbage on GPU where eval traps branded. Fix is a
  kernel-template guard, validatable locally via `scripts/hip_test.py`.
- **#689 / #693 / #951** (owner: #729) - the GPU and fused-reduce support
  cells. Read them past their titles: the silent halves are dead. HIP's
  `elem_kind` now returns a branded `unimplemented_rejection!(689, ...)` for
  every integer and narrow-float `Prim` instead of falling back to F32, and
  Metal rejects integer `abs` at its public emission boundary
  (`reject_integer_abs`, `chelis-backend-metal/src/emit.rs:280`) instead of
  emitting zeros. What is left in all three is capability: int64 kernel
  templates, the Metal integer path, and `emit_fused_reduce`'s f32-only
  arm. Each sits in the CI-pinned set and closes by landing kernels and
  validating through the local HIP gate, and none has moved.
- **#1112** (owner: #729) - the extent producer/consumer split is half
  migrated. 0.18.4 widened the host C carrier to `int64_t`, but
  `chelis_gpu_tensor` deliberately did not move, so the host and device ABIs
  now disagree on extent width. This is the only #729 child that v0.18.4 made
  more urgent rather than less.
- **#795** (owner: #730) - conv2d's present-but-non-literal stride/padding
  fall to `unwrap_or(1)`/`unwrap_or(0)` verbatim (census row 23); §C1.4
  raise-or-prove applies.
- **#906** (owner: #730) - eval stack-overflow abort on ~2-4k-element flat
  list literals; the WI-1 depth-budget pattern exists to apply.
- **#730 partials** - #699 and #722 are misread if taken at their titles: the
  silent-zero half is dead (Phase 1's raise at `lower.rs:10686` cites census
  row 1 and both issues), and what stays open is compiled integer *support*,
  which is a #729 capability-table decision rather than a #730 deliverable.
  #1058 keeps that shape - the branded rejection fires correctly and the
  capability is what is missing - while #1158 exited it on 2026-08-07: PR
  #1204 landed the capability, and the recursive-generic rejection is retired
  except for the #1226 residue. Genuinely #730's: #794 (the
  `extract_f64_value` par catch-all folds the first child where spec/03 says
  last, and `extract_usize_value` maps a negative `.dp` seed to 0 - both
  checker-unreachable today, so defense-in-depth per §C1.4), #960 (the
  chelis-python input side: raw `dtype: i32` fields, hardcoded DLPack code),
  #1137 (shrink over an expand-produced broadcast view emits a rank-0 output
  when consumed downstream), #1143 (rehome executable rejection authorities
  before closing defect instances), and #1157 (a C codegen ICE at
  `dag.rs:1753` whose site is a deliberate invariant assertion, so §C7.2
  routes it to `compiler_invariant!` rather than to a rejection).
- **#731 residue** - **#850**'s build half: its check half shipped as the
  orphan-`defsig` rule in `spec/03` §2.2, and a stdlib sweep backed all three
  signature-only regions (Parquet, SafeTensors, Xavier), but a call to a
  sig-declared export with no body still lowers to an undeclared C function.
  **#874**'s tag-keyed exemption remains (role totality landed), with #887
  parented under it. **#1125** (the stamped-Node ingress-parity sweep over
  match-arm and if-let `Expr::List` readers) and **#1134** (deciding which of
  `check_ir_program` and `check_typed_program` is correct on defsig-less
  forward value references) are the other two.
- **#997** (owner: #732) - JSON I/O runtime diagnostics format payloads
  through derived `Debug`, a direct §C1.6/§B2.4 violation that owns the
  structural `FO-DIAG` migration in `faithful_observation.md` §I2. Its
  tripwire annotations account for 36 tokens (7 eval + 18 JSON + 10 CSV + 1
  shared helper); one JSON token is explicitly cfg-test-only. This is the one
  issue keeping #732 open.
- **Structural/other** - #888 (compiler size arithmetic saturates ->
  memory-planning slot reuse at wrong capacity; **deliberately standalone**,
  held in the roadmap's unclaimed ledger with its disposition prose) and #889
  (owner: #893), whose reported band the v0.18.4 int64 widening repairs while
  the mechanism survives: `chelis_alloc`'s extent fold and byte multiply are
  still unchecked `*=` with a `bytes.max(1)` launder, and release builds carry
  no `overflow-checks`, so the silent threshold moved from 2^32 to 2^63 rather
  than closing. `chelis_alloc_view` reaches the wrong-value half without
  allocating at all; #868 (zero
  `Unsupported::with_span` call sites) and #886 (hand-assembled check JSON),
  both under #883 while #730 §C2 owns the span contract in prose; #870 (prove
  SIGABRT instead of degrading), #872 (depth-32 fuse projects to
  `Type::Unit`), #955 (nested Option/List composites fail to lower, and
  nothing in CI builds std-importing programs), #957 (the lowering/emission
  panic family: ~150 production sites by the issue's own 2026-07-30 census,
  ~44 of them rejections wearing panics; a raw re-count across `chelis-ir` and
  the three backend `src` trees returns 232 matching lines today, which
  includes in-file `#[cfg(test)]` modules and so is not directly comparable -
  the population needs re-measuring before Phase 4 §C7.2 sizes its ratchet),
  #958 (the einsum `unwrap_or`, token-anchored at
  `host.rs:8843`) - all owner #730.
- **Evidence gaps that could hide wrong answers** (standalone
  deferred-evidence items) - #737 (Metal typed kernels have never executed),
  the #735 RNG confirmation re-sweep (unblocked since 2026-07-20, never run;
  #735 is one of #733's four children), and #738/#754/#763 (no shell has ever
  run a compiled binary; the cross-lane gate is speced, Linux x86-64 first,
  and unbuilt).
- **v0.18.4 fallout** - **#1200 is repaired on `main` and live in the
  release.** PR #1208 replaced the Linearity-F2 region gate with per-binding
  component tracking: coral's unmodified suite recovers 74/74 and shoals'
  `curves_basis` clears, the destructured-component carve-out finally has
  normative text (spec/04 §8.3 plus `implicit_linearity.md` - both previously
  described copy insertion with no exception at all), and the repair sits
  after the tag, so a shell on 0.18.4 still hits the regression. The repair's
  own adversarial review filed the open residue, all unparented: **#1209**
  (alias chains resolve names rather than binding generations - both
  misroute directions, pinned as `#[ignore]`d tests), **#1211** (carrier
  identity and temp-name uniqueness follow-up), and **#1212** (an authored
  `__chelis_tmp0` in an enclosing block collides with a synthesized carrier
  and hides a use-after-consume). That re-poses #1200's old ownership
  question - see pending decision 2. **#1197** is unchanged: a pipe-stage
  lambda with a chained body still fails resugaring inside
  `chelis migrate surf`, and because the migrator preflights the whole batch,
  one such file aborts the entire run - so the tool the release names as its
  migration path can refuse to migrate a tree. No class parent, no owner,
  nothing in CI can see it.
- **Filed since 2026-08-06, all unparented** - a performance family:
  **#1205** (`chelis build` front-end time ~cubic in nested entry-body size,
  ~quadratic flattened: 42.9 s vs 7.6 s for the same 160 tensor ops),
  **#1206** (compiled recursive defs retain every frame's tensor
  intermediates: a 288-frame local-vol MC exhausts >11 GB at 60k width),
  **#1207** (typechecking superlinear in binding count - `Env::generalize`
  sweeps the whole environment per binding; a 433-line shoals module takes
  67 s; the investigation and levels plan sit on the open docs PR #1227).
  Two `soundness` finds: **#1214** (the HIP backend's `fused_in_place_spec`
  is missing chelis#933's caller-storage check, and a test asserts the unsafe
  shape) and **#1224** (the SMT property runner drops a where-clause
  assumption on the named-def gradient path - flaky spurious disproofs).
  **#1222** (a compiled binary prints correct results, then aborts at exit
  with `double free or corruption`). And the #1213/#943 excavations:
  **#1217** (`chelis test` re-validates its prepared-graph cache against a
  leaked bundled-stdlib tempdir, so a stdlib edit can run false-green or
  false-red until the bundle identity joins the cache key) and **#1225**
  (`parse_line_chars` recurses per character, so a valid CSV line beyond the
  lane's stack budget crashes `try_read_csv`: eval ~4 KiB, compiled
  ~100 KiB). #943 itself is half-repaired after the tag by PR #1213
  (compiled-lane CSV ingestion is linear - 3.06 GB -> 121 MB allocated on the
  9588-row probe - and the `string_slice`/`string_len`/`list_append` costs
  are retired); its remaining half, user-written recursive `append`
  accumulators, needs a static "uniquely referenced and not read again"
  consumption fact feeding `chelis_list_push`. The PR implemented and
  disproved the cheap dynamic `refcount == 1` route by oracle comparison
  (5 of 5 adversarial programs wrong, and the asymptotics survive at half
  the constant), so the static fact is the only live route.

## Prevention map (failure mode -> mechanism -> status)

| failure mode it prevents | mechanism | status |
|---|---|---|
| A new bare-float/raw-dtype numeric surface | §C6 census + tripwire; PR #1154's permanent dispositions (baseline regeneration cannot silently bless a change) | landed; #1160 (seam-identity relocation vocabulary) is the open design question, and it recurs on every dtype-migration rung |
| Op x dtype drift, hand-mirrored lists | #729 Phase 4 capability table | opened; table proper unbuilt |
| The next silent value substitution | #730's typed channels (landed) + Phase 4 ratchet totality (§C7) | Phase 4 unstarted (#990) |
| Gate drift / duplicate gates | #730 P3 gate contract + dedup + deletion | the gates themselves are done (PRs #1175 and #1186: one shared typed policy per target, zero `fn reject_` definitions left in `chelis-cli`, the syntactic `reject_*` source manifest, #697/#698/#705/#959 closed); the recorded contract, the deletion pass, and a semantic no-duplicate tripwire remain |
| A breaking release reaching shells before anyone runs their suites | the ecosystem-drift canary (auto-files per shell) + `conform bump-check` | neither detects nor gates this. The canary checks each shell out at `main` and never runs `chelis migrate surf`, so it only ever exercises pre-migration source - and #1200 appears only after migration rewrites record patterns to the mandatory v0.19 pun, which makes the existing leg structurally incapable of seeing it. The canary has also been red continuously since 2026-08-02 - including the 2026-08-21 run - so it gates nothing in practice either. The missing leg applies a release candidate's own named migration to each shell tree and then runs that shell's suite |
| Unhandleable AST states reappearing | #731/#908: stamped-only ingress (#1088), then carrier deletion (#1029) | **nothing in flight** - PR #1036 closed unmerged, #1088 is unowned, and #1129 holds the landing inventory |
| Spec silence and stale claims | #733 end-to-end | nothing active - the weakest link, with the #891/#904 twenty-builtins-no-spec instance as its measured cost |
| Guards existing but not running | #1089's inventory of oracles outside continuous jobs; #990's scheduled/change-gated package | partial: #732's Phase 2 oracle has a dedicated blocking job and Phase 3 runs nested continuously, and #729's Phase 1/2 oracles run nested inside Phase 3. Still unwired: `unrepresentable_domain_oracle.py` (#908), `loud_unsupported_phase2_oracle.py` (#730), `compiler_pipeline_oracle.py` (its three controls run in the gate's `lint-and-unit` stage, but the oracle itself is invoked nowhere - `grep -n compiler_pipeline_oracle scripts/gate.py` returns one comment line and no call - which is not what #1089 asks for), and the new `loud_unsupported_phase3_oracle.py`, which is red on its #912 leg by design. #1090 closed as refuted - the canary already auto-files into the shell repos (coral#23, school#189, hull#14, hello-chelis#19, octant#42, hydronnx#64) |
| Silent lane divergence | #754/#763 cross-lane gate; #738 shell compiled lanes | unblocked by #732 P2, undelivered |
| Agent-driven recurrence | #740's enforcement-ladder backlog; #895 executable plan inventories | entirely unchecked, dormant |
| Wrong issue closures | the liveness manifest (detects after the fact); a keyword-auto-close guard (prevention) | the guard is missing - three incidents (#716, #729, #912), one reverted overreach (#1151/#1159), and 22 more class closures on 2026-08-05 through the same unguarded path. The 2026-08-17/21 audit-and-pin wave (evidence-comment closures for #916/#646/#986; pinned-test PRs #1230-#1232 for #862/#1084/#683) is the closure discipline done right by hand - and the same wave produced the #1083 conflict (closed against PR #1231's own record, reopened the same day), which argues for the guard, not against the wave |

## The six biggest remaining gaps, elaborated

### 1. #730's gate half - delivered; the contract and the ratchet are not

The gates themselves are in place. What `77a74ea1` measures (re-verified for
this revision; every count and anchor below is unchanged from `841b6ddb`):

- `grep -c "fn reject_" crates/chelis-cli/src/main.rs` is **0**. The CLI
  defines no rejection policy of its own; it calls the typed compiler-api
  gates (`reject_unsupported_hip_ops`, `reject_unsupported_metal_ops`,
  `reject_unsupported_effect_ops` and its host-program sibling,
  `reject_host_only_builtins_before_host_lowering`) through
  `shared_compiler_gate`.
- `reject_unsupported_hip_ops` has exactly one definition, at
  `crates/chelis-compiler-api/src/compiler.rs:3836`.
  `COMPILED_HOST_ONLY_BUILTINS` has no occurrences left in `crates/`, and
  `HOST_ONLY_BUILTINS` is a single const at `compiler.rs:3331`.
- Effect and window rejection is scoped to the DAG or host program actually
  emitted, through one shared host-helper traversal, so the host-lane
  `dropout` path reaches a branded rejection citing #1192 rather than the C
  emitter's panic boundary.
- A syntactic `reject_*` source manifest walks free functions and `impl`
  methods across both crates (`phase3_gate_inventory.rs`).

The CLI also has zero `unimplemented_rejection!` uses, which is the correct
state rather than a gap: branding lives behind the shared typed policy, so the
same program gets the same diagnostic on every entry path - which is what the
drift between the two `reject_unsupported_hip_ops` copies originally broke.

**Phase 3 is still not complete, and the oracle says so out loud.**
`scripts/loud_unsupported_phase3_oracle.py` runs six legs; the first five
(shared gate + rejected-cell corpus + gate inventory, the sealed
diagnostic-kind contract, the typed rejection-authority tests, the
rejection-authority boundary, the live issue-authority manifest) pass, and the
sixth - the #912 root-realizability integration - does not. That leg runs
`cargo nextest run -p chelis-cli --test issue_912_root_boundary --run-ignored
all`. PR #1231 re-ran the target's six ignored cells against `main`, found
four quietly green (now active, after a review round caught three of them
passing vacuously), and declared the remaining two - both `todo!`
manifest-completeness cells - in an enforced `IGNORE_LEDGER` citing #1079: an
`#[ignore]` without a ledger row, or a stale row, now fails the target. The
leg is red on exactly those two cells until #1079 lands, which is the point. Also owed: the recorded gate
contract as a written rule
(a gate may only make a diagnostic earlier or more specific), deletion of
gates that duplicate emitter rejections (proven by the rejected-cells corpus),
a tripwire beyond the syntactic manifest - PR #1175's own scope note says the
manifest does not claim to detect a semantic reimplementation under an
unrelated name - and the first-error-vs-accumulate decision affirmed before
the contract freezes. Phase 4 (#990) is separate and entirely unbuilt: the
derived-universe tripwire over the real product-source manifest (Python and C
adapters, so a substitution cannot hide in a script), the ~150-site panic
triage (#957), #958, and the two CI legs. The #1151 same-day revert argued the
delivery must ship narrow; #1175 and #1186 both did, and that is the pattern
to keep.

### 2. #729's capability table - the mechanism that retires hand-mirrored lists

Op acceptance is hand-maintained lists, backend dispatch is hand-written match
arms, conformance is hand-curated files - three places every new op must
agree. The table inverts all three: checker acceptance derived from Table A
(op x dtype x surface, atom citation mandatory), dispatch skeletons
macro-generated so a `Supported` cell with no kernel is a compile error and a
kernel with no cell is dead code, conformance generated. Part of v0.19's
decision payload is already out ahead of the table as check-time rejections
(#724, #726), and two more pieces sit on `main` as exactly the hand-curated
matrices this phase exists to generate: the checked-cast product (PR #1189, in
0.18.4) and the scalar contract (PR #1188, after the tag). Every one of those
is work the table would otherwise have to redo as derived output. Still to
decide or ship: #753 (wrap ops exist only as spec text
plus `Unimplemented` cells), #759's `cast_saturate` rung, the #689/#693 GPU
cells, #965. Entry conditions: schema freeze plus five recorded open questions
(the surface axis for container callables is the one that bites). The merged
census permanence is the floor; #1160 needs deciding so the next dtype
migration can relocate seam identities without a hand-carried maintainer
override. Known inherited holes: #951, #957, #958, #960, and the capability
table's own stale seed-row statuses (#724/#726 "not yet landed", left for this
phase to refresh).

### 3. #1088 - unowned, and everything downstream of it is blocked

The compiler-api generic Deep ingress (`parse_str_strict`) skips the
declaration-role stamp - a weaker second front door through which unstamped
trees reach consumers today. Three things queue behind it: #731's §C4.2
successor acceptance (forbidden while a second ingress exists), the ingress
half PR #1036 was carrying before it closed unmerged, and #1029's deletion of
`Expr::List`/`Atom::Tag` - the make-illegal-states-unrepresentable payoff of
the entire #908 arc, still 1,014 and 77 occurrences deep respectively. This is
the only gap on this list with neither an owner nor a PR. #1129 exists to hold
what the closed branch work landed and what it did not, and the enabling work
is done (PR #1126's 31-reader `stamped_parts` sweep; a re-salvage baseline
exists in `phase3_stamped_ingress.rs` needing the `.kind().as_str()`
adaptation). It needs an owner; nothing else.

### 4. #912 Faces 1 and 4 - half of v0.19, and now a second plan's blocker

Face 1 (existence): the manifest walkers are `Expr::Node`-blind (#1082), so a
stamped program can produce a silently empty manifest; anonymous tuple/ADT
roots still drop from the manifest - the dotted-root expansion is a live
`TODO` at `chelis-effects/src/realizability.rs:506`, owned by #1083 (briefly
closed on 2026-08-21 against PR #1231's own record; reopened the same
day). Face 4 (artifact): `requires_main` reaches one production
caller that decides nothing (`compute_manifest_json`, which decorates
`eval --json`). All three `cmd_build_c_result` sites pass `None`, leaving
`c_source.contains("int main(")` as the deciding authority, and
`cmd_build_hip_host` never takes the parameter at all. spec/05 says a
source-text scanner cannot satisfy [05-UNS-1]. The dependency order is
written: #1082 -> #1079
(consume `ManifestedProgram`, delete the source_arch guard) -> thread
`requires_main` incl. HIP -> the dotted-root expansion -> the C1 f64
build-routing. #1084's surviving ask is delivered: PR #1231's enforced
`IGNORE_LEDGER` declares the two remaining `todo!` cells against #1079 and
un-ignored the four that had quietly gone green.
Jeff's chain; #1097 landed its fail-closed disposition piece;
#730's deliverable 6 lands inside it. Its cost is not confined to its own
class: #730's Phase 3 runner cannot go green until this chain does.

### 5. Twelve defects are owned but not scheduled

A defect with a parent class but no row in a remaining phase deliverable is
invisible to every oracle: nothing turns red while it stays open. Twelve sit
there, in three grades.

**No slot anywhere.** #795, #957, #958, #960, and #874 have a parent and no
owning row. So do four more that a sweep of #730's children turns up and that
no plan doc mentions at all: **#1137** (the shrink-over-expand rank-0 answer),
**#1138** (`compile_for_execution` declining grad/vmap transform entries),
**#1143** (rehome executable rejection authorities before closing instances,
which governs the Phase 3 oracle's fifth leg), and **#1157**.

**A slot that was never written.** **#794** is covered only by a roadmap line
calling it "#730 census extension rows", and `loud_unsupported.md` has no §C5
row for it at all. The citation points at a row that does not exist.

**A slot with no citation.** **#990** is covered by §C7.5 as a contract, but
that section never names the issue, so nothing links the two mechanically.

Each needs one of three things - a named phase or kill-table row in its owning
plan, an assignment to a v0.19 capability decision, or a recorded "standalone
fix, no phase dependency" line like the ledger already carries for #681 and
#888 - so that its delivery story is inspectable rather than implied by
parentage. #699 and #722 leave this list not because they gained a slot but
because their live remainder is #729 capability work.

The list is short because one two-step move clears these reliably, and it is
worth naming because it is needed again: give each unscheduled defect a named
home in its owning plan, then execute the homes as code. The homes for #729
live in the plan docs PRs #1167 and #1174 authored (docs-only, nothing else)
and in the code PRs #1181, #1188, and #1189 carry; #731's PP1/PP2 sit in
`checker_totality.md` beside the change set that implements them (PR #1178);
`loud_unsupported.md` carries LU1-LU6 for #870, #872, #906, #955, and #1150 on
the same pattern.

What that pattern produces is class mechanisms rather than point fixes, in
every instance so far: #851's fix is a single shared binder walk, #1147's is a
mandatory inference disposition on every `BuiltinDecl`, and #1150/#1152's is
one `CheckedCastPlan` over the whole source x target product - not a few added
match arms. The exception is the lane-level work: #690 and the pinned support
cells #689, #951, and #759's HIP cast half close only by landing kernels and
validating through the local HIP gate, and none of them has an owner.

### 6. The institutional layer - specified, largely not running

#733's Phase 0 is self-contained and Chelis-owned, so its dormancy has no
external excuse; the measured cost is visible in the #891/#904 instance
(twenty builtins and a prelude ADT with zero spec entries). Its four children
are #735 (the unauthored `with seed` / `with device` semantics), #797 (adopt
the central OpenSpec governance gate), #895, and #898 (four reference-only
reduce builtins); none has moved. #740's backlog is two items of eight, and
neither of the two binds anything: the mechanism index is hand-maintained and
the one per-crate guardrail covers one crate. #1089's
unwired oracles are empirically proven gaps - a deleted assertion produced no
CI failure while the unwired oracle caught it instantly - and three of the six
remain outside any continuous job. The keyword-auto-close guard is owed and
unowned after three incidents and one reverted overreach; the liveness
manifest only detects a wrong closure after the fact, on an unrelated PR's CI,
and all 22 of the most recent closures sit on that unguarded path. And
#754/#763/#737/#738 leave whole lanes evidence-free: Metal has never executed
a typed kernel, and no shell has ever run a compiled binary. #1200 is what
that costs - a linearity regression that shipped in 0.18.4, that only coral's
own suite could see, and whose repair (PR #1208) had to be driven by
downstream breakage rather than by any gate.

Ranked by leverage: give #1088 an owner, wire the remaining oracles, then
write down the gate contract that the two shipped slices now make cheap to
state. All three are small relative to what they unlock, while the capability
table and the #912 chain are the two real campaigns left before v0.19/v0.20.

## Open PRs

Six PRs are open besides the docs-only change carrying this report, and what
each retires when it lands:

| PR | what it retires | state |
|---|---|---|
| #1235 | Drops the superseded stamp-shell and aligns the python3 policy with the scripts | open |
| #1233 | `bump_compiler_pins.py` regenerates the checkpoint compile-fail fixture's lock on a version bump (part of #1128; #1234 holds the sibling pipeline-artifacts fixture gap) | open |
| #1227 | Publishes the #1207 typecheck-superlinearity investigation and levels plan (docs-only; fixes nothing) | open |
| #1203 | Commits the crate2nix graph and composes shared devenv tools | open |
| #1161 | Keys compiled-context caches on the build fingerprint rather than the release version: closes #1156, the one parentless issue the graph audit found | open |
| #838 | Conform central workflow wrappers (#788 family, Chelis-Lang/ci#5 prereq) | draft |

None touches the numeric-remediation plan set: they are cleanup, tooling,
cache, and conformance work, and between them they retire one open issue
(#1156). **Nothing in flight addresses #1197 or #1088**: the surviving
post-release regression and the unowned keystone are unstaffed, and so are
both remaining campaigns. (#1200 and #1201 left that sentence the right way -
repaired, by PRs #1208 and #1215.)

## Pending maintainer decisions

1. **What v0.18.4 did to anti-churn invariants 7 and 1 - two different
   questions, not one.** Invariant 7 makes an exported-signature change "0.19
   payload **by default** ... never a cut promised 'mechanical'". 0.18.4 was
   not promised mechanical, so the absolute clause held and only the default
   was overridden, by a release decision rather than an amendment here.
   Invariant 1 is a *scope* question, not a crossing: it governs #729 §C3's
   storage decision, and WireDag 4 -> 5 sealed the `Pad` carrier instead - a
   different surface that §C3's storage has still not followed. So the honest
   statement is "one default overridden, one scope question unanswered", and
   the call is: amend invariant 7 to say what governs an ABI change in a
   breaking patch cut (or record 0.18.4 as its stated exception), and decide
   whether invariant 1 reaches the WireDag schema at all.
2. **The linearity residue's owner and class.** DECIDED 2026-08-21: #1209,
   #1211, and #1212 are parented under #731 and delivered as one class
   change, PP3 in `spec/design/checker_totality.md` (the name-keyed
   binding-identity channel). The normative rule landed as [04-LIN-1] and
   [04-LIN-2] in `spec/04-type-system.md` §8.3; the delivery branch is
   `agent/1209-binding-generations`. (Original question: #1200 was
   repaired without ever getting an owner or class - PR #1208 landed
   classless - and its adversarial review filed the three issues
   checker-shaped and unparented while #731's oracle stayed green.
   #1209's binding-generation identity was the durable-fix shape PR #1208
   itself named.)
3. **The performance family's home.** #1205, #1206, and #1207 are the same
   shape from a lane no tracker owns: superlinear compile-path costs
   (front-end lowering, compiled recursion's memory retention, and
   `Env::generalize`'s per-binding environment sweep), each `area:perf`,
   each parentless. #1207 already has a levels plan on PR #1227. Decide
   whether they get a tracking hub (the one-tracking-issue-per-class rule
   fits: a recurring cost class with a shared oracle shape) or stay
   standalone ledger entries like #888.
4. **The graph-shape questions the audit surfaced**: confirm the #1023/#1029
   hub-with-a-parent arrangement is intended. (#895's second home is already
   recorded by comment, so nothing is owed there.)
5. **#1160** - the census seam-relocation mechanism; the issue itself asks
   for a dedicated red-team pass because it loosens a ratchet.
6. **spec/04 [04-NUM-4]/[04-NUM-5]** - their "not honored" parentheticals
   were removed on a negative search (no open bool-arithmetic or
   `fold_static_cond` issue found); if #892's Metal bool storage is judged
   inside [04-NUM-4]'s scope, restore that one parenthetical citing it.
7. The **§C5 census row 26 site-column anchors** in `loud_unsupported.md`
   are stale but frozen under B1; re-anchoring them is a deliberate
   baseline change, recorded in the PR #1163 body.
8. **The owned-but-unscheduled residue** (gap 5): #795, #957, #958, #960,
   #699, #722, #794, and #874 each need a named delivery slot.
9. **Which Surf follow-ons ride the v0.19 cut.** The canonical grammar itself
   shipped at 0.18.4, so what needs a call is whether #1171 (typed
   first-argument application-to-pipe promotion, explicitly not delivered by
   #1031), #1172 (structural spans, parented to #883), #1179, and #1180 ride
   v0.19 or a further patch - and whether #1197 blocks the migration path
   enough to warrant a 0.18.5. Note the calculus has shifted: `[Unreleased]`
   now carries the #1200 repair, the recursive-generic capability, and a
   BREAKING check-time polymorphic-recursion rejection, so the next cut is a
   migration-bearing release whether or not #1197 makes the bar.
10. **#1170's relationship to #730 Phase 4.** The C-lane capability tracker
   inventories the same hand-maintained exclusion lists that Phase 4's ratchet
   totality is meant to derive. Decide whether it is a Phase 4 acceptance cell
   or an independent surface before both build one.

## Standing process conditions

These are properties of how the repo currently works, each with the instance
that measures it.

1. **Nothing prevents a keyword auto-close.** A squash body's "Fixes #N"
   closes the issue whether or not the fix is complete, and the only guard is
   the liveness manifest's after-the-fact detection on an unrelated PR's CI.
   #716, #729, and #912 each sit on that history, reopened after the fact; the
   one mitigation attempt (#1151) is reverted as overreach, and the 22 class
   closures of 2026-08-05 rest on the same unguarded path - as do the August
   repair closures (#1158, #1200, #1201, #1216), each backed by a green
   oracle, which is the luck the guard would replace with a check.
2. **Merge governance holds; the verification habit is what backs it.** The
   #1096 rollout (no-bypass, squash-only, green up-to-date checks) governs
   every merge, which is why the 8/1-8/2 pattern the sprint audit counted - 22
   merges with zero approving reviews, several carrying self-reported
   verification later found false - cannot recur in that form. What actually
   catches the residue is execution over inspection - the discipline that
   refutes two claimed bugs before code is written and that catches the #730
   Phase 3 plan-doc over-claim inspection accepts.
3. **The filtered CI profile can hide a deterministic failure.** #1103 fixed
   #1095, but the fail-fast plus filtered-profile pattern that hid it is
   unchanged. One #749 lock cell
   (`c_nested_tensor_truncates_at_32_with_marker`) still skips silently
   without a C toolchain while every sibling panics - the only cell in its
   group that can vanish without turning anything red.
4. **Nothing enforces the close step when a fix lands.** Closures now carry
   evidence comments, but that is a convention, not a gate: the validation
   pass found ~29 fixed-but-open issues, several with threads describing their
   own fix as still in flight, and the plan docs carried an unperformed
   instruction ("close the issues on the PR #863 merge") for weeks. The
   2026-08-17/21 audit wave is the first systematic counter-move: fixed-but-
   open issues re-verified against `main` and closed with evidence comments
   (#916, #646, #986) or with the owed test landed first (#862 via PR #1230,
   #1084 via #1231, #683 via #1232). Still a convention rather than a gate -
   and the same wave shows the cost in the other direction: #1083, closed
   against PR #1231's own record and reopened the same day.
5. **An oracle that reports FAIL outranks a phase claim that reports done.**
   Both current examples are self-reports corrected by execution: #732's Phase
   2 command failed on `main` because a zero-cell ledger has no real
   classifier invocation to receipt, and #730's Phase 3 runner exits nonzero
   on the #912 leg rather than scoping the leg out. Both behaviors are
   correct. The hazard is the converse - an oracle nobody runs proves nothing,
   which is #1089.
6. **Release notes drift, and the guard against it is unbuilt.** v0.18.4 had
   to author its own section from the commits (`[Unreleased]` was empty and no
   commit since v0.18.3 touched `CHANGELOG.md` - 21 PRs, zero entries, per PR
   #1199's account, and `git log v0.18.3..v0.18.4 -- CHANGELOG.md` returns
   only the release commit), and to reopen the released `[0.18.3]` section,
   which documented 1 of its 20 commits and omitted a breaking change
   (chelis#1130's `shape()` return to int64, after 0.18.2's notes had told
   readers not to migrate onto the int32 form). Three consecutive releases now
   show the same drift. The current cycle breaks the pattern so far: every
   post-tag behavior PR (#1204, #1208, #1213, #1215, #1218) landed its
   `[Unreleased]` entry in the same change set, including the BREAKING
   polymorphic-recursion note with its reproducer. The proposed guard - **CI asserting that a PR diff
   adds no changelog line below the first `## [` header** - catches both the
   missing-entry case and the chelis#945 case where a rebase moves an entry
   into an already-released section.
7. **The plan set's oracles cannot see the ecosystem.** Every phase oracle in
   these five plans is repo-internal, so a change can pass all of them and
   still break a shell. #1200 is what that costs: a linearity regression that
   clears every gate the release ran and takes coral from 74 passing to 17,
   repaired main-side by PR #1208 only after coral reported it.
   #754/#763/#738 are where a shell-suite leg belongs, and "unblocked,
   undelivered" is no longer a theoretical position.
8. **The sub-issue graph is trustworthy as the assignment record, and it is
   not self-maintaining.** The roadmap's ledger says outright that "assignment
   now lives in the GitHub sub-issue graph", and the graph currently earns
   that: one missing link (#713) and three placements the prose described
   wrongly (#868, #886, #887) across every class-owned issue in these two
   documents. Keeping it that way costs roughly twenty API calls
   (`/issues/<tracker>/sub_issues` per class, `issue.parent` per claimed
   child) and belongs in the same automation that will eventually guard
   keyword auto-close.
