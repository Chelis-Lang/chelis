# Chelis Numeric Remediation - Status (2026-08-24)

This is the state of the numeric-remediation board. The execution basis for
this revision is `main` at `4e200061`. The shipped release is **v0.18.5**
(`fc3232ff`); `main` is eight commits ahead.

**One class is closed.** #732 (faithful observation) closed on 2026-08-21 on
its own stated condition, and its META #728 closed with it - the first class
closure since the plan set was authored in 2026-07. The board is four active
classes plus a closure record from here on; the #732 rows below say what the
class left behind rather than what it still owes.

v0.18.5 contains the post-v0.18.4 repair wave: scalar alignment, gate and
reshape fixes, pipeline-core extraction, recursive-generic build support,
linearity and CSV repairs, the #912 test wave, the rustc 1.98.0 pin,
integer-literal type rejection, the `chelis migrate surf` repair, authored `>`
operand order, the FO-DIAG boundary that closed #732, the cache-fingerprint
fix, and #731 PP3. None of those remains an unreleased main-only repair.

Every count and code claim here was measured on `841b6ddb` for the 2026-08-05
revision, re-measured on `254d6070` for the 2026-08-22 revision, and refreshed
against `4e200061` wherever this revision changes it. The command is
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
Wave-4 slots, **#732 is closed** - its Phase 3 delivered in v0.18.3 and its
last direct debt (#997) retired by PR #1250 - while #729 Phase 4A's
census-permanence slice is landed, Phase 4B's language/schema freeze is this
change, and the 4C-4E table/consumer/conformance work is unbuilt. #730
Phase 3 has its gates but not its contract and Phase 4 is unbuilt, and #733
has nothing.

After this change closes #898, the class trackers carry **54 open children**:
#729 has 27, #730 has 17, #731 has 7, #732 has none, and #733 has 3. The live
pre-merge graph is one higher because #898 remains open until this PR merges.
The #729 prerequisite split now names first-class count (#1287/#1291), the
zero-exception census (#1288), balanced reductions (#1290), own-width tensor
comparison (#1292), stdlib semantic alignment (#1293), exact builtin-atom
closure (#1294), all-active-dtype builtin parameters (#1295), the exact
v0.19 child set (#1296), legal compiled host effects (#1297), dynamic
axis/window execution (#1298), and direct typed subtraction/extrema (#1306),
rather than hiding those deliverables inside one phase checkbox.

`docs/CHELIS_SURFACE.md` remains the shipped-behavior guide while these
contracts are implemented; this planning/specification change does not rewrite
current behavior into future tense. The implementation children own the exact
guide updates in the same code/test/example changes: #1287 owns count and
WireDag v6, #1289 owns the tagged C carrier, #1293 owns the 83-definition
stdlib and sole JSON/CSV surfaces, #1295 owns numeric builtin parameters,
#1297 owns compiled host effects, #1298 owns runtime shape/movement/window
execution, and #1306 owns direct checked subtraction and stored-bit extrema
selection. Those existing guide statements are
implementation divergence, not compatibility authority or permission to
narrow the numbered specs.

This revision supplies #1306's direct `Sub`, `MinElem`, and
`ExtremaAdjoint` identities and the authoritative child command
`.venv/bin/python scripts/dtype_direct_arithmetic_oracle.py`, whose success
line is `DTYPE DIRECT ARITHMETIC ORACLE: PASS`. The executable normal-gate
surface covers exact typed kernels, IR/eval/AD, folding, WireDag v6, typed
target dispositions, compiled C behavior, HIP source generation, exhaustive
consumer compilation, and standing mutations. The four ignored HIP execution
cases remain a real-device manual gate:
`scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness direct_ -- --ignored --test-threads=1`;
expected success is four passed and zero failed. The current Darwin arm64
validation host has neither `hipcc` nor the documented ROCm wheel paths, so
that hardware receipt is **BLOCKED**, not passed or waived.

Everything still open falls into four groups: a defect inventory concentrated
in #730's support cells and #729's GPU lanes, two engineering campaigns (the
capability table and #912 root-boundary chain), the post-#1088 stamped-carrier
follow-ons now that PR #1285 has landed, and an institutional enforcement
layer that is specified and largely not running.

The weak point is downstream. v0.18.4 broke four boundaries in a patch cut -
canonical Surf, the int64 C ABI, declaration contracts, and WireDag schema 5 -
and exposed both a linearity regression (#1200) and a broken named migration
path (#1197). **v0.18.5 now ships both repairs** along with the rest of the
post-tag wave.
The same main-side wave also retired coral's other build-lane wall: recursive
generic host calls compile via bounded memoized monomorphization
(#1204/#1215/#1218), and `Coral.Frame` now builds, links, and runs against the
eval lane. Nothing in CI could see any of these; all were found downstream.
The remaining downstream risk is the planned v0.19 behavior/storage cut, not
an unshipped repair release.

## Scoreboard

| plan | landed | remaining |
|---|---|---|
| **#729 dtype semantics** | P0-P3 + §C6 census, all CI-wired (PRs #758, #956, #1033, #1049, #1054, #1065, #1118); the dtype ledger landed in PR #1181; [04-NUM-15] was authored in PR #1169 and enforced by PR #1189; scalar numeric families aligned in PR #1188 and shipped in v0.18.5. This change is Phase 4B: [05-OP-1..41], [04-NUM-16], spec/06's recursive List/ADT cotangents, reduction tie/NaN/infinity rules, direct stored-bit extrema selection and checked subtraction, canonical balanced sum/product/count trees, canonical gradient consumer-edge order by forward node ordinal and input slot, canonical scalar/tensor/recursive-List `to_string`, byte-exact runtime List/tuple/Dict/ADT observation, bool-only logical truth tables, exact NaN-aware comparisons, all-active-signed sparse indices across IR and public C, first-class multi-axis `count`, exact-only WireDag v6 numeric fields, the sole public JSON ADT, exact `IO` effect spelling, and zero-exception external-family semantics are frozen; `dtype_phase4b_oracle.py` is the authoritative freeze check | **Prerequisites before 4C:** #893/#1289 seal tensor access and replace the C ABI with exact tagged carriers; #1288 replaces every frozen capacity disposition with a total final authority map; #1290 lands the balanced reduction trees; #1287 lands `count` in eval/C and exact-only WireDag v6 decoding, while #1291 owns loud HIP/Metal count cells; #1292 removes the tensor-close f64 funnel; #1293 aligns all 83 recursively discovered stdlib numeric definitions, removes duplicate JSON/prelude identities, and removes the unimplemented generic `sample` export; #1294 proves exact atom authority for the union of every canonical Table-A IR/RISC identity and every sibling builtin declaration; #1295 implements the all-active-dtype parameter contracts; #1296 locks the complete v0.19 child set; #1297 implements the legal compiled host-effect builtins; #1298 implements runtime axes and target-independent reduction windows; and #1306 implements direct typed subtraction/extrema without arithmetic surrogates. No 4C key/cell type, macro, or partial row may land before #1294 and #1296 are green and merged, and no authored cell may claim support before its behavior prerequisite is executable. **4C:** populate Table A/B, host-constructor and sibling-builtin registries, external target dispositions, and exact effect-disposition rows, with authoritative `dtype_phase4c_oracle.py`. **4D:** derive checker/reporting, backend dispatch, build gates, host casts/ABI, #912 projections, effect lookup, and exported-stdlib dependency closures, with authoritative `dtype_phase4d_oracle.py`. **4E:** generate the complete executable product and final `dtype_phase4_oracle.py`, then fresh red team. No compatibility wrapper, reader, cast-counting idiom, unnumbered authority, grandfathered behavior, or capacity exception is accepted. #735 retains the unauthored device-selection meaning; its resource cases cannot default to support |
| **#730 loud unsupported** | P0-P2; P3's typed `RejectionAuthority` + sealed 48-kind `DiagnosticKind` slices (PR #1037); P3's gate contract in two slices - PR #1175 (one typed compiler-api policy behind both public build paths, the `reject_unsupported_hip_ops` pair deduped, `COMPILED_HOST_ONLY_BUILTINS` deleted, the C precision preflights removed, a syntactic `reject_*` source manifest over both crates) and PR #1186 (Metal and seeded-effect policy moved into those same typed definitions, one shared host tensor-helper traversal, entry-scoped pure-DAG effect rejection, the single Phase 3 runner). Closed #697, #698, #705, #959, and - via LU6 in PR #1189 - #1150, whose `_ => input` host-emission fallback is deleted. Closed #1158 via PR #1204 (after the tag): recursive generic host calls compile through bounded memoized monomorphization under the new [04-INF-2]/[04-INF-3] uniform-recursive-instantiation atoms (spec/04 §3.1.1) - a BREAKING check-time rejection of polymorphic recursion, lane-uniform - with PRs #1215/#1218 extending the same specialization to non-recursive generics (#1201) and recursive erased-dim generics (#1216), and the surviving fail-closed residue re-cited to the open tracker #1226. This revision makes the sixth Phase 3 leg executable: the integrated #912 manifested-root boundary | **Phase 3's executable oracle is green in this revision but completion awaits the fresh exact-head red-team pass**: `scripts/loud_unsupported_phase3_oracle.py` passes all six legs, including the #912 root-realizability interlock. Outside Phase 3, still owed: the recorded gate contract as a written rule, deletion of gates that duplicate emitter rejections, a no-duplicate-gates tripwire beyond the syntactic manifest, and the first-error-vs-accumulate decision affirmed before the contract freezes. Phase 4 (ratchet totality, #990) is entirely unbuilt - no phase-4 oracle script, no nightly workflow |
| **#731 checker totality** | P0-P3 implemented, incl. the construction gate (PR #1101) and authoring-scope fixes (PR #1136); **PP1 + PP2** (PR #1178): the function-parameter typing channel under [04-INF-1], the writeback information-ordering gate, the atom/primitive agreement matrix under [04-LIT-1] and `spec/03` §6.4, the orphan-`defsig` rule in `spec/03` §2.2, and the registered-builtin arm tripwire - closing #780, #783, #847, #851, #1131, #1147. PR #1285 closed #1088 by moving every compiler-API Deep ingress onto the declaration-role stamp, authoring [03-PROG-1..3], and wiring the #908 oracle continuously | The decode-once rework's fresh-context red team is named pending in `checker_totality.md`'s header, but the round ran on PR #855 (an exact-head architectural pass and an explicitly fresh local-subagent pass, both FAIL, findings folded before merge), so that line is stale text rather than unmet work; #850's build half; #1125's stamped-Node ingress-parity sweep; #1134's ingress-divergence decision; #874's tag-keyed exemption with #887 under it; carrier deletion last (#1029, whose stated precondition - no producer exists - is false: 1,024 `Expr::List` and 77 `Atom::Tag` matching lines under `crates/`, and the `Node::to_list` bridges plus `chelis-surf`'s desugar output are live producers) |
| **#732 faithful observation - CLOSED 2026-08-21** | All four phases (P3 via PRs #1099/#1115/#1118, shipped in v0.18.3); both known-red ledgers empty (`KNOWN_RED_CELLS` is `()` at `faithful_observation_phase2_oracle.py:159`, and `C_LANE_EXCLUDED` has no production occurrences left); tolerance table + #687 handshake shipped; Phase 2's empty-ledger acceptance repaired and continuously wired by PR #1185; and the last direct debt - #997's structural `FO-DIAG` diagnostic-rendering migration - retired by PR #1250, which is the tracker's own stated closing condition. META #728 closed with it | Nothing, and the closure says so in a form that can be checked: zero open children at close, both owning documents (`faithful_observation.md`, `remediation_roadmap.md`) audited by full read, and every live thread touching this class's artifacts assigned elsewhere by name - own-width eval tensor digits to #729's v0.19 D1 step, the `OBSERVATION_EXIT_SURFACES` derived-universe revisit to #730 Phase 4 §C7.1, root set/order/artifact acceptance to #912/#1023, the shell-invokable comparator gate to #754, the bool decode exception to #894, `chelis_format_shortest` hardening to the #729/#893 pair, and the scheduled full-matrix package to #990. #1059's C-host `to_string` is separate capability work and was re-homed under #1170 before the close, not left dangling. What outlives the tracker are the recurrence guards: the three no-third-formatter tripwire classes, the round-trip harness, and both phase oracles - Phase 2 as its own blocking job, Phase 3 nested inside the `Dtype Phase 0-3 Oracle` job, both on every non-docs-only PR |
| **#733 spec provenance** | #898's missing numeric-operation authorities are authored by this change, but no provenance phase is activated: no oracle suite, PR template, `governance` gate stage, or `spec/**` CODEOWNERS coverage | P0 remains Chelis-owned and self-contained; P1-P3 additionally wait on three Buoy prerequisites (the pinned revision's `devenv test` oracle, the `buoy.adapter-sdk/v1` pin, and approved host parsed-item schema). The remaining children are #735, #797, and #895. v0.20 is formally blocked on P3 |

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
  declaration-role stamp, plus the third non-strict `parse_deep` door at
  `compiler.rs:2903` that its body did not name - was closed by merged PR
  #1285, which migrated every door, added an ingress-parity oracle, and wired
  #908's behavioral oracle into a continuous job. #1029 is next behind it and
  is unblocked by the #1082 delivery below.
- **#912** (root boundary): the integrated #1079/#1082/#1083 delivery is in
  this revision. Manifest discovery walks stamped `Expr::Node` directly;
  `ManifestedProgram` is the compiler-API observation authority; Tensor roots
  receive their own DAG-derived input closure; statically fixed tuple/ADT
  roots expand structurally; C-target f64 roots route Host without narrowing;
  and C/HIP artifact kind comes from `requires_main()` with no emitted-source
  scan. Effectful nullary declarations are never auto-applied, variable-size
  built-in containers remain bare, and selected parameterized callables owe
  only the reachable inputs in their concrete DAG. Missing or unbound owed
  roots fail through [05-UNS-1] before a partial result or artifact. The two
  former `todo!` completeness cells are active, the ignore ledger is empty,
  and all seventeen `issue_912_root_boundary` cells pass. That
  unblocks #1029's carrier deletion and #730's Phase 3 integration leg. The
  complementary #1102 delivery now distinguishes a missing adjoint whose
  output-reachable dataflow is complete (an exact shaped zero) from an
  output-reachable unresolved callable result (the existing rootless
  placeholder). For #1148, spec/04 section 4.3 and [05-OP-33] require an f32
  matrix trace to have the rank-zero carrier `tensor[f32]`; current inference
  already has that shape. The #1148 change adds explicit f32 tests and corrects
  fixture signatures without changing trace implementation. It does not claim
  full per-dtype [05-OP-33] conformance.
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
- **#1288** (capacity-census final authority): this change lands the shared
  exact classifier and closes the exception path for every new or changed
  identity. Eighteen runtime descriptors are now structurally `Nonnumeric`,
  `chelis_tensor_shape` and Count's wire field are exact registered numeric
  operations, and the generic maintainer-override parser/path is gone. This is
  a foundation, not closure: 180 primary, 84 wire, and 17 binding rows remain
  in sealed foundation-era legacy cohorts for the immediate migration work.
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
  recursive-generic rejection (retired main-side by #1204/#1215/#1218), the
  quadratic compiled-lane CSV ingestion (repaired main-side by #1213), and the
  #1197 migrator batch abort (repaired main-side by #1243). `[Unreleased]`
  on `main` carries **three** BREAKING entries, which is why the 0.18.5 cut
  PR #1256 prepares is source-migrating several times over (decision 9):
  polymorphic recursion rejects at check time under [04-INF-2]/[04-INF-3]
  (#1204); an integer literal in a bare type position is a parse error rather
  than a fresh type variable (#1240, closing #1179); and `>` evaluates its
  operands in authored order, which changes the observable effect order of any
  program whose comparison operands carry effects (#1245, closing #1180). Two
  of the four boundaries land on the roadmap's anti-churn invariants; see
  pending decision 1 for what that does and does not settle.
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

### What the class trackers hold

Of the 139 issues parented to the five trackers, 54 remain open after this
change closes #898. #733 is the only class with nothing closed under it before
this merge, and #732 is now the only one with nothing open.

| tracker | open / total | open children |
|---|---|---|
| #729 | 27 / 65 | #689, #690, #693, #753, #759, #892, #951, #965, #1091, #1092, #1112, #1160, #1281, #1282, #1284, #1287, #1288, #1290, #1291, #1292, #1293, #1294, #1295, #1296, #1297, #1298, #1306 |
| #730 | 17 / 37 | #699, #722, #794, #795, #870, #872, #906, #955, #957, #958, #960, #990, #1058, #1137, #1138, #1143, #1157 |
| #731 | 7 / 25 | #850, #874, #1125, #1129, #1134, #1247, #1264 |
| #732 (closed) | 0 / 8 | - |
| #733 | 3 / 4 after this change | #735, #797, #895 |

Two of those rows moved for reasons the count hides. #731 absorbed and closed
#1209, #1211, and #1212, then gained #1247 and #1264 as new live children.
#733 loses #898 only when this change merges. #732 lost a child without closing it:
#1059 moved to #1170, whose subject - compiled-lane capability - is what
#1059's acceptance actually is.

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
  on top of #1132's [04-NUM-10] disposition and the earlier phase work.
  **PP3** (PR #1254) closes #1209, #1211, and #1212, and it is the class's
  cleanest closure so far because the fix is one mechanism rather than three
  patches. The checker had kept alias links, consumption marks, and
  destructured-component identity in maps keyed by the variable's *name
  string*, so an alias could cross a binding generation and land on the wrong
  binding. Every binding event now mints a generation id and a use-site name
  resolves to its innermost live generation exactly once, which makes both
  #1209 misroute directions - the false accept with wrong blame, and the false
  reject - unrepresentable rather than patched shape-by-shape, and turns
  #1212's authored-`__chelis_tmp0` collision into ordinary shadowing that can
  no longer change a verdict. The rule is normative, not incidental:
  [04-LIN-1] (binding identity) and [04-LIN-2] (the deliberately preserved
  two-closure capture verdict) in `spec/04-type-system.md` §8.3. Two entries
  do not mean what a closure usually means: **#847** is recorded
  non-reproducing at the PP1 baseline on PR #1178's word (its own thread
  records no such verification), and **#850** stays open because only its
  check half shipped.
- **#732.** All eight. Seven (#716, #723, #748, #749, #775, #1078, #1104) come
  from Phase 2/3 and the oracle hardening around them; the eighth, **#997**, is
  the FO-DIAG migration PR #1250 landed, and closing it closed the tracker.
  That closure is the class's own recurrence answer rather than a cleanup: the
  migration deleted `truncated_debug` instead of relocating it, so the JSON,
  CSV, and eval runtimes reach text through one boundary written beneath
  [05-OBS-1]'s `render_value`, and the tripwire baselines shrank in the same
  change set (eval.rs 31 -> 24, json.rs 18 -> 2, csv.rs 10 -> 4, mod.rs's row
  deleted). The harm was live, not hypothetical: `half::f16`'s `Debug` forwards
  to `to_f32()`, so a stored `f16` that every exit rendered as `0.1` appeared
  in a diagnostic as `0.099975586` - one value, two answers, which is the
  §B2.4 harm itself. **It also corrected a count this board and the design doc
  both carried**: of #997's 36 attributed tokens, 23 were product and **13**
  were `cfg(test)`, not the single one the JSON annotation claimed. Both
  documents had propagated the annotation's own number rather than re-deriving
  it, which is the ordinary way an inherited figure goes wrong.

Two issues outside the classes belong to the same story. **#796** is closed on
the maintainer call its row asked for, citing spec/05 §3.6.1's eval-only
`test_*` contract; the compiled-assertion request it carried lives at #1170.
**#862** closed on 2026-08-21 - the cheapest close on the board got made: PR
#1230 landed the named unit-valued-root regression test over three
sole-print-root programs, avoiding the three vacuity traps its disposition
notes documented.

**Two caveats on that 38.** All 22 of the 2026-08-05 class closures rest on a
PR-body keyword, the same mechanism that mis-closes #716, #729, and #912, and
the guard named in the prevention map is still owed (#1158's 2026-08-07
closure rode the same path, legitimately - PR #1204's oracle is green). And a
small number of open children is not the same as a small amount of open work:
**thirty-two** open issues now sit outside the class trackers, parented to
#1024, #883, #1170, or nothing. They fall into three filing generations.

**Ten predate the 2026-08-05 revision**: #1168, #1170, #1171, #1172, #1177,
#1182, #1183, #1184, #1192, #1198. Three left the group this revision by
repair - #1179 and #1180 (PRs #1240 and #1245, each a BREAKING correction) and
#1197 (PR #1243) - joining #1190/#1191/#1193/#1194 and the earlier
#1200/#1201.

**Eleven were filed between that revision and this one**, all still
parentless: the performance family (#1205 front-end superlinearity, #1206
recursive-def tensor retention, #1207 superlinear typechecking), two
`soundness` finds (#1214 HIP fused-in-place missing chelis#933's
caller-storage check, #1224 the SMT runner dropping a where-clause
assumption), #1222 (compiled-binary abort at exit), the #1213 excavations
(#1217 stale-stdlib false green, #1225 per-character CSV recursion wall), the
fail-closed residue tracker #1226, and two CI items (#1221, #1234). #1209,
#1211, and #1212 left this group the way the others should: they gained a
class (#731), and then that class shipped them.

**Eleven more were filed on 2026-08-21 alone**, all parentless, and they are
mostly the residue of the same day's five merges. From the #1197 repair, a
migrator/resugarer family: **#1241** (the v0.18 first-argument pipe-lambda
alias is never rewritten, so files using it cannot migrate at all),
**#1242** (a lone-argument call-first stage resugars to `f()`, which is a
different program), and **#1246** (the call-first stage strip frees a
repeated stage parameter and captures it in the enclosing scope) - the last
two both `soundness`. From the #1179/#1180 repairs, a checker pair:
**#1247** (integer type-application arguments are unenforced - wildcard on
type parameters, unconstrained on dimension parameters) and **#1248**
(comparison builtins disagree across the linearity and host typing tables,
so `<` consumes where `>` borrows). From the #1156 repair, **#1249** (the
prepared-reef-graph cache is keyed on the release version, not the compiler
build - the same class the fix just closed, one cache over) and **#1252** (no
regression test pins the cdylib image-resolution path that fix depends on).
And four process items: **#1237** (`rust-toolchain.toml` pins 1.98.0 while
the pinned rust-overlay tops out at 1.97.1), **#1244** (devenv hook
installation from a worktree corrupts the shared `commit-msg` hook for the
whole clone), **#1251** (the changelog-fragment decision, standing condition
6), and **#1253** (re-evaluate the `[profile.dev.package.sha2]` opt-level
bump).

The class-shaped-work-with-no-class observation that #1200/#1201 exposed now
has one worked answer and two open cases. The answer is the linearity residue:
#1209/#1211/#1212 became #731's PP3 on 2026-08-21 and closed on 2026-08-22 as
one mechanism, not three fixes. Still unanswered: the performance family
(#1205/#1206/#1207), and now the resugaring family, where #1242 and #1246 are
two faces of one defect in how a pipe stage's argument positions survive the
round trip, with #1241 adjacent in the migrator that consumes it. Both are
pending decision 3.

### The CI-pinned set (19, source-derived as of 2026-09-13)

`spec/design/loud_unsupported_issue_manifest.json` is generated from the exact
literal issue identities cited by production `unimplemented_rejection!`
invocations. Cargo supplies one package/target view for every repository-local
workspace member, including members outside `crates/`; the existing `syn`
inventory owner follows each non-test library or binary-like target's
production module graph, including literal `#[path]` sources outside the
target directory. One `test=false` source view excludes test-only items,
statements, expressions, arms, fields, arguments, and generic parameters
before include, macro, or module-wiring validation. The direct-construction
boundary consumes that same view. Rustc dep-info selects the same Cargo package
identities and is then independently reconciled against the graph through the
repository's existing configuration-closure parser.
Its current set is **#600, #689, #729, #759, #829, #879, #912,
#951, #1058, #1059, #1138, #1192, #1277, #1298, #1306, #1364, #1383, #1482,
and #1844**. Repeated citation sites collapse to one row; a missing cited row,
an uncited stale row, or a dynamic/malformed issue argument fails the
generator. The first regeneration removed #1291 because no production
rejection cites it; #1291's remaining hardware work and closing condition are
unchanged and no longer masquerade as rejection-construction authority.

Every `.rs` edit and every `Cargo.toml` edit anywhere in the repository
conservatively triggers Rejection Authority Liveness, which first rechecks
structural citation/generated-byte agreement and then checks every standing
row, so closing any listed issue reddens that run. The #1870 changed-row PR
narrowing and scheduled
standing-state canary are not part of this prerequisite slice. (#691 and #714 left the set via PR
#1164; their one-time closure by #1151 and un-closure by the #1159 revert is
the worked example of why closing a pinned issue from the roadmap top-down
breaks the build.)

The dedicated liveness job is the hosted owner of fresh production-graph,
compiler-closure, boundary, and tracker execution. Script-unit tests consume
fixtures or injected source evidence for those paths, avoiding four redundant
Cargo inventory/closure runs while preserving the dedicated job's fail-closed
checks.

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
- The five class **METAs** (#727, #703, #709, #728, #694, with #695 under
  #727) are *in* the sub-issue graph, and on the side a reader is likely to
  guess wrong. Each META is the **parent** of its tracker - #729's parent is
  #727, #730's is #703, #731's is #709, #732's is #728, #733's is #694 - so
  the narrow true claim is the converse of the obvious one: a META is not
  filed as a *child* of its own tracker. Per `AGENTS.md` that pairing is
  historical and not the pattern for a new class. #728 is the first to close,
  and it closed *with* its tracker rather than separately, which is the
  disposal route the rest should take: its closing comment names the class,
  points at #732's document audit for the evidence, and records that the
  recurrence guards outlive both issues.

Every parent question the plan docs leave open has an answer on the graph:
**#916** sits at #883, **#1148** at #912, **#1157** and **#1137** at #730,
**#1156** was parentless because it belongs to no class and closed on PR
#1161 (leaving #1249 and #1252 behind it, also parentless, also no class),
**#985/#995** are unlinked as closed dupes of #965, and **#895** is
at #733 while its body opens "Part of #740", and the "Also part of #740"
comment the one-parent convention requires is posted, so that pairing is
recorded rather than implied.

Two structural notes stand. #1023 and #1029 are both hubs that carry parent
#908, which is legal but worth confirming intended; of the two, only #1029
carries the `tracking` label, so under `AGENTS.md`'s "`-label:tracking` is the
work queue" rule #1023 sits *in* the queue as an ordinary work item
(`documentation`, `soundness`, `type-system`, three `area:*`). And #1170, the
C-lane capability tracker, is correctly its own root, now with two children:
#1192, filed by PR #1186 when the host-helper `dropout` panic became a branded
rejection needing a support owner, and **#1059**, re-homed from #732 on
2026-08-21 ahead of that tracker's close. The re-homing is the interesting
one, because it is the move #732's closure needed and the move the one-parent
rule makes non-obvious: #1059's acceptance is a compiled-lane capability
event, which is #1170's subject and not the formatter plan's, so the link
moved to where the oracle is - and an `Also part of #732` comment records the
provenance the single parent link would otherwise erase. #1059 stays open and
stays in the CI-pinned set; nothing about the class closure loosened it.

Four open children appear in no plan doc, and all four are legitimate:
**#1091** (the [04-NUM-14] cast trap breaking five shells) and **#1092**
(hydronnx's privatized `TensorValue.data`) under #729; **#1129** (the landing
inventory for the #1036/#1038 and #1037/#1042 branch work) under #731, which
carries more weight now that #1036 is closed unmerged; and **#797** (central
OpenSpec governance gate) under #733. #898 leaves this list because this
change authors its missing reduction contracts and closes it on merge.

## Live defect inventory

Each entry is tagged with its owning tracker class and describes `main` at
`4e200061`. All but the last two groups are parented sub-issues of a class this
doc covers; the caveat is that **parented != scheduled** - a parent gives
ownership and the oracle that proves the fix, but several of these appear in
no remaining phase deliverable of their owning plan (see gap 5 below). **No
open PR carries any of them**: the in-flight board holds no plan-set work.
That was briefly untrue - PR #1254 was #731's PP3 - and it stopped being true
by landing rather than by lapsing. #1227 documents #1207's investigation but
fixes nothing.

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
  after the tag, so a shell on 0.18.4 still hits the regression. The residue
  its adversarial review filed (#1209, #1211, #1212) is closed: it became
  #731's PP3 on 2026-08-21 and shipped on 2026-08-22 as PR #1254.
  **#1197 is also repaired on `main` and also live in the
  release** (PR #1243, three review rounds). Its root cause was wider than
  the report: the trigger is not a chained lambda body but the stage
  **callee**. `resugar_node`'s `T::App` arm ran finite-list and
  operator-application sugars before it could produce the `Expr::Apply` that
  `resugar_pipe_stage` strips its leading parameter off, so every
  operator-named primitive (`add`, `sub`, `mul`, `div`, `mod`, `eq`, `neq`,
  `cmplt`, `lte`, `gte`, `and`, `or`, `neg`, `not`) and `x |> Cons(Nil)` failed
  closed - spellings `chelis fmt --check` accepts, `spec/01` §3.6 and `spec/02`
  §8.2 teach, and the `prefer-pipe-operator` autofix emits. It was never only a
  migration problem: `chelis surf`, the decompiler, failed on the same input.
  The fix rebuilds the stage application from the Deep `app` node before those
  sugars apply, holding back only the callee. What the repair did **not** cover
  is now three open issues, all parentless: **#1241** (the general v0.18
  pipe-lambda alias is still never rewritten, so those files remain
  unmigratable), **#1242**, and **#1246** - the last two `soundness`, and both
  cases where resugaring produces a *different program* rather than failing
  closed, which is the worse half of the same boundary.
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
- **Filed 2026-08-21, all unparented** - a checker pair the #1179/#1180
  repairs surfaced without fixing: **#1247** (integer type-application
  arguments are unenforced - a type parameter takes any integer as a wildcard
  and a dimension parameter takes one unconstrained, so #1240 closed the
  parser half of #1179 and left the checker half standing) and **#1248**
  (`<` and `>` disagree between the linearity and host typing tables, so one
  consumes its operand where the other borrows - a consistency defect #1245's
  operand-order fix put in view). A cache pair behind the #1156 repair:
  **#1249** (the prepared-reef-graph cache is keyed on the release version,
  not the compiler build - the same defect one cache over from the one PR
  #1161 just fixed) and **#1252** (no regression test pins the cdylib
  image-resolution path that fix depends on). And two environment items:
  **#1237** (`rust-toolchain.toml` pins 1.98.0 while the pinned rust-overlay
  tops out at 1.97.1 - PR #1238 bumped the overlay, but the issue stays open
  on its third acceptance criterion, that `nix-packages.yml` run on pull
  requests at all, since this class went 18 days undetected) and **#1244**
  (installing the devenv hook from a worktree corrupts the shared `commit-msg`
  hook for the whole clone).

## Prevention map (failure mode -> mechanism -> status)

| failure mode it prevents | mechanism | status |
|---|---|---|
| A new bare-float/raw-dtype numeric surface | §C6 census + tripwire; PR #1154's permanent dispositions (baseline regeneration cannot silently bless a change) | landed; #1160 (seam-identity relocation vocabulary) is the open design question, and it recurs on every dtype-migration rung |
| Op x dtype drift, hand-mirrored lists | #729 Phase 4 capability table | 4A capacity ratchet landed; 4B decided-contract/schema freeze is this change; #1294 exact builtin-atom closure gates 4C; 4C tables, 4D consumers, and 4E conformance are unbuilt |
| The next silent value substitution | #730's typed channels (landed) + Phase 4 ratchet totality (§C7) | Phase 4 unstarted (#990) |
| Gate drift / duplicate gates | #730 P3 gate contract + dedup + deletion | the gates themselves are done (PRs #1175 and #1186: one shared typed policy per target, zero `fn reject_` definitions left in `chelis-cli`, the syntactic `reject_*` source manifest, #697/#698/#705/#959 closed); the recorded contract, the deletion pass, and a semantic no-duplicate tripwire remain |
| A breaking release reaching shells before anyone runs their suites | the ecosystem-drift canary (auto-files per shell) + `conform bump-check` | neither detects nor gates this. The canary checks each shell out at `main` and never runs `chelis migrate surf`, so it only ever exercises pre-migration source - and #1200 appears only after migration rewrites record patterns to the mandatory v0.19 pun, which makes the existing leg structurally incapable of seeing it. The canary has also been red continuously since 2026-08-02 - its last success is the 2026-08-01 run, and all twenty-four runs since, through 2026-08-21, have failed - so it gates nothing in practice either. The missing leg applies a release candidate's own named migration to each shell tree and then runs that shell's suite |
| Unhandleable AST states reappearing | #731/#908: stamped-only ingress (#1088), then carrier deletion (#1029) | PR #1285 landed the stamped-ingress and parity-oracle half and closed #1088. This revision's Node-native manifest walkers deliver #1082, so carrier deletion (#1029) is no longer blocked by the #912 consumer |
| Spec silence and stale claims | #733 end-to-end | nothing active - the weakest link, with the #891/#904 twenty-builtins-no-spec instance as its measured cost |
| Guards existing but not running | #1089's inventory of oracles outside continuous jobs; #990's scheduled/change-gated package | partial: #732's Phase 2 oracle has a dedicated blocking job and Phase 3 runs nested continuously, and #729's Phase 1/2 oracles run nested inside Phase 3. PR #1285 wired `unrepresentable_domain_oracle.py` (#908) into `gate.py`'s `integration` stage, which hosted CI runs in the `workspace-tests-shard` matrix on every non-docs-only PR, with `test_gate.py` locking the membership and the pairing to a nextest-installing job. Still unwired: `loud_unsupported_phase2_oracle.py` (#730), `compiler_pipeline_oracle.py` (its three controls run in the gate's `lint-and-unit` stage, but the oracle itself is invoked nowhere - `grep -n compiler_pipeline_oracle scripts/gate.py` returns one comment line and no call - which is not what #1089 asks for), and `loud_unsupported_phase3_oracle.py`; its former #912 red leg is green in this revision, but the runner is still not continuously invoked. #1090 closed as refuted - the canary already auto-files into the shell repos (coral#23, school#189, hull#14, hello-chelis#19, octant#42, hydronnx#64). **A fresh instance of this exact mode landed on 2026-08-21 outside the oracle inventory**, and its two intervals are worth keeping apart. The guard's dormancy is the long one: `nix-packages.yml` declares an unconditional `pull_request` trigger and has not executed since 2026-08-03, eighteen days, for a reason nobody has diagnosed - #1237's body deliberately leaves trigger bug, disabled workflow, and runner availability all open. The divergence it would have caught is the short one: the Nix and devenv lanes resolved a different rustc than CI only once stable moved to 1.98.0 after 2026-08-20, about a day before #1236's toolchain pin turned that silent disagreement into a loud failure. A guard that has been dark for eighteen days is not measured by the defect that happened to arrive on day eighteen |
| Silent lane divergence | #754/#763 cross-lane gate; #738 shell compiled lanes | unblocked by #732 P2, undelivered |
| Agent-driven recurrence | #740's enforcement-ladder backlog; #895 executable plan inventories | entirely unchecked, dormant |
| Wrong issue closures | the liveness manifest (detects after the fact); a keyword-auto-close guard (prevention) | the guard is missing - three incidents (#716, #729, #912), one reverted overreach (#1151/#1159), and 22 more class closures on 2026-08-05 through the same unguarded path. The 2026-08-17/21 audit-and-pin wave (evidence-comment closures for #916/#646/#986; pinned-test PRs #1230-#1232 for #862/#1084/#683) is the closure discipline done right by hand - and the same wave produced the #1083 conflict (closed against PR #1231's own record, reopened the same day), which argues for the guard, not against the wave. **#732's own closure is the sharpest illustration of why the guard is owed**, because the substance was right and the mechanism was still an accident. Right: #1059 was re-homed before PR #1250 merged, and the merge itself retired the last open child (#997), leaving zero at close; both owning documents were audited by full read, and every live thread had a named home elsewhere. Accidental: the close itself was a keyword auto-close at 23:31:48Z, two seconds after the merge, fired by narrative prose in PR #1250's body (`... then close #732`) rather than by any deliberate directive - the squash message carries no closing keyword at all. The full evidence record landed seventy-two seconds later, at 23:33:00Z, and its own first line says so. A guard would not have blocked this close; it would have made it deliberate |

## The six biggest remaining gaps, elaborated

### 1. #730's gate half - delivered; the contract and the ratchet are not

The gates themselves are in place. What `254d6070` measures (re-verified for
this revision; every count and anchor below is unchanged from `841b6ddb` and
`77a74ea1`):

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
`scripts/loud_unsupported_phase3_oracle.py` runs six legs. All six pass at
this revision: the shared gate + rejected-cell corpus + gate inventory, the
sealed diagnostic-kind contract and its mutations, the typed
rejection-authority tests, the rejection-authority boundary, the live
issue-authority manifest, and the #912 root-realizability integration. The
last leg runs `cargo nextest run -p chelis-cli --test
issue_912_root_boundary --run-ignored all`; its ledger is empty and all seventeen
cells pass. The #912 interlock no longer blocks Phase 3, so Phase 3 is
executable-oracle green; its completion claim still awaits this delivery's
fresh exact-head red-team pass. Also owed outside Phase 3: the recorded gate
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
arms, and conformance is hand-curated files - three places every new op must
agree. Phase 4B freezes the inversion: Table A is the complete
`BuiltinId x SurfaceClass x operand Prim x SemanticParams` semantic product,
with `SurfaceClass = Scalar | Tensor`; Table B adds
`eval | c-host | c-dag | hip | metal` only for supported A cells.
Container/boundary builtins use the sibling
`BuiltinId x SiblingDomain x SiblingCaseId x SemanticParams` registry and its
supported-row backend product; runtime, stdlib, and binding callables retain
§C6's exact-canonical-identity family registries. Recursive host types use the
companion constructor table.

The language and schema decisions are no longer open. [05-OP-11..16] own
reductions, [05-OP-17..19] own wrap arithmetic, [05-OP-20..22] own float
classification, and [05-OP-23..24] complete the named cast ladder without
`cast_round`. The five former schema questions are resolved in
`capability_table.md`. What remains is implementation: 4C table population,
4D generated checker/backend/root consumers, and 4E generated conformance.
Behavior-changing kernels or stable loud rejections ship at v0.19; the
behavior-preserving mechanism ships at v0.20. The merged census permanence is
the floor. #1160 remains a parallel seam-relocation decision, and inherited
holes include #951, #957, #958, and #960.

### 3. #1088 landed; #1029 is next behind it

The finding was that the compiler-api generic Deep ingress (`parse_str_strict`, and
the non-strict `parse_deep` at `compiler.rs:2903` that the issue body did not
name) skipped the declaration-role stamp - a weaker second front door through
which unstamped trees reached consumers.

**State: PR #1285 merged and closed #1088.** It migrated every compiler-API
door plus the named non-compiler-API stragglers
(`chelis-validate::validate_deep`, the `opaque-domain-construction` lint
rule, `chelis-e2e`'s snippet checker, `chelis-lsp`'s Deep document analysis,
and the `chelis-cli` style-gate fallback) onto the stamped carrier, each
field stamped in the role it occupies. It carries the requested parity
oracle, `crates/chelis-compiler-api/tests/phase3_stamped_ingress.rs`: one
accept/reject corpus driven through every module-text door, plus a
`syn`-based recurrence guard over the workspace's production sources. What
that guard actually covers is worth stating precisely, because a census is
not a proof: it resolves qualified paths, `use` imports under any local
name, module aliases, crate aliases (including the grouped-`self`,
`pub use`, and `extern crate` spellings), and glob imports, with every
identifier unraw-normalized first so a raw spelling such as
`chelis_deep::r#parser::r#parse_str` cannot slip a string comparison; and it
applies a deliberately conservative fail-closed policy to `macro_rules!`
bodies, whose tokens no path resolution can see through. It does not resolve
a cross-file re-export chain or a procedural macro from another crate;
closing those needs the visibility chokepoint that carrier retirement
(#1029) brings, not a source census. It decides the top-level form rule in the controlling
numbered spec ([03-PROG-1] through [03-PROG-3] in
`spec/03-deep-syntax.md`), which chapter 03's PEG had left unrestricted. And
it wires #908's own oracle continuously; see the "Guards existing but not
running" row for that wiring. The #1036 re-salvage that #1129 holds was
dispositioned in the same change: the read-only authoring stamped ingress
and the `UnknownForm` wire-preservation fix were salvaged, and the branch's
authoring-scope commits were already on main via PR #1136.

The remaining queue is #731's §C4.2 deletion clauses and #1029's deletion of
`Expr::List`/`Atom::Tag`. That make-illegal-states-unrepresentable payoff of
the entire #908 arc, still 1,024 and 77 matching lines deep respectively, is
next in the chain and is unblocked by this revision's #1082 delivery.

### 4. #912 Faces 1 and 4 - manifested boundary delivered for v0.19

The dependency chain was executed in order in one integration: #1082's
Node-native walkers, #1079's production `ManifestedProgram` consumption,
per-root input routing, C/HIP `requires_main()`, #1083's structural dotted
tuple/ADT expansion, and C1's target-aware f64 route. [05-OBS-7..11] now own
the normative root-manifest contract: automatic roots exclude effectful
nullaries, built-in variable-size containers do not expand, and a concrete
selected callable carries only its reachable runtime inputs. `source_arch.rs`
rejects both the old observe-and-discard shape and a restored emitted-C `main`
scan. The executable acceptance surface has an empty ignore ledger and passes
all seventeen cells, including the manifest-completeness and nullary-product
cells that previously blocked #730's sixth Phase 3 leg. The additive `spec/05`
change follows
`dtype_semantics.md` §B1: it changes no frozen [05-OP] atom or capability
schema, and the complete Phase 4B oracle passes. This delivery unblocks #1029.
The complementary #1102 value fix is now implemented with explicit
output-reachable callable-dependency evidence. The numbered specs settled
#1148's f32 rank/carrier question in favor of the existing rank-zero
`tensor[f32]` inference, and the #1148 change is limited to explicit f32 tests
and fixture-signature corrections. That evidence does not establish full
per-dtype [05-OP-33] conformance.

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
the same pattern. The newest instance is the cleanest, because both steps
happened at once and the whole cycle took a day: #1209, #1211, and #1212 spent
a month with no class and no slot, then got both - the parent link to #731 and
the PP3 row in `checker_totality.md` - inside the same change set that
implements them, and closed on its merge (PR #1254).

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
(twenty builtins and a prelude ADT with zero spec entries). Before this change
merges, its four children are #735 (the unauthored `with seed` / `with device`
semantics), #797 (adopt the central OpenSpec governance gate), #895, and #898.
#898's missing reduction authorities are delivered by this change, leaving
three open children afterward. #740's backlog is two items of eight, and
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

Ranked by leverage: wire the remaining oracles, then write down the gate
contract that the two shipped slices now make cheap to state. Both are small
relative to what they unlock, while the capability table and the #912 chain
are the two real campaigns left before v0.19/v0.20. #1088, the third entry on
this list at the last revision, landed through PR #1285.

## Open PRs

Eight PRs are open besides the specification change carrying this report:

| PR | what it retires | state |
|---|---|---|
| #1304 | #1292's own-width tensor closeness semantics; stacked on this change and intentionally blocked on #1288's zero-exception census | draft |
| #1303 | #1287's first-class bool-tensor `count` and exact-only WireDag v6; stacked on this change | draft |
| #1302 | #1293's exact exported stdlib numeric surface and removal of obsolete aliases | draft |
| #1301 | #1080's exhaustive typed `DeepTag` realizability classification | open |
| #1299 | #1294's exact builtin-atom closure; intentionally red until every valid uncovered identity has authority and obsolete identities are removed | draft |
| #1227 | Publishes the #1207 typecheck-superlinearity investigation and levels plan (docs-only; fixes nothing) | open |
| #1203 | Commits the crate2nix graph and composes shared devenv tools; also raises #1237's stakes, since it makes the devenv/Nix lane authoritative for every CI job | open |
| #838 | Conform central workflow wrappers (#788 family, Chelis-Lang/ci#5 prereq) | draft |

PR #1285 left this table by merging and closing #1088. The dtype prerequisite
stack is now visibly staffed through #1292, #1293, #1294, and first-class
`count`; the zero-exception census foundation is now this change, while its
remaining legacy-row migration and the generated-table slices remain to be
delivered in their declared dependency order.

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
2. **The linearity residue's owner and class - ANSWERED, and delivered
   inside twenty-four hours of the answer.** #1209, #1211, and #1212 became
   children of #731 on 2026-08-21 and closed on 2026-08-22 with PR #1254,
   which is PP3 in `checker_totality.md` and which authored [04-LIN-1] and
   [04-LIN-2] in `spec/04-type-system.md` §8.3. The original question stands
   recorded because the answer took a month to arrive and the shape of it is
   the reusable part: #1200 was repaired without ever getting an owner or a
   class (PR #1208 landed classless) while #731's oracle stayed green
   throughout, and what finally resolved it was not a triage pass but the fix
   turning out to *be* checker-totality work - the durable form #1208's own
   review had named, binding-generation identity, is exactly what #731 exists
   to own. Read together with decision 3 the lesson is narrow and usable: an
   unparented issue is not waiting for a hub to be invented, it is waiting for
   someone to notice which existing oracle turns green when it is fixed.
3. **Two unowned families' homes** - the same question, asked twice, and
   decision 2 is the argument for answering it: three parentless issues sat
   classless for a month, and closed within a day of a class claiming them.

   *The performance family.* #1205, #1206, and #1207 are the same shape from
   a lane no tracker owns: superlinear compile-path costs (front-end
   lowering, compiled recursion's memory retention, and `Env::generalize`'s
   per-binding environment sweep), each `area:perf`, each parentless. #1207
   already has a levels plan on PR #1227. Decide whether they get a tracking
   hub (the one-tracking-issue-per-class rule fits: a recurring cost class
   with a shared oracle shape) or stay standalone ledger entries like #888.

   *The resugaring family*, newer and with a soundness edge the performance
   one does not have. #1242 and #1246 are two faces of one boundary - a
   call-first pipe stage whose argument positions do not survive the Deep
   round trip, producing a *different program* rather than a failure - and
   #1241 is the migrator that consumes it. All three were filed by PR #1243's
   own review, all three are parentless, and nothing owns the Surf/Deep
   resugaring boundary. #1024 is the nearest existing hub (it holds #1171 and
   held #1179/#1180) but its subject is the canonical-grammar cut, not this.
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
9. **Which Surf follow-ons ride the v0.19 cut. The cut question itself is
   settled: v0.18.5 shipped through PR #1256 on 2026-08-23.** It carries the
   #1197 migrator repair, the recursive-generic
   capability, and all three BREAKING boundaries - polymorphic recursion at
   check time ([04-INF-2]/[04-INF-3]), integer literals rejected in bare type
   positions (#1179), and `>` evaluating operands in authored order (#1180) -
   so it is a source-migrating release, which is what the arithmetic here
   always implied. The version number is the interesting part of the call and
   is reasoned on that PR: the roadmap reserves the **v0.19 label for a defined
   payload** (the per-dtype storage break, the capability decisions, the
   manifested root completion), none of which this cut carries, and the
   migrator keys off the 0.18 line - a compiler at 0.19 whose migrator still
   speaks `--from 0.18` is a confusing pair. That reasoning is now the record
   of the shipped cut.

   What is left of this decision is only the Surf ride-which-cut question, and
   it is down to two: whether **#1171** (typed first-argument
   application-to-pipe promotion, explicitly not delivered by #1031) and
   **#1172** (structural spans, parented to #883) ride v0.19 or a further
   patch. #1179 and #1180 left this list by shipping in the 0.18.5 cut.
10. **#1170's relationship to #730 Phase 4 is decided.** Its container and
   boundary builtins use the sibling
   `BuiltinId x SiblingDomain x SiblingCaseId x SemanticParams` capability
   registry selected by their declarations, not numeric Table A's
   `SurfaceClass` axis. Exported callables retain §C6's semantic family
   registry. Runtime C exports and PyO3 bindings additionally use Phase 4C's
   total external target-disposition registry; exported stdlib definitions
   instead derive per-backend executability transitively from their checked
   bodies in Phase 4D. #730 owns the shared loud-rejection channel. Neither
   plan builds an independent support list.

## Standing process conditions

These are properties of how the repo currently works, each with the instance
that measures it.

1. **Nothing prevents a keyword auto-close.** A squash body's "Fixes #N"
   closes the issue whether or not the fix is complete, and the only guard is
   the liveness manifest's after-the-fact detection on an unrelated PR's CI.
   #716, #729, and #912 each sit on that history, reopened after the fact; the
   one mitigation attempt (#1151) is reverted as overreach, and the 22 class
   closures of 2026-08-05 rest on the same unguarded path - as do the August
   repair closures (#1158, #1200, #1201, #1216), all five of 2026-08-21's
   (#1179, #1180, #1197, #997, #1156) and all three of PR #1254's (#1209,
   #1211, #1212), each closing within two seconds of its PR's merge, all backed
   by a green oracle, which is the luck the guard would replace with a check.

   **#732 is the sixth of that evening, and it exposes a sub-mode the
   condition had not named: a PR body's narrative prose is directive
   surface.** Its `ClosedEvent` closer is PR #1250, at 23:31:48Z, two seconds
   after the merge - but #1250's squash message carries no closing keyword at
   all. What fired was a sentence of ordinary explanation in the PR *body*,
   reading `... then close #732` while arguing about where #1059 should be
   parented. Nobody wrote a directive; one got parsed. Only #728 was closed by
   hand that evening (its `ClosedEvent` has no closer at all, at 23:33:02Z).

   This exact pair has been through it before, which is the part that should
   settle the argument. On 2026-07-17 PR #742's squash body closed both: #728
   by an ordinary directive (`Fixes #728`) and **#732 by the same prose
   accident**, a line reading `Fixes #732 P1's format_element placement as ...`
   that GitHub read as `Fixes #732`. Both were reopened the next morning. So
   the tracker that the plan set just finished has been wrongly auto-closed
   twice, thirty-five days apart, by two different PRs, neither of which
   intended it - and on the second occasion the close happened to be correct,
   which is exactly the coincidence a guard exists to stop relying on.
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
   #1084 via #1231, #683 via #1232), and it continued on 2026-08-21 with the
   #732/#728 class closure, whose evidence comment names the closing condition,
   the PR that met it, the two documents audited, and where each surviving
   thread went. Read that alongside condition 1: the *evidence* was authored
   deliberately and the *close* fired by accident seventy-two seconds earlier,
   which
   is a fair summary of where this repo's closure discipline currently sits.
   Still a convention rather than a gate - and the same wave shows the cost in
   the other direction: #1083, closed against PR #1231's own record and
   reopened an hour later. **#1237 is the current live instance of the
   opposite failure**, and a benign one: PR #1238 repaired the breakage and
   deliberately said "Part of #1237" rather than "Closes", because the issue's
   third acceptance criterion (that `nix-packages.yml` run on pull requests at
   all) is not met. That is the convention working - but nothing distinguishes
   a deliberately-open issue like this one from the ~29 fixed-but-open ones the
   validation pass found, which is the gap.
5. **An oracle that reports FAIL outranks a phase claim that reports done.**
   Both current examples are self-reports corrected by execution: #732's Phase
   2 command failed on `main` because a zero-cell ledger has no real
   classifier invocation to receipt, and #730's Phase 3 runner exits nonzero
   on the #912 leg rather than scoping the leg out. Both behaviors are
   correct. The hazard is the converse - an oracle nobody runs proves nothing,
   which is #1089.
6. **Release notes drift; PR fragments are required.** v0.18.4 had
   to author its own section from the commits (`[Unreleased]` was empty and no
   commit since v0.18.3 touched `CHANGELOG.md` - 21 PRs, zero entries, per PR
   #1199's account, and `git log v0.18.3..v0.18.4 -- CHANGELOG.md` returns
   only the release commit), and to reopen the released `[0.18.3]` section,
   which documented 1 of its 20 commits and omitted a breaking change
   (chelis#1130's `shape()` return to int64, after 0.18.2's notes had told
   readers not to migrate onto the int32 form). Three consecutive releases now
   show the same drift.

   **The convention this document praised one revision ago has been reversed,
   and the reversal is worth reading as a trade rather than a retreat.** Until
   2026-08-21 the answer to drift was a per-PR `[Unreleased]` entry landed in
   the same change set, and the previous revision recorded five post-tag PRs
   (#1204, #1208, #1213, #1215, #1218) doing exactly that. The cost only
   appears with several PRs in flight at once, and then it appears every time:
   every entry inserts into the same section, so any two in-flight PRs conflict
   pairwise no matter how unrelated their code is. Measured on 2026-08-21 with
   three such PRs open - #1240, #1243, and #1245 each adding an `[Unreleased]`
   entry - #1240's merge forced #1245 to rebase and re-run its full CI
   (force-push at 20:46:18Z, commits re-dated 20:45:40Z), and it did so on the
   same afternoon the org's Actions budget was intermittently refusing to start
   jobs at all (condition 9). Per-PR `[Unreleased]` entries are therefore
   **disallowed** from here; release notes assemble from PR bodies in the
   interim, and **#1251** is the decision issue for the durable answer,
   towncrier-style `changelog.d/` fragments that carry one file per change and
   so cannot collide. PRs #1250 and #1161 are the first two merges under the
   new rule and touch `CHANGELOG.md` not at all. PR #1254 is the instructive
   third: it edited `[Unreleased]` without adding an entry, retracting a
   "Known gap, deferred" paragraph inside the existing #1200 entry that its own
   fix had just made false. Correcting a claim a merge falsifies is not what
   the reversal disallows, and the distinction is worth keeping when #1251 is
   designed - a fragment scheme has to leave a way to amend an already-written
   fragment, or it recreates this case as a conflict.

   #1251 supplies the [fragment contract](../../changelog.d/README.md), a tested
   local Python assembler, and a required `Changelog` CI check.

   Behavior-changing PRs author fragments; release assembly
   preserves historical sections and consumes the pending entries. GitHub
   Releases use the committed version section. Missing fragments and direct
   changelog edits outside reproducible assembly fail CI;
   `no-changelog` suppresses only the missing-fragment requirement. The executable
   acceptance command is `.venv/bin/python -m unittest scripts.test_changelog`.
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
9. **Merge governance has an external dependency nobody had written down:
   billing.** The #1096 rollout makes green, up-to-date checks mandatory on
   every repo with no bypass list, which is the right control and the reason
   condition 2 holds. It also means that when the org's GitHub Actions budget
   is exhausted, merging stops entirely - not degrades, stops - because the
   required checks cannot run. That happened on 2026-08-21 and the failure mode
   is easy to misread: jobs report `failure` in one to three seconds having
   executed zero steps, with the whole story in an annotation rather than a
   log - *"The job was not started because an Actions budget is preventing
   further use."* Two runs are the record: the post-merge run on `main` at
   20:44:57Z, where 13 of 15 jobs failed in 2-3 seconds and the 2 dependent
   jobs were skipped outright, every one of them having executed zero steps;
   and the #1250 branch's run at 20:57:55Z. By 21:00Z the budget was raised and
   the next run passed normally.
   Nothing in the repo detects or reports this state, and a red required check
   is indistinguishable at a glance from a real regression, which is the part
   worth fixing - the budget itself is an ops decision, but "CI is red for a
   reason no commit caused" deserves to be legible.
