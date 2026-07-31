# Recon: Proof-Stack Ground Truth vs the 2026-07 Roadmap

Date: 2026-07-15. Read-only audit of the factual claims in `proof_stack_roadmap_2026_07.md`
against authoritative sources. No repo was modified except to add this report. Every verdict
cites a command that was run or a file:line that was read, and names the checkout it was tested
against. Method summary in the appendix.

**Report home.** This file lives in the chelis `main` worktree (`~/Documents/scratch/chelis-proof`,
checked out at `5f86ddfa` = `origin/main` = release v0.16.1) — the main-branch working tree of the
single authoritative local chelis clone (`~/Documents/scratch/chelis`, whose own working tree is on
`seed-diligence`). Chelis over beacon because the roadmap's open action items are chelis-side and
the local beacon checkout is stale (§1).

---

## Headline results

1. **The roadmap is not a static proposal — it is being executed, by its own author, in the
   beacon repo, starting the same afternoon it was written.** The audited copy in `~/Downloads`
   is an intermediate snapshot (byte-identical to beacon commit `e440306`). Four roadmap items
   (0.1 beacon side, 0.3, 0.4, 2.1+2.3) landed on beacon main within hours, all authored and
   self-merged by rlronan with zero independent reviews, all **unreleased** (latest beacon
   release v0.1.8 predates everything).
2. **The version break (C1) is real but mis-framed:** "the next chelis release breaks every
   dispatch" was already stale when written — WireDag v3 shipped in chelis v0.15.0 on 2026-07-10,
   four days *before* the roadmap. The break is live across five released chelis versions against
   every released beacon, fixed on beacon main only, and **dormant in practice because every
   shell pins chelis =0.14.0 or older**.
3. **The "zero production callers" claim (C5) is false as stated.** FlukeBall's WC26 KellyBench
   lane discharged 338 real, LLM-authored strategy goals through a manually-registered
   `BeaconShim` against a SHA-pinned beacon binary (artifacts dated 2026-06-27/29), and that work
   was merged to flukeball `origin/main` ~2 hours *before* the roadmap's initial commit. What is
   true — and stronger than the roadmap says — is that `with_beacon` has **zero callers of any
   kind** on chelis main and `chelis prove` itself cannot route to beacon.
4. **The documentary authority chain exists and is largely evidence-backed** — the review doc,
   decision record, and both experiment dirs are committed on beacon main with raw logs backing
   most headline numbers to the digit. The earlier suspicion that they were absent was itself a
   stale-checkout artifact (they landed on beacon remote after the last local fetch). Governance
   caveat: the chain is single-author, self-merged, and the decision record's "accepted
   (maintainer-endorsed)" is the author endorsing his own decision; no independent acceptance
   record exists anywhere.
5. Of 18 audited claims: **9 confirmed** (mostly accurately cited, real), **4 partially true**
   (real core, overbroad or mis-scoped framing), **2 stale** (true when written, fixed the same
   day), and the remainder true-but-preference. Verdict-relevant detail in §2; consumer buckets
   in §5.

---

## 0. The roadmap document itself: provenance

The audited copy is `/home/jeff/Downloads/proof_stack_roadmap_2026_07.md` (untracked; mtime
2026-07-14 23:00 +0100; 36,571 bytes). It is an **intermediate snapshot** of a committed,
actively-updated document in `Chelis-Lang/beacon`:

| version | ref | notes |
|---|---|---|
| initial commit | beacon `452b5a5d`, 2026-07-14 **18:27:06Z** | Phase 0 titled "Stop the bleeding"; no items 0.4/0.5; "the time bomb" |
| **~/Downloads snapshot (audited)** | byte-identical to beacon `e440306` (19:17Z, "three accuracy corrections + finality pass") | adds 0.4/0.5; still says "item 23", "WI-H1..H6"; **no status markers** |
| HEAD copy | `docs/proof_stack_roadmap_2026_07.md` @ beacon main `5de54523` | status markers: 0.1 beacon side in review→merged (beacon#54), 0.3 **complete** (issues linked), 0.4 merged (beacon#53), 2.1+2.3 merged (beacon#55); WI-H7 added |

Evidence: `diff` of the Downloads file against `git show 452b5a5d:docs/...` (non-empty) and
against `git show e440306:docs/...` (byte-identical), in a fresh clone of beacon remote main.

Acceptance record: none beyond the author. Beacon PRs #50/#53/#54/#55/#56 were each authored
**and** self-merged by `rlronan` with zero reviews (`gh pr view … --json author,mergedBy,reviews`);
the companion issues (beacon#52, chelis#673, chelis#674) were filed by the same account in a
3-second burst at 19:32:16–18Z, 65 minutes after the roadmap's initial commit, with the
"0.3 complete" commit landing 24 seconds later.

---

## 1. Checkout topology (Phase 0)

Method: per-checkout `git remote/branch/status/worktree list/rev-list` (no fetches — cached
refs), cross-checked against remote HEADs via `gh api repos/Chelis-Lang/<repo>/commits/main`, and
fresh scratchpad clones of beacon and flukeball for remote-truth reads.

### Authoritative-source decisions used by every verdict

| repo | authoritative source used | why |
|---|---|---|
| chelis | `~/Documents/scratch/chelis` at `origin/main` (= remote main `5f86ddfa` = v0.16.1); file reads via the `chelis-proof` worktree at the same sha | cached origin/main matches remote main exactly; single primary clone |
| beacon | fresh clone of remote main `5de54523` (2026-07-14 21:28Z) | **local `~/Documents/scratch/beacon` is stale** (HEAD `4579ebe`, 2026-07-06); remote received 20 commits on 07-13/07-14 — including the roadmap itself, its evidence base, and fixes to its claims |
| flukeball | fresh clone (remote main `a1ec6287`, `origin/world-cup` `4bc0b97`); local `fluke-ball-wc-iso` only for machine-local pin facts | see divergence note below |

### Primary clones (Chelis-Lang)

| checkout | repo | branch | HEAD | dirty | vs remote main | latest release |
|---|---|---|---|---|---|---|
| chelis | chelis | seed-diligence | 6e5f3cbf | 0 | cache MATCH (5f86ddfa) | v0.16.1 (07-12) |
| beacon | beacon | main | 4579ebe | 0 | **STALE** (remote 5de54523, 07-14) | v0.1.8 (07-06) |
| whale | whale | codex/flukeball-46-outcome-helpers | e87b950 | 5 | MATCH | v0.1.13 |
| shoals | shoals | docs/note-load-sensitive-mc-tests | d089adf | 0 | MATCH | v0.23.0 |
| fluke-ball | flukeball | main | 05c5ebf | 0 | **401 BEHIND** origin/main (fast-forwardable, not divergent) | (none) |
| fluke-ball-wc-iso | flukeball | world-cup | 2a3e447 | 0 | STALE (260 behind) | — |
| hull | hull | main | 283a16d | 0 | MATCH | v0.1.6 |
| hydronnx | hydronnx | feat/pad-depthspace-mod-operator | e847deb | 0 | STALE (remote 4d960f73) | v0.1.8 |
| c-note | c-note | main | 2f89539 | 1 | MATCH | (none) |
| school | school | main | e6cd1ce | 0 | STALE (remote 38bc5d7d, 07-14) | v0.1.9 |
| nautilus / coral / octant / calcify / c-earchin / economoist / LaCaDiLE / hello-chelis / website / gtm | same-name | various | — | 0–19 | MATCH | v0.7.33 / v0.7.30 / v0.10.1 / v0.10.2 / v0.3.3 / v0.2.3 / — / v0.1.10 / — / — |

Org repos with **no local checkout**: `flukeball_house`, `flukeball_2`, `school-bootstrap`.

**FlukeBall divergence corrected.** The pre-audit briefing said fluke-ball local main was "401
ahead" of origin/main. Measured: it is 401 commits strictly **behind** (`git rev-list
--left-right --count`; merge-base equals the local tip `05c5ebf`). The 401-commit line is the
entire WC26 KellyBench experiment, developed on `world-cup` and merged to **remote main** via
`99dc0d2` "Merge world-cup into main: the WC26 KellyBench experiment (closed, null result)"
(2026-07-14 16:25:57Z). Flukeball remote main is authoritative and **already contains the beacon
integration**; the local primary is a pre-experiment snapshot.

### Worktrees and duplicates that matter

- The chelis clone owns **67 linked worktrees** sitting at three different
  `WIRE_DAG_SCHEMA_VERSION` values: **=3** (`chelis`@origin/main, `chelis-proof`,
  `chelis-wt-*`), **=2** (`chelis-fullprove`, `chelis-tooling`@release/0.12.0, `chelis-smt-*`,
  `chelis-deep-*`, `chelis-bug-fixes`), **=1** (`chelis-prove-pipeline`, `chelis-specs`,
  `chelis-rt-ws7*`, `chelis-z3`, `chelis-ws7-sollya`). Any citation resolved on a =1/=2 tree is
  anchored on a non-authoritative checkout; `chelis-prove-pipeline` (schema v1, +118 behind) is
  the known stale-never-use candidate.
- beacon worktrees: `beacon-constensor` (flukeball-bugs-beacon), `beacon-phase2e` (accepts schema
  {1} only — two generations stale). `beacon-bakeoff` is a separate local-only dossier repo (no
  `src/`), the source of the ±299.08 measurement used in C7.
- Where did the roadmap author work from? The stale local beacon (`4579ebe`) does **not** contain
  the experiment commits (`git cat-file` on `ad0c15c` fails there), yet the roadmap cites them —
  the author wrote from remote content while the local checkout lagged. Every roadmap code
  citation checked resolves exactly on `4579ebe` *and* (except where fixed same-day) on remote
  main; the citations were accurate, the *framing* is where staleness crept in (C1, C7).

---

## 2. Claim table (Phase 1)

Verdicts reflect **today's authoritative state**; temporal splits noted. "Checkout" names what
was actually tested. Full command/quote evidence for every row was captured by the investigation
agents and spot-verified by independent adversarial verifiers; the load-bearing items are inlined.

| # | claim (roadmap item) | verdict | classification |
|---|---|---|---|
| C1 | chelis emits WireDag v3, beacon accepts [1,2]; next chelis release breaks dispatch (0.1) | PARTIALLY TRUE | real defect (released seam); framing stale |
| C2 | guarding test `beacon_e2e.rs:274` is `#[ignore]`d and asserts v2 (0.1) | CONFIRMED | real defect |
| C3 | stale "currently 2" comment + seam-contract "==1" (0.1) | CONFIRMED | real defect |
| C4 | v3 diff touches only beacon-rejected features; minor parser change (0.1) | CONFIRMED | true (judgment word "minor" held up) |
| C5 | beacon plumbing has zero production callers; every goal ever from a test fixture (1.1) | PARTIALLY TRUE | narrow core real; as stated **wrong at authorship** |
| C6 | "beacon rejects tensors"; tensor-state decision largest open (3.1f) | PARTIALLY TRUE | real limitation, one-liner overbroad |
| C7 | erf envelope valid on [-300,300]; outside undocumented (2.9) | PARTIALLY TRUE | figure right; "undocumented" wrong; residual is doc-policy preference |
| C8 | architecture.md:204 carries falsified auto_LiRPA-rejects-Less justification (0.4) | STALE | fixed same day on beacon main (unreleased) |
| C9 | decision record = "the accepted graduation" | CONFIRMED (exists) | acceptance is self-marked, single-author |
| C10 | experiment dirs + named artifacts exist | CONFIRMED | all committed (one absent log noted) |
| C11 | headline measured numbers recorded in artifacts | 4× CONFIRMED, 2× PARTIALLY TRUE | see breakdown |
| C12 | --capabilities reports beacon on bare env var (0.2) | CONFIRMED | real defect |
| C13 | GoalShape::BoxRange has no syntax/CLI (1.2) | CONFIRMED | true-but-preference (documented staging) |
| C14 | zonotope-verified mode bypasses certificate lane at interval.rs:87 (2.8) | CONFIRMED | real defect (completeness, fail-closed) |
| C15 | zonotope lane accepts fewer ops than interval (2.7) | CONFIRMED | true-but-preference ("backwards" is opinion) |
| C16 | arb_gate.py hard-codes x86_64-linux (2.6, beacon#49) | CONFIRMED | real defect (already ticketed pre-roadmap) |
| C17 | WI-B4 CROWN core doesn't exist beyond affine substrate; plan calls it centerpiece (0.4) | PARTIALLY TRUE | code half confirmed; plan half fixed on beacon main, **chelis mirror still stale** |
| C18 | capabilities overclaim / wiring gap / tensor decision are "unticketed folklore" (0.3) | STALE | true at authorship; self-resolved 65 min later |

### C1 — the version break

- Tested on: chelis origin/main `5f86ddfa`; beacon-fresh HEAD `5de54523`, tag `v0.1.8`, stale
  local `4579ebe`.
- `schema.rs:1154` (chelis main): `pub const WIRE_DAG_SCHEMA_VERSION: u32 = 3;`. Released beacon
  (`git show v0.1.8:src/wire.rs`): `SUPPORTED_SCHEMA_VERSIONS: &[u64] = &[1, SUPPORTED_SCHEMA_VERSION]`
  with `SUPPORTED_SCHEMA_VERSION: u64 = 2` — membership check, v3 fails closed. Beacon **main**
  today: `&[1, 2, SUPPORTED_SCHEMA_VERSION]` with `= 3` (PR#54 `87efe418`, merged 07-14 21:28Z).
- The bump: `git log -S` → `4e596867` "feat(ir): runtime-symbolic movement bounds + reshape
  targets (closes chelis#616) (#627)", 2026-07-09; `git tag --contains` → **first released in
  v0.15.0 (2026-07-10)** — four days before the roadmap. "The next chelis release breaks every
  dispatch" was therefore already stale at authorship; the break was live in v0.15.0–v0.16.1.
- Is it real today? At the released-artifact level, yes: released chelis (≥v0.15.0) × any
  released beacon (≤v0.1.8) fails closed. At the consumer level it is **dormant**: every shell
  pins chelis =0.14.0 or older (§4, pin topology) and FlukeBall's beacon lane pins a v1-emitting
  chelis worktree against a 0.1.6 beacon binary. Zero release-consuming victims exist right now.
- Roadmap citation resolution: `wire.rs:8-9` resolves exactly on the stale local beacon
  (`4579ebe`, the author's vantage) and on every released beacon; on beacon main today the
  constants are at :27-29 with the new values.

### C2 — the ignored guarding test

- Tested on: chelis origin/main. `beacon_e2e.rs:230-231`:
  `#[ignore] // requires CHELIS_BEACON_BIN` / `fn beacon_e2e_bs_call_vec_live_shim_round_trip…`;
  line 274: `assert_eq!(beacon_evidence["schema_version"].as_u64(), Some(2));` — verbatim.
- When/why ignored: `git blame` → all three lines born together in `4841bf98` (2026-06-29, "Add
  live Beacon shim round-trip gate (#561)"), never touched since; the v3 bump did not update the
  file. Six tests in the file carry the same ignore (lines 139/175/199/230/278/298).
- Nuance the roadmap omits: this is not a disabled test but a **documented environment-gated
  manual gate** (exact command at `docs/verification_stack_handover.md:136` and
  `docs/design/beacon_subprocess_shim.md:222`, per the repo's manual-gate policy). The substance
  stands: no CI run exercises the cross-repo seam (beacon PR#54's own text: "beacon CI never
  reads the chelis repo; the cross-repo catch is the chelis-side e2e gate, still open"), the gate
  would fail if actually run today, and the break shipped undetected.

### C3 — stale comments

- Tested on: chelis origin/main. `graph_extract.rs:14-15`: "validates `schema_version <=
  WIRE_DAG_SCHEMA_VERSION` (currently `2`…" (the text is on line 15; roadmap cites :14 where the
  sentence begins — trivial). `docs/design/phase2_seam_contract.md:57-58`: "asserts
  `schema_version == 1`" — two versions behind. Both confirmed verbatim.
- A **third** stale site the roadmap does not list: `docs/verification_stack_handover.md:94`
  "Current state: Beacon accepts schema v2."

### C4 — what v2→v3 actually changed

- Tested on: chelis commit `4e596867`; beacon-fresh HEAD + `87efe418`; beacon v0.1.8.
- The diff changed exactly four wire payloads — `Reshape.new_shape` `Vec<WireDimInfo>`→`Vec<WireRtDim>`,
  `Pad.padding` and `Shrink.bounds` `(usize,usize)`→`WireRtDim` pairs, `Stride.strides`
  `Vec<usize>`→`Vec<WireRtDim>` — plus the new `WireRtDim` enum with `Sym`. Note the pair-payload
  changes alter the wire form even for fully-literal pads/shrinks/strides.
- Beacon parses all four payloads only as opaque arrays (`required_array`, wire.rs:536-570) and
  has **no eval arm** for any of the four ops — they hit the `unsupported op` catch-alls in every
  lane. PR#54 shipped acceptance as exactly the predicted minor parser change (two constants +
  probe + tripwire + tests), with one honest refinement PR#54 itself surfaced: a v3 node
  *outside* the selected root's ancestor cone is dropped by `project_selected_root` rather than
  rejected (argued sound because projection is ancestor-closed; dedicated off-cone test added).
- Verdict: the roadmap's assertion **held up when executed**, including against an org-wide grep
  (the four ops appear elsewhere only as chelis lint diagnostics).

### C5 — "zero production callers" (the suspected-wrong claim)

Split verdict along the two possible readings:

- **Restricted to chelis's in-tree dispatch: CONFIRMED, and stronger than stated.** On chelis
  origin/main, `with_beacon` (engine_registry.rs:115) has **zero callers of any kind — not even
  tests** (beacon_e2e.rs constructs `BeaconShim::new`/`from_env` and calls `registry.register()`
  directly); `with_builtin_engines()` never registers beacon; all four production dispatch sites
  (property_runner.rs:767/959/2310, obligation_engine.rs:557) use `with_builtin_engines()` bare.
  `chelis prove` cannot route any goal to beacon today.
- **As stated ("every goal beacon ever discharged came from a test fixture"): WRONG, and already
  wrong at authorship.** FlukeBall's WC26 admission gate is a genuine out-of-tree consumer of
  `beacon_shim.rs` + `graph_extract.rs` + the `DischargeRegistry`: `agentic_authoring` runs
  `redteam_beacon_bound` on each final LLM-authored strategy; a generated Rust producer extracts
  a WireDag from the authored source via `graph_extract::box_range_goal_from_source_entry`,
  registers `BeaconShim::new` against the SHA-pinned arb-oracle beacon binary
  (`registry.register(Box::new(BeaconShim::new(&request.beacon_bin, store)))` — flukeball
  origin/main blob, line 3162), and dispatches. **338 of 397** persisted
  `beacon_hardening_report.json` artifacts (runs 2026-06-27/29; e.g. field32: 32/32 with 32
  distinct core hashes) record `selected_engine: "beacon"` with `proved` (true-bound) and
  `disproved` (red-team false-bound) verdicts at sound_approximate/sound_over_approximation —
  authored-strategy goals, not fixtures.
- Timeline: the wiring lived on flukeball `origin/world-cup` since 2026-06-23 and was merged to
  flukeball **origin/main** at 16:25:57Z on 07-14 — ~2 hours *before* the roadmap's initial
  beacon commit (18:27:06Z; both timestamps from `git show -s --format='%aI %cI'`). The merge
  message marks the experiment "closed, null result" (a betting-performance null, not a beacon
  failure) — beacon's first production traffic already happened and concluded.
- Provenance of the drift: the underlying review doc contains the accurate narrow sentence
  ("with_beacon at line 115 has no production caller", proof_stack_beacon_review.md:251); roadmap
  1.1 broadened it to "every goal beacon ever discharged came from a test fixture", and
  chelis#674 repeats the broadened form in its title.
- Note: FlukeBall bypasses `with_beacon` itself (uses `register()` directly, exactly as
  engine_registry's out-of-tree-consumer doc comment prescribes), and builds chelis-prove from a
  pinned local worktree (`DEFAULT_CHELIS_REPO=.flukeball-bugs-worktrees/chelis-486`,
  beacon_hardening.py:71-77) — a library consumer of chelis-prove's Rust API, not of the CLI.
- Public evidence: `cproof.ai/flukeball` returns HTTP 403 — unverifiable (§6).

### C6 — the tensor claim

- Tested on: beacon-fresh HEAD (rejection sites quoted from `wire.rs`, `oracle.rs`, `domain.rs`),
  chelis main (`ws9_pricer_seam.rs`, `builtins.rs`, `lower.rs`), flukeball remote + gh (issue #38).
- "Beacon rejects tensors" is **overbroad**. Beacon parses the entire WireDag op vocabulary,
  accepts tensor-typed DAGs, and *proves* the `tensor[n,f32]` bs_call_vec pricer graph at the ATM
  singleton (`Verdict::Proved`, arb_point_256, 1e-12 agreement with the scalar reference — a
  CI-exercised arb-oracle test, not latent). Since PR#46 ("ConstTensor domain support
  (FlukeBall#38)", 2026-07-03 — present in the author's own vantage) it accepts `ConstTensor`
  literals by collapsing `Vec<f64>` to a scalar [min,max] hull.
- The precise limitation: **the abstract state is one scalar range per node over named scalar
  input boxes** — no tensor-shaped state anywhere (per-element ConstTensor bounds discarded; one
  noise symbol per tensor in the zonotope lane; the Arb oracle and the new WI-H1 searcher reject
  even ConstTensor). Elementwise ops over tensor values evaluate under hull semantics; every
  shape-semantic op (Sum/MaxReduce/MinReduce/ProdReduce/ReduceWindow, Reshape/Permute/Expand/
  Pad/Shrink/Stride, BlasMatmul) passes the `is_verifier_targetable` gate (36 of 48 ops,
  wire.rs:116-156) but is rejected at evaluation as `unsupported op` in every lane (the evaluated
  surface is ~24 scalar/elementwise ops; `is_interval_modeled()`, wire.rs:212, code-affirms it).
- The FlukeBall "scalar_to_tensor beacon-unprovable" bug confirms this shape: WC26 cores are
  tensor[f32]-typed and beacon proved them; what failed was `scalar_to_tensor`-routed literal
  constants being "OPAQUE to the Beacon prover's interval reasoning" / "exhaust[ing] the prover"
  while mathematically identical scale-derived constants proved. flukeball#38 is about
  flukeball's own authoring-contract ban, not beacon refusing tensors. (Mechanism is
  flukeball-doc-attested, not re-executed — §6. chelis lowers `scalar_to_tensor` as identity,
  lower.rs:7458.)
- hydronnx→beacon: **never wired** — zero beacon/WireDag references on hydronnx remote main or
  the local checkout.
- The careful formulation already exists in the author's own sibling doc:
  proof_stack_beacon_review.md §2 — "input boxes are scalars and tensor ops are rejected is the
  accurate form". The roadmap flattened it; beacon#52's "every tensor operation is rejected" is
  itself slightly overbroad given ConstTensor.
- Accurate one-liner: *"Beacon's abstract state is one scalar range per node over named scalar
  input boxes: tensor-typed elementwise DAGs evaluate under hull semantics, but every
  shape-semantic tensor op — reductions, movement ops, matmul — is rejected as unsupported at
  evaluation, and the Arb/searcher lanes reject even ConstTensor."*

### C7 — the erf envelope range

- Tested on: beacon-fresh (JSON parsed; **tests executed**), stale beacon `4579ebe`, chelis
  origin/main, beacon-bakeoff dossier.
- **[-300,300] figure: CONFIRMED.** The committed artifact (byte-identical in both repos,
  sha256 `2470666b…`, SHA-pinned in beacon `erf_envelope.rs:10-21`) spans exactly [-300, 300] in
  3 contiguous Gappa-proved boxes.
- **"Outside is undocumented behavior": WRONG as a description of code behavior, and already
  wrong at the author's own vantage.** Outside ±300 the envelope **fails closed** (no clamp, no
  extrapolation): `box_for`→None→`certified_bound/range` None; all three beacon lanes convert
  that to explicit Unsupported with reason "erf argument range is not covered by the committed
  WI-13 envelope" (interval.rs:2062+~2267, zonotope.rs:603, oracle.rs:~616 with dedicated
  `UnsupportedCause::SpecialFunctionArgumentOutOfEnvelope`). Executed in the disposable clone
  (isolated `CARGO_TARGET_DIR`): `outside_committed_range_is_not_certified` (asserts ±301→None)
  and `certified_erf_range_refuses_uncovered_or_invalid_arguments` — both green, default
  features. Chelis's `out_of_range_is_none` read (not executed; authoritative checkout is
  read-only) and the consumer entry point is rustdoc-documented ("or None if x is outside the
  covered range"). Test and artifact both present at `4579ebe` and in chelis since PR#533
  (2026-06-26).
- **Saturation arms: implemented from day one.** Tails [-300,-3] and [3,300] are certified
  constant ∓/±1 arms (eps 2.226e-5, proof_kind gappa, "subdivision + certified Taylor model +
  Gappa" per provenance); contiguity at the ±3 joins is enforced by `is_well_formed` with both
  adjacent boxes' eps covering the join point. Roadmap 2.9's option 2 ("add true saturation arms
  with certified error at the joins") is already satisfied within the covered range.
- **±299.08 prior measurement: CONFIRMED.** Dossier records the route pricing box producing erf
  arguments to ±299.07879573850227 — ~0.92 of argument headroom inside the ±300 edge; a ~+0.3%
  widening of the route-pricing box would push goals out of the envelope (fail-closed, so sound
  but unprovable).
- Residual truth in 2.9 (why not flat WRONG): no user-facing prose states "±300 is a fail-closed
  contract" — beacon's own plan (rewritten the same day by PR#53) still says "behavior outside is
  currently undocumented" at `docs/beacon_plan.md:227`, and no lane-level integration test drives
  |x|>300 end-to-end (the shared gate is unit-tested). What remains of 2.9 is a
  documentation/contract statement plus arms beyond ±300 — a policy choice, not a behavioral
  defect or missing safety path.

### C8 — the falsified CROWN finding

- Tested on: beacon-fresh HEAD + refs `e440306`/`94000cb`/`ad0c15c`; stale beacon `4579ebe`.
- The 40-line fix is real and committed since 2026-07-10: `bound_less.py` registers a custom
  auto_LiRPA op computing **the sound {0,1} hull on undecided Less predicates** (exact 0/1 when
  input boxes decide the comparison), with a fail-closed backward pass treating the node as a
  constant interval. `test_bound_less.py` checks exact unit semantics, fail-closed raising, and
  dense-sampled (500/box) concrete-execution containment under IBP and backward — **an empirical
  oracle, not Arb** (README: "NOT Arb-certified"). Rigorous certification lives in the separate
  `hybrid_certify.py` leg (python-flint, 256-bit balls).
- Effect: auto_LiRPA both **accepts** the graph (previously all 6 `onnx::Less` rejected at
  conversion) and **proves** the pricer goals — [0,20] unsat in 1.06 s / 66 domains, [8.0,13.2]
  unsat in 1.33 s / 194 domains, within ~0.06 of the true range [8.058,13.147]. (Numbers are
  README-recorded; no `.log` files are committed for this round — §6.)
- `architecture.md:204`: the stale "net-new branch handling work" justification was present at
  the author's vantage **and** on remote main at authorship (verified at `e440306`; the text
  dates to `5b7c685`, 2026-06-22, and no commit between 06-29 and PR#53 touched the file). PR#53
  (merged 20:37Z same day, executing roadmap 0.4) rewrote it — the section now sits at ~:215-228
  re-grounded as "checkability, not external-tool incapability". **Fixed on beacon main only;
  v0.1.8 predates it.** Verdict STALE: true when written, corrected the same day by the roadmap's
  own execution.

### C9 — the decision record and acceptance

- Tested on: beacon-fresh HEAD; gh PR metadata.
- `docs/searcher_certifier_decision_2026_07_10.md` exists on beacon main (161 lines; landed via
  PR#50, merged 07-14 20:37Z). It records four decisions — (1) graduate the searcher/certifier
  split into Beacon proper, (2) adopt the undecided-predicate hull-mask, (3) de-scope WI-B4 for
  the scalar/finance fragment with tensor goals routed to the external engine, (4) keep the
  zonotope lane — plus WI-H1..H7.
- **Acceptance is self-marked.** The doc says "Decision accepted 2026-07-10 … Status: accepted
  direction (maintainer-endorsed)". Every commit touching the decision record, review doc,
  roadmap, and both experiment dirs is authored by Robert Ronan (robert@burnin.ai /
  rlronan@gmail.com); PR#50 was authored and self-merged by rlronan, zero reviews, zero comments.
  No independent human acceptance artifact exists anywhere. Same for the roadmap doc itself (§0).
- The review doc's §6 does contain recommendations 1–10 (the roadmap's claimed spine), verified
  one-to-one against the roadmap's items.

### C10 — experiment artifacts

- Tested on: beacon-fresh HEAD (`git ls-tree -r`).
- **All named files are committed**: abcrown-probe README.md, `bound_less.py`,
  `test_bound_less.py`, `hybrid_certify.py`, `run_probe.py`, and `wiredag2onnx.py` (a real
  310-line committed script — the pre-audit suspicion it was README-only is wrong); hard-claims
  README.md, `hard_claims.py` (all four functions at attack:131 / certify_point_refutation:166 /
  search:201 / certify_leaves:262), `gen_specs.py` (`build_mono_onnx`:50), four `abcrown_*.log`,
  `run2.log`, `hard_claims_results.json`, surrogate artifacts, ONNX/vnnlib/yaml specs.
  `bs_call_main.v0_8_recon.json` lives at `tests/fixtures/` (not `experiments/`), consistent with
  how the docs cite it.
- **Missing: `run.log`** (round-2's run-1: widest-dim DFS, no attack phase). Consequence: the
  run-1 side of README finding 4's branching comparison and the "B5 burned the 2M cap in run 1"
  event rest on prose only.

### C11 — the headline numbers

| # | number | verdict | where recorded |
|---|---|---|---|
| a | 98–99.8% certification rate | **PARTIALLY TRUE** | true for the 1-D/2-D subset (B1 0.19–1.9%, B2 0.8%, B3 0% refinement); roadmap drops its own source's qualifier — decision record says "(12% on the hardest 3-D relational claim)" and committed run2.log shows the 3-D range claims at **~21–26%** Arb-refinement rates (2012/3767, 6093/16559). Each retelling (README → decision record → roadmap) drops a qualifier. |
| b | 2M-eval cap vs 81 samples + 1 Arb eval in 13 ms | CONFIRMED | run2.log:63-65 to the digit (counterexample at (s=96,d=2), 0.013 s, certified point, VERDICT: REFUTED); cap is committed constant `MAX_EVALS = 2_000_000` (hard_claims.py:52) and demonstrably fires (surrogate_run.log:40). Caveat: "burned in run 1" is prose-only (run.log uncommitted). |
| c | 2.9k → 327k leaves as margin shrank 100× | CONFIRMED | run2.log + hard_claims_results.json: dmin 0.5/0.1/0.02/0.005 → 2,874/15,811/81,608/327,457 leaves (~5× per 5× margin cut). |
| d | 159 s / 970k domains NN-vs-formula | CONFIRMED | abcrown_bs_nn_gap.log: "Result: unsat", "Time: 158.748…", "969854 domains visited"; interval side capped at 2000005 evals with no verdict (surrogate_run.log:40). |
| e | 3.9 s self-composition verification | CONFIRMED | abcrown_bs_mono.log: "Time: 3.8921…", 193 domains; bs_mono.onnx + spec committed. |
| f | 2.7e-5 fidelity gate | **PARTIALLY TRUE** | gate mechanism committed and fail-closed at tol=5e-4 (gen_specs.py:106); the measured 2.7e-5 exists only in README/PR prose (gen_specs stdout not committed). What committed code guarantees is ≤5e-4. |

Round-**one** (2026-07-10) numbers — ~1.3 s CROWN verify, ~8 ms/46-leaf hybrid certification,
2.9e-15/2.7e-6 fidelity legs — have **no committed run logs** at all; README/PR-body only.

### C12 — capabilities overclaim

- Tested on: chelis-proof worktree @ main `5f86ddfa`. `prove_capabilities()` (prove/mod.rs:2903-2918):
  `let beacon_available = std::env::var("CHELIS_BEACON_BIN").is_ok();` (:2905), emitted at :2913 —
  no dispatchability gating anywhere. **The overclaim is bigger than the roadmap notes**: the
  same function's `"engine_registry"` array (:2917) hardcodes `"beacon_shim"` as a registered
  engine, but no production registry ever contains it. chelis#673 (OPEN) tracks the
  `beacon_available` half; the issue as filed does not mention the engine_registry entry. The
  `"schema_version": 1` at :2909 is the prove-JSON output schema, not WireDag — the roadmap does
  not conflate them.

### C13 — BoxRange surface

- Tested on: chelis main. `GoalShape` (discharge.rs:121-134) has exactly `Smt` and `BoxRange`;
  zero references under chelis-surf or chelis-cli; no prove flag. Producer: graph_extract;
  fitness consumer: BeaconShim only (beacon_shim.rs:464/468); ad_rail.rs:53: "There is NO
  BoxRange engine in-tree". Classification: not a defect — the enum's own doc comment says
  "TYPE ONLY in Phase 1 (no authoring syntax, no engine discharges it yet)" and the roadmap
  itself schedules syntax as item 1.2. Accurate claim; the "blocked on it" framing is
  prioritization.

### C14 — the certificate-lane bypass

- Tested on: beacon-fresh HEAD (interval.rs/certificate.rs/zonotope.rs byte-identical to the
  authorship vantage `4579ebe` — every citation holds at both). interval.rs:87:
  `if matches!(options.oracle, OracleMode::ZonotopeVerified) { return check_dag_with_zonotope_verified(…) }`
  sits before `check_structural_certificates` at :96, which has no other caller — the
  zonotope-verified lane can never emit the fully-trusted structural `Proved`. PR#55's searcher
  does not change this (searcher.rs:3,11-12: "decision-maker, never a proof authority", "Nothing
  here is wired into the dispatch or CLI lanes yet") — 2.8/WI-H7 remains fully open.
- Classification: **completeness gap inside an intended mode split, not a correctness hole** —
  the lane fails closed and self-diagnoses the exact limitation (zonotope.rs:2167-2169 maps
  terminal_limit `zero_width_assertion_not_independently_certifiable` → binding limiter
  `exact_zero_without_certificate`). All three certificate strings the roadmap names exist
  verbatim in emitted reports (`additive_ssa_residual_cancellation` as the evidence
  `certificate` field; its oracle is `exact_cancellation_real`).

### C15 — zonotope op coverage

- Tested on: beacon-fresh HEAD, both op sets re-derived independently. Interval lane: Load,
  Const, ConstTensor, Cast, Add, Mul, Neg, Erf, Exp, Log, Sqrt, Div, Recip, Abs, MaxElem, Sin,
  Cos, Tan, Atan, Floor, Ceil, CmpLt, Copy, Drop. Zonotope lane: strict subset **missing Abs,
  MaxElem, Sin, Cos, Tan, Atan, Floor, Ceil** (8 ops). Both lanes fail closed on unmatched ops;
  ~25 further wire ops are unsupported in both. The coverage fact is exactly as claimed;
  "backwards" (2.7) is an engineering-priority opinion — no soundness consequence.

### C16 — arb gate portability

- Tested on: beacon-fresh HEAD. arb_gate.py:10 `TARGET = "x86_64-unknown-linux-gnu"`, haswell/
  no-pie RUSTFLAGS, no platform branch; blob unchanged since its introducing commit. beacon#49
  (OPEN, filed 2026-07-10 — **predates the roadmap**, so this item was correctly excluded from
  the "unticketed" list) documents it with three fix options.

### C17 — WI-B4 / the CROWN centerpiece

- Tested on: beacon-fresh HEAD + `git show 4579ebe2:docs/beacon_plan.md`; chelis origin/main.
- Code half CONFIRMED: `src/linear_relaxation.rs` (2,190 lines) is exactly "an internal affine
  substrate" — `#![allow(dead_code)]`, all `pub(crate)`, private mod; scalar affine forms over
  Add/Neg/scalar-Mul plus opt-in (default-off) tangent/secant relaxations for
  Exp/Log/Sqrt/Recip/Div; CmpLt and all tensor ops rejected; the "backward" pass supports affine
  ops only ("backward linear relaxation supports affine ops only", :1204-1208) — **no backward
  CROWN through relaxed nonlinearities, no alpha optimization**; three cli.rs tests enforce it is
  unreachable from protocol/check/dispatch. beacon#44 was scoped as substrate from the start.
- Plan half: true at authorship — the pre-rewrite plan called the CROWN-lineage engine "the
  centerpiece" and the linear-relaxation domain "the research-grade core … what makes Beacon
  prove anything useful on a real graph". PR#53 re-crowned zonotope+Arb and added an explicit
  "No CROWN backward pass" non-claim. **Residual live defect: the chelis mirror
  `spec/design/beacon_plan.md` (origin/main v0.16.1, lines 11+21) still carries the falsified
  centerpiece framing**; the sync is in-flight as open chelis PR#677 (docs-only).

### C18 — "unticketed folklore"

- Tested on: gh issue/PR trackers (full `--state all` list scans; gh search is unreliable on
  private repos — beacon has exactly 10 issues ever), Downloads snapshot, beacon-fresh history.
- True at authorship: exhaustive pre-07-14 sweep confirms none of the three items had a ticket
  (chelis#488 requested the --capabilities surface itself; chelis#553/beacon#41/chelis#532 track
  adjacent closed round-trip/selector work; nothing on the tensor decision). The scoping was also
  correct by exclusion (beacon#49 already existed).
- **Self-resolved 65 minutes later**: chelis#673, chelis#674, beacon#52 filed 19:32:16-18Z in a
  3-second burst, "0.3 complete" commit 24 s after. All three are ticketing-only and remain OPEN;
  zero implementation PRs behind them. The WireDag v3 break itself was never ticketed anywhere —
  it went straight to code (beacon PR#54, same day, unreleased).

---

## 3. Dependency and infrastructure weight (Phase 2)

### Current state

**beacon** (beacon-fresh @ 5de54523; release assets via gh):

- Direct always-on deps: inari 2.0, clap 4, serde/serde_json, sha2, base64, thiserror. One
  feature: `arb-oracle` → arb-sys 0.3.6 (+flint-sys). Cargo.lock: 91 `[[package]]` entries;
  default resolve 84 packages.
- **The default build is not C-free**: inari's default `gmp` feature pulls gmp-mpfr-sys 1.7.1 +
  rug, compiling GMP+MPFR from vendored C source into every beacon binary — unconditional, not an
  arb-oracle cost. Fresh clone needs stable Rust ≥1.85, a C toolchain (+m4), and an x86-64-v3 CPU
  (`.cargo/config.toml` pins `-Ctarget-cpu=haswell` globally — beacon as configured is x86-64
  Linux-first while chelis releases darwin-arm64 binaries).
- **Releases ship no Rust binary at all**: v0.1.8 assets are only the chelis reef package
  (.chb + .tar.zst). The `chelis-beacon` binary consumed by the subprocess shim is
  build-from-source only, and the sound Arb lanes require a self-built `--features arb-oracle`
  binary no release workflow ever builds. The release gate builds/tests default features only;
  arb-oracle is covered by a separate CI job via arb_gate.py.
- beacon's own reef.toml pins `compiler = "=0.14.0"` — two minor versions behind chelis v0.16.1,
  so beacon's CI gate exercises a v2-emitting chelis, not current.

**chelis** (chelis-proof worktree @ main; `cargo metadata --locked --offline`, read-only):

- 28-member workspace; 464-entry Cargo.lock (424 unique). Default workspace resolve: 400
  packages with **zero solver/numeric-engine deps** — cvc5, z3, Clarabel/BLAS, carcara,
  arb/flint/gmp all absent; default `chelis prove` is the solver-free fuzz tier.
- Engines are feature-gated in chelis-prove: `smt` (cvc5-sys builds cvc5 C++ from source — g++,
  cmake, libclang; ENABLE_GPL=OFF), `z3` (links prebuilt system libz3, never builds), `clarabel`
  (+sdp-openblas/accelerate; explicitly not shipped), `carcara` (git-pinned + system GMP/MPFR),
  `arb` (vendors FLINT+Arb C source; -fPIC + isolated caches baked into .cargo/config).
- Fresh clone hard prerequisite: repo-root `.venv/` (uv, Python 3.11) because chelis-python links
  libpython via pyo3 — a plain `cargo build --workspace` fails without it. **The released binary
  does not need Python**: release builds `-p chelis-cli --features smt` only (no pyo3 in that
  graph), ships a stripped `chelis` with cvc5 statically embedded (re-proved on the exact shipped
  tarball by verify_release_smt.py), plus libchelis_runtime.a + headers; links no z3, no BLAS, no
  FLINT/Arb, no libpython.
- CI split: per-PR carries a fast cvc5 smoke; z3/clarabel/carcara/arb lanes (including the Arb
  erf-envelope certifiers) run in smt-full-prove.yml nightly + manual dispatch. (This supersedes
  the earlier WS-7/WI-13 "arb feature exercised by zero workflows" finding — the Arb cross-check
  now runs nightly, still not per-PR.)

### Proposed additions

| proposal | what it drags in | shipped vs side-path | honest size |
|---|---|---|---|
| auto_LiRPA adapter (4.1) | Python 3.11.x exactly (`~=3.11.0`, matches house uv convention); torch ≥2.0,<2.12; numpy≥2 + tqdm/graphviz/appdirs; adapter adds onnx (19.1 MB) + onnxruntime (18.6 MB). **Git-commit pin only** — PyPI channel is dead (0.3 from 2022 pins torch<1.13; GitHub master is 0.7.2, 2026-06-11) | **Side path, zero shipped weight** — by the roadmap's own text: out-of-tree repo, subprocess seam, distinct "float-computed" label, "manual gate forever". 4.2/4.3 inherit this: 4.3 consumes the adapter's leaf boxes as *data* through beacon's existing proof-bundle mechanism | CPU venv ≈250 MB download / ~0.9–1.0 GB on disk (torch cpu wheel 190 MB → 0.70 GB unpacked). CUDA lane ~2.7 GB download across ~17 wheels / ~5–7 GB disk — **not needed**: every measured 4.1 number ran device:cpu, single-threaded. The probe used the abcrown competition harness; the productionized adapter (library-only) is smaller |
| Luna (4.4 / Appendix A) | C++20 + CMake ≥3.24; libtorch 2.5.1+ (genuinely no Python interpreter — pybind11 optional, `--no-python`); FetchContent Protobuf/ONNX/GTest at build. BSD-3. 51 commits, zero releases → git pin. No Less/Where ops (confirmed) | **The one proposal that could become shipped weight**: linked in-process (its whole appeal — `LUNA_BUILD_EXECUTABLE OFF` yields luna::core for embedding), beacon's shipped closure gains the libtorch runtime; kept behind the 4.1-style subprocess seam, it stays side-path. The roadmap defers the linkage shape to the spike and never states it | libtorch 2.5.1 CPU linux: 169 MB download / **0.75 GB unpacked** (Appendix A's "multi-hundred-MB" is accurate); cu124 zip 2.52 GB |
| dReal4 (Appendix A #2) | Bazel ≥4.2.1 (2021-era) in CI or vendored build; prebuilt .debs only for Ubuntu 22.04/20.04; C/C++ deps: ibex (**LGPL-3** — omitted from Appendix A's Apache-2.0 cell), coinor-CLP, nlopt, gmp, spdlog, fmt, picosat, bison/flex. No torch. **Upstream dormant**: last tag 2021-06, last commit 2023-12-23 (Appendix A's "unconfirmed" flag resolves negative) | **Hinges on linkage, and the in-tree precedent is shipped**: qualifier-filling engines (cvc5, z3) are linked in-process and cvc5 ships in the release binary via `--features smt`. The `DeltaComplete` qualifier dReal4 fills already exists on both sides of the seam (beacon report.rs:33, chelis discharge.rs:249). A cvc5-style adoption = libdreal + ibex/CLP/nlopt/gmp in the shipped smt-lane binary + a Bazel stage in release CI; a subprocess side engine stays machine-local. The roadmap is silent on this for dReal4 | libdreal_.so + dep chain; no measured figure (never built here) |

**Plain answer to the mission's question**: 4.1/4.2/4.3 add zero weight to any shipped artifact —
the cost is a ~1 GB machine-local pinned Python venv plus custody of a git-pinned research dep.
The two places a Phase-4 adoption could quietly grow a release artifact are **Luna linked
in-process** (~0.75 GB libtorch runtime into beacon's closure) and **dReal4 adopted the cvc5
way** (libdreal+deps into the chelis smt-lane binary + Bazel in release CI). The roadmap makes
the side-path distinction explicitly for 4.1 and flags Luna's libtorch caveat, but never states
the linkage shape for Luna and is entirely silent on it for dReal4.

---

## 4. What the roadmap misses (Phase 3)

1. **FlukeBall's Beacon integration — the roadmap's largest blind spot.** Zero mentions of
   FlukeBall anywhere in the roadmap, yet: `beacon_hardening.py` (flukeball origin/main blob,
   3,301 lines) owns the chelis#439 BeaconShim experiment policy — synthesizes a cargo project
   depending on chelis-prove from a pinned checkout, extracts WireDags from model-authored
   home-probability cores, registers BeaconShim manually, dispatches with the arb_box_256 oracle,
   and maps TierBResult into strategy-admission gating with runtime-equivalence probes (1e-6);
   `clamp_smoke.py` (487 lines) sha256-pins the beacon binary and persists ~12 evidence artifacts
   per run; `prove_compat.py` (1,032 lines) classifies chelis prove capability by binary. 338
   real discharges recorded. Consequence for roadmap consumers: **chelis-prove's Rust API surface
   (graph_extract, DischargeRegistry, TierBResult, WireDagByteStore) already has an external
   library consumer — Phase 3's dispatch-architecture rework would refactor exactly that
   surface.** Note: the *current* flukeball_house/flukeball_2 constrained-authoring campaign lane
   requires no beacon proofs (flukeball_2 surfaces only `chelis prove --samples 16 --seed 7` as
   an author self-check); the beacon lane was the WC26 experiment, closed with a null betting
   result. The lane is also machine-bound: DEFAULT_BEACON_BIN/DEFAULT_CHELIS_REPO are absolute
   paths on this workstation.
2. **Four shells are mechanical consumers of `chelis prove --json` today** — shoals
   (scripts/prove_gate.py, `--tier smt-only`), c-note (c-note-exec service.rs shells out to
   `prove --json --tier`, proto parses the NDJSON, exemplar_gate.rs locks it), c-earchin (CI and
   release workflows run `chelis prove … --spans … --json` every run), economoist
   (scripts/prove_gate.py). Roadmap items 1.2 and Phase 3 would land on a prove surface with four
   downstream gates parsing its output — a compatibility surface the roadmap never inventories.
   Zero shells consume beacon.
3. **Pin topology / who the break actually bites** (verified against remote HEAD): every
   ecosystem shell — whale, shoals, c-note, school, hull, octant, calcify, c-earchin, economoist,
   hello-chelis, flukeball (main and world-cup) — pins chelis **=0.14.0** (emits WireDag v2);
   fluke-ball-wc-iso is at =0.10.1 (v1), hydronnx at =0.8.0 (predates the WireDag producer
   entirely). Nobody consumes chelis ≥0.15.0. The released-level break is therefore latent
   everywhere; it fires on the first shell pin-bump to ≥0.15.0 unless a beacon release ships the
   PR#54 fix first. FlukeBall's beacon binary pin is 0.1.6 (two behind v0.1.8).
4. **Roadmap items already landed on beacon main** (all unreleased): 0.1 beacon side (PR#54:
   [1,2,3] + `--schema-versions` + version_drift_guard tripwire incl. v4 loud-rejection); 0.3
   (issues filed); 0.4 (PR#53 plan/architecture rewrite; chelis mirror sync open as PR#677);
   2.1+2.3 (PR#55 f64 searcher + hull-mask; PR#57 notes an adversarial review closed a NaN-fold
   soundness bug pre-merge — same single-author cycle). 2.5 partially exists: a committed golden
   differential fixture (wi_h4_golden_leaves_bs_call.json, 5-loose/46-tight leaves, exact-f64
   pinned against the python prototype, regenerable via scripts/gen_wi_h4_golden.py). Untouched:
   0.2, 0.5, 1.1, 1.2, 2.2, 2.4, 2.6–2.9, Phases 3/4/5. Chelis-side execution is zero code (only
   docs PR#677; remote main unchanged since v0.16.1).
5. **A schema probe surface already existed** before PR#54: `chelis-beacon protocol --json`
   emitted `wire_dag_schema_versions` (derived from the same constant) at the author's own
   vantage. Roadmap 0.1(c) proposes "a `--schema-versions` probe flag" as if no probe surface
   existed; PR#54's flag is an argv-only convenience over the same constant.
6. **The erf saturation arms and the fail-closed out-of-range tests already existed** (C7) — the
   roadmap's 2.9 presents as open design what is mostly shipped, tested behavior; the genuinely
   open part is a prose contract statement.
7. **beacon's own toolchain lag**: beacon pins chelis =0.14.0 in its reef.toml, so its CI never
   exercises a v3-emitting chelis — the very break the roadmap leads with is invisible to
   beacon's own gate (PR#54's tripwire trips at pin-update time by design).

---

## 5. Classification (the consumer's buckets)

### Real defects — true, worth acting on

| item | what is actually broken (today, authoritative refs) |
|---|---|
| C1 core | Released chelis (≥v0.15.0) × released beacon (≤v0.1.8) seam fails closed. Fixed on beacon main, **unreleased**; latent only because all shells pin =0.14.0. |
| C2 | beacon_e2e.rs:274 asserts v2 — the documented manual gate fails if run; no CI exercises the cross-repo seam in either repo. |
| C3 | Three stale doc/comment sites on chelis main (graph_extract.rs:15, phase2_seam_contract.md:58, verification_stack_handover.md:94). |
| C5 narrow | `with_beacon` has zero callers of any kind; `chelis prove` cannot dispatch to beacon (chelis#674 tracks it — with an overbroad title). |
| C6 core | No tensor-shaped abstract state; all shape-semantic ops rejected at evaluation; Arb/searcher lanes reject even ConstTensor (beacon#52 tracks the decision). |
| C11a | The 98–99.8% figure is over-claimed relative to the committed 3-D data (21–26% refinement); qualifier dropped at each retelling. |
| C12 | --capabilities env-var overclaim, plus the un-noted hardcoded `"beacon_shim"` in the engine_registry array (chelis#673 covers only the first half). |
| C14 | ZonotopeVerified early-return bypasses the structural-certificate lane (fail-closed completeness gap, self-diagnosed; WI-H7 open). |
| C16 | arb_gate.py x86_64-linux-only (beacon#49, pre-existing). |
| C17 residual | chelis mirror spec/design/beacon_plan.md still carries the falsified CROWN-centerpiece framing on main (PR#677 in flight). |

### Stale / wrong — discard the framing, keep nothing

| item | why |
|---|---|
| C1 framing | "The next chelis release breaks every dispatch" — the break had already shipped four days earlier (v0.15.0); and beacon main already accepts v3. |
| C5 as stated | "Every goal beacon ever discharged came from a test fixture" — false before the roadmap was committed (FlukeBall WC26: 338 authored-strategy discharges, merged to flukeball main ~2 h prior; on world-cup since 06-23). |
| C6 one-liner | "Beacon rejects tensors" — beacon proves tensor-typed elementwise DAGs; the author's own review doc §2 contains the accurate form. |
| C7 "undocumented" | Out-of-range behavior is fail-closed, unit-tested in both repos (executed green in this audit), rustdoc-documented; saturation arms exist as certified data. |
| C8 | architecture.md:204 was fixed by PR#53 the same day (beacon main only). |
| C18 | "Unticketed folklore" — self-resolved by the author's own 0.3 execution 65 minutes after authorship. |

### True but preference — accurate facts, decisions not defects

| item | the preference being smuggled |
|---|---|
| C4 | "Minor parser change" held up, but is a judgment the roadmap states as fact — it happened to be verified by PR#54's execution. |
| C7 residual | "Document ±300 as a contract or extend arms" is a documentation-policy choice; the safety behavior exists. |
| C13 | BoxRange-without-syntax is an explicitly documented Phase-1 staging choice; the roadmap itself schedules the syntax. |
| C15 | The op-coverage gap is real; "backwards" is an engineering-priority opinion with no soundness consequence. |
| Phase 3/4 redesigns | The transformation-layer rework (3.x) and rent-the-searcher portfolio (4.x) are presented as following from defects; the defects they cite (C5 narrow, C6, C12) are real but do not compel these particular architectures. They rest on the decision record — which is single-author, self-accepted (C9). The dispatch surface they would refactor already has an external consumer (FlukeBall) and four shell gates parsing prove --json (§4.1–4.2). |

### Governance observation (fact, not verdict)

The roadmap's entire authority chain — review doc, decision record, experiments, the roadmap
itself, and the same-day fixes it cites as validation — is one person (rlronan), self-merged,
zero independent reviews, with acceptance self-marked. The measured evidence is largely genuine
and raw-log-backed (C10/C11); the *decisions* have no second signature anywhere.

---

## 6. Unverifiable

| item | reason |
|---|---|
| cproof.ai/flukeball public lane | HTTP 403 on fetch (both /flukeball and site root). FlukeBall's beacon usage was instead established from repo evidence. |
| C8 measured numbers (round-one abcrown probe: 1.06 s/66, 1.33 s/194, ~1.3 s CROWN, ~8 ms hybrid, fidelity legs) | Recorded only in README/PR prose; no `.log` committed for the 07-10 round; committed yaml configs carry absolute macOS paths from the original run. Mechanisms (bound_less.py, hybrid_certify.py) are committed and readable. |
| C11f's 2.7e-5 | Gate mechanism committed (fail-closed ≤5e-4); the specific measurement exists only in README/PR prose. |
| Run-1 side of the branching-heuristic comparison and "B5 burned the 2M cap in run 1" | run.log not committed. |
| flukeball scalar_to_tensor opacity *mechanism* | Attested by flukeball docs ("OPAQUE…", "exhaust the prover"); not re-executed here. The outcome (identical constants proving when scale-derived) is artifact-backed. |
| "Beacon and the Proof Orchestrator: A Technical Overview" (roadmap source 5) | Uncommitted by its own admission; not found on any local or remote ref; review doc §7b confirms "not committed to any repo as of 2026-07-14". Its seven corrections are auditable only against the review doc's summary of it. |

---

## Appendix: method

- **Planning exploration**: three read-only agents (checkout-topology map; roadmap/citation
  extraction; source-file map across all duplicate checkouts). The citation-extraction pass
  initially concluded the authority-chain documents did not exist — a conclusion this audit
  overturned by fetching remote state; it is preserved here as a live demonstration of the
  stale-checkout failure mode the mission warned about.
- **Remote truth**: fresh clones of `Chelis-Lang/beacon` (main `5de54523`) and
  `Chelis-Lang/flukeball` (main `a1ec6287` + branches) in the session scratchpad; `gh api` for
  commits/PRs/issues/releases/trees of all other org repos.
- **Execution**: 11 parallel investigation clusters (C1–C4 seam; C5 traffic; C6 tensor; C7 erf —
  including executing beacon's out-of-range tests under an isolated `CARGO_TARGET_DIR`; C8+C17
  CROWN; C9–C11 authority chain; C12–C15 surfaces/lanes; C16+C18 tickets; dependency weight ×2;
  misses sweep), each followed by an independent adversarial verifier instructed to refute the
  verdicts with its own commands from different angles (git-show from refs vs worktree reads,
  gh api vs gh CLI, independent BFS over cargo metadata, re-parsing raw logs). All verifier
  corrections (six ignored tests not five; 18:27Z not 16:22Z for the roadmap's initial commit;
  fluke-ball 401 *behind* not ahead; remote blob line numbers for beacon_hardening.py) are
  incorporated above.
- **Hygiene**: no fetches or writes in any `~/Documents/scratch` repo; no issue/PR interactions;
  builds/tests only in scratchpad clones; the only repo write is this file.
