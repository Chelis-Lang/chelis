# HANDOVER — Verification Stack Phase 3 (orchestrator)

Written 2026-06-26 for the agent picking up the team-lead/orchestrator role.
You are driving a multi-agent effort autonomously: dispatch agents, run the full
PR loop (open / CI / admin-merge), red-team at every workstream exit, escalate
only genuine decisions.

Plan file (authoritative, read it):
`~/.claude/plans/big-new-collection-of-effervescent-galaxy.md`

---

## 0. STANDING RULES (verbatim, do not violate)

- **Always commit AND push** to the feature branch on green. No local-only state.
  `main` is protected (admin squash-merge only).
- **Red-team before every workstream-exit / PR-to-main**: a FRESH-CONTEXT local
  subagent that EXECUTES tests (not source review, not main-thread). Never merge
  on green-CI alone. If a spawned RT comes back dead/broken, close it and respawn;
  if you must validate yourself, say so explicitly (it does NOT satisfy the
  red-team requirement — re-dispatch a working fresh subagent).
- **Never opine on time / cost / prioritization.** Surface scope concretely; let
  the user pick on architectural / correctness axes. Default = build the full
  architecturally-complete thing.
- **Never touch `~/Documents/scratch/beacon-bakeoff`** (Beacon = a separate agent;
  coordinate only via the frozen seam in `docs/design/phase2_seam_contract.md`).
- **No AI authorship trailer** in commits (the commit-msg hook rejects
  `Co-Authored-By: claude/anthropic`). Commit with no trailer.
- **No absolute paths, no em-dash in Rust literals.** Conventional commits.
- **Reproduce-the-bug before fixing.** Agents repeatedly found issues already
  stale-fixed; reproduce first.
- **SendMessage `summary` field is HARD-CAPPED at 200 chars** (content unlimited).
  Keep summaries ~140 chars or the call errors and you waste a round-trip.
- Build hygiene: every concurrent build uses an isolated `CARGO_TARGET_DIR`
  (per-worktree). macOS `simd_math_fused_kernel_*` is a recurring environmental
  flake — auto-rerun the failed `macOS Smoke` job, don't block on it.

---

## 1. WHERE THINGS STAND

`main` = `abb1a815` as of writing (it advances FAST — a separate team's
`pis-ws*` workstreams merge constantly; PRs #504/#505/#511/#514/#515/#518/#531
etc. are THEIRS, not ours). Always re-fetch before acting.

### Phase 3 progress
- **Wave 1 (soundness + shipping): DONE, merged, red-team-passed.** Closed
  #426 (SMT false-prove), #436/#461 (faithful goal field + a chelis fmt bug),
  #434/#425/#417 finance-lowering honesty, #422 (ship `--features smt` in the
  release tarball). Red-team checkpoint 1 closed.
- **Wave 2 (engine roster):**
  - **WS-5 Z3 NRA engine + dispatcher try-until-discharge fall-through: MERGED**
    (#485). Laundering guard structural; cross-engine oracle (Z3==cvc5);
    `classify_smt_outcome` extracted public.
  - **WS-6 Clarabel SoS exact-cert engine: MERGED** (#497). A CRITICAL latent
    false-Exact gap was caught + fixed (the goal<->cert binding guard) + verified
    through the real dispatcher. Follow-up #496 (canonical discharge-attribution
    evidence key) filed.
  - **WS-7 Sollya+Gappa erf proof-term envelope: IN FLIGHT — THE ONLY REMAINING
    WAVE-2 PIECE. See section 2 (this is your immediate work).**

Red-team checkpoint 2 closes when WS-7 merges.

### Active worktrees (mine)
- `chelis-ws7-sollya` (branch `ws7-sollya-arb`): the WS-7 author. Local HEAD
  `5ca018c9` (AHEAD of pushed `3f795961` — WS-7 is mid-fix on the HIGH, see §2).
- `chelis-rt-ws7b` (`3f795961`, detached): the GOOD WS-7 red team (rt-ws7b),
  on STANDBY for the focused re-check. 3.3G target. Keep until WS-7 merges.
- `chelis-rt-ws7` (`27eb958b`, detached): the DEAD prior RT — safe to
  `git worktree remove --force`.
- `.claude/worktrees/ws7a-frontend`, `ws7b-std`: NOT mine (the other team).

### Live agents (address via SendMessage)
- `ws7-sollya` — WS-7 author, implementing the HIGH fix (§2).
- `rt-ws7b` — WS-7 red team, PASSED-with-one-HIGH, on standby for the re-check.

---

## 2. IMMEDIATE NEXT STEP — WS-7 HIGH finding (merge-blocker)

WS-7's erf envelope (committed hex-float coeffs, all-arms machine-checkable Gappa
proof at branch `ws7-sollya-arb`) is SOUND on the committed data, but rt-ws7b's
final pass found a **HIGH** (proof-integrity + a CLAUDE.md honesty-invariant
violation), which blocks merge:

**The finding:** the eps-to-proof link is gated by NOTHING that runs in CI.
Exploit (RT proved + reverted): set committed `central_eps = 3e-7` (unsound —
below both the proved 5.683e-7 and true erf), regen the bundle sha256, leave the
`.gappa` untouched -> `generate_erf_proof.py --check-only` PASSES and the default
Rust lock tests PASS; ONLY `certify_erf_envelope validate` (the Arb cross-check)
catches it. And `--features arb` is enabled by ZERO CI workflows
(`ci.yml` runs chelis-prove only under default/smt/carcara). PLUS
`docs/erf_envelope_regen.md` + `certify_erf_envelope.rs:15-16` FALSELY claim Arb
"re-validates every box on every CI run."

**The agreed fix (already routed to ws7-sollya — verify it lands):**
1. **WS-7 (no new deps, always-on default gate):** in `--check-only` and/or a
   DEFAULT-feature Rust test (runs every PR), re-derive + assert
   `central_eps_math >= max over the 16 central_<k>.gappa of each file's ASSERTED
   goal bound`, AND `central_eps_f64_rounding == central_rounding.gappa's asserted
   bound`. Parse the bounds literally from the `.gappa` goal lines (rt-ws7b did
   exactly this; its method is in memory `project_ws7_wi13_erf_ci_gap`). Add a
   tamper test (eps shrunk below the proved bound MUST fail the default gate).
2. **WS-7:** FIX the docs (`erf_envelope_regen.md` + `certify_erf_envelope.rs`
   comments) to state what actually runs where.
3. **WS-7:** ALSO incorporate rt-ws7b's additive hardening tests (in
   `chelis-rt-ws7b/crates/chelis-prove/src/erf_envelope.rs`:
   `rt_ws7b_adversarial_codec_and_committed_coeffs` +
   `rt_ws7b_codec_format_panics_on_subnormal`) and REBASE onto current `main`
   (expect a `Cargo.toml [features]` keep-both: z3/clarabel/arb all live there).
4. **YOU (CI is orchestrator-owned):** add the **arb CI lane** so the Arb
   cross-check actually runs (making the doc claim true). Recipe from WS-7:
   apt `gcc m4 make` + `libgmp-dev libmpfr-dev`; env `CFLAGS=-fPIC` + the
   `*_CACHE` redirects are already baked into `.cargo/config.toml`; steps:
   `python scripts/generate_erf_proof.py --check-only` (cheap, just needs
   `gappa`) + `cargo run -p chelis-prove --features arb --bin certify_erf_envelope
   -- validate crates/chelis-prove/data/erf_envelope.json` +
   `cargo test -p chelis-prove --features arb`. Add it to the `smt-build` job
   (it already has the C toolchain) like the z3/clarabel lanes — see §4 for the
   exact pattern. `arb` vendors FLINT/Arb from C source (~1-2 min cold).
5. **Focused re-RT (rt-ws7b, it's on standby):** confirm the eps=3e-7 exploit now
   FAILS on the default lane (no arb), the arb lane runs in CI, the docs match.
6. **THEN merge WS-7** (you take over the merge: add the CI lane, open the PR,
   admin-merge — see §4).

**Verified-sound facts about WS-7 (so you don't re-litigate):** all 69 Gappa
proofs machine-check; the proof checker is non-vacuous (gappa exits 1 on false
goals); a parser-independent 200k-point sample (`float.fromhex` of the hex coeffs
+ native Horner + mpmath) finds ZERO violations (central margin +4.4e-9); coeffs
are exact hex-floats so serde's non-correctly-rounded float parser is OFF the
trust path; release links nothing new (arb/sollya/gappa absent from default+smt
trees; the hex codec is in-crate, no dep). The THREE earlier proof-integrity
seams (coeff<->proof binding; f64-eval Horner rounding; serde decimal-vs-f64)
were all ELIMINATED, not absorbed — that rigor is the point (user chose Option B
== Sollya+Gappa machine-checkable proof term over Arb numerical enclosure
specifically for it; see memory `feedback_proof_stack_gets_soundness`).

---

## 3. WAVE 3 (next phase, after checkpoint 2 / WS-7 merge)

Per the plan:
- **WS-8 — cut chelis 0.11.0 + cascade.** 0.11.0 now ships `--features smt`
  (cvc5) in the tarball. Use the locked release procedure: `scripts/
  bump_compiler_pins.py` (bundles the #448 chelis-std-bundle/locks regen); the
  minimal Hull pin-bump + `run_conformance.py` self-verify; CHANGELOG. Then
  cascade the new pin to nautilus, coral, octant, shoals, school, hull per
  `spec/design/shell_repo_contract.md` §7. nautilus/coral are far behind —
  expect real breakage; escalate soundness/real regressions, don't paper over.
  HANDOFF NOTE from ws4-ship: the release `release.yml` e2e `--tarball` verify
  runs on the `v*` TAG (not PR CI) — WATCH it on the first real 0.11.0 run.
  glibc2.31 cvc5-build prereqs = tomli + python3-venv + m4 + libopenblas-dev
  (cached under the smt-glibc231 key). See memory `project_chelis_verif_phase3`.
- **WS-9 — #440-A vectorized-elementwise pricer (Shoals) + Beacon handshake.**
  Pure-tensor-DAG BS pricer entry (broadcast erf/N(d1)/N(d2) elementwise over
  tensor[n], no vmap/scalar-lane) so a real pricer DAG is a WireDag root Beacon
  can verify. Coordinate with the beacon-bakeoff agent on the byte-seam target +
  the refuted `oracle_verified` flag (the #439 follow-up). Memory
  `project_wi3_440_scalar_host_blocker` explains why pruning isn't the fix.

Red-team checkpoint 3 after Wave 3.

---

## 4. MECHANICS — the merge pattern (do this for WS-7, then WS-8 releases)

The orchestrator owns CI (`.github/workflows/`). Agents give a CI-lane recipe;
YOU add the lane. The established pattern (used for z3 + clarabel):

1. Engine RT passes. Engine author rebases its branch onto current `main`
   (Cargo.toml `[features]` keep-both is the usual conflict).
2. You take over the merge to add the CI lane WITHOUT the worktree-lock trap:
   the branch is checked out in the author's worktree, so you CANNOT
   `git checkout <branch>` in the main checkout (it errors "used by worktree").
   Instead, from the main checkout:
   `git checkout -b <branch>-merge origin/<branch>` (new local name) ->
   edit `.github/workflows/ci.yml` (add the lane to the `smt-build` job, which
   already has libopenblas-dev/gmp/mpfr; KEEP the job NAME unchanged — it is a
   branch-protection required check) -> commit (no AI trailer) ->
   `git push --force-with-lease=<branch>:<remote-sha> origin <branch>-merge:<branch>`
   -> `git checkout main && git branch -D <branch>-merge`.
   (GOTCHA I hit: do NOT run `git reset --hard origin/<branch>` while on `main` —
   it silently moves your local `main` ref. Recover with `git branch -f main
   origin/main`.)
3. `gh pr create --base main --head <branch>` (the body must avoid backticks/`$()`
   in an unquoted `--body` — use single quotes or `--body-file`).
4. Poll CI to completion; auto-rerun the macOS SIMD flake; admin-merge:
   `gh pr merge <n> --squash --admin --subject "..." --body "..."`.
5. Clean worktrees + `git reset --hard origin/main` for local main.

The `smt-build` job (`ci.yml`, name "SMT Feature Build (Linux)") is where the
z3 + clarabel lanes live (it has the cvc5/BLAS toolchain). Add the arb lane there
too (or a sibling job; arb needs gcc/m4/make + the .cargo PIC/cache config).

---

## 5. OPEN ISSUES / FOLLOW-UPS (tracked, not blocking)

- **#434** — SMT can't lower transcendental finance (Black-Scholes erf). This is
  what the WS-7 erf proof term ultimately UNBLOCKS (a later faithful-discharge
  step consumes the committed envelope). Stays open.
- **#463** — boolean and/or at a property goal site doesn't lower to Tier B
  (pre-existing, conservative, never false-proves). For a later workstream.
- **#496** — canonical discharge-attribution evidence key across engines (cvc5
  uses `solver:`, Clarabel uses `engine:`). Cosmetic; deliberate cross-engine
  schema decision.
- **WS-7 optional (not blocking the merge):** the default-lane eps-backing gate
  in §2 IS the fix; rt-ws7b also noted `--check-only` could re-derive the full
  eps independently — covered by the §2 fix.
- macOS SIMD flake; `graph_extract.rs` cfg(test) unused-import; Carcara
  productionization — cross-cutting, fold opportunistically.

---

## 6. KEY MEMORY FILES (read these — `~/.claude/projects/.../memory/`)

- `project_chelis_verif_phase3` — Wave 1/2/3 map, WS-8 release handoff.
- `feedback_proof_stack_gets_soundness` — WHY the WS-7 rigor (proof term >
  enclosure; the user's overriding principle for this phase).
- `project_ws5_z3_engine`, `project_ws6_clarabel_sos`, `project_ws7_arb_oracle`,
  `project_ws7_wi13_erf_ci_gap` (rt-ws7b's finding + the parse method).
- `project_chelis_shell_cascade`, `project_chelis_010_release` — release/cascade
  mechanics for WS-8.
- `feedback_red_team_failures`, `feedback_subagent_visibility` — RT discipline +
  the supervision lesson (I once reported a dead RT as "building" by miscounting
  procs; VERIFY a subagent's target dir/file-writes actually grow before trusting
  "it's running").

---

## 7. ONE-LINE RESUME

Re-fetch `main`; check `ws7-sollya` pushed its HIGH fix (default eps-backing gate
+ docs correction + hardening tests + rebase); add the arb CI lane; re-engage
`rt-ws7b` for the focused re-check (eps=3e-7 must fail the default lane, arb lane
runs in CI, docs match); admin-merge WS-7 -> checkpoint 2 closes -> start Wave 3
(0.11.0 release + cascade).
