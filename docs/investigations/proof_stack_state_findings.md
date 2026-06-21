# Chelis Proof-Stack Current-State Findings

- **Date:** 2026-06-21
- **Branch:** proof-recon
- **Document type:** Reconnaissance (executed-evidence map, not advocacy)
- **Authoritative binary:** `/tmp/chelis-smt-bin/chelis` (`chelis 0.8.0`, cvc5 statically
  linked via `cvc5-sys` / `cvc5-rs 0.3.2`; no external cvc5 on PATH)

This map serves the next proof-stack planning round. Its single job is to separate a
**shipped** guarantee from a **planned** one without softening either direction. Where a
doc and the binary disagree, **the binary wins** and the doc is flagged stale.

---

## 1. Purpose, axes, taxonomy, and verification stance

### The three axes

- **Capability** — what can actually be *proven* about recognizable content today, and at
  what proof tier. "Recognizable" means a property a quant-econ academic or derivatives
  quant names unprompted (put-call parity, monotonicity in spot, non-negativity, Bellman
  contraction), not a synthetic toy.
- **Integrity** — whether a green badge *means what it claims*: no laundering of a weaker
  guarantee into a stronger one, non-vacuity guarded, per-assumption provenance threaded,
  build-contingency honest, machine-arithmetic faithfulness disclosed.
- **Frontier** — what is plumbed or planned but not yet proving on real content: Beacon's
  interval/region verification, the dispatcher/solver roster, contract-abstraction reach,
  the c-earchin bridge.

### State-verdict taxonomy

| Verdict | Meaning |
|---|---|
| **PROVING** | The mechanism discharges a property on real content today, with executed evidence. Always carries a tier+qualifier (see §2). |
| **PLUMBED** | The mechanism exists and runs, but is not exercised on recognizable content, or the abstraction is too weak to close the recognizable goal. Wired, not yet proving. |
| **SPEC-ONLY** | Named in a design/spec/comment as a future seam. No implementing code path. |
| **ABSENT** | No dependency, feature, or code reference. Not present. |

### Verification stance

Every finding below rests on **executed evidence** — a command and its captured output, or
a source `file:line` for a structural claim that cannot be exercised without rebuilding.
Source inspection alone is treated as proof *only* for build-gating claims the brief
forbids exercising (rebuilding the non-SMT binary in this checkout). Every per-area agent
and every skeptic independently ran the SMT-binary control **first** and all matched the
orchestrator's control output, so all discharge JSON cited here is trustworthy. No source,
spec, or example files were edited; all probe files were throwaway under `/tmp`.

---

## 2. The PROVING-is-never-bare rule (read first)

**No PROVING claim in this document stands bare.** Every PROVING verdict carries its
*tier* and *qualifier*, because the tiers are not interchangeable:

- **PROVING-at-SMT-exact** (`composite_verdict=proven`, `proof_tier=smt`, `samples=0`,
  `arith_model=real`, non-vacuity established) — cvc5 discharged the goal over the reals.
- **PROVING-at-composite-green** (`composite_verdict=proven_modulo_fuzz_validated_contract`)
  — the SMT *structure* is proven, but a consumed transcendental contract (e.g. the
  normal-CDF range/reflection) is discharged by **fuzz** (8192 samples), never machine-
  checked. Strictly weaker than bare `proven`.
- **PROVING-at-fuzz** (`proof_tier=fuzz`, `samples=100`) — a sampled pass only. Measure-
  zero falsehoods can survive this tier.

A `PROVING-at-fuzz` claim is **not** a `PROVING-at-SMT-exact` claim. Conflating them is the
central integrity hazard this map exists to prevent.

### The build-contingency finding (the reason this rule has teeth)

The orchestrator established, *before* fan-out, that **the entire SMT guarantee depends on
the `--features smt` build**:

- A **default cargo build (no `--features smt`) is fuzz-only.** On it, measure-zero-false
  properties **false-green**: `square_needle` (false only at x=12345) and a corrupted
  economoist property (false only at d=1.0) both returned
  `composite_verdict=proven_modulo_fuzz_validated_contract`, `samples=100`, `status=passed`,
  with **no `proof_tier`, no counterexample** — plain text "100/100 passed."
- After rebuild with `--features smt`, the **same inputs refute**: `proof_tier=smt`,
  `samples=0`, exact counterexamples (x=12345.0; d=1.0,g=0.0,r=1.0).

The non-SMT build silently emits a **proven-flavored badge**
(`proven_modulo_fuzz_validated_contract`) while greening falsehoods that real SMT would
catch. The SMT-binary control exists precisely to catch this. Independently corroborated in
source by §C: without `cfg(feature="smt")`, `solve_property` returns a hard
`TierBResult::Timeout` (`tier_b.rs:99-103`) and `base_verdict` gates `Proven` to
`PropertyTier::Smt` only (`property_runner.rs:183-206`), so bare `proven` is structurally
unreachable on a non-SMT build — the laundering is *down to the fuzz qualifier*, not up.

**Every green in this document is conditional on the `--features smt` binary having run.**
This is tracked upstream as **chelis#422**.

---

## 3. Executive summary and net verdict

### Genuine current capability frontier (one sentence)

On the SMT-feature binary, the proof stack **proves recognizable economic structure
(Bellman, Markov, Gordon DDM) as full unqualified SMT-exact green over the reals**, and
**recognizable derivatives identities (put-call parity, C≤S, delta∈[0,1]) as composite-
green** (SMT structure modulo a fuzz-validated normal-CDF contract) — both single-
application / fixed-dimension, with convergence, general-n, and the transcendental→real
link all held out.

### The single most valuable next increment

**Wire the contract-abstraction path's recognizable greens into the reference shell
(shoals) and add the missing normal-CDF monotonicity invariant + d1≥d2 coupling.** Today
the recognizable derivatives greens (parity, C≤S) prove only on hand-built `/tmp`
fixtures; shoals ships its real BS properties as `def → bool` test helpers with **zero
`@property` declarations** (`chelis prove` finds `total:0`). And the next tranche of
recognizable derivatives facts (non-negativity C≥0, intrinsic lower bound C≥max(S−K·D,0),
monotonicity in spot) is blocked solely by an absent monotonicity invariant in
`ContractAbstraction`, not by any solver ceiling. This single increment converts a working-
but-unadopted mechanism into a proving reference shell and extends the recognizable-proven
set.

### Integrity gaps that must not ship unaddressed

1. **Build-contingency false-green (chelis#422).** A non-SMT build emits a proven-flavored
   badge while fuzz-greening measure-zero falsehoods. Must ship with a hard guard: the
   green badge must be unattainable, or loudly degraded, without the SMT feature.
2. **SMT "proven" over-approximates machine arithmetic, undisclosed — and fuzz does not catch
   it either.** int32 lowers to *unbounded* `integer_sort` (no overflow) and f32/f64 to `Real`
   (no NaN/Inf/rounding). `(n>100000): n*n>0` and `(x==x):f32` both badge `proven` though
   false on hardware, with `arith_model` hard-coded `"real"` and no over-approximation
   assumption emitted. The fuzz tier is **not** a reliable counterweight: its generator emits
   no NaN (`x==x` passes at 100k samples) and int32-overflow fuzz is pathologically slow, so
   the machine-arithmetic falsehood is invisible to *both* tiers — the stronger SMT badge is
   simply the one that asserts a guarantee about it.
3. **Call-form predicate silently drops to fuzz (the C-Note finding, unrepaired in 0.8.0).**
   A top-level `@property` written `gte(mul(x,x),0.0)` drops to fuzz and **reports
   `status=passed`** in auto mode; the same math operator-form `(x*x)>=0.0` proves at SMT.
   A measure-zero-false property that SMT *disproves* is reported PASSED in call-form.
4. **Upstream SMT-collapse hole (chelis#426) — real, but economoist mitigates it; no shipped
   green is unsound (red-team-corrected).** The lowering collapses *the same def called twice*
   in a difference expression to a constant, which would false-prove. economoist avoids it by
   inlining the Bellman operator with fully distinct argument expressions; verified — all 8
   Bellman greens are sound (every false sibling refutes at smt). The gap that must not ship
   is that the mitigation is **manual** and must be re-verified at every future Bellman /
   difference-property call site. (An earlier draft wrongly claimed 3 shipped economoist
   greens were unsound; see §6.)

### One-line state per brief area

- **§A (proving frontier — economoist/shoals/c-note):** PROVING. economoist = 17 full
  SMT-exact greens; shoals = 3 composite-greens; c-note = no finance green (honest-pending).
- **§B (contract-driven SMT abstraction):** PROVING for parity & C≤S on `/tmp` fixtures;
  monotonicity SPEC-ONLY (contract exists, abstraction wiring does not); not adopted in shoals.
- **§C (verdict algebra & integrity backbone):** PROVING/sound (weakest-link, non-laundering)
  but a **total order, not a lattice**; non-vacuity blocking; build-gating sound; machine-
  arithmetic over-approximation is the headline integrity gap.
- **§D (inequality→SMT lowering):** PROVING for operator-form; call-form is PLUMBED-only and
  silently fuzz-drops; two exact gate sites named.
- **§E (Beacon in-flight):** PLUMBED — interval region bound-propagation *does* land on the
  BS pricer; no branch-and-bound; zonotope/CROWN SPEC-ONLY; no chelis discharge seam.
- **§F (orchestrator/dispatcher/solver roster):** cvc5 PROVING (in-process FFI); Z3 SPEC-ONLY;
  Sollya/Arb-FLINT/Clarabel/Carcara/MetiTarski ABSENT; no async deep-prove lane.
- **§G (verified AD & special functions):** grad emits a verified adjoint graph (PROVING as
  capability); AD-derived Greeks ORACLE-VALIDATED-ONLY (PLUMBED for the proof stack); delta∈[0,1]
  proven over an *abstracted* N, not the AD delta; special-function recognition SPEC-ONLY;
  perturbed-branch policy ABSENT.
- **§H (c-earchin):** PLUMBED/superseded — dead path; emits Deep `.dp`, fuzz-only, zero
  sibling references.
- **§2 (integrity backbone):** non-vacuity, per-assumption provenance, contract trust-gate,
  corruption-flip all PROVING-sound; build-contingency and machine-arithmetic the live gaps.

---

## 4. Per-area findings

Each finding carries its **axis**, **state verdict**, and (for PROVING) **tier+qualifier**
and **recognizable/synthetic** mark.

### Economoist-premise tension, resolved plainly

The brief's earlier reading saw economoist's properties as fuzz-qualified. **That was a
non-SMT-binary artifact.** On the prescribed SMT binary the brief's full-SMT-green claim is
**CONFIRMED**: all 17 economoist structural properties return
`composite_verdict=proven`, `proof_tier=smt`, `samples=0`, `arith_model=real`,
non-vacuity established. The fuzz-qualified reading came from running a default (non-SMT)
build — exactly the build-contingency failure mode of §2. (The skeptic's variable-capture
caveat in §6 narrows *which* of those greens are *sound*, but does not revive the
fuzz-qualified reading: the report shape is genuinely full-SMT-green.)

### Section A — Chelis proving frontier (economoist, shoals, c-note)

- **economoist Bellman (8 props):** Capability / **PROVING** / SMT-exact, proven,
  non-vacuity-established / **recognizable.** Monotonicity, sup-norm boundedness, per-output-
  state contraction (the Banach premise) at n=2; monotonicity+bound at n=3. Each non-vacuity
  is a real cvc5 SAT model. `economoist/properties/bellman.ch:18-89`.
- **economoist Gordon DDM (5 props):** Capability / **PROVING** / SMT-exact, proven /
  **recognizable.** P>0, monotone in dividend, dP/dr<0, dP/dg>0, monotone in discount rate —
  stated in polynomial factor-sign form (a sound restatement of the quotient property under
  the stated guards, not a relaxation). `growth.ch:12-31`.
- **economoist Markov (4 props):** Capability / **PROVING** / SMT-exact, proven, **exact real
  identity == 1.0 (no epsilon)** / **recognizable.** Mass conservation + non-negativity at
  n=2, n=3, calling the *exported* `next_mass`/`mass3_next`. `markov.ch:8-43`.
- **economoist scope boundary:** Integrity / **PROVING** (scope-boundary integrity). Only
  fixed n=2, n=3 single-application facts ship; n=3 per-component contraction **held out with
  zero stub** (`grep -c bellman3_contraction => 0`); convergence/general-n held out (need
  induction). Honest non-overclaiming, stated in property file + model docs + issue draft.
- **economoist corruption-refutes:** Integrity / **PROVING** / SMT-exact refuted-with-exact-
  counterexample. A measure-zero Gordon falsehood `((d-7)^2)>0` refutes with cex d=7.0.
  `prove_gate.py` end-to-end exit 0.
- **shoals composites (3 props):** Capability / **PROVING** / **`proven_modulo_fuzz_validated_contract`**
  (SMT structure + fuzz-validated normal-CDF contract; **NOT** unqualified) / **recognizable.**
  put_call_parity_reflection, call_upper_bounded_by_spot (C≤S), delta_in_unit_interval. The
  normal-CDF calls are abstracted to fresh cvc5 symbols; the range/reflection contract is
  fuzz-discharged (8192 samples, max_error 0.0–5.55e-17, tol 1e-10) plus a cvc5 non-vacuity
  SAT check. The transcendental link to the real A&S CDF is fuzz-asserted, never machine-
  checked. `shoals/properties/composites.ch:43-58`.
- **shoals Greek qualifiers:** Capability / **PROVING** (delta) + numerics-oracle (higher
  Greeks) / **recognizable.** delta∈[0,1] is composite-green (range contract only). gamma/
  vega/rho/theta/vanna/volga are **not SMT properties** — they are FD-vs-analytic / AD oracle-
  validated via `scripts/oracle_greeks_gate.py`. Correct division of labor.
- **shoals single-body + A&S sharing + oracle harness:** Integrity / **PROVING** (oracle gate
  PASS). `bs_call_scalar → bs_call_f64 → n_cdf64 → erf64` (one CDF behind price and all
  Greeks); erf64 uses A&S 7.1.26 coefficients byte-identical to `Nautilus.Special.erf`
  (reimplemented in f64 because upstream is f32-only). Oracle composite corpus gate: 3
  composite greens, 1 corrupted-coupling flip (failed), 1 unknown-contract (unsupported),
  `all_ok=true`. `shoals/src/pricing.ch:4-16,52-99`.
- **shoals integrity probes:** Integrity / **PROVING** / SMT-exact refuted (corrupted flip) +
  unsupported (unknown contract). `put_call_parity_corrupted` flips to `failed` with cvc5 cex;
  `delta_unknown_contract` returns `unsupported`. Green is contract-dependent, not laundered.
- **c-note badge states:** Integrity / **PROVING** (badge mapping exercised). Four `PanelState`
  variants (`c-note-proto/src/lib.rs:165-176`): Proven(green)/FuzzValidated(amber)/Disproved(red)/
  Unknown(grey). Notable: c-note's **only** green fixture is green via its opaque-type
  `@invariant` SMT obligation, **not** by any `@property` predicate reaching SMT.
- **c-note put_call_parity:** Frontier / **PLUMBED** / amber=fuzz-only, honest-pending awaiting
  contract abstraction (architectural) / **recognizable.** Raw log/sqrt/exp coupling; even
  operator-form rewrite returns `unsupported` ("variable log_sk has no declared cvc5 term").
  The fix is to adopt shoals' normal_cdf-contract abstraction. (Also: the c-note BS fixture
  body is a placeholder missing the normal CDF entirely — `mul(s,d1)` not `s*N(d1)` — so it
  is not even real Black-Scholes.) Doc and binary agree (honest).
- **c-note inline-mechanism diagnosis:** Frontier / **PLUMBED** / call-form gte/gt/mul drops to
  fuzz (invocation-form-fixable to SMT-exact) / **recognizable.** `(x*x)>=0.0` proves at SMT;
  `gte(mul(x,x),0.0)` is unsupported. `exp(x)>=1.0` when x≥0 **proves at SMT**, `exp(x)>1.0`
  refutes (cex x=0.0), `exp(x)+exp(-x)>=2.0` returns honest "smt unknown." See **stale-doc
  flag** below.

**Stale-doc flag (binary wins):** c-note `OPEN_WORK.md:7-8` and the fixture name
`unsupported_smt.ch` attribute the amber to "transcendentals do not lower to SMT," but the
binary proves isolated `exp(x)>0` and `exp(x)>=1` (x≥0) at SMT via cvc5's sound
transcendental theory. The true blockers are two distinct causes bundled under one
inaccurate sentence: (a) call-form arithmetic sugar (recoverable by rewrite) and (b) the
log/sqrt composite coupling in the full BS body (needs contract abstraction).

**Integrity nuance (economoist Bellman inlining) — mitigation verified effective:** Bellman
properties inline the operator arithmetic rather than calling `bellman_state*` defs, because
chelis#426 (subtracting two calls of the same if/then/else-bodied def collapses to a constant
and would false-prove). The inlined arithmetic is term-by-term the exported body, pinned
numerically by `tests/bellman.ch`. **The mitigation works:** the red team confirmed all 8
Bellman greens are sound — a false-monotonicity sibling built on the identical inlined
structure refutes (failed/smt/samples 0, cex), and every economoist `*_guards_satisfiable`
witness refutes. The footgun is real but *upstream*; it does not reach any economoist green.
The residual risk is that the inlining mitigation is manual (see §6).

### Section B — Contract-driven SMT abstraction

**Brief premise correction:** `@opaque`/`@invariant` is **not** the transcendental-injection
mechanism. `@opaque`/`@invariant` (`opaque.rs`) is for opaque-*type* record invariants
(Probability-style value-class types), proven through producer obligations. The
transcendental-as-opaque-SMT-symbol mechanism is a *separate* feature: the per-property
`with contract = "<id>"` option + `ContractAbstraction` in `property_runner/smt_lower.rs`.
It abstracts **only** `normal_cdf`, injecting **only** range (`0≤N≤1`) and reflection
(`N(-x)=1-N(x)`) preconditions.

- **put-call parity (`/tmp` fixture):** Capability / **PROVING** / SMT-exact-modulo-fuzz-
  contract / **recognizable.** Proven with reflection contract (auto-includes range). cvc5
  discharges the algebraic obligation; the two normal_cdf contracts are fuzz-validated, so
  honestly capped at `proven_modulo_fuzz_validated_contract`.
- **call upper bound C≤S (`/tmp` fixture):** Capability / **PROVING** / SMT-exact-modulo-fuzz-
  contract / **recognizable.** Range-only contract; load-bearing (under-constraint control
  without contract → unsupported).
- **non-negativity C≥0 and intrinsic lower bound C≥max(S−K·D,0):** Capability / **PLUMBED** /
  **recognizable.** Both correctly FAIL at SMT with counterexamples under range-only. They
  hold only because d1>d2 and N is monotone, but the abstraction treats d1,d2,N(d1),N(d2) as
  independent free vars with **no monotonicity invariant and no d1≥d2 coupling.** Honest
  failure; the recognizable property is not reachable today.
- **monotonicity in spot:** Frontier / **SPEC-ONLY** / **recognizable.** The
  `std.normal_cdf.monotonicity` contract **exists as a fuzz record** (`contracts.rs:64-71`)
  but is **not wired into `ContractAbstraction`** (no monotonicity reference in
  `smt_lower.rs`). Declaring it does not abstract the calls → unsupported. The gap between
  contract registry and abstraction layer.
- **over-abstraction / under-constraint differential controls:** Integrity / **PROVING** /
  SMT-exact. Every green degrades to failed/unsupported when its load-bearing invariant is
  removed; every deliberately-false goal fails with a counterexample. **No laundering.**
- **shoals' real BS properties reach the prover?** Frontier / **PLUMBED** / **recognizable.**
  **NO.** They are `def → bool` test helpers consumed only by `chelis test`; `chelis prove`
  finds **zero `@property` declarations** (`total:0`). The recognizable-green is demonstrated
  only on synthetic `/tmp` fixtures, not the shipped reference shell.
- **contract trust gate:** Integrity / **PROVING** / SMT-exact. `ContractAbstraction` binds
  only the genuinely-linked `Std.Contracts.normal_cdf`; a bare file or a locally-defined
  linker-shaped name does **not** bind (locked by test). No spoofing path.

### Section C — Verdict algebra and integrity backbone

- **rollup soundness vs lattice claim:** Integrity / **PROVING** (structural). The rollup is
  **weakest-link minimum-soundness** (`rollup_composite` folds via `weakest()` over
  `weakness_rank`: Proven=0 < ProvenModuloFuzz=1 < ProvenModuloAxiom=2 < Unsupported=3 <
  Invalid=4 < Failed=5). It can only **degrade**, never upgrade — no laundering up.
  **CAVEAT:** it is a **total order on a single u8 scalar (a chain), NOT a lattice** of
  incomparable qualifiers. The brief's "lattice not chain" framing is **not literally
  satisfied**; a mixed fuzz(1)+axiom(2) discharge folds to the single badge
  `proven_modulo_asserted_axiom`, *dropping* the fuzz qualifier from the badge — the
  opposite of a union. The *safety* property (anti-laundering) holds; the *structure* does
  not. `composition.rs:63-80,284-291`.
- **qualifier distinguishability:** Integrity / **PROVING** (SMT/fuzz qualifiers) +
  **PLUMBED** (axiom). `composite_verdict` distinguishes the variants and each assumption
  carries `discharge.method`. **Collapse caveat:** `status` label and exit code bucket all
  three `proven*` qualifiers to "passed" — a CI keying on exit code alone cannot tell proven
  from fuzz-qualified. **`proven_modulo_asserted_axiom` is PLUMBED only** — no live (non-
  test) constructor of `DischargeMethod::Axiom` exists.
- **non-vacuity guard:** Integrity / **PROVING** / SMT-exact. **BLOCKING and exercised.** A
  jointly-unsat precondition set → `composite_verdict=invalid`, `status=unsupported`,
  `passed=0` (`non_vacuity.result=unsat`). unknown/timeout → unsupported, never green.
  Vacuous preconditions cannot prove anything. (Independently hammered by a skeptic across
  every tier and flag combination — see §6, claim holds.)
- **per-assumption provenance:** Integrity / **PROVING**. Every `AssumptionRecord` carries
  `discharge:{method,evidence}`, `non_vacuity:{status,evidence}`, and for std contracts
  `source_type`/`producer`. The CLI emits the full `assumptions[]` array. Injection is not
  tier-less.
- **build-contingency:** Integrity / **PROVING** (source-verified; cannot be exercised without
  rebuilding, which the brief forbids). A non-SMT build **cannot emit `proven`** — it caps at
  `proven_modulo_fuzz_validated_contract`, which is the *correct* (non-laundering-up)
  behavior, plus a stderr warning. See §2 for the orchestrator's executed confirmation that
  this still **false-greens measure-zero falsehoods at the fuzz tier** — the gap is real even
  though the badge cannot inflate to bare `proven`.
- **machine-arithmetic faithfulness:** Integrity / **PROVING** badge that is
  **SMT-unqualified vs hardware** / **recognizable.** int32 → unbounded `integer_sort` (no
  overflow); f32/f64 → `Real` (no NaN/Inf/rounding). `(n>100000):n*n>0` and `(x==x):f32`
  badge `proven` though false on hardware. **Neither tier catches it:** fuzz emits no NaN
  (`x==x` passes at 100k samples) and int32-overflow fuzz is pathologically slow, so the
  falsehood is invisible to fuzz too — the SMT badge is merely the one that *asserts* a
  guarantee over it. `arith_model` hard-coded `"real"`, no over-approximation assumption
  emitted. **The most consequential gap for a derivatives quant trusting a green badge.**
- **std-contract base records:** Integrity / **PROVING** with axiom-grade caveat /
  **recognizable.** SMT contract base records (`exp.positivity`, `exp.zero`) are **trusted
  constants** `{status:proved,solver:cvc5}` that are **never re-solved** — axiom-grade
  soundness wearing an `smt` method tag. Mitigated: the body postcondition is still solved
  (false bodies fail with counterexamples), and fuzz contracts correctly cap the consumer.

### Section D — Inequality-to-SMT lowering (operator-form vs call-form)

- **operator-form proves, call-form drops (the C-Note finding, still true in 0.8.0):**
  Capability / **PROVING** for operator-form, **PLUMBED** for call-form / **recognizable**
  (on call-option payoff non-negativity) and synthetic (on the isolation probes). The *same*
  true-over-reals inequality: `(a-b)>0.0` proves at SMT; `gt(sub(a,b),0.0)` drops to fuzz
  (auto) / unsupported (smt-only). **The drop is silent** (`status=passed` in auto). Worse
  than mere degradation: a measure-zero-*false* property that SMT *disproves* (cex x=12345)
  is reported PASSED in call-form because fuzz misses the counterexample.
- **two exact gate sites:** GATE 1 — `property_runner/smt_lower.rs:172-216` `surf_expr_to_smt`
  has no `Expr::Apply` arm in boolean position (`_ => None` at 214), rejecting any top-level
  call-form predicate (gt/gte/lt/lte/eq/neq). GATE 2 — `surf_arith` (`smt_lower.rs:296`)
  lowers non-whitelisted `Expr::Apply` to an *uninterpreted* function, then
  `classify_inlineability` returns None unless the name is in
  `chelis_pred::INTRINSIC_WHITELIST` = `[abs,min,max,sqrt,exp,log,sin,cos]`; sub/mul/add/div
  are absent. **Skeptic correction (see §6):** the brief's named site `tier_b_lower.rs` is
  **wrong** — that is the opaque-invariant producer-obligation path, which *does* handle
  call-form gte/mul. The actual top-level `@property` drop site is `property_runner/smt_lower.rs`.
- **green boundary, cross-checked on a quant property:** Capability / **PROVING** for
  operator-form / **recognizable.** GREEN: operator-form comparisons + arithmetic, and
  whitelisted intrinsic call-forms. AMBER (silently fuzz): any top-level call-form predicate;
  any non-whitelisted arith builtin call-form. Call-option payoff `max((s-k),0.0)>=0.0`
  proves at SMT; the identical `gte(max(sub(s,k),0.0),0.0)` is unsupported. The recognizable
  property is provable *only if the author happens to spell it operator-form.*

### Section E — Beacon in-flight state (Frontier)

- **graph-extraction seam:** Frontier / **SPEC-ONLY.** chelis still lowers exclusively to
  `SmtProperty`/`SmtExpr`; no IR-retaining property-to-DAG path. Beacon consumes a serialized
  **recon-captured DAG JSON fixture** out-of-band (`chelis-risc-dag/v0.8.0` exists only in
  `beacon/src/wire.rs:8`, emitted nowhere in chelis). The IR-discarded-at-the-SMT-boundary
  state is unchanged.
- **discharge-engine interface:** Frontier / **SPEC-ONLY.** No `DischargeEngine` trait, no
  engine-independent goal. `CompositeVerdict` has 6 variants, `DischargeMethod` 3 — none is
  a sound-over-approximation / interval method. Beacon's `Verdict::Proved` /
  `ProvedOracleUnverified` are crate-local and cannot flow into chelis.
- **box / output-range goal schema:** Frontier / **PLUMBED.** Native interval-box input +
  output-range goal exists and is exercised (`beacon.goal.v1`), but only Beacon-local
  (external JSON), not chelis property syntax.
- **domain ladder:** Frontier / **PLUMBED.** Only the **interval** domain is landed.
  Zonotope (WI-B3), CROWN/linear-relaxation (WI-B4), branch-and-bound (WI-B5) are SPEC-ONLY.
- **LOAD-BEARING: does Beacon bound over a REGION today?** Capability / **PROVING** /
  interval-domain region bound-propagation, Arb-256 outward-rounded oracle, **no branch-and-
  bound**, real-valued (no roundoff-soundness) / **recognizable.** **YES — real region bound-
  propagation, not point evaluation.** On the BS call pricer the output interval width scales
  with the input spot-box width: spot=100 → point [10.4506,10.4506]; [99,101] → ≈[5.33,15.59]
  (width ~10.3); [98,102] → ≈[0.25,20.73] (width ~20.5) — still a positive lower bound at this
  width; interval wrapping (a uselessly-wide / negative bound) appears only at wider boxes and
  is exhibited directly by `x*x` over [-2,3] → [-8.5,9.0] vs the true [0,9]. Bounded by the
  absence of branch-and-bound: a desk-width [95,105] box
  returns `unsupported` at the first straddling erf64 `CmpLt` (node 140). Region-bounding is
  real **only while the box is narrow enough that all 6 input-dependent erf64 branches
  resolve to one side.** (The §6 skeptic confirms and refutes the contrary claim; the only
  nuance is that the arb-box `Proved` path is **off by default** — opt-in `arb-oracle`
  feature.)
- **AD as verification target:** Frontier / **SPEC-ONLY.** No grad/adjoint graph is
  dispatched to Beacon; verified-Greeks (WI-B8) depends on the unbuilt CROWN rung.
- **§C laundering connection:** Integrity / **PLUMBED.** No sound-over-approximation badge
  exists in the upstream algebra → **no laundering vector through chelis.** The mirror risk:
  there is also no engine seam, so when WI-B8 wires Beacon in, the discharge interface must
  *add* a sound-over-approximation verdict + its real-valued/branch-coverage qualifier set.

**Doc/binary agreement (red-team-corrected — no stale-doc contradiction here):** the shipped
feature binary implements and passes the arb-box non-singleton region path (`oracle.rs`
`eval_dag_box` / `LoadMode::Box`, the `arb_box_*` tests, live BS-pricer region runs), and
Beacon's own docs already describe it: `architecture.md` lists **only** Sollya / zonotope /
CROWN / branch-and-bound / FP-roundoff under "Deferred By Design" — **not** the arb-box region
oracle, which its Phase-1C section describes as shipped. An earlier draft claimed
`architecture.md` filed the arb-box oracle under "Deferred By Design" and that the binary thus
"reverses its README"; that was **wrong** — doc and code agree. The only genuine subtlety is
naming: the README's "sound `proved` for singleton parsed-f64 only" caveat refers to the
arb-**point** oracle (`--oracle arb-point`), a different lane from the shipped arb-**box**
region path. (Environmental note: one agent could not link the arb-oracle build under default
flags — a flint-sys `-fPIC` / `R_X86_64_32` relocation — but the red team linked it with
`scripts/gate.py`'s `-Clink-arg=-no-pie` and ran 10 `arb_box_*` tests green; an environment
link issue, not a code defect.)

### Section F — Orchestrator, dispatcher, solver roster (Frontier)

- **dispatcher:** Frontier / **PROVING** (architecture). A shape-based router exists, but it
  routes between **tiers** (type → SMT → fuzz) keyed on `SmtAmenability`
  {Linear,Polynomial,Transcendental,Opaque}, **not between solver engines.** When SMT is
  attempted there is exactly one engine.
- **cvc5:** Frontier / **PROVING** / SMT-exact over `arith_model=real` (QF_NRA/QF_NRAT).
  Feature `smt = ["cvc5-rs"]` (0.3.2). **In-process FFI** — `cvc5-sys` statically compiled
  from C++ source into the binary (static C-ABI symbols, no dynamic lib, no external binary
  call). The optional `worker` subprocess is **crash-isolation** (re-exec of the same binary
  running the same in-process solve), not an external solver call.
- **Z3:** Frontier / **SPEC-ONLY.** Named only in a design comment behind the `Solver` trait
  ("future Z3, easy-smt subprocess"). No dep, no feature, no impl.
- **Sollya, Arb/FLINT, Clarabel, Carcara, MetiTarski:** Frontier / **ABSENT.** Zero deps,
  features, or code references in any workspace `Cargo.toml`. (cvc5's internal AlfPrinter is
  cvc5-internal, not a Carcara integration.)
- **deep-prove lane:** Frontier / **SPEC-ONLY.** No async/cancellable/result-cached lane.
  One synchronous prove path; `worker.rs` is crash isolation, not a deep lane.

**Cross-cutting soundness qualifier:** every SMT discharge carries `arith_model="real"`. f32
properties are proven **over the reals, not over IEEE-754 float32** (no rounding, overflow,
NaN). A `proof_tier:smt / proven` on an f32 property is **real-exact, not float-exact** —
every recognizable-PROVEN claim inherits this reals caveat. Transcendental SMT coverage is
narrow and cvc5-only: {exp,sqrt,sin,cos,abs,min,max}; **`log` is deliberately unsupported**
(no cvc5 LOG kind) and routes to fuzz.

### Section G — Verified AD and special functions

- **grad emits adjoint RISC graph:** Capability / **PROVING** (code-construction capability;
  the graph is structurally verified by `crate::verify`). Real reverse-mode AD with a
  structured rejection enum. `grad.rs:310,321-420`.
- **verified bounded first-order Greeks reachable via the proof stack?** Capability /
  **PLUMBED** / **recognizable.** **No.** No `@property` in shoals puts `grad(...)` inside a
  body that discharges through the proof stack. The AD-derived Greeks are **oracle-validated
  only** (FD/analytic via `chelis eval` to ~1e-9..1e-4). The grad graph computes correct
  Greeks; a bounded-Greek property over the AD value does not discharge.
- **delta∈[0,1] proven, over what?** Capability / **PROVING** /
  `proven_modulo_fuzz_validated_contract` / **recognizable.** Proven over an **abstracted
  `normal_cdf` symbol** carrying the fuzz-validated range contract — **NOT** over the AD-
  derived delta `grad(bs_call,wrt=s)` nor the real erf64 body. A different object than the AD
  delta. Integrity-guarded (corruption probe flips to failed; unknown contract → unsupported).
- **local A&S erf with input-dependent CmpLt branches:** Capability / **PROVING** (source-
  confirmed fact). erf64 = A&S 7.1.26 with two input-dependent branches (`if lt(ax,small)`,
  `if lt(x,0.0)`); these lower to `RiscOp::CmpLt`. AD differentiates via masked-select.
- **first-class special-function recognition layer:** Frontier / **SPEC-ONLY.** What ships is
  a **closed hardcoded contract allowlist** (`contracts.rs`: 8 fact-IDs — normal_cdf
  range/reflection/monotonicity, exp positivity/monotonicity/zero, log monotonicity/one —
  plus the `normal_cdf` impl/linker name), each fuzz-validated in Rust. erf/erfc are **not**
  in the SMT-lowerable set, so a property over the real erf64 body
  cannot discharge at Tier B — hence the corpus abstracts `normal_cdf`. Generalizing into
  auto-derived contracts for arbitrary user transcendentals is roadmap.
- **perturbed-branch policy:** Capability / **ABSENT.** No perturbed/smoothed/branch-
  sensitivity policy anywhere. AD treats the erf breakpoints as differentiable a.e. with zero
  gradient at the discontinuity and no perturbation. The proof stack offers no policy that
  would make a verified bounded-sensitivity claim across the branch sound.

### Section 2 — Integrity backbone (consolidated)

- **SMT-binary control (the substrate of trust):** Integrity / **PROVING** / SMT-exact. The
  measure-zero polynomial control refutes with `proof_tier=smt`, `samples=0`, cex x=12345.0.
  Run independently by *every* agent and *every* skeptic; all matched. cvc5 statically linked.
- **non-vacuity, per-assumption provenance, contract trust-gate, corruption-flip:** all
  **PROVING-sound** (detailed in §C and §B above and §6 below).
- **build-contingency (chelis#422) and machine-arithmetic over-approximation:** the two live
  integrity gaps (see §2 and §C). The released chelis tarball is documented (economoist
  `UPSTREAM_BUGS.md:48-57`) as fuzz-only — every green is conditional on the SMT-feature build.
- **oracle harnesses:** Integrity / **PROVING** (skeptic-confirmed RUN to green). economoist
  `oracle_harness.py` (exit 0; 8 cases + grad signs −800/+800 tied to the proven sign
  properties + 15-test suite) and shoals `oracle_greeks_gate.py` (exit 0; 14/14 groups, 63
  cells, a-priori non-self-widening bands). Both wired into local gate / nightly.

---

## 5. The recognizable-PROVEN set (headline output)

The point is *which named properties prove, and at what tier* — not how many.

### A. Full unqualified SMT-exact green (`proven`, `proof_tier=smt`, `samples=0`, non-vacuity-established, NO contract/fuzz) — economoist

These are recognizable to a quant-econ academic and pass economoist's `prove_gate.py` (which
enforces unqualified-green + non-vacuity + scope honesty):

- **Bellman optimality operator** — monotonicity, sup-norm boundedness, per-output-state
  contraction (the Banach premise) at n=2; monotonicity + bound at n=3. (8 properties.)
- **Gordon DDM** — P>0, monotone in dividend, dP/dr<0, dP/dg>0, monotone in discount rate
  (polynomial factor-sign form). (5 properties.)
- **Markov step** — exact mass conservation (==1.0, no epsilon) and non-negativity
  preservation at n=2 and n=3. (4 properties.)

All 17 are single-application / fixed-dimension; **convergence, general-n, and the n=3
per-component contraction are honestly held out.**

**Soundness note — all 17 are sound (red-team-corrected):** a real upstream SMT-lowering
hole (chelis#426) collapses *the same def called twice* in a difference expression to a
constant, which would false-prove. economoist **avoids** it by inlining the Bellman operator
with fully distinct `v*`/`w*` argument expressions (no def called twice). Verified: all 8
Bellman greens discharge proven/smt/samples 0, and a constructed false-monotonicity sibling
on the identical inlined structure **refutes** (failed/smt, cex). The earlier draft's "3 of
17 unsound" was wrong on every particular — the named properties `bellman_contraction`/
`bellman_contraction_lower` do not exist (real names `bellman_contraction_state0/_state1/…`)
and its cited repro does not reproduce. **0 of 17 unsound.**

### B. Composite green (`proven_modulo_fuzz_validated_contract`: SMT structure + fuzz-validated normal-CDF contract) — shoals

The headline recognizable derivatives facts. **NOT** unqualified `proven` — the
transcendental link to the real A&S CDF is fuzz-asserted (8192 samples), never machine-
checked. shoals is explicit in-source; integrity probes confirm contract-dependence:

- **Put-call parity** (reflection contract).
- **Call ≤ spot intrinsic upper bound** (range contract).
- **Call delta ∈ [0,1]** (range contract).

(These also prove on hand-built `/tmp` abstraction fixtures via §B. In the *shipped* shoals
shell they are reachable only via `properties/composites.ch`; shoals' other BS facts are
`def → bool` test helpers with no `@property`, reaching neither SMT nor fuzz via `prove`.)

### C. Not proven (amber/fuzz or unsupported) — c-note + the recognizable derivatives gaps

- **c-note:** no finance property reaches green; the only green fixture is the synthetic
  opaque-invariant, and even there the green is the SMT *invariant obligation*, not the
  `@property` predicate.
- **Recognizable-but-not-yet-provable (abstraction too weak):** non-negativity C≥0, intrinsic
  lower bound C≥max(S−K·D,0), monotonicity in spot — blocked by the missing normal-CDF
  monotonicity invariant + d1≥d2 coupling, not by a solver ceiling.

### Frontier (Beacon, recognizable, region-level not SMT)

- **Black-Scholes call-price output range over a spot interval** — interval region bound-
  propagation, Arb-256 oracle, `proved_oracle_unverified` (interval-only) by default or
  `proved` under the arb-oracle feature, *only for branch-non-straddling boxes*.

---

## 6. Integrity verdict

**Is the verdict algebra a true non-laundering lattice?** It is a **sound non-laundering
chain, not a lattice.** Weakest-link minimum-soundness holds (the fold can only degrade;
verified in source + by `/tmp/mixed_two_fuzz.ch` → `proven_modulo_fuzz_validated_contract`,
never `proven`). But it is a total order on a single u8 `weakness_rank`; a mixed fuzz+axiom
set folds to a single badge and *drops* the other qualifier. The safety property the lattice
was meant to provide IS met; the lattice structure is not. **Skeptic verdict: PARTIAL
(refuted=true)** on the "union-of-qualifiers lattice" framing, CLAIM_HOLDS on the anti-
laundering safety property.

**Is non-vacuity guarded?** **Yes, blocking and exercised.** A skeptic hammered it across
auto / smt-only / fuzz-only and adversarial flags (`--invariant-min-rate 0.0`, tiny
`--samples`) with false, absurd (x<x), and genuinely-true postconditions under three unsat
precondition forms. Every case returned `invalid`/`unsupported` (SMT) or `error` (fuzz);
**no vacuous precondition set ever greened.** Skeptic verdict: **CLAIM_HOLDS.**

**Is per-assumption provenance threaded?** **Yes.** Every `AssumptionRecord` carries
`discharge.method`, `non_vacuity.status`, and (for std contracts) `source_type`/`producer`,
emitted verbatim in the `assumptions[]` array.

**Did any laundering probe succeed?**

- **False invariant + exported producer:** **Blocked.** The per-producer SMT obligation
  `invariant:<Type>:<producer>` fails (cex __arg0=(- 1.0)); overall run exits 1. Skeptic:
  CLAIM_HOLDS.
- **Under-constraint / vacuous precondition:** **Blocked** (a true-but-vacuous `value==value`
  invariant cannot launder; removing the invariant turns the property unsupported, proving
  the abstraction does real work). Skeptic: CLAIM_HOLDS.
- **Boundary-bypass (record wrapper / list / alias / caller-callback):** **Rejected** as
  covered-or-rejected error (exit 3), never silently skipped.
- **Verdict rollup (weaker→stronger):** **Blocked** (weakest-link). The only refutation is
  the *lattice-structure* framing, not the safety property.

**Disagreement chain, resolved in favor of executed evidence (a workflow skeptic overstated;
the fresh red team corrected it):** A workflow Phase-2 skeptic claimed the chelis#426 collapse
bug made 3 economoist Bellman greens unsound. The **fresh red-team pass refuted that on every
particular**, and the orchestrator independently confirmed:
- The bug is **real upstream**: a wrapper `def h(v,r) = fmax(v,r)` *called twice* in a
  difference, e.g. `(h(v,r) - h(w,r)) <= 0.0`, proves at smt/samples=0 though false — the two
  calls collapse to one term. A genuine soundness hole in the lowering (chelis#426).
- It **does not reach economoist.** economoist inlines `fmax` with fully distinct `v*`/`w*`
  argument expressions (no def called twice), so the collapse cannot fire. Verified: a
  false-monotonicity sibling on the *identical* inlined structure **refutes**
  (failed/smt/samples=0, cex `g=1/2, v0=v1=-2, w0=-1, w1=-1/2, …`); all 8 Bellman greens are sound.
- The overstated claim was wrong on specifics: `bellman_contraction`/`bellman_contraction_lower`
  **do not exist** (real names `bellman_contraction_state0/_state1/_state0_lower/_state1_lower`),
  and the cited "two FALSE bellman_state0 props both proven" repro **does not reproduce** (both
  refute).

**Net: 0 of 17 economoist greens are unsound.** The chelis#426 hole is real and worth fixing
upstream, and the inlining mitigation is *manual* (a future Bellman property that calls a
shared if/then/else-bodied def twice would be exposed) — that fragility is the standing
integrity concern, not any shipped green.

**Net integrity verdict:** the green badge means what it claims, with honest qualifiers, on
the *correct binary* — **except** three live shipping gaps (build-contingency false-green,
machine-arithmetic over-approximation, call-form silent fuzz-drop), plus one real *upstream*
soundness hole (chelis#426 same-def-called-twice collapse) that economoist successfully but
*manually* mitigates by inlining — no shipped economoist green is unsound.

---

## 7. Frontier verdict

**Beacon's real landed surface vs its WI plan.** Beacon **does bound over a non-singleton
region** today (the interval domain, Arb-256 oracle) — output interval width provably scales
with input box width on the recognizable BS pricer. This is region bound-propagation, not
point evaluation with an interval oracle. **But:** no branch-and-bound, so it covers only
boxes narrow enough that all input-dependent erf64 branches resolve to one side (a desk-
width [95,105] box → unsupported). The arb-box `proved` path is **opt-in** (`arb-oracle`
feature, off by default; default checker emits `proved_oracle_unverified`). WI status:
WI-B1 (interval forward) and WI-B7 (Arb oracle) **landed**; WI-B2 partial; WI-B3 (zonotope),
WI-B4 (CROWN), WI-B5 (branch-and-bound), WI-B8 (verified Greeks), WI-B9 **SPEC-ONLY.** There
is **no chelis discharge seam** — Beacon consumes a recon-captured DAG JSON, not a live
chelis export, and its verdicts cannot flow into chelis.

**Dispatcher/solver roster.** The "dispatcher" is a per-property **tier** funnel (type → SMT
→ fuzz) keyed on `SmtAmenability`, not a multi-engine router.

| Solver | State | Feature | Notes |
|---|---|---|---|
| **cvc5** | **PROVING** | `smt` (cvc5-rs 0.3.2) | In-process static FFI; sole wired engine; arith_model=real |
| **Z3** | **SPEC-ONLY** | — | Named only in a `Solver`-trait design comment |
| **Sollya** | **ABSENT** | — | Zero references |
| **Arb/FLINT** | **ABSENT** | — | (Arb is used by Beacon, not by chelis-prove) |
| **Clarabel** | **ABSENT** | — | No convex/conic lane |
| **Carcara** | **ABSENT** | — | No external proof-certificate re-check |
| **MetiTarski** | **ABSENT** | — | Transcendentals go to cvc5 QF_NRAT (narrow); no log |

**Contract abstraction's reach on quant-named derivatives properties.** It reaches **put-call
parity and C≤S** (recognizable, composite-green) — and **only those**, plus delta∈[0,1].
The next tranche (non-negativity, intrinsic lower bound, monotonicity in spot) is blocked by
the absent monotonicity invariant + d1≥d2 coupling. Monotonicity is SPEC-ONLY: the contract
*exists as a fuzz record* but is not wired into `ContractAbstraction`. The mechanism is
real and sound, but **not adopted in the reference shell** (shoals' shipped BS properties are
`def → bool`, `total:0` under `prove`).

---

## 8. Negative space (an honest "unsupported + why" is a result)

- **Unsupported at 0.8.0:** scalar `max`/`min`/`abs` over f32 is an unbound variable
  (`tests_blocked/numerics/scalar_max.ch` pins "unbound variable: max"), forcing economoist
  to ship local `fmax`/`fabs` helpers and inline if/then/else.
- **Held out by design (no green claimed):** n=3 per-component Bellman contraction
  (nested-fmax goal-site lowering limit); convergence / fixed-point / stationarity /
  general-n (need induction).
- **Falls to fuzz:** the shoals normal-CDF contract (range/reflection, 8192 samples); any
  call-form top-level predicate (silently, see §D); `log`-bearing goals (no cvc5 LOG kind);
  opaque-binder quantifiers (cannot lower to SMT even under `--tier smt-only`).
- **Oracle-validated but NOT proven:** all shoals first/second-order Greeks (delta/vega/rho/
  theta, gamma/volga/vanna) and the shipped BS pricer — validated against analytic closed
  forms by `oracle_greeks_gate` (14/14, 63 cells), but shoals runs **no `chelis prove`** in
  any script/CI; its derivatives "properties" are point-tested via `chelis test`.
- **Recognizable but abstraction-too-weak:** non-negativity C≥0, intrinsic lower bound
  C≥max(S−K·D,0), monotonicity in spot.
- **Upstream-unsound-but-mitigated (no shipped green affected):** the chelis#426
  same-def-called-twice collapse false-proves a difference of two calls to one shared
  if/then/else-bodied def; economoist avoids it by inlining (all 8 Bellman greens verified
  sound). The hole is upstream; the mitigation is manual (§6).
- **Machine-arithmetic excluded silently — by BOTH tiers:** the SMT badge over-approximates
  (int32 → unbounded integer; f32/f64 → Real, no NaN/Inf/rounding) and does not flag it; and
  the fuzz tier does not catch these either (the generator emits no NaN — `x==x` passes at
  100k samples — and int32-overflow fuzz is pathologically slow, not returning within 60s at
  1000 samples). The machine-arithmetic falsehood is invisible to *both* tiers.

---

## 9. c-earchin (one de-weighted finding)

- **c-earchin's place:** Frontier / **PLUMBED** / **synthetic** (recognizable names, point-
  check bodies). **SUPERSEDED.** It is a standalone EARS→Chelis bridge that (1) emits Deep
  `.dp` in its live CLI path (not canonical Surf — the `emit_surf.rs::render_surf` path is
  orphaned, invoked by no CLI command), (2) has **zero corpus dependencies/usages** — no
  shoals/economoist/c-note module consumes it and there are **zero** `.ears` inputs anywhere
  (a registry *index* entry exists in `c-note/tools/reef-packages/index.json`, but nothing
  consumes it), and (3) produces
  only fuzz/concrete-eval witnesses that the SMT binary classifies as
  `proven_modulo_fuzz_validated_contract` (`samples:100`) and **never** routes to SMT even
  under `--tier smt-only`. Its witnesses reduce a recognizable property name (monotonicity,
  non-negativity) to **one concrete inequality on fixed literals** (e.g.
  `is_monotone_increasing(0.0,1.0,0.0,1.0)`) with an **empty quantifier** — concrete-eval-by-
  design, then fuzz-looped 100× by the CLI bridge path. The live proving frontends write
  `@property forall` directly in Surf `.ch` and reach the shared SMT runner; c-earchin's
  bridge is bypassed.

**The prior deep-tools "c-earchin boundary" finding is hereby marked reconnaissance of a
dead path.** Both prior emit-format claims were partly right; the **binary emits `.dp`**.

---

## 10. Net verdict (restated, full reasoning)

**The genuine capability frontier.** On the `--features smt` binary, the Chelis proof stack
**proves recognizable economic structure as full unqualified SMT-exact green over the
reals** — 17 economoist properties (Bellman monotonicity/bound/contraction, Markov mass-
conservation/non-negativity, Gordon DDM comparative statics), single-application and fixed-
dimension, with convergence and general-n honestly held out (all 17 verified sound; the
upstream chelis#426 collapse hole does not reach them — see §6). It **proves recognizable derivatives
identities as composite-green** — put-call parity, C≤S, delta∈[0,1] — where the polynomial
structure is SMT-proven over reals but the normal-CDF transcendental link is fuzz-validated,
never machine-checked. Beyond SMT, **Beacon bounds the BS call price over a non-singleton
spot region** via interval propagation with an Arb-256 soundness oracle, limited to branch-
non-straddling boxes. Grounded in: economoist `bellman.ch`/`growth.ch`/`markov.ch` prove
runs + `prove_gate.py`; shoals `composites.ch` prove run + `oracle_greeks_gate.py`; Beacon
BS-pricer box sweeps; and the SMT-binary control reproduced by every agent.

**The single most valuable next increment.** **Adopt the contract-abstraction recognizable
greens in the reference shell and extend the abstraction with the normal-CDF monotonicity
invariant + d1≥d2 coupling.** This is the highest-leverage move because the mechanism
already proves parity and C≤S soundly (§B differential controls confirm no laundering), yet
the shipped reference shell (shoals) carries **zero `@property` declarations** for its BS
facts (`total:0` under `prove`), and the *next* tranche of recognizable facts (non-negativity,
intrinsic lower bound, monotonicity-in-spot) is blocked by a single missing invariant — not
a solver ceiling. The monotonicity contract already exists as a fuzz record; only the
`ContractAbstraction` wiring (`smt_lower.rs`) is missing. Grounded in: §B's proven-on-`/tmp`-
fixtures + failed-non-negativity counterexamples + the SPEC-ONLY monotonicity wiring gap, and
§A's `chelis prove ... shoals → total:0`.

**The integrity gaps that must not ship unaddressed.**

1. **Build-contingency false-green (chelis#422).** A non-SMT build emits
   `proven_modulo_fuzz_validated_contract` while fuzz-greening measure-zero falsehoods
   (orchestrator control: `square_needle` false-only-at-12345 and a corrupted economoist
   property false-only-at-d=1.0 both green at samples=100 with no proof_tier/counterexample).
   The badge must be unattainable, or loudly degraded, without the SMT feature.
2. **SMT "proven" over-approximates machine arithmetic, undisclosed.** `(n>100000):n*n>0`
   (int32 overflow) and `(x==x):f32` (IEEE NaN) both badge `proven`; `arith_model` is hard-
   coded `"real"` with no over-approximation assumption. The stronger badge is the *less*
   machine-faithful one. (§C.)
3. **Call-form predicate silent fuzz-drop (C-Note finding, unrepaired in 0.8.0).** A top-level
   `gte(...)`/`gt(mul(...))` predicate drops to fuzz and reports `status=passed`; a measure-
   zero-false property SMT disproves is reported PASSED in call-form. Two exact gate sites
   named in `property_runner/smt_lower.rs` (the brief's `tier_b_lower.rs` attribution is
   wrong). (§D.)
4. **Upstream SMT-collapse hole (chelis#426), manually mitigated.** A difference of two calls
   to one shared if/then/else-bodied def collapses to a constant and false-proves; economoist
   avoids it by inlining with distinct args (all 8 Bellman greens verified sound — no shipped
   green is unsound). The gap: the mitigation is manual and must hold at every future call
   site; fix the collapse upstream. (§6.)

Each gap is grounded in executed probes recorded in §11.

---

## 11. Appendix — probe log (reproducible)

### Orchestrator control (established before fan-out)

```
# non-SMT (default) build — FUZZ-ONLY, FALSE-GREENS:
#   square_needle (false only at x=12345), corrupted economoist (false only at d=1.0)
#   => composite_verdict proven_modulo_fuzz_validated_contract, samples 100, status passed
#      NO proof_tier, NO counterexample; plain text "100/100 passed"
# --features smt build — SAME inputs REFUTE:
#   proof_tier smt, samples 0, counterexamples x=12345.0 ; d=1.0,g=0.0,r=1.0
#   economoist growth = 5 proven (smt, samples 0, non_vacuity cvc5 sat) + 5 guards witnesses failed
```

### SMT-binary control (run independently by every agent + every skeptic; all matched)

```
/tmp/chelis-smt-bin/chelis --version
  => chelis 0.8.0   (cvc5 statically linked via cvc5-sys / cvc5-rs 0.3.2; no external cvc5 on PATH)

/tmp/chelis-smt-bin/chelis prove --tier smt-only --json   # ((x-12345.0)*(x-12345.0)) > 0.0 under x>0.0
  => {"composite_verdict":"failed","proof_tier":"smt","samples":0,
      "counterexample":{"x":"12345.0"},"arith_model":"real"}   EXIT 1
```

### Capability — economoist (full SMT-exact green)

```
chelis prove --tier smt-only --json economoist/properties/bellman.ch
  => 8 proven/smt/samples0/non_vacuity-established + 8 guards witnesses refuted
chelis prove --tier smt-only --json economoist/properties/growth.ch
  => 5 Gordon proven/smt/samples0 + 5 witnesses refuted
chelis prove --tier smt-only --json economoist/properties/markov.ch
  => 4 proven/smt/samples0 (exact == 1.0) + 4 witnesses refuted
grep -c bellman3_contraction economoist/properties/bellman.ch   => 0   (n=3 contraction held out, no stub)
python3 economoist/scripts/prove_gate.py    => PASS exit 0 (unqualified SMT green; witnesses+twins refute)
CHELIS_SMT_BIN=... python3 economoist/scripts/oracle_harness.py
  => 8 cases OK; grad dP/dr=-800, dP/dg=+800; 15 passed 0 failed; exit 0
```

### Capability — shoals (composite green)

```
chelis prove --tier smt-only --json shoals/properties/composites.ch   # run from package dir
  => put_call_parity_reflection / call_upper_bounded_by_spot / delta_in_unit_interval:
       composite_verdict proven_modulo_fuzz_validated_contract, proof_tier smt, samples 0
       contract std.normal_cdf.* method=fuzz checked_samples=8192 max_error<=5.55e-17 tol=1e-10
       non_vacuity established (cvc5 sat)
     put_call_parity_corrupted => failed (flip, cvc5 cex)
     delta_unknown_contract   => unsupported (unknown contract)
chelis prove shoals/properties/pricing.ch --json   => {"total":0,"passed":0}  (def->bool, not @property)
CHELIS_SMT_BIN=... python3 shoals/scripts/oracle_greeks_gate.py
  => PASS 14/14 groups, 63 cells, binding+monotone+sign-fold green; all_ok=true; exit 0
```

### §B — contract abstraction (/tmp Reef package linking bundled chelis-std)

```
put_call_parity (reflection contract)       => proven_modulo_fuzz_validated_contract (smt)
call_le_spot (range + sign preconds)        => proven_modulo_fuzz_validated_contract (smt)
call_nonneg (range)                         => failed (cex N(d1)=N(d2)=1, s=1/2)
call_ge_intrinsic (range)                   => failed (cex N(d1)=N(d2)=0, s=2)
call_monotone_spot (monotonicity contract)  => unsupported (monotonicity not wired into ContractAbstraction)
OVER-ABSTRACTION: parity range-only         => failed (cex s=1/2)   # reflection load-bearing
UNDER-CONSTRAINT: C<=S no contract          => unsupported          # range load-bearing
FALSE-CLAIM: C<=0 / N(-x)==2-N(x)           => failed (cex)         # no laundering
TRUST-GATE: bare .ch                        => "did not bind any call to Std.Contracts.normal_cdf"
```

### §D — operator-form vs call-form lowering

```
op_form  (a-b)>0.0 where a>b             => smt / proven / samples 0
call_form gt(sub(a,b),0.0)               => auto: fuzz / proven_modulo_fuzz_validated_contract / samples 100
                                            smt-only: unsupported "does not lower to Tier B"
iso_op   abs(x)>=0.0                      => smt / proven        # whitelisted intrinsic call-form
iso_call gte(abs(x),0.0)                  => unsupported         # isolates GATE 1 (predicate-form) from GATE 2
payoff_op  max((s-k),0.0)>=0.0           => smt / proven        # recognizable call-option payoff
payoff_call gte(max(sub(s,k),0.0),0.0)   => unsupported (smt-only)
false_call gte(...)  (measure-zero false) => auto PASSED (fuzz misses cex)  # soundness gap, not just degradation
# gate sites: property_runner/smt_lower.rs:172-216 (GATE 1, no Expr::Apply arm, _=>None@214)
#             smt_lower.rs:296 + property_runner.rs:635-640 + chelis-pred INTRINSIC_WHITELIST (GATE 2)
```

### §C — verdict algebra & integrity

```
recog.ch (x*x>=0): smt-only => proven/smt/samples0 ; fuzz-only => proven_modulo_fuzz_validated_contract/fuzz/100
mixed_two_fuzz.ch (two fuzz contracts) => proven_modulo_fuzz_validated_contract (never proven)
vacuous.ch (x>5 AND x<1, false body)   => composite_verdict invalid, non_vacuity unsat, status unsupported, passed 0
  (also under auto, fuzz-only [generator exhausted error], --invariant-min-rate 0.0 — never green)
overflow.ch (n>100000: n*n>0)          => composite_verdict proven   (UNSOUND vs int32 wraparound)
nan_probe.ch (x==x : f32)              => composite_verdict proven   (FALSE under IEEE NaN)
contract_exp.ch (exp(x)>0, std.exp.positivity) => proven via baked-in {status:proved,solver:cvc5} (never re-solved)
contract_exp_false.ch (exp(x)>1.0)     => failed, cex x=0.0          (body still solved)
# weakness_rank chain: composition.rs:63-72 ; weakest fold: composition.rs:284-291
# Proven gated to PropertyTier::Smt: property_runner.rs:183-206 ; non-SMT solve_property hard Timeout: tier_b.rs:99-103
```

### §E — Beacon region propagation

```
BS spot-box sweep (red-team-measured bounds): spot=100 => point [10.4506,10.4506];
  [99,101] => ~[5.33,15.59] (width ~10.3); [98,102] => ~[0.25,20.73] (width ~20.5)   # width scales => region prop
  (interval-wrapping illustrated by x*x over [-2,3] => [-8.5,9.0] vs true [0,9])
BS arb-box spot [95,105] => unsupported, node_id=140 (CmpLt straddles boundary), resolved_comparisons=3
x*x arb-box over [-2,3]  => [-8.5,9.0] (interval wrapping; true [0,9])
default checker (OracleMode::None) => proved_oracle_unverified (interval-only, not a sound proof)
default binary --oracle arb-box (no feature) => invalid "not built with the arb-oracle feature"
arb-oracle feature build (RUSTFLAGS ... -Clink-arg=-no-pie): x in [1,2]->proved; [-50,50]->proved; exp[0,1]->proved; too-tight->unknown
# graph-extraction seam absent: chelis-risc-dag/v0.8.0 only in beacon/src/wire.rs:8 (0 hits in chelis crates)
```

### §F — solver roster

```
strings /tmp/chelis-smt-bin/chelis | grep cvc5   => cvc5-rs-0.3.2 + cvc5-sys built-from-source paths
nm /tmp/chelis-smt-bin/chelis | grep cvc5        => T cvc5_mk_true, T cvc5_add_plugin (static C-ABI)
ldd /tmp/chelis-smt-bin/chelis | grep -iE cvc5|z3 => empty (no dynamic solver lib; in-process FFI)
grep -rniE 'z3|sollya|arb-|flint|clarabel|carcara|metitarski' Cargo.toml crates/*/Cargo.toml => no matches
# dispatcher = tier funnel (SmtAmenability) not engine router: dispatch.rs:43-230
# worker.rs = crash isolation (re-exec same binary, same in-process solve), not a deep lane
```

### §G — AD & special functions

```
chelis eval 'grad((fn x -> if lt(x,0.0) then neg(x) else x), wrt=x)(2.0)'            => tensor([1.0])
chelis eval 'grad((fn x -> grad((fn y -> y*y*y), wrt=y)(x)), wrt=x)(2.0)'            => tensor([12.0])
# delta_in_unit_interval proven over ABSTRACTED N, not AD delta grad(bs_call,wrt=s)
# erf64 = A&S 7.1.26 with if lt(ax,small) + if lt(x,0.0) (pricing.ch:11-28); contracts.rs = closed 9-ID allowlist
# erf/erfc absent from SMT-lowerable set (tier_b_lower.rs:785-852); no perturbed-branch policy anywhere
```

### §6 — chelis#426 collapse: real upstream, NOT reaching economoist (red-team-corrected)

```
# UPSTREAM BUG (real): a shared def CALLED TWICE in a difference collapses to a constant
rt_collapse.ch: def h(v,r)=fmax(v,r); forall(v,w,r): (h(v,r)-h(w,r))<=0.0
  => proven/smt/samples0   (FALSE at v=1,w=0,r=-5)   <-- genuine soundness hole, chelis#426

# economoist DOES NOT use that shape: it INLINES fmax with distinct v*/w* args.
# False-monotonicity sibling on economoist's exact inlined structure:
/tmp/bellman_false_sibling.ch (Tv >= Tw + 1.0 under v<=w)
  => failed/smt/samples0, cex g=1/2,v0=v1=-2,w0=-1,w1=-1/2,...   <-- mitigation works, refutes
economoist/properties/bellman.ch (smt-only) => 8 proven/smt/samples0 + 8 witnesses failed
grep '@property bellman_contraction(_lower)? ' bellman.ch => ABSENT (real: *_state0/_state1/...)
# Corrected: 0 of 17 economoist greens unsound; the earlier "3 unsound" draft did not reproduce.
```

### §H — c-earchin

```
c-earchin translate sample.ears --inline   => emits Deep .dp (module {...} (def {...} (app {} (var {} ...))))
chelis prove --tier smt-only --json c-earchin/references/finance_options/options_rules.dp
  => every req_FIN_*: proven_modulo_fuzz_validated_contract, samples 100, NO proof_tier (bridge ignores --tier)
grep -rln c-earchin in shoals/economoist/c-note + find *.ears   => no corpus consumer, zero EARS inputs
  (only a registry index entry: c-note/tools/reef-packages/index.json catalogs c-earchin v0.2.2/v0.2.5)
```

---

## 12. Red-Team Validation

This document was validated by a **fresh-context adversarial subagent** (per the repo's Red
Team Protocol) that re-executed the load-bearing claims against the SMT binary and read full
source — it did **not** read the synthesis reasoning. The SMT-binary control passed in its
hands (measure-zero polynomial → `failed`/`smt`/samples 0/cex x=12345.0). Outcome: the
capability frontier, the next-increment recommendation, and **3 of the 4** integrity gaps are
**supported by executed evidence**; **one flagship finding was wrong and has been corrected**;
two HIGH and several LOW doc errors were folded in.

**Confirmed by re-execution:** RT-C2 (machine-arith over-approximation: `(n>100000):n*n>0`
int32 and `(x==x):f32` both badge `proven`/smt), RT-C3 (call-form silent fuzz-drop incl. a
genuine false-green; gate site `property_runner/smt_lower.rs:172-216`, not `tier_b_lower.rs`),
RT-C5 (verdict rollup = sound `weakness_rank` chain, not a lattice), RT-C6 (economoist 17 /
shoals 3 composite / shoals `pricing.ch` `total:0`), RT-C7 (non-vacuity blocking across every
tier+flag), and **all 10 citation spot-checks** (no misses).

**Corrections folded into this revision:**
- **CRITICAL — RT-C1 (the flagship soundness finding was wrong).** The earlier draft claimed
  the chelis#426 collapse bug made *3 economoist Bellman greens unsound*. Re-execution shows:
  the bug is **real upstream** (a shared def called twice in a difference false-proves) but
  reaches **zero** economoist properties (economoist inlines `fmax` with distinct args — a
  false-monotonicity sibling on that exact structure **refutes**); the named properties
  `bellman_contraction`/`bellman_contraction_lower` **don't exist**; the cited repro **doesn't
  reproduce**. Rewritten throughout (§2/§3/§4/§5/§6/§8/§10/§11) to "**0 of 17 unsound**; the
  upstream hole is real but economoist's *manual* inlining mitigation holds and must be
  re-verified at every future call site."
- **HIGH — RT-C4 Beacon.** The "`architecture.md` files the arb-box oracle under *Deferred By
  Design* / binary reverses its README" stale-flag was **fabricated** — doc and code already
  agree (the README's singleton caveat is the arb-*point* lane, distinct from the shipped
  arb-*box* region path). Removed. The §E/appendix interval numbers were wrong ([98,102] is
  ≈[0.25,20.73], a positive lower bound, not the stated [−0.84,21.74]); corrected to
  red-team-measured bounds; interval-wrapping illustrated by `x*x` instead.
- **MEDIUM — "fuzz tier is IEEE-faithful" overstatement.** Fuzz catches neither NaN (`x==x`
  passes at 100k samples) nor int-overflow (pathologically slow); the machine-arithmetic
  falsehood is invisible to **both** tiers. Corrected in §2/§3/§C/§8.
- **LOW.** c-earchin "zero references" → "zero corpus consumers (a registry index entry
  exists)"; "9-ID allowlist" → "8 fact-IDs + impl name"; int32-fuzz slowness recorded in §8.

These corrections are grounded in the red team's executed evidence (and the orchestrator's
independent re-run of the Bellman false-sibling and property-name checks); the capability and
frontier verdicts are unchanged.
