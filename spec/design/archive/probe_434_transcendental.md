# chelis#434 probe report — transcendental SMT discharge (p16–p20)

Probe-gated slice for the transcendental-envelope de-narrowing decision. These
five probes exist so the team lead's investment call on the research-risk 60%
(compound-argument interval propagation + hot-path wiring + release) rests on
**run evidence**, not source reading. Companion runnable evidence:
`crates/chelis-prove/tests/probe_434_transcendental_envelopes.rs` (p18, p20).

Pin/context: branch off `origin/main` at **v0.15.0-1** (`b8589260`). v0.15.0
(IR/AD movement bounds) is unrelated to the prove path. Environment: z3 present
(`~/.local/bin/z3`), **cvc5 links via `cvc5-rs` under `--features smt`** (builds
clean here), **gappa present** (`/usr/bin/gappa`), **sollya NOT installed**.

## Headline

- **The envelope machinery is real but inert.** The `AbstractSubterm` transform
  and `TransformationPipeline` have **zero production callers** (p16). Nothing in
  dispatch / tier_b / obligation_engine / property_runner constructs or runs
  them. The certified `erf` envelope is consumed only by tests and the
  (out-of-tree) Beacon relaxation.
- **Model-authoring conventions do NOT sidestep compound-argument propagation
  for the flagship Black-Scholes goal (p17).** Two independent blockers: the
  lowering inlines let-bindings so there is no bare variable to attach a range
  precondition to, and — deeper — freeing the transcendental arguments as
  bounded vars destroys the functional coupling (`d2 = d1 − σ√t`; `s` vs
  `k·e^{−rt}`) that BS positivity actually depends on. **Recommendation: the
  research-risk 60% is genuinely required; do not gate it on an authoring-escape
  hatch.**
- **The residual solver step is not the risk.** cvc5 discharges the residual
  polynomial goals an abstract-subterm transform emits, including a nonlinear
  `s·exp_abs` term, within timeout, and honestly disproves an un-entailed one
  (p20). The verdict lattice already projects the certificate correctly:
  `special_function_certified` alone → `unsupported`; `+ real_arithmetic` →
  `proven_modulo_real_arithmetic` (p18).
- **The envelope RECIPE generalizes to exp; the CERTIFY anchor does not, yet
  (p19).** The float-proposer fits `exp` over [-2,2] to 1.76e-12 empirical
  sup-norm error by swapping the truth oracle. But the two sound-eps anchors are
  both erf-specialized / env-blocked: the Arb certifier hardcodes
  `arb_hypgeom_erf`, and the Sollya→Gappa proof bundle needs Sollya, which is not
  installed here.

## Probe verdicts

### p16 — the pipeline never engages in a real `chelis prove` run today. CONFIRMED.

Static + build evidence. Every reference to `TransformationPipeline`,
`apply_all`, and `AbstractSubterm` across `crates/` is one of: the definition
itself, the `pub use` re-export in `lib.rs:116`, or code inside `#[cfg(test)]`.
There is **no** non-test caller — `dispatch.rs`, `tier_b.rs`,
`obligation_engine{,.rs}`, and `property_runner{,.rs}` contain no `transform` /
`abstract` reference on the discharge path (the only `abstract*` hits there are
the unrelated `ContractAbstraction` for `normal_cdf`). The transform is a
built-but-unwired capability: a goal carrying `erf(x)` reaches the SMT lowering
as an uninterpreted `Apply("erf",[…])`, fails the inlineability classifier
(`property_runner.rs:817`), and drops to fuzz — the envelope is never consulted.

### p17 — can an authoring convention sidestep compound propagation? NO (for the #434 goal).

`shoals/src/pricing.ch`: `d1_64`/`d2_64` are **top-level `def`s, not let-vars**;
the oracle `BS_SOURCE` (`tests/transcendental_finance_lowering.rs`) binds
`log_sk`/`d1`/`d2` as **let-vars** inside `bs_call`. Either way the transcendental
**arguments are compound**: `log(s/k)`, `exp(−r·t)`, `normal_cdf(d1)` with `d1`
a compound expression. Concretely:

1. **Expressibility blocker.** SMT lowering **inlines every let-binding** into
   the substitution map (`property_runner/smt_lower.rs:978-993`); unknown
   transcendentals lower to `Apply(name, args)` with the argument fully expanded
   (`:804`, `:976`). So `log(s/k)` → `Apply("log",[Div(s,k)])` — a compound
   argument. `extract_variable_range` (`abstract_subterm.rs:258`) matches **only
   bare `Var`**. And property preconditions are gathered **only from
   `property.params`** (the forall-quantified vars) and the `where` clauses over
   them (`property_runner.rs:735-774`); a let-bound intermediate is never a
   variable and cannot carry a precondition. So "explicit range precondition on a
   let-bound transcendental argument" is **not expressible** — the intermediate
   does not survive lowering as a nameable var. To pin the argument it must be a
   bare property parameter, i.e. a **model rewrite**, not a convention.

2. **Semantic blocker (the deeper one).** Even after lifting the arguments to
   bounded parameters, BS positivity's truth relies on the **coupling** between
   them. `bs_call = s·N(d1) − k·e^{−rt}·N(d2)`. Free `N(d1), N(d2)` as
   independent `[0,1]` vars and the property is **false**: take `N(d1)→0,
   N(d2)→1 ⇒ value → −k·e^{−rt} < 0`. The existing `normal_cdf` contract lane
   already abstracts `N(·)→[0,1]` with a **reflection** precondition
   (`smt_lower.rs:111-127`), but reflection couples only `N(x)` with `N(−x)`, not
   `d2 = d1 − σ√t`, so it cannot recover positivity. Encoding the real coupling
   as explicit preconditions is tantamount to hand-writing the compound
   propagation, and strict positivity needs essentially the full BS relationship
   (moneyness ↔ d1), not just monotonicity + coupling.

**Verdict:** authoring conventions help only for **simple single-transcendental,
monotone, output-range-sufficient** properties where the argument can honestly be
a bare bounded parameter (and where the certified envelope's *output* band is
enough). They do **not** sidestep compound-argument interval propagation for the
Black-Scholes flagship. The 60% is required.

### p18 — a discharge carrying `SpecialFunctionCertified` end-to-end. CAPTURED.

Run through the SAME `base_verdict_from_discharge` seam the property runner uses
(`composition.rs:649`). Serialized `composite_verdict` tokens:

| discharge qualifiers (at `SoundApproximate`)      | `composite_verdict` token             |
|---------------------------------------------------|---------------------------------------|
| `{special_function_certified}`                    | `"unsupported"`                       |
| `{special_function_certified, real_arithmetic}`   | `"proven_modulo_real_arithmetic"`     |

The certified envelope alone is honestly **not a proof badge** — it reaches
`proven_modulo_real_arithmetic` only once the residual's own over-reals discharge
contributes `real_arithmetic`, and both caveats are then disclosed in the
`qualifiers` array. Locked by
`tests/probe_434_transcendental_envelopes.rs::p18_*` (green under `--features smt`
and default).

### p20 — cvc5-in-release discharges the residual polynomial goal. CONFIRMED.

Hand-constructed the residual shape `AbstractSubterm::apply` emits — a fresh
envelope-bounded var in place of the transcendental subterm — and ran it through
`Cvc5Engine` (`--features smt`, 10 s timeout):

- `log` residual (`__log_abs_0 ∈ [−1e−6, 0.70]`, prove `< 1`): **Proved**.
- nonlinear `s·__exp_abs_0` (`s ∈ [1,2]`, `exp_abs ∈ [0.9,1.1]`, prove `> 0`):
  **Proved** — the residual stays in cvc5's NRA fragment after abstraction.
- un-entailed (`exp_abs ∈ [0.9,1.1]`, prove `< 1.0`): **Disproved** — honest, not
  laundered.

The solver step is not the research risk; the argument-range propagation that
produces a *tight enough* fresh-var band is.

### p19 — trial exp envelope via the erf recipe. PARTIAL (proposer generalizes; certify anchor blocked).

- **Proposer half generalizes cleanly.** Reusing the exact
  `generate_erf_envelope.py::central_coeffs` approach (`numpy.polyfit` at
  Chebyshev nodes vs an mpmath truth) with `mpmath.erf → mpmath.exp` and box
  `[-2,2]`: a **degree-14** central polynomial fits `exp` to an **empirical
  sup-norm error of 1.76e-12** over the box. exp is monotone, so no saturation
  tails inside the box — one central arm. The committed erf proposer also runs
  clean here (numpy 2.4.6, mpmath 1.4.1). The proposer is function-agnostic given
  a truth oracle.
- **Neither sound-eps anchor is exercisable here.**
  - *Arb certifier* (`arb_oracle.rs`) hardcodes `arb_hypgeom_erf`
    (`rigorous_erf_enclosure`, `:151/:188`). The `ErfEnclosure` framework itself
    "carries no special-function identity" (`:62`) — generalizing to exp is a
    one-line swap to `arb_exp` plus function-keying — but it is **not** a
    no-code-change step, and the `arb` feature vendors FLINT/Arb (heavy build,
    not run here).
  - *Sollya→Gappa proof bundle* (`generate_erf_proof.py`) requires Sollya at
    `.local/bin/sollya`, which is **not installed**
    (`setup_proof_toolchain.py` builds it from source — heavy). **gappa itself IS
    present** (`/usr/bin/gappa`), so only the Sollya remez/Taylor-model half is
    the env gap.

**Verdict:** the recipe generalizes; the sound-eps step needs (a) mechanical
function-keying of the Arb certifier and (b) Sollya install for the machine-
checked Gappa half. Both are tractable, neither is exercisable in this
environment as-is. The proposer draft + numbers are reproducible via the probe
script (kept out-of-tree; see the PR body).

## Tooling availability (for the de-narrowing checklist)

| tool   | status here                    | needed for                          |
|--------|--------------------------------|-------------------------------------|
| cvc5   | links via `cvc5-rs` (`--features smt`) | residual discharge (p20) — WORKS |
| z3     | `~/.local/bin/z3`              | cross-engine oracle                 |
| gappa  | `/usr/bin/gappa`               | machine-checked eps proof (present) |
| sollya | **absent**                     | remez + certified Taylor model (BLOCKS Gappa half) |
| Arb    | `arb` feature vendors FLINT/Arb (not built) | rigorous eps cross-check |

## What this slice deliberately does NOT do

Per the probe-gate: no production wiring of the pipeline into dispatch, no
`composition.rs` verdict-projection change, no compound-argument propagation, no
release. Those are the research-risk 60%, a separate team-lead decision. The
mechanical generalization (function-keyed envelope + registry finder + generated
envelopes) lands in this same slice but stays **behind the same unwired
engagement point** — the flagship BS-positivity oracle stays honestly
`unsupported` until the wiring lands.
