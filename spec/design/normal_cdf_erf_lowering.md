# normal_cdf → erf lowering (chelis#434 design note)

A mechanism switch that lets the standard-normal CDF be discharged through the
**certified `erf` envelope** instead of the **trusted `normal_cdf` contract**.
Implemented (behind the unwired pipeline) in
`crates/chelis-prove/src/transformations/normal_cdf_erf.rs`.

## The identity

`Φ(x) = ½·(1 + erf(x/√2))` — exact for the standard normal CDF. The pre-pass
`lower_normal_cdf` rewrites every `normal_cdf(x)` in a goal's postcondition to
`0.5 + 0.5·erf(x · (1/√2))`, where `1/√2` is the nearest `f64`
(`std::f64::consts::FRAC_1_SQRT_2`). The rewrite is a pure `SmtExpr → SmtExpr`
identity; it introduces no approximation of its own.

## Two lanes, one CDF

| lane | mechanism | qualifier | status |
|------|-----------|-----------|--------|
| **contract** (`ContractAbstraction`, `property_runner`) | `N(x)` → fresh `[0,1]` var + optional reflection `N(−x)=1−N(x)` | trusted contract (fuzz-validated) | **unchanged, default** |
| **envelope** (this pre-pass → `abstract_subterm`) | `N(x)` → `½(1+erf(x/√2))`, `erf` bounded by its Sollya/Gappa/Arb-certified envelope | `SpecialFunctionCertified` | new, **unwired** |

The envelope lane upgrades the CDF from a *trusted assumption* to a *certificate-
backed* bound. This note and its code add ONLY the envelope-lane pre-pass; the
contract lane is not removed or altered. Both sit behind the same engagement
point, and the pipeline has no production caller today (probe p16), so nothing in
a real `chelis prove` run changes.

## The wiring boundary — and a concrete blocker it exposes

`NormalCdfToErf` is a `Transformation` meant to run *before*
`abstract_subterm::AbstractSubterm` in the (future) pipeline. Composing them
today exposes a real limitation, test-locked in
`lowering_is_exact_but_discharge_needs_compound_arg_propagation`:

- The lowering is **done and exact** — `normal_cdf` disappears, `erf` appears.
- But the `erf` argument is `x/√2`, a **compound** expression (`Mul(x, 1/√2)`),
  and the abstract-subterm range extractor handles **only a bare `Var`** in this
  slice (extending it is compound-argument interval propagation — the deferred
  research-risk 60%). So `AbstractSubterm` **declines**: the goal is returned
  unchanged.

Even this trivial **affine** argument (`a·x`, a constant scale) needs argument
propagation. The control test `bare_arg_erf_identity_would_abstract` confirms the
identity shape itself is fine — an `erf(x)` with a bare bounded `x` DOES abstract
— so the sole blocker is the `x/√2` scale.

**Implication for the de-narrowing decision:** the first, smallest piece of the
60% is affine-argument propagation (`a·x + b`). It is required even to discharge
`normal_cdf(x)` with a bare `x`, let alone the Black-Scholes `normal_cdf(d1)`
where `d1` is a full compound of `log`/`sqrt`. This reinforces
`spec/design/probe_434_transcendental.md` p17: authoring conventions cannot
sidestep it, because the CDF→erf identity itself manufactures a compound argument.

## When the pipeline is wired

The intended order is `NormalCdfToErf` → `AbstractSubterm` → residual → SMT.
Wiring is the deferred step; when it lands with (a) affine/compound argument
propagation and (b) the erf envelope reachable, `normal_cdf` goals discharge as
`proven_modulo_real_arithmetic` (the residual's over-reals `RealArith` qualifier
composed with `SpecialFunctionCertified`; probe p18), not the trusted-contract
badge.
