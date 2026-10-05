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

## The wiring boundary — affine argument propagation has landed

`NormalCdfToErf` is a `Transformation` meant to run *before*
`abstract_subterm::AbstractSubterm` in the (future) pipeline. Composing them:

- The lowering is **exact** — `normal_cdf` disappears, `erf` appears.
- The `erf` argument `x/√2` is **affine** (`Mul(x, 1/√2)`). This bounded
  increment added **affine-argument propagation** (`a·x + b·y + … + c`) to the
  abstract-subterm range extractor, so `normal_cdf(x)` with a bounded `x` now
  **abstracts** via the certified `erf` envelope — the case that declined before
  this slice. Test-locked in
  `normal_cdf_of_bare_var_abstracts_via_affine_propagation`.

The **remaining** boundary is genuinely NONLINEAR arguments. `normal_cdf(x·y)`
lowers to `erf((x·y)/√2)`, and `(x·y)/√2` is var×var — non-affine, so
abstract-subterm still declines (test `normal_cdf_of_nonlinear_arg_still_declines`).
The Black-Scholes `normal_cdf(d1)` with `d1 = (log(s/k) + …)/(σ√t)` is this
nonlinear case: full compound propagation over `log`/`sqrt`/products is the
remaining research-risk work (see `spec/design/archive/probe_434_transcendental.md` p17).

**Soundness of affine propagation:** the extractor parses the argument into a
canonical coefficient-merged affine form and evaluates its interval by summing
`coeff·[var range]` per variable. Merging coefficients first removes the
interval-arithmetic dependency error (`x − x` is exactly `[0,0]`, not `[−1,1]`).
It is **fail-closed**: any non-affine node (var×var, division by a non-constant or
by zero, a transcendental `Apply`, an `Ite`/comparison), or any unbounded leaf
variable, returns `None` and the transform declines. Adversarial tests cover each
in `abstract_subterm.rs` (`nonaffine_fails_closed`,
`affine_unbounded_leaf_fails_closed`, `affine_repeated_var_merges_no_dependency_error`).

## When the pipeline is wired

The intended order is `NormalCdfToErf` → `AbstractSubterm` → residual → SMT.
Wiring is the deferred step; when it lands with (a) the remaining compound
argument propagation and (b) the erf envelope reachable, `normal_cdf` goals
discharge behind the `SpecialFunctionCertified` qualifier. Note the eventual
wiring must project this to the distinct honest tier
`proven_modulo_certified_envelope` (per the sprint plan) rather than reusing
`proven_modulo_real_arithmetic`; that verdict-projection change is deliberately
out of this slice (probe p18 documents the current projection).
