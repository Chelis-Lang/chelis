`chelis prove` no longer reports an SMT disproof as a property failure when no
machine value witnesses it. A Tier B disproof is decided over the real numbers,
so its counterexample is an exact real point: an exact rational need not have a
binary64 value at which the property is false, and an irrational algebraic
witness has none at all. `1.0f64 / ((x * x) - 2.0f64)` is infinite over the
reals at `x = sqrt(2)` and finite at every double in `[1, 2]`, yet a property
asserting that was reported failed under `--tier auto` and `--tier smt-only`
alike.

The witness is now rounded to each binder's declared width and evaluated
concretely, the same route a square-root domain witness takes, and the outcome
is reported in the additive `machine_arithmetic` field:

- `confirmed` — the preconditions admit the rounded witness and the body is
  false there. The result is a definite `failed` carrying that machine value as
  the counterexample, rather than the reals-hedged
  `disproved_modulo_real_arithmetic` with the solver's model term.
- `unconfirmed` — the witness is not a machine counterexample. `--tier auto`
  samples the property and reports what sampling found; `--tier smt-only`, which
  has no sampling tier, keeps the over-the-reals disproof and records the field
  beside it.
- `unchecked` — the witness could not be read or evaluated. The disproof is
  unchanged: an evaluation limit is not evidence about the property.

See [#3971](https://github.com/Chelis-Lang/chelis/issues/3971).
