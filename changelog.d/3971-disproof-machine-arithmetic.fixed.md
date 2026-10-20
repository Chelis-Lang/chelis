`chelis prove` no longer reports an SMT disproof as a property failure when the
witness it rests on is not a machine value. A Tier B disproof is decided over
the real numbers, so its counterexample is an exact real point: no binary64
value equals an irrational, and an exact rational need not be representable
either. `1.0f64 / ((x * x) - 2.0f64)` is disproved finite over the reals at
`x = sqrt(2)` and is finite at every double in `[1, 2]`, yet a property
asserting that was reported failed under `--tier auto` and `--tier smt-only`
alike.

A readable witness is now rounded to each binder's declared width and evaluated
concretely, the same route a square-root domain witness takes; an irrational one
is classified from the model term itself, since no machine value equals it.
Either way the outcome is reported in the additive `machine_arithmetic` field:

- `confirmed` — the preconditions admit the rounded witness and the body is
  false there. The result is a definite `failed` carrying that machine value as
  the counterexample, rather than the reals-hedged
  `disproved_modulo_real_arithmetic` with the solver's model term.
- `unconfirmed` — the witness is not a machine counterexample. The result is
  `unsupported` (exit `2`): there is no basis for a failure, because the only
  point the solver exhibited is not one, and none for a pass either, because the
  property was decided false over the reals and the check establishes nothing
  about any other input. `--tier auto` samples first, so a sampled
  counterexample is still a definite failure; only a clean run is `unsupported`.
- `unchecked` — the witness could not be read or evaluated. The disproof is
  unchanged: an evaluation limit is not evidence about the property.

`unconfirmed` does not mean the property holds. `(x * x) != 11.0f64` on `[3, 4]`
is `unconfirmed`, because `sqrt(11)` is irrational, and is nevertheless false at
`3.3166247903554`, whose square is exactly `11.0`.

See [#3971](https://github.com/Chelis-Lang/chelis/issues/3971).
