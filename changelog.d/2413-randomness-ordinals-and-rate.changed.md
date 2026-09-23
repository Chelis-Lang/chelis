The randomness rules now say which random-call ordinals each draw takes:
- an `if` or `match` consumes ordinals only for its selected arm;
- `vmap` of a random function assigns each row the ordinals of evaluating the rows in order, so `vmap(f)(x)` equals `stack([f(x[i]) ...])` bit for bit;
- `par` assigns them as sequential evaluation of its branches in source order would;
- leaving a nested `with seed` restores the enclosing seed and ordinal.

`grad` through a `dropout` rate that depends on a differentiated input is now a structural rejection (`RandomSelectionParameter`). The previous pathwise rate cotangent was a biased estimate of the derivative of the expected result; write `stop_gradient(rate)` to treat such a rate as constant.

Several lanes do not yet implement these rules; [#2413](https://github.com/Chelis-Lang/chelis/issues/2413) tracks them.
