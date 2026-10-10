`grad` accumulates a value's cotangent contributions with no base leaf. A
value with one contribution has that contribution as its gradient, so a `-0`
gradient stays `-0`: the gradient of `sum(a * w)` with respect to `a` at
`w = -0`, or of a dead `relu` unit with a negative input, is now `-0`, where it
was `+0`. Two or more contributions combine in the same adjacent-pair tree as
`sum`. For a value used three or more times, the tree pairs the contributions
differently from before, so the last bits of such a gradient can change. A
parameter that receives no contribution still gets an exact `+0` gradient. The
evaluator and the C lane agree. See
[#3419](https://github.com/Chelis-Lang/chelis/issues/3419).
