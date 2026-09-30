An exported `def` that draws can no longer escape with its randomness
unhandled, because a draw takes its key as an argument. Previously such a
`def` scored 1.0 at `chelis check`, and then the C tensor graph drew silently
while the C host path aborted. The keyless form is now an arity error at
`chelis check` and does not build, and the form that takes a `key` works in
both C paths and draws the same bits as `chelis eval`. Fixes
[#2318](https://github.com/Chelis-Lang/chelis/issues/2318).
