An exported `def` that draws can no longer escape with its randomness
unhandled, because a draw takes its key as an argument. Previously such a
`def` scored 1.0 at `chelis check`, and then one compiled lane drew silently
while the other aborted. The keyless form is now an arity error at
`chelis check` and does not build, and the form that takes a `key` builds in
both compiled lanes and draws the same bits as `chelis eval`. Fixes
[#2318](https://github.com/Chelis-Lang/chelis/issues/2318).
