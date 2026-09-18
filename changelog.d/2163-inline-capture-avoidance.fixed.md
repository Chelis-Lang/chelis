Host-lane inlining is now capture-avoiding. Substituting a caller's argument
into a callee whose `fn` or `let` binder spelled one of that argument's free
names captured it, so the compiled C silently computed a different value from
`eval` with no diagnostic. A colliding binder is renamed to a deterministic
fresh name before the substitution descends under it. See
[#2163](https://github.com/Chelis-Lang/chelis/issues/2163).
