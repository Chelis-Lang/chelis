A `match` arm guard is now typed the way an `if` condition is: the checker
unifies it with `bool`. Before, a guard whose type was still pending, such as
`| v if eq(v, cast(0, p)) =>` under a dtype-family binder, was rejected with
`match arm guard must be bool, got ?350`, while the same expression was
accepted as an `if` condition. A guard that cannot be `bool` is still rejected.
See [#2444](https://github.com/Chelis-Lang/chelis/issues/2444).
