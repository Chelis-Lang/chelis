`chelis build` now compiles a program whose `if` consumes an owned value on one
branch and not the other, instead of rejecting it with an internal ownership
diagnostic. The branch arm that does not consume the value now releases it, so
both paths reach the join in the same state. Programs this affected typically
bind a value from a block expression and then return it from only one arm --
the shape a `nan`-sentinel guard produces, as in a matrix inverse that returns
the real result or a NaN fill. `chelis check`, `chelis test` and `reef build`
all accepted these programs already; only the C build rejected them, so the
failure appeared only at the point of emitting code. `match` has the same
limitation still, tracked separately. See
[#2477](https://github.com/Chelis-Lang/chelis/issues/2477) and
[#2520](https://github.com/Chelis-Lang/chelis/issues/2520).
