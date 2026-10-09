A generic rank-2 helper that builds a broadcast with `insert` sizes read from
`shape(x, axis)` and then multiplies evaluates again instead of being refused
as a checker dimension with two extents. A refusal raised while a helper's
dimensions are actualized is now reported as a diagnostic rather than
unwinding, and `chelis test` renders such a diagnostic instead of
"test panicked (no message)". A run-time `insert` extent that differs from the
declared result extent still traps `Domain`.
See [#3340](https://github.com/Chelis-Lang/chelis/issues/3340).
