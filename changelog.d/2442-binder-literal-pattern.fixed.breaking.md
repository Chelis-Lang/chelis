A literal pattern whose scrutinee is a rigid dtype-family binder is now
checked at every instantiation the binder admits, under
`spec/04-type-system.md` [04-PAT-1], [04-LIT-2], and [04-INF-6]. Before, the
checker skipped it, so `| 300 =>` under `p: Int` (out of range at `i8`),
`| 70000.0 =>` under `p: Float` (infinite at `f16`), and `| 0 =>` under
`p: Numeric` (it never matches a float) all checked at score 1. The last one
also gave wrong answers at run time. A literal pattern under an unbounded
binder is rejected too, because such a binder admits non-primitive types.

Each rejection names the binder, its family, and the member where the
pattern fails. It also names a repair that checks under the binder and
matches exactly the values equal to the literal, with no cast that can trap:

- the pattern in the family's own kind, such as `| 0.0 =>`;
- a comparison in an `if` ahead of the match, over the literal bound at the
  binder, such as `if eq(x, cast(0, p)) then ...`, or over the scrutinee
  widened to `i64` or `f64`, such as `if eq(cast(x, i64), 300i64) then ...`;
- deleting an arm that no instantiation can match.

See [#2442](https://github.com/Chelis-Lang/chelis/issues/2442).
