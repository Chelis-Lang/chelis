The checker decides every operand obligation a declaration leaves open when the
declaration closes ([04-INF-1], [04-INF-6], [04-INF-9]). Previously each of
these checked with score 1:

- A call waiting on an operand whose type nothing in the declaration determines,
  such as `def size(k) -> i64 = numel(k)`, or a `take`, `where` or
  `uniform_like` over a lambda parameter that is never applied, was dropped
  ([#2518](https://github.com/Chelis-Lang/chelis/issues/2518)). It is now
  decided at an arbitrary type, so an operation that admits only some operand
  types is rejected, naming the declaration, and an operation that admits every
  type is still accepted. The collection operations `len`, `index`, `append`
  and `concat` follow the same rule and diagnostic rather than a separate one.
  So does `eq`, which admits no function: a wrapper such as
  `def equal(x, y) = eq(x, y)` is rejected, as `def equal[a](x: a, y: a) -> bool`
  already was. A `to_tensor([])` whose element type nothing determines is now a
  check error rather than an evaluation or code-generation error.
- `dict_get`, `dict_contains`, `dict_remove` and `dict_insert` over an operand
  typed by an authored binder, `to_tensor` and `dict_of` over a list whose
  element an unbounded binder types, and a tuple projection or field read on a
  binder ([#2523](https://github.com/Chelis-Lang/chelis/issues/2523)). Each is
  now decided at the binder's instantiations, with the operation's own
  diagnostic. `dict_of([])` and `to_tensor([])` still take their element type
  from a declared result.
- An axis argument whose type was still a variable, including one typed by an
  authored binder or bound later to `i64`, as in `cumsum(t, ax)` with `ax: q`
  and `q: Int` ([#2523](https://github.com/Chelis-Lang/chelis/issues/2523)).
  An axis-domain argument is `i32` ([05-DIM-3]), so it is now constrained to
  `i32`, and a binder there is reported as narrowed to `i32`.
- A tuple projection or field read on the parameter of a `let`-bound lambda let
  the lambda generalize, so each use returned an unconstrained type and a false
  declared result checked. The lambda now stays monomorphic until its first
  application, like one carrying any other obligation, and a top-level lambda
  whose projections only a later declaration's call would resolve is rejected,
  since a later declaration is not a binding site ([04-INF-1]). A field read on
  an opaque type through such a lambda is now reported as the opaque-boundary
  violation it is, rather than as an unverifiable access.
- A dtype-family requirement that narrowed an authored binder only while the
  declaration boundary replayed a suspended call, such as a `cast_trunc` inside
  a returned closure whose parameter only the declared type fixes
  ([#2537](https://github.com/Chelis-Lang/chelis/issues/2537)). The binder's
  declared contract is now checked after the last such replay.

Two routes decide on the types a recursive group determines rather than
before them. A declared header whose result or parameter type is omitted is
typed, inside its recursive group, at the one type its body determines
([04-INF-5], [04-INF-2]); each recursive call used to see a fresh variable in
place of the omitted type, so `def step(n: i32) = ... step(n - 1).0` over an
`i32` result checked with score 1 and failed in `eval`. And a group member's
open obligations are decided when the whole group has been inferred, so a
sibling declared later can still determine the type they wait on. Recursive
definitions with annotated parameters and an omitted result keep checking when
they read their own result's tuple or record fields or pass it to `take`,
and the ill-typed `step(n - 1).0` is rejected with the access's own diagnostic.
