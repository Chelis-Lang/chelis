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

A recursive group whose members omit types is typed by [04-INF-2] and
[04-INF-5] together ([#2590](https://github.com/Chelis-Lang/chelis/issues/2590)).
Inside its group, a member whose declared header omits a type, such as its
result, shares its authored binders with every in-group reference, and each
reference takes its own instance of each omitted type. When the group
completes, each reference is typed at the type the member's body determines,
which it never narrows: an instance of an omitted type must be that type
itself or a fully concrete type, and anything else is polymorphic recursion
([04-INF-3]), reported at the reference with both types. A use that disagrees
with the body-determined type is also reported at the reference. Previously
each recursive call saw a fresh variable that nothing tied to the body, so
`def step(n: i32) = ... step(n - 1).0` over an `i32` result checked with score
1 and failed in `eval`. A group member's open obligations are decided when the
whole group has been inferred, so a sibling declared later can still determine
the type they wait on, and omitted types still generalize when the group
completes.

- Now accepted: members with authored binders and an omitted result that were
  rejected as recursive calls at another instantiation, across two or three
  members, over named-dimension tensors and under `grad`; an in-group call
  that instantiates an omitted type at a fully concrete type beside an
  authored binder, as `f(x, 3i32, n - 1)` in `def f[a](x: a, y, n: i32)`; an
  omitted result used at a concrete type by an in-group call, as
  `append(mk(n - 1), 1i32)` where `mk` returns `[]`; a literal pattern on a
  sibling's omitted result; and a `cast` of a sibling's omitted result to a
  bounded binder.
- Now rejected: a call inside the group that instantiates such a member's
  binders at other types, for example `f(y, x, n - 1)` inside
  `def f[a, b](x: a, y: b, n: i32) = ...`. It identifies two authored binders
  ([04-INF-6]), and the diagnostic names the repair: write the omitted types,
  after which a call at another instantiation of the member's own binders is
  admitted ([04-INF-2]). Some of these checked with score 1 before and failed
  in `eval`; others, such as a swapped `swap[a, b]` with its result omitted,
  checked and ran. A member whose every type is written is still referenced at
  its declared scheme.
- Now rejected: an in-group reference that instantiates an omitted type at
  another of the member's own omitted types, or at a type containing a
  variable ([04-INF-3]), such as `swap(y, x, n - 1)` in
  `def swap(x, y, n: i32)`, or two calls `f(x, n)` and `f(y, n)` that would
  give one member's two omitted types one type. `main` checked and ran some
  of these; their twins with the types written as authored binders check. A
  call at a concrete type whose result the body returns makes the body itself
  determine that type: after
  `def pick(x, n: i32) = if eq(n, 0) then x else pick(3i32, n - 1)`,
  `pick("s", 2)` is rejected, where it checked with score 1 and failed in
  `eval`.
- Now rejected: an ill-typed use of a member's omitted result, with the
  operation's own diagnostic or at the reference: `step(n - 1).0` over an `i32`
  result, `numel` over a list result, and an `eq` over a list reached through
  a recursive call, as the direct `eq([1i32], [1i32])` already was.
  Previously the check was dropped. So is `vmap(g)(k)` over a member's
  omitted tensor result `k`, as its direct form `vmap(g)(to_tensor(...))` and
  its written twin already were; `main` checked and ran it.
