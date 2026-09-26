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
- A `grad` of an operand whose type was still a variable, such as the lambda
  parameter in `fn (g) -> grad(g)(3.0f32)`, published a result nothing checked
  ([#2626](https://github.com/Chelis-Lang/chelis/issues/2626)). It is now
  decided when the operand binds, so that lambda applied to a function with a
  tensor result is rejected with `grad`'s own diagnostic, and
  `def h(g) = grad(g)` is rejected at the declaration boundary.

A recursive group whose members omit types is typed by [04-INF-2] and
[04-INF-5] together ([#2590](https://github.com/Chelis-Lang/chelis/issues/2590)).
Inside its group, a member whose declared header omits a type, such as its
result, shares its authored binders with every in-group reference, and each
reference takes its own instance of each omitted type. When the group
completes, each reference is typed at the type the member's body determines,
which it never narrows: an instance of an omitted type must be that type
itself or a fully concrete type, and anything else is polymorphic recursion
([04-INF-3]), reported at the reference with both types. The verdict is
decided on the group's solved types, so it is the same in every declaration
order. A use that disagrees with the body-determined type is also reported at
the reference. Previously
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
- Now rejected, in every declaration order: in-group calls that swap or merge
  two of a member's omitted types ([04-INF-2], [04-INF-3]), since a published
  signature is never narrowed to fit an in-group use. The call can do it
  directly, as `swap(y, x, n - 1)` in `def swap(x, y, n: i32)` or
  `f(x, x, n - 1)` in `def f(x, y, n: i32)`, or through a sibling, as the two
  calls `f(x, n)` and `f(y, n)` to a sibling that calls back with one type for
  both, or `g(x, y, n - 1)` to a sibling whose body gives its two parameters
  one type. The diagnostic tells the author to write the member's signature
  with explicit type binders. `main` checked and ran these. The fully written
  twin of the swap, `def swap[a, b](x: a, y: b, n: i32) -> (a, b)`, still
  checks. `main` already rejected a pair whose calls swap the omitted types
  around the group, such as `b(y, x, n - 1)` inside `def a(x, y, n: i32)` with
  `b` calling `a(u, v, n - 1)`, and it stays rejected in every order.
- Now rejected: a group whose omitted types would grow with every in-group
  call, such as `def f(n: i32) = if eq(n, 0) then [] else [f(n - 1)]`, as
  polymorphic recursion ([04-INF-3]); `main` checked it. The
  polymorphic-recursion diagnostics no longer suggest moving the call into a
  separate non-recursive helper `def`: that helper would call into the group
  and be called from it, so it would join the group.
- A call at a concrete type whose result the body returns makes the body itself
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
- `grad` decided whether its function's result was a floating scalar, and
  whether each parameter was differentiable, where it was inferred, while a
  type that the group determines for a member writing no signature was still
  a variable, so the verdict depended on the declaration order
  ([#2626](https://github.com/Chelis-Lang/chelis/issues/2626)). It now waits
  for that type and is decided on it, in every order. Now accepted in every
  order: `d = grad(fn (z: f32) -> f(z, 0i32))` inside `g`, where `def f(x, n)`
  returns `mul(x, x)`, which `main` checked with `f` declared first and
  rejected with `g` first as "grad requires a scalar floating output", a false
  rejection; `grad(f, wrt=x)` and `grad(fn (z) -> f(z, 0i32))` over such an
  `f`'s parameter, which `main` rejected with `g` first; and
  `grad(ev)(y, 0i32)` over a sibling `def ev(x, n)` whose parameter only that
  call determines, which `main` rejected in every order. In every order, a
  result that is not a floating scalar is rejected with `grad`'s own
  diagnostic, and a parameter the group makes an integer is rejected under
  `wrt` with it and left out of the gradient without `wrt`, where `main`
  reported a variable's type in one order. `grad` over any other variable is still decided where it is
  inferred, which gives one verdict in every order: over a generic function
  instantiated at the call, inside a generic function, over a lambda parameter
  a later application determines, and over a member whose signature omits only
  some types, as in `def f(x, n: i32) -> f32`, whose every reference takes a
  fresh instance of them. `chelis build --target c` does not implement `grad`
  of a generic function.
