# `pad_sequences_to` and runtime-shaped tensor builders: design note (chelis#143 follow-up)

Design note rather than diagnosis: `pad_sequences_to` is harder than
sub-issue (A) because its output dim is not statically derivable from
a literal — it's a runtime `int64` value. Threading shape info through
requires either compile-time evaluation, a dependent-type-lite sig form,
or a documented "this stays Wildcard until you ascribe" policy.

Cross-references:

- Parent issue: `chelis#143`; sub-issues `chelis#158` (A) and
  `chelis#159` (B)
- Sub-issue (A) diagnosis: `docs/archive/investigations/issue_143a_to_tensor_shape_erasure_diagnosis.md`
- Sub-issue (B) diagnosis: `docs/archive/investigations/issue_143b_let_ascription_no_propagation_diagnosis.md`
- Builtin registration: `crates/chelis-types/src/builtins.rs:875-876`
  (`pad_sequences` is `generic_binop`, `pad_sequences_to` is `generic_triop`)
- Spec: `spec/04-type-system.md` (no dim-from-int-literal rule)

This is a design discussion, not a bug to fix. The decision affects
how the chelis#143 repro should ultimately catch its shape mismatch.

## Why this is different from sub-issue (A)

The chelis#143 repro:

```chelis
q = pad_sequences_to([[cast(0.0, f32)]], cast(8, int64), cast(0.0, f32))
k = pad_sequences_to([[cast(0.0, f32)]], cast(5, int64), cast(0.0, f32))
scaled_dot_product_attention(q, k, v, scale)
```

For `to_tensor([1.0, 2.0, 3.0])`, the desugarer can read `3` from
`items.len()` at parse time. For `pad_sequences_to(input, pad_target,
pad_value)`, the pad target is a runtime int64 value
(`cast(8, int64)`). The type checker today has no const-evaluation
machinery, no refinement type for "int64 known at compile time," and
no way to express "the output's second dim equals the value of
argument 2."

So even if sub-issue (A) ships and gives `to_tensor` shape-aware
inference, `pad_sequences_to` still produces a `Wildcard` second dim
unless something else changes.

## Option survey

### (a) Const-folding for `cast(<int_literal>, int64)`

When `pad_sequences_to`'s pad-target argument is statically
`cast(<integer-literal>, int64)`, propagate the literal's value as a
`Lit` dim into the output's second dim. Outside that exact shape
(variable arg, function-call arg, arithmetic result), fall back to
`Wildcard`.

Scope: small implementation, narrow utility. The repro pattern is
exactly this case, so the static catch lands on the original issue. A
real user threading a non-literal pad target gets no static catch.

Pros: easy; covers the issue's stated repro.

Cons: looks like a fix but doesn't address the general problem.
Encourages "best-effort static catch" framing instead of a principled
shape-tracking story. Users writing `let n = 8 in pad_sequences_to(..., cast(n, int64), ...)`
expect the same behavior as the literal version and don't get it.

### (b) Dependent-typing-lite sig form

Allow a `[name: T]` syntax in a sig that says "this argument's
*value* is bound to type-level name." Concretely:

```chelis
sig pad_sequences_to: list[list[T]] -> [n: int64] -> T -> tensor[?, n, T]
```

The type checker then unifies the call site's second argument's
*value* (if known) against `n`. When the value is a `cast(8, int64)`
literal (or a binding traced back to one), `n := 8` and the output is
`tensor[?, Lit(8), T]`. When the value is opaque, `n` stays as a
dim variable and propagates.

This is a real language change: parser, type checker, IR, backends,
spec. The benefit is generality — once the form exists, similar
runtime-shape builtins (`split`, `reshape`, `slice`) can use it.

Scope: large. Multi-phase rollout (parser → type checker → spec → all
builtins that should adopt it).

Pros: principled. Handles the broader class of "value-tracked dims"
that recurs across tensor APIs. Aligns with existing partial-static
inference moves like §P10b.

Cons: significant scope. The chelis#143 user gets a static catch
only after a multi-quarter project. Risk of bikeshedding the surface
syntax (the `[n: int64]` notation is illustrative, not proposed).

### (c) Document `Wildcard` return, depend on (B) for user-side ascription

Accept that `pad_sequences_to`'s output dims are runtime-determined
and document the return type as `tensor[Wildcard, Wildcard, T]`. Push
the user to ascribe the let-binding (`q: tensor[1, 8, f32] = pad_sequences_to(...)`),
which — once sub-issue (B) is fixed — properly constrains the var
chain into the sig.

Scope: documentation + sub-issue (B) prerequisite. No compiler change
specific to `pad_sequences_to`.

Pros: smallest change. Honest about runtime-dynamic shapes.
Composable with sub-issue (B)'s fix.

Cons: ergonomics: users re-type the dims when they're "obvious" from
the call site. A library author writing
`pad_sequences_to(input, cast(8, int64), default)` has to ascribe the
output every time. Easy to forget; easy to ascribe wrong.

### (d) `pad_sequences_to` returns Wildcard; type-check the ascription only

A weaker version of (c): the sig says `tensor[Wildcard, Wildcard, T]`,
and users must ascribe the output. Combine with a lint rule that
requires the ascription whenever `pad_sequences_to`'s result flows
into a sig with shared dim vars. The lint catches the most common
foot-shoot (the chelis#143 case) without compiler changes.

Scope: medium. New lint rule. Depends on sub-issue (B) for the
ascription to actually constrain anything.

Pros: pragmatic; user sees a warning at the exact point shape-erasure
matters. Adds no language complexity.

Cons: yet another lint rule the user can squelch or work around. Not
truly type-system-enforced.

## Recommendation

**Short term: ship (c) + sub-issue (B).** Document the return type,
land (B), and tell the chelis#143 user to ascribe. The static catch
lands at the call site, with no compiler change beyond what (B)
requires. This unblocks Chelis-Lang/school's negative-test predicates.

**Medium term: revisit (b).** Once §P10b expands to cover shape (per
sub-issue (A)'s recommendation), the "value-tracked dims" pattern is
the natural next axis. Schedule the design work after sub-issue (A)
ships and the spec amendment for shape-aware contextual literals is
in place.

**(a) is rejected.** Looks like a fix; behaves like a trap.

**(d) is parallel-with-(c).** A lint rule is harmless and could ship
alongside (c) if a user-side reminder turns out to be valuable.

## Caveats

`pad_sequences_to` is not the only runtime-shaped builtin. The same
analysis applies to:

- `pad_sequences` (`crates/chelis-types/src/builtins.rs:875`,
  `generic_binop`)
- `reshape`, `split`, `gather`, `scatter` (search `builtins.rs` for
  the `generic_*op` registrations)

Any compiler-level fix for `pad_sequences_to` (option (b) or (a))
should be designed to cover the full set, or it will leak the same
shape-erasure footgun on a sibling. (c) is the only option that
generalizes for free.

## What this means for chelis#143

The chelis#143 issue is reopened pending sub-issues (A) and (B).
Once (B) ships, the issue's original repro can be rewritten as:

```chelis
q: tensor[1, 8, f32] = pad_sequences_to([[cast(0.0, f32)]], cast(8, int64), cast(0.0, f32))
k: tensor[1, 5, f32] = pad_sequences_to([[cast(0.0, f32)]], cast(5, int64), cast(0.0, f32))
scaled_dot_product_attention(q, k, v, scale)
```

…and the call to `scaled_dot_product_attention` will trip
`DimensionMismatch` at type-check. The ascription is load-bearing;
the user pays a typing cost but gets the static catch the spec
implies.

The unascribed form of the issue's repro will catch via (A) once
`to_tensor` literal-shape inference ships, but `pad_sequences_to`-
derived tensors will continue to require ascription until (b) is
designed and shipped.
