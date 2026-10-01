# `to_tensor` / `pad_sequences_to` shape erasure diagnosis (chelis#143 sub-issue A)

Diagnosis pass for the chelis#143 sub-bug surfaced during PR #151
red-team. The original commit (1681b52) was a no-op; the bug surfaced
in the issue's repro is in tensor-builder typing, not `unify_dim`.

Cross-references:

- Issue: `chelis#158` (this sub-issue); parent `chelis#143` ("sig dim
  variables not unified across parameters at type-check (0.7.10)")
- Sibling diagnosis: `docs/archive/investigations/issue_143b_let_ascription_no_propagation_diagnosis.md`
- Design note for the harder case: `docs/archive/investigations/issue_143_pad_sequences_to_design_note.md`
- Pinning test: `crates/chelis-types/tests/issue_143a_to_tensor_shape_erasure.rs`
  (one `#[ignore]` probe + two passing counter-probes; the probe
  still fails as expected after the sub-issue (B) prototype lands —
  the (A) bug is genuinely independent of (B))
- Unify-layer corroboration test: `crates/chelis-types/src/unify.rs`
  `unify_dim_cannot_recover_lit_from_two_var_tensors` (proves the
  fix-site must be upstream of `unify_dim`)
- Prior-work signal: `crates/chelis-types/tests/contextual_tensor_literal.rs:1-21`
  (WS-B2 §P10b / §5.6, element-type contextual narrowing — *not* shape
  inference)
- Spec: `spec/02-surf-syntax.md` §P10b, `spec/04-type-system.md` §5.6

No code changes for sub-issue (A) in this branch — only a unify-layer
corroboration test (see "Unify-layer corroboration" below). The full
fix for (A) requires spec amendment work and a desugar change.

## Bug surface

A function whose `sig` declares the same dim variable on multiple
parameter positions should reject calls whose tensors have mismatched
concrete values for that dim. At `chelis check` time today, the check
silently passes when the call-site tensors come from `to_tensor` or
`pad_sequences_to`:

```chelis
sig pair_id: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(x, y) = x
def caller() -> tensor[3, f32] = {
  a = to_tensor([1.0, 2.0, 3.0])
  b = to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
  pair_id(&a, &b)
}
```

The static check today produces only an unrelated `TypeMismatch` on
the return type — no `DimensionMismatch` is emitted for the
`pair_id(&a, &b)` call, despite the sig demanding a shared `n`:

```
[TypeMismatch] def 'caller' body doesn't match declared signature:
  body has type `() -> &tensor[Var(DimVar(40)), f32]`,
  declared type is `() -> tensor[Lit(3), f32]`
```

The sig dim var `Var(DimVar(40))` is unbound. The mismatch surfaces
only at runtime when the function body does a shape-checking primitive
(`matmul`, etc.), as documented in the chelis#143 repro from
Chelis-Lang/school.

The shared-dim sig machinery itself is fine — the counter-probe in
`issue_143a_to_tensor_shape_erasure.rs::parameter_ascription_with_shared_sig_dim_does_trip_dim_mismatch`
passes today: when `caller` accepts `a: &tensor[3, f32]` and
`b: &tensor[5, f32]` as direct parameter ascriptions, the `pair_id(a, b)`
call correctly trips `DimensionMismatch`. The bug is upstream of `unify_dim`.

## Bug site

`to_tensor` is registered as a fully polymorphic unary op:

`crates/chelis-types/src/builtins.rs:873`

```rust
generic_unop("to_tensor", &mut env, &mut vg);
```

`generic_unop` at `crates/chelis-types/src/builtins.rs:387-396` produces:

```rust
let scheme = Scheme {
    tvars: vec![input, output],
    dvars: vec![],
    body: Type::Fn(vec![Type::Var(input)], Box::new(Type::Var(output))),
};
```

Reading the scheme: `to_tensor : ∀α β. α → β`. No dim variables; no
constraint between input and output. When the user writes
`to_tensor([1.0, 2.0, 3.0])`, the type checker:

1. Desugars the list literal to a `(app Cons 1.0 (app Cons 2.0 (app Cons 3.0 Nil)))`
   chain (`crates/chelis-surf/src/desugar.rs:1436-1442`,
   `desugar_list_literal`).
2. Wraps the chain in a bare `(app to_tensor <chain>)`
   (`crates/chelis-surf/src/desugar.rs:1500`, `desugar_list_as_tensor_literal`).
3. Unifies the chain's inferred type against `Type::Var(input)`, leaving
   `Type::Var(output)` free.
4. Forces `output` into a tensor shape at the call site (the sig param
   demands a tensor). The unifier produces `tensor[Wildcard, f32]`
   because there is no constraint on the dim.

`pad_sequences_to` is the same shape (`crates/chelis-types/src/builtins.rs:876`):

```rust
generic_triop("pad_sequences_to", &mut env, &mut vg);
```

→ `pad_sequences_to : ∀α β γ δ. α → β → γ → δ`. No dim info.

The list length **is observable at desugar time** (`Expr::List(Vec<Expr>, Span)`
in `crates/chelis-surf/src/ast.rs:125`; `items.len()` is available at
the entry to `desugar_list_as_tensor_literal`), but the desugarer
discards it before handing the chain to a generic `to_tensor`.

## Prior work in the area

`crates/chelis-types/tests/contextual_tensor_literal.rs` (the WS-B2
suite) already handles a related-but-different problem: contextual
**element-type** narrowing for tensor literals (spec §P10b / §5.6).
When the user writes `xs: tensor[3, f64] = [1.0, 2.0, 3.0]`, the
desugarer narrows the unsuffixed `1.0` literals to `f64` instead of
their `f32` default — see `crates/chelis-surf/src/desugar.rs:670-682`
(Position 1) and the test suite's four-position enumeration. That
work pins *element types*, not *shapes*. There is no companion path
for "the literal's outer length determines a `Lit` dim."

The closed set of four contextual positions in §5.6 is element-type-only
by design; extending it to shape inference is a spec change, not a bug
fix. See `crates/chelis-surf/src/desugar.rs:1476-1479`:

> Future agents reading this code: do NOT silently extend the closed
> set. Adding new positions ... is a spec change requiring an
> amendment to §5.6 / §P10b.

## Option survey

### (a) Special-case `to_tensor(list_literal)` in desugar — recommended

When the desugarer sees `to_tensor` applied to an `Expr::List` whose
elements are all-statically-known (or recursively `Expr::List` of
all-statically-known), emit a shape-annotated tensor type. Concretely:

- Replace the bare `(app to_tensor <Cons-chain>)` at
  `crates/chelis-surf/src/desugar.rs:1500` with a form that carries
  `(t-tensor [Lit(<outer-len>), Lit(<inner-len>), ...] <prec>)` as a
  type ascription on the resulting application.
- Drive the chain through a Deep `ascribe` node so the type checker
  immediately unifies `to_tensor`'s output `Type::Var` against the
  shape-annotated tensor type.

Risk: rank > 2 nested literals with ragged inner lengths must produce
a clear error rather than a silently truncated shape. The desugarer
should reject ragged shapes (today it silently produces a `Cons` chain
whose typed length is the outer count only).

Scope estimate: medium. One new helper in `chelis-surf/desugar.rs`,
~20-40 lines including the rank-walk and ragged-shape guard. One new
test position in `contextual_tensor_literal.rs` (positions 5-8 for
shape-aware contextual literals, or a separate file if the spec keeps
the four-position set element-only). Spec amendment required.

### (b) Special-case `to_tensor` in the type checker

Replace the `generic_unop("to_tensor", ...)` registration with a custom
scheme that introspects its argument. Today's checker doesn't have a
"look at the call's argument and emit a shape" hook; the natural place
is in `crates/chelis-types/src/infer.rs` near `infer_call` /
`infer_app`, with a name-based fast path for `to_tensor`.

Risk: encodes a builtin-name special case in the inference path, where
today special cases are pushed into desugar. Less idiomatic given the
desugar precedent for `cast`, `Cons`, and the element-type narrowing
work.

Scope estimate: medium-large. Comparable line count to (a) but in a
hotter path.

### (c) Defer to user-side ascription

Require users to write `q: tensor[3, f32] = to_tensor(...)` and let
the let-binding ascription propagate. This depends on sub-issue (B)
being fixed first. Even with (B) fixed, the ergonomics regress
relative to `q = to_tensor([1.0, 2.0, 3.0])` — the user re-types what
the literal already states.

Useful as a workaround in the meantime once (B) is fixed; not a real
solution.

### (d) Defer entirely (do nothing)

The chelis#143 user gets `chelis check` score 1.0 with `errors=[]` for
a program whose tensors fail at runtime. The static-catch guarantee
that the sig system advertises is decorative. Rejected.

## Recommendation

Pursue (a). It matches the prior pattern (desugar pushes contextual
type info; checker stays generic), keeps the inference path simple,
and lets the spec amendment be local to §P10b / §5.6.

Order of operations:

1. Land (B) first — that fix is smaller and unblocks ascription as a
   workaround for users on (A) before (a) lands.
2. Amend `spec/02-surf-syntax.md` §P10b and `spec/04-type-system.md`
   §5.6 to add shape inference for full-static literal contexts.
3. Implement (a) in `chelis-surf/desugar.rs` with the ragged-shape
   guard.
4. Remove the `#[ignore]` from
   `crates/chelis-types/tests/issue_143a_to_tensor_shape_erasure.rs::to_tensor_mismatched_list_lengths_should_trip_dim_mismatch`
   and confirm it passes.
5. `pad_sequences_to` remains broken until its design note is resolved
   (see sibling doc).

## Why this matters

The `sig` system is Chelis's main differentiator: dim-tracked tensors
catch shape bugs before execution. With shape erasure live in the most
common tensor-builder paths (`to_tensor`, `pad_sequences_to`), the
sig guarantee is purely advisory for any code that builds tensors from
literals — which is essentially all user code.

Downstream (Chelis-Lang/school) currently routes around the gap by
asserting on runtime-fail substrings in negative tests; the static
catch the spec implies is not actually shipped. Worth filing as a
distinct workstream rather than rolling into the next phase
miscellaneously.

## Unify-layer corroboration

The pre-existing diagnosis above (file-by-file source-reading)
concluded that the bug must be upstream of `unify_dim` because
`to_tensor`'s scheme contains no dim variables — there is no place
for shape info to enter the type system through that builtin. The
diagnosis is now corroborated empirically by a focused unit test:

`crates/chelis-types/src/unify.rs::tests::unify_dim_cannot_recover_lit_from_two_var_tensors`

The test directly constructs the scenario the chelis#158 repro hits:

- Two sig-instantiated tensor parameters that share a single dim
  variable `d_sig` — modeling `pair_id : &tensor[d_sig, f32] ->
  &tensor[d_sig, f32] -> ...`.
- Two caller tensors with fresh dim variables `d_caller_a` and
  `d_caller_b` (modeling the shape-erased output of `to_tensor`).
- Symmetric case: two caller tensors with `Dim::Wildcard` (the
  actual variant that to_tensor's generic output forces into the
  unifier when paired with a tensor-shaped sig position).

For both shapes, the test unifies each caller arg against the
matching sig position and asserts that after both unifications,
`subst.apply_dim(Dim::Var(d_sig))` resolves to *something that is
not a `Lit`* — specifically `Dim::Var(_)` or `Dim::Wildcard`. The
test passes today.

What this proves:

1. **Unify is correct.** Given `Var`/`Wildcard`-only inputs, the
   unify-layer behavior is exactly what the type theory demands:
   no constraint, no error, no manufactured value.
2. **The fix site is upstream.** A future "make unify smarter"
   proposal cannot fix #158 without violating this contract — the
   information needed to bind `d_sig` to a concrete value never
   reaches unify in the first place.
3. **The minimum-change point for (A)** is exactly where the
   diagnosis recommends: in the desugar pass for `to_tensor` over
   list literals, where `items.len()` is observable and can be
   threaded into the resulting tensor type as a `Lit` dim.

The micro-test pins this contract so any future PR claiming to fix
#158 in `unify_dim` can be checked against the asserted property:
if the test fails under your change, your fix is in the wrong place.

## Why the (B) prototype does not also fix (A)

The (B) prototype patch (in `infer_let`, on this same investigation
branch) makes block-scoped let-binding ascriptions propagate into
generic-builtin RHSes. So an ASCRIBED let-binding now correctly
constrains `to_tensor`'s output:

```chelis
// Post-(B), with explicit ascription, the shape is now constrained:
a: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])  // a: tensor[Lit(3), f32]
```

But the (A) probe test uses UNASCRIBED let-bindings:

```chelis
a = to_tensor([1.0, 2.0, 3.0])  // a: tensor[Var(?), f32]
b = to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
```

With no ascription, no metadata is injected by the desugarer, and
the (B) prototype has no hook to grab onto. The (A) probe continues
to fail with the same `body has type () -> &tensor[Var(DimVar(40)),
f32]` diagnostic before and after (B) lands. The bugs are
orthogonal: (B) covers the ascribed-but-erased path; (A) covers the
unascribed-but-should-be-inferrable path.
