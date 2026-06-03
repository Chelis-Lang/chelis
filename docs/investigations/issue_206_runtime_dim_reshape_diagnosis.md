# chelis#206 diagnosis: `reshape(x, [cast(shape(x, axis), int64), ...])` loses symbolic dim

## Symptom

The reproducer

```chelis
sig flatten_batch: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def flatten_batch(x) = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(4, int64)])
```

reports

```text
def 'flatten_batch' body doesn't match declared signature:
  body has type `(&tensor[Var(DimVar(_)), Lit(4), f32]) -> tensor[Wildcard, f32]`,
  declared type is `(&tensor[Var(DimVar(_)), Lit(4), f32]) -> tensor[Var(DimVar(_)), Lit(4), f32]`
```

The neighbor pattern `expand(b, 0, shape(x, cast(0, int32)))` works on a
declared `&tensor[n, 4, f32] -> &tensor[4, f32] -> tensor[n, 4, f32]`.

## Where the body type comes from

`crates/chelis-types/src/infer.rs::infer_reshape_app` is the handler for
`reshape(x, shape_list)`. After inferring `x`'s type and unifying the
shape list against `List<Int64>`, the `Type::Tensor(_, precision)` arm
runs:

```rust
let dims = list_literal_dims(shape_expr).unwrap_or_else(|| {
    let rank = list_literal_len(shape_expr).unwrap_or(1);
    vec![Dim::Wildcard; rank]
});
return Type::Tensor(dims, precision);
```

`list_literal_dims` only recognizes dim-list elements as
`Dim::Lit(<int>)` -- it walks the list (or `Cons` chain), extracts an
`i64` per element via `extract_int_for_dim`, and returns `None` if any
element is not a concrete integer (or `cast(N, int{32,64})` of a
concrete integer). On the reproducer, `cast(shape(x, cast(0, int32)),
int64)` is not a concrete integer, so the entire list extraction
returns `None` and the rank-correct but value-blind fallback
`vec![Dim::Wildcard; rank]` runs. That `Wildcard` rank-N vector cannot
unify with the sig's `[Var(DimVar(_)), Lit(4)]` dim list, so the body
type collapses to `tensor[Wildcard, f32]` (rank-1 after the dim
mismatch surfaces upstream), and the def-body check reports the
mismatch.

## Why `expand` "works"

`check_expand_signature` (same file) does NOT compute a precise output
type when the size arg is a non-literal expression. Its size-extraction
chain tries `extract_int_literal` first, then
`symbolic_dim_ref_name` (which recognizes only a bare `var` like `n`),
and falls back to `subst.apply(result_ty)` on the polymorphic fresh
ret-var. That fresh var unifies with whatever the sig declares -- the
sig provides the constraint and `expand` accepts it. So `expand` works
because it never asserts a specific output dim list; it lets the sig
shape the result type. `reshape`, in contrast, actively asserts
`vec![Wildcard; rank]` and forces a mismatch.

## Fix axis

The minimal fix is to extend `list_literal_dims` (or add a parallel
recognizer) so a single element of the shape list can resolve to
`Dim::Name(d)` or pass through the input's dim at that axis when the
element has the syntactic shape

```text
cast( shape( x, lit_axis ), int64 )
```

where `x` is the same input tensor being reshaped and `lit_axis` is a
concrete integer literal (possibly wrapped in `cast(int32)`).

Two recognizer surfaces:

1. `cast` may surface as the `(cast {} INNER TARGET)` tag form or as
   `(app {} (var {} cast) INNER TARGET)`. Both must be peeled.
2. `shape` always surfaces as `(app {} (var {} shape) X AXIS)`.

`X` must be a `(var {} NAME)` whose bound `NAME` matches the reshape
input expression's bound name. The simplest comparison uses the
syntactic form of the reshape input: when both sides are the same
`(var {} NAME)`, they reference the same value. (More elaborate cases
-- aliased lets, etc. -- fall back to `Wildcard`, which still preserves
the original behavior.)

Once a dim-list element is recognized, the dim we inject is the
corresponding entry of the input tensor's resolved dim list at axis
`lit_axis`. We then build the output dim list by mixing recognized
entries (input's dim at axis) with literal entries (`Dim::Lit`) and
fall back to `Wildcard` only for entries that are neither.

## Sibling sweep

- `expand` (read above) uses a different, looser strategy and relies
  on the sig to shape the result. The runtime-dim cases there are
  covered by `symbolic_dim_ref_name` for direct `var` refs and by the
  sig's constraint for other cases. Adding the same recognizer there
  would help, but: a sweep for downstream-failing reproducers under
  `expand` is out of scope unless the issue surfaces one.
- `view` is NOT a current Chelis builtin (no entry in
  `crates/chelis-types/src/builtins.rs::BUILTIN_NAMES`).
- `broadcast_to` is NOT a current Chelis builtin (only `broadcast_pair`
  exists, and it takes two tensors -- not a dim list).
- No other builtin in `tensor_unop` / `tensor_expand_to_out` /
  `generic_binop_first_borrow` registrations consumes a runtime dim
  list besides `reshape` and `expand`.

So the fix's blast radius is limited to `infer_reshape_app`.

## IR-lowering interaction

`crates/chelis-ir/src/lower.rs::lower_reshape_app` extracts a dim list
from `args[1]` via `extract_dim_list` (literal-only) and falls back to
`ty.dims.clone()` if extraction fails. `ty.dims` is the typer-emitted
output type. If the typer produces the right symbolic dims, IR lowering
inherits them automatically -- no IR-side change required.

## Acceptance

The two reproducer functions both type-check. The four positive
fixtures in `crates/chelis-types/tests/issue_206_runtime_dim_reshape.rs`
go green. The four negatives (literal-only baseline, foreign-tensor
shape source, arithmetic wrapper) keep their current behavior.
