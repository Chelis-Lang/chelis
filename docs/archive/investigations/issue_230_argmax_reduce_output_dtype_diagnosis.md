# Issue #230: argmax_reduce / argmin_reduce output dtype is input dtype, not int64

## Symptom

```chelis
def forward(x: tensor[2, 3, f32]) -> tensor[2, int64] = argmax_reduce(x, 1)
```

`chelis check` rejects this with a `TypeMismatch`:

```
def 'forward' body doesn't match declared signature:
  body has type `(tensor[Lit(2), Lit(3), f32]) -> tensor[Lit(2), f32]`,
  declared type is `(tensor[Lit(2), Lit(3), f32]) -> tensor[Lit(2), int64]`
```

The body type is `tensor[Lit(2), f32]` (input dtype preserved) but the
canonical contract per `packages/chelis-std/src/tensor/reduce.ch` is
`tensor[Lit(2), int64]` (output is always int64 indices, regardless
of the input dtype).

`argmin_reduce` has the same shape of bug.

## Bisect

A bisect across the five commits between `v0.7.13` and `v0.7.14`
(merged PRs #224 / #223 / #225 / #227 / #228) shows the buggy output
type was already present at `v0.7.13` (and at every earlier 0.7.x
release back to `v0.7.0`). The `check_reduction_signature` function
was added in `9212769 fix: normalize negative axes in IR lowering` and
the `out_dims.remove(axis)` plus "result precision is operand
precision for non-sum reductions" semantics have been in place since
then.

This is therefore a **latent typing bug surfaced by hydronnx PR #2's
ArgMax compile-run-parity test** when hydronnx retargeted its
build-test from `v0.7.13` to `v0.7.14` — not a regression introduced
by any of the five 0.7.13 -> 0.7.14 commits. Bisect commands and
output:

```sh
git checkout v0.7.13 && cargo run -p chelis-cli --bin chelis --quiet -- \
  check /tmp/issue230.ch
# -> body has type ... tensor[Lit(2), f32], declared ... tensor[Lit(2), Lit(1), int64]

git checkout v0.7.14 && cargo run -p chelis-cli --bin chelis --quiet -- \
  check /tmp/issue230.ch
# -> body has type ... tensor[Lit(2), f32], declared ... tensor[Lit(2), Lit(1), int64]

git checkout v0.7.0  && cargo run -p chelis-cli --bin chelis --quiet -- \
  check /tmp/issue230.ch
# -> body doesn't match declared signature (older diagnostic format,
#    same underlying mismatch)
```

The hydronnx PR #2 CI message reporting `Lit(1)` keepdim shape in the
body (rather than the rank-reduced rank-2 shape the direct CLI
reproducer produces) is an artifact of hydronnx's emitted Surf source
shape; the dtype half of the mismatch is identical. The actionable
diagnosis is the same in both cases: output **dtype** is wrong.

## Root cause

`crates/chelis-types/src/infer.rs::check_reduction_signature` at line
~12391 hard-codes "result precision = operand precision" for every
reduction except `sum`:

```rust
let result_prec: TensorPrec = if name == "sum" {
    // §5.7.1 widening table for narrow integer operands
    ...
} else {
    prec.clone()        // every other reduction reuses input precision
};
```

`argmax_reduce` and `argmin_reduce` are reduction-family names but
their output semantics are different: they emit element indices, not
reduced operand values. The canonical std signatures
(`packages/chelis-std/src/tensor/reduce.ch`):

```text
sig min:     &tensor[a, b, p] -> int32 -> tensor[b, p]
sig prod:    &tensor[a, b, p] -> int32 -> tensor[b, p]
sig argmax:  &tensor[a, b, p] -> int32 -> tensor[b, int64]
sig argmin:  &tensor[a, b, p] -> int32 -> tensor[b, int64]
```

declare argmax/argmin output as `int64`, distinct from min/prod which
preserve `p`. The IR lowering and host-runtime evaluator already
collapse the reduced axis correctly (rank `n` -> rank `n-1`); only the
**dtype** label needs the special case.

`crates/chelis-ir/src/dag.rs::RiscOp::Argmax` already documents the
intended Phase 3j-pre semantics:

> per Phase 3j-pre, argmax/argmin logically return integer indices.
> The evaluator stores them as f64 integer-valued floats and the
> type system carries whatever precision the caller assigns ...

The note's "whatever precision the caller assigns" was correct for
the int64-on-the-caller's-signature path but wrong for the inferred
let-binding path where the user does not write the type. The type
system needs to canonically assign `int64` itself; let-binding sites
that don't write the type now resolve to int64 and match the std
signatures.

The host-runtime / IR evaluator continues to store integer-valued
floats internally, per the Phase 3j-pre Batch 1 caveat; the int64
type-system label is independent of the storage layout. The
`TODO(phase3j): widen backend runtime to carry Int64 tensors
natively.` in `dag.rs` tracks the eventual storage widening — out of
scope for this fix.

## Sibling sweep

Other reductions in the family (`sum`, `max_reduce`, `min_reduce`,
`prod_reduce`, `mean`) correctly preserve input dtype per
`spec/04-type-system.md` §5.7.1 and per the std signatures. They have
no analogous "always-int64" rule because they return reduced operand
values, not indices. The fix is narrow to `argmax_reduce` and
`argmin_reduce`; the sibling-sweep tests in
`issue_230_argmax_reduce_output_dtype.rs` lock that the other
reductions stay precision-preserving.

## Fix

`check_reduction_signature` adds an `argmax_reduce`/`argmin_reduce`
branch that returns `TensorPrec::Concrete(Prim::Int64)` for the
result precision, independent of the input precision. Shape
computation (`out_dims.remove(axis)`) is unchanged. The fix is
~3 lines and surgical to one helper.

The acceptance oracle is
`crates/chelis-types/tests/issue_230_argmax_reduce_output_dtype.rs`:
all 11 tests pass, including positive (f32/int32/int64 inputs all
produce int64 outputs), negative (wrong declared output dtype is
rejected with the int64 body type surfaced in the diagnostic), and
the sibling-sweep regression guard on the other reductions. The
host-runtime parity tests
(`crates/chelis-compiler-api/tests/issue_230_argmax_argmin_runtime.rs`)
keep passing — the IR evaluator was already producing the correct
index values; only the tensor dtype label changes.
