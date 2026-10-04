# Implicit Linearity

**Status:** Implemented verified ownership lowering over the `copy-drop` foundation.
Its conservative scope-end lifetime strategy is superseded as a target contract by
`compiled_value_ownership.md`; that plan retains last-use release as successor work.
**Owning specs:** `spec/03-deep-syntax.md`, `spec/04-type-system.md`,
`spec/05-risc-primitives.md`, and `spec/design/borrow_typed_primitives.md`.

## Summary

Chelis keeps explicit `copy()` and `drop()` source forms for compatibility, but the
compiler now owns routine linearity ceremony. The shipped implementation inserts drops
at scope exits and copies at consuming fan-out sites. The controlling [04-LIN-8]
contract requires the successor ownership lowering to place terminal releases at the
last-use/post-dominance boundary and before tail calls; that is correctness work for
compiled heap values, not an optional allocation optimization.

The verified `OwnershipProgram` makes `RiscOp::Copy` and `RiscOp::Drop` real
ownership operations: every lowered linear value has exactly one terminal path,
and every backend receives the borrow/move/clone disposition before emission.
Slot planners treat terminal consumes and `Drop` as authoritative live-range
closes.

## Read-only numeric families

The controlling borrow rule is spec/05 §1.3.1. Comparisons, elementwise
extrema, and unary activations leave their tensor inputs live. The checker
uses the comparison identity enumeration for every member; host coarse typing
returns an operand-shaped bool tensor for every direct tensor comparison,
and a scalar bool for scalar or recursive equality. A comparison containing a
host-evaluated operand stays on the host lane so runtime operand agreement is
checked before comparison. The C emitter supplies that family through its
existing runtime or typed elementwise comparison arms.

The regression oracle is `cargo test -p chelis-cli --test
issue_1248_read_only_families`: family and operator reuse succeeds, reuse after
`realize` fails, and host-produced tensor comparisons execute with eval/C parity
while incompatible shape or dtype inputs fail checking.

## Drop Insertion

An owned linear value without a consuming use receives an inserted `Drop`. The shipped
foundation places it at the end of the containing scope; the successor pass moves it to
the earliest post-dominating point after last use as [04-LIN-8] requires. Multi-branch
flow is branch-local: if a value is consumed on one branch and not the other, the
unconsumed branch receives a `Drop`. `if` and `match` branches are analyzed
independently.

Loops introduce an iteration scope. Values defined inside a loop are consumed or
dropped by the end of that iteration. Values defined outside a loop and consumed inside
the loop require the existing loop-aware ownership rules; cases where the compiler
cannot prove a single terminal path remain hard errors.

`Drop` is not a numerical computation. AD treats it as a gradient sink. For an owned
source, tensor slot planners close the dropped value's live range there. C and HIP emit the exact
descriptor release selected by the verified directive; Metal consumes the same
directive as a typed no-device-owner disposition. A `Drop` over a borrowed DAG `Load`
is instead a typed logical discard: it emits no release and leaves the external owner
live; Metal consumes that disposition without inventing a device owner. Independent
liveness may refine allocation details but must not extend a program owner past an
owned terminal operation.

## Copy Insertion

The compiler inserts `Copy` for source-level consuming fan-out: a value used in more
than one non-borrow consuming position. Copies are inserted for all consume sites before
the final consume site; the final consume site takes the original.

Borrows do not count as fan-out. Multiple `&T` uses share the same source. A value
passed once to a consuming function after any number of borrows is not fan-out and does
not receive a copy.

Explicit source `copy()` lowers to the same `RiscOp::Copy` used for inserted copies.
Cost and fitness signals intentionally do not distinguish explicit and inserted
copies.

## Destructured Components

Copy insertion has one carve-out, stated normatively at `spec/04-type-system.md` §8.3:
a **destructured component** does not receive an inserted copy at consuming fan-out.
The earlier consuming use is a hard error, and the fix is an authored `copy()`.

A component is a binding introduced by a `let` whose pattern is not a single name. It
is projected out of the destructured value — the desugarer emits a `tuple-get` per
component into a synthesized intermediate — so a component is a fresh owned value
rather than an alias of a shared source. That is what disqualifies it from copy
insertion, which the rest of this section justifies by the source and the copy sharing
one DAG node.

The exception is a property of the **binding**, not of a region of the program and
not of a name. It attaches to each binding the destructuring `let` introduces and to
nothing else. In particular it does not attach to:

- the value that was destructured. That value is consumed in the enclosing scope
  where the destructuring `let`'s right-hand side was evaluated, and its own fan-out
  is copied normally.
- any binding that merely appears after the destructuring `let`. Statements in a block
  nest as let-bodies, so a region-shaped reading of this rule silently covers the
  entire remainder of the block.
- a discarding `let _ = e`, which introduces no component at all. The value is still
  evaluated for effect, and the intermediate that holds it is unnameable.
- a component's name after an ordinary `let` re-binds it. The new binding is an
  ordinary one.

### New declaration regions

A closure body, a `match` arm, and an `if` or `match` branch each open a new
declaration region. Names the region works on are re-declared in it, and a new
declaration is an ordinary binding: the component exception does not cross into the
region. A component's fan-out inside a closure body or a branch is therefore copied
like any other value's.

This is a consequence of the rule being per-binding, not a separate rule, and it is
uniform across the three region kinds. A destructuring `let` authored *inside* a region
introduces components of that region, which are excepted there on the same terms.

Consuming a component inside a branch and again after the branch joins is still fan-out
on one binding: the branch's consume survives the join, and the later use is the error
the carve-out describes. The region boundary governs which bindings are components, not
whether consumes propagate out of it.

Two things follow, and they are easy to conflate:

- *Which* binding a region-crossing consume lands on is fixed by the binding's
  identity, and identity does not belong to a region. A component's value lives in a
  synthesized carrier, and that stays true inside a branch even though the region has
  cleared the component *exception* for the names it re-declares. Resolving a carrier by
  asking the region-relative question loses the carrier inside every branch, which lets a
  branch consume fail to survive the join — the opposite of the rule above.
- The join carries a branch's consume out onto a binding whose outer record is an
  **alias**, and it does so only for a **component carrier**. An alias record is
  bookkeeping, never a destruction, so for a carrier the branch's real consume must
  replace it. An ordinary `let y = x` records the same shape for an unrelated reason,
  and promoting it there would make a later *borrow* of `y` fail after one branch
  consumed `x`. Ordinary aliases keep their existing behavior; the promotion is
  carrier-only.

### Aliases

`let y = p` where `p` is a component records an alias, not a destruction: both names
reach the same node in lowered IR, exactly as for a non-component source. Taking two
such aliases is not fan-out. A *consuming* use resolves through the alias chain to the
component, so a component consumed twice through aliases is the same hard error as one
consumed twice directly, and the diagnostic names the binding the source actually
wrote.

An alias binds to the **binding generation** it was taken against, per
`spec/04-type-system.md` [04-LIN-1], and the link never re-resolves. The checker
implements this by minting a unique id per binding event (`BindingId` in
`linearity.rs`) and keying every alias link, consumption mark, and component mark on
ids; a name is resolved to its innermost live generation exactly once, at the use
site that records the fact. Two consequences the name-keyed implementation got wrong
(chelis#1209), both now structural:

- Re-binding the source's name after `y = a` leaves `y` pointing at the original
  `a`. A double consume of `y` lands on that generation's carrier and is reported
  against `y`; the fresh `a` is untouched and stays consumable.
- A later destructuring `let` that reuses the source's name marks its own new
  generation. An alias taken against the ordinary older generation never inherits
  the component restriction, so the fan-out it always had stays copyable.

The same identity dissolves the authored-carrier collision (chelis#1212): a
user-written `__chelis_tmp0` and a desugarer-minted carrier of the same spelling are
distinct generations, so the collision is ordinary shadowing and hides nothing. The
capture rule stays deliberately narrow under generation identity: per [04-LIN-2] a
closure capture consumes the binding it names, ordinary aliases are distinct for
capture, and only a component (or an alias of one) forwards to its carrier. Because
that carve-out keeps two captures on one alias chain order-sensitive, the sorted
`free_vars` order in `check_fn` remains semantically necessary, not cosmetic.

## Tensor-Carrying ADTs

Implicit linearity extends to ADTs whose definitions transitively carry a
tensor. The reference rule lives in `spec/04-type-system.md` §8.4; this
section covers how it plugs into copy/drop insertion.

An ADT `T` is **tensor-carrying** iff some variant of `T` has a field whose
type contains a tensor, considered transitively through tuples, ADT
instantiations, and other tensor-carrying ADTs. Recursive and mutually
recursive ADTs are resolved by least-fixed-point.

Owned `T` values are linear. The same auto-borrow, auto-copy, and auto-drop
rules used for `tensor[...]` apply to `T`:

- An unconsumed local `T` receives an inserted end-of-scope `Drop`.
- A single owned `T` used in two consuming positions is fan-out: the
  earlier site receives an inserted `Copy`, the final site takes the
  original.
- Passing owned `T` where `&T` is expected auto-borrows.
- `match` on owned `T` consumes the scrutinee per §8.3.

`t-fn` is intentionally excluded from the carrier relation: a field whose
type is a function that happens to take or return a tensor does not make
the enclosing ADT tensor-carrying. Closures that capture tensors are
governed by the §8.3 capture rule and do not need ADT carrier participation
to be correctly handled.

Cross-package transitivity is preserved. When the linearity checker runs
against a library context plus new code, both halves of the ADT declaration
set are resolved in one fixed-point pass, so a new-code wrapper around a
library tensor-carrying ADT is recognized as carrying without re-walking
the library's bodies.

The carrier-set rule presumes type-name uniqueness within the program
(`spec/04-type-system.md` §8.5). Duplicate `deftype` / `typealias` names
are rejected as `DuplicateDefinition` before linearity runs.

## Preserved Hard Errors

Implicit handling replaces missing-local-drop and ordinary consume-fan-out diagnostics.
The following remain errors:

- invalid borrow syntax, including borrowing non-variables where a direct variable
  borrow is required
- storing, returning, or capturing borrows where borrow escape is disallowed
- passing `&T` to an owned `T` parameter without an explicit or inserted copy
- impossible branch or loop ownership states where a single terminal path cannot be
  established
- recursive or cyclic consume cases outside the v1 signature-inference scope

## Signature Inference

Inference applies only to unannotated parameters of non-recursive functions. A
parameter is inferred read-only when no operation in the function body consumes it.
Primitive calls use their declared signatures. User-function calls use written
signatures, inferred signatures already available in the module, or imported inferred
signatures from export metadata.

Type checking builds a call graph and marks direct and mutual recursion cycle members.
Inference is disabled for parameters of recursive-cycle members. Written `&T` or `T`
annotations are authoritative. Unannotated parameters in recursive cycles keep the
existing owned/default semantics.

Module export metadata records inferred signatures as a per-function map from
parameter index to inferred type. Importing modules consume this metadata before their
own inference pass. `chelis check --show-inferred` prints inferred signatures alongside
written signatures; default `check` output is unchanged.

## Fitness And Cost

Unified copy cost is computed only from lowered `RiscOp::Copy` nodes. If an older path
counts source-level `copy()` calls, Workstream C must remove or bypass it so explicit
and inserted copies are counted once.

`chelis cost <file>` reports copy cost after lowering. Human output is the default.
`--json` emits this stable shape:

```json
{
  "file": "path/to/file.ch",
  "functions": [
    {
      "name": "function_name",
      "copy_count": 1,
      "bytes_copied": 1024
    }
  ],
  "total_copy_count": 1,
  "total_bytes_copied": 1024
}
```

For tensors with symbolic dimensions, `bytes_copied` is omitted for that function and
from `total_bytes_copied`; the human output prints the symbolic byte formula when
available.

## Migration And Linting

Existing source with explicit `copy()` and `drop()` remains valid.

Executable examples should move toward the implicit style. Fixture baseline updates
must record before/after fitness data in machine-readable CSV or JSON. Expected deltas
from newly visible copy costs are documented; unexpected IR or fitness deltas block the
integration merge. The current executable-example baseline lives in
`crates/chelis-cli/tests/fixtures/copy_drop_fixture_fitness_baseline.json`.

## IR Consumer Audit

Adding `RiscOp::Copy` and `RiscOp::Drop` changes the IR wire shape. The implementation
must audit evaluator, AD, optimizer/fusion, C/HIP/Metal backends, compiler API
serialized DAGs and caches, CLI tooling, and tests. Any serialized cache or public wire
consumer affected by the enum expansion must receive the same additive schema/version
treatment used by prior IR schema changes: old caches are invalidated or regenerated,
and current clients see explicit `Copy` and `Drop` nodes.
