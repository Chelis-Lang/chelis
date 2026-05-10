# Implicit Linearity

**Status:** Phase 0 contract for the `copy-drop` work.
**Owning specs:** `spec/03-deep-syntax.md`, `spec/04-type-system.md`,
`spec/05-risc-primitives.md`, and `spec/design/borrow_typed_primitives.md`.

## Summary

Chelis keeps explicit `copy()` and `drop()` source forms for compatibility, but the
compiler now owns routine linearity ceremony. It inserts drops at scope exits and
copies at consuming fan-out sites. Drop timing is end-of-scope, not last-use. Last-use
optimization is a future tightening.

Lowered IR makes the result explicit. `RiscOp::Copy` and `RiscOp::Drop` are real IR
nodes, and every lowered linear value has exactly one terminal path: a consuming use or
a `Drop`. Slot planners treat `Drop` as the authoritative live-range close.

## Drop Insertion

An owned linear value that reaches the end of its containing scope without a consuming
use receives an inserted `Drop`. Multi-branch flow is branch-local: if a value is
consumed on one branch and not the other, the unconsumed branch receives a `Drop`.
`if` and `match` branches are analyzed independently.

Loops introduce an iteration scope. Values defined inside a loop are consumed or
dropped by the end of that iteration. Values defined outside a loop and consumed inside
the loop require the existing loop-aware ownership rules; cases where the compiler
cannot prove a single terminal path remain hard errors.

`Drop` is a liveness marker, not a computation. Evaluators and backends emit no runtime
operation for it. AD treats it as a gradient sink for the dropped value. Backends and
slot planners must use `Drop` to close the dropped value's live range; when `Drop` is
present, independent liveness may refine allocation details but must not extend the
value past the drop.

## Copy Insertion

The compiler inserts `Copy` for source-level consuming fan-out: a value used in more
than one non-borrow consuming position. Copies are inserted for all consume sites before
the final consume site; the final consume site takes the original.

Borrows do not count as fan-out. Multiple `&T` uses share the same source. A value
passed once to a consuming function after any number of borrows is not fan-out and does
not receive a copy.

Explicit source `copy()` lowers to the same `RiscOp::Copy` used for inserted copies.
Cost and training signals intentionally do not distinguish explicit and inserted
copies.

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

Existing source with explicit `copy()` and `drop()` remains valid. The
`redundant-linearity-call` lint flags removable explicit calls as warnings for
user-facing `chelis lint` and `chelis check`. The lint is advisory in this release and
does not fail style gates or fixture/test compilation paths unless explicitly invoked.

Executable examples should move toward the implicit style. Fixture baseline updates
must record before/after fitness data in machine-readable CSV or JSON. Expected deltas
from newly visible copy costs are documented; unexpected IR or fitness deltas block the
integration merge. The current executable-example baseline lives in
`docs/copy_drop_fixture_fitness_baseline.json`.

## IR Consumer Audit

Adding `RiscOp::Copy` and `RiscOp::Drop` changes the IR wire shape. The implementation
must audit evaluator, AD, optimizer/fusion, C/HIP/Metal backends, compiler API
serialized DAGs and caches, CLI tooling, and tests. Any serialized cache or public wire
consumer affected by the enum expansion must receive the same additive schema/version
treatment used by prior IR schema changes: old caches are invalidated or regenerated,
and current clients see explicit `Copy` and `Drop` nodes.
