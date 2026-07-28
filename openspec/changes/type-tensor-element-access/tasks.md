# Tasks

Ordered. Phase 0 settles the questions that change the shape of everything after it.

**Prerequisite:** `seal-tensor-data-pointer` is landed. Typing an accessor that callers
can bypass with a direct cast buys nothing, and doing this first means the seal has to be
redone against the new signatures.

## Phase 0 — Measure before designing

- [ ] **0.1** Count internal access sites and classify each as element-generic (the body
      works for any `T` given bounds) or dtype-dispatched (the body differs per dtype).
      The ratio decides whether `dispatch` is called a handful of times or dozens, and
      whether the visitor needs a return-type parameter or can be monomorphic.
- [ ] **0.2** List every operation's supported-dtype subset. `cmplt` rejects
      bf16/f16/i8/i16; `cumsum` and `trace` reject bool. Decide whether the guard lives
      in the visitor or stays per-op.
- [ ] **0.3** Settle the `u16` / `u8` / `u32` element-type gap **jointly with
      `seal-tensor-data-pointer` Phase 0**, not twice. `bf16`/`f16` cast to `*mut u16`
      and have no `TensorElement` impl for the same reason bool had none: the logical
      type is not the storage type. `Bool8` is the precedent.
- [ ] **0.4** Decide whether `Tensor<T>` owns or borrows. The runtime frees tensors
      explicitly through `chelis_free`; a handle that implies ownership will fight that.
- [ ] **0.5** Confirm `#[repr(transparent)]` over `*mut chelis_tensor` gives identical
      size and alignment, and that passing it through `extern "C"` is not required
      anywhere (it should not be — conversion happens inside the boundary).

## Phase 1 — Build the types alongside the existing ones

- [ ] **1.1** Add `Tensor<T: TensorElement>`, `#[repr(transparent)]`, with the phantom
      element type.
- [ ] **1.2** Checked constructor returning `Result<Self, DtypeMismatch>`. The check runs
      in release builds; it is not a `debug_assert`.
- [ ] **1.3** Element pointer accessor with **no** dtype comparison, and a doc comment
      stating why the absence is sound rather than an oversight.
- [ ] **1.4** `DtypeVisitor` and the single `dispatch` entry point, exhaustive over
      `RuntimeDType`.
- [ ] **1.5** Negative test: a handle of one element type does not coerce to another.
      A `compile_fail` doctest is the natural form — **and it runs nowhere today**.
      `chelis-runtime` has no doctest stage, so either widen the gate's
      `cargo test -p chelis-types --doc` stage in this change set or the oracle is
      decorative. See `AGENTS.md`, "Doctests only run where something invokes them."
- [ ] **1.6** Negative test: constructing a handle against a mismatched tag fails, and
      the error names both dtypes.
- [ ] **1.7** Negative test: a dtype added to the vocabulary without extending `dispatch`
      fails the build.
- [ ] **1.8** Assert the handle is pointer-sized and pointer-aligned.

## Phase 2 — Migrate, in reviewable batches

Safe to split, unlike the bool migration. Every accessor introduced is a typed spelling
of a cast that already exists, so no byte moves and a half-migrated tree is inconsistent
rather than memory-unsafe. Batches should still be per-operation-family so a reviewer can
check one dtype story at a time.

- [ ] **2.1** Convert the element-generic sites first — they need bounds, not new logic.
- [ ] **2.2** Convert the dtype-dispatched sites, replacing each hand-written
      `match dtype` with a `dispatch` call. Expect this to surface arms that differ for
      no reason; note them rather than silently unifying behaviour.
- [ ] **2.3** Convert the raw-byte sites (`where`'s `copy_nonoverlapping`, reshape,
      cast). These are legitimately untyped and need a distinct accessor, not a typed
      one — the requirement is that asking for bytes is deliberate and visible.
- [ ] **2.4** Convert each `extern "C"` entry point to dispatch on its first line.
- [ ] **2.5** After each batch, report the remaining unconverted count. "Estimated" is
      not acceptable here; the bool migration's near-miss came from believing a site
      list that was compiled by reading rather than by counting.

## Phase 3 — Remove the untyped path

- [ ] **3.1** Delete `data_as_f32` / `data_as_f32_const`. They exist only for f32 sites
      predating the trait and buy nothing a typed accessor does not.
- [ ] **3.2** Delete `data_ptr_unchecked`, whose `debug_assert` is a type's work done at
      runtime in debug builds only.
- [ ] **3.3** Land 3.1 and 3.2 in a commit that changes **nothing else**, so that its
      compiling is the evidence no site was missed. A removal commit that also fixes
      fallout destroys that evidence.

## Phase 4 — Record

- [ ] **4.1** Update `docs/gap_synthesis.md`: this closes the mechanism behind
      `CRuntime-I32Storage-F1` and `CRuntime-BoolStorage-F1` rather than their instances,
      which were fixed by hand.
- [ ] **4.2** Record what remains reachable. Sealing plus typing does not touch the C
      backend, where `elem_type` returning the wrong string caused the single
      highest-impact bug in the bool migration. Say so plainly rather than implying the
      class is closed.
- [ ] **4.3** Add the `CExpr<T>` codegen argument to the backlog with that evidence
      attached, as a separate proposal.

## Deliberately not in this change

- **The C backend.** Emitted C is text; no Rust type prevents returning `"float"` for
  bool. Phantom-typed `CExpr<T>` is the analogous move and belongs in its own proposal
  with its own evidence.
- **Removing the FFI boundary.** `#[repr(C)]` structs crossing a C ABI cannot be typed by
  generics. The goal is one untyped line per exported function, not zero.
- **Eliminating runtime checks.** The check moves to the constructor. A tensor arriving
  from C is still validated once, and must be.
