# Tasks: seal-tensor-data-pointer

The seal is the only irreversible step and it is last. Everything before it is a typed
respelling of an existing cast, so a half-migrated tree is correct-but-inconsistent
rather than wrong. That property is what makes this migration safe where the bool one is
not; do not trade it away for a shorter sequence.

## Phase 0 — Decide the gap, before any call site moves

- [ ] **0.1** Characterise the single `*const u32` site. If it is an f32 bit-pattern read it is a latent instance of the int32 defect class and is reported as a finding, not migrated as-is.
- [ ] **0.2** Decide `bf16` / `f16`. They cast to `*mut u16` today and have no `TensorElement` impl, because the logical type is a 16-bit float and `u16` is only its storage — structurally the same as bool-as-f32. Either introduce `Bf16Bits` / `F16Bits` on the `Bool8` precedent (`#[repr(transparent)]`, every bit pattern an inhabitant, explicit narrowing) or route them through raw bytes. Record which and why.
- [ ] **0.3** Confirm the `*const u8` sites are genuinely untyped — memcpy, allocation, release — and not element access wearing a byte disguise.
- [ ] **0.4** Re-count all access sites at the current tip. The figures in the proposal (4 external, 63 internal) were taken at `f0290cde` and will drift.
- [ ] **0.5** Confirm `#[repr(C)]` layout is unaffected by field privacy on this specific struct: capture `size_of`, `align_of`, and every field offset as a baseline to compare against after the seal.

## Phase 1 — Build the accessor surface

- [ ] **1.1** Create the module owning `chelis_tensor`. Leave `data` `pub` for now; this phase adds accessors without restricting anything.
- [ ] **1.2** Add the raw-bytes accessor. Make it deliberately awkward: `unsafe`, named for what it is, documented as unsuitable for element access. If it is pleasant it will be used to rebuild the hole this change closes.
- [ ] **1.3** Add the pointer-hygiene accessors — null test and the release path — which never dereference and should not require element access.
- [ ] **1.4** Add whatever Phase 0.2 decided for `bf16` / `f16`.
- [ ] **1.5** Test the accessor surface, including that raw bytes cannot be used to reconstruct typed access without the caller saying so explicitly.
- [ ] **1.6** Negative test: the raw-bytes accessor's documentation states the element-access exclusion, and a doc check enforces that the statement is present.

## Phase 2 — Migrate, in reviewable batches

Each batch is independently revertable and changes no behavior.

- [ ] **2.1** The ~8 `*const u8` byte sites → the raw-bytes accessor.
- [ ] **2.2** The null checks, struct construction, and release path → pointer hygiene.
- [ ] **2.3** The typed casts for the six dtypes that already have `TensorElement` impls (`f32`, `f64`, `i8`, `i16`, `i32`, `i64`) → `data_ptr` / `data_ptr_unchecked`.
- [ ] **2.4** The `u16` sites, per the Phase 0.2 decision.
- [ ] **2.5** `chelis-python`'s four sites (lines 248, 249, 260, 261 at `f0290cde`) → pointer hygiene.
- [ ] **2.6** After each batch: workspace clippy `-D warnings`, `cargo fmt --check`, and the runtime plus e2e suites.
- [ ] **2.7** Any site whose behavior changes under migration is a **finding**. Report it, fix it separately with its own regression test, and do not absorb it into a batch.

## Phase 3 — Seal

- [ ] **3.1** Confirm the remaining out-of-module access count is zero.
- [ ] **3.2** Seal the field, in a commit that changes **nothing else**. Its successful compilation is the evidence that no site was missed; a commit that also fixes fallout destroys that evidence.
- [ ] **3.3** Compare `size_of`, `align_of`, and field offsets against the Phase 0.5 baseline. They must be identical.
- [ ] **3.4** Compare the checked-in and generated C headers against their pre-change bytes. They must be identical.
- [ ] **3.5** Negative test: a direct cast of the storage pointer from outside the module fails to compile. A `compile_fail` doctest is the natural form — note that doctests run in only two places in this repo, so site it where something actually invokes it.
- [ ] **3.6** Confirm foreign access still works: the C-backend and e2e suites exercise compiled C reading the field through its own declaration.

## Phase 4 — Record

- [ ] **4.1** Update `docs/gap_synthesis.md`: this closes the *mechanism* behind `CRuntime-F32Coupling`, `CRuntime-I32Storage-F1`, and the eight-site int32 instance, which is distinct from closing any individual entry.
- [ ] **4.2** Record what is still not fixed: divergent match arms (the `dispatch_dtype!` macro, deliberately out of scope), and fixture conventions that encode the defect.
- [ ] **4.3** State the honest scope of the guarantee — direct casts are now unwritable outside one module; inside it they remain possible by construction, which is why that module should stay small.
- [ ] **4.4** Name the acceptance oracle: the seal commit compiling, plus the layout and header equality checks.

## Deferred, deliberately

- **The bool storage migration** (`CRuntime-BoolStorage-F1`). Should follow this change, because sealing turns its missed sites into compile errors rather than silent over-reads. `Bool8` is already staged.
- **`dispatch_dtype!`.** Complementary — privacy forces an accessor, but not the *right* accessor. Kept separate so neither change waits on the other.
- **Barnacle's `boundary_safety_restricted_field_access`.** The fallback if module privacy proves infeasible; costs a nightly Dylint toolchain.
- **`pub(crate)` as a cheaper substitute.** Rejected on evidence: all eight bugs were runtime-internal, so crate-level privacy prevents none of them.
