# Tasks

## Phase 0 — Settled in `design.md`; confirm against the code

The four open questions are argued and closed in [`design.md`](design.md). Three of them
constrain where the code can live, so they were not safe to leave to implementation.
What remains is confirming the premises still hold.

- [x] **0.2** **Markers move to `chelis-vocab`.** No backend depends on `chelis-runtime`,
      where the element types live today — backends emit text that calls the runtime, they
      do not link it. A trait over runtime types is therefore unreachable from the lanes
      without inverting that layering.
- [x] **0.3** **Vocab owns the constraint; each lane owns its spelling.** There is no
      single C name — MSL `bool`, C99 `unsigned char`, HIP C++ — and a vocab-owned name
      would make the dependency bottom enumerate backends.
- [x] **0.4** **`Bf16Bits(u16)` / `F16Bits(u16)`, not `half`'s types.** Not for soundness:
      unlike `bool`, `half::bf16` has no validity invariant, so an impl over it would be
      sound. For dependencies — vocab has none, and
      `crate_purity.rs::manifest_declares_no_dependencies` fails if that changes.
      `chelis-backend-hip` also has no `half` dependency today.
- [ ] **0.1** Enumerate every element-spelling site: `CEmitter::elem_type`,
      `DtypeArm::elem_t`, `HipEmitter::dtype_c_type`, `HipEmitter::elem_kind`,
      `msl_type`, `host_sizeof_expr`. Record which return a *type name*, which return a
      *sizeof expression*, and which return a *kernel family* — three different things,
      and only the first two are width-constrained.
- [ ] **0.5** Confirm the dependency facts above still hold at implementation time. The
      whole design rests on backends not depending on `chelis-runtime`; if that changed,
      the marker relocation may be unnecessary.
- [ ] **0.6** Confirm relocating `Bool8` to `chelis-vocab` does not break its `no_std`
      purity or its zero-dependency ratchet. It is plain data with a `const fn`, so it
      should not, but `crate_purity.rs` is the arbiter.

## Phase 1 — Build the binding

- [ ] **1.1** Relocate the element markers to `chelis-vocab` and add `Bf16Bits` /
      `F16Bits`. `chelis-runtime` implements `TensorElement` over them rather than over
      its own types.
- [ ] **1.2** Define the per-lane trait shape and the `lane_element!` macro that emits an
      impl **and** its width assertion together. The macro is the mechanism: if a lane can
      hand-write an impl, it can omit the assertion, and agreement goes back to being
      maintained by hand — which is what produced the HIP overflow.
- [ ] **1.3** The assertion must be a `const` block, not a `debug_assert`. A debug-only
      check is what `data_ptr_unchecked` used, and it is a type's work done at runtime.
- [ ] **1.4** Implement for every currently-emittable element type, per lane.
- [ ] **1.4** Negative test: binding a type to a dtype of a different width fails to
      compile. **This is a `compile_fail` doctest and it runs nowhere today** — see
      `AGENTS.md`, "Doctests only run where something invokes them." Widen the gate's
      doctest stage to the owning crate in this change set, or the oracle is decorative.
- [ ] **1.5** Negative test: a dtype with no binding produces a loud failure naming the
      dtype and the lane.

## Phase 2 — Migrate the emitters

- [ ] **2.1** `chelis-backend-c`: `elem_type` and `DtypeArm::elem_t`.
- [ ] **2.2** `chelis-backend-hip`: `dtype_c_type`. Note this is the site that overflowed;
      the fix itself lands with `CRuntime-BoolStorage-F1` in chelis#894, and this change
      is what makes the *class* unwriteable rather than that instance fixed.
- [ ] **2.3** `chelis-backend-metal`: `msl_type`, and reconcile `host_sizeof_expr`, which
      emits a C expression rather than a type name.
- [ ] **2.4** `elem_kind` (HIP) selects a *kernel family*, not a type. Decide whether it
      is in scope: it mapped `Prim::Bool` to the f32 family, which is how bool kernels
      came to write four-byte values. It is width-relevant but not a spelling.

## Phase 3 — Record

- [ ] **3.1** Update `docs/gap_synthesis.md`. This is the compile-time half of the
      cross-lane class; the parity probes are the executable half.
- [ ] **3.2** State what remains uncovered: a kernel *body* that writes a wrong-width
      value into a correctly typed buffer still compiles. Name `probe-lane-dtype-abi` as
      the mechanism for that, and do not imply the class is closed.
- [ ] **3.3** Amend `type-tensor-element-access`'s non-goals, which currently exclude the
      backends on the grounds that emitted C is text. That is true of kernel bodies and
      false of element spellings, and HIP is the evidence.
