# Tasks

## Phase 0 — Decide what a binding is

- [ ] **0.1** Enumerate every element-spelling site: `CEmitter::elem_type`,
      `DtypeArm::elem_t`, `HipEmitter::dtype_c_type`, `HipEmitter::elem_kind`,
      `msl_type`, `host_sizeof_expr`. Record which return a *type name*, which return a
      *sizeof expression*, and which return a *kernel family* — they are three different
      things and only the first two are width-constrained.
- [ ] **0.2** Decide whether the binding lives in `chelis-vocab` or in a shared backend
      crate. `chelis-vocab` is `no_std` with zero dependencies and is the dependency
      bottom; a C type name is arguably vocabulary, but a *lane's* type name is arguably
      not. Argue it rather than defaulting.
- [ ] **0.3** Decide how lanes that disagree legitimately are expressed. Metal emits MSL,
      C emits C99, HIP emits HIP C++. `bool` is one byte in all three, but the names
      differ (`bool` vs `unsigned char`). Either the binding is per-lane, or it carries a
      per-lane name — the second is smaller but couples the lanes.
- [ ] **0.4** Settle whether `bf16`/`f16` get element newtypes here. They cast to
      `*mut u16` with no element type, for the same reason bool had none: the logical type
      is not the storage type. `Bool8` is the precedent, and this is the third change to
      need this decision — settle it once, with `seal-tensor-data-pointer` Phase 0 and
      `type-tensor-element-access` Phase 0.

## Phase 1 — Build the binding

- [ ] **1.1** Define the trait with `DTYPE` and the lane spelling(s).
- [ ] **1.2** Add the `const` size assertion. It must be a `const` block, not a
      `debug_assert` — a debug-only check is what `data_ptr_unchecked` used and it is a
      type's work done at runtime.
- [ ] **1.3** Implement for every currently-emittable element type.
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
