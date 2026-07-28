# Tasks

**Atomic, like the runtime migration and for the same reason.** A one-byte allocation with
four-byte writers is a heap overflow, not an inconsistency, so no intermediate state is
shippable. The allocation side has *already* moved — it derives from the vocabulary — so
this branch starts from the broken state rather than reaching it partway through.

## Phase 0 — Write the probe first

Deliberately before the fix. The probe is what tells us the site list is complete, and a
site list compiled by reading rather than by testing is how the overflow got here.

- [ ] **0.1** Add `crates/chelis-backend-hip/tests/dtype_abi_parity.rs`, modelled on
      Metal's `dtype_abi_width_parity.rs`. Map HIP's emitted device type to a `Repr` via
      `dtype_c_type` — driven by the emitter's own function, not a fixed table, so a
      change to the emitter moves the probe's input and the comparison is what fails.
- [ ] **0.2** Confirm it **fails** against the current tree for bool, and passes for every
      other dtype. A probe that passes before the fix proves nothing.
- [ ] **0.3** Cover the three decisions that must agree with each other, not just with the
      runtime: `dtype_c_type`, `elem_kind`, and the device allocation width. The overflow
      came from those disagreeing among themselves.
- [ ] **0.4** Failure message names the direction: the runtime's native byte is the agreed
      encoding, and widening HIP back to four bytes would satisfy an equality check while
      reverting `CRuntime-BoolStorage-F1`.

## Phase 1 — Migrate, in one commit

### 1.0 — Make the spelling a compile-time property, before fixing the arms

So the fix is *checked* rather than asserted. The pattern is already proven against this
tree: planting `Bool => (f32, "float", RuntimeDType::Bool)` produced
``error[E0080]: HIP element type `float` for Bool does not have that dtype's declared width``.

- [ ] **1.0a** Replace `dtype_c_type`'s hand-written match with a macro-generated table:
      `Prim => (witness_type, c_spelling, RuntimeDType)`. The witness is a Rust primitive
      whose size models the C type — `u8` for `unsigned char`, `f32` for `float`, `i32`
      for `int32_t`. It does **not** need to be `Bool8`, so nothing moves between crates.
- [ ] **1.0b** The macro emits, per arm,
      `const _: () = assert!(size_of::<witness>() as u32 == dtype.byte_width(), "...")`.
      `RuntimeDType::byte_width` is already `const fn`. `Prim::runtime_dtype` is **not**
      const, so name the `RuntimeDType` in the table rather than deriving it.
- [ ] **1.0c** There must be no hand-written arm path. If a spelling can be added without
      its assertion, agreement is back in human hands — which is what produced this
      overflow.
- [ ] **1.0d** Negative test: a deliberately wrong binding fails the build. A
      `compile_fail` doctest is the natural form and **runs nowhere today** (`AGENTS.md`:
      "Doctests only run where something invokes them"). Either wire a doctest stage for
      this crate in this change set, or record that the oracle is manual and give the exact
      command that demonstrates it.

### 1.1 onward — the arms

- [ ] **1.1** `dtype_c_type`: bool binds to `(u8, "unsigned char")`. With 1.0 in place this
      is a table edit the compiler checks, not a claim.
- [ ] **1.2** `elem_kind`: remove the `Prim::Bool` arm. Bool is not a float family; it
      routes through the typed templates as i8/i16/i32/i64 already do. Confirm every
      caller either handles the absence or is unreachable for bool.
- [ ] **1.3** `ElemKind::one_lit_bool` / `zero_lit_bool`: emit `1` / `0`. If (1.2) makes
      these unreachable for bool outputs, delete them rather than leaving a
      correct-but-dead float convention for someone to reach for.
- [ ] **1.4** `emit_typed_scalar_local`: give bool its own arm; it currently shares the
      f32 bit-pattern reconstruction.
- [ ] **1.5** `bytes_per_element` in `emit.rs` and `memory.rs`: derive from
      `byte_width`. Both are part of this fix because this branch is what makes them wrong.
- [ ] **1.6** `dtype_bytes` in `tests/bf16_f16_matmul.rs`: derive. A harness that restates
      widths can pass while the code it tests is wrong — the `dtype_op_matrix` failure
      mode, where 77 fixtures could not detect a defect their helper shared.
- [ ] **1.7** `dtype_kernel_suffix` keeps `_bool`; the suffix names a dtype, not a width.
      Confirm the kernels that suffix selects are now the typed ones.

## Phase 2 — Prove it

- [ ] **2.1** The Phase 0 probe passes.
- [ ] **2.2** `cargo nextest run -p chelis-backend-hip` green.
- [ ] **2.3** Workspace green — this branch is stacked, so the base is already known green
      at `06c58f19`, and any new failure is this change's.
- [ ] **2.4** **Run the HIP manual gate.** This workstation has a reconciled ROCm stack, so
      the class of bug this change fixes is locally observable rather than theoretical.
      Use `scripts/hip_test.py` per `AGENTS.md`; a plain `cargo test --ignored` inherits
      only the `environment.d` defaults and segfaults hipBLAS-linked binaries at exit,
      which reads as a code regression and is purely environmental.
- [ ] **2.5** Record the gate command and its result in the phase docs. If a bool tensor
      turns out not to be reachable on the HIP path from surface syntax today, say so —
      that downgrades the defect from live to latent without making the emitter any less
      wrong.

## Phase 3 — Record

- [ ] **3.1** `docs/gap_synthesis.md`: `CRuntime-BoolStorage-F1`'s entry claims closure.
      Amend it to say the runtime closed first and the HIP lane followed, because the
      intervening state was a device-side overflow and that is the reusable lesson.
- [ ] **3.2** Note in the entry that the migration's atomicity was scoped to one crate when
      it needed to be scoped to every lane holding the dtype.
- [ ] **3.3** Cross-reference `probe-lane-dtype-abi`: HIP's probe lands here, the pattern
      generalises there.

## Deliberately not in this change

- **Generalising the binding to the C and Metal lanes.** `bind-lane-element-types` does
  that. Note its design assumes the element markers must relocate to `chelis-vocab`; the
  witness pattern proven here shows that is unnecessary, so that proposal should be
  re-derived and is likely much smaller than it currently claims.
- **The other four duplicate width tables.** Repo-wide deletion is
  `derive-lane-element-widths`. The two HIP ones are here because this branch breaks them.
- **Probes for other lanes.** `probe-lane-dtype-abi`.
