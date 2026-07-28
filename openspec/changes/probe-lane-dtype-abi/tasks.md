# Tasks

Cheapest item in this family and the only one covering what compile-time work cannot.
Metal's probe is the reference implementation; this is largely "do that, for each lane".

**Scope note:** the HIP probe lands with the `CRuntime-BoolStorage-F1` HIP fix in
chelis#894, because a fix without its oracle is how the overflow got there. This change
covers generalising the pattern and the remaining lanes.

## Phase 0 — Inventory

- [ ] **0.1** List every lane boundary that moves tensor bytes: Metal host/device, HIP
      host/device, the C backend's emitted-host to runtime path, `chelis-python`'s buffer
      surface. Not all are host/device; a boundary is anywhere two components hold the
      same buffer under separately declared types.
- [ ] **0.2** For each, record what it currently checks, if anything. Metal has
      `dtype_abi_width_parity.rs`; HIP has nothing; the C backend has
      `host_sizeof_expr`-shaped checks scattered across test files.
- [ ] **0.3** Decide whether the probe is shared code or duplicated per lane. Shared is
      DRY-er; duplicated keeps each lane's mapping legible and avoids a shared helper that
      must know every lane's spelling conventions. Metal's version is ~120 lines, so
      duplication is affordable.

## Phase 1 — Generalise the Metal probe

- [ ] **1.1** Extract the shape: emitted-type → `Repr`, compare against
      `RuntimeDType::repr`, fail with both encodings named.
- [ ] **1.2** Keep the mapping driven by the emitter's own function rather than by a fixed
      table, so a change to the emitter moves the probe's input and the *comparison* is
      what fails. Metal's `metal_device_repr` maps `msl_type(prec)`; this is the property
      that makes the probe track the code instead of a snapshot of it.
- [ ] **1.3** Preserve the direction-naming failure message. Metal's says not to close a
      mismatch by widening Metal to f32; without that, the cheapest fix is the wrong one.

## Phase 2 — Cover the remaining lanes

- [ ] **2.1** HIP. **Lands in chelis#894** with the bool fix, since that branch is what
      breaks it. Must cover `dtype_c_type`, `elem_kind`, and the device allocation width,
      because the overflow came from those disagreeing with each other rather than from
      any one of them being independently wrong.
- [ ] **2.2** The C backend's emitted-host path.
- [ ] **2.3** `chelis-python`'s buffer surface, if 0.1 finds it qualifies.

## Phase 3 — Keep new lanes honest

- [ ] **3.1** Decide the mechanism that makes a new lane inherit this: a checklist in the
      backend docs, a conformance entry, or a test that enumerates backend crates and
      asserts each has a probe. The third is mechanical and catches the actual failure —
      a lane existing without one.
- [ ] **3.2** Add the rule to the owning spec, phrased as the finding rather than as an
      instruction: the lane with a probe is the lane whose defect was found.

## Phase 4 — Record

- [ ] **4.1** `docs/gap_synthesis.md`: this is the executable half of the cross-lane
      class. `derive-lane-element-widths` and `bind-lane-element-types` are the
      compile-time halves.
- [ ] **4.2** Record the width-versus-representation evidence in one place rather than in
      each probe: the width check caught bool because `1 != 4` and was blind to int32
      because `4 == 4`, in the same file, at the same time.

## Deliberately not in this change

- **Executing GPU kernels on the per-PR gate.** These probes compare emitted text and
  metadata. Running device work stays a documented manual gate with its own command and
  success condition.
- **Checking kernel bodies.** A kernel writing a wrong-width value into a correctly typed
  buffer is not caught by comparing element encodings. That needs execution, which is the
  manual gate, or phantom-typed emission, which is a much larger change.
