# Tasks

Mechanical and low-risk. Deleting a table that agrees with the source changes nothing;
deleting one that disagrees fixes a bug. Either way the derived value is correct by
definition.

**Scope note:** the two HIP tables are part of the `CRuntime-BoolStorage-F1` fix and land
with it in chelis#894, because that branch is what makes them wrong. This change covers
the rest.

## Phase 0 — Find them all

- [ ] **0.1** Enumerate every mapping from a dtype or precision to a byte count outside
      `chelis-vocab`. Grep for `=> 1,` / `=> 2,` / `=> 4,` / `=> 8,` near a `Prim` or
      `RuntimeDType` match, and for `sizeof(` in emitted-C helpers.
- [ ] **0.2** For each, record whether it currently agrees with `byte_width`. A
      disagreement is a live bug and should be reported before it is silently corrected
      by the deletion.
- [ ] **0.3** Note which are in test harnesses. A harness that restates widths can pass
      while the code it tests is wrong, which is the `dtype_op_matrix` failure mode from
      `CRuntime-I32Storage-F1` — the fixture helper shared the code's misunderstanding, so
      77 fixtures could not detect it.

## Phase 1 — Delete

- [ ] **1.1** `chelis-backend-metal`: derive `metal_elem_size` and reconcile
      `host_sizeof_expr` (it emits a C *expression*, so it needs the name binding from
      `bind-lane-element-types`, not just a width — record which half is deferred).
- [ ] **1.2** `chelis-backend-c/tests/exec_compile.rs`: `c_sizeof_width` maps a C
      expression string back to a width. Keep the mapping if it is the only way to check
      emitted text, but assert it against `byte_width` rather than restating it.
- [ ] **1.3** `chelis-backend-hip/tests/bf16_f16_matmul.rs`: `dtype_bytes`.
- [ ] **1.4** Any site found in 0.1 not listed above.

## Phase 2 — Keep it deleted

- [ ] **2.1** Decide the guard: a lint, a grep-based test, or review discipline. A test
      that greps for the shape is honest about being heuristic and is still better than
      nothing; a lint is stronger and costs a nightly toolchain.
- [ ] **2.2** Whatever is chosen, it must name the reason at the failure site, not just
      report a match. "A duplicated width table is invisible exactly when the widths
      agree and the encodings do not" is the sentence that makes the rule stick.

## Phase 3 — Record

- [ ] **3.1** `docs/gap_synthesis.md`: this closes the *width* half of the cross-lane
      class. The encoding half stays open and belongs to `probe-lane-dtype-abi`.
- [ ] **3.2** State plainly that this does not prevent a lane naming a C type whose width
      disagrees with the dtype — that is `bind-lane-element-types`, and it is the half
      that actually caused the HIP overflow.
