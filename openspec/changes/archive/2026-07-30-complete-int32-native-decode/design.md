## Context

The base commit is PR #964 at `5e81c966`. It maps `RuntimeDType::I32` to `Repr::TwosComplement32` and preserves four-byte native int32 storage.

Current writers already store native `i32` values. Seven runtime consumers still read those bytes through `data_as_f32` or `data_as_f32_const`.

The affected consumers are comparison, scatter-add, `where`, `cumsum`, `trace`, `clamp`, and `einsum`. The C host emitter also maps `DtypeArm::I32` to `float`.

PR #894 contains corrections for these sites. Its bool migration and development-environment changes belong to separate changes.

The base already corrects int32 tensor formatting. This change does not replace or duplicate that correction.

`spec/04-type-system.md` [04-NUM-8] owns exact int32 arithmetic width. `spec/design/loud_unsupported.md` section C6.2 owns the runtime dtype consumer inventory.

OpenSpec records the implementation plan. It does not replace either owning document.

## Goals / Non-Goals

**Goals:**

- The runtime decodes every affected int32 operand through an `i32` view.
- Generated C uses `int32_t` for `DtypeArm::I32` element access.
- Regression tests distinguish native int32 values from the same bytes read as `f32`.
- The `data_as_f32` boundary rejects int32 tensors in debug builds.
- The runtime dtype consumer documentation states the shipped storage and access rules.

**Non-Goals:**

- The change does not migrate bool storage or add `Repr::Bool8`.
- The change does not alter dtype IDs, widths, allocation, or public C signatures.
- The change does not implement the integer overflow trap contract from [04-NUM-3].
- The change does not add arithmetic-width vocabulary from chelis#899.
- The change does not seal `chelis_tensor.data` from chelis#893.
- The change does not modify HIP, Metal, the development environment, or tensor formatting.

## Decisions

### Use the physical element type at every corrected access

Each `RuntimeDType::I32` arm uses `i32::data_ptr_unchecked` or an existing generic `i32` loop. This approach aligns access with `Repr::TwosComplement32`.

The bool arms remain on the `f32` compatibility view. `Repr::BoolInBinary32` still describes that active representation.

A shared four-byte view was rejected. Equal width does not make IEEE binary32 and two's-complement int32 interchangeable.

### Reuse existing typed loops

Comparison, `cumsum`, `trace`, `clamp`, and `einsum` use their existing generic loops with `i32`. Scatter-add follows the existing native integer arm.

This choice removes special float bodies and adds no int32 algorithm. It also preserves each operation's current overflow behavior.

This change does not claim completion of [04-NUM-3]. The tests use in-range values and isolate representation errors.

### Give generated C an int32 element type

`DtypeArm::elem_t()` maps `DtypeArm::I32` to `int32_t`. The mapping remains separate from the bool and f32 mappings.

Generated int32 max and min operations use integer comparisons. They do not call `fmaxf` or `fminf`, which lose integer precision above 2²⁴.

A width-derived C type was rejected. Both int32 and f32 have four-byte storage but require different pointer types.

### Add a debug ratchet at the f32 compatibility boundary

`data_as_f32` and `data_as_f32_const` accept only `F32` and the current payload-encoded `Bool` representation in debug builds.

A negative test passes an int32 tensor to this boundary and requires an assertion failure. Positive tests keep valid f32 and bool access available.

The assertion is a development ratchet, not the memory-safety boundary. The public untyped pointer remains open under chelis#893.

### Use values that expose each old decode

The regression suite uses these evidence classes:

- Comparison uses a negative value because `-1i32` becomes a NaN through an f32 view.
- `where` uses `i32::MIN` because its bits become negative zero through an f32 view.
- Accumulation tests use in-range values outside the float denormal accident range.
- `clamp` uses signed values and bounds that fail under bit reinterpretation.
- `einsum` uses integer multiplication that underflows through the float reinterpretation.

A deliberate local mutation restores the old f32 view. The related regression must fail for the predicted reason.

### Update implementation contracts without changing language semantics

The change updates section C6.2 to name physical representations and representation-compatible access. It also corrects `CRuntime-I32Storage-F1` and stale code comments.

The numbered specifications already require exact int32 behavior. This change adds no new language rule.

### Use one acceptance oracle

The authoritative oracle is `cargo nextest run -p chelis-runtime -p chelis-backend-c` from the repository toolchain.

Local C tests use the repository `gcc` command, which can resolve to clang on macOS. Hosted Linux CI remains the final GNU C evidence.

## Risks / Trade-offs

- **Risk: The port copies stale PR #894 context.** The implementation ports behavior onto PR #964. It does not replay commits without review.
- **Risk: Bool access changes by accident.** Separate bool arms and positive bool tests preserve `BoolInBinary32` behavior.
- **Risk: A direct pointer cast bypasses the debug ratchet.** The consumer audit and chelis#893 track this residual exposure.
- **Risk: Generic integer loops expose overflow debt.** In-range fixtures isolate this change, and [04-NUM-3] stays open.
- **Risk: Local and hosted C compilers differ.** The local oracle catches source and execution errors, and hosted CI supplies GNU C evidence.

## Migration Plan

1. Add the failing runtime and emitter regression tests.
2. Demonstrate each failure against the old f32-view behavior.
3. Port the seven runtime corrections from PR #894 onto PR #964.
4. Change the generated C int32 element type to `int32_t`.
5. Add the f32-boundary debug ratchet and its positive and negative tests.
6. Update section C6.2, `CRuntime-I32Storage-F1`, and stale code comments.
7. Run the authoritative oracle and the repository OpenSpec validation command.

If a regression appears, revert the runtime, emitter, test, and documentation edits together.

No stored-data migration or ABI rollback step is necessary.

## Open Questions

None.
