# Proposal: seal-tensor-data-pointer

## Why

`chelis_tensor.data` is a `pub` untyped `*mut u8`. Any code in reach can cast it to any
pointer type, and nothing checks that the type matches the tensor's `dtype` tag. That is
the mechanism behind the largest defect class this runtime has had.

The most recent instance, fixed on `harden-vocabulary-kernel` (`c9b99cae`, `521b4323`,
`975e333b`): **eight accessors plus the C backend's codegen decoded native int32 storage
through an f32 view.** `cmplt`, `where`, scatter add-mode, `cumsum`, `trace`, `clamp`,
`einsum`, and `tensor_to_string`. Each failed differently — wrong only for negatives,
only at `i32::MIN`, only outside the denormal range, or across the whole domain — so none
looked broken in isolation.

Its predecessor, `CRuntime-F32Coupling`, was a four-PR workstream that locked 110
fixtures. It did not prevent this one.

### Why the existing guards are insufficient

Three mechanisms already exist, and the int32 defect went through all of them:

- **`TensorElement::data_ptr`** checks the tag and returns `Err(DtypeMismatch)`. It is
  *optional*; the wrong arms simply did not call it.
- **`data_as_f32`'s dtype assert** (added this week) gates that helper. A direct
  `(*t).data as *const f32` bypasses it entirely, which five bool sites were doing until
  `f0290cde` funnelled them.
- **`dtype_op_matrix.rs`'s 77 fixtures** caught none of it, because its helper
  `alloc_vec_with_values` wrote I32 as `value as f32` — the same misunderstanding the code
  had. A test that shares the code's wrong assumption cannot detect it.

Every one of those is a convention. The field being `pub` is what makes them all
optional.

### The measurement that decides the shape of this change

Counted at `f0290cde`:

| scope | sites | character |
| --- | --- | --- |
| outside `chelis-runtime` | **4** | all `chelis-python`: `is_null` checks and `free()` — memory management, not typed reads |
| inside `chelis-runtime/src/lib.rs` | **63** | ~33 typed casts, 8 raw-byte casts, plus null checks and struct construction |

**All eight bugs were inside `chelis-runtime`.** So `pub(crate)` — the cheap option, and
all the external count would justify on its own — prevents none of them. It closes a hole
nobody was falling through.

Prevention requires the field private to a **module**, so that the 63 internal sites must
also go through accessors. That is the whole cost of this change, and it is the reason it
needs a proposal rather than a commit.

## What Changes

- **`chelis_tensor` moves into its own module** with `data` private to it. `#[repr(C)]`
  does not require the Rust field to be `pub`; the C side uses its own header
  declaration, so the ABI is unaffected.
- **A typed accessor surface** sufficient to retire every legitimate use:
  - typed element access via `TensorElement`, for the six dtypes that have impls;
  - a raw-bytes accessor for the memcpy and allocation paths that are genuinely untyped;
  - pointer-hygiene accessors for the null checks and the free path.
- **The 63 internal sites migrate in reviewable batches**, not one commit.
- **`chelis-python`'s four sites** move to the pointer-hygiene accessors.
- **The field is sealed last**, once nothing outside the module touches it.

### The gap that will need a decision

Three cast targets have no `TensorElement` impl and cannot simply be migrated:

| target | used for | why there is no impl |
| --- | --- | --- |
| `u16` | `bf16` / `f16` storage | the dtype's logical type is not `u16`; it is a 16-bit float carried in one |
| `u8` | raw byte copies | not an element type at all |
| `u32` | one site, to be characterised | unknown; may be an f32 bit-pattern read |

`bf16` / `f16` are the interesting case, and they are the same shape as bool: a logical
type carried in a storage type that is not itself. The `Bool8` newtype added in
`44a4e1db` is the precedent — a `#[repr(transparent)]` element type that every byte value
inhabits, so access is defined and the narrowing is explicit. Phase 1 decides whether
`bf16`/`f16` get the same treatment or whether the raw-bytes accessor covers them.

### Non-Goals

- **Not the bool storage migration.** `CRuntime-BoolStorage-F1` is separate and is
  already staged behind `Bool8`. This change should land first, because it makes that
  migration's missed sites into compile errors rather than silent over-reads.
- **No behavior change.** Every accessor is a typed spelling of a cast that exists today.
  A behavioral difference discovered during migration is a defect found, and is reported
  and fixed separately rather than folded in.
- **No C ABI change.** The struct layout, field order, and header are untouched.
- **Not `pub(crate)`.** Explicitly rejected above: it would not have prevented any of the
  eight bugs.
- **No new dtype and no storage-encoding change.** This change moves access, not bytes.
- **Not a Dylint adoption.** Barnacle's `boundary_safety_restricted_field_access` is the
  fallback if module privacy proves infeasible, and it carries a nightly toolchain cost
  (`harden-vocabulary-kernel` task 0.3). Privacy is preferred because it is enforced by
  the compiler everyone already runs.

## Capabilities

### New Capabilities

- `tensor-storage-encapsulation`: the requirement that a tensor's storage pointer be
  reachable only through accessors that name an element type, that the raw-bytes escape
  be narrow and justified, that the C ABI be unaffected by the encapsulation, and that a
  partial migration be structurally prevented from shipping.

### Modified Capabilities

None. No existing requirement changes; this makes enforceable a convention that
`TensorElement`'s own safety contract already states informally.

## Impact

- `crates/chelis-runtime/src/lib.rs` — the module extraction and 63 call sites. The
  bulk of the change.
- `crates/chelis-runtime/src/` — a new module owning `chelis_tensor` and the accessors.
- `crates/chelis-python/src/lib.rs` — four sites at lines 248, 249, 260, 261.
- `crates/chelis-runtime/tests/` — the accessor surface needs its own tests, including
  that a raw-bytes accessor cannot be used to reconstruct typed access without saying so.
- `docs/gap_synthesis.md` — record that this closes the mechanism behind
  `CRuntime-F32Coupling`, `CRuntime-I32Storage-F1`, and the eight-site instance, as
  distinct from closing any individual entry.
- Relationship to `harden-vocabulary-kernel`: depends on it. `Bool8`, the `data_as_f32`
  assert, and the direct-cast funnelling all land there, and this change starts from that
  state.
- Relationship to `CRuntime-BoolStorage-F1`: should precede it, per the Non-Goals.
