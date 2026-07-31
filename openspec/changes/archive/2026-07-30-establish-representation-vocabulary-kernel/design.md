## Context

On `main`, `RuntimeDType::byte_width()` is a direct match over logical dtypes. This table cannot distinguish equal-width encodings with different bit meanings.

`chelis-vocab` also owns a C source renderer. The renderer requires `String`, `format!`, and `std`, which prevents a dependency-bottom `no_std` boundary.

The checked-in C header has a byte-equality test. That test proves deterministic rendering but does not execute the generated decoder.

This change extracts the representation foundation from draft PR #894. PR #896 can add arithmetic width after this foundation lands.

## Goals / Non-Goals

**Goals:**

- Define each active runtime encoding with one closed `Repr` variant.
- Preserve every current runtime dtype ID, macro, width, and generated header byte.
- Make `chelis-vocab` a dependency-free `no_std` kernel.
- Put C source rendering in `chelis-runtime`.
- Compare generated C behavior with the Rust vocabulary.
- Prepare a small source surface for later Verus and Buoy work.

**Non-Goals:**

- Do not add arithmetic width from PR #896.
- Do not correct native int32 decoder defects from PR #894.
- Do not migrate bool storage from four bytes to one byte.
- Do not seal or type the runtime tensor data pointer from issue #893.
- Do not change the compiler toolchain or backend support matrix.
- Do not claim formal proof in this change.

## Decisions

### Name physical encodings, not logical dtypes

`Repr` names the physical bit encoding. Planned variants are:

- `Ieee754Binary16`
- `Ieee754Binary32`
- `Ieee754Binary64`
- `Bfloat16`
- `TwosComplement8`
- `TwosComplement16`
- `TwosComplement32`
- `TwosComplement64`
- `BoolInBinary32`

`BoolInBinary32` names the current ABI debt. It remains four bytes and reports that it is payload encoded.

A later atomic bool migration can replace this variant with `Bool8`. PR A does not add the future variant because the closed vocabulary describes active encodings only.

A width-only enum was rejected. IEEE binary32 and two's-complement int32 have the same width but are not interchangeable.

### Make representation the only source of width

`RuntimeDType::repr()` maps each runtime dtype to one `Repr`. `RuntimeDType::byte_width()` delegates to `Repr::byte_width()`.

The public width return type remains `usize`. This choice prevents an unrelated Rust API change.

A second per-dtype width match was rejected because it can drift from the representation map.

### Keep decoding total and allocation-free

`chelis-vocab` uses `core` only. Unknown effect-symbol errors borrow the source symbol instead of creating a `String`.

This change updates all workspace callers in the same change set. The borrowed error is a Rust API change but not a runtime ABI change.

Dropping the unknown symbol was rejected because the current diagnostic includes it.

### Move rendering to the artifact owner

`chelis-runtime` owns `include/chelis_runtime_dtype.h`, so it also owns `dtype_header::render_runtime_dtype_c_header()`.

The renderer continues to read `RuntimeDType::ALL`, IDs, macro names, and derived widths from `chelis-vocab`. The checked-in header remains byte-identical.

Keeping the renderer in `chelis-vocab` was rejected because source emission requires allocation and is shell work.

### Use structural and executable evidence

The purity suite reads the manifest and source because it guards declared crate properties. It rejects dependencies, `alloc`, removal of `no_std`, and removal of the unsafe-code prohibition.

Closed-domain tests cover every `RuntimeDType::ALL` value, unique IDs, decode round-trips, representations, and widths. Exhaustive Rust matches provide compile-time coverage when enum variants change.

The C behavior test compiles one probe against the generated header. The probe accepts a dtype ID and prints the derived width.

The Rust test invokes the probe for every valid tag and compares the result with `RuntimeDType::byte_width()`. A separate invocation supplies an invalid tag and requires an unsuccessful process exit.

The byte-equality test remains because behavior agreement does not prove that the shipped header matches the renderer.

### Use the repository C compiler contract

The C behavior test uses the repository `gcc` command, which can map to clang on macOS. The cross-platform compiler shim must land first or this change must rebase onto it.

Adding separate compiler discovery to this test was rejected because the repository already owns that policy.

## Risks / Trade-offs

- **Risk: `BoolInBinary32` can appear to endorse known debt.** → Its name and payload flag make the debt explicit. The proposal excludes the migration.
- **Risk: The moved public renderer path breaks a downstream Rust caller.** → Repository search shows one test consumer. The change updates all workspace callers.
- **Risk: Source-text purity tests can miss indirect capabilities.** → The empty manifest, `no_std`, and no `alloc` form the primary compiler-enforced boundary.
- **Risk: The C test depends on a working compiler command.** → Land the cross-platform compiler shim first and use CI as the final platform oracle.
- **Risk: These tests are not formal proofs.** → Record only tested and structurally-enforced evidence. Add Verus and Buoy in a later change.

## Migration Plan

1. Land or rebase onto the cross-platform C compiler shim.
2. Add positive and negative test stubs from the capability specification.
3. Add `Repr` and derive widths without any value change.
4. Move C-header rendering and update its single consumer.
5. Convert `chelis-vocab` to `no_std` and borrowed decode errors.
6. Run the crate tests, C probe, and repository gate.

A revert restores the old direct width match and renderer location. This change changes no stored data or runtime wire value.

## Open Questions

None.
