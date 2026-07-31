## 1. Baseline and Test Stubs

Baseline at `0709d6be`: `chelis_runtime_dtype.h` is 850 bytes. Its SHA-256 is `d1167988df9d149077d5f20f65b4fda9fe260e6f9ee969bc9c489ef9168d1d5f`.

- [x] 1.1 Rebase onto the cross-platform C compiler shim and record the baseline runtime dtype header bytes.
- [x] 1.2 Add failing tests for `Repr`, representation identity, current widths, payload status, and complete `RuntimeDType` mapping.
- [x] 1.3 Add failing purity tests for dependencies, `alloc`, `no_std`, unsafe code, and source-renderer ownership.
- [x] 1.4 Add failing C-probe tests for every valid dtype tag and one invalid tag.
- [x] 1.5 Demonstrate that the new tests fail on the baseline or on one deliberate local mutation per guarded property.

## 2. Representation Vocabulary

- [x] 2.1 Add the closed `Repr` enum, its complete `ALL` list, derived byte widths, and payload-encoding classification.
- [x] 2.2 Map every `RuntimeDType` to one `Repr` value and delegate `RuntimeDType::byte_width()` to that value.
- [x] 2.3 Preserve every existing dtype ID, macro, width, name, and current four-byte bool representation.
- [x] 2.4 Add round-trip and uniqueness tests for all runtime dtype IDs and mappings.

## 3. Dependency-Bottom Kernel

- [x] 3.1 Move `render_runtime_dtype_c_header()` and its test import to `chelis-runtime::dtype_header`.
- [x] 3.2 Make `chelis-vocab` use `no_std`, `core`, no allocation, no unsafe code, and no dependencies.
- [x] 3.3 Change unknown effect-symbol errors to borrow the source symbol and update all workspace callers.
- [x] 3.4 Add the crate lint table for lossy cast and arithmetic classes.
- [x] 3.5 Make the purity suite fail if a later change restores dependencies, allocation, `std`, unsafe code, or source rendering.

## 4. Cross-Language Evidence

- [x] 4.1 Keep the generated C dtype header byte-identical to the checked-in header.
- [x] 4.2 Compile a C probe and compare every valid tag result with `RuntimeDType::byte_width()`.
- [x] 4.3 Execute the C invalid-tag path and require an unsuccessful process exit with no fallback width.
- [x] 4.4 Prove that the C-probe test detects one deliberately incorrect generated `case` result.

## 5. Validation

- [x] 5.1 Run `cargo nextest run -p chelis-vocab` and require all tests to pass.
- [x] 5.2 Run the focused `chelis-runtime` header and C-probe tests and require all cases to pass.
- [x] 5.3 Run `openspec validate establish-representation-vocabulary-kernel --strict` and require success.
- [x] 5.4 Run `python3 scripts/gate.py --local` as the authoritative completion oracle.
- [x] 5.5 Compare the generated header and public ABI values against `main` and require no wire difference.
