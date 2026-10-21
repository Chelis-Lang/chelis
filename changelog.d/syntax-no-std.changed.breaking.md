`chelis-deep`, `chelis-pred`, `chelis-surf`, and `chelis-macros` use `no_std` with `alloc`. Parser, formatter, desugarer, and decompiler callers pass explicit `StackServices`. Native callers use `chelis-syntax-host::NativeStack`. Module identity checks move from `chelis-surf` to `chelis-syntax-host`. `chelis-unord` uses portable storage. Syntax and wire contracts remain unchanged.

The syntax acceptance command checks isolated embedded builds, native parity, allocation costs, and failure controls. Linux integration support and macOS nightly validation execute the complete command. See [#3979](https://github.com/Chelis-Lang/chelis/issues/3979).

The acceptance command fetches the pinned baseline for shallow checkouts.
It rejects matching snapshots with missing nested evidence.

Standalone rejection fixtures retain locked dependencies on the resource crates.
Historical cache producer bytes and artifact hashes remain unchanged.
Current cache tests use separate live helpers.
Acceptance tests share one support module.
