Move primitive identities, sealed numeric values, operations, activation graphs, observation, and agreement into `chelis-numeric-core` with unconditional `no_std`. Callers use canonical modules without compatibility exports. Default `alloc` adds tensors and owned text. Optional `serde` preserves stored-value transport. Host crates retain schemas, I/O, native allocation, and the runtime C ABI.

The shared CORE-MATH provider supports bare targets without a C library. Acceptance compares immutable-main behavior, wire bytes, schemas, generated C, traps, and allocation costs. See [#3993](https://github.com/Chelis-Lang/chelis/issues/3993).
Schema metadata ownership uses verified codec definitions. Payload publication still needs an admitted declaration.
