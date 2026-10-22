Move primitive identities, sealed numeric values, operations, activation graphs, observation, and agreement into `chelis-numeric-core` with unconditional `no_std`. Callers use canonical modules without compatibility exports. Default `alloc` adds tensors and owned text. Optional `serde` preserves stored-value transport. Host crates retain schemas, I/O, native allocation, and the runtime C ABI.

The shared CORE-MATH provider supports bare targets without a C library. Acceptance compares immutable-main behavior, wire bytes, schemas, generated C, traps, and allocation costs. See [#3993](https://github.com/Chelis-Lang/chelis/issues/3993).
Schema metadata ownership uses verified codec definitions. Payload publication still needs an admitted declaration.

Compile probes retain the vocabulary failure module.
Source spelling checks read the canonical primitive owner.
Host tests compile the bare-target math helpers.
Schema entry points retain floating-point guards.
Generated-C checks link each program with its exact staged runtime archive.
Baseline cleanup removes the complete isolated cache.

Runtime source records include numeric code and CORE-MATH C inputs. Development consumers reject stale Rust and C sources. Rebuilt consumers retain the existing staging contract.
