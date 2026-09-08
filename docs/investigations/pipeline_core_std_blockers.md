# Standard-Library Blockers for the Pipeline Core

`chelis-pipeline-core` requires `std`.

This inventory records direct core use and transitive blockers in its five direct dependencies. It does not claim current `#![no_std]` support.

## Direct Core Use

| Blocker class | Confirmed use |
|---|---|
| Collections and allocation | Core artifacts use `Vec`, `String`, `Box`, `BTreeMap`, and the order-free `chelis-unord` hash collections. |
| Global state | The core declares no global mutable state. |
| Stack support | The core calls lower crates that use stack growth. |
| Panic behavior | Core errors implement `std::error::Error`. Core tests use `panic!` for failed fixtures. |
| Operating-system use | Core production code does not use files, processes, sockets, or environment variables. |
| Dependency features | The crate uses the default feature sets of all five lower crates. |

## Transitive Lower-Crate Blockers

### `chelis-deep`

`chelis-deep` uses standard collections and allocated strings throughout its AST and authoring APIs. Its default dependencies include `serde`, `thiserror`, and `miette`.

### `chelis-types`

`chelis-types` uses standard collections, `Arc`, `Mutex`, cells, and thread-local state. Cancellation and linked-program guards use thread-local storage.

Type inference uses `stacker::grow` and `stacker::remaining_stack`. The `stacker` dependency uses platform stack and memory services.

The checker also retains panic paths for internal invariant failures. Its default dependencies include `serde`, `thiserror`, `half`, and `stacker`.

### `chelis-effects`

`chelis-effects` uses standard map and set collections. It also depends on `chelis-types`, so it inherits the type-checker blockers.

### `chelis-ir`

`chelis-ir` uses standard collections, thread-local state, and `OnceLock`. Lowering uses `std::panic::catch_unwind` to convert selected panic payloads into diagnostics.

Lowering uses `stacker::grow` and `stacker::maybe_grow`. The IR crate also retains panic paths for internal compiler invariants.

The default dependencies include `serde`, `thiserror`, and `stacker`. `chelis-ir` also inherits blockers from `chelis-types` and `chelis-effects`.

### `chelis-unord`

`chelis-unord` wraps standard-library hash collections behind an order-free API and uses `serde` for canonical cache serialization. Its storage therefore requires `std` even though observable iteration is unavailable.

## Scope

This change does not assign work to remove these blockers. A separate approved change must define any future `#![no_std]` work.
