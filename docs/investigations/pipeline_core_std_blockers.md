# Standard-Library Blockers for the Pipeline Core

`chelis-pipeline-core` requires `std`.

This inventory records direct core use and transitive blockers in its seven direct dependencies.
It does not claim current `#![no_std]` support for `chelis-pipeline-core`.

## Direct Core Use

| Blocker class | Confirmed use |
|---|---|
| Collections and allocation | Core artifacts use `Vec`, `String`, `Box`, `BTreeMap`, and the order-free `chelis-unord` hash collections. |
| Global state | The core declares no global mutable state. |
| Stack support | The core calls lower crates that use stack growth. |
| Panic behavior | Core errors implement `std::error::Error`. Core tests use `panic!` for failed fixtures. |
| Operating-system use | Core production code does not use files, processes, sockets, or environment variables. |
| Dependency features | Existing owners retain their declared features. Portable owners disable dependency `std` defaults. |

## Transitive Lower-Crate Blockers

### `chelis-deep`

`chelis-deep` uses unconditional `no_std` with allocation-backed syntax data.
Its parser takes explicit `chelis-stack-services` resources.
Native pipeline callers use `chelis-syntax-host::NativeStack`.
This portable dependency does not make the pipeline core portable.

### `chelis-types`

`chelis-types` uses standard collections, `Arc`, `Mutex`, cells, and thread-local state. Cancellation and linked-program guards use thread-local storage.

Type inference uses `stacker::grow` and `stacker::remaining_stack`. The `stacker` dependency uses platform stack and memory services.

The checker also retains panic paths for internal invariant failures. Its default dependencies include `serde`, `thiserror`, `half`, and `stacker`.

### `chelis-effects`

`chelis-effects` retains checked reconstruction, target admission, realizability, and root policy.
Its `chelis-types` dependency retains the host checker blockers.
Portable effect inference and annotation reside in `chelis-effects-core`.

### `chelis-effects-core`

`chelis-effects-core` uses unconditional `no_std` with explicit `alloc`.
It owns effect inference, annotation, handler validation, declared-row checks, and shared effect error data.
The caller supplies allocation and stack resources.
This portable dependency does not make the pipeline core portable.

### `chelis-ir`

`chelis-ir` uses standard collections, thread-local state, and `OnceLock`. Lowering uses `std::panic::catch_unwind` to convert selected panic payloads into diagnostics.

Lowering uses `stacker::grow` and `stacker::maybe_grow`. The IR crate also retains panic paths for internal compiler invariants.

The default dependencies include `serde`, `thiserror`, and `stacker`. `chelis-ir` also inherits blockers from `chelis-types` and `chelis-effects`.

### `chelis-unord`

`chelis-unord` uses unconditional `no_std` and allocation-backed ordered storage behind its order-free collection interface.
Its portable `serde` dependency retains canonical cache serialization.
This dependency does not need host services.

### `chelis-dag-core`

`chelis-dag-core` owns graph data and pure graph passes.
It uses `alloc`, portable numeric data, checked names, and deterministic lookup tables without host services.
Public remap and binding maps retain `chelis-unord`.
Host lowering, evaluation, target policy, diagnostic stages, and production orchestration remain outside this dependency.
This portable dependency does not make `chelis-pipeline-core` portable.


## Scope

This change does not assign work to remove these blockers. A separate approved change must define any future `#![no_std]` work.
