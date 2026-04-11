# Phase 3m: Rust Runtime Rewrite

## Goal

Replace the growing C runtime implementation with a Rust static library while cleaning
up the compiled host-value ABI before any more runtime-heavy Phase 3 work lands.

This is not a language-semantics phase. Chelis programs keep the same observable
meaning. The change is to the compiled runtime contract, ownership model, and build
surface used by generated C and HIP host code.

## Why 3m Exists

Phase `3c` and `3d` introduced a real host-value lane: strings, lists, dicts, tuples,
`Option`, and print/debug all compile through the generated host program path. The
original C runtime was acceptable when it mostly handled tensors plus a few helpers.
It is now the wrong substrate for the next Phase 3 work:

- `3g` adds file I/O, CSV/JSON, tokenizer loading, and batching
- `3i` adds time/date and exact-decimal runtime support
- later shell work (`school`, `coral`, `treasure`) depends on a stable host runtime

If those features are added to `chelis_runtime.c`, Chelis accumulates more manual memory
management, more ad hoc container code, and more C-side ownership bugs exactly where the
language is growing fastest.

`3m` is the emergency correction: move runtime implementation into Rust now, while the
host-value surface is still small enough to migrate cleanly.

## Scope

### In Scope

- new crate: `crates/chelis-runtime/`
- Rust implementation of the runtime as `libchelis_runtime.a`
- ownership of `chelis_runtime.h` moves to the runtime crate
- ABI cleanup for host values: opaque handles + explicit retain/release
- generated host C updated to use accessors and retain/release instead of field peeking
- `chelis build` updated to copy/link the Rust runtime library
- test and doc migration away from `chelis_runtime.c`

### Out Of Scope

- changing Chelis language semantics
- replacing generated tensor kernels
- changing the evaluator contract
- changing the HIP kernel ABI
- adding new language/runtime features beyond what current codegen already needs

## ABI Decision

This phase takes the **ABI cleanup now** path.

### Tensors stay layout-visible

`chelis_tensor` remains a source-visible `#[repr(C)]` struct with the current stable
field order. Generated numeric C and HIP code dereference tensor fields directly in hot
loops, so tensor layout stability is required.

### Host values become opaque

`chelis_string`, `chelis_list`, `chelis_tuple`, and `chelis_dict` stop exposing concrete
field layouts. They become opaque runtime-managed handles declared in
`chelis_runtime.h`.

Generated host code must not read `.data`, `->len`, `->items`, or `->entries` directly.
It must call runtime accessors.

### Explicit ownership

The current compiled host path has no first-class retain/release contract for host
values. `3m` adds one:

- per-handle `retain` / `release`
- `chelis_value_retain`
- `chelis_value_release`

Generated host code becomes responsible for releasing owned temporaries and balancing
container insertion/return paths. The runtime owns internal storage; codegen owns local
handle lifetime discipline.

## Runtime Structure

Create a new workspace crate:

```text
crates/chelis-runtime/
  Cargo.toml
  include/chelis_runtime.h
  src/
    lib.rs
    tensor.rs
    blas.rs
    string.rs
    list.rs
    tuple.rs
    dict.rs
    value.rs
    print.rs
    io.rs
```

Implementation rules:

- crate type: `staticlib`
- non-tensor host values use Rust-owned storage behind opaque handles
- strings preserve current character-count semantics for `len` and slicing
- dict iteration preserves insertion order
- print/debug output remains byte-for-byte compatible with the current surface unless
  an active spec says otherwise

## Header Contract

`chelis_runtime.h` remains the C ABI contract, but it changes in `3m`:

- `chelis_tensor` stays source-visible
- host-value structs become opaque forward declarations
- add accessor APIs for every current generated-code field peek
- add retain/release APIs for handles and `chelis_value`

The compiler and tests must treat the header shipped by `chelis-runtime` as the only
source of truth.

## Build Surface

### Output Artifacts

`chelis build` for C or HIP now emits:

- generated program source (`.c` / `_hip.cpp`)
- generated function header when applicable
- `chelis_runtime.h`
- `libchelis_runtime.a`
- `chelis_hip_runtime.h` for HIP builds

It no longer emits `chelis_runtime.c`.

### Runtime Discovery

Runtime library discovery order is fixed and explicit:

1. `CHELIS_RUNTIME_DIR`
2. path relative to `std::env::current_exe()`
3. hard error with a clear message

Do not search arbitrary directories.

Development expectation:

- `cargo run -p chelis-cli` users set `CHELIS_RUNTIME_DIR` to the Cargo build output
  directory containing `libchelis_runtime.a`

Installed-binary expectation:

- the runtime library is found relative to the installed `chelis` executable

Required failure text shape:

- it must clearly mention `libchelis_runtime.a`
- it must mention `CHELIS_RUNTIME_DIR`
- it must tell the user to install Chelis correctly or set the environment variable

## Execution Plan

1. land this spec and sync active planning docs
2. add `3m` test stubs from this spec before runtime implementation
3. create `chelis-runtime` and move the runtime header into it
4. implement the Rust runtime modules
5. update host codegen to use accessors and retain/release
6. update CLI build output and runtime discovery
7. migrate compile/run tests away from `chelis_runtime.c`
8. remove `crates/chelis-backend-c/runtime/chelis_runtime.c`

## Acceptance Oracle

Authoritative oracle:

```sh
cargo test -p chelis-cli phase3m_rust_runtime_acceptance_oracle -- --nocapture
```

That oracle must prove all of the following in one named suite:

- a mixed tensor + scalar/string/list/dict program builds through `chelis build --target c`
- the emitted output contains `chelis_runtime.h` and `libchelis_runtime.a`
- the emitted output does not contain `chelis_runtime.c`
- the printed compile line links `-lchelis_runtime`
- the generated host C no longer reads host-value fields directly
- the compiled binary matches `chelis eval`

## Supporting Gates

Default repo gate:

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Runtime-focused suite:

```sh
cargo test -p chelis-runtime
```

## Manual HIP Gate

This gate is not part of the default workspace run and must stay documented as manual:

```sh
CHELIS_RUNTIME_DIR=<runtime-dir> cargo test -p chelis-cli phase3m_rust_runtime_hip_manual_gate -- --ignored --nocapture
```

Expected success condition:

- `chelis build --target hip` emits `_hip.cpp`, `chelis_runtime.h`,
  `libchelis_runtime.a`, and `chelis_hip_runtime.h`
- the emitted program links and runs on the local ROCm machine
- output matches `chelis eval` for the same mixed host+tenso​r program

## Required Tests

### Runtime crate

- string length/slice use character counts, not byte counts
- string concat/trim/contains/starts_with/ends_with match the current surface
- list and dict accessors match codegen expectations
- dict preserves insertion order
- handle retain/release is safe under clone/drop patterns
- `chelis_value_retain` / `chelis_value_release` dispatch correctly by tag
- tensor ABI layout assertions remain stable

### CLI / build

- runtime discovery prefers `CHELIS_RUNTIME_DIR`
- fallback to executable-relative runtime works
- missing runtime produces the documented hard error
- `chelis build` copies `libchelis_runtime.a`, not `chelis_runtime.c`
- printed compile lines mention `-lchelis_runtime`

### Codegen

- generated host programs no longer emit `.data` on strings or `->len` on host
  collections
- generated host programs balance retain/release across branches, loops, and returns
- evaluator-vs-compiled agreement still holds for mixed programs

## Red-Team Checklist

Before calling `3m` healthy enough to unblock `3g`, red-team these concrete surfaces:

- no active docs still claim `chelis build` emits `chelis_runtime.c`
- no generated host code still peeks into non-tensor runtime struct fields
- no active tests compile `chelis_runtime.c`
- runtime discovery failures are explicit and actionable
- mixed-program compiled execution agrees with `chelis eval` on both C and HIP paths

