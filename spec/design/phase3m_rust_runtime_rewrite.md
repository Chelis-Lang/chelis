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
- later shell work (`nautilus`, `coral`, `shoals`) depends on a stable host runtime

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

This phase took the **ABI cleanup now** path described below. The tensor half is a
historical delivery record, not the current target contract: chelis#1286 and
[`compiled_value_ownership.md`](compiled_value_ownership.md) supersede it with one
opaque, refcounted heap model and a verified ownership boundary.

### Superseded tensor decision

`chelis_tensor` remains a source-visible `#[repr(C)]` struct with the current stable
field order. Generated numeric C and HIP code dereference tensor fields directly in hot
loops, which was why this delivery kept the layout visible. The successor cut removes
that reachability atomically: generated code uses tagged read/unique-write access and
the shared reuse proof, while the public handle becomes opaque. No compatibility ABI
retains the visible layout.

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

`chelis_runtime.h` remains the C ABI contract. The historical `3m` cut made these
changes:

- `chelis_tensor` stays source-visible
- host-value structs become opaque forward declarations
- add accessor APIs for every current generated-code field peek
- add retain/release APIs for handles and `chelis_value`

The successor chelis#1286 cut replaces the first bullet with an opaque tensor declaration,
adds tensor/storage lifetime and guarded tagged-data operations, and removes
`chelis_free`, `chelis_alloc_view`, and ambiguous value-conversion aliases in one
change set.

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
When a generated object-mode header would otherwise export a source-level
`main`, the emitted C symbol is renamed to `<program>__main` so downstream
drivers can link their own `main(void)` without collision.

### Runtime Staging

`spec/08-backends.md` §2.1 owns the rule; this section records the
implementation. The CLI carries its runtime: the `chelis-runtime-bundle` crate
embeds, through `chelis-runtime-bundle-macro`, the static archive produced by the
`chelis-runtime` compilation the same build links, and re-exports that
compilation's public headers. `chelis build` stages exactly those bytes, verifies
the written archive against the carried SHA-256, writes
`chelis_runtime.receipt.json`, and reports
`Staged runtime <dir>/libchelis_runtime.a (sha256 <digest>)`. Nothing searches for
a runtime archive, so archives that other configurations or commits leave in a
target directory are never read.

- A set `CHELIS_RUNTIME_DIR` fails `chelis build` before any output is written;
  the message tells the user to unset it.
- `chelis runtime export <dir>` writes the carried runtime for packaging; release
  tarballs ship its output and build with `sealed-runtime`.
- A test or oracle that needs an instrumented runtime builds its consumer with
  that runtime feature, or links an exact instrumented Cargo artifact itself.
- The Python extension stages its carried runtime into the artifact directory
  and rejects a set `CHELIS_RUNTIME_DIR` (chelis#1354).
- A development build checks its runtime's sources before staging.
  `crates/chelis-runtime/build.rs` records the SHA-256 of each declared input,
  relative to the workspace root, as `chelis_runtime::build_record::SOURCES`, and
  has Cargo rerun it when a declared root changes. A development bundle embeds
  the path of its checkout and, before `chelis build`, `chelis runtime export`
  or `compile_and_load` does other work, compares the record with that checkout
  and fails with the changed, removed and added paths. A runtime compiled
  outside the workspace, such as a per-crate Nix build, records its missing
  roots, and a development build refuses it. Sealed builds (`sealed-runtime` on
  the CLI and the extension) carry no checkout path and skip the check.
  `crates/chelis-runtime-bundle/tests/declared_runtime_inputs.rs` checks the
  declared roots against Cargo's dep-info for each runtime configuration. The
  workspace manifest, `.cargo/config.toml` and `rust-toolchain.toml` are outside
  the declared inputs. Because the embedded path is a `CARGO_MANIFEST_DIR` value,
  Kache keys the development bundle, and the CLI and extension crates that link
  it, per checkout; the runtime's record carries no path, so the runtime stays
  shareable across checkouts.

## Execution Plan

1. land this spec and sync active planning docs
2. add `3m` test stubs from this spec before runtime implementation
3. create `chelis-runtime` and move the runtime header into it
4. implement the Rust runtime modules
5. update host codegen to use accessors and retain/release
6. update CLI build output and runtime staging
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
- the printed compile line links the staged `libchelis_runtime.a` by path
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
cargo test -p chelis-cli phase3m_rust_runtime_hip_manual_gate -- --ignored --nocapture
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

- `chelis build` stages the runtime the CLI carries, whatever other archives
  exist near the executable
- a set `CHELIS_RUNTIME_DIR` fails the build before any output is written
- `chelis build` copies `libchelis_runtime.a`, not `chelis_runtime.c`
- printed compile lines name the staged `libchelis_runtime.a` by path

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
- `chelis build` stages only the runtime the CLI carries and rejects a set
  `CHELIS_RUNTIME_DIR` before writing output
- mixed-program compiled execution agrees with `chelis eval` on both C and HIP paths
