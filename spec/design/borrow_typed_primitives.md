# Borrow-Typed Primitives

**Status:** Implemented for compiler `0.7.0` and `chelis-std` `0.3.0`.
**Owning specs:** `spec/02-surf-syntax.md`, `spec/03-deep-syntax.md`,
`spec/04-type-system.md`, and `spec/05-risc-primitives.md`.

## Summary

Chelis now distinguishes owned tensor values from read-only borrows. Surf writes the
borrow type as `&T`; Deep represents it as `(t-ref {} T)`. Read-only tensor primitives
and standard-library wrappers take borrowed inputs and return owned outputs. Passing an
owned value where a borrow is expected auto-borrows at ordinary call sites and through
the pipe operator.

The reverse direction is deliberately asymmetric: passing `&T` where owned `T` is
expected is a type error. Code that needs a new owned value must write `copy(x)`.

## Linearity Model

The linearity checker is parameter-type driven. A call consumes an argument only when
the resolved parameter type is owned. Borrowed parameters do not consume their owner.
This replaces the previous name-based read-only exception list as the main rule, with
builtin metadata used only to type unresolved builtins consistently.

Local owned bindings still require an explicit consume on every path. Chelis does not
perform implicit local drop. The new borrowed-but-never-consumed diagnostic reports the
borrow sites and the missing final consume separately from the fully unused-value case.
Function parameters are treated as ownership-transfer boundaries, so an owned parameter
may be borrowed through the function body and leave the function scope without a local
`drop` ceremony.

## Backend Boundary

Borrowing is erased before IR and backend lowering. The IR DAG already represents reads
from live tensors by reference-like handles, so C, HIP, Metal, AD, and `vmap` do not
need a new borrow payload. The type checker and linearity checker enforce the borrow
discipline; lowerings see the same owned tensor IR they saw before.

## Standard Library

`chelis-std` `0.3.0` retypes read-only tensor helpers as borrowed inputs and removes
the old noisy `copy()` fan-out patterns from source and examples. The bundled std
artifacts are regenerated from the new source and embedded in `chelis-std-bundle`.
Post-build verification must confirm the embedded package reports `0.3.0` and that a
borrow-typed primitive such as `matmul` checks through the bundled loader.

## Macro Surface

Macros operate on Deep after Surf desugaring and before type checking. Macro code that
matches type tags must either handle `t-ref` or explicitly reject it. Existing macro
tests must include at least one macro-expanded corpus path after borrow-typed code is
present, so `(borrow ...)` and `t-ref` are not silently treated as transparent trivia.

## Regression Coverage

Required coverage for this feature:

- owned-to-borrow auto-borrow succeeds for read-only builtins
- `&T` to owned `T` fails unless wrapped in `copy`
- `copy(&x)` yields an owned tensor
- `x |> f` auto-borrows when `f` takes `&T`
- a borrowed local owner that is never consumed reports the specialized diagnostic
- returning a borrow from a function is rejected
- `grad(f)(x)` works when `f` takes `&tensor`
- `vmap(f, axis=0)(xs)` works when `f` takes `&tensor`
- Deep validation accepts `t-ref` and keeps the 61-tag vocabulary closed
- macro-expanded programs continue through check/build when borrow nodes are present

## Downstream Propagation

Propagation targets are `nautilus`, `coral`, `octant`, and `hello-chelis` when local
checkouts exist. Each shell gets the same audit:

- bump compiler pins to `=0.7.0` and dependency pins to `chelis-std` `0.3.0`
- audit exported Surf function signatures and retype read-only tensor or frame inputs
  to borrows where the body does not consume them
- remove redundant `copy()` calls used only for read-only fan-out
- keep explicit `copy()` where ownership is intentionally forked for a later consume
- run the shell's documented check/build/test gate

Bootstrap-list updates stay deferred until downstream shell artifacts are actually
published. This avoids repeating the Octant `0.3.3` cycle where bootstrap metadata was
advanced before every downstream artifact existed.

### Local Propagation Status

As of the `0.7.0` compiler implementation, the local `chelis-std` `0.3.0` package has
been built and published into the local Reef registry. Local downstream checkouts for
`nautilus`, `coral`, `octant`, and `hello-chelis` were verified and their manifest pins
were advanced to the new compiler/std versions.

Source-level downstream migration is still gated on Nautilus. Its first `reef build`
under `0.7.0` fails on numerous local tensors that are borrowed by read-only calls but
not explicitly consumed before scope exit. Those sites should be fixed by the
per-function audit above: add real `drop(...)` calls for local scratch tensors that are
only borrowed, retype exported read-only inputs as `&T`, and remove redundant copies only
after each function's ownership story is clear. Coral, Octant, and hello-chelis should
not publish their bumped pins until Nautilus `0.7.0` is built and published.
