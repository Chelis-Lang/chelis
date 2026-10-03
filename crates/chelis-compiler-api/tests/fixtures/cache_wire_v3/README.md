# Previous numeric wire cache fixtures

These files were emitted and read successfully by the actual compiler at
`010354f91d1efd210037f4090f6e862aa7a68773`, using the accompanying `producer.rs`.
The producer checks that both sub-context readers hit without rewriting the
files and that the compiled-context reader returns a fresh context. The old
formats are compiled context 17, stdlib 15, and dependency library 11. Each
fixture contains a checked library with floating-point arithmetic. The stdlib,
compiled disk and worker artifacts also contain actual scalar and tensor-storage
payloads: exact int64 values above 2^53, positive scalar zero, negative tensor
zero and an adjacent f64 value. The dependency-library cache carries checked
source state and has no lowered DAG/storage payload.

`producer.json` records the producer build identity, source declarations,
keys, and SHA-256 identities. The key-input files are the actual ordered bytes
hashed by that producer. The compatibility test changes only their format
version to isolate version admission from build-identity invalidation.

The compatibility test compiles `producer.rs` against the current compiler API
for its helpers, so the copy here passes the embedded chelis-std runtime to
`compile_reef_context` and `CompiledContext::load_if_fresh`, which take it
explicitly. At the producer commit those calls have no runtime argument; remove
it when regenerating.

To regenerate, create an isolated worktree at the producer commit, copy
`producer.rs` to `crates/chelis-compiler-api/examples/cache_wire_fixture_producer.rs`,
and run it with a task-owned `CHELIS_REEF_HOME` and output directory:

```sh
CHELIS_REEF_HOME="$PWD/target/fixture-reef-home" cargo run -p chelis-compiler-api --example cache_wire_fixture_producer -- "$PWD/target/cache-wire-fixtures"
```

Copy the three cache files, worker handoff, two key-input files and producer manifest here;
record their SHA-256 identities and the producer source identity in the
manifest. New-producer/current-reader hits are generated during the test.
Do not regenerate old fixtures using the current compiler.

The same harness also emits `worker-unversioned.bin` through the old public
worker encoder and decodes it with the old worker reader. Both run in the same
executable because the compiler compatibility identity includes that executable's
image identity. The manifest binds the worker bytes with the other fixtures.
The current reader must reject these unversioned bytes at the envelope boundary;
a current writer and reader use the same checked envelope as the disk cache.

The manifest records actual old `ScalarValue` and `TensorStorage` positional
bytes read from the lowered DAG. The current test constructs their expected
values and old/current encodings independently, checks old floating payload
decoder rejection, and requires matching numeric payloads after current cache
reconstruction. Recomputed checksums on changed scalar/storage zero bits must
still fail the lowered-library comparison on stdlib, compiled disk and worker
reads. This evidence does not adopt the separate native AST/checker cache codec
as a complete numeric wire surface.

The immediate predecessor also contains the checked `fixture_literal` declaration
with result `tensor[4, f32]`. The producer and current roundtrip controls inspect
that exact result type. This reusable library retains the function as a checked
definition; no claim is made that its cached DAG contains an `ExtentWitness`.
Direct witness transport is exercised by `wire_extent_witness.rs`. General
imported-call execution remains governed by #1277.
