# Typed graph fixtures

`test_capacity_census_graph.py` is supporting evidence for
`spec/design/dtype_semantics.md` §C6. Run it from the repository root:

```sh
.venv/bin/python -m unittest discover -s scripts -p test_capacity_census_graph.py
```

The tests run the pinned Rust toolchain, rather than consume a saved rustdoc
snapshot. `graph.rs` exercises private helpers, a public reexport, a normalized
alias and recursive generic types. A second source-to-rustdoc run changes a
numeric argument to `bool` and checks the resulting capacity change. The small
locked `serde/` crate establishes the actual rustdoc representation of serde
attributes, derived implementations and a custom serializer. Its build uses
one Cargo job and this worktree's `target/graph-fixture-target` cache.

Configuration closure records this directory as standalone fixture source owned
by the script suite. The suite rejects any missing or additional Rust file
outside its two actual rustdoc inputs, so that directory entry cannot silently
admit a new uncompiled fixture.

Synthetic artifact mutations exercise unresolved imports, unsupported types,
growing generic substitutions, discriminator and width changes, role swaps,
ambiguous registrations, and mixed numeric siblings. The fixed point records
finite declaration-leaf/primitive pairs; it does not enumerate an infinite
recursive value path. The growing-generic case is synthetic because rustdoc's
automatic trait expansion itself does not terminate promptly for that fixture.

The engine supports explicit type parameters, ordinary recursive nominal types,
aliases, tuples, concrete arrays, references and the exact standard containers
listed in its adapter table. Missing defining artifacts, const/defaulted generic
parameters, opaque/associated types, unknown serde options and custom serializers
fail closed. Root discovery enumerates public type exports; selecting all
published serialization roots remains the live integration's responsibility.

The structural recognizer supports only the decided u64 source-coordinate
fields and u64 input slot with its separate int32 axis. It checks canonical type
identity, complete graph identity, exact role structure and derived serde
identity. A matching digest or arbitrary tagged number supplies no authority.
The witness is structural evidence only: it does not prove measured provenance,
owner/slot validation, runtime metadata adoption or execution of a decode path.

These fixtures do not activate the live census, change wire versions, retire a
baseline row, or replace the `capacity_census_wire` acceptance oracle. The atomic
wire migration still owes actual serializer/decoder and cache executions,
execution-backed adapters for custom codecs (including JSON/binary dispatch),
final authority for every discovered leaf and the complete C6 mutation matrix.
