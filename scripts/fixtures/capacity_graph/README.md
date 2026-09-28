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
aliases, tuples, arrays, references and the exact standard containers listed in
its adapter table. Nondefault `usize` const parameters accept decimal literals
and declared-parameter substitutions, including `Hex<4/8/16>` and `[T; N]`.
Missing defining artifacts, computed/defaulted generic arguments, opaque or
associated types, unknown serde options and unsupported custom serializers fail
closed. Root discovery enumerates public type exports; selecting all
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

`test_capacity_census_wire_adapters.py` extends this supporting evidence with
actual `chelis-types` rustdoc and the developer-only `wire_codec_probe` example.
The verifier checks the private canonical scalar/storage carriers, real
JSON/positional enum mirrors, const bit widths and actual serde implementation
identities. The Python suite owns expected observations and records each selected
case's execution and outcome under `target/coordination/`. Its matrix compares
actual own-width storage with independent JSON and bincode expectations, including
all reduced-float bit patterns and paired rejection cases. A source-byte change
invalidates the receipt even when Git's index hides the change.

The canonical adapter can expose the numeric capacity of an IEEE bit-string
field only after checking its enclosing exact dtype/width relationship and actual
codec execution. `String`, `HexBits` alone, an arbitrary tagged numeric enum or
a caller-provided descriptor supplies no authority. The adapter does not cover
source admission, fixed-report codecs, scoped reference validation, operation
registrations or cache version adoption; the complete live wire oracle remains
the atomic integration's responsibility.

The API-profile adapter suite additionally uses the workspace-owned
`chelis-compiler-api/examples/wire_schema_probe.rs` observation driver:

```sh
.venv/bin/python scripts/test_capacity_census_wire_schema.py
```

The verifier owns one lease across clean/build/rustdoc/probe, builds types and
API examples in the same API feature profile, and records actual per-case
selection/execution/outcomes. It follows private report, tensor, and JSON envelope
mirrors and checks their concrete field/generic/serde relationships. Its cases
cover fixed-number admission (including nonnegative extents), report counters and
ordered parameters, exact JSON envelope versions/dispatch, source coordinates
and checked UTF-8 access, selected DAG/reference boundaries, and inference/result
identity widths. The canonical scalar/storage binary codec remains separately
executed; the raw-JSON envelope adapters do not claim bincode compatibility.

`serialization_candidates()` inventories exported serde types and aliases across
all public modules. A candidate is not proof of JSON publication. Imported types
need defining artifacts; an unknown custom serializer remains a candidate and
fails graph discovery. Public JSON protocol roots, private published manifest
roots, and internal cache products require explicit producer/consumer ownership
reconciliation in live integration. These fixtures and named API roots do not
establish complete publication-entry discovery, cache compatibility, Python
facade execution, source materialization, or final numeric authority.
