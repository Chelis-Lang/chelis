# typed-tensor-access

## ADDED Requirements

### Requirement: A tensor handle names the element type it stores
Internal runtime code SHALL hold tensors through a handle parameterised by an element
type implementing `TensorElement`, so that two tensors of different dtypes have different
Rust types.

#### Scenario: Handles of different dtypes do not unify
- **WHEN** a function expects a handle whose element type is `f32`
- **AND** it is passed a handle whose element type is `Bool8`
- **THEN** the program does not compile

#### Scenario: The wrong-width accessor is unreachable
- **WHEN** code holding a bool tensor handle attempts to obtain an `f32` element pointer
- **THEN** the program does not compile, rather than reading four bytes per one-byte slot

#### Scenario: The handle carries no runtime cost
- **WHEN** the handle's size and alignment are compared to a raw tensor pointer
- **THEN** they are identical

### Requirement: Obtaining a typed handle checks the dtype exactly once
Construction of a typed handle from an untyped tensor pointer SHALL verify that the
tensor's dtype tag matches the handle's element type, and SHALL report a mismatch rather
than producing a handle.

#### Scenario: A mismatched dtype is refused at construction
- **WHEN** a handle for element type `T` is constructed from a tensor whose tag names a
  different dtype
- **THEN** construction fails with a dtype mismatch naming both the expected and actual
  dtype

#### Scenario: Element access performs no further dtype check
- **WHEN** an element pointer is taken from a successfully constructed handle
- **THEN** no dtype comparison occurs, because the handle's existence is the evidence

#### Scenario: The check is present in release builds
- **WHEN** the constructor runs in a release build
- **THEN** the dtype is still verified, rather than being elided as a debug-only assertion

### Requirement: The runtime dtype tag becomes a static element type in exactly one place
The conversion from a runtime dtype tag to a statically known element type SHALL happen
through a single dispatch entry point, and that dispatch SHALL be exhaustive over the
closed dtype vocabulary.

#### Scenario: A missing dtype arm fails the build
- **WHEN** a dtype is added to the runtime vocabulary and the dispatch is not extended
- **THEN** the program does not compile

#### Scenario: A hand-written match cannot reach typed access
- **WHEN** code matches on a dtype tag outside the dispatch and attempts to obtain typed
  element access
- **THEN** the program does not compile, because the dispatch is the only constructor of
  a typed handle

#### Scenario: An unsupported dtype is refused loudly
- **WHEN** an operation that supports a subset of dtypes is invoked with one outside that
  subset
- **THEN** it fails with a message naming the operation and the dtype, rather than
  decoding the buffer at the wrong width

### Requirement: The foreign function surface is unchanged
Introducing typed handles SHALL NOT change any exported symbol, struct layout, or
generated header, and foreign callers SHALL be unaffected.

#### Scenario: Exported signatures are unchanged
- **WHEN** the exported C functions are compared before and after
- **THEN** their names and parameter types are identical

#### Scenario: The C headers are unchanged
- **WHEN** the checked-in and generated C headers are compared before and after
- **THEN** they are byte-identical

#### Scenario: Conversion happens inside the boundary
- **WHEN** an exported function receives an untyped tensor pointer
- **THEN** it converts to a typed handle within its own body, rather than requiring the
  caller to supply one

### Requirement: Migration cannot silently half-land
Because a partially migrated tree compiles and passes tests, the change SHALL make
completion mechanically checkable rather than a matter of review.

#### Scenario: Remaining untyped access is enumerable
- **WHEN** the migration is in progress
- **THEN** the count of unconverted internal access sites is reported by a command, not
  estimated

#### Scenario: No byte layout changes during migration
- **WHEN** any subset of sites has been converted
- **THEN** every buffer is written and read at the same offsets and widths as before, so
  an incomplete migration is inconsistent rather than memory-unsafe

#### Scenario: The untyped accessor is removed, not deprecated
- **WHEN** the migration completes
- **THEN** the untyped element accessors no longer exist, so a later site cannot reach
  for one
