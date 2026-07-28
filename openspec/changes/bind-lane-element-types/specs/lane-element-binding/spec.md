# lane-element-binding

## ADDED Requirements

### Requirement: A lane's element spelling comes from a type that knows its width
A backend SHALL obtain the source-level type name it emits for a dtype from a binding
that also names that dtype, rather than from a literal written at the emission site.

#### Scenario: A literal spelling is not available
- **WHEN** an emitter needs the C or MSL type name for a dtype
- **THEN** it reads it from the binding, and a hand-written string literal in that
  position does not compile

#### Scenario: The spelling follows the representation
- **WHEN** a dtype's representation changes
- **THEN** every lane's emitted type name changes with it, without edits at the emission
  sites

### Requirement: A spelling whose width disagrees with its dtype fails the build
The binding SHALL assert at compile time that the bound type's size equals the dtype's
declared width.

#### Scenario: A mismatched binding does not compile
- **WHEN** a type is bound to a dtype whose width differs from the type's size
- **THEN** the build fails, naming both

#### Scenario: Changing a width without changing its type fails everywhere at once
- **WHEN** a representation's width changes and the bound element type does not
- **THEN** every lane fails to build, rather than the lanes that derive their width
  succeeding while the lanes that restate it corrupt memory

#### Scenario: The assertion holds in release builds
- **WHEN** the workspace is built in release
- **THEN** the check still applies, because it is a compile-time assertion rather than a
  debug assertion

### Requirement: A dtype with no lane binding is a loud gap
A backend asked to emit an element type for a dtype it has no binding for SHALL fail with
the dtype and the backend named.

#### Scenario: An unsupported dtype names itself
- **WHEN** an emitter reaches a dtype with no binding
- **THEN** the failure names the dtype and the lane, rather than emitting a default type

#### Scenario: No wildcard spelling arm
- **WHEN** the binding lookup is written
- **THEN** it has no wildcard arm yielding a type name, because such an arm silently gave
  new dtypes whichever spelling happened to be the default
