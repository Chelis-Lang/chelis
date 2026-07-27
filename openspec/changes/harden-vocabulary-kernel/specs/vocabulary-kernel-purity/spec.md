# Vocabulary Kernel Purity: harden-vocabulary-kernel

## ADDED Requirements

### Requirement: The dependency-bottom vocabulary crate is free of the standard library
The crate that owns Chelis' closed cross-boundary vocabularies SHALL compile without the Rust standard library. Its purity SHALL be a property the compiler enforces, not a claim stated in a doc comment. Where the crate needs a facility the standard library would supply, it SHALL take the `core` equivalent or do without.

#### Scenario: The crate declares and honors no_std
- **WHEN** the vocabulary crate is compiled
- **THEN** it is compiled without the standard library, and any reintroduction of a standard-library path fails the build rather than passing review

#### Scenario: A doc comment no longer carries the guarantee
- **WHEN** a reader consults the crate's stated purity claim
- **THEN** the claim is backed by a compiler-enforced attribute, and the prose does not assert a boundary wider than what is enforced

#### Scenario: The dependency graph stays empty
- **WHEN** the crate's manifest is inspected
- **THEN** it declares no dependencies, and adding one is a reviewable manifest change rather than an invisible transitive addition

### Requirement: The vocabulary decode path performs no allocation
Decoding a vocabulary symbol or an ABI tag SHALL NOT allocate. An error reporting an unrecognized input SHALL NOT take ownership of a heap copy of that input. The decode functions SHALL be total: every input SHALL produce either a recognized variant or a specific rejection, and no input SHALL produce a panic.

#### Scenario: An unknown symbol is rejected without allocating
- **WHEN** the effect-kind decoder receives a symbol outside the closed vocabulary
- **THEN** it returns a rejection that names the offending symbol without copying it to the heap

#### Scenario: Decoding cannot panic
- **WHEN** the decoder receives any input its signature admits, including a malformed or absent one
- **THEN** it returns a value, and no input reaches a panicking path

#### Scenario: The rejection remains diagnosable
- **WHEN** a caller receives a rejection for an unknown symbol
- **THEN** the caller can still render a diagnostic naming that symbol

### Requirement: Code generation does not live in the vocabulary crate
A function whose contract is to produce source text for another language SHALL NOT reside in the dependency-bottom vocabulary crate. The vocabulary crate SHALL expose the data such a generator reads; the generator SHALL reside on the shell side of the boundary, together with the test that compares its output against a checked-in artifact.

#### Scenario: The generator is relocated
- **WHEN** the vocabulary crate is inspected for functions that build source text
- **THEN** none are present, and the generator that formerly lived there resides with the component that owns the generated artifact

#### Scenario: The generated artifact still matches
- **WHEN** the relocated generator runs
- **THEN** its output is byte-identical to the checked-in header the runtime compiles against, as it was before relocation

#### Scenario: The vocabulary data remains the single source
- **WHEN** the relocated generator produces the artifact
- **THEN** every tag identifier, tag value, and byte width in the output derives from the vocabulary crate rather than being restated

### Requirement: ABI quantities do not use architecture-sized integers
A value that describes the Chelis runtime ABI SHALL be represented by a fixed-width integer type. It SHALL NOT be represented by a type whose width varies with the compilation target's pointer size, because the ABI it describes does not vary that way.

#### Scenario: The dtype byte width is fixed-width
- **WHEN** the byte width of a runtime dtype is requested
- **THEN** the returned type is fixed-width, and its value is identical on every supported target

#### Scenario: Consumers convert explicitly
- **WHEN** a consumer needs the byte width in a pointer-sized context, such as an allocation size
- **THEN** the conversion is written at the call site rather than being supplied by the vocabulary's return type

### Requirement: Lint configuration is declared, not inherited by default
The crate SHALL declare its lint configuration rather than relying on the toolchain's default groups. Lint classes relevant to its content SHALL be enabled explicitly, including those governing lossy numeric conversion and unchecked arithmetic. A lint class SHALL NOT be off merely because nothing declared it on.

#### Scenario: The crate declares its lints
- **WHEN** the crate's manifest is inspected
- **THEN** a lint configuration is present, and the classes it enables are visible without reading a toolchain default

#### Scenario: Lossy conversion and unchecked arithmetic are diagnosed
- **WHEN** source in the crate performs a numeric conversion that can lose a value, or arithmetic that can overflow
- **THEN** the configured lints diagnose it rather than passing silently

#### Scenario: A relaxation is explicit and justified
- **WHEN** a configured lint is relaxed for an item
- **THEN** the relaxation is scoped to that item and carries a recorded reason

### Requirement: The runtime dtype layout is asserted at compile time
The in-memory size of the runtime dtype vocabulary and the correspondence between each variant's tag value and its generated C constant SHALL be asserted at compile time. A change to the representation SHALL fail the build rather than a test.

#### Scenario: A representation change fails the build
- **WHEN** the vocabulary's representation attribute or size changes
- **THEN** compilation fails

#### Scenario: A discriminant change fails the build
- **WHEN** a variant's tag value diverges from the constant emitted for it in generated code
- **THEN** compilation fails

### Requirement: The closed-vocabulary invariant is enforced semantically
The requirement that a closed vocabulary have no fallback variant SHALL be enforced by a rule that understands Rust's structure, not by matching text against source files. A dispatch over a closed vocabulary SHALL NOT use a catch-all arm. Where a source-text assertion is retained because no semantic rule subsumes it, the retention SHALL be recorded with the reason.

#### Scenario: A catch-all over a closed vocabulary is rejected
- **WHEN** a dispatch over a closed vocabulary enum uses a top-level catch-all pattern
- **THEN** the enforcing rule rejects it, and adding a variant to the vocabulary reopens every affected dispatch

#### Scenario: The rule is not defeated by formatting
- **WHEN** a consumer that satisfies the invariant is reformatted, or a comment is added containing the matched text
- **THEN** the enforcement result is unchanged, because the rule inspects structure rather than characters

#### Scenario: Superseded text assertions are removed, not duplicated
- **WHEN** a semantic rule subsumes an existing source-text assertion
- **THEN** that assertion is removed, and the change records which rule replaced it

#### Scenario: Surviving text assertions are justified
- **WHEN** a source-text assertion is retained after the semantic rule is adopted
- **THEN** the reason no semantic rule covers it is recorded alongside it
