# Tensor Storage Encapsulation: seal-tensor-data-pointer

## ADDED Requirements

### Requirement: A tensor's storage pointer is reachable only through accessors
The field holding a tensor's storage pointer SHALL NOT be accessible outside the module
that defines it. Code needing the buffer SHALL obtain it through an accessor. A cast of
the storage pointer to an element type SHALL NOT be expressible outside that module.

#### Scenario: A direct cast outside the owning module fails to compile
- **WHEN** code outside the owning module casts the storage pointer to an element type
- **THEN** compilation fails, rather than the cast being permitted and checked by convention

#### Scenario: The compiler is the enforcement, not review
- **WHEN** a contributor adds a new access site
- **THEN** the only forms that compile are the accessors, so a wrong form cannot reach review

#### Scenario: A comment cannot authorise a direct cast
- **WHEN** documentation asserts that a particular direct decode is correct
- **THEN** that assertion cannot be acted on outside the owning module, because the cast does not compile

### Requirement: The C ABI is unaffected by encapsulation
Restricting host-language access to the storage pointer SHALL NOT change the tensor
struct's layout, field order, or size, and SHALL NOT change the generated or checked-in C
header. Foreign code SHALL continue to access the field through its own declaration.

#### Scenario: Layout is unchanged
- **WHEN** the tensor struct's size, alignment, and field offsets are compared before and after
- **THEN** they are identical

#### Scenario: The C header is unchanged
- **WHEN** the checked-in and generated C headers are compared before and after
- **THEN** they are byte-identical

#### Scenario: Foreign access still works
- **WHEN** compiled foreign code reads or writes the storage pointer through its own declaration
- **THEN** it behaves as before, because host-language visibility does not affect the ABI

### Requirement: Typed element access is the ordinary path and raw byte access is deliberate
An accessor that yields a pointer to a named element type SHALL check, or have already
established, that the tensor's dtype tag matches that element type. A raw-bytes accessor
SHALL exist only for uses that are genuinely untyped, SHALL be documented as unsuitable
for element access, and SHALL NOT be more convenient than the typed path.

#### Scenario: Typed access establishes the dtype
- **WHEN** element access is obtained for a named element type
- **THEN** the tensor's dtype tag has been checked against that type, or the check is discharged by the caller under a stated obligation

#### Scenario: Raw bytes are for untyped uses only
- **WHEN** a caller needs the buffer for a byte copy, an allocation, or a release
- **THEN** the raw-bytes accessor serves it, and its documentation states that element access is not among its uses

#### Scenario: The safe path is the easy path
- **WHEN** a contributor reads an element and reaches for the most convenient accessor
- **THEN** that accessor is the typed one

#### Scenario: An element type without a typed accessor is decided, not defaulted
- **WHEN** a dtype's storage has no element type providing typed access
- **THEN** whether to introduce one or to route it through raw bytes is recorded as a decision, rather than falling to raw bytes by omission

### Requirement: The migration cannot ship partially
Sealing the field SHALL occur only when no access outside the owning module remains, and
SHALL be a change that does nothing else. Each preceding batch SHALL be independently
revertable and SHALL NOT change behavior.

#### Scenario: The seal is its own change
- **WHEN** the field is sealed
- **THEN** that change contains no other edit, so its successful compilation is the evidence that no site was missed

#### Scenario: Intermediate states are correct
- **WHEN** the migration is partially complete
- **THEN** the code is correct but inconsistent, because every accessor introduced is a typed spelling of a cast that already existed

#### Scenario: A batch is revertable alone
- **WHEN** any pre-seal batch is reverted
- **THEN** the result compiles and behaves as it did before that batch

#### Scenario: A behavioral difference is a finding
- **WHEN** migrating a site changes what it computes
- **THEN** it is reported as a defect and handled separately, rather than being absorbed into the migration
