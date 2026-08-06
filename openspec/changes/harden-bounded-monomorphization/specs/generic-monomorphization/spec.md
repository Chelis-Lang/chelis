# Generic Monomorphization: harden-bounded-monomorphization

## ADDED Requirements

### Requirement: Specialization emission is deterministic

Repeated lowering of one program SHALL produce byte-identical native artifacts.
Speculative lowering performed to answer a question and then discarded — a
probe — SHALL leave no observable trace on specialization state: it SHALL NOT
add a specialized definition to the emitted program, SHALL NOT reorder the
emitted definitions, and a failure inside probe-only specialization work SHALL
NOT fail the build. Every emitted specialization SHALL be one a surviving call
site references.

#### Scenario: Byte-identical emitted C across repeated builds

- **WHEN** `chelis build` compiles the same program repeatedly, including a
  program whose source order places a probed caller before the definitions it
  wraps
- **THEN** every build's emitted C artifact is byte-identical to the first

#### Scenario: Probe-only specializations are not emitted

- **WHEN** a speculative probe lowers a body containing a recursive generic
  call whose instantiation no surviving call site uses
- **THEN** the emitted C contains no specialized definition for that
  instantiation

### Requirement: Specialized symbols are compiler-internal

A monomorphized specialization is an implementation detail of one generic
definition: its name is derived from an interned signature and changes whenever
the program's instantiation set does. The published host header SHALL NOT
declare a specialized symbol. Entry selection SHALL NOT choose a specialized
symbol as the preferred tensor entry, and tensor-signature classification SHALL
NOT count one, so the presence of a tensor-typed specialization SHALL NOT
change the authored entry's compiled ABI. Emitted C SHALL retain whatever
internal prototypes are needed for specializations to reference definitions
emitted later.

#### Scenario: Published header omits specializations

- **WHEN** `chelis build` succeeds on a program containing recursive generic
  host calls
- **THEN** the emitted header declares the authored surface only, with no
  specialized symbol, and the emitted C still compiles and links cleanly

#### Scenario: Authored entry keeps its ABI

- **WHEN** a program's authored preferred tensor entry coexists with a
  specialization whose signature is also tensor-shaped
- **THEN** the compiled entry ABI is the authored entry's, unchanged from the
  same program with the generic calls removed

### Requirement: Symbolic tensor dimensions are diagnosed, not deferred

A checked type application whose terms are concrete may still disagree with its
call site in tensor dimensions, because dimension spellings are not type-term
slots. Before emitting a specialization, lowering SHALL compare the
instantiated parameter and result types against the types the call supplies,
tensor shapes included. A disagreement SHALL reject through the
unsupported-case response contract (`spec/05-risc-primitives.md` §7,
[05-UNS-1..5]), naming the instantiated type and the supplied type, and SHALL
NOT surface as an internal error at a later stage. Agreement — literal or
symbolic dimension spellings alike — SHALL specialize: a recursive generic
instantiated at a symbolic-dimension tensor payload compiles and runs.

#### Scenario: Symbolic-dim payload compiles with eval parity

- **WHEN** a recursive generic structure is instantiated at a `tensor[n, f32]`
  payload and built for a native target
- **THEN** the build succeeds, the binary links and runs, and its output
  matches the eval lane on the same program

#### Scenario: Dimension disagreement rejects cleanly

- **WHEN** a recursive generic call's instantiated signature disagrees with the
  call's supplied tensor dimensions
- **THEN** the build fails with a branded `unsupported:` diagnostic naming both
  types, no internal-error text is emitted, and no C artifact is written

### Requirement: One specialization per callee identity and instantiation

Specializations SHALL be interned by the callee's canonical definition
identity, not its source spelling: distinct spellings that resolve to one
definition (a package-qualified and a short reference to the same reef'd def)
SHALL share one specialization per instantiation. Distinct instantiations SHALL
NOT share an emitted symbol; if symbol derivation would assign one symbol to
two distinct canonical signatures, lowering SHALL fail loudly rather than emit
a definition that serves either.

#### Scenario: Qualified and short spellings intern one symbol

- **WHEN** a reef'd package's recursive generic def is called through a
  qualified spelling and a short spelling at the same instantiation in one
  program
- **THEN** the emitted C contains exactly one specialized definition for that
  instantiation

#### Scenario: Symbol collision fails loudly

- **WHEN** symbol derivation maps two distinct canonical signatures to one
  symbol
- **THEN** lowering fails with an internal diagnostic naming both signatures,
  and no C artifact is written
