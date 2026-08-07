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

#### Scenario: A failed probe restores the specialization state

- **WHEN** a speculative probe encounters a lowering error after specialization
  state already exists
- **THEN** the probe reports no summary rejection and restores every state field
  to its pre-probe value
- **AND** later real lowering reports the genuine lowering error

### Requirement: Specialized symbols are compiler-internal

A monomorphized specialization is an implementation detail of one generic
definition. The host program SHALL carry explicit provenance for each function
through concrete-type and C-ABI projection. Symbol spelling SHALL NOT determine
that provenance. The published host header SHALL NOT declare a specialization.
Entry selection SHALL NOT choose a specialization as the preferred tensor
entry, and tensor-signature classification SHALL NOT count one. An authored
function whose valid name matches the specialization mangling grammar SHALL
remain authored surface. Every specialization declaration and definition SHALL
have translation-unit-local C linkage in binary mode and object mode. Authored
object-mode exports SHALL keep external linkage. Emitted C SHALL retain the
internal prototypes that specializations need for forward references.

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

#### Scenario: Authored mangling-shaped name stays public

- **WHEN** an authored function has a valid snake_case name that ends in
  `__mono_` plus 16 lowercase hexadecimal digits
- **THEN** the published header declares that function and entry selection
  treats it as authored

#### Scenario: Specializations do not collide across objects

- **WHEN** two generated object-mode C files contain the same specialization
  symbol
- **THEN** each specialization has translation-unit-local linkage and the two
  object files link into one relocatable object without a duplicate symbol

### Requirement: Symbolic tensor dimensions specialize from the call site

A specialization's parameter and result types SHALL be the call site's own
checked types: no separately instantiated signature exists for tensor
dimensions to disagree with, so a dimension disagreement SHALL NOT be
representable at the specialization boundary, and none may surface as an
internal error at a later stage. Symbolic and literal dimension spellings
SHALL specialize alike: a recursive generic instantiated at a
symbolic-dimension tensor payload compiles and runs.

#### Scenario: Symbolic-dim payload compiles with eval parity

- **WHEN** a recursive generic structure is instantiated at a `tensor[n, f32]`
  payload and built for a native target
- **THEN** the build succeeds, the binary links and runs, and its output
  matches the eval lane on the same program

#### Scenario: Dimension spellings do not fork specializations

- **WHEN** one recursive generic instantiation's payload tensor reaches the
  call with its dimensions spelled symbolically
- **THEN** the emitted C contains exactly one specialized definition for that
  instantiation, and no dimension-related internal error is emitted at any
  stage

### Requirement: One specialization per callee identity and instantiation

Specializations SHALL be interned by the callee's canonical definition
identity, not its source spelling. Resolution SHALL prefer an exact canonical
identity before a terminal-name fallback. A terminal-name fallback SHALL
succeed only when exactly one definition matches. Distinct spellings that
resolve to one definition SHALL share one specialization per instantiation.
Distinct definitions with one terminal name SHALL retain separate identities.
Distinct instantiations SHALL NOT share an emitted symbol. If symbol derivation
assigns one symbol to two distinct canonical signatures, lowering SHALL fail
loudly rather than emit a definition that serves either.

#### Scenario: Qualified and short spellings intern one symbol

- **WHEN** a reef'd package's recursive generic def is called through a
  qualified spelling and a short spelling at the same instantiation in one
  program
- **THEN** the emitted C contains exactly one specialized definition for that
  instantiation

#### Scenario: Package definitions with one terminal name stay distinct

- **WHEN** two package modules define recursive generic functions with one
  terminal name and callers use both canonical identities
- **THEN** native output matches eval and emitted C contains one distinct
  specialization for each definition

#### Scenario: Symbol collision fails loudly

- **WHEN** symbol derivation maps two distinct canonical signatures to one
  symbol
- **THEN** lowering fails with an internal diagnostic naming both signatures,
  and no C artifact is written
