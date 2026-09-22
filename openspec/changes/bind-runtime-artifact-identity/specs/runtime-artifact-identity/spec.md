## ADDED Requirements

### Requirement: Runtime compatibility has an independent build identity

Runtime-consuming compiler components SHALL carry an expected versioned runtime build
identity independently of candidate discovery. Compatibility SHALL require equality
of the runtime source/dependency/build-input closure, public-header/ABI identity,
target, effective runtime features, the compile configuration the runtime compilation
observes directly, and toolchain identity. Identity SHALL record observed configuration
effects, not the configuration files or profile names that produce them.
Relevant dirty and untracked inputs SHALL participate by content. Release versions,
filenames, paths, mtimes, and compiler image IDs SHALL NOT substitute for that identity.
The CLI and Python extension SHALL each carry their own expectation for the runtime
recipe they support; unrelated compiler-only features SHALL NOT alter that recipe.

This identity obligation complements the generated-artifact contract in
`spec/08-backends.md` §2 and compiled-loading contract in `spec/11-ffi.md` §1.4;
it does not redefine their language or callable-ABI semantics.

#### Scenario: Independently packaged matching components
- **WHEN** a compiler and runtime built in different directories carry the same runtime build identity
- **THEN** relocation and different compiler image bytes do not make that runtime incompatible

#### Scenario: Runtime inputs differ without a release bump
- **WHEN** runtime source, a transitive runtime dependency, headers, target, effective features, observed compile configuration, or code-affecting toolchain inputs differ
- **THEN** the differing runtime is incompatible even if release versions and filenames are unchanged

#### Scenario: Python and an adjacent CLI are different builds
- **WHEN** Python compiles a callable artifact while an unrelated CLI is installed nearby
- **THEN** runtime compatibility is determined by the extension's expectation, not the CLI or Python interpreter image

### Requirement: Runtime archives carry build-bound identity evidence

A runtime archive SHALL contain a versioned private identity record emitted by its
compilation. Identity production SHALL cover changes, additions, and deletions in the
declared build-input closure. Consumers SHALL decode the record without executing the
candidate and reject missing, duplicate, malformed, or unsupported records. Packaging
SHALL preserve and verify the record rather than stamp old bytes from current source.
The archive's full content digest SHALL be observed separately from build identity.

#### Scenario: A stamped archive is renamed or copied
- **WHEN** the same valid archive bytes are copied to a canonical distribution filename
- **THEN** their embedded build identity is preserved and remains independently verifiable

#### Scenario: An old or stripped archive has no usable record
- **WHEN** a candidate's record is missing, duplicated, malformed, unsupported, or removed by packaging
- **THEN** it cannot become a verified runtime through a filename, adjacent metadata file, or post-build source observation

### Requirement: Shared selection is independent of recency and enumeration

Production CLI staging, Python compiled execution, and runtime-consuming test and
benchmark harnesses SHALL use one owned identity selection policy. Selection SHALL
consider all candidates in the declared search scope, admit only verified matches,
and succeed only for one distinct matching archive-content identity. Byte-identical
copies SHALL count as one artifact. Distinct matching byte identities SHALL be an
ambiguity error. Canonical and hashed names SHALL have equal compatibility standing.
A malformed or incompatible candidate SHALL NOT displace a unique valid match.

#### Scenario: Correct runtime remains usable among stale candidates
- **WHEN** one valid matching archive is accompanied by incompatible archives and mtimes or enumeration order are changed
- **THEN** selection still succeeds with the same matching archive bytes

#### Scenario: No verified match or multiple distinct matches exist
- **WHEN** no verified archive matches, or more than one distinct matching content identity exists
- **THEN** selection fails before linking or reporting a usable compiled result and diagnoses the candidates and expected identity

#### Scenario: Canonical and hashed files are identical copies
- **WHEN** canonical and hashed candidate names contain identical valid matching bytes
- **THEN** they are one selectable artifact rather than a false ambiguity

### Requirement: Overrides restrict discovery without weakening verification

`CHELIS_RUNTIME_DIR` SHALL restrict discovery to the specified directory and SHALL NOT
waive identity validation. Harness `CHELIS_RUNTIME_LIB` pins SHALL restrict
discovery to one absolute regular-file target under the same identity rules. A harness
receiving both variables SHALL accept the pair only when the pinned file lies inside
the override directory and SHALL reject any other pair. Invalid, inaccessible,
empty, or incompatible explicit selections SHALL fail without default fallback.

#### Scenario: An explicit directory contains one compatible artifact
- **WHEN** the directory override contains one verified matching archive identity
- **THEN** that artifact is selected with directory-override provenance

#### Scenario: A bad override has a valid default elsewhere
- **WHEN** an explicit directory or harness file pin is invalid or incompatible and a default location contains a matching archive
- **THEN** selection fails rather than silently using the default

#### Scenario: A harness pin lies inside the override directory
- **WHEN** both variables are supplied to a harness and the pinned file is a regular file inside the override directory
- **THEN** that file is validated under the identity rules and selected with exact-file provenance

#### Scenario: Harness overrides conflict
- **WHEN** both variables are supplied to a harness and the pinned file is not inside the override directory
- **THEN** the harness rejects the request before staging or linking

### Requirement: Verified runtime bytes survive staging and native consumption

Staging SHALL verify that the copied archive bytes and embedded identity agree with
the selection, and that shipped headers match its public-header identity. Publication
SHALL be atomic with respect to a successful current-attempt artifact receipt. Changed
bytes or unresolved verification SHALL prevent usable-result publication. Python and
native harness linking SHALL consume the verified staged artifact explicitly, not
rediscover a library through an ambient linker search. Python failures SHALL use the
compiler/build/runtime error boundary in `spec/11-ffi.md` §1.

#### Scenario: The intended runtime is staged and linked
- **WHEN** a verified archive and matching headers are staged successfully and Python or a native harness links them
- **THEN** the linked runtime input is the verified staged byte identity

#### Scenario: A candidate changes during copying
- **WHEN** replacement or mutation makes the copied bytes differ from the verified selection
- **THEN** staging fails and publishes no successful current-attempt receipt or usable compiled result

#### Scenario: Archive and headers disagree
- **WHEN** the selected runtime identity does not bind the public runtime headers being emitted
- **THEN** compilation/staging rejects the bundle before claiming success

### Requirement: Receipts distinguish staging from linked execution

Successful staging SHALL persist a versioned receipt with expected and observed runtime
build identity, full archive digest, source/staged paths, selection mode, and compiler
image provenance or its explicit unavailability. The CLI SHALL also name the staged
archive in its build artifact report on stdout. Linking consumers SHALL bind the actual linker input
and resulting native artifact digest to the receipt. Cached Python artifacts and
subsequent loads of those artifacts SHALL verify the matching receipt and native bytes
before reuse; missing or stale evidence SHALL NOT be repaired by assuming compatibility.
A staged receipt SHALL NOT claim native linking or execution.

#### Scenario: CLI emits source artifacts only
- **WHEN** `chelis build` succeeds
- **THEN** its diagnostic and receipt identify the staged runtime without claiming that a native compiler or program ran

#### Scenario: A valid compiled Python artifact is reused
- **WHEN** its native bytes, runtime expectation, and persisted receipt match
- **THEN** reuse preserves the original verified runtime provenance

#### Scenario: Cached native bytes or evidence are replaced
- **WHEN** a cached artifact lacks its receipt or disagrees with the recorded native digest or required runtime identity
- **THEN** it is rejected or rebuilt through verified compilation before becoming usable

### Requirement: Source-based evidence cannot certify a stale correct runtime

Source-worktree consumers and source-certification harnesses SHALL validate the live
runtime input closure against their independently recorded expected build inputs.
Missing or changed required inputs SHALL fail freshness before a success claim.
Sealed distribution consumers SHALL use their packaged expectation without requiring
a checkout or inferring source identity from cwd. Source versus distribution mode
SHALL be explicit build provenance, not an automatic fallback on unavailable source.

#### Scenario: A source mutation is not rebuilt
- **WHEN** runtime inputs are mutated while an old consumer and stale correct archive remain available to a source-worktree or source-certification run
- **THEN** freshness fails and the old pair cannot provide passing evidence for the mutated source

#### Scenario: A consistently rebuilt mutant is executed
- **WHEN** a broken runtime and its consuming compiler are rebuilt from matching inputs while a stale correct archive remains present
- **THEN** the matching mutant is selected and its actual behavior is observed instead of being masked by the stale archive

#### Scenario: An installed package has no source checkout
- **WHEN** a sealed distribution contains a matching compiler or extension and runtime/header bundle
- **THEN** it works after relocation without probing an unrelated checkout or cwd

### Requirement: Packages and acceptance evidence cover the consuming boundary

Runtime-producing Cargo, Nix, release/container, and Python-wheel paths SHALL bind
compiler expectations to their actual runtime/header outputs. This supplements the
complete-toolchain/runtime requirements in
`openspec/specs/nix-package-outputs/spec.md`; package paths alone are not identity.
A Python wheel SHALL supply its matching runtime without requiring another CLI install.

Runtime-selection acceptance SHALL execute a CLI build/stage/native-link/run witness
and an actual Python compile/load/call witness, each with exact expected results and
artifact receipts. It SHALL include positive controls, incompatible/missing/ambiguous
selection, overrides, ordering permutations, and a real source-level broken-runtime
mutation with a stale correct archive present. Fixture archives SHALL come from their
exact producing build, not from the selector whose correctness is under test.
Acceptance SHALL name its consumer/configuration coverage and disposition every known
selector before claiming defect-class closure. Missing required evidence SHALL NOT
be a passing skip; accelerator execution and evidence produced only outside hosted
lanes SHALL be reported separately from hosted CPU/staging coverage. Schema
validation and documentation CI SHALL NOT imply runtime acceptance.

#### Scenario: Source-free distributions execute their own runtime
- **WHEN** installed toolchain and wheel witnesses run without a developer target directory or source checkout
- **THEN** they execute using their own verified runtime bundle and retain artifact receipts

#### Scenario: A permissive oracle hides the defect
- **WHEN** a mutation selects stale correct bytes, a positive witness only errors, or required CLI/Python execution is skipped
- **THEN** the runtime-selection acceptance suite fails rather than reporting successful defect closure

#### Scenario: CPU acceptance passes without accelerator execution
- **WHEN** CPU and artifact-staging witnesses pass but a hardware execution lane has not run
- **THEN** receipts distinguish the successful covered rows from the unrun accelerator rows and do not assert accelerator execution
