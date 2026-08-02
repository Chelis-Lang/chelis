## ADDED Requirements

### Requirement: Proof-bearing edit success
The edit API SHALL return an opaque `ValidatedModule` only after type, effect, and linearity checks accept the complete edited module. Only the compiler-owned whole-module check function SHALL construct this type.

`ReplacementReport` SHALL contain the proof type and SHALL NOT contain `checks_clean` or a public raw module field. The proof type SHALL provide read-only and consuming access to its Deep expressions.

#### Scenario: A valid edit returns proof
- **WHEN** the whole-module check accepts an edited module
- **THEN** the result contains a `ValidatedModule` whose expressions equal the checked module

#### Scenario: A rejected edit returns no proof
- **WHEN** type, effect, or linearity checks reject an edited module
- **THEN** the edit API returns the existing tagged error and no `ValidatedModule`

#### Scenario: A caller tries to forge success
- **WHEN** a compile-fail fixture constructs `ValidatedModule` or a false successful `ReplacementReport`
- **THEN** Rust rejects the fixture because the proof constructor is private

### Requirement: Typed and aligned root artifacts
The pipeline SHALL use separate opaque types for all root names, tensor root names, declared tensor outputs, and forward node aliases.

These types SHALL use `IrName` instead of untyped `String` keys at the pipeline boundary.

Every successful DAG-backed construction SHALL create `NamedRoots` through one smart constructor. This constructor MUST require exact positional alignment between `TensorRootNames` and `Dag::roots()`.

A successful empty DAG SHALL use exact alignment in strict and host-only modes. If tensor root names exist, construction SHALL return `PipelineRejection::RootCount`.

A selected `AllowHostBackend` policy SHALL permit a successful empty DAG. This explicit host result SHALL use the empty constructor because the host lane emits the output.

An accepted nonfatal lower rejection SHALL also use the empty constructor. No constructor SHALL truncate a mismatched pair with `zip`.

`LoweredCompilation::into_parts` SHALL return a named `LoweredParts` product. It SHALL NOT return adjacent root maps in a tuple.

#### Scenario: Tensor names align with DAG roots
- **WHEN** a tuple-valued tensor output lowers to two DAG roots
- **THEN** `NamedRoots` binds each `IrName` to the root at the same position

#### Scenario: Root counts differ
- **WHEN** normal DAG construction receives different tensor-name and DAG-root counts
- **THEN** construction returns the existing root-count rejection before it creates `LoweredCompilation`

#### Scenario: A consumer swaps root map roles
- **WHEN** a compile-fail fixture passes `ForwardNodeIndex` where `NamedRoots` is required
- **THEN** Rust rejects the fixture because the map types differ

#### Scenario: A consumer decomposes lowered output
- **WHEN** a consumer takes ownership of `LoweredCompilation`
- **THEN** it receives named `LoweredParts` fields for the checked state, DAG, declared roots, and forward index

#### Scenario: A successful empty DAG has no tensor names
- **WHEN** normal lowering succeeds with an empty DAG and no tensor root names
- **THEN** the exact constructor returns an empty `NamedRoots`

#### Scenario: A strict successful empty DAG loses tensor roots
- **WHEN** strict lowering succeeds with an empty DAG but `TensorRootNames` is not empty
- **THEN** construction returns `PipelineRejection::RootCount`

#### Scenario: The selected host backend emits a tensor-typed output
- **WHEN** `AllowHostBackend` receives a successful empty DAG for a host-lane output
- **THEN** the explicit host result contains empty `NamedRoots` and preserves the C build output

#### Scenario: An accepted nonfatal rejection has no DAG roots
- **WHEN** the selected host policy accepts a nonfatal lower rejection and creates an empty DAG
- **THEN** `NamedRoots` is empty and public output remains equal to the current output

### Requirement: Exclusive semantic outcomes
`complete_checks` SHALL return a narrow `SemanticRejection` that contains only effect or linearity errors. The full pipeline boundary SHALL convert this rejection into `PipelineRejection`.

`LayeredCheck` SHALL represent clean, effect-rejected, and linearity-rejected states as exclusive enum variants. No variant SHALL contain both effect and linearity error collections.

#### Scenario: Complete semantic checks accept
- **WHEN** effect and linearity checks accept a typed program
- **THEN** `complete_checks` returns `CheckedCompilation` without an error variant

#### Scenario: Effect checks reject
- **WHEN** effect checks reject a typed program
- **THEN** `SemanticRejection` and `LayeredCheck` expose only effect errors

#### Scenario: Linearity checks reject
- **WHEN** effects accept and linearity checks reject a typed program
- **THEN** `SemanticRejection` and `LayeredCheck` expose only linearity errors

#### Scenario: A caller creates two layered error classes
- **WHEN** a compile-fail fixture tries to create one layered result with both error classes
- **THEN** Rust rejects the fixture because no enum variant has that shape

### Requirement: Typed AST integration at the target branch
A pipeline migration SHALL compile against the target branch's current typed Deep AST.

It SHALL use current typed name variants or typed accessors. It SHALL NOT restore a removed raw AST variant.

Conflict resolution SHALL preserve existing target-branch compiler observations. This change SHALL retain the #912 realizability manifest observation around the migrated compile path.

The migration SHALL preserve target-branch typed `.dp` ingestion and the typed `Node` wire bridge. It SHALL preserve property verification and producer obligations for typed `Node` input.

Strict Deep parsing SHALL reject unknown tags and malformed known tags after lenient stamping produces a typed fallback.

Lenient Deep fragment parsing SHALL retain malformed structures for checker diagnostics. It SHALL still reject a function node without a body.

Typed Deep parsing SHALL accept canonical parameter lists in `deftype` and `typealias` declarations.

Typed macro expansion SHALL accept the compiler-internal `defmacro` fallback. It SHALL resolve macro calls whose variable is a typed `Node`.

The Deep validator SHALL reject reopened modules and reserved linker names in typed `Node` input.

Deep lint and trace consumers SHALL visit metadata and children in typed `Node` input.

Compiler API authoring SHALL accept typed `Node` declarations. It SHALL insert them at the correct declaration index in typed modules.

Opaque-value decoding SHALL collect field types, invariants, and constants from typed `Node` input.

Root collection SHALL read tags, metadata, and children from both transitional `List` values and typed `Node` values.

#### Scenario: The implementation uses a removed AST variant
- **WHEN** the rebased pipeline refers to `Atom::Symbol` after the target branch replaced it with `Atom::Name`
- **THEN** the target build rejects the change before acceptance evidence is recorded

#### Scenario: A compiler conflict overlaps an observation hook
- **WHEN** conflict resolution changes the compile path that contains the #912 realizability manifest observation
- **THEN** the resolved path retains that observation and its existing tests

#### Scenario: Typed Deep ingestion rejects a bare runtime name
- **WHEN** `.dp` input contains a bare name in a runtime expression position
- **THEN** the CLI returns the target typed-parser error and does not format the input

#### Scenario: Strict Deep parsing receives an unknown tag
- **WHEN** `.dp` input contains a tagged form outside the closed vocabulary
- **THEN** strict parsing rejects the input with the existing unknown-tag diagnostic

#### Scenario: Strict Deep parsing receives a malformed known tag
- **WHEN** `.dp` input omits metadata from a tag in the closed vocabulary
- **THEN** strict parsing rejects the input with the existing metadata diagnostic

#### Scenario: Lenient parsing receives a malformed checker fixture
- **WHEN** a checker test parses a malformed type or expression fragment
- **THEN** the checker receives the original structure and emits its existing diagnostic

#### Scenario: Lenient parsing receives a function without a body
- **WHEN** a function node contains parameters but no body
- **THEN** parsing rejects the node before a path operation can use it

#### Scenario: Typed Deep parsing receives a generic ADT
- **WHEN** a canonical `deftype` contains a parameter list and variants
- **THEN** typed parsing preserves the parameter list and the type checker accepts valid construction

#### Scenario: Typed macro expansion receives an internal definition
- **WHEN** lenient parsing produces a `defmacro` fallback and a typed `Node` call
- **THEN** macro expansion substitutes the call and preserves call-site module attribution

#### Scenario: Typed Deep input reaches root collection
- **WHEN** a typed `Node` module contains a tuple-valued root declaration
- **THEN** root collection returns the same ordered names as the transitional `List` form

#### Scenario: Typed Deep input crosses the wire boundary
- **WHEN** a typed `Node` crosses the compiler API wire adapter
- **THEN** the wire value preserves its tag, metadata, binders, and syntax children

#### Scenario: A typed Deep property reaches the shared property runner
- **WHEN** the CLI sends a typed `Node` property to the shared property runner
- **THEN** the runner verifies the property and rejects malformed property metadata

#### Scenario: A typed Deep opaque producer reaches verification
- **WHEN** a typed `Node` module contains an invariant and an exported producer
- **THEN** constructor generation and producer obligations retain their positive and negative results

#### Scenario: Typed Deep input forges module identity
- **WHEN** a typed `Node` file reopens a module or uses a reserved linker name
- **THEN** Deep validation rejects the file with the existing identity diagnostic

#### Scenario: Typed Deep input constructs an opaque value outside its module
- **WHEN** a typed `Node` file directly constructs or updates an opaque domain value
- **THEN** the blocking Deep lint rejects the file

#### Scenario: Typed Deep input carries trace spans
- **WHEN** a typed `Node` fixture carries span IDs in nested nodes
- **THEN** trace collection returns every ID that the sidecar declares

#### Scenario: Typed Deep authoring adds a function
- **WHEN** compiler API authoring appends a typed declaration or inserts it after a target function
- **THEN** the result preserves the declaration order and passes the complete module check

#### Scenario: Typed Deep authoring receives a bad declaration
- **WHEN** compiler API authoring receives a declaration with an invalid shape
- **THEN** it rejects the request before it edits the module

#### Scenario: Typed Deep decode receives an opaque value
- **WHEN** a typed `Node` program declares an opaque type, an invariant, and a constant
- **THEN** decode accepts a valid value and rejects a value that violates the invariant

#### Scenario: Typed Deep decode receives malformed invariant metadata
- **WHEN** a typed `Node` program declares malformed invariant metadata
- **THEN** decode rejects the value as an invariant failure

### Requirement: Authoritative artifact-type oracle
The existing `.venv/bin/python scripts/compiler_pipeline_oracle.py` command SHALL remain the authoritative completion oracle. It SHALL run the proof, root, semantic-outcome, checkpoint, consumer-parity, and compile-fail tests.

The oracle SHALL run after the implementation compiles against the target branch. Stale-branch results SHALL NOT count as final acceptance evidence.

#### Scenario: All artifact boundaries hold
- **WHEN** the authoritative oracle runs after all consumers migrate
- **THEN** all positive tests, negative tests, compile-fail tests, and parity tests pass

#### Scenario: A weak artifact shape returns
- **WHEN** a negative fixture restores a Boolean success marker, raw root tuple, or parallel layered error vectors
- **THEN** the authoritative oracle fails and identifies the broken boundary

### Requirement: Continuous artifact compile-fail enforcement
The canonical per-PR gate SHALL run `cargo test -p chelis-compiler-api --doc`.

The same gate SHALL run `.venv/bin/python scripts/check_checkpoint_compile_fail.py`. Command-list unit tests SHALL NOT substitute for either executable control.

#### Scenario: Hosted CI checks compile-time artifact boundaries
- **WHEN** hosted CI runs the `lint-and-unit` gate stage
- **THEN** it executes the compiler-API doctests and the raw-checkpoint compile-fail fixture

## MODIFIED Requirements

### Requirement: Canonical production pipeline owner
`chelis-compiler-api` SHALL own production orchestration in the upper-consumer scope. This scope contains `chelis-compiler-api`, `chelis-cli`, and `chelis-e2e`.

Production functions in this scope that need two or more semantic stages SHALL delegate to this owner. Lower-layer crates SHALL retain individual stage implementations.

`chelis-reef` package artifact construction SHALL remain one explicit dependency exception. It can run type, effect, and linearity stages directly because `chelis-compiler-api` depends on `chelis-reef`.

Issue #1012 owns removal of this exception through a dependency-bottom pipeline core. No other production crate SHALL use this exception.

The source architecture guard SHALL inspect production Rust files under exactly these roots:

- `crates/chelis-compiler-api/src`
- `crates/chelis-cli/src`
- `crates/chelis-e2e/src`

The guard SHALL NOT claim coverage for all workspace production files.

The source architecture guard SHALL detect direct stage sequences. It SHALL also detect equivalent sequences composed through local helper calls.

It SHALL use one inventory for all guarded files. It SHALL propagate reachable stages through each crate-local call graph until the stage sets reach a fixed point.

The guard SHALL preserve local call-site multiplicity. It SHALL resolve local calls through path prefixes, named imports, glob imports, and local function-value aliases.

Alias bindings SHALL remain specific to one execution path and lexical scope. Typed, parenthesized, and assigned callable values SHALL use the same resolver.

Callable results from `if` and `match` expressions SHALL remain path-specific. Known callable elements in an array iterable SHALL bind to their exact iteration.

If a local helper invokes a callable parameter, the guard SHALL substitute the caller's callable argument into that helper path.

The guard SHALL inspect implementation methods and trait default methods. An invoked local macro SHALL resolve direct canonical calls and canonical imports.

A local macro identity SHALL include its lexical block scope. An inner definition SHALL NOT replace an outer definition after the inner block ends.

The guard SHALL track executable paths. Return, break, and continue SHALL isolate unreachable statements from their paths.

An uninvoked closure or nested function body SHALL NOT contribute stages to its enclosing function.

The guard SHALL classify a canonical stage by its owning module and function identity. A shared final function name SHALL NOT change that identity.

The guard SHALL resolve a crate-local receiver method from a typed local receiver. It SHALL preserve callable argument positions for method and qualified-method syntax.

The guard SHALL preserve the zero-iteration exit path for `while` and `for` loops. A return from the loop body SHALL terminate only the body path.

The guard SHALL calculate a fixed point across repeated loop iterations. It SHALL preserve labeled break and continue targets.

The guard SHALL retain stable Boolean facts across loop iterations. It SHALL preserve exact values and counts for syntax-known array iterables.

Pattern bindings SHALL shadow outer callable aliases in `if let`, `while let`, `match`, and `for` scopes.

The guard SHALL preserve full import targets. These targets SHALL include `extern crate` aliases.

The guard SHALL resolve local type aliases before receiver method lookup.

It SHALL NOT classify an unrelated receiver, import, type alias, or external call only from its final name.

A focused helper that reaches only one semantic stage SHALL remain valid. A production function that reaches two or more stages through helpers SHALL be rejected.

#### Scenario: Production consumer delegates
- **WHEN** a CLI, edit, compiler-API, or E2E path needs a full semantic check
- **THEN** the path requests a compiler-API pipeline goal instead of calling the stages in sequence

#### Scenario: Production consumer duplicates the sequence
- **WHEN** a guarded production file outside the owner orchestrates two or more canonical semantic stages
- **THEN** the source architecture guard rejects the file and identifies the duplicated stage calls

#### Scenario: Reef package artifact path uses the dependency exception
- **WHEN** `chelis-reef` checks a fully linked package before artifact or schema output
- **THEN** the documented dependency exception permits its direct type, effect, and linearity sequence

#### Scenario: Production consumer composes stage helpers
- **WHEN** a production function calls separate local helpers that reach type and effect stages
- **THEN** the source architecture guard rejects the orchestrating function and identifies both reachable stages

#### Scenario: Repeated helper calls reach different stages
- **WHEN** one production path calls a conditional helper twice and the two call sites reach different stages
- **THEN** the source architecture guard rejects the caller and identifies both stages

#### Scenario: A qualified local helper composes stages
- **WHEN** a production function calls a local helper through a qualified path and directly reaches another stage
- **THEN** the source architecture guard rejects the function and identifies both stages

#### Scenario: Helpers in separate files compose stages
- **WHEN** a caller reaches one stage through a helper from another guarded file and reaches a second stage directly
- **THEN** the source architecture guard rejects the caller and identifies both stages

#### Scenario: A local alias composes stages
- **WHEN** a caller reaches one stage through an imported helper alias or local function-value alias
- **THEN** the source architecture guard resolves the alias before it calculates the caller's stage set

#### Scenario: A callable alias uses another valid Rust form
- **WHEN** a typed, parenthesized, or branch-assigned callable value selects a canonical stage
- **THEN** the source architecture guard resolves that selected stage on each execution path

#### Scenario: A local binding shadows a callable alias
- **WHEN** an inner lexical block binds an unrelated callable under the same name as an outer stage alias
- **THEN** the source architecture guard uses the inner binding only inside that block

#### Scenario: A branch expression returns a callable
- **WHEN** an `if` or `match` expression selects a canonical stage callable
- **THEN** the source architecture guard retains the selected callable on each applicable path

#### Scenario: An array iterable contains stage callables
- **WHEN** a `for` loop invokes known callable elements from an array iterable
- **THEN** the source architecture guard composes the stages in exact iteration order

#### Scenario: A higher-order helper invokes its parameter
- **WHEN** a local helper invokes a callable parameter that receives a canonical stage argument
- **THEN** the source architecture guard adds that stage to the caller's path

#### Scenario: A higher-order helper ignores its parameter
- **WHEN** a local helper receives but does not invoke a canonical stage argument
- **THEN** the source architecture guard does not add that stage to the caller's path

#### Scenario: A trait default method duplicates stages
- **WHEN** a trait default method directly reaches two canonical semantic stages
- **THEN** the source architecture guard rejects the method and identifies both stages

#### Scenario: Two stage modules export one function name
- **WHEN** type and effect modules export canonical functions with the same final name
- **THEN** the source architecture guard classifies each function from its complete module identity

#### Scenario: A typed receiver invokes a higher-order method
- **WHEN** a local receiver method invokes a callable parameter with a canonical stage argument
- **THEN** the source architecture guard adds that stage for method and qualified-method syntax

#### Scenario: A local type alias names a receiver
- **WHEN** a typed receiver uses a local alias for a crate-local method owner
- **THEN** the source architecture guard resolves the alias before method lookup

#### Scenario: An unrelated typed receiver uses a local method name
- **WHEN** an external receiver type uses the same method name as a crate-local helper
- **THEN** the source architecture guard does not resolve that call to the local helper

#### Scenario: A loop body can run zero times
- **WHEN** a `while` or `for` body returns but the loop can run zero times
- **THEN** the source architecture guard preserves the exit path and analyzes later stage calls on it

#### Scenario: Repeated loop iterations compose stages
- **WHEN** separate iterations can select body paths that reach different canonical stages
- **THEN** the source architecture guard rejects the function and identifies both stages

#### Scenario: A loop branch uses a stable Boolean
- **WHEN** an immutable Boolean selects one semantic stage on every iteration
- **THEN** the source architecture guard does not create a path through the unselected branch

#### Scenario: An array loop has zero or one iteration
- **WHEN** an array iterable contains zero or one stage-bearing element
- **THEN** the source architecture guard does not compose stages from impossible extra iterations

#### Scenario: A loop control expression terminates a path
- **WHEN** break or continue precedes another stage call in the same loop body path
- **THEN** the source architecture guard excludes the unreachable stage call from that path

#### Scenario: A labeled break targets an outer loop
- **WHEN** a nested loop breaks to an outer label
- **THEN** the source architecture guard terminates the applicable outer body path

#### Scenario: A pattern shadows a callable alias
- **WHEN** a branch or loop pattern reuses the name of an outer callable alias
- **THEN** the source architecture guard uses the pattern binding only in its lexical scope

#### Scenario: An invoked local macro contributes a stage
- **WHEN** a local macro calls a canonical stage through a direct path or imported alias
- **THEN** the source architecture guard includes that stage in the caller's execution path

#### Scenario: A nested macro shadows an outer macro
- **WHEN** an inner block defines a macro with the same name as an outer macro
- **THEN** each call resolves the definition from its lexical block

#### Scenario: An extern crate alias names a canonical module
- **WHEN** `extern crate` renames a canonical stage module
- **THEN** the source architecture guard preserves the canonical import target

#### Scenario: A return terminates one branch
- **WHEN** one branch reaches a stage and returns before a later stage call
- **THEN** the source architecture guard does not combine the two stages on that branch

#### Scenario: A callable body is not invoked
- **WHEN** a function defines a closure or nested function but does not invoke it
- **THEN** the source architecture guard does not add that callable body's stages to the function

#### Scenario: A local callable body is invoked
- **WHEN** a function invokes a local closure or nested function that reaches a canonical stage
- **THEN** the source architecture guard adds that stage to the caller's execution path

#### Scenario: An unrelated receiver or import uses a stage name
- **WHEN** an unrelated receiver or external import has the same final name as a canonical semantic stage
- **THEN** the source architecture guard does not classify that call as the stage

#### Scenario: Focused helper calls one stage
- **WHEN** a production helper reaches only one canonical semantic stage
- **THEN** the source architecture guard accepts the helper unless another caller composes it with a second stage
