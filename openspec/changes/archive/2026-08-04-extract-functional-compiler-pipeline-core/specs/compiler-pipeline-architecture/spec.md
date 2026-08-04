## MODIFIED Requirements

### Requirement: Canonical production pipeline owner
`chelis-pipeline-core` SHALL own production semantic transitions in the guarded scope. This scope contains the core, compiler API, Reef, CLI, and E2E crates.

`chelis-compiler-api` SHALL own source preparation and dynamic goal dispatch. Production functions outside the core that need two or more semantic stages SHALL delegate.

Lower-layer crates SHALL retain individual stage implementations. `chelis-reef` SHALL use the core and SHALL NOT retain a direct semantic sequence.

The source architecture guard SHALL inspect production Rust files under exactly these roots:

- `crates/chelis-pipeline-core/src`
- `crates/chelis-compiler-api/src`
- `crates/chelis-reef/src`
- `crates/chelis-cli/src`
- `crates/chelis-e2e/src`

The guard SHALL permit canonical multi-stage orchestration only in the core owner. It SHALL NOT claim coverage for all workspace production files.

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

#### Scenario: Facade consumer delegates
- **WHEN** a CLI, edit, compiler-API, or E2E path needs a full semantic check
- **THEN** the path requests a compiler-API pipeline goal instead of calling the stages in sequence

#### Scenario: Reef package artifact path delegates
- **WHEN** `chelis-reef` checks a fully linked package before artifact or schema output
- **THEN** the path passes owned expanded Deep through `chelis-pipeline-core`

#### Scenario: Production consumer duplicates the sequence
- **WHEN** a guarded production file outside the core owner orchestrates two or more canonical semantic stages
- **THEN** the source architecture guard rejects the file and identifies the duplicated stage calls

#### Scenario: Reef recreates the removed exception
- **WHEN** a Reef negative fixture calls type, effect, and linearity stages directly
- **THEN** the source architecture guard rejects the fixture and identifies all three stages

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

### Requirement: Continuous artifact compile-fail enforcement
The canonical per-PR gate SHALL run `cargo test -p chelis-compiler-api --doc`.

The same gate SHALL run `cargo test -p chelis-pipeline-core --doc` and `.venv/bin/python scripts/check_checkpoint_compile_fail.py`.

Command-list unit tests SHALL NOT substitute for these executable controls.

#### Scenario: Hosted CI checks compile-time artifact boundaries
- **WHEN** hosted CI runs the `lint-and-unit` gate stage
- **THEN** it executes both pipeline doctest suites and the raw-checkpoint compile-fail fixture

## ADDED Requirements

### Requirement: Dependency-bottom semantic core
The workspace SHALL contain an unpublished `chelis-pipeline-core` crate. Its direct production dependencies SHALL be exactly the approved lower compiler crates.

The approved direct dependencies SHALL be `chelis-deep`, `chelis-types`, `chelis-effects`, and `chelis-ir`.

The resolved dependency graph from the core SHALL NOT reach compiler API, Reef, source preparation, macros, backends, schemas, wire adapters, or caches.

`chelis-compiler-api` and `chelis-reef` SHALL depend on the core. The core SHALL NOT depend on either upper crate.

A dependency guard SHALL inspect both the core manifest and the resolved workspace graph. It SHALL fail closed on an unknown direct dependency.

#### Scenario: Core uses only approved lower dependencies
- **WHEN** the dependency guard inspects the shipped core manifest and resolved graph
- **THEN** it accepts the four approved direct dependencies and finds no path to an upper crate

#### Scenario: Core adds a forbidden direct dependency
- **WHEN** an in-memory negative fixture adds `chelis-reef` to the core manifest
- **THEN** the dependency guard rejects the fixture and names `chelis-reef`

#### Scenario: Core reaches an upper crate through another dependency
- **WHEN** a negative graph fixture adds a transitive path from the core to `chelis-compiler-api`
- **THEN** the dependency guard rejects the fixture and displays the complete path

#### Scenario: Core adds an unknown dependency
- **WHEN** an in-memory negative fixture adds a dependency that is not in the approved set
- **THEN** the dependency guard rejects the fixture instead of classifying it as harmless

### Requirement: Owned expanded-Deep boundary
The core SHALL accept an owned expanded Deep program. It SHALL NOT parse source, desugar Surf, expand macros, or prune an entry.

`chelis-compiler-api` SHALL complete source preparation before it constructs the core carrier. Reef SHALL complete graph links and source expansion before construction.

The carrier SHALL own its Deep expressions. A borrowed source buffer or source-schema type SHALL NOT cross the core boundary.

#### Scenario: Compiler API prepares source
- **WHEN** a compiler API request contains Surf source and an entry name
- **THEN** the facade parses, expands, prunes, and passes owned expanded Deep to the core

#### Scenario: Reef prepares linked Deep
- **WHEN** Reef checks a linked package
- **THEN** Reef passes its owned expanded Deep to the same core carrier without source-schema conversion

#### Scenario: Source dependency enters the core
- **WHEN** a negative manifest fixture adds `chelis-surf` or `chelis-macros` to the core
- **THEN** the dependency guard rejects the fixture as a boundary violation

### Requirement: Stable compiler API facade
Existing `chelis_compiler_api::pipeline` imports SHALL compile without a direct `chelis-pipeline-core` dependency. The facade SHALL re-export moved artifacts and wrap policy functions.

`PipelineRequest`, `PipelineGoal`, `PipelineOutcome`, `PreparationError`, and `PipelineRejection` SHALL remain facade-owned public types.

The facade SHALL preserve current cancellation stages and rejection variants. Core artifacts SHALL NOT gain machine-facing wire traits.

The compiler API SHALL retain source preparation, dynamic goal selection, native-error conversion, host policy selection, and backend policy selection.

#### Scenario: Existing consumer imports only compiler API
- **WHEN** a compile fixture imports the current pipeline types and functions through `chelis_compiler_api::pipeline`
- **THEN** the fixture compiles without a direct core dependency

#### Scenario: Rejected facade request keeps its stage
- **WHEN** parse, type, effect, linearity, lower, root-count, or cancellation rejection occurs
- **THEN** the facade returns the current `PipelineRejection` variant with exact current data

#### Scenario: Core artifact enters a wire model
- **WHEN** a compile-fail fixture requires a core checked or lowered artifact to implement `serde::Serialize`
- **THEN** Rust rejects the fixture because the artifact has no wire implementation

#### Scenario: Backend policy enters the core graph
- **WHEN** a negative dependency fixture adds a backend crate to the core
- **THEN** the dependency guard rejects the fixture and names the backend dependency

### Requirement: Reef uses canonical semantic transitions
The Reef package artifact and schema paths SHALL use the core type, effect, and linearity transitions. Reef SHALL NOT call those stages in sequence.

The Reef adapter SHALL preserve its linked-program guard. It SHALL preserve exact accepted artifacts and exact rejected error text.

#### Scenario: Valid linked package keeps its artifact
- **WHEN** a valid linked package passes through the core transitions
- **THEN** Reef produces the same checked package, archive content, schema content, and hashes as the baseline

#### Scenario: Invalid linked package keeps its rejection
- **WHEN** a linked package fails type, effect, or linearity checks
- **THEN** Reef returns the same error text and order as the baseline

#### Scenario: Linked internal names remain accepted
- **WHEN** linked Deep contains valid internal mangled names
- **THEN** the linked-program guard remains active for the complete core semantic check

#### Scenario: Reef restores direct stage calls
- **WHEN** a negative source fixture restores the former Reef helper sequence
- **THEN** the source guard rejects the fixture and identifies the repeated stages

### Requirement: Core state remains legal by construction
The moved core artifacts SHALL preserve the current legal-state guarantees. Rejected states SHALL expose no checked program, DAG, or partial root map.

Exact root construction SHALL reject different tensor-name and DAG-root counts. Explicit accepted host results SHALL retain empty named-root products.

`SemanticRejection` SHALL contain only effect or linearity errors. Core lower errors SHALL contain only lower diagnostics or root-count failures.

#### Scenario: Semantic success creates checked state
- **WHEN** type, effect, and linearity stages accept owned expanded Deep
- **THEN** the core returns `CheckedCompilation` with canonical root metadata

#### Scenario: Root counts differ
- **WHEN** exact root construction receives different name and root counts
- **THEN** the core returns a root-count error before it creates a lowered state

#### Scenario: Caller swaps root map roles
- **WHEN** a compile-fail fixture passes a forward-node index where declared roots are required
- **THEN** Rust rejects the fixture because the artifact types differ

#### Scenario: Rejection exposes success state
- **WHEN** a compile-fail fixture requests a checked product from a semantic rejection
- **THEN** Rust rejects the fixture because the rejection has no success accessor

### Requirement: Standard-library blocker inventory
The change SHALL record why `chelis-pipeline-core` still requires `std`. The inventory SHALL cover direct core use and transitive lower-crate blockers.

The inventory SHALL distinguish collections, allocation, global state, stack support, panic behavior, operating-system dependencies, and dependency features.

The inventory SHALL NOT claim current `#![no_std]` support or assign removal work without separate approval.

#### Scenario: Reviewer reads the blocker inventory
- **WHEN** a reviewer checks the extracted crate's portability status
- **THEN** the inventory states that the core requires `std` and names each confirmed blocker class

#### Scenario: Documentation claims current no-std support
- **WHEN** a documentation negative fixture states that the extracted core supports `#![no_std]`
- **THEN** the documentation guard rejects the claim because the blocker inventory remains nonempty

### Requirement: Core extraction acceptance oracle
The authoritative completion oracle SHALL remain `.venv/bin/python scripts/compiler_pipeline_oracle.py`.

The oracle SHALL run core tests, both pipeline doctest suites, dependency guards, source guards, Reef parity, and current consumer parity.

The oracle SHALL contain positive and negative evidence for every new boundary. OpenSpec validation SHALL remain structural evidence only.

#### Scenario: Extracted core satisfies all boundaries
- **WHEN** the authoritative oracle runs after the migration
- **THEN** all core, facade, Reef, guard, compile-fail, and consumer parity controls pass

#### Scenario: Forbidden dependency is planted
- **WHEN** the dependency-guard negative fixture adds a shell dependency to the core
- **THEN** the authoritative oracle fails and names the forbidden dependency path

#### Scenario: Reef duplicate is planted
- **WHEN** the source-guard negative fixture restores a direct Reef semantic sequence
- **THEN** the authoritative oracle fails and names the duplicated stages and source path
