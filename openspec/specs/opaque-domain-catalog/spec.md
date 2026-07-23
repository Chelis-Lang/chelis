# Opaque Domain Catalog

## Purpose

Define how `opaque-domain-construction` discovers, prepares, scopes, and reuses the Surf declaration catalog needed for cross-file opacity diagnostics.

## Requirements

### Requirement: Catalog discovery uses the canonical lint walk
For a directory lint root, `opaque-domain-construction` SHALL derive its Surf catalog candidates from the entries admitted by the canonical `chelis-lint` walker. Catalog discovery MUST honor the same skipped-directory policy as normal lint dispatch, including `target/`, `.git/`, `node_modules/`, `__pycache__/`, `.venv*`, nested agent worktrees, and generated opaque-invariant programs.

#### Scenario: Excluded Surf source cannot affect the catalog
- **WHEN** a skipped directory contains a valid `.ch` file declaring an opaque type and an admitted `.ch` file constructs a same-named non-opaque type
- **THEN** the skipped declaration does not enter the catalog and does not create an `opaque-domain-construction` violation

#### Scenario: Admitted Surf source contributes to the catalog
- **WHEN** an admitted `.ch` file declares an opaque type and another admitted `.ch` file directly constructs that type outside its defining module
- **THEN** the lint reports the existing `opaque-domain-construction` violation

### Requirement: Catalog preparation is bounded to once per lint invocation
`chelis_lint::lint` SHALL prepare the opaque-domain Surf catalog no more than once per invocation and SHALL reuse that immutable corpus catalog for every admitted Surf file dispatched during that invocation. Each admitted Surf catalog candidate MUST be read and parsed no more than once for catalog preparation during the invocation.

#### Scenario: Multiple checked Surf files share one prepared catalog
- **WHEN** one lint invocation admits multiple `.ch` files
- **THEN** catalog preparation runs once and all `opaque-domain-construction` checks consume the same prepared corpus catalog

#### Scenario: Catalog preparation does not repeat per checked file
- **WHEN** a lint invocation admits `N` Surf files
- **THEN** catalog discovery and catalog parsing are bounded by the admitted corpus size rather than being repeated `N` times

### Requirement: Opaque-site lookup is indexed by leaf and defining module
The prepared catalog MUST index opaque definitions by type leaf and defining module. Checking one record construction, cast, or typed update MUST NOT scan opaque declarations with unrelated leaves. The fail-closed untyped Deep update check MUST use an index of defining modules rather than scan every declaration.

#### Scenario: Distinct declarations and sites remain linear
- **WHEN** the admitted corpus contains `N` distinct opaque type leaves and `N` construction sites
- **THEN** catalog construction is bounded by the declarations and each site performs a bounded leaf/module lookup rather than an `N`-element catalog scan

#### Scenario: Same-leaf definitions retain module identity
- **WHEN** multiple modules define opaque types with the same leaf
- **THEN** the leaf index retains every defining module and preserves defining-module allowance plus out-of-module rejection

### Requirement: Catalog state is isolated to one invocation
Prepared opaque-domain catalog state MUST NOT outlive or leak across `chelis_lint::lint` invocations. A later invocation SHALL rebuild from its own admitted entries so file additions, removals, and edits are visible without process restart.

#### Scenario: A later invocation sees a new opaque declaration
- **WHEN** one invocation completes, an admitted Surf file is changed to declare an opaque type, and lint is invoked again with the same rule objects
- **THEN** the second invocation uses the changed declaration and reports any resulting out-of-module construction violation

#### Scenario: A later invocation drops a removed declaration
- **WHEN** one invocation catalogs an opaque declaration, that declaration is removed, and lint is invoked again
- **THEN** the second invocation does not reuse the removed declaration

### Requirement: Existing opaque-domain semantics remain unchanged
The optimization SHALL preserve existing Surf diagnostics, per-file module-less shadowing, single-file root behavior, fail-soft handling of unreadable or unparsable corpus candidates, and Deep-source checking. Deep-source checks MUST NOT depend on the repository-wide Surf catalog.

#### Scenario: Defining-module construction remains allowed
- **WHEN** a Surf module directly constructs an opaque type that it defines
- **THEN** `opaque-domain-construction` reports no violation

#### Scenario: Out-of-module construction remains rejected
- **WHEN** a Surf module directly constructs an admitted opaque type defined by another module
- **THEN** `opaque-domain-construction` reports the same violation class and message semantics as before the optimization

#### Scenario: Single-file lint remains self-contained
- **WHEN** the lint root is one `.ch` file
- **THEN** catalog preparation uses that file and preserves the current single-file result

#### Scenario: Deep lint remains corpus-independent
- **WHEN** the lint checks a Deep source file
- **THEN** opaque-domain declarations and constructions are derived from that Deep source without a repository-wide Surf catalog walk
