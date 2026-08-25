## ADDED Requirements

### Requirement: Build checks the selected program before pruning

Under the front-end gate in [`spec/04-type-system.md` §2.5](../../../../../spec/04-type-system.md), `chelis build` SHALL complete type, effect, and linearity checks for every definition in the selected program.

The selected program SHALL contain the input source or the complete linked Reef package target. The semantic checks SHALL occur before eval-only removal or general reachability pruning.

A semantic error in an unreachable definition SHALL fail the build. Reachability and eval-only taint SHALL NOT hide that error.

A source file outside the selected program SHALL remain outside this requirement.

#### Scenario: Direct eval-only definition contains a type error

- **WHEN** an unreachable definition directly references an eval-only builtin and contains a type error
- **THEN** `chelis build` rejects the selected program with that type error before it removes the definition

#### Scenario: Transitive eval-only wrapper contains an error

- **WHEN** an unreachable wrapper references an eval-only-tainted definition at any depth and contains a semantic error
- **THEN** `chelis build` rejects the selected program before transitive eval-only removal can hide the error

#### Scenario: Unreachable ordinary definition contains an error

- **WHEN** an unreachable definition contains a semantic error and does not reference an eval-only builtin
- **THEN** `chelis build` rejects the selected program through the same complete semantic gate

#### Scenario: Check and build see the same semantic error

- **WHEN** a selected program contains an error inside an unreachable eval-only-tainted definition
- **THEN** `chelis check` and `chelis build` both reject the error with the same diagnostic kind and source location

#### Scenario: Cache modes preserve build rejection

- **WHEN** a selected program fails the complete semantic gate under cold, warm, and cache-disabled builds
- **THEN** all three builds reject it with byte-identical build diagnostics

#### Scenario: Unselected source contains an error

- **WHEN** a source file is outside the selected source or linked Reef package target
- **THEN** `chelis build` does not include that file in this semantic gate
