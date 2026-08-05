# Compiler Pipeline Architecture Delta: enforce-pipeline-core-guards-in-ci

## MODIFIED Requirements

### Requirement: Standard-library blocker inventory
The change SHALL record why `chelis-pipeline-core` still requires `std`. The inventory SHALL cover direct core use and transitive lower-crate blockers.

The inventory SHALL distinguish collections, allocation, global state, stack support, panic behavior, operating-system dependencies, and dependency features.

The inventory SHALL NOT claim current `#![no_std]` support or assign removal work without separate approval.

The documentation guard SHALL detect a current `#![no_std]` support claim about the core across the common phrasings, for both the `no_std` and `no-std` spellings. Detection SHALL cover adjective forms (`no_std compatible`, `no_std ready`, `no_std support`), verb forms (`supports`, `provides`, `enables`, `is`), and environment forms (`runs in`, `works in`, `compiles as`, `portable to`) a `no_std` target or environment.

The guard SHALL NOT reject a statement that the core requires `std`, a statement that `no_std` is a future non-goal, a hypothetical blocker line that describes what `no_std` would require, or any other line that names `no_std` without asserting a current crate capability. The guard SHALL evaluate an explicit negation, future-target, or hypothetical frame before any affirmative match, so a line such as "supports std only, not no_std" is not a false claim. The guard is a best-effort natural-language heuristic; the required-statement, required-section, and required-crate checks remain the guarantee.

#### Scenario: Reviewer reads the blocker inventory
- **WHEN** a reviewer checks the extracted crate's portability status
- **THEN** the inventory states that the core requires `std` and names each confirmed blocker class

#### Scenario: Documentation claims current no-std support
- **WHEN** a documentation negative fixture states that the extracted core supports `#![no_std]`
- **THEN** the documentation guard rejects the claim because the blocker inventory remains nonempty

#### Scenario: A false claim uses an environment phrasing
- **WHEN** a documentation negative fixture states that the core is "portable to no_std targets" or "works in a no_std environment"
- **THEN** the documentation guard rejects the claim as a false current portability statement

#### Scenario: A false claim uses the hyphen spelling
- **WHEN** a documentation negative fixture states that the core is "no-std compatible" or is "portable to no-std targets"
- **THEN** the documentation guard rejects the claim, treating `no-std` the same as `no_std`

#### Scenario: A hypothetical blocker line is not a false claim
- **WHEN** the inventory states that "no_std would require a custom allocator"
- **THEN** the documentation guard does not reject the line because it describes a blocker, not a current capability

#### Scenario: A true requires-std statement is not a false claim
- **WHEN** the inventory states that the core "supports std only, not no_std"
- **THEN** the documentation guard does not reject the statement and validation passes

#### Scenario: A future-target non-goal is not a false claim
- **WHEN** the inventory frames `no_std` as a future target that is not a current capability
- **THEN** the documentation guard does not reject the statement

## ADDED Requirements

### Requirement: Continuous pipeline-core boundary guard enforcement
The canonical per-PR gate SHALL run the pipeline-core dependency guard, the pipeline-core documentation guard, and the pipeline-artifact compile-fail fixture. Hosted CI SHALL run the same three controls through the gate's `lint-and-unit` stage.

The manual completion oracle SHALL remain and SHALL continue to run the same three controls, but it SHALL NOT be the only enforcement point for the dependency boundary, the no_std documentation contract, or the facade artifact compile-fail boundary.

Command-list unit tests SHALL NOT substitute for these executable controls. The gate parity test SHALL confirm the workflow runs the stage that produces the three commands.

#### Scenario: Hosted CI enforces the pipeline-core boundary
- **WHEN** hosted CI runs the `lint-and-unit` gate stage
- **THEN** it executes the dependency guard, the documentation guard, and the pipeline-artifact compile-fail fixture

#### Scenario: A forbidden dependency is added on a PR
- **WHEN** a pull request adds a forbidden direct or transitive dependency to `chelis-pipeline-core`
- **THEN** the per-PR gate fails on the dependency guard without requiring a manual oracle run

#### Scenario: A false portability claim is added on a PR
- **WHEN** a pull request adds a current `#![no_std]` support claim to the blocker inventory
- **THEN** the per-PR gate fails on the documentation guard without requiring a manual oracle run

#### Scenario: A facade artifact boundary is broken on a PR
- **WHEN** a pull request removes a required artifact-misuse compile error at the `chelis_compiler_api::pipeline` boundary
- **THEN** the per-PR gate fails on the pipeline-artifact compile-fail fixture without requiring a manual oracle run
