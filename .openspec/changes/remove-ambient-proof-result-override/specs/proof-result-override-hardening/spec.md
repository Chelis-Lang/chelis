## ADDED Requirements

### Requirement: Ambient variables cannot substitute proof results
Production proof paths SHALL NOT read `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` or another process environment variable that substitutes a solver observation, soundness classification, qualifier, degradation, or composite verdict.

#### Scenario: Hostile proved request is inert
- **WHEN** the production proof entry point runs with the removed variable set to `proved`
- **THEN** engine execution and verdict mapping are identical to a run with the variable absent

#### Scenario: Invalid hostile value is inert
- **WHEN** an unrecognized result-substitution variable or invalid former value is present
- **THEN** production proof behavior does not parse it, report it as a proof observation, or change the verdict

### Requirement: Tests inject raw outcomes explicitly
Tests that require controlled solver outcomes SHALL use an explicitly constructed dev-only test-support target or harness crate that supplies raw proved, disproved, timeout, unknown, or error outcomes through ordinary Rust values. Normal production dependency/feature graphs SHALL have no edge to that target and SHALL expose no scripted-result constructor. The fixture MUST NOT directly construct final trust-bearing verdict fields.

#### Scenario: Explicit proved fixture uses normal mapping
- **WHEN** a test supplies a raw proved outcome through the explicit fixture
- **THEN** the existing production mapper determines the resulting status and trust fields

#### Scenario: Scripted fixture cannot ship as production authority
- **WHEN** a normal production dependency builds `chelis-prove`
- **THEN** no public production feature or constructor exposes the scripted outcome fixture or grants it production trust

### Requirement: Existing fail-closed mapping is preserved
Removing the ambient shortcut SHALL preserve established mapping for real or explicitly scripted timeout, unknown, malformed/error, disproved, and proved raw outcomes.

#### Scenario: Timeout remains non-green
- **WHEN** the explicit fixture or real engine supplies timeout
- **THEN** the resulting proof artifact remains non-green with the established timeout classification

#### Scenario: Proved still requires an actual raw outcome
- **WHEN** no engine or explicit test fixture supplies a proved outcome
- **THEN** removal of the environment branch cannot create a proved result by fallback
