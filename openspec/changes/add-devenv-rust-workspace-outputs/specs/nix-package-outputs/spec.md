## MODIFIED Requirements

### Requirement: Rust packages use a pinned automatic crate2nix graph
The flake and Devenv MUST pin crate2nix 0.15.0. Each package interface MUST generate its graph with crate2nix `generatedCargoNix` and import from derivation.

The repository MUST provide one shared workspace graph helper. The root flake adapter and the Devenv workspace import MUST call that helper with their owned inputs.

The root flake MUST enable import from derivation. The repository MUST NOT track a generated `Cargo.nix` file or a graph-input digest.

The generator source MUST contain `Cargo.lock`, the root manifest, every workspace manifest, and the complete workspace source. The source filter MUST exclude generated and ignored build directories.

Each interface MUST generate one graph. The graph MUST contain the `chelis-cli`, `chelis-runtime`, and `chelisup` workspace members. The compiler member MUST enable the `smt` feature.

The generator MUST resolve each dependency through Nix fetchers. It MUST set `CARGO_NET_OFFLINE=true` before Cargo reads the prepared sources.

Crate builds MUST remain network-independent.

The shared `cvc5-sys` override MUST provide the fixed cvc5 tree and libclang. It MUST NOT fetch cvc5 during the build.

The shared source override MUST provide a workspace source view to `chelis-cli`, `chelis-compiler-api`, `chelis-cove`, and `tree-sitter-chelis`.

Other crates MUST keep crate-local sources.

The root flake and Devenv MUST use one shared artifact assembly helper for `chelis`, `chelis-runtime`, and `chelisup`.

The native check set MUST inspect each generated crate2nix artifact and its required workspace members. Generation or import failure MUST fail the check.

Each native Nix job MUST evaluate both graphs for its matching system. Cross-system inventory checks MUST not require a foreign-system generator build.

#### Scenario: Separate product members share crate outputs
- **WHEN** one interface builds the compiler, runtime, and installer packages
- **THEN** one crate2nix graph builds their common dependencies as shared crate derivations

#### Scenario: Both package interfaces use the shared graph rules
- **WHEN** native CI evaluates the root flake and Devenv package graphs
- **THEN** both graphs use the same feature selection, crate overrides, source policy, and artifact assembly

#### Scenario: A Cargo graph input changes
- **WHEN** `Cargo.lock`, a workspace manifest, or workspace source changes
- **THEN** Nix gives each affected generator a new derivation identity
- **AND** package evaluation imports the new graph without a repository refresh

#### Scenario: The repository contains a generated graph
- **WHEN** the repository tracks `Cargo.nix` or a graph-input digest
- **THEN** the package contract test fails

#### Scenario: The cvc5 override is absent
- **WHEN** `cvc5-sys` does not receive the fixed cvc5 tree in either package interface
- **THEN** the SMT compiler build fails without a network fallback

#### Scenario: A crate loses an external compile-time asset
- **WHEN** one listed crate receives only its crate directory
- **THEN** the crate build fails because a required header or grammar file is absent

#### Scenario: Import from derivation is disabled
- **WHEN** the Nix configuration prohibits import from derivation
- **THEN** package evaluation fails before crate compilation

#### Scenario: Devenv generates one graph per output
- **WHEN** Devenv creates separate crate2nix graphs for the three product outputs
- **THEN** the static package contract test fails

### Requirement: Nix and Devenv share input revisions
`flake.lock` and `devenv.lock` MUST pin the same Nixpkgs, Rust overlay, and crate2nix source revisions.

The repository MUST provide a parity checker with positive and negative tests. A mismatch MUST fail before a package build can count as accepted.

The checker MUST reject a floating crate2nix reference and a lock entry that differs from the configured Devenv pin.

#### Scenario: Shared pins match
- **WHEN** the parity checker reads both committed lock files
- **THEN** it exits with status 0 and reports matching Nixpkgs, Rust overlay, and crate2nix revisions

#### Scenario: One Nixpkgs pin changes
- **WHEN** a fixture changes the Nixpkgs revision in only one lock file
- **THEN** the checker exits with a nonzero status and names the Nixpkgs nodes

#### Scenario: One Rust overlay pin changes
- **WHEN** a fixture changes the Rust overlay revision in only one lock file
- **THEN** the checker exits with a nonzero status and names the Rust overlay nodes

#### Scenario: One crate2nix pin changes
- **WHEN** a fixture changes the crate2nix revision in only one lock file
- **THEN** the checker exits with a nonzero status and names the crate2nix nodes

#### Scenario: Devenv uses a floating crate2nix input
- **WHEN** `devenv.yaml` selects a crate2nix branch or tag without an exact revision
- **THEN** the static Devenv input test fails

### Requirement: Native Nix checks protect every supported system
Each supported system MUST define checks for package construction, package contents, executable behavior, launcher shell lint, app paths, SMT activation, and lock parity.

CI MUST run the complete root flake check set, the Nix flake contract suite, and all four Devenv package outputs on native `x86_64-linux` and `aarch64-darwin` builders.

The native jobs MUST check the artifact layout and executable behavior of each Devenv output. They MUST run the SMT verification fixture against the Devenv compiler.

The `x86_64-linux` job MUST run on every code pull request and push to `main`. A docs-only pull request MUST skip this job through the shared job-level detector.

The `aarch64-darwin` job MUST run identical package steps as a documented manual dispatch gate. Default CI does not run this job.

A system MUST NOT count as supported from evaluation-only evidence.

The supported-system list and the named native CI jobs MUST have exact parity. The repository script suite MUST fail when either list contains an unmatched system.

The authoritative completion oracle MUST be the two successful native CI check jobs: the pull-request `x86_64-linux` job and the dispatched `aarch64-darwin` job.

#### Scenario: Linux checks pass
- **WHEN** CI runs the complete flake check set and Devenv package checks on `x86_64-linux`
- **THEN** every package and contract check exits with status 0

#### Scenario: A Devenv package output drifts
- **WHEN** a Devenv output violates its shared artifact or behavior contract
- **THEN** the native package job fails before the system can count as supported

#### Scenario: Devenv compiler lacks SMT support
- **WHEN** the Devenv compiler does not discharge the native cvc5 fixture
- **THEN** the native package job fails

#### Scenario: A docs-only pull request skips the Linux job
- **WHEN** a pull request changes only documentation paths
- **THEN** the shared detector reports docs-only and the `x86_64-linux` job skips with a successful context

#### Scenario: The generated launcher fails shell lint
- **WHEN** the built chelisup launcher fails `bash -n` or `shellcheck`
- **THEN** the launcher lint check fails the native check set

#### Scenario: macOS checks pass
- **WHEN** a manual CI run executes the complete flake and Devenv package checks on `aarch64-darwin`
- **THEN** every package and contract check exits with status 0

#### Scenario: One package check fails
- **WHEN** any required package or contract check fails on a supported system
- **THEN** the native system job fails and the change does not satisfy the completion oracle

#### Scenario: A supported system has no native job
- **WHEN** the supported-system list contains a system without a named native CI job
- **THEN** the repository script suite fails before the system can count as supported

### Requirement: Documentation explains the Nix channel boundary
Contributor documentation MUST show the package names and commands for `nix build`, `nix run`, `devenv build`, and flake checks.

The documentation MUST state that Devenv generates its own workspace graph and does not evaluate the root flake.

The documentation MUST state that both interfaces share package rules, but dirty worktrees can produce different derivation identities.

The documentation MUST list the supported systems. It MUST state that Nix is an additive source-build channel and that `chelisup` remains the release installer.

The documentation MUST NOT claim separate Nix packages for internal crates, the Python extension, or `chelis-std`.

#### Scenario: A Nix user reads the build instructions
- **WHEN** the user reads the source-build documentation
- **THEN** the documentation identifies every public package, app, Devenv output, supported system, and check command

#### Scenario: A contributor compares package interfaces
- **WHEN** the contributor reads the Devenv package documentation
- **THEN** the documentation explains the shared contracts and separate graph ownership

#### Scenario: A user needs version routing
- **WHEN** the user reads the Nix installation boundary
- **THEN** the documentation directs release toolchain routing to `chelisup`
