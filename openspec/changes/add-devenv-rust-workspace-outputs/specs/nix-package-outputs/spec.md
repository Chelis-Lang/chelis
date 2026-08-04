## MODIFIED Requirements

### Requirement: Rust packages use one pinned checked-in crate2nix graph
The flake and Devenv MUST pin crate2nix 0.15.0. The repository MUST track one generated `Cargo.nix` file.

The repository MUST provide one shared workspace graph helper. The root flake adapter and the Devenv workspace import MUST call that helper with their owned inputs.

Both package interfaces MUST import the same checked-in `Cargo.nix`. Package evaluation MUST NOT use import from derivation.

Each interface MUST instantiate one graph. The graph MUST contain the `chelis-cli`, `chelis-runtime`, and `chelisup` workspace members. The compiler member MUST enable the `smt` feature.

The repository MUST record a digest of `Cargo.lock`, the root manifest, and every workspace manifest in `Cargo.nix`.

The Python suite and root native checks MUST reject an absent or stale digest. They MUST parse workspace members at the input boundary.

An exact regeneration check MUST use the pinned crate2nix source, the configured Rust toolchain, and offline Cargo dependencies.

The regeneration command MUST use `--no-default-features`, `--features chelis-cli/smt`, and `--output Cargo.nix`. It MUST compare the complete regenerated file.

The Linux native job MUST run exact regeneration when a graph input changes. A workflow dispatch MUST select exact regeneration.

If the change detector fails or omits its output, the Linux job MUST run exact regeneration.

The macOS native job MUST NOT regenerate the platform-independent graph. It MUST import and build the checked-in graph.

Graph inputs include `Cargo.nix`, `Cargo.lock`, every `Cargo.toml`, crate2nix pins, toolchain pins, and graph-generation code or configuration.

The exact regeneration source MUST include tracked Cargo configuration and crate2nix configuration or hash files when present.

An ordinary Rust source change MUST NOT select exact graph regeneration. The normal Linux package build MUST still run for that change.

Crate builds MUST remain network-independent.

The shared `cvc5-sys` override MUST provide the fixed cvc5 tree and libclang. It MUST NOT fetch cvc5 during the build.

The shared source override MUST provide a workspace source view to `chelis-cli`, `chelis-compiler-api`, `chelis-cove`, and `tree-sitter-chelis`.

Other crates MUST keep crate-local sources.

The root flake and Devenv MUST use one shared artifact assembly helper for `chelis`, `chelis-runtime`, and `chelisup`.

The native checks MUST inspect the checked-in graph and its required workspace members. An absent graph or failed import MUST fail the check.

#### Scenario: Separate product members share crate outputs
- **WHEN** one interface builds the compiler, runtime, and installer packages
- **THEN** one crate2nix graph builds their common dependencies as shared crate derivations

#### Scenario: Both package interfaces use the shared graph rules
- **WHEN** native CI evaluates the root flake and Devenv package graphs
- **THEN** both interfaces import the same graph with the same overrides and artifact assembly

#### Scenario: A Cargo graph input changes
- **WHEN** `Cargo.lock` or a workspace manifest changes without a matching `Cargo.nix`
- **THEN** the graph digest check fails and names `Cargo.nix`
- **AND** the Linux native job selects exact regeneration

#### Scenario: The generated graph changes
- **WHEN** `Cargo.nix` differs from pinned crate2nix 0.15.0 output
- **THEN** the Linux exact regeneration check fails

#### Scenario: Graph-generation configuration changes
- **WHEN** tracked Cargo or crate2nix configuration changes
- **THEN** the Linux native job selects exact regeneration
- **AND** the regeneration source contains the changed configuration

#### Scenario: Rust source changes without a graph change
- **WHEN** a pull request changes Rust source but no graph input
- **THEN** the Linux job builds the packages without exact graph regeneration

#### Scenario: The change detector fails
- **WHEN** the change job fails or omits `cargo_graph_changed`
- **THEN** the Linux package job runs exact graph regeneration

#### Scenario: The cvc5 override is absent
- **WHEN** `cvc5-sys` does not receive the fixed cvc5 tree in either package interface
- **THEN** the SMT compiler build fails without a network fallback

#### Scenario: A crate loses an external compile-time asset
- **WHEN** one listed crate receives only its crate directory
- **THEN** the crate build fails because a required header or grammar file is absent

#### Scenario: Import from derivation is introduced
- **WHEN** package evaluation generates or imports a derived Cargo graph
- **THEN** the static package contract test fails

#### Scenario: Devenv creates one graph per output
- **WHEN** Devenv imports separate crate2nix graphs for the three product outputs
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

The `aarch64-darwin` job MUST run the same package and behavior steps as a manual gate. It MUST omit exact graph regeneration.

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
- **THEN** the documentation explains the shared graph and separate package instantiation

#### Scenario: A user needs version routing
- **WHEN** the user reads the Nix installation boundary
- **THEN** the documentation directs release toolchain routing to `chelisup`
