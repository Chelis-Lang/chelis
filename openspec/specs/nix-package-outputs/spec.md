# Nix Package Outputs Specification

## Purpose

Define the locked Nix packages, applications, artifact layouts, supported systems, and native checks for Chelis.

## Requirements

### Requirement: The repository exposes locked Nix package outputs
The repository MUST track `flake.nix` and `flake.lock`. The flake MUST export packages for `x86_64-linux` and `aarch64-darwin`.

For each supported system, the flake MUST export `chelis`, `chelis-runtime`, and `chelisup`. `packages.default` MUST resolve to the same derivation as `packages.chelis`.

The flake MUST NOT claim support for a system without the complete native check set.

#### Scenario: A user lists package outputs
- **WHEN** a user runs `nix flake show`
- **THEN** each supported system lists `chelis`, `chelis-runtime`, `chelisup`, and `default`

#### Scenario: A user builds the default package
- **WHEN** a user runs `nix build .`
- **THEN** Nix builds the same derivation as `nix build .#chelis`

#### Scenario: An unsupported system is inspected
- **WHEN** a user inspects package outputs for a system outside the supported list
- **THEN** the flake does not present that system as a supported Chelis package target

### Requirement: The default package contains the complete Chelis toolchain
`packages.chelis` MUST contain `bin/chelis`, `lib/libchelis_runtime.a`, and all public runtime headers.

The compiler MUST use the workspace version and the `smt` Cargo feature. The build MUST use `Cargo.lock` and MUST complete without network access in the build sandbox.

The package MUST preserve the artifact boundaries of the release toolchain. It MUST NOT add test-only binaries or internal workspace executables.

#### Scenario: The complete toolchain builds
- **WHEN** a native builder builds `packages.chelis`
- **THEN** the output contains the compiler, runtime static library, and five public runtime headers

#### Scenario: The compiler runs
- **WHEN** a check invokes `bin/chelis --version` and `bin/chelis --help`
- **THEN** both commands exit with status 0 and the version matches the workspace manifest

#### Scenario: SMT support is absent
- **WHEN** the final Nix compiler does not discharge the SMT verification fixture through cvc5
- **THEN** the package check fails

#### Scenario: A build attempts network access
- **WHEN** a Rust crate or the cvc5 build requests an undeclared network source
- **THEN** the sandboxed Nix build fails instead of fetching the source

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

### Requirement: The runtime package exposes the C runtime contract
`packages.chelis-runtime` MUST contain `lib/libchelis_runtime.a` and these headers under `include/`:

- `chelis_runtime.h`
- `chelis_runtime_dtype.h`
- `chelis_blas.h`
- `chelis_simd.h`
- `chelis_math.h`

The package MUST NOT contain the `chelis` or `chelisup` executable.

#### Scenario: A C consumer uses the runtime package
- **WHEN** a native check compiles and links a minimal C program against `packages.chelis-runtime`
- **THEN** the program links with the static library and includes the public headers

#### Scenario: A runtime artifact is missing
- **WHEN** the runtime library or one required header is absent
- **THEN** the runtime package check fails and names the missing path

#### Scenario: The runtime package gains a product executable
- **WHEN** `packages.chelis-runtime` contains `bin/chelis` or `bin/chelisup`
- **THEN** the runtime package check fails

### Requirement: The chelisup package exposes a closure-safe installer
`packages.chelisup` MUST contain a launcher at `bin/chelisup` and the real installer at `libexec/chelisup`. It MUST NOT install a `bin/chelis` shim in the package output.

The launcher MUST create `$CHELIS_HOME/nix-gcroots/chelisup.next` before it delegates an `install` command. This staging root MUST point to the complete Nix package output.

After a successful install, the launcher MUST promote `$CHELIS_HOME/nix-gcroots/chelisup` and remove the staging root. After a failed install, it MUST preserve the prior stable root.

If the failed install copied a new binary, the launcher MUST promote `$CHELIS_HOME/nix-gcroots/chelisup.partial`. This partial root MUST protect that binary.

The launcher MUST remove the staging root after it preserves the required package closure.

`chelisup install` MUST copy the real installer into `$CHELIS_HOME/bin/chelis`. The launcher MUST replace `$CHELIS_HOME/bin/chelisup` with the Nix launcher after each successful installer copy.

The real installer MUST NOT contain paths or cleanup logic for Nix GC roots. The Nix launcher MUST remove all three roots after the real `self uninstall` command succeeds.

The package MUST use the workspace version and `Cargo.lock`.

#### Scenario: A user builds chelisup
- **WHEN** a user runs `nix build .#chelisup`
- **THEN** the result contains a runnable launcher and the internal installer payload

#### Scenario: The Nix installer creates release shims
- **WHEN** a user runs `nix run .#chelisup -- install <ver>`
- **THEN** the launcher stages the new GC root before the installer copies its executable
- **AND** the launcher restores itself at `$CHELIS_HOME/bin/chelisup`
- **AND** the launcher promotes the stable GC root after success

#### Scenario: A Nix installer update fails before a copy
- **WHEN** an install fails before it copies the new installer
- **THEN** the launcher removes the staging root and preserves the prior stable root

#### Scenario: A Nix installer update fails after a partial copy
- **WHEN** an install fails after it copies either new executable
- **THEN** the launcher promotes the partial root and preserves the prior stable root
- **AND** the launcher removes the staging root

#### Scenario: A prior install left a staging root
- **WHEN** the next install finds `$CHELIS_HOME/nix-gcroots/chelisup.next`
- **THEN** the launcher preserves every closure that matches an installed executable
- **AND** the launcher removes the stale staging root before the new attempt

#### Scenario: Nix garbage collection runs after installation
- **WHEN** the installed shim still depends on the Nix package closure
- **THEN** the GC root keeps that closure live

#### Scenario: A user removes Nix-installed chelisup
- **WHEN** the user runs the installed Nix launcher with `self uninstall`
- **THEN** the real installer removes both executable copies
- **AND** the launcher removes all three GC roots

#### Scenario: The real installer runs without the Nix launcher
- **WHEN** the real installer runs `self uninstall` directly
- **THEN** it does not read or remove a Nix GC root

#### Scenario: The installer package collides with the compiler
- **WHEN** the `chelisup` output contains `bin/chelis`
- **THEN** the package check fails

### Requirement: The flake exposes runnable applications
For each supported system, the flake MUST export `apps.chelis` and `apps.chelisup`. `apps.default` MUST resolve to `apps.chelis`.

Each app MUST use the executable from its corresponding package output. The app MUST NOT invoke a host command outside the Nix store.

#### Scenario: A user runs the compiler app
- **WHEN** a user runs `nix run .#chelis -- --version`
- **THEN** Nix executes `packages.chelis/bin/chelis`

#### Scenario: A user runs the installer app
- **WHEN** a user runs `nix run .#chelisup -- --help`
- **THEN** Nix executes `packages.chelisup/bin/chelisup`

#### Scenario: An app points outside its package
- **WHEN** an app program path does not belong to its corresponding package
- **THEN** flake evaluation or the app contract check fails

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
- **WHEN** the built `chelisup` launcher fails `bash -n` or `shellcheck`
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
