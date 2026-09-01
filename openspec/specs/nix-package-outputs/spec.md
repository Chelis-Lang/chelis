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

### Requirement: Rust packages use a pinned automatic crate2nix graph
The flake MUST pin crate2nix 0.15.0. The package build MUST generate the graph with crate2nix `generatedCargoNix` and import from derivation.

The flake MUST enable import from derivation. The repository MUST NOT track a generated `Cargo.nix` file or a graph-input digest.

The generator source MUST contain `Cargo.lock`, the root manifest, every workspace manifest, and the complete workspace source.

The graph MUST contain the `chelis-cli`, `chelis-runtime`, and `chelisup` workspace members. The compiler member MUST enable the `smt` feature.

The generator MUST resolve each dependency through Nix fetchers. It MUST set `CARGO_NET_OFFLINE=true` before Cargo reads the prepared sources.

Crate builds MUST remain network-independent.

The `cvc5-sys` crate derivation MUST receive the fixed cvc5 tree and libclang. It MUST NOT fetch cvc5 during the build.

`chelis-cli`, `chelis-compiler-api`, `chelis-cove`, and `tree-sitter-chelis` MUST receive a workspace source view for external compile-time assets.

Other crates MUST keep crate-local sources.

The native check set MUST inspect the generated crate2nix artifact and the required workspace members. Generation or import failure MUST fail the check.

Each native Nix job MUST evaluate the graph for its matching system. Cross-system inventory checks MUST not require a foreign-system generator build.

#### Scenario: Separate product members share crate outputs
- **WHEN** Nix builds the compiler, runtime, and installer packages
- **THEN** crate2nix builds their common dependencies as shared crate derivations

#### Scenario: A Cargo graph input changes
- **WHEN** `Cargo.lock`, a workspace manifest, or workspace source changes
- **THEN** Nix gives the generator a new derivation identity
- **AND** package evaluation imports the new graph without a repository refresh

#### Scenario: The repository contains a generated graph
- **WHEN** the repository tracks `Cargo.nix` or a graph-input digest
- **THEN** the flake contract test fails

#### Scenario: The cvc5 override is absent
- **WHEN** the `cvc5-sys` crate does not receive the fixed cvc5 tree
- **THEN** the SMT compiler build fails without a network fallback

#### Scenario: A crate loses an external compile-time asset
- **WHEN** one listed crate receives only its crate directory
- **THEN** the crate build fails because a required header or grammar file is absent

#### Scenario: Import from derivation is disabled
- **WHEN** the Nix configuration prohibits import from derivation
- **THEN** package evaluation fails before crate compilation

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
`flake.lock` and `devenv.lock` MUST pin the same Nixpkgs source revision and Rust overlay revision.

The repository MUST provide a parity checker with positive and negative tests. A mismatch MUST fail before a package build can count as accepted.

#### Scenario: Shared pins match
- **WHEN** the parity checker reads both committed lock files
- **THEN** it exits with status 0 and reports matching shared input revisions

#### Scenario: One Nixpkgs pin changes
- **WHEN** a fixture changes the Nixpkgs revision in only one lock file
- **THEN** the checker exits with a nonzero status and names the Nixpkgs nodes

#### Scenario: One Rust overlay pin changes
- **WHEN** a fixture changes the Rust overlay revision in only one lock file
- **THEN** the checker exits with a nonzero status and names the Rust overlay nodes

### Requirement: Native Nix checks protect every supported system
Each supported system MUST define checks for package construction, package contents, executable behavior, launcher shell lint, app paths, SMT activation, and lock parity.

The native Nix workflow MUST run the complete check set and the Nix flake contract suite on native `x86_64-linux` and `aarch64-darwin` builders when it is manually dispatched or a GitHub release is published. Each configured event MUST run both native jobs. Pull requests, pushes, and scheduled events MUST NOT invoke the native Nix workflow. A system MUST NOT count as supported from evaluation-only evidence.

The supported-system list and the named native CI jobs MUST have exact parity. The repository script suite MUST fail when either list contains an unmatched system.

The authoritative completion oracle MUST be both successful native check jobs from one manually dispatched or published-release workflow run.

#### Scenario: Linux checks pass
- **WHEN** CI runs the complete flake check set on `x86_64-linux`
- **THEN** every package and contract check exits with status 0

#### Scenario: Routine repository activity does not schedule native Nix
- **WHEN** a pull request, push, or scheduled event occurs
- **THEN** the native Nix workflow schedules neither supported-system job

#### Scenario: The generated launcher fails shell lint
- **WHEN** the built `chelisup` launcher fails `bash -n` or `shellcheck`
- **THEN** the launcher lint check fails the native check set

#### Scenario: A manual dispatch checks both supported systems
- **WHEN** a maintainer manually dispatches the native Nix workflow
- **THEN** both native jobs execute the complete check set and every package and contract check exits with status 0

#### Scenario: A published release checks both supported systems
- **WHEN** GitHub publishes a release
- **THEN** both native jobs execute the complete check set and every package and contract check exits with status 0

#### Scenario: One package check fails
- **WHEN** any required package or contract check fails on a supported system
- **THEN** the native system job fails and the change does not satisfy the completion oracle

#### Scenario: A supported system has no native job
- **WHEN** the supported-system list contains a system without a named native CI job
- **THEN** the repository script suite fails before the system can count as supported

### Requirement: Documentation explains the Nix channel boundary
Contributor documentation MUST show the package names and commands for `nix build`, `nix run`, and flake checks.

The documentation MUST list the supported systems. It MUST state that Nix is an additive source-build channel and that `chelisup` remains the release installer.

The documentation MUST NOT claim separate Nix packages for internal crates, the Python extension, or `chelis-std`.

#### Scenario: A Nix user reads the build instructions
- **WHEN** the user reads the source-build documentation
- **THEN** the documentation identifies every public package, app, supported system, and check command

#### Scenario: A user needs version routing
- **WHEN** the user reads the Nix installation boundary
- **THEN** the documentation directs release toolchain routing to `chelisup`
