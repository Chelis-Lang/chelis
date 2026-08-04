## ADDED Requirements

### Requirement: Chelis extends Devenv Rust imports for its virtual workspace
The repository MUST define a local Devenv module that provides `config.chelis.rust.importWorkspace`.

The function MUST accept the workspace path and an exact argument set. An unknown argument MUST fail at the function boundary.

The function MUST obtain crate2nix through `config.lib.getInput`. It MUST use `config.languages.rust.toolchainPackage` for Cargo and rustc.

The function MUST return one crate2nix graph for the complete filtered workspace. The graph MUST expose product crates through `workspaceMembers`.

The local module MUST NOT disable, replace, copy, or override the built-in Devenv Rust module. It MUST NOT require a root Cargo package.

#### Scenario: Devenv imports the Chelis workspace
- **WHEN** the package output module requests the Chelis workspace graph
- **THEN** the graph contains `chelis-cli`, `chelis-runtime`, and `chelisup` under `workspaceMembers`
- **AND** Cargo and rustc come from the configured Devenv toolchain

#### Scenario: A caller supplies an unknown import argument
- **WHEN** a caller passes an argument outside the declared workspace import argument set
- **THEN** Nix evaluation fails at the workspace import boundary

#### Scenario: The workspace adapter selects rootCrate
- **WHEN** the local module references `cargoNix.rootCrate`
- **THEN** the static Devenv composition test fails

#### Scenario: The built-in Rust module is replaced
- **WHEN** the configuration disables or overrides the pinned Devenv Rust module
- **THEN** the static Devenv composition test fails

## MODIFIED Requirements

### Requirement: Devenv exposes the canonical Nix packages
The `devenv/package-outputs.nix` module MUST define `outputs.chelis`, `outputs.chelis-runtime`, `outputs.chelisup`, and `outputs.default`.

The module MUST select `chelis-cli`, `chelis-runtime`, and `chelisup` from one Devenv-owned `workspaceMembers` graph for the native system.

Each Devenv output MUST satisfy the artifact and behavior contract of the corresponding root flake package. `outputs.default` MUST remain the same derivation as `outputs.chelis` within Devenv.

The Devenv module MUST use the shared Chelis crate overrides and artifact assembly. It MUST NOT evaluate the repository flake, call `languages.rust.import`, select `rootCrate`, or define a second graph.

The workspace source MUST use the repository source filter. It MUST NOT copy ignored build directories into the Nix store.

#### Scenario: A contributor builds one Devenv output
- **WHEN** a contributor runs `devenv build outputs.chelis`
- **THEN** Devenv builds the SMT-enabled compiler and runtime from the Devenv workspace graph
- **AND** the result satisfies the public `chelis` package contract

#### Scenario: A contributor builds all Devenv outputs
- **WHEN** a contributor runs `devenv build`
- **THEN** Devenv builds `chelis`, `chelis-runtime`, `chelisup`, and the default alias

#### Scenario: A Devenv output evaluates the root flake
- **WHEN** the package output module calls `builtins.getFlake` or selects a flake package
- **THEN** the static Devenv composition test fails

#### Scenario: A Devenv output creates another workspace graph
- **WHEN** the package output module calls the workspace generator more than once
- **THEN** the static Devenv composition test fails

#### Scenario: A required workspace member is absent
- **WHEN** the generated graph lacks `chelis-cli`, `chelis-runtime`, or `chelisup`
- **THEN** package evaluation fails before an output can count as accepted

#### Scenario: A Devenv output imports the unfiltered repository
- **WHEN** the workspace module passes an unfiltered worktree to crate2nix
- **THEN** the static Devenv composition test fails before Nix copies ignored build directories

### Requirement: The shell provides the Chelis development tools
The common shell MUST provide Rust from `rust-toolchain.toml`, rust-analyzer, Python 3.11, uv, `cargo-nextest`, `cargo-llvm-cov`, CMake, Git, pkg-config, and ShellCheck.

The Rust module MUST set `languages.rust.toolchainFile`. It MUST provide rust-analyzer through `languages.rust.lsp.package`.

The Linux shell MUST also provide GCC, OpenBLAS, and Valgrind. The macOS shell MUST use the system Accelerate framework instead of OpenBLAS.

#### Scenario: Common tools are available
- **WHEN** a contributor enters the shell on Linux or macOS
- **THEN** each common tool responds to its version command

#### Scenario: The Rust language server package is absent
- **WHEN** the Rust module does not define `languages.rust.lsp.package`
- **THEN** the static Devenv version test fails

#### Scenario: Linux tools are available
- **WHEN** a contributor enters the shell on Linux
- **THEN** GCC, OpenBLAS, and Valgrind are available from the Nix environment

#### Scenario: macOS omits the Linux BLAS package
- **WHEN** a contributor evaluates the package list on macOS
- **THEN** the shell does not add OpenBLAS or Valgrind
