# Cross-Platform Devenv Specification

## Purpose

Define the reproducible contributor shell, platform compiler commands, Python environment, smoke checks, and contributor documentation for Linux and macOS.

## Requirements

### Requirement: The repository provides reproducible Devenv inputs
The repository MUST track `devenv.nix`, `devenv.yaml`, `devenv.lock`, and the local modules under `devenv/`.

The lock file MUST pin all resolved input revisions.

`devenv.yaml` MUST pin the Devenv module input to release `v2.2.2`. `devenv.lock` MUST resolve that input to commit `b8030c58deafc013fc51791377fe8fea4dadcb00`.

`devenv.yaml` MUST set the exact string constraint `require_version: ">=2.2.0, <=2.2.2"`, which the CLI enforces before Nix evaluation. This is a closed, reviewed CLI compatibility range: the repository-owned atomic load-export task MUST replace the non-atomic upstream implementation throughout the range. Because the upstream `v2.2.2` tag builds a 2.2.2 CLI while its module source still advertises `latest-version` 2.2.1, the local Devenv configuration MUST also override `devenv.latestVersion` to 2.2.2. The Devenv CLI MUST reject versions below 2.2.0 and above 2.2.2.

`devenv.yaml` MUST pin the shared `nixpkgs` and `rust-overlay` inputs to exact commit revisions, not floating references. `devenv.lock` MUST resolve each shared input to its configured revision.

The repository MUST ignore `.devenv/` and `.devenv.flake.nix`. These paths contain generated local state, not reproducible inputs.

The repository lint policy MUST exclude both generated paths. It MUST keep the tracked root files and local modules in the editable corpus.

#### Scenario: The repository checks the Devenv release pin
- **WHEN** the configured URL or locked revision differs from Devenv `v2.2.2`
- **THEN** the static Devenv version contract test fails

#### Scenario: The CLI version is inside the reviewed range
- **WHEN** a contributor runs Devenv 2.2.0, 2.2.1, or 2.2.2
- **THEN** the CLI accepts the version requirement

#### Scenario: The pinned modules carry stale release metadata
- **WHEN** the `v2.2.2` modules advertise their upstream 2.2.1 `latest-version` value
- **THEN** the local module overrides that value to 2.2.2 before reporting its update target

#### Scenario: The CLI version is outside the reviewed range
- **WHEN** a contributor runs Devenv below 2.2.0 or above 2.2.2
- **THEN** the closed YAML constraint rejects the configuration before Nix evaluation or shell activation

#### Scenario: The CLI version requirement is absent
- **WHEN** `devenv.yaml` omits `require_version: ">=2.2.0, <=2.2.2"`
- **THEN** the static Devenv version contract test fails

#### Scenario: A shared input uses a floating reference
- **WHEN** `devenv.yaml` names a shared input by branch or tag instead of a full commit revision
- **THEN** the static Devenv version contract test fails

#### Scenario: A shared input lock drifts from its pin
- **WHEN** the locked revision of a shared input differs from the configured revision
- **THEN** the static Devenv version contract test fails

#### Scenario: A clean checkout evaluates the shell
- **WHEN** a contributor runs `devenv test` from a clean checkout
- **THEN** Devenv resolves the tracked inputs without an uncommitted input file

#### Scenario: Generated state stays untracked
- **WHEN** Devenv creates `.devenv/` or `.devenv.flake.nix`
- **THEN** Git ignores each generated path

#### Scenario: Generated state stays outside repository lint
- **WHEN** a contributor runs `chelis lint --check .` after Devenv creates local state
- **THEN** the lint excludes both generated paths and still checks the tracked Devenv inputs

### Requirement: The shell composes local configuration modules
The root `devenv.nix` MUST import these local modules:

- `./devenv/toolchains.nix`
- `./devenv/entry-shell.nix`
- `./devenv/commands.nix`
- `./devenv/generated-files.nix`
- `./devenv/git-hooks.nix`
- `./devenv/smoke-tests.nix`
- `./devenv/package-outputs.nix`
- `./devenv/release-outputs.nix`

`devenv.yaml` MUST NOT declare local module imports. Those imports belong in `devenv.nix`.

`devenv.yaml` can declare remote imports for shared configuration. It MUST remain the owner of Devenv inputs and CLI options.

The root `devenv.nix` MUST contain only the module imports. Each local module MUST own one configuration concern.

#### Scenario: A local module is absent
- **WHEN** a required local import is absent or duplicated in `devenv.nix`
- **THEN** the static Devenv composition test fails

#### Scenario: YAML declares a local import
- **WHEN** `devenv.yaml` lists a `./`, `../`, or `.nix` import entry
- **THEN** the static Devenv composition test fails

#### Scenario: An evaluator reads the root module
- **WHEN** a Devenv evaluator reads `devenv.nix` without the root YAML imports
- **THEN** it discovers all eight local modules

#### Scenario: A contributor evaluates the composed shell
- **WHEN** a contributor runs `devenv test`
- **THEN** Devenv combines all eight local modules
- **AND** the composed configuration passes the shell contract

### Requirement: Devenv exposes the canonical Nix packages
The `devenv/package-outputs.nix` module MUST define `outputs.chelis`, `outputs.chelis-runtime`, `outputs.chelisup`, and `outputs.default`.

Each Devenv output MUST reuse the corresponding package from the root flake for the native system. `outputs.default` MUST remain the same derivation as `outputs.chelis`.

The Devenv module MUST NOT define a second crate2nix graph or an independent package derivation. It MUST evaluate the root flake through a local Git input. It MUST NOT copy ignored build directories into the Nix store.

#### Scenario: A contributor builds one Devenv output
- **WHEN** a contributor runs `devenv build outputs.chelis`
- **THEN** Devenv builds the same derivation as `nix build .#chelis`

#### Scenario: A contributor builds all Devenv outputs
- **WHEN** a contributor runs `devenv build`
- **THEN** Devenv builds `chelis`, `chelis-runtime`, `chelisup`, and the default alias

#### Scenario: A Devenv output selects another flake surface
- **WHEN** the package-output module selects flake checks, apps, or an independent package graph
- **THEN** the static Devenv composition test fails

#### Scenario: A Devenv output imports the unfiltered repository
- **WHEN** the package-output module passes the complete worktree to `builtins.getFlake`
- **THEN** the static Devenv composition test fails before Nix copies ignored build directories

### Requirement: The shell composes shared ecosystem tools from ci
The shell MUST obtain crate2nix and OpenSpec from the shared ci consumer Devenv module, composed through `devenv.yaml` (a `ci` input plus the `ci/devenv/consumer` import). Neither tool may be pinned or built independently in this repository.

The shell MUST expose `config.outputs.crate2nix` and `config.outputs.openspec`, built with the repository's own `pkgs`. It MUST provide a `regenerate-crate2nix` command that refreshes or `--check`s the committed `Cargo.nix` graph using `config.outputs.crate2nix`. The Devenv smoke check MUST run the freshness `--check` and MUST verify OpenSpec reports the pinned version.

#### Scenario: The shell builds the shared tools
- **WHEN** a contributor runs `devenv build outputs.crate2nix outputs.openspec`
- **THEN** Devenv builds crate2nix and OpenSpec from the ci consumer module

#### Scenario: The repository pins a shared tool independently
- **WHEN** the shell builds or pins its own crate2nix or OpenSpec instead of composing ci
- **THEN** the static Devenv composition test fails

#### Scenario: The committed graph drifts
- **WHEN** the committed `Cargo.nix` no longer matches a fresh crate2nix generation
- **THEN** the `regenerate-crate2nix --check` smoke task fails

### Requirement: The repository catalogs inactive Git hooks
The repository MUST declare the `git-hooks` input in `devenv.yaml`. The input MUST follow the configured `nixpkgs` input.

The repository MUST declare this Git hook catalog in `devenv/git-hooks.nix`:

- `actionlint`
- `check-added-large-files`
- `check-case-conflicts`
- `check-executables-have-shebangs`
- `check-json`
- `check-merge-conflicts`
- `check-python`
- `check-symlinks`
- `check-toml`
- `check-yaml`
- `detect-private-keys`
- `end-of-file-fixer`
- `fix-byte-order-marker`
- `forbid-new-submodules`
- `mixed-line-endings`
- `nixfmt`
- `rustfmt`
- `shellcheck`
- `trim-trailing-whitespace`

Each listed catalog entry MUST set `enable = false`. Devenv MUST NOT install or run a listed catalog hook by default.

The `nixfmt` entry MUST remain inactive. The `rustfmt` entry MUST use check mode.

The `shellcheck` entry MUST select only `crates/chelisup/bootstrap/chelisup.sh`. The whitespace entry MUST preserve Markdown line breaks.

Devenv MUST NOT define an enabled hook that installs a `commit-msg` hook of its own. An installer that records an absolute path to one worktree governs every worktree of a clone, because they share one hooks directory.

The repository MUST track a `commit-msg` hook at `.githooks/commit-msg`. It MUST resolve its repository at run time and MUST NOT name any single worktree.

The tracked hook MUST invoke `scripts/check_commit_message.py` with a repository-managed Python interpreter. It MUST reject prohibited AI authorship markers, MUST accept ordinary commit messages, and MUST report the matched marker when it rejects a message.

Devenv MUST copy the tracked hook into the shared hooks directory for contributors who use the Devenv shell. It MUST resolve that directory with `git rev-parse --git-common-dir` so a linked worktree installs to the same place. It MUST NOT reach the hook through `core.hooksPath`, which is repository-scoped while a tracked file is branch-scoped: a worktree on a branch without the file would then run no hook and accept the commit silently.

Devenv MUST report when `core.hooksPath` is set to a location other than the shared hooks directory, because the installed hook cannot run there.

The repository MUST retain the cargo-husky user hook and development dependency for contributors who use the manual setup.

The cargo-husky hook MUST use minimal POSIX `sh`. It MUST invoke `scripts/check_commit_message.py` with a repository-managed Python interpreter.

The repository MUST ignore `.pre-commit-config.yaml`. Devenv generates this local file only while a catalog hook is enabled; with every entry disabled it generates none.

#### Scenario: A contributor enters the shell with the hook catalog
- **WHEN** a contributor runs `devenv shell`
- **THEN** no listed catalog hook is active
- **AND** Devenv copies the tracked `commit-msg` hook into the shared hooks directory

#### Scenario: A contributor runs tests through the manual setup
- **WHEN** a contributor runs `cargo test` outside Devenv
- **THEN** cargo-husky installs the `commit-msg` hook
- **AND** the hook invokes the shared checker with a repository-managed Python interpreter

#### Scenario: A listed catalog entry becomes active without a policy change
- **WHEN** any listed catalog entry sets `enable = true`
- **THEN** the static Devenv contract test fails

#### Scenario: A commit message contains an AI authorship marker
- **WHEN** the custom hook receives that commit message
- **THEN** the hook exits with a nonzero status
- **AND** the diagnostic identifies the matched marker

#### Scenario: A commit message contains no AI authorship marker
- **WHEN** the custom hook receives that commit message
- **THEN** the hook exits with status 0

### Requirement: The shell provides platform-correct compiler commands
On macOS, the shell MUST provide `gcc` and `g++` commands that invoke the Nixpkgs Darwin compiler wrappers from `pkgs.stdenv.cc`.

The shell MUST NOT use host compiler aliases as the shim implementation. These aliases are not pinned by the Devenv inputs.

The macOS shims MUST suppress only the unused wrapper-argument warning. Other warnings MUST remain errors when a caller passes `-Werror`.

On Linux, the shell MUST provide GCC from Nixpkgs. Linux MUST NOT use the macOS compiler shims.

On Apple silicon macOS, Rust build scripts targeting `aarch64-apple-darwin` MUST receive target-specific `CC` and `CXX` values that name the SDK-aware Nixpkgs compiler wrapper. Because the build is native, the shell MUST set `CRATE_CC_NO_DEFAULTS=1` so cc-rs does not add the redundant `arm64-apple-macosx` target alias. A clean `tree-sitter-chelis` build MUST NOT emit the Nix cc-wrapper multi-target warning, and the tree-sitter parser agreement suite MUST remain green.

#### Scenario: macOS compiles C through the expected command
- **WHEN** a macOS contributor compiles valid C with `gcc -Werror -fsyntax-only`
- **THEN** the pinned Nixpkgs C compiler wrapper succeeds

#### Scenario: macOS compiles C++ through the expected command
- **WHEN** a macOS contributor compiles valid C++17 with `g++ -std=c++17 -Werror -c`
- **THEN** the pinned Nixpkgs C++ compiler wrapper succeeds

#### Scenario: Host aliases do not satisfy the shell contract
- **WHEN** the macOS shims are absent and host `gcc` or `g++` aliases exist
- **THEN** `devenv test` rejects the host aliases

#### Scenario: Code warnings remain errors
- **WHEN** a macOS contributor compiles source with a deliberate warning and passes `-Werror`
- **THEN** the compiler command fails despite the wrapper-warning suppression

#### Scenario: Linux uses GCC
- **WHEN** a Linux contributor runs `gcc --version` inside the shell
- **THEN** the command reports the GCC package from Nixpkgs

#### Scenario: tree-sitter builds for the native Darwin target
- **WHEN** the Darwin smoke task builds `tree-sitter-chelis` in a fresh target directory
- **THEN** all generated C and C++ parser sources compile through the target-specific SDK-aware Nixpkgs compiler wrapper
- **AND** no cc-wrapper target-mismatch warning appears
- **AND** the parser agreement tests pass

### Requirement: The shell provides the Chelis development tools
The common shell MUST provide Rust from `rust-toolchain.toml`, rust-analyzer, Python 3.11, uv, `cargo-nextest`, `cargo-llvm-cov`, CMake, Git, mdBook 0.5.2, Pyright, pkg-config, and the repository-owned Kache 0.16.0 package.

The Rust module MUST set `languages.rust.toolchainFile`. It MUST provide rust-analyzer through `languages.rust.lsp.package`.

The Linux shell MUST also provide GCC, OpenBLAS, and Valgrind. The Linux Clarabel feature MUST link this OpenBLAS package through `openblas-src/system`. It MUST NOT build a separate OpenBLAS source tree. The macOS shell MUST use the system Accelerate framework instead of OpenBLAS.

#### Scenario: Common tools are available
- **WHEN** a contributor enters the shell on Linux or macOS
- **THEN** each common tool responds to its version command

#### Scenario: The Rust language server package is absent
- **WHEN** the Rust module does not define `languages.rust.lsp.package`
- **THEN** the static Devenv version test fails

#### Scenario: Linux tools are available
- **WHEN** a contributor enters the shell on Linux
- **THEN** GCC, OpenBLAS, and Valgrind are available from the Nix environment

#### Scenario: Clarabel builds on Linux
- **WHEN** CI activates the Clarabel feature in the Linux shell
- **THEN** Cargo links the OpenBLAS package from Devenv
- **AND** Cargo does not build an OpenBLAS source tree

#### Scenario: macOS omits the Linux BLAS package
- **WHEN** a contributor evaluates the package list on macOS
- **THEN** the shell does not add OpenBLAS or Valgrind

### Requirement: Devenv provides focused CI profiles
The shell MUST define `ci`, `sanitizers`, and `smt` profiles.

The `ci` profile MUST set development and test Cargo debug information to zero. On Linux, it MUST request a GNU linker build ID for Rust binaries. This linker flag MUST NOT apply on macOS. The `sanitizers` profile MUST extend `ci`.

The `sanitizers` profile MUST own the C sanitizer flags and runtime options. The sanitizer flags MUST include optimization for the Nix fortify contract.

The `smt` profile MUST extend `ci`. It MUST set `CVC5_DIR` from the lazy `outputs.cvc5-dir` Devenv output.

The base shell MUST NOT set `CVC5_DIR`. Thus, ordinary shell activation does not force the cvc5 closure.

#### Scenario: A contributor reproduces the CI Cargo environment
- **WHEN** a contributor activates the `ci` profile
- **THEN** Cargo development and test builds omit debug information

#### Scenario: Linux CI builds a Rust test binary
- **WHEN** Cargo links a Rust test binary through the Linux `ci` profile
- **THEN** the binary contains a GNU linker build ID
- **AND** image identification uses the linker-ID fast path

#### Scenario: A contributor runs the sanitizer profile
- **WHEN** a contributor activates the `sanitizers` profile
- **THEN** the profile provides the complete sanitizer environment
- **AND** it retains the CI Cargo policy

#### Scenario: A contributor runs the SMT profile
- **WHEN** a contributor activates the `smt` profile
- **THEN** `CVC5_DIR` identifies the pinned cvc5 prebuilt tree
- **AND** it retains the CI Cargo policy

#### Scenario: A contributor enters the base shell
- **WHEN** no SMT profile is active
- **THEN** the shell does not add the cvc5 output to the environment
### Requirement: The shell owns its Kache compiler wrapper
The shell MUST build Kache 0.16.0 from source commit `a21d020142b1248537cd548ccde99a04c0a44820` and fixed-output source and Cargo hashes. It MUST apply the repository patches `nix/patches/kache-0.16.0-chelis-contract.patch` and `nix/patches/kache-0.16.0-relocatable-macos-executables.patch`, and the Nix derivation MUST run Kache's upstream unit tests.

The shell MUST set `RUSTC_WRAPPER` to that exact package's `kache` binary, set `KACHE_CONFIG` to the tracked `.kache.toml`, and force `KACHE_DISABLED=0`. The tracked policy MUST ignore file-backed `KACHE_*` overrides, cache user-facing executables, and retain a 30-second heartbeat.

The managed wrapper MUST take precedence over an absent, differently versioned, or hostile user Cargo wrapper configuration. The ordinary non-Devenv Cargo workflow MAY continue to honor user configuration.

Kache doctor MUST exit nonzero when it reports a genuine issue. An intentionally absent local-only daemon MAY remain informational, but stale locks MUST remain a genuine issue. Heartbeats MUST show a positive ETA only below the historical typical duration, MUST say when that duration is reached, and MUST report elapsed time over typical after it is exceeded.

The executable-cache regression MUST start with a fresh probe-local Kache store shared only by its cold and warm clean checkouts. Each checkout MUST own its Cargo target beneath its verified workspace root so Kache can normalize workspace identity. For every warm local-hit event, its exact cache key MUST resolve to local entry metadata with the same key and crate identity. Only a warm target candidate whose exact filename is marked executable by that entry metadata counts as restored. The regression MUST reject the executable-bypass disposition and scan every classified restored executable and its debug bundle for the cold checkout path. On Darwin, every classified executable MUST have an adjacent dSYM bundle; `dwarfdump --uuid` MUST read both members of every pair and return identical, nonempty UUID sets.

On macOS, a cached debug executable MUST use a private `strip -S` store copy after Kache has produced its self-contained dSYM. The cached dSYM MUST omit donor-path relocation metadata, and Kache MUST skip cache publication when it cannot produce the path-clean executable/dSYM pair. The compiler-owned cold output MUST remain untouched.

The relocatable executable/dSYM representation MUST use cache-key schema 28 rather than Kache 0.16.0's upstream schema 27. A consumer using the relocatable representation MUST therefore miss every entry produced under schema 27 instead of restoring an executable or dSYM that predates path sanitization. The executable-cache regression MUST reject restored executable metadata carrying any schema other than 28, then prove that a current-producer/current-consumer hit remains path-clean and UUID-matched.

The managed environment MUST expose an oracle-only `KACHE_SCHEMA_27_WRAPPER` built from the same pinned Kache source with the Chelis doctor/heartbeat contract patch but without the schema-28 relocation patch. This fixture MUST NOT be selected as Cargo's active wrapper. Its only supported use is producing the legacy entry for the executable-cache representation-boundary regression.

#### Scenario: Host Cargo configuration names another wrapper
- **WHEN** the smoke control supplies a user Cargo config naming an absent host wrapper
- **THEN** Cargo still compiles through the exact repository-owned `RUSTC_WRAPPER`

#### Scenario: Correctness does not depend on a cache hit
- **WHEN** the same smoke fixture compiles with no Rust compiler wrapper
- **THEN** the build succeeds with the same source contract

#### Scenario: A clean warm checkout restores a test executable
- **WHEN** the executable-cache regression builds the selected suites in its cold checkout and then in its warm checkout
- **THEN** both runs share one otherwise-fresh Kache store and distinct checkout-owned Cargo targets
- **AND** a warm local-hit cache key resolves to metadata that marks the exact warm-target executable filename
- **AND** the warm report contains restored bytes and no executable-cache bypass
- **AND** neither a classified restored executable nor its debug bundle contains the cold checkout path
- **AND** every restored Darwin executable and dSYM bundle has the same nonempty UUID set

#### Scenario: A legacy executable entry cannot cross the representation boundary
- **WHEN** the schema-27 fixture populates an otherwise-fresh probe store and the schema-28 wrapper compiles the same source from a clean checkout
- **THEN** the schema-28 report contains current-consumer misses and no local hit
- **AND** the probe store contains distinct schema-27 and schema-28 entries
- **AND** a subsequent schema-28 producer/consumer pair still satisfies the exact executable, path-clean, and Darwin UUID checks

#### Scenario: Darwin debug metadata is absent, malformed, or mismatched
- **WHEN** a classified restored executable lacks an adjacent dSYM bundle, either artifact yields no Mach-O UUID, or their UUID sets differ
- **THEN** the executable-cache regression fails closed

#### Scenario: Doctor finds a genuine issue
- **WHEN** any nondowngraded doctor check fails
- **THEN** doctor reports the issue count and exits nonzero

#### Scenario: A compile exceeds its historical duration
- **WHEN** heartbeat elapsed time is greater than the typical duration
- **THEN** the heartbeat reports time over typical
- **AND** it does not render a positive ETA

### Requirement: Python analysis has a closed repository-owned source scope
The repository MUST track `pyrightconfig.json`. Its include roots MUST cover every tracked Python source and test, and its excludes MUST reject Devenv state, virtual environments, Cargo targets, Node modules, Git metadata, bytecode caches, and directory-symlink escapes.

The executable scope checker MUST fail when a tracked Python file lies outside the include roots, when an include root can escape through a directory symlink, or when a required generated-tree exclusion is absent.

#### Scenario: Pyright enumerates the repository
- **WHEN** a contributor runs the Pyright scope smoke task
- **THEN** every tracked Python file belongs to an intended include root
- **AND** no generated environment, target, sibling worktree, or Nix-store tree belongs to the analysis source set

#### Scenario: A new Python root is unclassified
- **WHEN** a tracked Python file is added outside every include root
- **THEN** the scope checker exits nonzero and names the file

### Requirement: Devenv manages the Python environment
The shell MUST enable Python 3.11, uv, and the Devenv Python virtual environment.

Devenv MUST store the environment at `$DEVENV_STATE/venv`. It MUST activate that environment before `devenv:enterShell`.

The shell MUST set `PYO3_PYTHON` to `$DEVENV_STATE/venv/bin/python`. It MUST NOT use the system Python for PyO3.

Devenv MUST NOT create, replace, or modify the repository-root `.venv`. The manual setup path owns that environment outside Devenv.

#### Scenario: Devenv creates its Python environment
- **WHEN** a contributor enters the shell without `$DEVENV_STATE/venv/bin/python`
- **THEN** uv creates the environment with Python 3.11
- **AND** Devenv activates the environment

#### Scenario: PyO3 uses the active Devenv interpreter
- **WHEN** a contributor builds a crate that uses PyO3 inside the shell
- **THEN** `PYO3_PYTHON` names the active Devenv virtual environment interpreter

#### Scenario: A manual environment exists
- **WHEN** a repository-root `.venv` exists before shell activation
- **THEN** Devenv preserves that environment without a change

### Requirement: The shell provides developer commands
The shell MUST provide these commands through Devenv scripts:

- `chelis-gate`
- `chelis-reap-orphans`
- `chelis-exec-preflight` on macOS
- `chelis-z3-test` on Linux
- `chelis-hip-test` on Linux

Each command MUST use the configured Devenv Python package. Each command MUST forward all arguments to its existing tested Python file. Each command MUST run that file under the active Devenv virtual environment interpreter when one exists, resolved at run time rather than from an interpolated path.

The command definitions MUST NOT duplicate the Python implementation inside Nix.

#### Scenario: A contributor runs a command
- **WHEN** a contributor runs a supported command in the shell
- **THEN** Devenv invokes its repository Python file with all supplied arguments

#### Scenario: A command runs under the active interpreter
- **WHEN** a contributor runs a supported command in the shell
- **THEN** the repository Python file runs under the active Devenv virtual environment interpreter

#### Scenario: A platform-specific command is unavailable
- **WHEN** a contributor enters the shell on an unsupported platform
- **THEN** Devenv omits that platform-specific command

### Requirement: Devenv creates compiler probe files
Devenv MUST create these read-only compiler probe files from `devenv/generated-files.nix`:

- `.devenv/generated/compiler-probes/valid.c`
- `.devenv/generated/compiler-probes/warning.c`
- `.devenv/generated/compiler-probes/valid.cpp`

The compiler smoke tasks MUST compile these files. They MUST NOT create equivalent source with inline shell heredocs.

#### Scenario: Devenv materializes compiler probes
- **WHEN** a contributor enters the shell
- **THEN** each compiler probe is a Devenv-managed symlink
- **AND** each probe contains its declared source

#### Scenario: A compiler probe is absent
- **WHEN** a required compiler probe is absent from the declarative file module
- **THEN** the static Devenv composition test fails

### Requirement: The shell checks its compiler contract
`devenv test` MUST check the required tool versions and the Python interpreter version.

It MUST compile valid C and C++ translation units under `-Werror`. It MUST also prove that a deliberate code warning still fails.

The smoke checks MUST run before `devenv:enterTest` as these named tasks. The Python, Kache, Pyright, and Darwin tree-sitter tasks MUST run after `devenv:python:virtualenv`. The C and C++ compiler tasks MUST run after `devenv:files`, which materializes their declared compiler probes. Those prerequisite edges MUST be explicit so every accepted Devenv CLI schedules consumers only after their generated inputs exist. The tasks MUST NOT declare themselves after `devenv:enterShell`, because that makes ordinary shell entry execute the acceptance suite before admitting a command:

- `chelis:toolchain-test`
- `chelis:python-test`
- `chelis:c-compiler-test`
- `chelis:cpp-compiler-test`
- `chelis:kache-test`
- `chelis:pyright-test`
- `chelis:docs-test`
- `chelis:darwin-tree-sitter-test`

The smoke-check graph MUST NOT start a Devenv service or long-running process.

#### Scenario: The shell smoke check passes
- **WHEN** all required tools and compiler commands satisfy the contract
- **THEN** `devenv test` runs all eight named tasks
- **AND** each task reports success
- **AND** `devenv test` exits with status 0

#### Scenario: A smoke task can race its generated prerequisite
- **WHEN** a Python-backed task does not follow `devenv:python:virtualenv` or a compiler task does not follow `devenv:files`
- **THEN** the static Devenv composition test fails
- **AND** the task graph cannot count as acceptance evidence

#### Scenario: A managed compiler command is missing
- **WHEN** either `gcc` or `g++` does not resolve to a Nix store package
- **THEN** `devenv test` exits with a nonzero status and names the unmanaged command

#### Scenario: The shim hides a code warning
- **WHEN** the deliberate warning compiles successfully under `-Werror`
- **THEN** `devenv test` exits with a nonzero status

### Requirement: Concurrent shell entry publishes one complete export payload
The repository MUST override `devenv:enterShell` so `$DEVENV_DOTFILE/load-exports` is written to a same-directory temporary file, flushed, and installed by atomic rename with executable mode. Concurrent writers MUST expose either the previous complete payload or one new complete payload, never a missing or partial file.

A committed regression MUST launch concurrent noninteractive entries in an isolated temporary checkout. Each entry MUST either complete its entry dependencies and run its payload with exit zero, or exit nonzero without running the payload. A task failure followed by a successful payload is a failure.

#### Scenario: Shell entries overlap
- **WHEN** eight warm noninteractive Devenv entries start concurrently
- **THEN** no entry reports a missing `load-exports`
- **AND** no entry runs its payload after an entry-task failure

### Requirement: Repeated commands have a supported low-overhead path
The supported low-overhead path for repeated work MUST be one persistent `devenv shell` session. Entry checks run once before that session admits commands; subsequent commands inherit the exact activated toolchain without reevaluating or reactivating the shell.

For a single noninteractive command after one successful shell realization, contributors MAY use `devenv shell --no-reload -- <command>`. They MUST rerun normal entry after changing Devenv inputs or modules.

The repository MUST provide a benchmark that reports wall, user CPU, and system CPU separately for ordinary entry, `--no-reload`, task-skipped activation diagnosis, and payloads repeated inside one persistent session. The report MUST decompose reload/evaluation, activation, entry-task, and payload estimates without claiming that wall time equals runner consumption.

#### Scenario: A contributor runs repeated focused commands
- **WHEN** the contributor enters one persistent Devenv shell and runs multiple commands
- **THEN** every command inherits the activated environment
- **AND** the fixed Devenv entry cost is paid once rather than once per command

#### Scenario: The benchmark runs
- **WHEN** the startup benchmark collects at least three warm samples per mode
- **THEN** it reports wall and CPU medians separately
- **AND** it reports the persistent payload cost and the four-part decomposition

### Requirement: Native package CI uses the reviewed portable Devenv base
Each native Nix package job MUST invoke `Chelis-Lang/ci/actions/setup-devenv@<revision>` once before its runner verification step. `<revision>` MUST be one immutable 40-character commit SHA.

That SHA MUST equal the configured and locked ci Devenv input revision. Each native job MUST invoke `authenticate-private-ci-input` at the same revision before its first project Devenv command.

Each job MUST use `devenv-ci bash --noprofile --norc -e -o pipefail {0}` as its default shell for `run` steps. The Linux job can invoke `reclaim-ubuntu-runner-disk` before setup with its explicit host shell.

The workflow MUST NOT duplicate direct Nix, Cachix, Devenv, App-token, or Nix-authentication bootstrap. The reviewed setup action supplies the pinned Nix and Devenv versions.

Each job MUST run project Python helpers through `devenv-retry --profile ci shell --no-tui -- python`.

Runner-system and sandbox probes MUST use the portable default shell. These probes MUST NOT start a nested project environment.

Each job MUST NOT install host uv, create the repository-root `.venv`, or invoke a direct Devenv-state interpreter path.

Each job MUST run `devenv-retry test --no-tui` and `devenv-retry build --no-tui outputs.chelis outputs.chelis-runtime outputs.chelisup`. The Devenv cache MUST NOT replace the complete native flake check.

The public Devenv cache does not contain the custom Chelis cvc5 derivation. Each job can reuse the prebuilt cvc5 toolchain closure from the repository Actions cache, keyed by the closure derivation name.

#### Scenario: Native CI checks the development shell
- **WHEN** either native Nix package job runs
- **THEN** the job invokes setup and fixed-scope authentication at the ci input revision
- **AND** each Nix and Devenv `run` step uses the portable shell
- **AND** each project Python helper uses the activated project interpreter
- **AND** each runner probe uses the portable Devenv interpreter
- **AND** the job runs all nine named Devenv tasks
- **AND** the job builds all three named Devenv package outputs
- **AND** the job runs the complete native flake check

#### Scenario: Native CI bypasses the portable base
- **WHEN** a job omits setup, authentication, the portable shell, or the shared revision
- **THEN** the native workflow contract fails

#### Scenario: Native CI recreates a Python environment
- **WHEN** a job invokes host uv, creates `.venv`, or names a direct Devenv-state interpreter
- **THEN** the native workflow contract fails

#### Scenario: The Linux runner needs disk reclamation
- **WHEN** the Linux native package job starts on a GitHub-hosted Ubuntu runner
- **THEN** the shared fixed disk action runs before `setup-devenv`

#### Scenario: The custom cvc5 output is absent from public caches
- **WHEN** the native flake check requires the custom non-GPL cvc5 derivation and the repository closure cache has no entry
- **THEN** Nix builds that derivation from source

### Requirement: The shell runs the C backend test tier on macOS
The macOS shell MUST run the existing `chelis-backend-c` tests that invoke `gcc` and `g++`.

The change MUST NOT disable, ignore, or weaken a test to make this tier pass.

#### Scenario: The C backend suite passes in the shell
- **WHEN** a macOS contributor runs `devenv shell -- cargo nextest run -p chelis-backend-c`
- **THEN** the suite exits with status 0 and compiler-dependent tests run with the shell commands

#### Scenario: The compiler shims are absent
- **WHEN** the same suite runs in a macOS shell without the compiler shims
- **THEN** the compiler smoke checks fail before the result can count as acceptance evidence

### Requirement: Contributor documentation describes the optional shell
The contributor documentation MUST show the Devenv activation, smoke-check, package-output, and developer commands.

It MUST state that the macOS command shims invoke the Nixpkgs clang wrapper from `pkgs.stdenv.cc`, not host Apple clang.

The documentation MUST keep the manual setup path. It MUST describe Devenv as optional for local work and required by native Nix CI only.

It MUST NOT describe Devenv as a product requirement.

#### Scenario: A contributor selects Devenv
- **WHEN** a contributor reads the source-build prerequisites
- **THEN** the documentation provides the Devenv activation, smoke-check, package-output, and backend-test commands

#### Scenario: A contributor selects manual setup
- **WHEN** a contributor does not use Devenv
- **THEN** the existing rustup, uv, and platform C-toolchain instructions remain available
