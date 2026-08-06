## Context

Devenv documents `outputs` as its package interface. It also documents language import functions as the standard package source for each language.

The Rust import function gets crate2nix through `config.lib.getInput`. It also uses `languages.rust.toolchainPackage` for Cargo and rustc.

The function returns `cargoNix.rootCrate.build`. Chelis has a virtual Cargo workspace, so its crate2nix graph exposes product crates only under `workspaceMembers`.

Chelis also requires a fixed cvc5 tree, libclang, workspace source overrides, SMT features, runtime headers, and the closure-safe chelisup launcher.

The current Devenv package module calls `builtins.getFlake` on the repository. This call preserves package identity, but it makes Devenv depend on the flake package path.

Devenv documents custom local modules as the extension mechanism for project policy. It also supports direct outputs and typed output options.

## Goals / Non-Goals

**Goals:**

- Make Devenv own the Rust package graph for `devenv build`.
- Use Devenv's crate2nix input and configured Rust toolchain.
- Generate one graph for the complete Chelis virtual workspace.
- Select the three product crates from `workspaceMembers`.
- Preserve SMT support, source overrides, package layouts, and launcher behavior.
- Keep the root flake as the public downstream Nix API.
- Share package rules between the flake and Devenv paths.
- Keep ordinary `devenv shell` and `devenv test` evaluation lazy.
- Fail before compilation when an input, member, override, or package contract is absent.

**Non-Goals:**

- Do not add a `[package]` table to the root Cargo manifest.
- Do not split the Cargo workspace into standalone package trees.
- Do not fork, copy, disable, or replace the Devenv Rust module.
- Do not change crate2nix.
- Do not remove the root flake or its public apps and checks.
- Do not guarantee equal Nix store paths for dirty worktrees.
- Do not change any language, compiler, runtime, CLI, backend, ABI, or generated-code contract.
- Do not add packages for internal crates, Python, documentation, or `chelis-std`.

## Decisions

### D1: Add a Chelis workspace import as a local Devenv extension

Add `devenv/rust-workspace.nix`. This module defines `config.chelis.rust.importWorkspace` as a lazy function.

The function accepts a workspace path and an exact argument set. Unknown arguments fail at the function boundary.

The function returns the crate2nix graph. It does not return one package and does not select `rootCrate`.

The option uses a Chelis namespace because the built-in Devenv API does not define a workspace graph contract.

A future Devenv release can replace this extension after it exposes an equivalent graph API.

An alternative copies and replaces `languages/rust.nix`. That choice duplicates the complete Devenv Rust module and creates pin drift.

An alternative changes `languages.rust.import` with `lib.mkForce`. That choice changes a built-in contract and still cannot express a graph return type.

### D2: Use Devenv-owned inputs and the configured toolchain

The extension obtains crate2nix through `config.lib.getInput`. `devenv.yaml` pins crate2nix 0.15.0 as a non-flake input.

The graph builder uses `config.languages.rust.toolchainPackage` for Cargo and rustc. It does not construct another Rust toolchain.

The parity checker compares the crate2nix, Nixpkgs, and Rust overlay revisions across `devenv.lock` and `flake.lock`.

A missing input, a floating input, or a lock mismatch fails before a package build counts as evidence.

An alternative relies on the default crate2nix reference inside Devenv. That choice permits unreviewed version drift.

### D3: Share one workspace graph implementation

Extract the workspace graph rules from `nix/packages.nix` into a focused helper under `nix/`.

The helper receives these typed inputs:

- the package set
- the source root
- the crate2nix source
- the configured Rust toolchain
- the fixed cvc5 package set

The helper generates one crate2nix graph. It enables `chelis-cli/smt` and selects no product package itself.

The Devenv extension and the root flake adapter call the same helper. Each interface owns its input graph, but both interfaces share package rules.

The helper uses `cratePkgs.buildRustCrate`. It retains crate2nix's default overrides before it adds Chelis overrides.

An alternative calls `config.languages.rust.import` three times. That function requires `rootCrate`, and three calls duplicate graph generation.

An alternative imports `nix/packages.nix` directly from the output module. That path hides the workspace API and couples Devenv to the flake adapter.

### D4: Keep Chelis crate overrides in one helper

Move the `cvc5-sys` override and workspace source overrides into a shared Chelis override function.

The `cvc5-sys` override supplies the fixed cvc5 tree, libclang, and pkg-config. It prevents a network fallback.

The source override supplies workspace assets to these crates:

- `chelis-cli`
- `chelis-compiler-api`
- `chelis-cove`
- `tree-sitter-chelis`

Other crates retain crate-local sources.

The graph helper fails if a required workspace member or override is absent. It does not use a default package or an empty override set.

### D5: Share product artifact assembly

Extract product assembly into a helper that accepts the three selected crate derivations.

The helper creates these product packages:

- `chelis`, with the compiler, runtime archive, and public headers
- `chelis-runtime`, with only the runtime archive and public headers
- `chelisup`, with the launcher and real installer payload

The helper retains the current closure-safe chelisup launcher without a text copy in Devenv.

The root flake and Devenv output module call the same assembly helper. Package layout and behavior remain one contract.

Store path equality is supporting evidence only. Native package checks decide contract parity.

### D6: Expose direct Devenv outputs

`devenv/package-outputs.nix` forces the shared graph only when an output value is requested.

The module selects these members:

- `workspaceMembers."chelis-cli"`
- `workspaceMembers."chelis-runtime"`
- `workspaceMembers."chelisup"`

It defines `outputs.chelis`, `outputs.chelis-runtime`, `outputs.chelisup`, and `outputs.default`. The default value aliases the Devenv `chelis` derivation.

The module contains no `builtins.getFlake`, `rootCrate`, second graph call, or independent launcher text.

The static composition test rejects each forbidden path. Negative fixtures also remove one member, one override, and one output alias.

### D7: Preserve source filtering and lazy shell evaluation

The workspace helper uses the existing filtered source policy. It excludes `.git`, `.devenv`, `.venv`, `target`, and other generated directories.

The local function and output values remain lazy. `devenv shell` and `devenv test` do not force crate2nix graph generation.

`devenv build` forces the graph and imports its generated Nix expression. Import-from-derivation failure is a build failure.

A static test rejects an unfiltered repository path. A native probe confirms that shell evaluation does not build a product package.

### D8: Use native CI as the acceptance oracle

The authoritative completion oracle is the named native Nix package workflow on both supported systems.

The Linux job runs for code pull requests and pushes to `main`. The macOS job remains a documented manual dispatch gate.

Each native job runs these checks:

- Devenv input and composition tests
- the complete root flake check set
- `devenv test --no-tui`
- all four Devenv package outputs
- package layout and executable checks against the Devenv outputs
- SMT verification against the Devenv compiler

Local `devenv build --no-tui` output is supporting evidence. A local run cannot replace the required macOS hosted evidence.

## Risks / Trade-offs

**[The local extension drifts from a future Devenv workspace API]** → Use the `chelis` namespace and keep the adapter small. Remove it after an equivalent pinned API exists.

**[Two interface evaluations create different store paths]** → Share inputs, graph rules, overrides, and assembly. Test behavior and layout instead of dirty-worktree path identity.

**[Import from derivation increases evaluation cost]** → Keep one graph thunk and force it only for package outputs.

**[The cvc5 override disappears from one path]** → Keep one shared override function and run an SMT fixture against each native Devenv compiler.

**[A source filter removes a compile-time asset]** → Keep explicit workspace source overrides and negative fixtures for every covered crate.

**[A Devenv input drifts from the flake pin]** → Extend the lock parity checker and fail before package acceptance.

**[The built-in Rust helper later changes behavior]** → Do not override it. The Chelis extension remains isolated from that contract.

## Migration Plan

1. Add failing static tests for the workspace import, graph members, input pins, overrides, outputs, and forbidden self-flake path.
2. Add failing native fixtures for package layout, SMT support, and the default alias.
3. Pin crate2nix in Devenv and extend lock parity tests.
4. Extract the shared graph, override, and artifact assembly helpers.
5. Add the local Devenv workspace import module.
6. Replace the self-flake output module with workspace member selection.
7. Run the Linux native job and the manual macOS job.
8. Update specifications, contributor documentation, and the changelog.

Rollback removes the Devenv package output modules and their CI steps. The public root flake remains the stable package path during rollback.

## Open Questions

No question blocks implementation.

A separate upstream proposal can ask Devenv to expose a generic workspace graph import. This change does not depend on that proposal.
