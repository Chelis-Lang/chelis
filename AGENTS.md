# Chelis semantic core

This branch contains the existing production semantic core, not the Chelis product workspace. The canonical owner is `chelis-pipeline-core`; its complete local production dependency closure is `chelis-ir`, `chelis-types`, `chelis-effects`, `chelis-deep`, `chelis-pred`, `chelis-vocab`, and `chelis-unord`.

- Work in this dedicated worktree. Do not alter the primary checkout or another agent's files.
- Preserve the sealed checked-product, library-identity, root-order, effect, AD, and ownership contracts. Do not substitute miniature algorithms for production code.
- Frontend preparation, CLI/LSP, target emitters, runtime packaging, and ecosystem tooling are outside this repository. Tests use actual Deep/AST inputs, not frontend dependencies or mock parsers.
- Keep meaningful acceptance/rejection and executable behavior tests. Do not retain source-text, file-layout, wording-only, or empty passing tests.
- During parallel edits, each agent owns disjoint files and skips builds/tests/linters/formatters. The integration owner validates the completed workspace. Eval state is shared: namespace all persistent variables.

## Toolchains and checks

`rust-toolchain.toml` preserves the production Rust compiler pin. `flake.lock` pins the native development environment and the compatible Charon/Aeneas pair. `verification/toolchain.json` records the Rust source and Lean reference revisions. Charon uses its own matching compiler, independently of the native Rust pin.

```text
nix develop --command cargo test --workspace --locked --features chelis-pipeline-core/lowering-trace --jobs 2
nix develop --command cargo clippy --workspace --locked --all-targets --all-features --jobs 2 -- -D warnings
nix develop --command cargo fmt --all --check
```

The optional `generalize-sweep-oracle` feature is a separate diagnostic configuration. Preserve its assertions and report original-input mismatches; do not hide them by weakening tests or changing the production checker during fixture migration.

Use task-owned Cargo target directories for independent builds. Keep temporary tools and probes under `.work/`; do not publish them as core code. Python tooling must use the pinned environment, not an ambient interpreter.

Formal verification requires actual Charon extraction, Aeneas translation, and Lean-checked correspondence theorems. A generated file, passing numerical sample, or lowering trace is not that proof. Do not hide unsupported behavior with opaque shims, no-op replacements, feature gates, or new axioms. State reference-domain restrictions and numerical/runtime trust boundaries explicitly.
