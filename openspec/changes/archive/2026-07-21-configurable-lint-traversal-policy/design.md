## Context

The #603 change established one canonical walker result per lint invocation and made corpus-aware rules consume those admitted entries. The remaining walker policy is still encoded as path literals in `is_skip_dir`, while chelis#740 calls for a general lint-side ignore mechanism for generated or genuinely immutable trees. Raw `.gitignore` is not a valid contract: it represents version-control intent, may match tracked files differently from Git, and BurntSushi `ignore` defaults also consult hidden, parent, global, `.ignore`, and Git-exclude sources that can make local and CI lint different.

Chelis already requires rule exceptions to carry resolvable spec cross-references. Whole-tree exclusions are stronger than rule exceptions, so they need at least the same accountability and must remain visible as structured policy.

## Goals / Non-Goals

**Goals:**

- Remove repository path literals from `walker.rs`.
- Keep safe defaults for loose files and downstream repositories.
- Allow repository-specific whole-tree exclusions without Rust changes.
- Make every exclusion typed, spec-linked, deterministic, and explainable.
- Preserve one walk, explicit targets, hidden source, subdirectory path anchoring, source parity, and #603 prepared-state behavior.

**Non-Goals:**

- Honoring `.gitignore`, `.ignore`, global Git configuration, or parent ignore files.
- Adding arbitrary CLI `--exclude` patterns that bypass review.
- Replacing rule-specific exceptions or inline `allow`/`keep`.
- Parallelizing traversal or source parsing.
- Allowing policy to hide editable source merely because it currently violates lint.

## Decisions

### 1. Compose a shipped baseline with one nearest repository policy

`crates/chelis-lint/default_policy.toml` will be embedded into the crate and contain the cross-repository infrastructure/build/dependency exclusions currently hard-coded in `walker.rs`. The nearest ancestor `chelis-lint.toml` will add repository-specific entries, initially the generated opaque-invariant program corpus. This preserves downstream behavior when no repository policy exists while moving policy values out of traversal code.

Repository discovery starts at a file target's parent or a directory target itself and chooses the nearest ancestor only. Patterns are anchored to the directory containing that policy. Multiple implicit policy layers are rejected in favor of one obvious repository contract.

Alternative rejected: require a config for every directory lint. That would break loose projects and every downstream shell before coordinated migration. Alternative rejected: raw `.gitignore`; it conflates commit and lint scope.

### 2. Use `ignore` as an engine with all ambient filters disabled

The walker will use `ignore::WalkBuilder`, but call `standard_filters(false)` and keep hidden paths visible. Policy entries will be compiled with `ignore::gitignore::GitignoreBuilder` so patterns have documented gitignore-style path semantics without reading any Git ignore file. A custom entry predicate will prune matching directories before descent.

This retains `ignore`'s tested traversal/matching implementation while making the policy inputs exclusively Chelis-owned. The direct `walkdir` dependency and error type can be removed if no public caller still needs them; `ignore` itself continues to use `walkdir` internally.

### 3. Use a strict structured schema

The schema is:

```toml
version = 1
spec = "spec/01-nomenclature.md" # required for repository policy

[[exclude]]
pattern = "tests/corpus/opaque_invariants/programs/"
class = "generated"
cross_ref = "§12.2"
```

Serde structs use `deny_unknown_fields`. `class` is a closed enum: `infrastructure`, `build`, `dependency`, `generated`, `immutable`. Every pattern is compiled independently so `TraversalPolicy::exclusion_for(path)` can return the exact policy record for diagnostics and tests. Repository cross-references are verified against the declared spec file before traversal. The shipped baseline is tripwire-tested against the canonical Chelis spec at build/test time.

Alternative rejected: a plain `.chelis-lintignore`. It would externalize patterns but lose mandatory classification and cross-reference metadata.

### 4. Preserve explicit-root and canonical-entry invariants

Depth zero overrides exclusion matching only. Therefore a directly named excluded regular file or directory remains auditable, while entry-kind and canonical policy-boundary validation still apply before traversal or source loading. Nested matches are pruned. Paths are normalized relative to the policy root, not the per-target walk root, mirroring the existing workspace-root fix for diagnostic exceptions.

`walker::walk` remains the only recursive discovery entry point. Rules receive its resulting entry vector for corpus-wide state. A bounded ancillary lookup needed to preserve direct or explicit-file behavior must pass `TraversalPolicy` admission for the file and its parents; excluded content may not influence an admitted verdict indirectly. The opaque direct-check compatibility path may invoke the same canonical walker, never construct an independent traversal.

### 5. Separate policy errors from filesystem walk errors

Introduce `TraversalPolicyError` for discovery, I/O, TOML, pattern, spec, version, and cross-reference failures. `LintError` will expose policy and `ignore::Error` variants with paths and actionable messages. No malformed policy fallback is allowed.

### 6. Lock policy with behavioral and tripwire tests

Tests will cover every migrated baseline entry, repository extension, no-config defaults, all disabled ambient ignore sources, hidden workflows, malformed schema, invalid patterns, unresolved references, subdirectory and absolute targets, explicit-root override, sibling parity, and interaction with the prepared opaque catalog. A source tripwire will reject reintroducing skip-name literals in `walker.rs`.

The authoritative acceptance oracle is:

```text
CARGO_TARGET_DIR=target/agents/issue-740 cargo build -p chelis-cli --bin chelis
CARGO_BIN_EXE_chelis="$PWD/target/agents/issue-740/debug/chelis" CARGO_TARGET_DIR=target/agents/issue-740 cargo nextest run -p chelis-lint
CARGO_BIN_EXE_chelis="$PWD/target/agents/issue-740/debug/chelis" CARGO_TARGET_DIR=target/agents/issue-740 cargo nextest run -p chelis-cli --test lint_traversal_policy
```

## Risks / Trade-offs

- **[Risk] `ignore` defaults silently hide source.** → Disable all standard filters and prove hidden/Git-ignored paths remain admitted.
- **[Risk] Repository config can hide editable code.** → Require closed classification plus resolvable spec reference; document that whole-tree exclusion is only for generated, vendored, infrastructure, or immutable content.
- **[Risk] Pattern anchoring drifts under subdirectory lint.** → Store the policy root separately and test relative, absolute, and subdirectory targets.
- **[Risk] Embedded defaults and active spec drift.** → Parse the shipped TOML in normal code and tripwire every entry's cross-reference against `spec/01-nomenclature.md`.
- **[Trade-off] Independent matchers add small per-entry overhead.** → The policy has few entries and pruning avoids much larger traversal and parsing costs; optimize only with measurement.
- **[Trade-off] This branch is stacked on #603.** → Keep its PR based on the #603 branch until that change merges, then rebase onto `main`.

## Migration Plan

1. Write failing schema, matching, ambient-ignore, explicit-root, and prepared-catalog tests.
2. Add dependencies and `TraversalPolicy` loading/matching.
3. Switch the canonical walker to configured `ignore::WalkBuilder` and migrate current values into policy TOML.
4. Update §12, CHANGELOG, and downstream contract material if repository policy becomes required there.
5. Run the authoritative oracle, local gate, and a fresh red-team pass.

Rollback restores the previous walker and hard-coded predicate; policy files are additive and contain no persisted runtime data.

## Open Questions

None for this phase. A future CLI may expose `chelis lint --explain-path`, but this change only requires an explainable library policy API and deterministic diagnostics on policy failure.
