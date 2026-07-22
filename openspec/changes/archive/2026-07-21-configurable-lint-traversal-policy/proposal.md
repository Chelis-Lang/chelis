## Why

`chelis-lint` currently embeds repository-specific directory names directly in `walker.rs`, so adding or retiring generated, vendored, or immutable trees requires a Rust code change and risks another rule-specific traversal fork. Part of chelis#740 is to make whole-tree lint scope explicit, reviewable, and mechanically justified without conflating lint policy with `.gitignore` or machine-local Git configuration.

## What Changes

- Add a versioned root `chelis-lint.toml` containing structured traversal exclusions with a pattern, closed classification, and mandatory spec cross-reference.
- Replace `walker.rs`'s hard-coded path names with a policy loaded from the nearest `chelis-lint.toml` ancestor and matched through BurntSushi's `ignore` crate.
- Preserve deterministic behavior by disabling `.gitignore`, `.ignore`, parent, global, hidden-file, and Git-exclude defaults.
- Preserve explicit-target behavior: a path named directly by the user is linted even when a repository exclusion matches it, while matching nested directories are pruned.
- Keep traversal exclusions distinct from rule-specific `Exception` and inline `allow`/`keep` mechanisms.
- Add an explainable policy API and positive/negative tests for config discovery, malformed policy, cross-reference validation, hidden source, subdirectory/absolute roots, explicit targets, and every migrated exclusion.

## Capabilities

### New Capabilities

- `lint-traversal-policy`: Defines deterministic, structured, workspace-anchored discovery exclusions for `chelis lint`.

### Modified Capabilities

None. `opaque-domain-catalog` already consumes the canonical walker's admitted entries; changing how that walker obtains its policy does not alter the catalog contract.

## Impact

- Affected crate: `chelis-lint`; affected command surface: `chelis lint` and built-in style-gate traversal.
- New repository contract file: `chelis-lint.toml`.
- New direct dependencies: `ignore`, `serde`, and `toml`; the direct `walkdir` dependency can be removed if no longer exposed by the walker API.
- Active documentation: `spec/01-nomenclature.md` will define traversal-exclusion semantics and permitted classifications.
- No `.gitignore`, global Git ignore, `.ignore`, or machine-local configuration will influence lint results.
