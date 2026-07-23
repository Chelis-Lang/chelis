## 1. Write Policy Tests First

- [x] 1.1 Add schema tests for valid versioned policy plus failures for unsupported versions, unknown fields/classes, missing fields/specs, invalid patterns, and unresolved cross-references.
- [x] 1.2 Add discovery and matching tests for shipped defaults, nearest repository policy, workspace anchoring, subdirectory/absolute targets, no-config fallback, sibling parity, directory pruning, and exact exclusion metadata.
- [x] 1.3 Add negative-parity tests proving `.gitignore`, `.ignore`, parent/global Git ignores, `.git/info/exclude`, and hidden paths cannot alter admitted entries.
- [x] 1.4 Add explicit-root and integration tests proving directly named excluded files/directories are linted, nested exclusions still prune, and opaque prepared catalogs consume exactly the configured canonical entries.

## 2. Implement Structured Traversal Policy

- [x] 2.1 Add `ignore`, `serde`, and `toml` dependencies and a strict `TraversalPolicy` module with typed schema, nearest-ancestor discovery, shipped-default composition, pattern compilation, exact match explanation, and actionable errors.
- [x] 2.2 Create the shipped baseline policy TOML and root `chelis-lint.toml`, migrating every current hard-coded exclusion into the appropriate artifact with class and `§12.2` cross-reference.
- [x] 2.3 Add cross-reference verification against the repository policy's declared spec and a tripwire validating every shipped baseline entry against the canonical spec.

## 3. Replace Walker Hard-Coding

- [x] 3.1 Replace direct `WalkDir` and `is_skip_dir` use with a fully configured `ignore::WalkBuilder` whose ambient standard filters are disabled and whose sole pruning source is `TraversalPolicy`.
- [x] 3.2 Preserve depth-zero admission, directory/file classification, symlink behavior, stable violation ordering, walk error propagation, and one canonical entry vector for rule preparation and checks.
- [x] 3.3 Update `LintError` and direct opaque-rule compatibility paths for policy/ignore errors without silent fallback.
- [x] 3.4 Add a source tripwire preventing path-name exclusion literals or an independent filesystem walker from returning to rule or walker code.

## 4. Documentation and Acceptance

- [x] 4.1 Add `spec/01-nomenclature.md` §12.2 defining exclusion classes, schema, explicit-target semantics, deterministic non-Git inputs, and the traversal-exclusion versus diagnostic-exception boundary.
- [x] 4.2 Add a `CHANGELOG.md` entry linking chelis#740 and update active CLI documentation with policy discovery and failure behavior.
- [x] 4.3 Inspect orphaned processes, build the current CLI, and run the authoritative `chelis-lint` oracle in `target/agents/issue-740`.
- [x] 4.4 Run `cargo fmt --all -- --check`, focused clippy with `-D warnings`, strict OpenSpec validation, and `.venv/bin/python scripts/gate.py --local`.
- [x] 4.5 Run a fresh local red-team agent with `target/agents/redteam-740`; execute adversarial policy, CLI, hidden-source, explicit-root, and prepared-catalog probes and record residual limitations.

### Adversarial Validation Record

The original fresh local red-team run used `target/agents/redteam-740` and exercised malformed and escaping policy paths, hidden and Git-ignored sources, explicit excluded roots, standalone CLI failure behavior, and the opaque prepared catalog. A later independent follow-up found two missed architecture gaps: `doc-filename-convention` could read an excluded Cargo manifest through its bounded package-name scan, and the crate guardrail existed only as a regular `CLAUDE.md` instead of the repository-standard canonical `AGENTS.md` plus symlink.

The follow-up adds an executable admitted-versus-excluded manifest pair, parent-aware policy admission for ancillary manifests, canonical prepared package-name state, and a symlink tripwire for the crate guardrail. Residual limitation: elapsed-time behavior is not benchmarked against a massive generated tree; the traversal and ancillary-input boundaries are locked structurally instead. Future rules that consult ancillary files must consume canonical entries or add the same parent-aware admission proof.
