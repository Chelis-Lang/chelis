# Tasks: harden-lint-traversal-edges

## 1. Spec-first failing tests

- [x] 1.1 Flip `explicit_source_shaped_special_entry_is_omitted` in `crates/chelis-lint/tests/traversal_policy.rs` to assert `chelis_lint::lint` returns a `LintError::Policy` naming the root and a non-regular-entry reason (test fails red against current code)
- [x] 1.2 Flip the escaping-root half of `explicit_symlink_directory_root_must_resolve_inside_policy_boundary` to assert a loud error naming the root and boundary-escape reason; keep the internal-link half asserting successful lint of `inside.ch`
- [x] 1.3 Add negative-parity test: a symlink root resolving to a mismatched kind (file link named as directory target) fails loudly with a kind-mismatch reason
- [x] 1.4 Add positive-parity test: discovered (non-explicit) special files and escaping symlinks below an admitted root are still silently omitted and the invocation succeeds
- [x] 1.5 Add child-process probe test: cwd below the policy root plus target `.` discovers the repository `chelis-lint.toml` (repository exclusion applies); assert relative and absolute spellings admit identical entry sets
- [x] 1.6 Add standalone CLI tests in `crates/chelis-cli/tests/lint_traversal_policy.rs`: `chelis lint --check <inadmissible-explicit-root>` exits nonzero and prints the root path and reason; `chelis lint --check .` from a subdirectory cwd applies repository exclusions

## 2. Loud explicit-root rejection (D1)

- [x] 2.1 Add `TraversalPolicyError::InadmissibleExplicitRoot { path, reason }` (reason distinguishes non-regular kind, kind-mismatched link target, boundary-escaping link, unresolvable link) with `Display` coverage
- [x] 2.2 Rework `is_admitted_explicit_entry` (or add a `Result`-returning sibling) so the rejection reason is produced where the checks run; update `walker::walk` to return the error instead of `Ok(Vec::new())`
- [x] 2.3 Confirm the style gate path: a rejected explicit root surfaces through `run_lint_for_single_file` as the existing loud `lint-traversal-policy` violation; add a style-gate unit test for a socket-shaped input file
- [x] 2.4 Run tasks 1.1–1.4 and 1.6 (rejection half) green; verify the nonexistent-root test still passes unchanged

## 3. cwd-insensitive policy discovery (D2)

- [x] 3.1 In `TraversalPolicy::load_for`, resolve the ancestor-search start against `std::env::current_dir()` for relative targets before `find_repository_policy`; leave matcher roots and entry-path presentation lexical
- [x] 3.2 Verify `path_relative_to_scope` handles the absolute-policy-root/relative-entry mix introduced by 3.1; extend it only if a failing test demands it
- [x] 3.3 Run tasks 1.5 and 1.6 (discovery half) green; confirm `relative_subdirectory_target_uses_workspace_anchoring` and the existing relative-probe suite stay green

## 4. Shared policy per invocation (D3)

- [x] 4.1 Add `walk_with_policy(root, &TraversalPolicy)`; keep `walk(root)` as load-then-delegate
- [x] 4.2 Extend `Rule::prepare_run` with a `&TraversalPolicy` parameter; update all in-tree implementors; `doc-filename-convention` consumes the shared policy instead of calling `load_for` in `prepare_doc_filename_state`
- [x] 4.3 Thread one policy load through `lint()`; assert via test instrumentation or a filesystem-permission trick that the policy TOML is read at most once per `lint()` invocation
- [x] 4.4 Confirm `repeated_lint_invocations_rebuild_opaque_catalog_state` still passes (no cross-invocation caching)

## 5. Depth-zero recheck scoping (D4)

- [x] 5.1 Scope the post-yield `is_admitted` recheck in `walker::walk` to `entry.depth() == 0`
- [x] 5.2 Run the full symlink/special-entry corpus in `traversal_policy.rs` as the regression oracle for discovered-entry admission

## 6. Docs and spec sync (D5)

- [x] 6.1 Update `spec/01-nomenclature.md` §12.2: explicit-root rejection is loud with path and reason; nearest-ancestor discovery is cwd-resolved and spelling-independent
- [x] 6.2 Update `docs/book/src/cli.md` lint section to document the loud rejection and cwd-insensitive discovery
- [x] 6.3 Add `CHANGELOG.md` entries (loud rejection under Fixed/BREAKING note; discovery fix under Fixed)
- [x] 6.4 Verify the delta spec scenarios in this change each map to at least one test added in group 1

## 7. Gates and evidence

- [x] 7.1 `cargo nextest run -p chelis-lint` and the standalone CLI traversal suite green in an isolated `CARGO_TARGET_DIR`
- [x] 7.2 `python3 scripts/gate.py --local` green (clippy `-D warnings`, fmt, `chelis lint --check .`, changed-crate nextest)
- [x] 7.3 Record one quiet-machine `chelis lint --check .` wall-clock run (clean worktree) in the change's validation notes as the #603 linearity evidence
- [ ] 7.4 Red team: fresh local subagent executes the flipped-assertion tests, the CLI rejection path, and the subdirectory-cwd probe against the spec deltas

### Validation record (2026-07-23)

- Gate: `gate.py --local` exit 0 — workspace clippy `-D warnings`, `cargo fmt --check`, `chelis lint --check .`, per-crate nextest: chelis-cli 1587/1587, chelis-lint 346/346 (isolated `CARGO_TARGET_DIR=target/agents/pr839`, rebased onto main 33dcf882 / v0.17.1).
- 7.3 timing: `/usr/bin/time ./target/agents/pr839/debug/chelis lint --check .` on this checkout (23.5 GiB `target/` policy-excluded; one concurrent cargo job on another repo): exit 0, 520.9s wall / 358.8s user / 12.6s sys, ~27 MB max RSS, debug binary. Traversal cost is negligible (sys 12.6s); wall is per-rule CPU linear in the admitted corpus — the #603 O(files × repo-walk) pathology is gone (each admitted Surf candidate parses once per invocation, locked by `catalog_parses_each_admitted_surf_candidate_once_per_invocation`).
- Environmental note: without `cargo` on PATH the same invocation exits 1 on 7 exception-covered paths — `detect_lint_workspace_root` shells out to `cargo locate-project`, and without it workspace-rooted exceptions cannot anchor. Pre-existing behavior, unchanged by this change.
