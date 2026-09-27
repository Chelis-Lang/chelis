# PR #2684 red-team round 1

Reviewed exact pushed head `7538f66a6d27a2b2674d6397a0bf15533bc17b44` in `/Users/robertronan/chelis-worktrees/source-arch-perf-20260927`.

**Satisfied; round closed.** No P0, P1, P2, or P3 findings. No out-of-scope defects identified.

The change replaces cloning a frontier that is immediately overwritten with an explicit frontier argument. All other collector fields keep the same clone/reference semantics. No visitor executes between the former fork and frontier replacement. Closure construction still uses the same parent binding depth plus one for both parameter binding types and the body visitor. Allocation order changes, but visitor order, lexical metadata, path contents, and multiplicity do not. The path data have derived clones and no custom clone/drop side effects.

Executed in the supplied worktree with its warm target after the parent released the CPU slot:

- `python3 scripts/reap_orphans.py`: no repository build/test processes.
- `cargo nextest run -p chelis-compiler-api --lib -E 'test(source_arch::)'`: **80 passed**, 470 unrelated tests skipped; nextest run `a5afc0aa-827e-47d2-92cb-6b9211849c28`. Build took 12.31 seconds; tests took 73.582 seconds.
- Production `guarded_upper_consumers_do_not_recreate_the_semantic_pipeline`: **PASS, 73.535 seconds**.

Positive controls covered single-stage helpers, external aliases, uninvoked closures, mutually exclusive branches, correlated branch-result aliases, lexical shadowing, invariant conditions, and zero/one iteration cases. Negative controls exercised planted duplicate pipelines, closure and higher-order callback composition, match/if callable results, known callable iterables, repeated while/for execution, call multiplicity, local macros and imports, and production source-discovery bypass cases. Existing adversarial fixtures suffice for this mechanical change; no tracked-source probes were inserted.

The production pass validates current scanner findings on this head. The parent-supplied historical timing baselines were not remeasured, as directed; this independent candidate run is consistent with the reported candidate timing. No general speedup guarantee or exhaustive Rust-source analysis claim is established. Hosted CI, package expansion, and merge readiness remain the author's responsibility. No CLI, language, or normative specification contract changes are introduced by this private scanner refactoring; the production guard and existing acceptance/rejection fixtures exercise its product.

Final verification: `git status --porcelain=v1` empty. `python3 scripts/reap_orphans.py` reported no build/test processes. `.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/source-arch-perf-20260927/target` at 2026-09-27T19:03:38Z reported **FREE**, exact reviewed head, 0 modified/staged/untracked/unmerged files, no scoped processes. No probes or mutations require restoration. Reviewer remains available for repair verification.
