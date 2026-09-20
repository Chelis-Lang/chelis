`python3 scripts/gate.py --fast` now classifies the changed path set through
the change-owned planner's own rule lookup and refuses a push that adds a file
no `[[path_rule]]` in `.config/ci-test-targets.toml` routes. The planner
itself cannot run locally, because in `pull_request` mode it requires a
two-parent synthetic merge, so until now a new tracked file passed every local
check and then failed `Plan Changed Integration Tests` in CI. The check prints
the same `unclassified changed path: <path>` sentence CI prints, from one
shared spelling, and reports every unrouted path rather than stopping at the
first as CI does.

It derives its own changed set when it runs, so a file a regeneration created
earlier in the same `--fast` is classified in that run. It matches the
planner's rename handling, classifying both sides of a move rather than only
the destination, and it ignores untracked work, which CI never sees and
nothing can route. It is not a proof that CI will agree: it reads the working
tree, while CI composes base and candidate Cargo metadata inside the synthetic
merge, so a change to the package set itself can be ambiguous to the planner
and clean locally.

Four paths that no rule routed are now routed, three of them generated
artifacts whose only consumer runs nightly. `ci_change_owned.py
classify-paths` is available on its own, with `--from-git` for the derived
set.
