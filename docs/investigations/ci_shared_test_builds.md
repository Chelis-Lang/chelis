# Shared Linux test builds

The September 8 CI sample showed each workspace shard spending about five
minutes compiling tests, with the same compilation repeated by four
generalization workers under their own feature configuration. Workspace shard
2 then spent another 5:49 on support oracles. This change addresses the shared
test-build and support-command portions of [#1502](https://github.com/Chelis-Lang/chelis/issues/1502)
and [#1503](https://github.com/Chelis-Lang/chelis/issues/1503).

## Build and execution ownership

| Configuration | Producer | Consumers |
| --- | --- | --- |
| Default workspace features | `workspace-test-build` | Two workspace hash shards and their support nextest commands |
| `chelis-types/generalize-sweep-oracle` | `generalization-test-build` | Four generalization hash shards |

Each producer compiles a complete test archive with nextest 0.9.136 and executes
its existing test-selection census on that warm configuration. The normal
workspace producer is the sole writer of the Linux workspace Cargo cache.
Consumers keep their existing test profiles, explicit filters, hash partitions,
JUnit artifacts, and required aggregates. Their `needs` dependencies require
the producer to finish before downloading an artifact from that workflow run.
Missing artifacts, wrong build identity, corruption, and failed test selection
remain failures. No artifact lookup reaches into a different run or branch.

The manifest binds the archive's SHA-256 to the checked-out commit,
configuration, platform/architecture, workspace path, and nextest version.
Producer and consumer checkouts use the same GitHub revision and path. The
path restriction is deliberate: existing tests use compile-time absolute
source paths. This is a same-path hosted handoff, not a relocatable release
artifact. Nextest extracts into the checkout's `target` tree so compiled-in
CLI paths remain valid. The producer and consumers install the same Python
and C dependencies as the existing workers.

Consumers run `cargo fetch --locked --target x86_64-unknown-linux-gnu` after
restoring the dependency cache. Some test binaries execute offline Cargo compile
controls, which still require registry metadata and sources. A restored cache
may predate a dependency; executing an archive does not implicitly fetch it.
The explicit fetch supplies that prerequisite without compiling the workspace.

The runtime static library is an explicit extra archive input. A Cargo
`test --no-run` build supplies the exact current `chelis_runtime` artifact
filenames through JSON messages; nextest reuses that build and includes those
files. A cached filename or glob is never the authority for this handoff.
Missing current artifact messages or files fail archive creation.

Artifacts are compressed by nextest, uploaded without a second compression
pass, and retained for one day. A rerun may overwrite the same configuration's
artifact within that run; its identity is still checked. The archive contains
the full build, so execution profiles continue to own exclusions and ignored
tests. See the [nextest archive contract](https://nexte.st/docs/ci-features/archiving/).

Workspace shard 1 owns the lowering-trace and front-end performance support
slice; shard 2 owns the unrepresentable-domain slice. The complete unsliced
`gate.py integration --support-only` command still executes all three in their
original order. Focused support nextest commands reuse the default workspace
archive with equivalent package, binary, and test filters. The front-end
performance ignored integration suite runs with `--run-ignored only` and one
test thread. Feature-specific lowering-trace builds, rustdoc commands, the
annotation-scaling Cargo test, and runtime/release-profile oracles retain
their separate builds. Those remaining configurations keep #1502/#1503 open.

## Validation and performance evidence

```console
.venv/bin/python -m unittest scripts.test_ci_test_archive scripts.test_gate scripts.test_compiler_front_end_performance scripts.test_hosted_validation
.venv/bin/python -m unittest scripts.ci_archive_execution
```

The executable probe creates a tiny Rust fixture with a runtime static library
and a CLI, archives both feature configurations, removes the entire producer
target, compares original and reused test lists, and runs passing,
failing, absent, and explicitly ignored selections. It also rejects a different
configuration and corrupted archive, and requires both the CLI and static
library to work after extraction. The workspace producer runs this probe
before building Chelis. A cold-registry probe rejects offline compilation before
the locked fetch, accepts it afterward, and still rejects an actual type error.
Workflow tests reject missing or late dependency fetches, missing producer dependencies
and cross-run downloads; existing gates still check exact partition coverage,
support ownership, cache writers, docs-only gating, and aggregate failure.

Hosted CI on the candidate is the full execution oracle. Its logs must show
one archive build per configuration, all two/four partitions executing, both
support slices passing, and green required aggregates. The expected saving is
four repeated workspace compilations per run, plus package-level recompilation
in the support nextest commands. Archive transfer/extraction and producer queue
time are new costs; wall-clock improvement must be measured from hosted runs,
with queue time separated from execution time. No measured speedup is claimed
before those runs complete.
