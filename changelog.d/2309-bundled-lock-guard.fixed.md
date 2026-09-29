`python3 scripts/gate.py --fast` now reaches the `chelis-reef` bundled-lock
guard. That guard holds both halves of one invariant — every committed bundled
`reef.lock` must pin the embedded std bundle's hashes, and the compiled
`chelis-std-bundle` rlib must embed the bytes that are on disk — and no local
gate invocation ran either one. They are lib unit tests in a crate the
changed-crate stage never selects, because the paths that invalidate them belong
to other crates (`crates/chelis-cli/tests/fixtures/**/reef.lock`) or to no crate
at all (`examples/**/reef.lock`, and the root `Cargo.toml` whose workspace
version the verdict keys on). A change that moved the bundle hash therefore
passed `--fast` and failed in CI. The new leg runs when a `packages/chelis-std/`
or `crates/chelis-std-bundle/` path, any `reef.lock`, or the root `Cargo.toml`
changed, and stays off every other change.

The guard also reports every stale lock in one run rather than panicking on the
first. The discovery walk is sorted, so a tree with three drifted locks
surrendered one per CI round. See
[#2309](https://github.com/Chelis-Lang/chelis/issues/2309).
