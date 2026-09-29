`python3 scripts/gate.py --fast` now reaches the `chelis-reef` bundled-lock
guard. That guard holds both halves of one invariant — every committed bundled
`reef.lock` must pin the embedded std bundle's hashes, and the compiled
`chelis-std-bundle` rlib must embed the bytes that are on disk. They are lib
unit tests in a crate that no path invalidating them selects: those paths belong
to other crates (`crates/chelis-cli/tests/fixtures/**/reef.lock`) or to no crate
at all (`examples/**/reef.lock`, and the root `Cargo.toml` whose workspace
version the verdict keys on), so the changed-crate stage never reached them.
(`--validation` does run them when `chelis-reef` itself changed — which is a
fourth invalidating class, and now triggers the leg too.) A change that moved
the bundle hash therefore passed `--fast` and failed in CI. The new leg runs when a `packages/chelis-std/`
or `crates/chelis-std-bundle/` path, any `reef.lock`, or the root `Cargo.toml`
changed, or when the guard's own crate changed, and stays off every other
change.

The guard also reports every stale lock in one run rather than panicking on the
first. The discovery walk is sorted, so a tree with three drifted locks
surrendered one per CI round. See
[#2309](https://github.com/Chelis-Lang/chelis/issues/2309).
