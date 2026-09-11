The runtime-extent class oracle (`scripts/runtime_extent_oracle.py`) runs again. Its
per-target expected test receipts moved out of hand-maintained Python tuples into the
reviewed manifest `scripts/runtime_extent_oracle_targets.json`, which two independent
readers enforce: the oracle builds its cargo commands from the same rows it checks, and
a new `--fast` tripwire parses each named source and fails the pre-push gate when a
rename, an unregistered new test, or a new `#[ignore]` moves a row's real inventory.
Renaming a selected test previously broke the oracle silently for everyone who did not
run it. See [#1742](https://github.com/Chelis-Lang/chelis/issues/1742).
