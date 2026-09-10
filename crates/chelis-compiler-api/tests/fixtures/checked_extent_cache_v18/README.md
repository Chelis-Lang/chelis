# Previous checked-extent cache producer

`context-v18.ctx` is an actual cache from the unmodified compiler before
checked-extent transport, not a current payload with its version changed.
`producer.json` records its source, dependency, observed wrong result, binary
hash and cache hash. The old producer successfully read its own cache before
these bytes were passed to the new consumer.

The producer was built from tree `05c0b0c390dcf1f4e47c614376937d40f631c687`,
the conflict-free integration of PR #1688 head
`00d176c589e7d308c6cfa0b4b1afed2b3c6cf386` and main
`ad9b6c248f6752febed9b314ff762303dff1022e` (#1664).
The synthesized local commit was `77988d22dd0548797c394ce584b07f6314ecc5e3`;
merged main `4f3812e2f6cbdf6cc940195a891912f975d54b35` has the same tree.

The generating command is `scripts/runtime_extent_cache_compatibility.py`
with `--previous` and `--current` pointing to separately built binaries,
`--previous-head` and `--current-head` naming their revisions, `--version`
matching the compiler package version, `--report` selecting an output JSON,
and `--fixture-dir` selecting this directory. It also executes matching and
mismatching reshape, singleton-broadcast and literal-insert controls against
old/current and current/current disk caches. The current consumer rejects
this v18 file before parsing its positional payload or reading its old paths.
