# Previous named-claim cache producer

`chelis-ctx-v19.ctx` is an actual compiled-context cache written by the
compiler at this pull request's base, `e813415d0`, before `ExtentWitness`
carried named claims and before WireDag reached v11. It is not a current
payload with its version edited. `producer.json` records the producer's
revision and binary hash, the source and dependency it compiled, the cache
hash, the magic the producer wrote, and the result that producer got.

The sibling `checked_extent_cache_v18/` holds a producer from several versions
earlier. It proves only that the magic line rejects a stale file. This fixture
is the IMMEDIATELY preceding producer, so it is the one that proves the
rejection this pull request's version bumps depend on.

The generating command is `scripts/runtime_extent_cache_compatibility.py` with
`--previous` and `--current` pointing at separately built binaries,
`--previous-head` and `--current-head` naming their revisions, `--version`
matching the compiler package version, `--report` selecting an output JSON, and
`--fixture-dir` selecting this directory. It also executes matching and
mismatching reshape, singleton-broadcast and literal-insert controls against
old/old, old/current and current/current disk caches; all six rows passed at
`e813415d0` against `de35dda3e`.

The `.ctx` file is named for the magic its bytes carry, so a later bump cannot
leave a file claiming a format it does not hold.
