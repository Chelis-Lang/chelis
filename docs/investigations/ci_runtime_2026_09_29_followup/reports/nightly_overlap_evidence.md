# Runtime extent nightly overlap, 2026-09-29

Candidate source: `2ed139844a82319fcd491cd9f1b2f5a2b80ad576`, nightly run `36517834756`.
The target manifest and heavy-e2e workflow have no changes between that commit
and current `origin/main` at `465a3623be19bd68ada6c39aa7552b63e0ccf8c3`.

- `scripts/runtime_extent_oracle_targets.json` registers 62 target rows: 61
  execute tests and one lists ignored HIP inventory. The final oracle creates
  63 distinct commands when its Python self-test command is included.
- The 61 executing rows name 662 unique `(JUnit classname, testcase name)`
  identities. All 662 appear in the downloaded full-workspace or dtype JUnit
  artifacts from this same run: 644 in full-workspace, 38 in dtype, and 20
  in both. This checks name presence, not identical feature closures, target
  configuration, environment, or the runtime oracle's per-target success
  receipts.
- The full-workspace JUnit observed 1,304 seconds of testcase time for those
  identities; dtype observed 1,242 seconds, including 1,132 seconds for
  `wire_schema_numeric_fields_match_the_reviewed_baseline`. These are sums of
  per-test durations, not job wall times. The runtime extent oracle job ran
  from 03:40:39 to 04:14:21 UTC (about 33m42s); its main oracle step ran
  from 03:41:32 to 04:14:18. Its per-target timing is not retained.
- Phase A row `wire.capacity_census` points to
  `wire_capacity.wire_schema_numeric_fields_match_the_reviewed_baseline`.
  The runtime extent oracle currently executes the entire 18-test wire
  census target; the dtype worker also executes all 18 tests.

This is a candidate for an authenticated same-head execution receipt shared
across the nightly jobs. Before reusing any result, prove equivalent build
feature/target/environment contracts, exact selected and executed tests,
success, source/head identity, and corpus ownership. Keep the standalone
`runtime_extent_oracle.py --phase final` command exhaustive; a nightly
aggregate can validate receipts after independently scheduled workers finish.
