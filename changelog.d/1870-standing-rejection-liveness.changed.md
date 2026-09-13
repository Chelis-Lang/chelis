Run the full source-derived standing rejection-issue manifest against GitHub
daily and on manual dispatch from `main`. Any non-success result, including a
cancelled or skipped canary job, opens or updates one `nightly-failure`
tracking issue. A constant concurrency group prevents report scripts from
overlapping without cancelling the running workflow; bursts may coalesce to
the newest pending dispatch rather than run FIFO. Failure consolidates
exact-title duplicates and the next successful run closes every match. This is
the standing liveness canary only, not the broader loud-unsupported Phase 4
nightly matrix.
