Run the full source-derived standing rejection-issue manifest against GitHub
daily and on manual dispatch from `main`. Any non-success result, including a
cancelled or skipped canary job, opens or updates one `nightly-failure`
tracking issue; the next successful run closes it. This is the standing
liveness canary only, not the broader loud-unsupported Phase 4 nightly matrix.
