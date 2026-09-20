The informational `PR Package Expansion` lane now gives each shard an
80-minute execution deadline inside a 90-minute job limit, replacing 16 inside
20. The deadline is a backstop against a hung command rather than a schedule,
and at the old size it was deciding coverage instead: across the 38 dispatches
measured under the current planner the longest shard projects to a median of
1580s and a maximum of 4410s, so 45 minutes would still cut six of them and 60
would still cut one. Three of four shards finish inside fifteen minutes either
way, so the median dispatch costs about 81 job-minutes instead of 64, with a
worst case of 360 against 80 if every shard ran to the new limit. The two
limits move together, because a shard that reaches the script's deadline still
writes and uploads a partial receipt while one the job limit kills writes
none; the ten minutes between them is consumed by job setup, which runs before
the executor's clock starts, rather than by finalization.
`SOFT_BUDGET_SECONDS` is now derived as ten minutes below the deadline rather
than set independently, and the warning window is bounded as a fraction of the
deadline so it cannot degenerate into a threshold every shard exceeds or none
reaches.
