The informational `PR Package Expansion` lane now gives each shard an
85-minute execution deadline inside a 90-minute job limit, replacing 16 inside
20. The deadline is a backstop against a hung command rather than a schedule,
and at the old size it was deciding coverage instead: across the 38 dispatches
measured under the current planner the longest shard projects to a median of
1580s and a maximum of 4410s, so 45 minutes would still cut six of them and 60
would still cut one. Three of four shards finish inside fifteen minutes either
way, so the median dispatch costs about 81 job-minutes instead of 64. The two
limits move together, because a shard that reaches the script's deadline still
writes and uploads a partial receipt while one the job limit kills writes
none. `SOFT_BUDGET_SECONDS` is now derived as ten minutes below the deadline
rather than set independently, so the early warning it records cannot drift
into a threshold every shard exceeds.
