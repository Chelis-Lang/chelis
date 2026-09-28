# PR #2683 expansion shard timing audit

Run [36342641704](https://github.com/Chelis-Lang/chelis/actions/runs/36342641704) validated exact PR head `5810744986c43aee6353673c6f8e21f2986cedb0`. It selected 360 `chelis-cli` integration targets; all executed and passed. The executed assignment uses `duration-lpt-v1` introduced in #2678, not the compatibility `sha256-modulo-v1` mapping retained in `plan.json`.

| Shard | LPT balancing weight | Actual elapsed | Workspace build | Sum of group list commands | Sum of group run commands | JUnit case-time sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,088.032s | 1,089.494s | 194.004s | 89.7s | 805.8s | 1,513.4s |
| 1 | 1,088.034s | 1,481.060s | 194.742s | 85.6s | 1,200.7s | 2,243.1s |
| 2 | 1,088.032s | 616.078s | 119.213s | 60.7s | 436.1s | 849.0s |
| 3 | 1,088.035s | 1,048.769s | 184.549s | 84.6s | 779.6s | 1,524.8s |

The long shard's excess is almost entirely in test execution, not in its workspace build or test listing. Its four longest grouped `nextest` runs took 355.3, 277.0, 224.7 and 219.5 seconds; shard 2's longest took 122.7 seconds. JUnit names on shard 1 include `every_lane_returns_the_taken_arm_or_its_claims_trap_at_a_claimed_join` (187.9s), `independent_grad_entry_failures_are_ordered_on_eval_and_c` (52.9s), and the #1293 keyed-random grad cases (46–47s). Shard 3's `std_initialisers_validate_before_the_draw_is_consumed_under_grad_in_eval_and_c` took 194.4s. The LPT weights are derived from serial per-target measurements while execution groups up to 16 targets in each command; they are useful ordering inputs, not wall-time predictions. The four runners also differ in build time, so one run cannot isolate stale weights from hardware/cache variation or interactions among tests.

Conclusion: #2678 now balances by test identity and measured duration, but it did not make this sampled run balanced. Updating the weight source with recent command/test observations is a lower-risk next experiment than random package hashing. Work stealing would require a coordinated per-target claim/receipt protocol across workers and could compromise deterministic exact-head coverage if built casually. The telemetry merged in #2682 now preserves these command and JUnit timings for further samples. This audit alone does not justify another scheduling PR.
