# Report: chelis#1829 bisect

Worktree: `~/chelis-worktrees/1829-bisect`, detached (renamed before your "do not rename"
arrived, with nothing running).

## 1. First bad commit

`22b193cf7d3df991cc0fa073864cd6df5df7ce89` — **PR #1693**, "Preserve computed reshape claims
and broadcast unit checks through calls". Parent: `062c29c19` (#1736).

**Correction:** you said no commit in the window is a runtime-extent change. `22b193cf7` is
exactly that: fifteen `extents` commits in its squashed body, adding
`changelog.d/checked_extent_transport.fixed.md` and
`crates/chelis-ir/tests/runtime_extent_checked_transport.rs`. I did not weight the search;
the bisect landed there on measurement.

## 2. Confirming pair (back to back, same worktree)

| commit | `csv_io_end_to_end_is_exact_and_byte_stable` |
|---|---|
| `062c29c19` (parent) | `Summary [8.764s]`, 11.5 s wall, PASS |
| `22b193cf7` (culprit) | **>480 s**, timed out at a 480 s cap |

At least 55x. Reaped after that timeout: PIDs 60030, 60722; every passing step reaped nothing,
so no measurement ran on a polluted machine.

## 3. Calibration and threshold

`df46fae13` = `Summary [7.999s]`. `e813415d0` = **>600 s**, timed out at a 600 s cap (reaped
6843, 90880). Threshold **120 s**: 15x the good end, so load or build noise cannot flip a
verdict, and 5x below the bad bound. Five steps, no ambiguous middle: every step was either
~8.4 s or a full timeout.

## 4. Bare standalone program (`#1362` row: measured, not inferred)

A 13-line program parsing a 30-byte JSON document through `Std.Io.Json`, run directly, fresh
package directory and reef home per measurement, `reef.lock` present:

| | `chelis check` | `chelis eval --file` |
|---|---|---|
| `df46fae13` | 2.11 s | 1.80 s |
| `062c29c19` (parent) | 2.10 s | 1.80 s |
| `22b193cf7` (culprit) | 2.19 s | **>300 s** (timed out) |

**The front end is not the cost.** `check` is flat; evaluation carries all of it, >=167x. That
reconciles your two observations: the `chelis check` I saw the test spawn is a fast
preliminary phase, and #1829's interpreter `sample` is right. It reaches a bare user program,
so the Tier 2 reading holds on measurement.

## 5. #1823

Body updated with the dispatch outcome; all five citations repointed to #1829 (verified no
`1830` remains). Still a draft, head `a466d5f6c`, code untouched.

## 6. Unfinished

I did **not** measure `json_io_end_to_end` or the two `issue_1314_json_bigint` rows, so I
cannot say they share this commit; the bare JSON program regressing at `22b193cf7` is
evidence for the JSON path, not proof for those rows. Both bad-end figures are bounds, not
values: I never observed either run complete.
