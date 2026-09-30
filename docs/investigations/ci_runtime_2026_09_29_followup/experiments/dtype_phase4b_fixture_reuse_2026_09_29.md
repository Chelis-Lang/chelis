# Phase 4B oracle fixture reuse

Worktree: `/Users/robertronan/chelis-worktrees/dtype-phase4b-fixture-reuse`
Base: `e16b2f795045e26a366cfbba570b73a0c6278ea3` (freshly fetched `origin/main`)
Local commit/head: `d11894935cd2fdd5e72beece28ac070c27328400` (`perf(ci): reuse Phase 4B contract test fixture`)

Changed path: `scripts/test_dtype_phase4b_oracle.py` only. I inspected its owner, `scripts/dtype_phase4b_oracle.py`, and did not change it. `ContractValidationTests` now prepares the 27 real contract files once per class, snapshots their bytes, and restores changed bytes after each case. The oracle still reads real files from the fixture. `FrozenContractChangeTests` still uses separate Git repositories per case.

The new control runs a negative mutation followed by a positive validation and limits fixture copying across the two cases to one 27-file preparation. Before changing setup, this command failed with `54 not less than or equal to 27`; after the setup change, it passed:

```sh
.venv/bin/python scripts/test_dtype_phase4b_oracle.py ContractFixtureReuseTests.test_cases_share_preparation_but_not_mutations
```

All commands ran from the task worktree above. Timed commands and results:

| Source state | Exact timed command | Result | Wall time |
|---|---|---:|---:|
| Base, before test/control edits | `/usr/bin/time -p .venv/bin/python scripts/test_dtype_phase4b_oracle.py > target/dtype-phase4b-before.log 2>&1` | 286 tests, OK | 65.94s |
| Candidate, with control and reuse | `/usr/bin/time -p .venv/bin/python scripts/test_dtype_phase4b_oracle.py > target/dtype-phase4b-after-first.log 2>&1` | 287 tests, OK | 52.02s |

Observed local improvement: 13.92s, about 21.1%. This is the closest honest before/after comparison, not a same-head comparison: the candidate adds one control. It is one run per state on a machine with other activity, so it is not a hosted CI prediction. The baseline and candidate logs remain in the task worktree's ignored `target/` directory.

Additional check: `git diff --check` passed before commit. No cargo, full gate, push, or PR was run. Remaining risk: the class fixture assumes the normal serial `unittest` execution within one process; its reset restores the frozen contract file bytes used by existing cases. Future cases that mutate other fixture state would need corresponding reset coverage.

Final task worktree status: clean, branch `perf/dtype-phase4b-fixture-reuse`, one local commit ahead of `origin/main`. No oracle or fixture test process remains running.
