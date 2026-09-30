# Phase 4B shared-fixture audit

Task worktree: `/Users/robertronan/chelis-worktrees/dtype-phase4b-fixture-reuse`
Base: `e16b2f795045e26a366cfbba570b73a0c6278ea3`
Final head: `bb94278a366c59054eefe04f27964c51a25563c5`
Changed path: `scripts/test_dtype_phase4b_oracle.py` only. The owner `scripts/dtype_phase4b_oracle.py` remains unchanged.

## Audit

I inspected the entire `ContractValidationTests` class (now lines 30-4298): 228 test methods. The AST scan found 162 direct `write_text` call sites and 128 calls to the class's `replace` helper. It found no other test-body filesystem mutation calls (create, delete, rename, link, chmod, utime, subprocess, or open), and no assignment to `self` attributes in test methods. Fixture-root writes in those cases are therefore text replacements; their original paths are under `CONTRACT_FILES`. A reverse-order run executed all 228 cases successfully and recorded 25 distinct fixture paths written, with zero paths outside `CONTRACT_FILES`.

The earlier shared-fixture teardown had a real isolation leak even though the existing suite passed: a new file or directory, a moved contract file, or a file/directory mode or time edit could reach the next case. The new negative control created an extra directory, files and a symlink, moved one contract file, and changed file/directory modes and times. Before the repair, this exact command failed because `extra/` remained:

```sh
.venv/bin/python scripts/test_dtype_phase4b_oracle.py ContractFixtureReuseTests.test_cases_restore_extra_paths_and_metadata
```

Teardown now removes extra entries without following symlinks, recreates missing contract paths, and restores file bytes and file/directory mode and access/modification times. The control also covers an extra file inside an existing fixture directory and an unreadable extra directory. The earlier positive/negative reuse control remains.

## Commands and results

Commands ran in the task worktree. After repair:

```sh
.venv/bin/python scripts/test_dtype_phase4b_oracle.py ContractFixtureReuseTests
```

Result: 2 tests, OK. `git diff --check` passed.

The reverse-order audit used this command:

```sh
.venv/bin/python - <<'PY'
import sys, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, 'scripts')
import test_dtype_phase4b_oracle as t
original_write = Path.write_text
written = set()
def record_write(path, *args, **kwargs):
    root = getattr(t.ContractValidationTests, 'root', None)
    if root is not None:
        try:
            written.add(path.relative_to(root))
        except ValueError:
            pass
    return original_write(path, *args, **kwargs)
names = unittest.defaultTestLoader.getTestCaseNames(t.ContractValidationTests)
suite = unittest.TestSuite(t.ContractValidationTests(name) for name in reversed(names))
with mock.patch.object(Path, 'write_text', record_write):
    result = unittest.TextTestRunner(verbosity=0).run(suite)
extra = sorted(written - set(t.CONTRACT_FILES))
print(f'reversed_cases={len(names)} written_fixture_paths={len(written)} unexpected_paths={extra}')
sys.exit(not result.wasSuccessful() or bool(extra))
PY
```

Result: 228 tests, OK, `written_fixture_paths=25`, `unexpected_paths=[]`.
This run used the repaired precommit source; the final edit only reordered the
symlink check in the removal helper. The full suite below ran on the exact commit.

Exact-head full-suite command:

```sh
/usr/bin/time -p .venv/bin/python scripts/test_dtype_phase4b_oracle.py > target/dtype-phase4b-after-audit-head.log 2>&1
```

Result at `bb94278a3`: 288 tests, OK, real 53.70s. The original base run was 286 tests, OK, real 65.94s with the same timed command and output file `target/dtype-phase4b-before.log`: a measured local reduction of 12.24s (18.6%). One run per state, with other machine activity and two new controls, is not a hosted CI prediction.

Remaining limit: the fixture is serial within one `unittest` process. The audited cases neither inspect inode/ctime identity nor change extended attributes; those are not reset by a shared directory and would need per-case roots if a future test begins depending on them.

Final status: clean worktree, branch `perf/dtype-phase4b-fixture-reuse`, two local commits ahead of `origin/main`; no push, PR, cargo, full gate, or running fixture test process.
