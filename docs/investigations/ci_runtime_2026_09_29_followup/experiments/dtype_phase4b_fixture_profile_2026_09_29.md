# Phase 4B fixture cost follow-up

Compared the same 228 `ContractValidationTests` cases on original fresh-root
`e16b2f795045e26a366cfbba570b73a0c6278ea3` and shared-root
`bb94278a366c59054eefe04f27964c51a25563c5`. Both passed. No source
changed. The detached profile worktree was removed; the task worktree is clean
at `bb94278a3`, two local commits ahead of `origin/main`. No push or PR.

| 228-case phase (inclusive wall seconds) | Original | Shared | Fresh-root cache probe |
|---|---:|---:|---:|
| Class fixture preparation | 0.000 | 0.010 | 0.000 |
| Per-case setup, 228 calls | 1.100 | 0.000 | 0.727 |
| Copy/write within setup phases | 0.926 (6,156 copies) | 0.008 (27 copies) | 0.548 (6,156 writes) |
| `validate_contract`, 492 calls | 33.775 | 32.645 | 33.483 |
| Other test-body work | 2.773 | 2.720 | 2.678 |
| Per-case teardown, 228 calls | 0.400 | 0.370 | 0.376 |
| Temporary-directory cleanup *inside teardown or class cleanup* | 0.400 (228) | 0.001 (1) | 0.376 (228) |
| Whole 228-case suite | 38.057 | 35.754 | 37.273 |

The direct setup/teardown saving of the shared fixture is about **1.12s**.
Another 1.13s of its 2.30s suite difference is faster oracle execution in
that run, with the same 492 calls. The earlier 65.94s versus 53.70s
full-module difference cannot be attributed to fixture preparation. These
single local runs do not predict hosted CI.

The cache probe pre-read the 27 contract files once and wrote those bytes into
each original `setUp` destination. It kept a new `TemporaryDirectory`, real
files, and cleanup **for every case**. It passed all 228 cases and saved 0.37s
of setup. Reflinks need platform-specific fallbacks; hard links share mutations.

**Recommendation:** do not ship the shared-fixture commits for CI speed.
Keep the original fresh-per-test fixture. The cache needs a hosted case for
source-read cost; 0.37s locally does not justify implementation. Future speed
work should target the 33s oracle body while preserving negative cases.

## Commands

From each worktree, the original and shared profiles used the inline
command below with the corresponding `base` or `shared` log path. The cache
probe used the original worktree, `cached-fresh` log path, and replacement
snippet below. The logs remain in this research `target/` directory.

```sh
/usr/bin/time -p .venv/bin/python - > /Users/robertronan/chelis-worktrees/ci-selection-assessment/target/dtype-phase4b-base-subphases.log 2>&1 <<'PY'
import functools, json, shutil, sys, tempfile, time, unittest
from collections import Counter, defaultdict
sys.path.insert(0, 'scripts')
import test_dtype_phase4b_oracle as t
seconds, calls = defaultdict(float), Counter()
def timed(label, function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try: return function(*args, **kwargs)
        finally:
            seconds[label] += time.perf_counter() - start
            calls[label] += 1
    return wrapper
cls = t.ContractValidationTests
names = unittest.defaultTestLoader.getTestCaseNames(cls)
cls.setUp = timed('setup', cls.setUp)
cls.tearDown = timed('teardown', cls.tearDown)
for name in names: setattr(cls, name, timed('body', getattr(cls, name)))
original_class_setup = cls.setUpClass
cls.setUpClass = classmethod(lambda klass: timed('class_setup', original_class_setup)())
t.oracle.validate_contract = timed('oracle', t.oracle.validate_contract)
shutil.copyfile = timed('copyfile', shutil.copyfile)
tempfile.TemporaryDirectory.cleanup = timed('temp_cleanup', tempfile.TemporaryDirectory.cleanup)
suite = unittest.TestSuite(cls(name) for name in names)
start = time.perf_counter()
result = unittest.TextTestRunner(verbosity=0).run(suite)
seconds['suite'] = time.perf_counter() - start
seconds['body_without_oracle'] = seconds['body'] - seconds['oracle']
print(json.dumps({'tests': result.testsRun, 'ok': result.wasSuccessful(),
    'seconds': {k: round(v, 3) for k, v in sorted(seconds.items())},
    'calls': dict(sorted(calls.items()))}, sort_keys=True))
sys.exit(not result.wasSuccessful())
PY
```

For the cache probe, the exact replacement was:

```python
from pathlib import Path
blobs = {t.REPO_ROOT / r: (t.REPO_ROOT / r).read_bytes() for r in t.CONTRACT_FILES}
def cached_copy(source, destination):
    Path(destination).write_bytes(blobs[Path(source)])
    return destination
shutil.copyfile = timed('cached_write', cached_copy)
```
