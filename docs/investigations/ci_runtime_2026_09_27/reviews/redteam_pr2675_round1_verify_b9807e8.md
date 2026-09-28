# PR #2675 round 1 repair verification

**Verdict: closed; reviewer satisfied.** Local unpushed head `b9807e86450627460c0e6a68c87911418a005a5e` repairs the in-scope P2 primary-diagnostic selection finding from the initial report. No new finding arose from the repair.

The repair at `scripts/capacity_census_cache_publication.py:164-166` selects the earliest line-start `error:` or `error[code]:` in stderr and takes at most 3,500 characters from there. It therefore selects by position rather than favoring every coded error over every plain one. The accompanying regression covers the original order.

## Executed evidence

1. Re-ran the original real-rustc reproduction, with the assertion changed to require the first plain error. Exact command:

```sh
PYTHONPATH=scripts .venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from capacity_census_cache_publication import COMPILE_CASES, CompileOutcome, CachePublicationError, validate_compile_outcomes
with TemporaryDirectory() as d:
    source = Path(d) / 'errors.rs'
    source.write_text('#[derive(DoesNotExist)]\nstruct A;\nfn x() { missing_crate::x(); }\n')
    rustc = subprocess.run(['rustc', '--edition=2024', '--crate-type=lib', '--emit=metadata', '-o', str(Path(d) / 'errors.rmeta'), str(source)], capture_output=True, text=True)
    stderr = 'warning: earlier note\n' + rustc.stderr
    outcomes = tuple(CompileOutcome(case.name, 1 if case.name == 'library' else (0 if case.success else 1), stderr if case.name == 'library' else ((case.error + ' ' + case.diagnostic) if case.error else ''), 'source', ('rustc',)) for case in COMPILE_CASES)
    try:
        validate_compile_outcomes(outcomes)
    except CachePublicationError as failure:
        message = str(failure)
        print('rustc first:', next(line for line in rustc.stderr.splitlines() if line.startswith('error')))
        print('census first:', message.split(': ', 2)[-1].splitlines()[0])
        print('first preserved:', 'DoesNotExist' in message, 'bounded:', len(message) < 4500)
        assert 'DoesNotExist' in message and 'E0433' in message and len(message) < 4500
    else:
        raise AssertionError('expected census rejection')
PY
```

Both `rustc first` and `census first` printed `error: cannot find derive macro `DoesNotExist` in this scope`; `first preserved: True bounded: True`.

2. `PYTHONPATH=scripts .venv/bin/python -m unittest scripts.test_ci_rebase_reuse scripts.test_capacity_census_cache_publication` passed 31 tests.

3. Ran an inline `PYTHONPATH=scripts .venv/bin/python - <<'PY'` matrix calling `validate_compile_outcomes` with three unexpected-library-failure stderr shapes: coded error before plain error, plain error after 1,000 note lines, and no error marker. Each report began with the expected first diagnostic or safe fallback; message lengths were 125, 125, and 3,580 characters respectively, all under the 4,500-character assertion. The exact stderr variants were `warning: setup\nerror[E0433]: first coded\nerror: later plain\n`, `warning: setup\n` + `note\n` × 1,000 + `error: first plain\nerror[E0433]: later coded\n`, and `unclassified rustc failure\n` + `detail\n` × 1,000. This checks the adjacent selection paths introduced by the repair.

No full census build was needed for this local diagnostic repair. All probe files lived in `TemporaryDirectory` and were removed; no tracked source was mutated.

Final status: `git status --porcelain=v1` was empty; `git diff --exit-code` and `git diff --cached --exit-code` succeeded; `.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/ci-reuse-expansion-2673` reported `VERDICT: FREE`, exact head `b9807e86450627460c0e6a68c87911418a005a5e`, clean tree, free lease, no scoped processes, and no Git operation at 2026-09-27T16:04:19Z.
