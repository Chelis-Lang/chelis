# PR #2675 red-team round 1 — 874c83d8b3f4920a650a0b57e9d6248dbadad76c

## Findings

### P2 — in scope — diagnostic selection skips the first rustc error

`scripts/capacity_census_cache_publication.py:163-170` searches all stderr for `error[` before it searches for `error:`. Rustc can emit a plain `error:` first and a coded `error[E…]` second. The census then starts its bounded excerpt at the second error, and the first error is absent from the CI message. This contradicts both the PR's “first bounded compiler error” claim and issue #2673's request to show at least the first error. The excerpt remains bounded and still shows a compiler error, so this is P2 residual diagnostic work, not a correctness or CI admission failure. Defect class: wrong primary-diagnostic selection. The changelog claim is affected by the same behavior.

Exact executed reproduction from the supplied worktree (the temporary Rust file and output directory were removed by `TemporaryDirectory`):

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
        print('first omitted:', 'DoesNotExist' not in message, 'bounded:', len(message) < 4500)
        assert 'DoesNotExist' not in message and 'E0433' in message and len(message) < 4500
    else:
        raise AssertionError('expected census rejection')
PY
```

Observed:

```text
rustc first: error: cannot find derive macro `DoesNotExist` in this scope
census first: error[E0433]: cannot find module or crate `missing_crate` in this scope
first omitted: True bounded: True
```

The source has two independent, plausible compile failures; this is not a malformed rustc transcript. The prefixed stderr also verifies behavior when a diagnostic is not at the beginning. A single coded rustc error behind a long prefix was separately confirmed to appear in a bounded message.

## Coverage and results

- `PYTHONPATH=scripts .venv/bin/python -m unittest scripts.test_ci_rebase_reuse scripts.test_capacity_census_cache_publication`: 30 passed.
- In fresh temporary Git repositories, both `base-rebase` and `base-merge` were classified by `ci_candidate_lifecycle.classify_update`, their synthetic merge commits were marked shallow, and `evaluate_rebase` selected `docs` with prior receipt run 202. With a code delta at `crates/fixture/src/rebase_delta.rs`, both selected `targeted`, package `fixture`, and `run_rust=True`. Thus both shallow paths reached the existing trusted receipt and owner-frontier checks.
- Negative controls in those same temporary repositories: no receipt or `reuse_eligible=False` selected `full`; a candidate whose raw second parent was the old PR head selected `full` with “head parent does not match”; an unmapped delta selected `full`. A raw commit object with `parent invalid-parent` was rejected by `_candidate_parents` as a malformed SHA. The raw object was written only into the temporary repository, which was removed.
- The cache module comment names exactly the six crate names in the fixture's `required` set: `chelis_compiler_api`, `chelis_ir`, `chelis_unord`, `serde`, `sha2`, and `bincode`. The new changelog describes the intended bounded diagnostic and the two shallow update forms. Existing `docs/ci_validation.md` and workflow wiring were inspected for the candidate shape and trusted-base verifier path; no executable examples changed.

Additional exact commands were `PYTHONPATH=. .venv/bin/python - <<'PY'` inline fixture probes using `RebaseRepository`, `merge_base_into_feature`, `make_candidate_shallow`, `evaluate`, `ci_candidate_lifecycle.classify_update`, and `ci_rebase_reuse.evaluate_rebase`; and a `git hash-object --literally -t commit -w --stdin` malformed-parent probe inside a temporary fixture repository. Output: `base-rebase base-rebase docs 202 full full`, `base-merge base-merge docs 202 full full`; targeted outputs were `targeted ['fixture'] True full` for both modes; the malformed parent raised `invalid current synthetic candidate parents: candidate parent must be a lowercase 40-character SHA`.

## Unvalidated and final status

I did not run the full cache-publication census or any new heavyweight Rust build. No hosted run of the future trusted-base version of `ci_rebase_reuse.py` was observed; the current candidate's hosted run uses the already trusted base copy, as the PR states. These limits do not affect the executed unit and adversarial evidence above.

Final supplied-worktree probe: `.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/ci-reuse-expansion-2673` reported `VERDICT: FREE`, exact head `874c83d8b3f4920a650a0b57e9d6248dbadad76c`, clean tree (0 modified, 0 staged, 0 untracked, 0 unmerged), free lease, no scoped processes, and no Git operation at 2026-09-27T16:00:30Z. `git status --porcelain=v1`, `git diff --exit-code`, and `git diff --cached --exit-code` were empty/successful. No tracked source was mutated.
