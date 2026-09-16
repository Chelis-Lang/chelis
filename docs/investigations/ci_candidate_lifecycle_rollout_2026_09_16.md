# CI candidate lifecycle rollout assessment, 2026-09-16

This is a point-in-time assessment of pull requests opened after the candidate
lifecycle changes in #2099. It is historical evidence, not a CI contract. The
repeatable measurement surface is
[`scripts/ci_pr_lifecycle_report.py`](../../scripts/ci_pr_lifecycle_report.py).

## Finding

The latest candidate remained reasonably bounded, but the complete pull-request
lifecycle did not. In the fully post-change cohort #2105, #2106, and
#2111-#2115:

| Measure | Observed |
| --- | ---: |
| CI candidates | 38 |
| package-expansion dispatches | 10 |
| cumulative CI/Hull raw job-minutes | 3,207.033 |
| sum of each PR's latest-candidate CI/Hull raw job-minutes | 645.267 |
| lifecycle amplification | 4.97x |
| package-expansion raw job-minutes | 282.033 |
| all measured hosted raw job-minutes | 3,489.067 |
| explicit agent-wait minutes | 129.413 |

These are raw summed job-minutes: for every hosted job, its finish time minus
its start time, then summed. They are not workflow elapsed time, billing-rounded
runner minutes, agent waiting, or dollar cost. Concurrent jobs make raw
job-minutes larger than workflow wall time. Runner rates and billing rounding
were not part of the evidence, so this assessment does not convert the result
to an exact cost.

The 4.97x ratio means the final candidate's validation burden was not the main
problem. Repeated candidates and repeated package expansions accumulated nearly
five times the CI/Hull work represented by the cohort's latest candidates.

The trace-backed cause ledger attributed all 38 candidates and all 90 hosted
runs:

| Cause | Candidates | CI/Hull raw job-minutes | expansion raw job-minutes | explicit agent-wait minutes |
| --- | ---: | ---: | ---: | ---: |
| initial candidate | 7 | 443.750 | 0 | 0 |
| review repair | 11 | 922.433 | 108.617 | 36.179 |
| ordinary content push | 2 | 85.833 | 0 | 0 |
| non-conflicting rebase/base update | 0 | 0 | 0 | 0 |
| trivial or hand-resolved conflict rebase | 3 | 308.700 | 20.617 | 35.758 |
| base retarget/stack collapse | 1 | 109.917 | 0 | 0 |
| CI-policy or CI repair | 14 | 1,336.400 | 0 | 57.475 |
| package-expansion rerun | 0 | 0 | 152.800 | 0 |
| unknown | 0 | 0 | 0 | 0 |

The largest cost was CI-policy and CI-repair churn, followed by review repairs.
No non-conflicting rebase occurred in the cohort. Conflict rebases were costly
because the targeted selector failed closed, but they were not the dominant
cause of the observed amplification.

## Why #2099 did not reduce rebase work

No routine "refresh from main" rebases were found in this cohort. The four
post-open base updates were necessary because of conflicts, stack movement, or
base-sensitive overlap. Avoiding gratuitous rebases therefore would not have
removed this cohort's rebase cost.

The targeted-rebase selector introduced by #2099 had no effect. Both CI and Hull
invoked `scripts/ci_rebase_reuse.py`, which queries GitHub through `gh api`, but
the selector steps omitted `GH_TOKEN`. The verifier failed closed, so every
otherwise eligible necessary rebase used full CI. This is an authentication
defect in the rollout, not evidence that all four rebases required full
validation.

## Pull-request and trace findings

### #2111

#2111 had a healthy two-candidate lifecycle. Its second candidate was a genuine
review repair, not branch-refresh churn.

### #2112

#2112 required a stack-conflict rebase. It paid full CI because the targeted
selector could not authenticate, not because the available evidence established
that the complete suite was necessary.

### #2113

#2113 produced nine candidates and four package expansions. The first reviewer
was explicitly stopped after its first P1 finding. Later rounds then found
sibling instances of the same transform-erasure defect class. The serial review
shape produced additional repair candidates, and package expansion was
dispatched before the review surface had settled.

### #2114

#2114 had four reviewer attempts on the same SHA: a context cap, a malformed tool
call, an interruption, and then a successful review. Those attempts consumed
about 35 local agent minutes without creating additional hosted workflow runs.
That is agent/reviewer overhead, not hosted CI compute and not automatically an
agent-wait-for-CI interval.

### #2115

#2115 produced ten candidates and three package expansions. The change-owned
planner reported the first unmapped path and stopped, so subsequent pushes
revealed further paths one at a time. Several package-expansion dispatches were
labelled final before candidate planning and review had settled. The standing
reviewer was correctly retained across the later semantic rebase.

### #2105 and #2106

No local creator traces existed for #2105 or #2106, so their agent decisions and
waiting intervals cannot be reconstructed from this machine. GitHub evidence
showed two avoidable early documentation pushes on #2105 and serial
alias-defect review repairs on #2106, but it does not establish what an agent
was doing between runs.

## Waiting and causation audit

GitHub can establish run IDs, PR associations for pull-request events, heads,
timestamps, attempts, job execution, and workflow windows. It cannot reliably
establish why a candidate was pushed, whether a conflict was trivial or
semantic, whether an agent actually waited, or whether the agent did useful work
while CI ran.

The audit therefore used workflow timestamps only for hosted execution and used
local traces for agent waiting. Fourteen intervals had explicit start and end
events showing that the agent had entered and left a CI wait. They totalled
129.413 minutes: 36.179 for review repairs, 35.758 for conflict rebases, and
57.475 for CI-policy or CI repairs. No supported wait interval belonged to an
initial candidate, ordinary content push, non-conflicting rebase, retarget, or
package-expansion-only rerun.

The known roughly 35-minute #2114 reviewer-retry interval is excluded because
the agent was trying to obtain a valid review, not waiting for hosted CI. No
creator traces existed for #2105 or #2106, so their hosted causes use public
GitHub evidence and their agent-wait totals remain zero rather than inferred.
Three jobs had inverted start and completion timestamps and were excluded from
minute totals; the affected workflow runs were 35019801558, 35027127544, and
35070318237.

Per-cause totals require each candidate or run to be linked to a cause with
evidence. The report tool accepts those facts in an attribution ledger and
marks all other rows `inferred` or `unknown`.

The fixed cause vocabulary is:

- `initial-candidate`
- `review-repair`
- `ordinary-content-push`
- `non-conflicting-rebase-base-update`
- `trivial-or-hand-resolved-conflict-rebase`
- `base-retarget-stack-collapse`
- `ci-policy-ci-repair`
- `package-expansion-rerun`
- `unknown`

For each cause the report separately totals candidate count, workflow-run and
attempt counts, CI/Hull raw job-minutes, package-expansion raw job-minutes,
summed workflow wall-minutes, and ledger-supplied agent-wait minutes.

## Repeatable report

### Live collection

Live collection queries the PR records and the three owning workflows, fetches
every job and rerun attempt, writes a normalized snapshot, and emits JSON plus
Markdown:

```console
python3 scripts/ci_pr_lifecycle_report.py \
  --repository Chelis-Lang/chelis \
  --prs 2105,2106,2111-2115 \
  --since 2026-09-12T00:00:00Z \
  --attribution-ledger /path/to/attribution.json \
  --snapshot-output /tmp/ci-lifecycle-github.json \
  --json-output /tmp/ci-lifecycle-report.json \
  --markdown-output /tmp/ci-lifecycle-report.md
```

The `--since` boundary is mandatory for live collection so the workflow scan is
bounded. Pull-request workflow runs carry their PR association. Manual
package-expansion and retarget dispatches do not carry enough safe PR/head
identity in the workflow-run listing; give those run IDs explicit
`run_attributions` rows. Without a ledger assertion that the manual-run scope
is complete, the report labels retarget and package-expansion counts as lower
bounds rather than silently assigning dispatches to a PR. Completed
pull-request runs can also lose their `pull_requests[]` association; for those,
the collector requires an exact head repository/ref match inside the PR's
activity window.

### Offline rerun

The normalized snapshot makes the GitHub part reproducible without another API
query:

```console
python3 scripts/ci_pr_lifecycle_report.py \
  --prs 2105,2106,2111-2115 \
  --github-input /tmp/ci-lifecycle-github.json \
  --attribution-ledger /path/to/attribution.json \
  --json-output /tmp/ci-lifecycle-report.json \
  --markdown-output /tmp/ci-lifecycle-report.md
```

### Attribution ledger

The ledger keeps human or trace-derived facts separate from GitHub facts. A
minimal example is:

```json
{
  "schema": "chelis-ci-pr-lifecycle-attribution-v1",
  "manual_run_scope_complete": true,
  "manual_run_scope_evidence": "audited manual runs for the cohort time window",
  "candidate_attributions": [
    {
      "pr_number": 2112,
      "head_sha": "0123456789abcdef0123456789abcdef01234567",
      "candidate_sha": "89abcdef0123456789abcdef0123456789abcdef",
      "cause": "trivial-or-hand-resolved-conflict-rebase",
      "evidence": "trace session and turn identifying the conflict resolution"
    }
  ],
  "run_attributions": [
    {
      "run_id": 123456789,
      "pr_number": 2112,
      "head_sha": "0123456789abcdef0123456789abcdef01234567",
      "cause": "package-expansion-rerun",
      "evidence": "dispatch record for the exact reviewed head"
    }
  ],
  "agent_waits": [
    {
      "wait_id": "session-id:turn-42",
      "pr_number": 2112,
      "candidate_sha": "89abcdef0123456789abcdef0123456789abcdef",
      "cause": "trivial-or-hand-resolved-conflict-rebase",
      "started_at": "2026-09-15T14:03:00Z",
      "ended_at": "2026-09-15T14:11:30Z",
      "run_ids": [123456780, 123456781],
      "evidence": "trace events showing the agent entered and left a CI wait"
    }
  ]
}
```

Candidate and run attribution rows require an evidence string. Agent waits
require a stable wait ID, timestamps, cause, evidence, and either `head_sha` or
`candidate_sha`; use the synthetic candidate identity when a stacked pull
request keeps the same Git head across different tested merges. Related run IDs
are recorded when available and are checked against the PR and supplied
identity. Set `manual_run_scope_complete` only after every retarget and
package-expansion dispatch for the cohort has a run row, and name that audit in
`manual_run_scope_evidence`; otherwise the report labels those counts as lower
bounds. The tool rejects unsupported completeness claims, unknown cause
spellings, duplicate identities, unknown or mismatched wait-run links, and
negative intervals.

## Interpretation

This cohort does not show the intended targeted-rebase effect because the
selector never authenticated. It does show that measuring only the latest
candidate hides the dominant spend mechanism. CI-policy or CI repairs alone
consumed 1,336.400 CI/Hull raw job-minutes, and review repairs consumed another
922.433. Necessary conflict rebases consumed 308.700; routine non-conflicting
rebases consumed none. Premature package-expansion reruns added 152.800
raw job-minutes.

Future rollout checks should publish both latest-candidate and
cumulative-lifecycle figures and retain the attribution ledger needed to
explain their difference. They should also report explicit trace-backed
agent-wait intervals separately from hosted execution and reviewer overhead.
