# CI queue capacity and status reporting

On 2026-09-08 at 20:40 UTC, Chelis had 60 running hosted jobs (55 Ubuntu,
5 macOS) and 31 queued jobs (24 Ubuntu, 7 macOS). This matches GitHub Team's
60-job total and 5-job macOS limits. The sample covers repository resource use;
the PR timing investigation includes only PRs opened since 2026-09-01.

In [run 34272131841](https://github.com/Chelis-Lang/chelis/actions/runs/34272131841),
`macOS Smoke` and `CI Test Telemetry` were runnable at 20:33:17 UTC but still had
no assigned runner at the snapshot. Both aggregate jobs run on Ubuntu. Across
31 successful first-attempt recent PR runs, their median execution times were
7 seconds and 22 seconds respectively. The delay was acquiring capacity.

## Short-job capacity

The `CHELIS_CONTROL_RUNNER` repository variable selects a separate runner label
for changes detection, short validation, and aggregate/report jobs in CI,
Conformance, and Changelog. Without the variable, these jobs use `ubuntu-latest`.
The workflow's dependencies, status names, and failure checks are unchanged.
Tests and builds continue using their existing runners.

Provision one GitHub larger-runner pool with this configuration:

| Setting | Value |
| --- | --- |
| Name/label | `chelis-control` |
| OS | GitHub Ubuntu 24.04 x64 image |
| Size | 2-core Linux, 8 GB RAM, 75 GB storage |
| Maximum concurrent runners | 4 |
| Static IP | Disabled |
| Runner group | `chelis-control`, restricted to `Chelis-Lang/chelis` |

Larger runners have separate concurrency capacity and are billed even for public
repositories. The four-runner cap bounds concurrency, not a monthly spending cap.
The pool may still have provisioning delays; it removes contention with the
60 standard runners rather than guaranteeing instant starts. See GitHub's
[limits](https://docs.github.com/en/actions/reference/limits#job-concurrency-limits-for-github-hosted-runners)
and [larger-runner reference](https://docs.github.com/en/actions/reference/runners/larger-runners).

Create the group with repository access restricted before creating the runner.
Use the current machine-size and image catalogs from the
[hosted-runner API](https://docs.github.com/en/rest/actions/hosted-runners), then
verify the resulting pool and group access. Set `CHELIS_CONTROL_RUNNER` to
`chelis-control` only after the pool is available, and inspect the assigned runner
on the next short job. Removing the variable returns subsequent jobs to standard
Ubuntu; already queued jobs retain the label with which they were scheduled.

Provisioning was not performed when this change was authored: the CLI credential
could read workflows but the organization runner API returned HTTP 403. This
document specifies the proposed configuration, not evidence of an active pool.

## Concurrency support request

Submit the following to GitHub Support from an organization owner account:

> Please increase GitHub-hosted Actions concurrency for Chelis-Lang (Team plan)
> from 60 to 120 total jobs, and from 5 to 15 macOS jobs. Chelis-Lang/chelis has
> multiple active compiler PRs, each with Linux test partitions and two macOS
> partitions. At 2026-09-08 20:40 UTC we observed 60 running jobs (55 Ubuntu,
> 5 macOS) and 31 queued jobs (24 Ubuntu, 7 macOS). In run 34272131841, required
> aggregate job 102227301764 and reporting job 102227301841 were runnable at
> 20:33:17 UTC but had no runner assigned. Their typical execution times are
> only seconds. Recent PR #1635's run 34270120202 took 51:03, including a macOS
> worker's 23:35 queue delay and 24:50 execution. We are reducing redundant
> compilation and moving short jobs to a bounded larger-runner pool. Please
> confirm the available total and macOS limits and any additional requirements.

This text is ready to submit; no support case is claimed here. Runner
administration and support submission require the corresponding account access.

## Monitor one PR head

```console
.venv/bin/python scripts/ci_status.py 1638 --head FULL_HEAD_SHA
.venv/bin/python scripts/ci_status.py 1638 --head FULL_HEAD_SHA --watch
```

The newest check run for each context and app supersedes earlier workflow suites,
including canceled runs caused by PR edits. Legacy commit statuses are still
checked independently when the requirement permits any app.

The tool reads classic branch protection and effective branch rules, paginates
all current check runs and commit statuses, and separates required checks from
other work. It rechecks the head, PR comparison base, live base branch tip, and
open state after each snapshot. The separate ref read matters because GitHub's
PR `baseRefOid` can retain the older comparison commit after `main` advances.
Use `--watch` as a background process, as required by the agent contract.

`required_passed` means only that the observed required status checks pass on
that head. It does not imply mergeability, review approval, or satisfaction of
validation requirements outside branch protection. GitHub accepts `success`,
`neutral`, and `skipped` check conclusions. Missing requirements, failed API reads,
or a changed PR identity never produce readiness. The tool needs read access to
both protection APIs; it fails closed if either is unavailable.

Exit codes: 0 means required checks pass and no observed other check failed
(other checks may still be pending); 1 means a required or other check failed;
2 means required checks are pending or missing in a single snapshot; 3 means
status could not be established. A watch exits on 0, 1, or 3. In particular,
queued `CI Test Telemetry` stays visible under `other_pending` without being
presented as a branch-protection blocker. A queued `macOS Smoke` remains required.

Validation:

```console
.venv/bin/python -m unittest scripts.test_ci_status scripts.test_ci_control_runners
```
