# The 2026-09-10 CI cadence change and what merged behind it

On 2026-09-10 at 18:04 UTC, `062c29c19` (#1736) moved a large part of the test suite
off the per-pull-request path and into the nightly `heavy-e2e.yml` workflow. Over the
following day, nine integration rows across five test targets went red on `main` with no
check reporting it, and nine issues were filed against the result.

This directory is the investigation into which merges caused that, written while the
repairs were being made. It is history and evidence rather than a contract: nothing here
is normative, and where it describes `main` it describes `main` on 2026-09-11.

## The finding, in one paragraph

**One pull request accounts for nearly all of it.** `22b193cf7` (#1693) is the immediate
child of the cadence change and merged **fourteen minutes after it**. Its squash carries
36 commit messages across 74 files, +6835/-519, under a subject line naming none of the
mechanisms that broke. Six of the nine rows, and five separately filed issues, bisect to
it. The failures are not a diffuse consequence of running fewer checks; they are one
large change landing in the exact window where nothing could see it.

## The files

| file | what it is |
|---|---|
| [`regression_analysis.md`](regression_analysis.md) | The main writeup. Which pull requests caused which breaks, the nightly's own failures, merge volume and exposure, a repair-cost triage, which job catches what, and §14's mechanism-by-mechanism account of #1693. |
| [`attribution_of_the_unowned_rows.md`](attribution_of_the_unowned_rows.md) | How the six unattributed `chelis-cli` rows of chelis#1776 were measured to one commit, using a four-signal probe over a sampled commit ladder rather than a per-test bisect. |
| [`interpreter_probe_diagnosis.md`](interpreter_probe_diagnosis.md) | chelis#1829: why `chelis eval` on a JSON program went from about two seconds to over five minutes. An unmemoized, exponentially branching kernel-decision probe that #1693 exposed by removing an accidental guard. |
| [`interpreter_probe_bisect.md`](interpreter_probe_bisect.md) | The measurement behind that diagnosis, with the confirming parent/child pair. |
| [`interpreter_probe_design.md`](interpreter_probe_design.md) | chelis#1835: the structural repair for the same defect class, replacing a thread-local armed flag with a borrow-checked session so that forgetting to arm it stops compiling. Written as a design, not as a decision. |
| [`reproduce_interpreter_probe.py`](reproduce_interpreter_probe.py) | The reproduction harness. Times `check` and `eval` as separate invocations, with a fresh reef home and package directory per measurement, because `~/.cache/chelis` is workstation-wide and will otherwise serve a result computed at a different commit. Roughly five seconds per commit, against forty-five minutes for the affected test rows. |

## Why it is worth keeping

Three things in here are reusable beyond the incident.

**The measurement method.** `attribution_of_the_unowned_rows.md` records a probe that
reads several independent signals off one build, with a passing row as a negative control,
and tri-valued signals so a commit that fails to build cannot read as the regression. That
localises several independent boundaries from one commit ladder instead of one bisect per
symptom.

**The repair ledger.** Every issue closed during this work was verified on `main` and not
inferred from a merge, because merging closes nothing in this repository and a merged
pull request is not proof its issue is resolved. §14.5 records what each closure actually
rests on.

**The recurrence.** Two further defects of the identical shape landed *during* the
investigation, from #1828, and were found only because an agent rebased across them. That
is the argument for chelis#1824 being the structural item: the defect is not one bad
merge, it is a per-pull-request selection whose purpose nobody has decided, which keeps
admitting the same class a day at a time.

## Status of these files

Point-in-time artifacts, **not maintained automation**. The harness is exempt from the
phase-gate and example-corpus policies for the same reason the probe corpus in
[`../probes/`](../probes/) is: it is neither an executable example nor a gate script, and
it describes the tree as it stood on 2026-09-11. It is kept here rather than in that
directory because `../probes/README.md` indexes one named sweep and should keep matching
its own contents.

The live trackers are the issues, not these documents. Where a file here describes an
issue as open that has since closed, the issue is right and the file is history.
