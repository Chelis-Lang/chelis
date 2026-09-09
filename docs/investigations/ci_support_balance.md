# Balance integration support across workspace workers

In [September 8 run 34278889672](https://github.com/Chelis-Lang/chelis/actions/runs/34278889672),
workspace shard 1 finished in 15:00 and shard 2 in 21:51. Shard 2 spent its
last 5:57 running all three integration support commands: lowering-trace and
front-end performance consumed about 2:27, then unrepresentable-domain consumed
about 3:30. This change addresses the support scheduling portion of
[#1503](https://github.com/Chelis-Lang/chelis/issues/1503).

## Execution ownership

The existing two workers retain their workspace test partitions. After those
tests, shard 1 runs the lowering-trace and front-end performance commands;
shard 2 runs the unrepresentable-domain command. Both reuse their own warm
workspace build. The full `gate.py integration --support-only` command retains
all three commands in their original order. No test filters, feature flags,
required aggregates, cache writers, runner types, or artifact retention change.

The default-feature selection census runs on shard 1 before lowering-trace
changes the warm feature configuration. Its generalization counterpart stays
on generalization shard 1. Moving the default census after lowering-trace would
rebuild the workspace merely to enumerate it.

```console
python3 scripts/gate.py integration --support-only --support-slice frontend
python3 scripts/gate.py integration --support-only --support-slice domain
```

The slices derive commands from the canonical integration stage. Tests bind
them to the exact existing commands, require their ordered union to equal the
full support stage, and reject missing or duplicated workflow ownership and
incorrect census order. Invalid slice values or use outside integration
support-only mode fail argument validation. Both worker results continue to
feed the same fail-closed workspace and integration aggregates.

Holding the measured durations constant, moving 2:27 of work would reduce the
longer workspace job from 21:51 to about 19:24; the other would become about
17:27. This is a scheduling estimate, not a guaranteed workflow speedup.
Hosted CPU performance, queue delays, and the other required jobs can dominate.
The change adds no jobs or archives; total compute should remain similar, with
cache effects and per-job billing rounding determined by the hosted run.

## Validation

```console
.venv/bin/python -m unittest scripts.test_gate scripts.test_hosted_validation
python3 scripts/gate.py --fast
```

Hosted CI on the pushed candidate is the execution oracle. It must execute both
workspace partitions, each support command once on its assigned worker, both
selection censuses, and the existing required aggregates. Compare JUnit receipt
identities with a same-code baseline, then separate queue time, execution time,
and rounded runner minutes when reporting performance. Build sharing remains
outside the landing change; #1502 and the remaining #1503 work stay open.
