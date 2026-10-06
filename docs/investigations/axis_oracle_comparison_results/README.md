# Axis oracle campaign evidence

The [investigation](../axis_oracle_comparison.md) explains the frozen baseline,
scoring rules and limitations. [mutants.csv](mutants.csv) records every mutant;
[costs.csv](costs.csv) compares setup and baseline execution;
[bounds.csv](bounds.csv) retains each calibration trial.
[summary.json](summary.json) contains counts and frozen bounds.
[manual-classifications.json](manual-classifications.json) explains every
unwitnessed survivor.

[receipts.tar.gz](receipts.tar.gz) preserves the original JSON receipts and log
bytes, the frozen runner, generated Lean cases, setup measurements and
supplementary probes. It is approximately 5.1 MiB compressed. Its
[index](receipts-index.json) records the SHA-256 and length of every evidence
file, with original paths for content-addressed logs. Supplementary probes do
not enter the canonical campaign score. No tool installation or build target
is included.

To regenerate the tables without installing any proof tool, use a uv-managed
Python 3.11 in a checkout of this PR. Extract the archive into an empty directory:

```text
.venv/bin/python -m tarfile -e docs/investigations/axis_oracle_comparison_results/receipts.tar.gz /tmp/axis-receipts
```

Then run the reporter (the following is one command):

```sh
.venv/bin/python scripts/axis_oracle_report.py /tmp/axis-receipts/raw \
  --manual docs/investigations/axis_oracle_comparison_results/manual-classifications.json \
  --output /tmp/axis-tables \
  --extras /tmp/axis-receipts/setup-costs.json \
  --log-store /tmp/axis-receipts/logs \
  --vermilion-cases /tmp/axis-receipts/vermilion-cases
```

The reporter verifies log hashes, complete oracle execution, frozen Kani settings,
manual classifications and Vermilion's case identity and four-function lowering.
It resolves log paths in memory and retains recorded labels alongside explicit
adjudications. The original receipts are not rewritten. The archive includes
its own identical index, so it can be shared separately from this directory.

To repeat measurements, follow the investigation's isolated-checkout instructions
at frozen commit `d68ae1fc1cce832275977fecc3a705640703a99d`. The archived runner is
an audit copy; running it from the archive would give it a different repository
root. Absolute command paths in receipts describe the original machine and are
provenance, rather than portable installation locations. Calibration provenance
documents the unchanged admission/normalization trials imported from the
preceding freeze. Results measure this operator inventory on the recorded host,
not all possible defects or a hardware-independent capacity limit.
