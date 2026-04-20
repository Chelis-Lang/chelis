# Local Downstream Gates

`nautilus_local_gate.py` runs Chelis against the real Nautilus checkout with a
local compiler binary instead of a release artifact.

Typical local loop:

```sh
cargo build -p chelis-cli
python3 scripts/nautilus_local_gate.py baseline
python3 scripts/nautilus_local_gate.py tensor-grad
python3 scripts/nautilus_local_gate.py tensor-fold
python3 scripts/nautilus_local_gate.py eval-imports
```

Notes:

- By default it uses `target/debug/chelis` and a sibling `../nautilus`
  checkout.
- `baseline` is the shipped-surface proof: it delegates to Nautilus's
  `tests/run_static_checks.py` and `tests/run_numeric_tests.py`.
- `eval-imports` works on a temporary copy of Nautilus so it can rewrite the
  `reef.toml` compiler pin to the local Chelis version without touching the real
  downstream checkout.
- `tensor-grad` and `tensor-fold` are local blocker canaries for the native C
  build path. They exist to keep blocker debugging local and cheap; release
  publishing should happen only after the relevant canaries and the baseline are
  green.
