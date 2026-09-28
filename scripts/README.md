# Script owners

Run Python tools with this worktree's `.venv/bin/python`. Commands and gate
coverage are listed in [`docs/local_gate.md`](../docs/local_gate.md).

| Area | Owner |
|---|---|
| Local validation | [`gate.py`](gate.py), [`test_gate.py`](test_gate.py) |
| Generated artifacts | [`regen_all.py`](regen_all.py), [`test_regen_all.py`](test_regen_all.py); each leg names its writer and checks |
| CI selection and contracts | [`ci_change_owned.py`](ci_change_owned.py), [`.config/ci-test-targets.toml`](../.config/ci-test-targets.toml), [`docs/ci_validation.md`](../docs/ci_validation.md) |
| Releases | [`changelog.py`](changelog.py), [`bump_compiler_pins.py`](bump_compiler_pins.py), [`changelog.d/README.md`](../changelog.d/README.md) |
| Local downstream checks | [`nautilus_local_gate.py`](nautilus_local_gate.py), [`docs/manual_gates.md`](../docs/manual_gates.md) |

The `loc-report` command is owned by [`py/src/chelis_tools/loc_report.py`](../py/src/chelis_tools/loc_report.py).
It prints Markdown to stdout; pass `--output PATH` to save a report.
