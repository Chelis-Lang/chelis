When a checkout has its own interpreter (its Devenv state venv, or else its
`.venv`) and `PYO3_PYTHON` is unset, `python3 scripts/gate.py` now runs on that
interpreter. This holds even when another checkout's venv comes first on
`PATH`. Previously the gate adopted that foreign venv and exported it to every
stage, and the runtime-representation oracle's capacity census refused it at
the end of a `--validation` run. An owned `.venv` that does not start as the
venv now stops the gate with exit 2 instead of re-launching it. See
[#2511](https://github.com/Chelis-Lang/chelis/issues/2511).
