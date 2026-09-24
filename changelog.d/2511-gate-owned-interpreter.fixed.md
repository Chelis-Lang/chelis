`python3 scripts/gate.py` now runs on the checkout's own interpreter, which is
its Devenv state venv or else its `.venv`. It does so even when another
checkout's venv comes first on `PATH`. Previously the gate adopted that
foreign venv and exported it to every stage, and the runtime-representation
oracle's capacity census refused it at the end of a `--validation` run. See
[#2511](https://github.com/Chelis-Lang/chelis/issues/2511).
