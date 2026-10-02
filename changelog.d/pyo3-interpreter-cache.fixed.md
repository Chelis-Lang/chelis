CI and local gates refresh cached PyO3 build configuration when the selected
Python interpreter changes behind an unchanged venv path. Native builds reject
discovery overrides and CI uses only the selected interpreter's library directory.
