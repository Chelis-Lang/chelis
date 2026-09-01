{
  config,
  lib,
  pkgs,
  ...
}:

let
  # Devenv copies path-valued script bodies into the Nix store. This adapter
  # loads the repository file so scripts that resolve paths from __file__ keep
  # the correct repository root.
  #
  # It also hands the script the interpreter Devenv activated, not the one that
  # runs the adapter. `scripts.<name>.package` can only name a Nix package, so
  # every launcher starts under the bare store CPython, which is neither what a
  # developer gets by typing `python` in the shell nor what PYO3_PYTHON points
  # at. scripts/gate.py reads that divergence correctly as an unmanaged runtime
  # and re-executes through uv, which then fails against Devenv's own
  # UV_PYTHON_PREFERENCE (chelis#1421).
  #
  # DEVENV_STATE is read at run time rather than interpolated, so the store
  # script names no worktree and still works when it is reached from outside a
  # Devenv shell, where there is no venv and sys.executable is already correct.
  runPython = relativePath: ''
    import os
    import sys

    script = "${config.devenv.root}/${relativePath}"
    state = os.environ.get("DEVENV_STATE")
    interpreter = sys.executable
    if state:
        activated = os.path.join(state, "venv", "bin", "python")
        if os.access(activated, os.X_OK):
            interpreter = activated
    os.execv(interpreter, [interpreter, script, *sys.argv[1:]])
  '';
in
{
  scripts = {
    "chelis-gate" = {
      description = "Run the Chelis repository gate";
      package = config.languages.python.package;
      exec = runPython "scripts/gate.py";
    };

    "chelis-reap-orphans" = {
      description = "List or remove orphaned Chelis build processes";
      package = config.languages.python.package;
      exec = runPython "scripts/reap_orphans.py";
    };
  }
  // lib.optionalAttrs pkgs.stdenv.isDarwin {
    "chelis-exec-preflight" = {
      description = "Check macOS first-exec health";
      package = config.languages.python.package;
      exec = runPython "scripts/preflight_exec_probe.py";
    };
  }
  // lib.optionalAttrs pkgs.stdenv.isLinux {
    "chelis-z3-test" = {
      description = "Run tests against a prebuilt Z3 library";
      package = config.languages.python.package;
      exec = runPython "scripts/z3_test.py";
    };

    "chelis-hip-test" = {
      description = "Run HIP tests with the reconciled local environment";
      package = config.languages.python.package;
      exec = runPython "scripts/hip_test.py";
    };
  };
}
