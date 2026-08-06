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
  runPython = relativePath: ''
    import os
    import sys

    script = "${config.devenv.root}/${relativePath}"
    os.execv(sys.executable, [sys.executable, script, *sys.argv[1:]])
  '';
in
{
  scripts = {
    "chelis-gate" = {
      description = "Run the Chelis repository gate";
      package = config.languages.python.package;
      exec = runPython "scripts/gate.py";
    };

    # Regenerate or `--check` the committed Cargo.nix graph. crate2nix comes
    # from the ci consumer module (config.outputs.crate2nix); cargo is on the
    # shell PATH from languages.rust.
    "regenerate-crate2nix" = {
      description = "Regenerate or freshness-check the committed Cargo.nix graph";
      exec = ''
        export PATH="${config.outputs.crate2nix}/bin:$PATH"
        exec "${config.languages.python.package}/bin/python" \
          "${config.devenv.root}/scripts/regenerate_cargo_nix.py" "$@"
      '';
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
