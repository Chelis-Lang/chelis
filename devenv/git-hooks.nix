{ config, ... }:

let
  commitMessageChecker = ../scripts/check_commit_message.py;
in
{
  # Keep this catalog inactive until a separate policy change enables hooks.
  git-hooks.hooks = {
    actionlint.enable = false;
    check-added-large-files.enable = false;
    check-case-conflicts.enable = false;
    check-executables-have-shebangs.enable = false;
    check-json.enable = false;
    check-merge-conflicts.enable = false;
    check-python.enable = false;
    check-symlinks.enable = false;
    check-toml.enable = false;
    check-yaml.enable = false;
    detect-private-keys.enable = false;
    end-of-file-fixer.enable = false;
    fix-byte-order-marker.enable = false;
    forbid-new-submodules.enable = false;
    mixed-line-endings.enable = false;

    nixfmt.enable = false;

    rustfmt = {
      enable = false;
      settings.check = true;
    };

    shellcheck = {
      enable = false;
      files = "^crates/chelisup/bootstrap/chelisup\\.sh$";
    };

    trim-trailing-whitespace = {
      enable = false;
      args = [ "--markdown-linebreak-ext=md" ];
    };

    no-ai-authorship = {
      enable = true;
      name = "Reject AI authorship markers";
      entry = "${config.languages.python.package}/bin/python ${commitMessageChecker}";
      language = "system";
      package = config.languages.python.package;
      pass_filenames = true;
      stages = [ "commit-msg" ];
    };
  };
}
