{ ... }:

{
  tasks."chelis:install-commit-hook" = {
    description = "Install the tracked commit-msg hook into the shared hooks directory";
    after = [ "devenv:enterShell" ];
    exec = ''
      set -eu

      # Hooks live in the COMMON git directory, which every worktree of a
      # clone shares. In a linked worktree `.git` is a file, so resolve it
      # rather than assuming `.git/hooks` exists here.
      # --path-format=absolute because --git-common-dir alone returns a
      # RELATIVE path from the main worktree and an absolute one from a linked
      # worktree; the core.hooksPath comparison below is a string compare and
      # would miss an equivalent path spelled the other way.
      hooks_dir="$(git rev-parse --path-format=absolute --git-common-dir)/hooks"
      template="$(git rev-parse --show-toplevel)/.githooks/commit-msg"

      if [ ! -f "$template" ]; then
        printf '%s\n' "commit hook: $template is missing; skipping install" >&2
        exit 0
      fi

      mkdir -p "$hooks_dir"
      if ! cmp -s "$template" "$hooks_dir/commit-msg"; then
        # Stage under a PID-unique name, then rename. Two things matter here.
        # `cp` truncates before it writes and git silently accepts a commit
        # whose hook is zero length, so copying over the live hook would leave
        # a window with the guard off; `mv` within one directory is atomic.
        # The name must also be unique per installer: every worktree of a clone
        # writes to this one shared directory, so concurrent devenv shells race,
        # and `cp` preserves the destination inode and mode. A shared staging
        # name therefore lets one installer truncate another's already-executable
        # file to zero bytes and the other publish it. mktemp rather than a
        # PID-derived name: `$$` in a subshell reports the parent's PID, so it
        # is not per-installer under every invocation shape.
        staged="$(mktemp "$hooks_dir/commit-msg.XXXXXX")"
        cp "$template" "$staged"
        chmod 755 "$staged"
        mv "$staged" "$hooks_dir/commit-msg"
      fi
      # Unconditionally, not inside the copy branch: `cmp` compares content, so
      # an installed copy that lost its executable bit matches the template and
      # would never be repaired. Git skips a non-executable hook silently, which
      # is the one failure mode this whole change exists to prevent.
      chmod +x "$hooks_dir/commit-msg"

      # An installed hook that cannot run is worse than none, because it
      # looks configured. core.hooksPath overrides the common directory, so
      # say so loudly rather than silently installing into a dead path.
      configured="$(git config --get core.hooksPath || true)"
      if [ -n "$configured" ] && [ "$configured" != "$hooks_dir" ]; then
        printf '%s\n' "commit hook: core.hooksPath is set to $configured, so the" >&2
        printf '%s\n' "hook installed at $hooks_dir will NOT run. Run:" >&2
        printf '%s\n' "  git config --unset core.hooksPath" >&2
      fi
    '';
  };

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

    # Disabled because prek bakes an absolute --config path naming whichever
    # worktree installed it into the shared hooks directory, so one worktree
    # governed every worktree of the clone and all of them lost the ability to
    # commit once that one was deleted (chelis#1409). The check now installs
    # from the tracked `.githooks/commit-msg` template above, which resolves
    # its repository at run time and therefore names no worktree.
    no-ai-authorship = {
      enable = false;
      name = "Reject AI authorship markers";
    };
  };
}
