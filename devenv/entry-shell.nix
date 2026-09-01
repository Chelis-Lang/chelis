{ config, lib, ... }:

{
  # Supported Devenv 2.2.x CLIs write this shared file through a
  # remove/write/chmod sequence. Concurrent shell entries can observe it
  # between those operations. Publish one complete payload by same-directory
  # atomic rename instead.
  tasks."devenv:enterShell".exec = lib.mkForce ''
    ${config.languages.python.package}/bin/python \
      ${config.devenv.root}/scripts/write_devenv_load_exports.py \
      "$DEVENV_DOTFILE/load-exports"
  '';
}
