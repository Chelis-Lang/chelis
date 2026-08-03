{ ... }:

{
  imports = [
    ./devenv/toolchains.nix
    ./devenv/commands.nix
    ./devenv/generated-files.nix
    ./devenv/git-hooks.nix
    ./devenv/smoke-tests.nix
  ];
}
