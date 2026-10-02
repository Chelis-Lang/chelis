{
  ...
}:

let
  toolchain = builtins.fromTOML (builtins.readFile ./devenv-toolchain.toml);
in
{
  imports = [ ./devenv/shared/default.nix ];

  _module.args = {
    inherit toolchain;
    sharedRoot = ./.;
  };
}
