On `aarch64-linux`, where no release toolchain exists, `chelisup install` now
names the Nix route: `chelisup link` of the `chelis` package of the Chelis
flake, for example
`chelisup link "$(nix build --no-link --print-out-paths github:Chelis-Lang/chelis/vX.Y.Z#chelis)"`.
On NixOS without nix-ld, a dynamic release build cannot start. The error then
names `programs.nix-ld.enable = true` and the same `chelisup link` route, and
no longer names Alpine. chelisup takes the file `/etc/NIXOS` as the NixOS
signal. See [#3915](https://github.com/Chelis-Lang/chelis/issues/3915).
