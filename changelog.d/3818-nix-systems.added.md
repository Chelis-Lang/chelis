The Nix flake supports `aarch64-linux` in addition to `x86_64-linux` and
`aarch64-darwin`. `checks.aarch64-linux.lane-check` produces a native gate
verdict on arm64 Linux, and a native CI job on `ubuntu-24.04-arm` runs the
complete check set. `x86_64-darwin` is not supported, because nixpkgs removes
that system for 26.11 and Devenv publishes no build for it. See
[#3818](https://github.com/Chelis-Lang/chelis/issues/3818).
