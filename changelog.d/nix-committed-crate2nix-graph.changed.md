The Nix flake imports a committed `Cargo.nix` and no longer needs import from
derivation. One host can now evaluate the packages of both supported systems
without a builder for the other system. A change to `Cargo.lock`, a workspace
manifest, or a crate target must also regenerate `Cargo.nix`. The new
`Cargo.nix freshness` CI job checks that file. See
[#3611](https://github.com/Chelis-Lang/chelis/issues/3611).
