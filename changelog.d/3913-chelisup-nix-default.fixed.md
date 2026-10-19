The Nix `chelisup` package keeps its launcher at `$CHELIS_HOME/bin/chelisup`
after `chelisup default`. Before this change, `default` copied the real
installer over the launcher, and a later `chelisup link` of a Nix store
toolchain added no garbage-collector root. The launcher now keeps the
garbage-collector root of the package around `default` as it does for
`install`, so `default` also stops when no `nix-store` is found. Through the
launcher, `chelisup update` and `chelisup self update` name the `nix run` and
`nix profile install` commands of a newer release, and they no longer advise
the bootstrap installer. See
[#3913](https://github.com/Chelis-Lang/chelis/issues/3913).
