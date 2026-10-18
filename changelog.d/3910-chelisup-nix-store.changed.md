The Nix `chelisup` package no longer brings its own Nix. Its launcher adds
garbage-collector roots with the `nix-store` of the Nix installation of the
user. It looks on `PATH` first and then in the standard Nix profile
directories. Thus the `chelisup` closure no longer holds Nix 2.34.7 and its
libraries. If the launcher finds no `nix-store`, `chelisup install` and
`chelisup link` of a store path stop with a message before they change a root.
See [#3910](https://github.com/Chelis-Lang/chelis/issues/3910).
