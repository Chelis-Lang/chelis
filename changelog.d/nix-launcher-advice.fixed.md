The Nix `chelisup` launcher now gives Nix upgrade advice when it refuses a newer release. It no longer sends users to the bootstrap installer, which replaces the launcher. Help and missing-operand requests need no `nix-store` and change no GC root. Failed installs remove duplicate partial roots. A launcher restoration failure names the destination file.

See [#3942](https://github.com/Chelis-Lang/chelis/issues/3942) and [#3957](https://github.com/Chelis-Lang/chelis/issues/3957).
