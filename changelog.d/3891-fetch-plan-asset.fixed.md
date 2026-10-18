`chelis reef fetch-plan` rejects an unsafe `binary` lock entry. The `name` must be a
bare file name. The `asset` must be a bare file name of ASCII letters, digits, `.`,
`-`, `_`, or `~`. The diagnostic names the lock entry. Previously the command copied
an `asset` such as `../x` or `a?b` into `file` and into the download URL. See
[#3891](https://github.com/Chelis-Lang/chelis/issues/3891).
