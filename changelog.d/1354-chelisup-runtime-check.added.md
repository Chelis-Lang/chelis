`chelisup install` now checks a release's runtime files before placing it. It
runs the unpacked `chelis runtime export` and refuses the release unless the
exported archive and each receipt-listed header are regular non-symlink files
whose SHA-256 matches the receipt, and the shipped archive and headers match
those same digests. A refused release leaves the store untouched. Releases up
to 0.18.11 predate the export and install unchecked, with a warning. A refused
release newer than the running chelisup also says how to get the latest
chelisup, and a release whose `chelis` the operating system refuses to execute
is reported as such. See [#1354](https://github.com/Chelis-Lang/chelis/issues/1354).
