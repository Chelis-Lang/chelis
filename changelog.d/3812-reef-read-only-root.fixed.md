`chelis build`, `check`, `cost`, and `prove` accept a package with no
`reef.lock` in a root that the process cannot write. A source in a read-only Nix
store is an example. The commands resolve the dependency graph in memory and
write no `reef.lock`, `.reef-write.lock`, or `.gitignore` into that root.
Previously they stopped with `Permission denied` on the project write lock. A
writable root still receives the repaired `reef.lock`. See
[#3812](https://github.com/Chelis-Lang/chelis/issues/3812).
