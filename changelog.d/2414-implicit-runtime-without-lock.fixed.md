A `Std.*` import in a Reef package now resolves on the first run of every
command, whether or not the package has a `reef.lock`. Previously the bundled
`chelis-std` runtime entered the package graph only through the lockfile, so
in a package from `chelis reef init` the first `chelis check`, `build`, `test`
or `prove` reported ``unresolved import `Std.Io.Json` `` and wrote the lock,
`chelis eval --file` failed on every run, and `chelis reef build` failed even
with a lock present. Manifest resolution and lockfile reconstruction now both
add the runtime as an implicit dependency. See
[#2414](https://github.com/Chelis-Lang/chelis/issues/2414).
