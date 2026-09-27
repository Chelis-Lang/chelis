A Reef package now links only the standard-library modules that its own
modules, its dependencies, and the entries a command runs import, directly or
through other standard-library modules. Previously every package that had the
bundled `chelis-std` runtime in its graph linked, type-checked, cached and
decoded the whole standard library, so `chelis eval --file`, `chelis test`,
`chelis check` and `chelis build` in a package that imports nothing from it
paid for all of it. A loose `chelis eval --file` snippet, a `chelis test` of
any directory, and a Python `eval(..., project_root=...)` source still import
any standard-library module; the compiled context is cached separately for
each set of linked modules. See
[#2558](https://github.com/Chelis-Lang/chelis/issues/2558) and
[#2672](https://github.com/Chelis-Lang/chelis/issues/2672).
