`CHELIS_HOME` now moves the Reef registry and the compiler caches. Without
`CHELIS_REEF_HOME`, the registry is at `$CHELIS_HOME/reef` and the caches are
at `$CHELIS_HOME/reef/.cache`. Previously a build that set only `CHELIS_HOME`
wrote them under `HOME`, and in a Nix build it failed to write the registry.
See [#3917](https://github.com/Chelis-Lang/chelis/issues/3917).

An empty `CHELIS_REEF_HOME`, `CHELIS_HOME`, or `XDG_CACHE_HOME` now counts as
unset in every reader. Previously an empty `CHELIS_REEF_HOME` put the registry
and the prepared-graph cache in the working directory. The layout without these
variables does not change. If you set `CHELIS_HOME` and keep packages in
`~/.chelis/reef`, move them to `$CHELIS_HOME/reef` or set `CHELIS_REEF_HOME`.
