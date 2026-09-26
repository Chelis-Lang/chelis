`chelis build` stages the runtime archive built into the `chelis` binary instead
of searching for the most recently modified `libchelis_runtime.a`, so a stale or
mutated archive left in a build directory is no longer linked by mistake. The
build report adds a `Staged runtime <path> (sha256 <digest>)` line, the output
directory gains `chelis_runtime.receipt.json`, and the printed compile commands
name the staged archive by path instead of `-L<dir> -lchelis_runtime`. Setting
`CHELIS_RUNTIME_DIR` is now an error for `chelis build`; unset it. See [#1354](https://github.com/Chelis-Lang/chelis/issues/1354).
