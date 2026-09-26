`chelis.compile_and_load` stages the runtime archive built into the Python
extension beside the compiled library and links that archive by path, instead of
searching for the most recently modified `libchelis_runtime.a` or taking one from
`CHELIS_RUNTIME_DIR`. Setting `CHELIS_RUNTIME_DIR` is now an error for
`compile_and_load`; unset it. Compiled-artifact metadata records the SHA-256 of
that runtime and of the compiled library, and `chelis.load` refuses an artifact
whose digests are missing or differ, before opening its library. Artifacts
compiled by an earlier version must be recompiled. See
[#1354](https://github.com/Chelis-Lang/chelis/issues/1354).
