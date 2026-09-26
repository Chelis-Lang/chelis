A development build of `chelis` or of the Python extension no longer stages a
runtime older than its checkout. The runtime records a SHA-256 of each of its
declared source inputs when it is compiled, and `chelis build`,
`chelis runtime export` and `compile_and_load` compare that record with the
checkout before other work, failing with the changed, removed and added files
and a rebuild remedy. A sealed build reads no checkout; the Python extension
gains the `sealed-runtime` feature the CLI already has. See
[#1354](https://github.com/Chelis-Lang/chelis/issues/1354).
