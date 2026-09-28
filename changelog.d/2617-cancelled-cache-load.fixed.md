A `--timeout` or other cancellation that fires while a compiler cache is being
loaded is now reported as the cancellation alone. Previously the chelis-std and
dependency typecheck caches and the compiled-context cache classified the
abandoned load as a decode failure, so `chelis eval --timeout` could print
`... unusable (cache decode error: ...: chelis::eval::cancelled); rebuilding
and overwriting` before the timeout message. A valid cache file is left in
place; a file that genuinely fails to decode still warns and is rebuilt. See
[#2617](https://github.com/Chelis-Lang/chelis/issues/2617).
