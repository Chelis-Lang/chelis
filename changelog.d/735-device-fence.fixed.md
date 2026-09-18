`chelis build --target c` now rejects accelerator, unknown, empty, and malformed
`with device(...)` selectors with `BuildTargetMismatch` before writing artifacts.
Host programs remain accepted without a resource region or with `cpu` and
well-formed `cpu:<label>` selectors. Previously selectors such as `cuda:0` and
`metal` were silently erased and compiled for host execution. See
[#735](https://github.com/Chelis-Lang/chelis/issues/735).
