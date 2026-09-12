Fatal tensor-helper diagnostics raised directly during a helper-summary probe
now return through the existing error channel after probe cleanup, instead of
unwinding out of `eval` or silently exiting the CLI with status 101. This changes
diagnostic transport, not insert/concat support or the existing deferral of errors
already returned by nested probes. See [#1922](https://github.com/Chelis-Lang/chelis/issues/1922).
