`chelis check`'s per-file path can no longer fail without producing a report.
The functions that check one file return the report directly instead of a
`Result`, so a failure has nowhere to propagate to and must be carried as a
diagnostic. Building the inferred-signature rows and validating the measured
report were the last two paths that could exit without one; both now report
the problem instead. `chelis check <dir>` consequently emits a report for
every file it walks, where a file that failed to be read or formatted
previously produced a bare message in the report's place. See
[#886](https://github.com/Chelis-Lang/chelis/issues/886).
