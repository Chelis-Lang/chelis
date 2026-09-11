`chelis check`'s per-file path can no longer fail without producing a report.
The functions that check one file return the report directly instead of a
`Result`, so a failure has nowhere to propagate to and must be carried as a
diagnostic. The visible change is `chelis check --show-inferred` on a program
whose inferred signatures cannot be built -- for example one with a negative
dimension extent, itself a checker defect tracked by
[#1768](https://github.com/Chelis-Lang/chelis/issues/1768). On a single file
it previously exited `1` with nothing on stdout; it now emits a report whose
`errors` array names the problem, prints the same message to stderr, and
exits `2`. In directory mode such a file previously produced
`{"file":...,"error":"..."}` in place of a report and the run ended with
`one or more files failed to check` and exit `1`; it now contributes a report
like any other file and the run exits `2`. A measured report that falls
outside its own numeric domain is likewise reported rather than propagated.
See [#886](https://github.com/Chelis-Lang/chelis/issues/886).
