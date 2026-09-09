`chelis check` now reports a failure that occurs before the checker runs
through the same JSON report as every other failure, instead of writing a
message to stderr and leaving stdout empty. An unreadable or non-UTF-8 file,
a style-gate rejection and a preparation failure all emit a report whose
`errors` array names the problem, so a consumer can no longer confuse "no
diagnostics" with "the diagnostics were not transported".

**These paths now exit `2` rather than `1`**, matching every other non-empty
error list, and directory mode exits `2` rather than `1` when a file in the
walk fails for one of these reasons. Exit `1` no longer means "the tool could
not run": the exit status is a function of the `errors` array alone. The
directory walk failing to enumerate a directory at all still exits `1` with
no output, which chelis#1678 owns.

The human message is still written to stderr on every one of these paths, and
is now also written on three that previously failed silently: an unreadable
`.dp`, a missing file, and a read failure inside a reef package. The
directory-mode summary line `one or more files failed to check` is gone,
replaced by the per-file diagnostics now carried in each entry's report. See
[#886](https://github.com/Chelis-Lang/chelis/issues/886).
