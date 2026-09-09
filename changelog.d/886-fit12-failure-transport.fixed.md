`chelis check` now reports a failure that occurs before the checker runs
through the same JSON report as every other failure, instead of writing a
message to stderr and leaving stdout empty. An unreadable or non-UTF-8 file,
a style-gate rejection, a preparation failure and a layered-check failure all
emit a report whose `errors` array names the problem, so a consumer can no
longer confuse "no diagnostics" with "the diagnostics were not transported".
These paths now exit `2` rather than `1`, matching every other non-empty
error list and converging the Surf arm on the Deep arm, which already
reported an unreadable file this way. The human-readable message is still
written to stderr. See
[#886](https://github.com/Chelis-Lang/chelis/issues/886).
