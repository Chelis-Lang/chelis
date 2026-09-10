`chelis check`'s per-file path can no longer fail without producing a report.
The functions that check one file return the report directly instead of a
`Result`, so a failure has nowhere to propagate to and must be carried as a
diagnostic. The one visible consequence: `chelis check --show-inferred` on a
program whose inferred signatures cannot be built -- for example one with a
negative dimension extent -- now emits a report whose `errors` array names
the problem and exits `2`, where it previously exited `1` with nothing on
stdout. A measured report that falls outside its own numeric domain is
likewise reported rather than propagated. See
[#886](https://github.com/Chelis-Lang/chelis/issues/886).
