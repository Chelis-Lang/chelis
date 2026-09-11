The runtime-extent class oracle demands `RUNTIME EXTENT ORACLE: PASS` from
phase B rather than reporting rows short of exit, and `--phase final` now
reaches its row report instead of refusing on the withdrawn phase C. Every
recorded row of both phases is at an exit state, and the acceptance runner over
the extent-claim fixtures runs as an ordinary test. See
[#1277](https://github.com/Chelis-Lang/chelis/issues/1277).
