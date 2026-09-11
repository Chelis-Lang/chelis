The runtime-extent class oracle demands `RUNTIME EXTENT ORACLE: PASS` from
phase B rather than reporting rows short of exit, and `--phase final` now
reaches its row report instead of refusing on the withdrawn phase C. Every
recorded row of both phases is at an exit state, the nightly extended-validation
workflow runs that one completion command, and the acceptance runner over the
extent-claim fixtures runs as an ordinary test. See
[#1277](https://github.com/Chelis-Lang/chelis/issues/1277).

The example census in `issue_1739_diagonal_runtime_bound` derives its corpus
from the examples directory instead of a hand-written count, and records the
two examples the C capability gate refuses under
[#1192](https://github.com/Chelis-Lang/chelis/issues/1192). The shipped-example
roster stays in `parity.rs`, where the Phase 3 faithful-observation oracle
freezes its definition. See
[#1787](https://github.com/Chelis-Lang/chelis/issues/1787).
