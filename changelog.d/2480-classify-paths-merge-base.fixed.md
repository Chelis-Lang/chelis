`scripts/gate.py --fast` no longer fails a branch that is behind `main` on
paths the branch never touched. Its changed-path classification diffed the
tip of `origin/main` against the branch, which charged the branch with every
path `main` changed after the fork. It now diffs from the merge base, as
the gate's crate selection already did, and it judges a deleted file's
retirement against that same commit. A branch with no history in common with
`origin/main` now fails the check instead of being diffed tree against tree,
and a missing `origin/main` fails with a message that says to fetch it. See
[#2480](https://github.com/Chelis-Lang/chelis/issues/2480).
