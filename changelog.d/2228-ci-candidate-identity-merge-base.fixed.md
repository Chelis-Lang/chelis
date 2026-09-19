CI resolves a pull request's patch base when its target branch has advanced
past the branch point. The candidate clone's backstop fetches no longer pass
`--depth`, which recorded the target snapshot in `.git/shallow` and made git
ignore the parents that commit already had; `git merge-base` then reported no
common ancestor, the candidate-identity step failed, and two required contexts
never started, leaving the pull request unmergeable. The identity script also
now distinguishes a truncated clone from parents that genuinely share no
history. See [#2228](https://github.com/Chelis-Lang/chelis/issues/2228).
