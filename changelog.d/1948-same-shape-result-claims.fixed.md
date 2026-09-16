Declared result extent guards on same-shape operations now validate every
distinct positive-rank operand path first, then attribute any result mismatch
to the primitive that produced the returned value. Operand ordering and
rank-zero inputs no longer select or erase the claim. See
[#1948](https://github.com/Chelis-Lang/chelis/issues/1948).
