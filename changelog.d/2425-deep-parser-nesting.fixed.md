`chelis prove` no longer aborts with a stack overflow on a deeply nested Deep
program. It now runs on the same grown stack segment as `chelis check`, so a
program `check` accepts also proves; previously a 2,000-deep application chain
that `check` accepted aborted `prove` with exit 134. The Deep parser runs on a
stack segment of its own and rejects input nested deeper than that segment
holds with a located diagnostic, ``Deep input nests deeper than the parser
supports at byte N``, identically in every command, where both `check` and
`prove` previously aborted. See
[#2425](https://github.com/Chelis-Lang/chelis/issues/2425).
