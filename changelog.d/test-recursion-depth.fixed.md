`chelis test` reaches the same recursion depth as `chelis eval --file`. Each
test was evaluated on a fixed 32 MiB worker thread, which held about a tenth of
the recursion `chelis eval` allows, so a recursive test that passed under
`chelis eval` could fail with "evaluation stack budget exhausted during a
function call". See [#3742](https://github.com/Chelis-Lang/chelis/issues/3742).
