`grad` now differentiates a dimension-generic helper called from a generic
function, and a generic helper used at several sizes in one differentiated
function. Before, a generic `gather` helper called from a generic loss failed
backward-graph verification (`gather ... has output dims [.., Named("d61",
Some(3))], expected [.., Lit(3)]`), and a spatially generic helper whose `conv`
result fed another helper trapped at its second size (``extent `d101`: claimed
= 4, add axis 2 = 2``). Each use now gets its own dimensions, as every use of a
generic signature must; actuals that disagree within one use still trap.
`chelis eval` also now agrees with the C lane on a root whose result keeps a
named dimension a parameter binds, such as `outer(s: tensor[seq, f32]) ->
tensor[seq, f32]` called from a nullary `main`: it evaluates the root, and
traps when the body breaks the claim, where it used to refuse with `missing
symbolic dimension binding`. See
[#3462](https://github.com/Chelis-Lang/chelis/issues/3462) and
[#3415](https://github.com/Chelis-Lang/chelis/issues/3415).
