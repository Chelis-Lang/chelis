The compiled ownership safety oracle now follows the consuming
`string_concat` contract: a uniquely owned left operand moves into the result,
while the right operand and each surviving result owner are released exactly
once. The regression tests still pin the original alias-retain guarantees and
now also reject a fallback to the cloning concat. See
[#2269](https://github.com/Chelis-Lang/chelis/issues/2269).
