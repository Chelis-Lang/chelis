`pad` accepts a fill computed at run time in `chelis build --target c`, and
`grad` differentiates through it on the evaluator and C lanes. Previously the C
build refused a runtime fill even without `grad`, and `grad` refused it even
when only the padded tensor was differentiated. The fill is a scalar of exactly
the tensor's dtype. Its cotangent sums the output cotangent over the padded
cells, with every cell holding an input element counted as exact +0, so a
learned fill now trains. Integer and bool fills stay forward-only. The C lane
does not yet build the fill's gradient when the padded tensor has a runtime
extent ([#3412](https://github.com/Chelis-Lang/chelis/issues/3412)), and the
`chelis eval --file` forward path still carries integer fills through f64 and
refuses bool fills ([#3510](https://github.com/Chelis-Lang/chelis/issues/3510)).
The C build of `grad` with respect to a rank-2 operand with runtime extents
stops at an emission panic that a literal fill reaches too
([#3511](https://github.com/Chelis-Lang/chelis/issues/3511)).

`grad` over a tensor with a parameter's runtime extent no longer fails in the
evaluator with "bool storage cannot enter a numeric IR kernel" when a Bool
`where` mask is padded after the operand. See
[#3389](https://github.com/Chelis-Lang/chelis/issues/3389).
