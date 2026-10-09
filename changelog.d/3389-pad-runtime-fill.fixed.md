`pad` accepts a fill computed at run time on every lane, and `grad` differentiates
through it. Previously `chelis build --target c` refused a runtime fill even
without `grad`, and `grad` refused it even when only the padded tensor was
differentiated. The fill is a scalar of exactly the tensor's dtype. Its
cotangent sums the output cotangent over the padded cells, with every cell
holding an input element counted as exact +0, so a learned fill now trains.
Integer and bool fills stay forward-only. See
[#3389](https://github.com/Chelis-Lang/chelis/issues/3389).
