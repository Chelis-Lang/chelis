Added a validated Clarabel adapter and an optional evaluator path for checked
`Clarabel.Qp.solve` calls over dense `f64` tensors. The package returns distinct
solved and stopped cases for non-SDP cones. Compiled C calls and the exact
optimizer proof contract remain unavailable. See
[#3310](https://github.com/Chelis-Lang/chelis/pull/3310).
