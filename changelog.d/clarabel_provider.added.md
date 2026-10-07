Added a validated Clarabel adapter and optional evaluator and compiled-C paths
for checked `Clarabel.Qp.solve` calls over dense `f64` tensors. The package
returns distinct solved and stopped cases for non-SDP cones. An opt-in ideal
QP property contract proves fixed literal zero/nonnegative-cone claims
conditional on a source-bound exact-real optimizer axiom. See
[#3310](https://github.com/Chelis-Lang/chelis/pull/3310).
