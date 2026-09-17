`grad` now preserves checked function metadata for captured callable values, so
typed Jacobian wrappers can differentiate a target that closes over a model
function on both evaluator and generated-C lanes. See [#676](https://github.com/Chelis-Lang/chelis/issues/676).
