Add `is_nan`, `is_finite`, and `is_infinite` for scalar and tensor `f16`, `bf16`, `f32`, and `f64` values in the evaluator and C backend. Classification reads stored bits without conversion and preserves tensor dimensions through direct calls, callable aliases, and borrowed callbacks, including rank-zero tensors. HIP and Metal report a typed unsupported-feature diagnostic.

WireDag requires schema 29, binary Shell metadata requires format 8, and Reef package schemas require format 5. Rebuild older package metadata and update graph consumers.
