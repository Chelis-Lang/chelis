pub const TENSOR_DEEP: &str = r#"(module {}
  frag.tensor
  (export {} passthrough)
  (defsig {}
    passthrough
    (t-fn {eff: (effects {})}
      (t-tensor {} (d-lit {} 4) (t-prim {} f32))
      (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
  (def {}
    passthrough
    (fn {}
      (params {}
        (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
      (var {} x))))
"#;

pub const TENSOR_WELL_TYPED_BODY: &str = "(app {} (var {} relu) (var {} x))";

pub const TENSOR_EFFECTING_BODY: &str =
    "(app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))";

pub const TENSOR_LINEARITY_BODY: &str =
    "(let {} (bind {} y (realize {} (var {} x))) (app {} (var {} add) (var {} x) (var {} y)))";

pub const LIVE_MALFORMED_CAST_BODY: &str = "70.0(as)(f32)";
