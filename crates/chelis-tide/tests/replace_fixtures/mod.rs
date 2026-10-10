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

/// `debug` performs `IO`, which the declared-pure `passthrough` forbids. (A
/// random draw is no effect: it takes an explicit key.)
pub const TENSOR_EFFECTING_BODY: &str = "(app {} (var {} debug) (var {} x))";

pub const TENSOR_COPY_REPAIRED_BODY: &str =
    "(let {} (bind {} y (realize {} (var {} x))) (app {} (var {} add) (var {} x) (var {} y)))";

/// `drop` is terminal, so a later use cannot be repaired by inserting `copy`.
pub const TENSOR_LINEARITY_BODY: &str =
    "(let {} (bind {} gone (app {} (var {} drop) (var {} x))) (var {} x))";

pub const LIVE_MALFORMED_CAST_BODY: &str = "70.0(as)(f32)";

pub const ADD_TENSOR_IDENTITY: &str = r#"(defsig {}
  added_passthrough
  (t-fn {eff: (effects {})}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
(def {}
  added_passthrough
  (fn {}
    (params {}
      (y {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
    (var {} y)))
"#;

/// A declared-pure function whose body performs `IO` through `debug`.
pub const ADD_TENSOR_EFFECTING: &str = r#"(defsig {}
  added_noisy
  (t-fn {eff: (effects {})}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
(def {}
  added_noisy
  (fn {}
    (params {}
      (y {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
    (app {} (var {} debug) (var {} y))))
"#;

pub const ADD_TENSOR_COPY_REPAIRED: &str = r#"(defsig {}
  added_alias_twice
  (t-fn {eff: (effects {})}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
(def {}
  added_alias_twice
  (fn {}
    (params {}
      (y {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
    (let {}
      (bind {} z (realize {} (var {} y)))
      (app {} (var {} add) (var {} y) (var {} z)))))
"#;

pub const ADD_TENSOR_LINEARITY: &str = r#"(defsig {}
  added_alias_twice
  (t-fn {eff: (effects {})}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
(def {}
  added_alias_twice
  (fn {}
    (params {}
      (y {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
    (let {}
      (bind {} gone (app {} (var {} drop) (var {} y)))
      (var {} y))))
"#;

pub const ADD_DECL_SHAPE_ERROR: &str = "(export {} added_passthrough)";
