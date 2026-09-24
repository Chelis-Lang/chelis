//! Post-Phase checker-totality contracts for deferred shape checks (#780)
//! and rigid declaration dimensions (#847).

use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn diagnostics(source: &str) -> Vec<CheckError> {
    let deep = chelis_deep::parser::parse_str(source).expect("Deep fixture");
    match check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("{:?}: {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn unannotated_lambda_replays_matmul_shape_check_after_first_use() {
    let errors = diagnostics(
        r#"(defsig {}
  driver
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 9) (d-lit {} 9) (t-prim {} f32))))

(def {}
  driver
  (fn {}
    (params {} (good {type: (t-var {} _)}))
    (let {span: "surf:60..105"}
      (bind {}
        f
        (fn {span: "surf:68..93"}
          (params {} a)
          (app {span: "surf:78..93"}
            (var {span: "surf:78..84"} matmul)
            (var {span: "surf:85..86"} a)
            (var {span: "surf:88..92"} good))))
      (app {span: "surf:96..103"}
        (var {span: "surf:96..97"} f)
        (var {span: "surf:98..102"} good)))))
"#,
    );
    assert!(
        errors.iter().any(|error| {
            error
                .message
                .contains("def 'driver' body doesn't match declared signature")
                && error.message.contains("tensor[4, 4, f32]")
                && error.message.contains("tensor[9, 9, f32]")
        }),
        "the ordinary matmul rule must replay to a resolved 4x4 result, not merely give up; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn deferred_shape_lambda_is_monomorphic_after_first_use() {
    let errors = diagnostics(
        r#"(defsig {}
  driver
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 5) (d-lit {} 6) (t-prim {} f32))
    (t-tensor {} (d-lit {} 6) (d-lit {} 7) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 4) (t-prim {} f32))))

(def {}
  driver
  (fn {}
    (params {}
      (a {type: (t-var {} _)})
      (b {type: (t-var {} _)})
      (c {type: (t-var {} _)})
      (d {type: (t-var {} _)}))
    (let {span: "surf:123..203"}
      (bind {}
        f
        (fn {span: "surf:131..156"}
          (params {} x y)
          (app {span: "surf:144..156"}
            (var {span: "surf:144..150"} matmul)
            (var {span: "surf:151..152"} x)
            (var {span: "surf:154..155"} y))))
      (let {}
        (bind {}
          first
          (app {span: "surf:167..174"}
            (var {span: "surf:167..168"} f)
            (var {span: "surf:169..170"} a)
            (var {span: "surf:172..173"} b)))
        (let {}
          (bind {}
            second
            (app {span: "surf:186..193"}
              (var {span: "surf:186..187"} f)
              (var {span: "surf:188..189"} c)
              (var {span: "surf:191..192"} d)))
          (var {span: "surf:196..201"} first))))))
"#,
    );
    assert!(
        errors.iter().any(|error| {
            error.message.contains("dimension mismatch")
                || error.message.contains("tensor dimension mismatch")
                || error.message.contains("cannot unify")
        }),
        "the second application must conflict with the first concrete instantiation; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn unconstrained_identity_lambda_remains_polymorphic() {
    let errors = diagnostics(
        r#"(defsig {} keep_poly (t-fn {} (t-prim {} i32) (t-prim {} string)))

(def {}
  keep_poly
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (let {span: "surf:35..108"}
      (bind {}
        identity
        (fn {span: "surf:50..69"}
          (params {} value)
          (var {span: "surf:64..69"} value)))
      (let {}
        (bind {}
          one
          (app {span: "surf:78..89"}
            (var {span: "surf:78..86"} identity)
            (var {span: "surf:87..88"} x)))
        (app {span: "surf:92..106"}
          (var {span: "surf:92..100"} identity)
          (lit {span: "surf:101..105", type: (t-prim {} string)} "ok"))))))
"#,
    );
    assert!(
        errors.is_empty(),
        "an unconstrained identity has no deferred semantic obligation: {}",
        summary(&errors)
    );
}

#[test]
fn unresolved_shape_obligation_rejects_at_declaration_boundary() {
    let errors = diagnostics(
        r#"(defsig {}
  unresolved
  (t-fn {} (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32)) (t-var {} _)))

(def {}
  unresolved
  (fn {}
    (params {} (good {type: (t-var {} _)}))
    (fn {span: "surf:43..68"}
      (params {} a)
      (app {span: "surf:53..68"}
        (var {span: "surf:53..59"} matmul)
        (var {span: "surf:60..61"} a)
        (var {span: "surf:63..67"} good)))))
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(
                error.kind,
                chelis_types::errors::CheckErrorKind::TypeMismatch
            )
        }),
        "an unbound shape obligation must reject at the declaration boundary; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn synthesized_defsig_hole_does_not_disable_shape_obligation() {
    let errors = diagnostics(
        r#"(defsig {}
  helper
  (t-fn {} (t-var {} _) (t-tensor {} (d-lit {} 9) (d-lit {} 8) (t-prim {} f32))))

(def {}
  helper
  (fn {}
    (params {} a)
    (app {span: "surf:38..50"}
      (var {span: "surf:38..44"} matmul)
      (var {span: "surf:45..46"} a)
      (var {span: "surf:48..49"} a))))
"#,
    );
    assert!(
        errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::TypeMismatch
        )),
        "a return annotation's synthesized `(t-var _ )` parameter slot is still an unknown outer constructor; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn authored_bare_type_hole_does_not_disable_shape_obligation() {
    let errors = diagnostics(
        r#"(defsig {}
  helper
  (t)
  (t-fn {} (t-var {} t) (t-tensor {} (d-lit {} 9) (d-lit {} 8) (t-prim {} f32))))

(def {}
  helper
  (fn {}
    (params {} a)
    (app {span: "surf:55..67"}
      (var {span: "surf:55..61"} matmul)
      (var {span: "surf:62..63"} a)
      (var {span: "surf:65..66"} a))))
"#,
    );
    assert!(
        errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::TypeMismatch
        )),
        "an authored bare type variable still leaves the operand constructor unknown; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn derived_projection_keeps_unannotated_parameter_ownership() {
    let errors = diagnostics(
        r#"(defsig {}
  driver
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 9) (d-lit {} 9) (t-prim {} f32))))

(def {}
  driver
  (fn {}
    (params {} (good {type: (t-var {} _)}))
    (let {span: "surf:60..121"}
      (bind {}
        f
        (fn {span: "surf:68..101"}
          (params {} pair)
          (app {span: "surf:81..101"}
            (var {span: "surf:81..87"} matmul)
            (tuple-get {span: "surf:88..94"}
              (var {span: "surf:88..92"} pair)
              (lit {type: (t-prim {} i32)} 0))
            (var {span: "surf:96..100"} good))))
      (app {span: "surf:104..119"}
        (var {span: "surf:104..105"} f)
        (tuple {span: "surf:106..118"}
          (var {span: "surf:107..111"} good)
          (var {span: "surf:113..117"} good))))))
"#,
    );
    assert!(
        errors.iter().any(|error| {
            error
                .message
                .contains("def 'driver' body doesn't match declared signature")
                && error.message.contains("tensor[4, 4, f32]")
                && error.message.contains("tensor[9, 9, f32]")
        }),
        "a projection-derived type variable must retain its lambda-parameter ownership; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn top_level_shape_helper_must_resolve_at_its_own_boundary() {
    let errors = diagnostics(
        r#"(def {}
  helper
  (fn {}
    (params {} a)
    (app {span: "surf:17..29"}
      (var {span: "surf:17..23"} matmul)
      (var {span: "surf:24..25"} a)
      (var {span: "surf:27..28"} a))))

(defsig {}
  d1
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))))

(def {}
  d1
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:82..91"}
      (var {span: "surf:82..88"} helper)
      (var {span: "surf:89..90"} x))))

(defsig {}
  d2
  (t-fn {}
    (t-tensor {} (d-lit {} 7) (d-lit {} 7) (t-prim {} f32))
    (t-tensor {} (d-lit {} 7) (d-lit {} 7) (t-prim {} f32))))

(def {}
  d2
  (fn {}
    (params {} (y {type: (t-var {} _)}))
    (app {span: "surf:144..153"}
      (var {span: "surf:144..150"} helper)
      (var {span: "surf:151..152"} y))))
"#,
    );
    assert!(
        errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::TypeMismatch
        )),
        "later declarations are not binding sites for an earlier top-level declaration; got:\n{}",
        summary(&errors)
    );
}

#[test]
fn unresolved_non_lambda_shape_value_keeps_the_existing_wildcard_contract() {
    let errors = diagnostics(
        r#"(defsig {}
  manual_matmul
  (t-fn {}
    (t-tensor {} (d-lit {} 8) (d-lit {} 16) (t-prim {} f32))
    (t-tensor {} (d-lit {} 16) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 8) (d-lit {} 4) (t-prim {} f32))))

(def {}
  manual_matmul
  (fn {}
    (params {} (a {type: (t-var {} _)}) (b {type: (t-var {} _)}))
    (let {span: "surf:94..171"}
      (bind {}
        ae
        (app {span: "surf:103..121"}
          (var {span: "surf:103..109"} insert)
          (var {span: "surf:110..111"} a)
          (lit {span: "surf:113..114", type: (t-prim {} i32)} 2)
          (lit {span: "surf:116..120", type: (t-prim {} i64)} 4)))
      (let {}
        (bind {}
          be
          (app {span: "surf:129..147"}
            (var {span: "surf:129..135"} insert)
            (var {span: "surf:136..137"} b)
            (lit {span: "surf:139..140", type: (t-prim {} i32)} 0)
            (lit {span: "surf:142..146", type: (t-prim {} i64)} 8)))
        (app {span: "surf:150..169"}
          (var {span: "surf:150..153"} sum)
          (app {span: "surf:154..165"}
            (var {span: "surf:154..157"} mul)
            (var {span: "surf:158..160"} ae)
            (var {span: "surf:162..164"} be))
          (lit {span: "surf:167..168", type: (t-prim {} i32)} 1))))))
"#,
    );
    assert!(
        errors.is_empty(),
        "[04-INF-1] defers unannotated lambda parameters, not an ordinary let-bound wildcard: {}",
        summary(&errors)
    );
}

#[test]
fn unannotated_scatter_elements_replays_dtype_checks_after_first_use() {
    let errors = diagnostics(
        r#"(defsig {}
  driver
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))))

(def {}
  driver
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (let {span: "surf:124..224"}
      (bind {}
        scatter_at
        (fn {span: "surf:141..185"}
          (params {} x i u)
          (app {span: "surf:157..185"}
            (var {span: "surf:157..173"} scatter_elements)
            (var {span: "surf:174..175"} x)
            (var {span: "surf:177..178"} i)
            (var {span: "surf:180..181"} u)
            (lit {span: "surf:183..184", type: (t-prim {} i32)} 1))))
      (app {span: "surf:188..222"}
        (var {span: "surf:188..198"} scatter_at)
        (var {span: "surf:199..203"} data)
        (var {span: "surf:205..212"} indices)
        (var {span: "surf:214..221"} updates)))))
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("i32") || error.message.contains("i64")),
        "deferred scatter_elements must reject floating-point indices: {}",
        summary(&errors)
    );
}

#[test]
fn consistent_deferred_scatter_elements_is_accepted() {
    let errors = diagnostics(
        r#"(defsig {}
  driver
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} i32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))))

(def {}
  driver
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (let {span: "surf:124..224"}
      (bind {}
        scatter_at
        (fn {span: "surf:141..185"}
          (params {} x i u)
          (app {span: "surf:157..185"}
            (var {span: "surf:157..173"} scatter_elements)
            (var {span: "surf:174..175"} x)
            (var {span: "surf:177..178"} i)
            (var {span: "surf:180..181"} u)
            (lit {span: "surf:183..184", type: (t-prim {} i32)} 1))))
      (app {span: "surf:188..222"}
        (var {span: "surf:188..198"} scatter_at)
        (var {span: "surf:199..203"} data)
        (var {span: "surf:205..212"} indices)
        (var {span: "surf:214..221"} updates)))))
"#,
    );
    assert!(
        errors.is_empty(),
        "valid deferred scatter_elements rejected: {}",
        summary(&errors)
    );
}

#[test]
fn generic_jacobian_wrapper_keeps_n_and_m_independent() {
    let errors = diagnostics(
        r#"(defsig {}
  jacobian_row
  (n m)
  (t-fn {}
    (t-fn {}
      (t-ref {} (t-tensor {} (d-var {} n) (t-prim {} f32)))
      (t-ref {} (t-tensor {} (d-var {} m) (t-prim {} f32)))
      (t-tensor {} (d-var {} m) (t-prim {} f32)))
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} m) (t-prim {} f32))
    (t-tensor {} (d-var {} m) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  jacobian_row
  (fn {}
    (params {}
      (model {type: (t-var {} _)})
      (theta {type: (t-var {} _)})
      (x_data {type: (t-var {} _)})
      (output_seed {type: (t-var {} _)}))
    (let {span: "surf:191..490"}
      (bind {}
        target
        (fn {span: "surf:204..428"}
          (params {}
            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
            (x_local {type: (t-tensor {} (d-var {} m) (t-prim {} f32))})
            (seed_local {type: (t-tensor {} (d-var {} m) (t-prim {} f32))}))
          (let {span: "surf:309..428"}
            (bind {}
              prediction
              (app {span: "surf:328..355"}
                (var {span: "surf:328..333"} model)
                (var {span: "surf:334..345"} theta_local)
                (var {span: "surf:347..354"} x_local)))
            (app {span: "surf:360..424"}
              (var {span: "surf:360..376"} tensor_to_scalar)
              (app {span: "surf:377..423"}
                (var {span: "surf:377..380"} sum)
                (app {span: "surf:381..408"}
                  (var {span: "surf:381..384"} mul)
                  (var {span: "surf:385..395"} prediction)
                  (var {span: "surf:397..407"} seed_local))
                (cast {span: "surf:410..422"}
                  (lit {span: "surf:415..416", surf_literal_style: "unsuffixed", type: (t-prim {} i32)} 0)
                  (t-prim {} i32)))))))
      (app {span: "surf:431..488"}
        (grad {span: "surf:431..460", wrt: (var {} theta_local)}
          (var {span: "surf:436..442"} target)
          (lit {type: (t-prim {} i32)} 0))
        (var {span: "surf:461..466"} theta)
        (var {span: "surf:468..474"} x_data)
        (var {span: "surf:476..487"} output_seed)))))
"#,
    );
    assert!(
        errors.is_empty(),
        "independently declared rigid dimensions must not collapse: {}",
        summary(&errors)
    );
}

#[test]
fn a_body_that_really_requires_equal_rigid_dims_still_rejects() {
    let errors = diagnostics(
        r#"(defsig {}
  wrong
  (n m)
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} m) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  wrong
  (fn {}
    (params {} (x {type: (t-var {} _)}) (y {type: (t-var {} _)}))
    (var {span: "surf:75..76"} y)))
"#,
    );
    assert!(
        errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::DimensionMismatch
                | chelis_types::errors::CheckErrorKind::TypeMismatch
        )),
        "a genuine n-versus-m contract violation must stay rejected"
    );
}
