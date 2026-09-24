//! chelis#1292: `test_assert_close_tensor` is float-only and its tolerance
//! has exactly the tensor element dtype.

use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

const FLOAT_DTYPES: &[&str] = &["f16", "bf16", "f32", "f64"];
const REJECTED_DTYPES: &[&str] = &["i8", "i16", "i32", "i64", "bool"];

fn diagnostics(source: &str) -> Vec<CheckError> {
    let deep = chelis_deep::parser::parse_str(source).expect("Deep fixture");
    match check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(report) => report.errors,
    }
}

fn rendered(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

fn assert_one_active_float_error(errors: &[CheckError], dtype: &str, route: &str) {
    assert_eq!(
        errors.len(),
        1,
        "{route} at {dtype} must emit exactly one diagnostic, never an empty-error fallback or a cascade; got:\n{}",
        rendered(errors)
    );
    let error = &errors[0];
    assert!(
        matches!(error.kind, CheckErrorKind::PrecisionMismatch),
        "{route} at {dtype} must emit the active-float PrecisionMismatch; got:\n{}",
        rendered(errors)
    );
}

#[test]
fn every_active_float_dtype_and_same_dtype_tolerance_are_accepted() {
    for dtype in FLOAT_DTYPES {
        let source = format!(
            r#"(defsig {{}}
  check
  (t-fn {{}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  check
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:85..145"}}
      (var {{span: "surf:85..109"}} test_assert_close_tensor)
      (var {{span: "surf:110..116"}} actual)
      (var {{span: "surf:118..126"}} expected)
      (var {{span: "surf:128..131"}} tol)
      (lit {{span: "surf:133..144", type: (t-prim {{}} string)}} "own-width"))))
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.is_empty(),
            "{dtype} must be accepted with its own tolerance dtype; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn shared_precision_generic_wrapper_is_accepted_for_every_active_float_dtype() {
    for dtype in FLOAT_DTYPES {
        let source = format!(
            r#"(defsig {{dtype_bounds: {{p: float}}}}
  close
  (p)
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-var {{}} p)))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-var {{}} p)))
    (t-var {{}} p)
    (t-unit {{}})))

(def {{}}
  close
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:100..158"}}
      (var {{span: "surf:100..124"}} test_assert_close_tensor)
      (var {{span: "surf:125..131"}} actual)
      (var {{span: "surf:133..141"}} expected)
      (var {{span: "surf:143..146"}} tol)
      (lit {{span: "surf:148..157", type: (t-prim {{}} string)}} "generic"))))

(defsig {{}}
  check
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  check
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:255..283"}}
      (var {{span: "surf:255..260"}} close)
      (var {{span: "surf:261..267"}} actual)
      (var {{span: "surf:269..277"}} expected)
      (var {{span: "surf:279..282"}} tol))))
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.is_empty(),
            "shared-precision generic wrapper must accept {dtype}; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn function_value_aliases_preserve_the_active_float_domain() {
    for dtype in REJECTED_DTYPES {
        let fixtures = [
            (
                "top-level alias",
                format!(
                    r#"(def {{}} close_alias (var {{span: "surf:15..39"}} test_assert_close_tensor))

(defsig {{}}
  bad
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  bad
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:134..175"}}
      (var {{span: "surf:134..145"}} close_alias)
      (var {{span: "surf:146..152"}} actual)
      (var {{span: "surf:154..162"}} expected)
      (var {{span: "surf:164..167"}} tol)
      (lit {{span: "surf:169..174", type: (t-prim {{}} string)}} "top"))))
"#,
                ),
            ),
            (
                "local alias",
                format!(
                    r#"(defsig {{}}
  bad
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  bad
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (let {{span: "surf:92..182"}}
      (bind {{}}
        close_alias
        (var {{span: "surf:110..134"}} test_assert_close_tensor))
      (app {{span: "surf:137..180"}}
        (var {{span: "surf:137..148"}} close_alias)
        (var {{span: "surf:149..155"}} actual)
        (var {{span: "surf:157..165"}} expected)
        (var {{span: "surf:167..170"}} tol)
        (lit {{span: "surf:172..179", type: (t-prim {{}} string)}} "local")))))
"#,
                ),
            ),
            (
                "nested alias",
                format!(
                    r#"(def {{}} close_alias (var {{span: "surf:15..39"}} test_assert_close_tensor))

(def {{}} nested_alias (var {{span: "surf:55..66"}} close_alias))

(defsig {{}}
  bad
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  bad
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:161..206"}}
      (var {{span: "surf:161..173"}} nested_alias)
      (var {{span: "surf:174..180"}} actual)
      (var {{span: "surf:182..190"}} expected)
      (var {{span: "surf:192..195"}} tol)
      (lit {{span: "surf:197..205", type: (t-prim {{}} string)}} "nested"))))
"#,
                ),
            ),
            (
                "higher-order alias",
                format!(
                    r#"(def {{}} close_alias (var {{span: "surf:15..39"}} test_assert_close_tensor))

(def {{}}
  invoke
  (fn {{}}
    (params {{}} f actual expected tol)
    (app {{span: "surf:80..120"}}
      (var {{span: "surf:80..81"}} f)
      (var {{span: "surf:82..88"}} actual)
      (var {{span: "surf:90..98"}} expected)
      (var {{span: "surf:100..103"}} tol)
      (lit {{span: "surf:105..119", type: (t-prim {{}} string)}} "higher-order"))))

(defsig {{}}
  bad
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  bad
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:215..257"}}
      (var {{span: "surf:215..221"}} invoke)
      (var {{span: "surf:222..233"}} close_alias)
      (var {{span: "surf:235..241"}} actual)
      (var {{span: "surf:243..251"}} expected)
      (var {{span: "surf:253..256"}} tol))))
"#,
                ),
            ),
            (
                "returned alias",
                format!(
                    r#"(def {{}}
  return_close
  (fn {{}}
    (params {{}})
    (var {{span: "surf:22..46"}} test_assert_close_tensor)))

(def {{}}
  returned_alias
  (app {{span: "surf:64..78"}} (var {{span: "surf:64..76"}} return_close)))

(defsig {{}}
  bad
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  bad
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:173..222"}}
      (var {{span: "surf:173..187"}} returned_alias)
      (var {{span: "surf:188..194"}} actual)
      (var {{span: "surf:196..204"}} expected)
      (var {{span: "surf:206..209"}} tol)
      (lit {{span: "surf:211..221", type: (t-prim {{}} string)}} "returned"))))
"#,
                ),
            ),
            (
                "stored alias",
                format!(
                    r#"(def {{}}
  close_pair
  (tuple {{span: "surf:14..66"}}
    (var {{span: "surf:15..39"}} test_assert_close_tensor)
    (var {{span: "surf:41..65"}} test_assert_close_tensor)))

(def {{}}
  stored_alias
  (tuple-get {{span: "surf:82..94"}}
    (var {{span: "surf:82..92"}} close_pair)
    (lit {{type: (t-prim {{}} i32)}} 0)))

(defsig {{}}
  bad
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  bad
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:189..234"}}
      (var {{span: "surf:189..201"}} stored_alias)
      (var {{span: "surf:202..208"}} actual)
      (var {{span: "surf:210..218"}} expected)
      (var {{span: "surf:220..223"}} tol)
      (lit {{span: "surf:225..233", type: (t-prim {{}} string)}} "stored"))))
"#,
                ),
            ),
        ];

        for (route, source) in fixtures {
            assert_one_active_float_error(&diagnostics(&source), dtype, route);
        }
    }
}

#[test]
fn aliased_higher_order_calls_accept_every_float_at_rank_zero_and_multiple_ranks() {
    for dtype in FLOAT_DTYPES {
        for tensor_type in [
            format!("(t-tensor {{}} (t-prim {{}} {dtype}))"),
            format!("(t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype}))"),
            format!(
                "(t-tensor {{}} (d-lit {{}} 2) (d-lit {{}} 3) (d-lit {{}} 4) (t-prim {{}} {dtype}))"
            ),
        ] {
            let source = format!(
                r#"(def {{}} close_alias (var {{span: "surf:15..39"}} test_assert_close_tensor))

(def {{}} nested_alias (var {{span: "surf:55..66"}} close_alias))

(def {{}}
  return_close
  (fn {{}}
    (params {{}})
    (var {{span: "surf:88..100"}} nested_alias)))

(def {{}}
  returned_alias
  (app {{span: "surf:118..132"}} (var {{span: "surf:118..130"}} return_close)))

(def {{}}
  close_pair
  (tuple {{span: "surf:146..176"}}
    (var {{span: "surf:147..161"}} returned_alias)
    (var {{span: "surf:163..175"}} nested_alias)))

(def {{}}
  stored_alias
  (tuple-get {{span: "surf:192..204"}}
    (var {{span: "surf:192..202"}} close_pair)
    (lit {{type: (t-prim {{}} i32)}} 0)))

(def {{}}
  invoke
  (fn {{}}
    (params {{}} f actual expected tol)
    (app {{span: "surf:245..281"}}
      (var {{span: "surf:245..246"}} f)
      (var {{span: "surf:247..253"}} actual)
      (var {{span: "surf:255..263"}} expected)
      (var {{span: "surf:265..268"}} tol)
      (lit {{span: "surf:270..280", type: (t-prim {{}} string)}} "positive"))))

(defsig {{}}
  check
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} {tensor_type})
    (t-ref {{}} {tensor_type})
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  check
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:380..423"}}
      (var {{span: "surf:380..386"}} invoke)
      (var {{span: "surf:387..399"}} stored_alias)
      (var {{span: "surf:401..407"}} actual)
      (var {{span: "surf:409..417"}} expected)
      (var {{span: "surf:419..422"}} tol))))
"#,
            );
            let errors = diagnostics(&source);
            assert!(
                errors.is_empty(),
                "aliased higher-order call must accept {tensor_type}; got:\n{}",
                rendered(&errors)
            );
        }
    }
}

#[test]
fn aliases_still_reject_mixed_precision_and_rank() {
    for (label, source) in [
        (
            "mixed precision",
            r#"(def {} close_alias (var {span: "surf:15..39"} test_assert_close_tensor))

(defsig {}
  bad
  (t-fn {eff: (effects {} test)}
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f32)))
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f64)))
    (t-prim {} f32)
    (t-unit {})))

(def {}
  bad
  (fn {}
    (params {}
      (actual {type: (t-var {} _)})
      (expected {type: (t-var {} _)})
      (tol {type: (t-var {} _)}))
    (app {span: "surf:133..186"}
      (var {span: "surf:133..144"} close_alias)
      (var {span: "surf:145..151"} actual)
      (var {span: "surf:153..161"} expected)
      (var {span: "surf:163..166"} tol)
      (lit {span: "surf:168..185", type: (t-prim {} string)} "mixed precision"))))
"#,
        ),
        (
            "mixed rank",
            r#"(def {} close_alias (var {span: "surf:15..39"} test_assert_close_tensor))

(defsig {}
  bad
  (t-fn {eff: (effects {} test)}
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f32)))
    (t-ref {} (t-tensor {} (d-lit {} 2) (d-lit {} 1) (t-prim {} f32)))
    (t-prim {} f32)
    (t-unit {})))

(def {}
  bad
  (fn {}
    (params {}
      (actual {type: (t-var {} _)})
      (expected {type: (t-var {} _)})
      (tol {type: (t-var {} _)}))
    (app {span: "surf:136..184"}
      (var {span: "surf:136..147"} close_alias)
      (var {span: "surf:148..154"} actual)
      (var {span: "surf:156..164"} expected)
      (var {span: "surf:166..169"} tol)
      (lit {span: "surf:171..183", type: (t-prim {} string)} "mixed rank"))))
"#,
        ),
    ] {
        let errors = diagnostics(source);
        assert!(
            errors.len() == 1
                && matches!(
                    errors[0].kind,
                    CheckErrorKind::PrecisionMismatch | CheckErrorKind::DimensionMismatch
                ),
            "{label} through an alias must produce one structural mismatch; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn arbitrary_tensor_ranks_are_accepted_with_one_shared_shape() {
    for (tensor_type, tolerance_type) in [
        ("(t-tensor {} (t-prim {} f32))", "f32"),
        (
            "(t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))",
            "f32",
        ),
        (
            "(t-tensor {} (d-lit {} 2) (d-lit {} 3) (d-lit {} 4) (t-prim {} f64))",
            "f64",
        ),
    ] {
        let source = format!(
            r#"(defsig {{}}
  check
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} {tensor_type})
    (t-ref {{}} {tensor_type})
    (t-prim {{}} {tolerance_type})
    (t-unit {{}})))

(def {{}}
  check
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:99..154"}}
      (var {{span: "surf:99..123"}} test_assert_close_tensor)
      (var {{span: "surf:124..130"}} actual)
      (var {{span: "surf:132..140"}} expected)
      (var {{span: "surf:142..145"}} tol)
      (lit {{span: "surf:147..153", type: (t-prim {{}} string)}} "rank"))))
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors.is_empty(),
            "{tensor_type} must satisfy the arbitrary-rank signature; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn independently_polymorphic_tolerance_cannot_bypass_same_dtype_constraint() {
    for (tensor_dtype, tolerance_dtype) in [
        ("f16", "bf16"),
        ("bf16", "f32"),
        ("f32", "f64"),
        ("f64", "f32"),
    ] {
        let source = format!(
            r#"(defsig {{}}
  unsafe_close
  (p q)
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-var {{}} p)))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-var {{}} p)))
    (t-var {{}} q)
    (t-unit {{}})))

(def {{}}
  unsafe_close
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:111..169"}}
      (var {{span: "surf:111..135"}} test_assert_close_tensor)
      (var {{span: "surf:136..142"}} actual)
      (var {{span: "surf:144..152"}} expected)
      (var {{span: "surf:154..157"}} tol)
      (lit {{span: "surf:159..168", type: (t-prim {{}} string)}} "generic"))))

(defsig {{}}
  bad
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {tensor_dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {tensor_dtype})))
    (t-prim {{}} {tolerance_dtype})
    (t-unit {{}})))

(def {{}}
  bad
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:272..307"}}
      (var {{span: "surf:272..284"}} unsafe_close)
      (var {{span: "surf:285..291"}} actual)
      (var {{span: "surf:293..301"}} expected)
      (var {{span: "surf:303..306"}} tol))))
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            !errors.is_empty(),
            "an independently-polymorphic {tolerance_dtype} tolerance must not bypass the {tensor_dtype} tensor constraint"
        );
    }
}

#[test]
fn shared_precision_generic_wrapper_rejects_non_float_instantiations() {
    for dtype in REJECTED_DTYPES {
        let source = format!(
            r#"(defsig {{dtype_bounds: {{p: float}}}}
  close
  (p)
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-var {{}} p)))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-var {{}} p)))
    (t-var {{}} p)
    (t-unit {{}})))

(def {{}}
  close
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:100..158"}}
      (var {{span: "surf:100..124"}} test_assert_close_tensor)
      (var {{span: "surf:125..131"}} actual)
      (var {{span: "surf:133..141"}} expected)
      (var {{span: "surf:143..146"}} tol)
      (lit {{span: "surf:148..157", type: (t-prim {{}} string)}} "generic"))))

(defsig {{}}
  bad
  (t-fn {{eff: (effects {{}} test)}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  bad
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:253..281"}}
      (var {{span: "surf:253..258"}} close)
      (var {{span: "surf:259..265"}} actual)
      (var {{span: "surf:267..275"}} expected)
      (var {{span: "surf:277..280"}} tol))))
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors
                .iter()
                .any(|error| { matches!(error.kind, CheckErrorKind::PrecisionMismatch) }),
            "generic {dtype} instantiation must receive the float-domain diagnostic; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn every_non_float_tensor_dtype_is_rejected_at_check_time() {
    for dtype in REJECTED_DTYPES {
        let source = format!(
            r#"(defsig {{}}
  check
  (t-fn {{}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {dtype})))
    (t-unit {{}})))

(def {{}}
  check
  (fn {{}}
    (params {{}} (actual {{type: (t-var {{}} _)}}) (expected {{type: (t-var {{}} _)}}))
    (app {{span: "surf:75..147"}}
      (var {{span: "surf:75..99"}} test_assert_close_tensor)
      (var {{span: "surf:100..106"}} actual)
      (var {{span: "surf:108..116"}} expected)
      (cast {{span: "surf:118..132"}}
        (lit {{span: "surf:123..126", surf_literal_style: "unsuffixed", type: (t-prim {{}} f32)}} 0.0)
        (t-prim {{}} f32))
      (lit {{span: "surf:134..146", type: (t-prim {{}} string)}} "float-only"))))
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors
                .iter()
                .any(|error| { matches!(error.kind, CheckErrorKind::PrecisionMismatch) }),
            "{dtype} must receive the float-domain diagnostic; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn tolerance_dtype_must_equal_the_tensor_dtype() {
    let errors = diagnostics(
        r#"(defsig {}
  check
  (t-fn {}
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f64)))
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f64)))
    (t-prim {} f32)
    (t-unit {})))

(def {}
  check
  (fn {}
    (params {}
      (actual {type: (t-var {} _)})
      (expected {type: (t-var {} _)})
      (tol {type: (t-var {} _)}))
    (app {span: "surf:85..146"}
      (var {span: "surf:85..109"} test_assert_close_tensor)
      (var {span: "surf:110..116"} actual)
      (var {span: "surf:118..126"} expected)
      (var {span: "surf:128..131"} tol)
      (lit {span: "surf:133..145", type: (t-prim {} string)} "same dtype"))))
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| { matches!(error.kind, CheckErrorKind::PrecisionMismatch) }),
        "a mixed f64/f32 application must reject as a precision mismatch; got:\n{}",
        rendered(&errors)
    );
}

#[test]
fn non_float_tolerance_is_rejected_at_check_time() {
    for dtype in ["i32", "bool"] {
        let source = format!(
            r#"(defsig {{}}
  check
  (t-fn {{}}
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} f32)))
    (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} f32)))
    (t-prim {{}} {dtype})
    (t-unit {{}})))

(def {{}}
  check
  (fn {{}}
    (params {{}}
      (actual {{type: (t-var {{}} _)}})
      (expected {{type: (t-var {{}} _)}})
      (tol {{type: (t-var {{}} _)}}))
    (app {{span: "surf:85..151"}}
      (var {{span: "surf:85..109"}} test_assert_close_tensor)
      (var {{span: "surf:110..116"}} actual)
      (var {{span: "surf:118..126"}} expected)
      (var {{span: "surf:128..131"}} tol)
      (lit {{span: "surf:133..150", type: (t-prim {{}} string)}} "float tolerance"))))
"#,
        );
        let errors = diagnostics(&source);
        assert!(
            errors
                .iter()
                .any(|error| { matches!(error.kind, CheckErrorKind::PrecisionMismatch) }),
            "{dtype} tolerance must receive the float-domain diagnostic; got:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn tensor_shapes_and_dtypes_must_match() {
    for source in [
        r#"(defsig {}
  bad
  (t-fn {}
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f32)))
    (t-ref {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)))
    (t-unit {})))

(def {}
  bad
  (fn {}
    (params {} (actual {type: (t-var {} _)}) (expected {type: (t-var {} _)}))
    (app {span: "surf:73..140"}
      (var {span: "surf:73..97"} test_assert_close_tensor)
      (var {span: "surf:98..104"} actual)
      (var {span: "surf:106..114"} expected)
      (cast {span: "surf:116..130"}
        (lit {span: "surf:121..124", surf_literal_style: "unsuffixed", type: (t-prim {} f32)} 0.0)
        (t-prim {} f32))
      (lit {span: "surf:132..139", type: (t-prim {} string)} "shape"))))
"#,
        r#"(defsig {}
  bad
  (t-fn {}
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f32)))
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f64)))
    (t-unit {})))

(def {}
  bad
  (fn {}
    (params {} (actual {type: (t-var {} _)}) (expected {type: (t-var {} _)}))
    (app {span: "surf:73..140"}
      (var {span: "surf:73..97"} test_assert_close_tensor)
      (var {span: "surf:98..104"} actual)
      (var {span: "surf:106..114"} expected)
      (cast {span: "surf:116..130"}
        (lit {span: "surf:121..124", surf_literal_style: "unsuffixed", type: (t-prim {} f32)} 0.0)
        (t-prim {} f32))
      (lit {span: "surf:132..139", type: (t-prim {} string)} "dtype"))))
"#,
        r#"(defsig {}
  bad
  (t-fn {}
    (t-ref {} (t-tensor {} (d-lit {} 2) (t-prim {} f32)))
    (t-ref {} (t-tensor {} (d-lit {} 2) (d-lit {} 1) (t-prim {} f32)))
    (t-unit {})))

(def {}
  bad
  (fn {}
    (params {} (actual {type: (t-var {} _)}) (expected {type: (t-var {} _)}))
    (app {span: "surf:76..142"}
      (var {span: "surf:76..100"} test_assert_close_tensor)
      (var {span: "surf:101..107"} actual)
      (var {span: "surf:109..117"} expected)
      (cast {span: "surf:119..133"}
        (lit {span: "surf:124..127", surf_literal_style: "unsuffixed", type: (t-prim {} f32)} 0.0)
        (t-prim {} f32))
      (lit {span: "surf:135..141", type: (t-prim {} string)} "rank"))))
"#,
    ] {
        let errors = diagnostics(source);
        assert!(
            !errors.is_empty(),
            "mismatched tensors must be rejected before evaluation"
        );
    }
}
