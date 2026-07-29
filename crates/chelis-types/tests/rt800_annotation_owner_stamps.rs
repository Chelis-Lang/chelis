//! RT-800: annotation consumes the authoritative result of the owning
//! inference traversal. It must not re-infer the same node in a fresh scope.

use std::sync::Arc;

use chelis_deep::Expr;
use chelis_types::{
    build_type_env_from_library, check_ir_program, check_ir_with_context, check_typed_program,
};

fn surf(source: &str) -> Vec<Expr> {
    let parsed = chelis_surf::parser::parse_str(source).expect("Surf source should parse");
    chelis_surf::desugar::desugar_program(&parsed)
}

fn expanded_surf(source: &str) -> Vec<Expr> {
    chelis_macros::expand_program(&surf(source), &chelis_macros::ExpansionOptions::default())
        .expect("Surf owner fixture should expand")
        .into_exprs()
}

fn tag(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    // Decode-once: the spelling comes from the decoded tag, never a raw
    // element-0 string.
    list.tag().map(|tag| tag.as_str())
}

fn carries_type_stamp(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    let Some(Expr::Map(meta, _)) = list.elements.get(1) else {
        return false;
    };
    meta.entries.iter().any(|(key, _)| key == "type")
}

fn collect_tag_stamp_state(expr: &Expr, wanted: &str, out: &mut Vec<bool>) {
    match expr {
        Expr::Atom(_, _) => {}
        Expr::List(list, _) => {
            if tag(expr) == Some(wanted) {
                out.push(carries_type_stamp(expr));
            }
            for child in &list.elements {
                collect_tag_stamp_state(child, wanted, out);
            }
        }
        Expr::Map(meta, _) => {
            for (_, value) in &meta.entries {
                collect_tag_stamp_state(value, wanted, out);
            }
        }
        Expr::MetaExpr(meta, _) => {
            collect_tag_stamp_state(&meta.expr, wanted, out);
            for (_, value) in &meta.entries {
                collect_tag_stamp_state(value, wanted, out);
            }
        }
    }
}

fn tuple_fold_program() -> Vec<Expr> {
    surf(
        r#"
def f[n](xs: tensor[n, f32]) -> (tensor[n, f32], int64) = {
  idxs = range(cast(0, int64), numel(copy(xs)))
  state0 = (to_tensor(map(fn (x: f32) -> cast(0.0, f32), to_list(copy(xs)))), cast(0, int64))
  step = fn (state, i) -> {
    acc = state.0
    total = state.1
    (acc, add(total, i))
  }
  fold(step, state0, idxs)
}

out = f(to_tensor([1.0, 2.0, 3.0]))
"#,
    )
}

#[test]
fn polymorphic_tuple_fold_has_one_owner_across_both_public_checks() {
    let program = tuple_fold_program();
    check_typed_program(&program).expect("typed annotation must consume owner inference");
    check_ir_program(&program).expect("IR annotation must consume owner inference");
}

#[test]
fn sequential_let_and_generic_closure_keep_owner_stamps() {
    let checked = check_ir_program(&surf(
        r#"
def choose[a](flag: bool, x: a) -> a = match flag with {
  | true => { id = fn (y) -> y; id(x) }
  | false => x
}

def twice[a](x: a) -> a = {
  id = fn (y) -> y
  first = id(x)
  second = id(first)
  second
}

answer = twice(choose(true, cast(7, int64)))
"#,
    ))
    .expect("let/match/closure annotation must consume the owning types");

    for wanted in ["fn", "app"] {
        let mut stamps = Vec::new();
        for expr in checked.annotated_exprs() {
            collect_tag_stamp_state(expr, wanted, &mut stamps);
        }
        assert!(!stamps.is_empty(), "fixture must contain `{wanted}` nodes");
        assert!(
            stamps.iter().all(|stamped| *stamped),
            "every `{wanted}` owner must retain an authoritative type stamp: {stamps:?}"
        );
    }
}

#[test]
fn owner_stamps_are_isolated_across_context_and_parallel_sessions() {
    let library = surf("def id[a](x: a) -> a = x");
    let context = Arc::new(
        build_type_env_from_library(&library).expect("generic library context should build"),
    );
    let snippets = ["left = id(cast(1, int64))", "right = id(cast(2.0, f32))"];

    let handles: Vec<_> = snippets
        .into_iter()
        .map(|source| {
            let context = Arc::clone(&context);
            let program = surf(source);
            std::thread::spawn(move || {
                check_ir_with_context(&context, &program)
                    .expect("parallel annotation session must remain isolated")
            })
        })
        .collect();

    for handle in handles {
        let checked = handle.join().expect("checker thread should not panic");
        let mut app_stamps = Vec::new();
        for expr in checked.annotated_exprs() {
            collect_tag_stamp_state(expr, "app", &mut app_stamps);
        }
        assert_eq!(app_stamps, vec![true]);
    }
}

#[test]
fn structural_child_roles_do_not_require_runtime_owner_stamps() {
    let fixtures = [
        (
            "grad wrt selector",
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))
             (def {} g (grad {} (var {} f) (lit {type: (t-prim {} int32)} 0)))",
        ),
        (
            "vmap axis selector",
            "(defsig {} f (t-fn {}
                (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (vmap {} (var {} f) (lit {type: (t-prim {} int32)} 0)))",
        ),
        (
            "tuple and record selectors with cast type syntax",
            "(deftype {} Box ()
                (variant {} Box (field {} value (t-prim {} int64))))
             (def {} pair (tuple {}
                (cast {} (lit {type: (t-prim {} int32)} 1) (t-prim {} int64))
                (record {} Box (kv {} value
                    (cast {} (lit {type: (t-prim {} int32)} 2) (t-prim {} int64))))))
             (def {} selected (tuple-get {} (var {} pair)
                (lit {type: (t-prim {} int32)} 1)))
             (def {} answer (access {} (var {} selected) value))",
        ),
        (
            "let binders and match patterns",
            "(def {} answer
                (let {} (bind {} x (lit {type: (t-prim {} int32)} 1))
                    (match {} (var {} x)
                        (arm {} (pat-lit {} 1) ()
                            (lit {type: (t-prim {} int32)} 2))
                        (arm {} (pat-as {} y (pat-wild {})) ()
                            (var {} y)))))",
        ),
    ];

    for (label, source) in fixtures {
        let program = chelis_deep::parser::parse_str(source).expect("valid Deep fixture");
        check_ir_program(&program).unwrap_or_else(|result| {
            panic!(
                "{label} must classify structural children outside runtime ownership: {result:?}"
            )
        });
    }
}

#[test]
fn concat_reuses_nested_runtime_child_types_without_reinference() {
    let valid = surf(
        r#"
def combine(a: tensor[2, f32], b: tensor[3, f32]) -> tensor[5, f32] = {
  pair = (a, b)
  concat([pair.0, pair.1], 0)
}
"#,
    );
    check_ir_program(&valid).expect("nested tuple/list children retain canonical runtime types");

    let malformed = surf(
        r#"
def bad(a: tensor[2, f32]) = {
  pair = (a, cast(1.0, f32))
  concat([pair.0, pair.1], 0)
}
"#,
    );
    let result = check_ir_program(&malformed).expect_err("mixed tensor/scalar list must reject");
    assert!(
        result.errors.iter().any(|error| {
            matches!(
                error.kind,
                chelis_types::errors::CheckErrorKind::TypeMismatch
                    | chelis_types::errors::CheckErrorKind::DimensionMismatch
            )
        }),
        "malformed nested concat needs an owning type error: {:?}",
        result.errors
    );
    assert!(
        result
            .errors
            .iter()
            .all(|error| !error.message.contains("owner-stamp invariant")),
        "malformed parity must not fail through an internal owner lookup: {:?}",
        result.errors
    );
}

#[test]
fn module_helper_dependency_orders_keep_body_owner_stamps() {
    for source in [
        r#"
module Forward
def caller(a, b: tensor[4, f32]) = helper(a, b)
def helper(x, y: tensor[4, f32]) = add(x, y)
"#,
        r#"
module Backward
def helper(x, y: tensor[4, f32]) = add(x, y)
def caller(a, b: tensor[4, f32]) = helper(a, b)
"#,
    ] {
        let checked = check_ir_program(&surf(source)).expect("module dependency order checks");
        let mut stamps = Vec::new();
        for expr in checked.annotated_exprs() {
            collect_tag_stamp_state(expr, "app", &mut stamps);
        }
        assert!(!stamps.is_empty());
        assert!(
            stamps.iter().all(|stamped| *stamped),
            "every module body app retains its primary owner stamp: {stamps:?}"
        );
    }
}

#[test]
fn raw_top_level_def_and_cast_keep_primary_owner_stamps() {
    let program = chelis_deep::parser::parse_str(
        r#"(def {} answer
              (cast {}
                (lit {type: (t-prim {} int32)} 7)
                (t-prim {} int64)))"#,
    )
    .expect("raw Deep owner fixture must parse");
    let checked = check_ir_program(&program)
        .unwrap_or_else(|result| panic!("raw def/cast owners must check: {:?}", result.errors));

    for wanted in ["def", "cast", "lit"] {
        let mut stamps = Vec::new();
        for expr in checked.annotated_exprs() {
            collect_tag_stamp_state(expr, wanted, &mut stamps);
        }
        assert_eq!(
            stamps,
            vec![true],
            "the primary driver must stamp the original `{wanted}` owner"
        );
    }
}

#[test]
fn guarded_match_arm_records_the_guard_owner_before_annotation() {
    let program = chelis_deep::parser::parse_str(
        r#"(def {} choose
              (match {}
                (lit {type: (t-prim {} bool)} true)
                (arm {}
                  (pat-lit {} true)
                  (lit {type: (t-prim {} bool)} true)
                  (lit {type: (t-prim {} int32)} 1))
                (arm {}
                  (pat-wild {})
                  ()
                  (lit {type: (t-prim {} int32)} 2))))"#,
    )
    .expect("guarded-arm owner fixture must parse");
    let checked = check_ir_program(&program)
        .unwrap_or_else(|result| panic!("guarded arm must check: {:?}", result.errors));

    let mut lit_stamps = Vec::new();
    for expr in checked.annotated_exprs() {
        collect_tag_stamp_state(expr, "lit", &mut lit_stamps);
    }
    assert_eq!(
        lit_stamps,
        vec![true, true, true, true],
        "scrutinee, guard, and both bodies need authoritative stamps"
    );

    let invalid = chelis_deep::parser::parse_str(
        r#"(def {} choose
              (match {}
                (lit {type: (t-prim {} bool)} true)
                (arm {}
                  (pat-wild {})
                  (lit {type: (t-prim {} int32)} 1)
                  (lit {type: (t-prim {} int32)} 2))))"#,
    )
    .expect("non-boolean guard fixture must parse");
    let errors = check_ir_program(&invalid)
        .expect_err("a non-boolean match guard must reject")
        .errors;
    assert_eq!(
        errors.len(),
        1,
        "guard mismatch must report once: {errors:?}"
    );
    assert!(
        matches!(
            errors[0].kind,
            chelis_types::errors::CheckErrorKind::TypeMismatch
        ) && errors[0].message.contains("guard")
            && !errors[0].message.contains("owner-stamp invariant"),
        "the guard owner must emit the user-facing mismatch: {errors:?}"
    );
}

#[test]
fn nominal_header_scope_reaches_generated_metadata_finalization() {
    let checked = check_ir_program(&expanded_surf(
        r#"
module Repro.ParamsOwner

type Params =
  | Params { w: tensor[2, f32] }

def loss(p: Params, x: tensor[2, f32], y: tensor[2, f32]) -> f32 = {
  match p with {
    | Params { w: w } => {
      d = sub(mul(&x, &w), y)
      sum(mul(&d, &d), cast(0, int32)) |> tensor_to_scalar
    }
  }
}
x = to_tensor([cast(2.0, f32), cast(3.0, f32)])
y = to_tensor([cast(1.0, f32), cast(1.0, f32)])
out = grad(fn (p: Params) -> loss(p, x, y))(
  Params { w: to_tensor([cast(0.5, f32), cast(-1.0, f32)]) }
)
"#,
    ))
    .unwrap_or_else(|result| {
        panic!(
            "precollected nominal headers must reach generated metadata: {:?}",
            result.errors
        )
    });
    assert!(checked.signature_inference().functions.contains_key("loss"));
}
