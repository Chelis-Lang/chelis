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

fn tag(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    match list.elements.first() {
        Some(Expr::Atom(chelis_deep::Atom::Symbol(tag), _)) => Some(tag),
        _ => None,
    }
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
