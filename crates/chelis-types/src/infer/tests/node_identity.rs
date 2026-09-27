//! Owner stamps and the deferred call ledger are keyed by node identity.
//!
//! The identity is assigned when a checker entry numbers its copy of the
//! program, so it follows the node rather than its content or its address: two
//! structurally equal subtrees are different owners, and the `Node::to_list`
//! view inference dispatches on names the nodes it copies.

use super::*;

/// `(tag {} children...)` in the requested carrier, with every span zero so
/// equal subtrees are equal values.
fn form(stamped: bool, tag: DeepTag, children: Vec<deep::Expr>) -> deep::Expr {
    if stamped {
        deep::Expr::node(tag, deep::Metadata::default(), children, Span::new(0, 0))
    } else {
        node_expr(tag, children)
    }
}

fn int(value: i64) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Int(value), Span::new(0, 0))
}

fn float(value: f64) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Float(value), Span::new(0, 0))
}

/// `x = 1; ({x}, (x = 2.5; {x}))`: the two `(block {} (var {} x))` reads are
/// equal values of different types (a `block` carries its value's type stamp;
/// a bare `var` carries none).
fn shadowed_program(stamped: bool) -> Vec<deep::Expr> {
    let f = |tag, children| form(stamped, tag, children);
    let read_x = f(
        DeepTag::Block,
        vec![f(DeepTag::Var, vec![symbol_expr("x")])],
    );
    let inner = f(
        DeepTag::Let,
        vec![
            f(
                DeepTag::Bind,
                vec![symbol_expr("x"), f(DeepTag::Lit, vec![float(2.5)])],
            ),
            read_x.clone(),
        ],
    );
    let body = f(
        DeepTag::Let,
        vec![
            f(
                DeepTag::Bind,
                vec![symbol_expr("x"), f(DeepTag::Lit, vec![int(1)])],
            ),
            f(DeepTag::Tuple, vec![read_x, inner]),
        ],
    );
    vec![f(DeepTag::Def, vec![symbol_expr("main"), body])]
}

/// The annotated type of every `block` and `lit` node, in preorder.
fn stamped_types(exprs: &[deep::Expr]) -> Vec<(DeepTag, String)> {
    fn walk(expr: &deep::Expr, out: &mut Vec<(DeepTag, String)>) {
        let deep::ExprCarrier::DecodedNode(tag, meta, children) = expr.carrier() else {
            return;
        };
        if matches!(tag, DeepTag::Block | DeepTag::Lit) {
            let ty = meta
                .ty()
                .map(|ty| {
                    chelis_deep::printer::print_canonical_flat(std::slice::from_ref(
                        ty.expression(),
                    ))
                })
                .unwrap_or_default();
            out.push((tag, ty));
        }
        for child in children {
            walk(child, out);
        }
    }
    let mut out = Vec::new();
    for expr in exprs {
        walk(expr, &mut out);
    }
    out
}

/// REGRESSION TEST. Equal subtrees are distinct owners: each read of `x` carries
/// the type of the binding it reads, in both carriers, and the stamped
/// carrier (whose children inference reaches only through the `to_list`
/// view) is annotated exactly like the list carrier.
#[test]
fn equal_subtrees_keep_their_own_owner_stamps_in_both_carriers() {
    let listed = crate::check_ir_program(
        &shadowed_program(false),
        crate::ProgramOrigin::Unlinked,
        &mut crate::NativeServices::new(None),
    )
    .unwrap_or_else(|report| panic!("list carrier: {:?}", report.errors));
    let stamped = crate::check_typed_program(
        &shadowed_program(true),
        crate::ProgramOrigin::Unlinked,
        &mut crate::NativeServices::new(None),
    )
    .unwrap_or_else(|report| panic!("stamped carrier: {:?}", report.errors));

    let types = stamped_types(listed.annotated_exprs());
    let [
        (DeepTag::Lit, outer_lit),
        (DeepTag::Block, outer_var),
        (DeepTag::Lit, inner_lit),
        (DeepTag::Block, inner_var),
    ] = types.as_slice()
    else {
        panic!("unexpected annotated nodes {types:?}");
    };
    assert_ne!(
        outer_lit, inner_lit,
        "the two bindings have different types"
    );
    assert_eq!(
        outer_var, outer_lit,
        "the outer `x` reads the outer binding"
    );
    assert_eq!(
        inner_var, inner_lit,
        "the inner `x` reads the shadowing binding"
    );
    assert_eq!(
        stamped_types(stamped.annotated_exprs()),
        types,
        "the stamped carrier is annotated like the list carrier"
    );
}

fn check_messages(source: &str, stamped: bool) -> Vec<String> {
    let exprs = chelis_deep::parse_and_stamp_file(source).expect("Deep fixture");
    let result = if stamped {
        crate::check_typed_program(
            &exprs,
            crate::ProgramOrigin::Unlinked,
            &mut crate::NativeServices::new(None),
        )
    } else {
        crate::check_ir_program(
            &exprs,
            crate::ProgramOrigin::Unlinked,
            &mut crate::NativeServices::new(None),
        )
    };
    match result {
        Ok(_) => Vec::new(),
        Err(report) => report.errors.iter().map(|e| e.message.clone()).collect(),
    }
}

/// Two sibling calls that both suspend on an unresolved operand: `len(t)` and
/// `len(u)` are the arguments of one `add`.
const SIBLING_SUSPENDED_CALLS: &str = r#"
(defsig {} f (t-fn {} (t-adt {} List (t-prim {} i32)) SECOND (t-prim {} i64)))
(def {} f (fn {} (params {} (a {type: (t-var {} _)}) (b {type: (t-var {} _)}))
  (let {} (bind {} g (fn {} (params {} t u)
      (app {} (var {} add) (app {} (var {} len) (var {} t)) (app {} (var {} len) (var {} u)))))
    (app {} (var {} g) (var {} a) (var {} b)))))
"#;

/// REGRESSION TEST. Keyed by the address of the `to_list` view, which lives
/// in the inference frame, the two sibling calls of a stamped program shared
/// one ledger key: the second call's suspension was dropped as a duplicate of
/// the first, its own route never replayed, and `len(u)` on an `i32` was
/// rejected only by a later, different check. Keyed by node identity each call
/// keeps its own entry and both carriers report the route's own diagnostic.
#[test]
fn sibling_suspended_calls_in_a_stamped_program_keep_distinct_keys() {
    let lists = SIBLING_SUSPENDED_CALLS.replace("SECOND", "(t-adt {} List (t-prim {} i32))");
    let _ = take_post_app_key_log();
    assert_eq!(check_messages(&lists, true), Vec::<String>::new());
    let keys = take_post_app_key_log();
    let eager: Vec<_> = keys
        .iter()
        .filter(|(_, during_replay)| !*during_replay)
        .map(|(key, _)| *key)
        .collect();
    assert_eq!(eager.len(), 2, "one suspension per call; log {keys:?}");
    assert!(
        eager[0].is_some() && eager[0] != eager[1],
        "the sibling calls must keep distinct keys; log {keys:?}"
    );

    let scalar = SIBLING_SUSPENDED_CALLS.replace("SECOND", "(t-prim {} i32)");
    let stamped = check_messages(&scalar, true);
    assert!(
        stamped
            .iter()
            .any(|message| message == "len expects List or Dict input, got i32"),
        "the second call's own route must reject its operand; got {stamped:?}"
    );
    assert_eq!(
        stamped,
        check_messages(&scalar, false),
        "both carriers decide the program the same way"
    );
}
