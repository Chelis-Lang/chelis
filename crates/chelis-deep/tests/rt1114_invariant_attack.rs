//! Permanent adversarial net for the chelis#1109 construction-scan skip
//! (filed out of the PR #1114 red team).
//!
//! `Node::validate` stops its raw-vocabulary scan at an already-stamped
//! `Expr::Node` child. That is sound only because of an inductive
//! invariant: a `Node` exists only if it cleared this same scan, so no
//! raw closed-vocabulary tag can sit anywhere beneath one. The invariant
//! rests on the construction surface staying closed -- `try_new`, `new`,
//! and `Deserialize` all routing through `validate`, the fields staying
//! private to the `node` module, every mutator revalidating the whole
//! candidate, and no mutable child borrow escaping.
//!
//! One test locking that is a thin thread. Every test here drives a
//! DIFFERENT public entry point with a smuggled raw tag and asserts the
//! same thing: the gate REJECTS it. If a future change opens a hole -- a
//! new mutable accessor, a mutator that forgets to revalidate, a
//! constructor that skips `validate` -- one of these turns red, and the
//! skip in `find_raw_vocabulary_tag_below_gate` stops being sound the
//! moment they do.
//!
//! Note what CANNOT be written here: no test constructs a `Node` that
//! actually carries a raw tag beneath it. That impossibility is the
//! invariant, so every attack below is either rejected at the outer gate
//! or rejected earlier, when the inner node the attack needed refused to
//! be built.

use serde_json::Value;

use chelis_deep::node::{Node, NodeError};
use chelis_deep::parser::parse_str;
use chelis_deep::path::{ResolveError, splice_function_body, spliced_function_def};
use chelis_deep::span::Span;
use chelis_deep::tag::DeepTag;
use chelis_deep::{Atom, Expr, MetaExpr, Metadata, UnknownFormData};

fn sp() -> Span {
    Span::new(0, 0)
}

/// A pre-decode-once node: a closed-vocabulary tag surviving as a raw head
/// string instead of a stamped `Node`. The stamper builds a `Node` for every
/// decodable head, so only a hand-built `UnknownForm` carries one. This is
/// the payload every attack tries to smuggle past the gate.
fn raw(tag: &str) -> Expr {
    Expr::UnknownForm(Box::new(UnknownFormData {
        head: tag.to_string(),
        meta: Metadata::default(),
        children: vec![Expr::Atom(Atom::Int(0), sp())],
        span: sp(),
    }))
}

/// A clean stamped node, for positions an attack needs to look legitimate.
fn stamped_lit() -> Expr {
    Expr::node(
        DeepTag::Lit,
        Metadata::default(),
        vec![Expr::Atom(Atom::Int(1), sp())],
        sp(),
    )
}

fn meta_with(_key: &str, value: Expr) -> Metadata {
    let mut metadata = Metadata::default();
    metadata
        .insert(chelis_deep::annotations::MetadataValue::PropertySeed(
            chelis_deep::annotations::RuntimeExpression::try_new(value).unwrap(),
        ))
        .expect("valid extension fixture");
    metadata
}

/// Assert the construction gate rejects `payload` in child position and
/// names the raw tag it found.
fn assert_child_rejected(payload: Expr, expected_tag: &str) {
    let result = Node::try_new(DeepTag::App, Metadata::default(), vec![payload]);
    match result {
        Err(NodeError::RawVocabularyTag { ref raw_tag, .. }) if raw_tag == expected_tag => {}
        other => panic!("expected a RawVocabularyTag(`{expected_tag}`) rejection, got {other:?}"),
    }
}

/// Assert the construction gate rejects `payload` in metadata position.
fn assert_metadata_rejected(payload: Expr, expected_tag: &str) {
    let error = chelis_deep::annotations::RuntimeExpression::try_new(payload).unwrap_err();
    assert!(error.to_string().contains(expected_tag), "{error}");
}

// ── Leg 1: the Deserialize entry point ───────────────────────────────
//
// Serde is the one surface that can EXPRESS the forbidden shape: JSON can
// describe a stamped Node carrying a raw tag, which no Rust caller can
// build. The custom `Deserialize` routes through `try_new`, so the inner
// node refuses to decode and the whole tree fails -- which is exactly how
// the induction survives a wire boundary.

/// `&mut` into the `children` array of a serialized `Expr::Node`.
fn node_children_mut(value: &mut Value) -> &mut Vec<Value> {
    value
        .get_mut("Node")
        .and_then(|node| node.get_mut(0))
        .and_then(|payload| payload.get_mut("children"))
        .and_then(Value::as_array_mut)
        .expect("serialized Expr::Node is {\"Node\":[{tag,meta,children},span]}")
}

/// `&mut` into the `meta.entries` array of a serialized `Expr::Node`.
fn node_meta_entries_mut(value: &mut Value) -> &mut Vec<Value> {
    value
        .get_mut("Node")
        .and_then(|node| node.get_mut(0))
        .and_then(|payload| payload.get_mut("meta"))
        .and_then(|meta| meta.get_mut("entries"))
        .and_then(Value::as_array_mut)
        .expect("serialized Expr::Node is {\"Node\":[{tag,meta,children},span]}")
}

fn json_of(expr: &Expr) -> Value {
    serde_json::to_value(expr).expect("Expr serializes")
}

/// A stamped `(var {} x)` node, whose single child slot the attacks
/// overwrite with the raw payload.
fn stamped_var_json() -> Value {
    json_of(&Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("x".to_string()), sp())],
        sp(),
    ))
}

fn assert_deserialize_rejected(value: Value, expected_tag: &str) {
    let text = serde_json::to_string(&value).expect("attack payload serializes");
    let error = serde_json::from_str::<Expr>(&text)
        .expect_err("a raw vocabulary tag must not decode into a stamped Node");
    let message = error.to_string();
    assert!(
        message.contains(&format!("raw closed-vocabulary tag `{expected_tag}`")),
        "deserialization must fail with the raw-vocabulary diagnostic, got: {message}"
    );
}

#[test]
fn deserialize_rejects_a_raw_tag_directly_below_a_stamped_child() {
    let mut inner = stamped_var_json();
    node_children_mut(&mut inner)[0] = json_of(&raw("lit"));

    let mut outer = json_of(&Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![stamped_lit()],
        sp(),
    ));
    node_children_mut(&mut outer)[0] = inner;

    assert_deserialize_rejected(outer, "lit");
}

#[test]
fn deserialize_rejects_a_raw_tag_two_stamped_levels_down() {
    let mut bottom = stamped_var_json();
    node_children_mut(&mut bottom)[0] = json_of(&raw("app"));

    let mut middle = json_of(&Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![stamped_lit()],
        sp(),
    ));
    node_children_mut(&mut middle)[0] = bottom;

    let mut top = json_of(&Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![stamped_lit()],
        sp(),
    ));
    node_children_mut(&mut top)[0] = middle;

    assert_deserialize_rejected(top, "app");
}

#[test]
fn deserialize_rejects_a_raw_tag_in_a_stamped_child_metadata() {
    let mut inner = stamped_var_json();
    node_meta_entries_mut(&mut inner).push(Value::Array(vec![
        Value::String("property_seed".to_string()),
        json_of(&raw("let")),
    ]));

    let mut outer = json_of(&Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![stamped_lit()],
        sp(),
    ));
    node_children_mut(&mut outer)[0] = inner;

    assert_deserialize_rejected(outer, "let");
}

#[test]
fn deserialize_rejects_a_raw_tag_in_the_outermost_node_itself() {
    // The control for the three above: when the raw tag is NOT below a
    // stamped child, the outer node's own scan is what rejects it.
    let mut outer = json_of(&Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![stamped_lit()],
        sp(),
    ));
    node_children_mut(&mut outer)[0] = json_of(&Expr::BareList(vec![raw("if")], sp()));

    assert_deserialize_rejected(outer, "if");
}

#[test]
fn deserialize_accepts_the_same_tree_without_the_smuggled_tag() {
    // Negative parity: the attack payloads above must fail for the raw tag,
    // not because the hand-built JSON was malformed.
    let clean = Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![Expr::node(
            DeepTag::Var,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("x".to_string()), sp())],
            sp(),
        )],
        sp(),
    );
    let text = serde_json::to_string(&json_of(&clean)).expect("clean tree serializes");
    let decoded: Expr = serde_json::from_str(&text).expect("clean tree decodes");
    assert_eq!(decoded, clean);
}

// ── Leg 2: the post-construction mutators ────────────────────────────
//
// Each mutator revalidates the WHOLE candidate node before committing, so
// none of them can widen a node that was clean at construction. Each test
// also asserts the original is untouched, since a mutator that rejected
// after mutating would leave a live node the induction no longer covers.

#[test]
fn try_replace_child_rejects_a_raw_tag_below_carriers() {
    let mut node = Node::try_new(
        DeepTag::App,
        Metadata::default(),
        vec![stamped_lit(), stamped_lit()],
    )
    .expect("clean node constructs");

    let rejected = node.try_replace_child(1, Expr::BareList(vec![raw("tuple")], sp()));
    assert!(matches!(
        rejected,
        Err(NodeError::RawVocabularyTag { ref raw_tag, .. }) if raw_tag == "tuple"
    ));
    assert_eq!(node.expr_child(1), &stamped_lit(), "rejection is atomic");
}

#[test]
fn try_replace_children_rejects_a_raw_tag_beside_a_stamped_child() {
    // The replacement keeps a legitimate stamped sibling in front of the
    // payload: stopping the scan at that sibling must not stop the scan of
    // the rest of the vector.
    let mut node =
        Node::try_new(DeepTag::App, Metadata::default(), vec![stamped_lit()]).expect("clean node");

    let rejected = node.try_replace_children(vec![
        stamped_lit(),
        Expr::UnknownForm(Box::new(UnknownFormData {
            head: "future-form".to_string(),
            meta: Metadata::default(),
            children: vec![raw("fn")],
            span: sp(),
        })),
    ]);
    assert!(matches!(
        rejected,
        Err(NodeError::RawVocabularyTag { ref raw_tag, .. }) if raw_tag == "fn"
    ));
    assert_eq!(node.child_count(), 1, "rejection is atomic");
}

#[test]
fn try_replace_meta_rejects_a_raw_tag_below_nested_carriers() {
    let metadata = meta_with("probe", stamped_lit());
    let before = metadata.clone();
    let buried = Expr::BareList(
        vec![Expr::MetaExpr(
            MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(raw("match")),
            },
            sp(),
        )],
        sp(),
    );
    let error = chelis_deep::annotations::RuntimeExpression::try_new(buried).unwrap_err();
    assert!(error.to_string().contains("match"));
    assert_eq!(metadata, before, "rejection is atomic");
}

// ── Leg 3: the path.rs splice surface ────────────────────────────────
//
// `splice_function_body` and `spliced_function_def` rewrite inside a
// stamped tree. They rebuild inside-out through `try_replace_child`, so
// every payload shape below has to come back as `InvalidStampedRewrite`
// with the source program untouched.

const MODULE: &str = r#"(module {}
  attack.probe
  (export {} identity)
  (defsig {} identity (t-fn {} (t-prim {} f32) (t-prim {} f32)))
  (def {}
    identity
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (var {} x))))
"#;

/// The raw-payload shapes a splice can carry. Nothing here puts the raw
/// tag BELOW a stamped node, because nothing can: the shapes are every
/// way a caller can wrap one in an unvalidated carrier instead.
fn splice_payload_shapes() -> Vec<(&'static str, Expr, &'static str)> {
    vec![
        ("bare raw form", raw("lit"), "lit"),
        (
            "raw under a BareList",
            Expr::BareList(vec![raw("app")], sp()),
            "app",
        ),
        (
            "raw in an UnknownForm child",
            Expr::UnknownForm(Box::new(UnknownFormData {
                head: "future-form".to_string(),
                meta: Metadata::default(),
                children: vec![raw("if")],
                span: sp(),
            })),
            "if",
        ),
        (
            "raw as a sibling after a stamped node",
            Expr::BareList(vec![stamped_lit(), raw("let")], sp()),
            "let",
        ),
        (
            "raw in a MetaExpr wrapping a stamped node",
            Expr::MetaExpr(
                MetaExpr {
                    metadata: Metadata::default(),
                    expr: Box::new(Expr::BareList(vec![stamped_lit(), raw("tuple")], sp())),
                },
                sp(),
            ),
            "tuple",
        ),
    ]
}

#[test]
fn splice_function_body_rejects_every_raw_payload_shape() {
    let module = parse_str(MODULE).expect("fixture parses");

    for (label, payload, expected_tag) in splice_payload_shapes() {
        let result = splice_function_body(&module, "attack.probe.identity", payload);
        let Err(ResolveError::InvalidStampedRewrite { message, .. }) = &result else {
            panic!("{label}: a raw `{expected_tag}` must not cross the gate, got {result:?}");
        };
        assert!(
            message.contains(&format!("raw closed-vocabulary tag `{expected_tag}`")),
            "{label}: the rewrite error must name the raw tag, got: {message}"
        );
        assert_eq!(
            module,
            parse_str(MODULE).expect("fixture parses"),
            "{label}: the source program remains unchanged"
        );
    }
}

#[test]
fn spliced_function_def_rejects_every_raw_payload_shape() {
    let module = parse_str(MODULE).expect("fixture parses");

    for (label, payload, expected_tag) in splice_payload_shapes() {
        let result = spliced_function_def(&module, "attack.probe.identity", payload);
        let Err(ResolveError::InvalidStampedRewrite { message, .. }) = &result else {
            panic!("{label}: a raw `{expected_tag}` must not cross the gate, got {result:?}");
        };
        assert!(
            message.contains(&format!("raw closed-vocabulary tag `{expected_tag}`")),
            "{label}: the rewrite error must name the raw tag, got: {message}"
        );
    }
}

#[test]
fn splice_accepts_a_clean_stamped_body() {
    // Negative parity for the two tests above: the same splice path with a
    // clean stamped body succeeds, so the rejections are about the raw tag.
    let module = parse_str(MODULE).expect("fixture parses");
    let spliced = splice_function_body(&module, "attack.probe.identity", stamped_lit())
        .expect("a clean stamped body splices");
    assert_ne!(spliced, module);
}

// ── Leg 4: MetaExpr and the rest of the carrier surface ──────────────
//
// The below-gate scan skips exactly one edge: an `Expr::Node` child.
// Every other carrier must still be walked to any depth, in child and in
// metadata position alike.

#[test]
fn metaexpr_expr_slot_is_scanned_in_both_positions() {
    let carrier = Expr::MetaExpr(
        MetaExpr {
            metadata: Metadata::default(),
            expr: Box::new(Expr::BareList(vec![raw("lit")], sp())),
        },
        sp(),
    );
    assert_child_rejected(carrier.clone(), "lit");
    assert_metadata_rejected(carrier, "lit");
}

#[test]
fn metaexpr_metadata_is_scanned_in_both_positions() {
    let value = serde_json::json!({ "MetaExpr": [{"entries": [["property_seed", raw("app")]], "expr": stamped_lit()}, sp()] });
    let error = serde_json::from_value::<Expr>(value).unwrap_err();
    assert!(error.to_string().contains("app"));
    assert_metadata_rejected(raw("app"), "app");
}

#[test]
fn a_six_deep_carrier_chain_is_scanned_to_the_bottom() {
    let mut chain = raw("match");
    for _ in 0..3 {
        chain = Expr::BareList(vec![chain], sp());
        chain = Expr::MetaExpr(
            MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(chain),
            },
            sp(),
        );
    }
    assert_child_rejected(chain.clone(), "match");
    assert_metadata_rejected(chain, "match");
}

#[test]
fn stamped_nodes_interleaved_in_the_chain_do_not_blind_the_scan() {
    // Alternating stamped / unvalidated levels. Each stamped node is clean
    // (it has to be -- see the next test), and the raw tag rides the
    // unvalidated carriers all the way down beside them.
    let mut chain = raw("tuple");
    for _ in 0..3 {
        chain = Expr::BareList(vec![stamped_lit(), chain], sp());
        chain = Expr::UnknownForm(Box::new(UnknownFormData {
            head: "future-form".to_string(),
            meta: meta_with("sibling", stamped_lit()),
            children: vec![chain],
            span: sp(),
        }));
    }

    assert_child_rejected(chain.clone(), "tuple");
    assert_metadata_rejected(chain, "tuple");
}

#[test]
fn every_public_entry_point_refuses_to_place_a_raw_tag_below_a_stamped_node() {
    // The induction in one test: to hide a raw tag under a stamped node an
    // attacker must first BUILD that stamped node, and every way in
    // refuses. This is why the skip is sound, and it is what turns red if
    // a fourth construction path ever appears.
    let payload = vec![Expr::BareList(vec![raw("lit")], sp())];

    // 1. The boundary constructor.
    assert!(matches!(
        Node::try_new(DeepTag::App, Metadata::default(), payload.clone()),
        Err(NodeError::RawVocabularyTag { .. })
    ));

    // 2. The internal constructor.
    let panicked =
        std::panic::catch_unwind(|| Node::new(DeepTag::App, Metadata::default(), payload.clone()));
    assert!(panicked.is_err(), "Node::new must panic on the violation");

    // 3. The typed producer.
    let panicked = std::panic::catch_unwind(|| {
        Expr::node(DeepTag::App, Metadata::default(), payload.clone(), sp())
    });
    assert!(panicked.is_err(), "Expr::node must panic on the violation");

    // 4. Metadata rejects the raw payload before it can reach any AST carrier.
    assert_metadata_rejected(raw("lit"), "lit");

    // 5. The wire boundary (covered in depth by leg 1).
    let mut smuggled = stamped_var_json();
    node_children_mut(&mut smuggled)[0] = json_of(&raw("lit"));
    assert_deserialize_rejected(smuggled, "lit");

    // 6. The mutators (covered in depth by leg 2).
    let mut node =
        Node::try_new(DeepTag::App, Metadata::default(), vec![stamped_lit()]).expect("clean node");
    assert!(node.try_replace_children(payload).is_err());
    assert_eq!(node.child_count(), 1);
}
