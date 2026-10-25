use super::*;

#[test]
fn synthesized_effects_preserve_typed_members_and_decoded_wire_tags() {
    let mut effects = EffectSet::new();
    effects.insert(Effect::Accum);
    effects.insert(Effect::Io);
    effects.insert(Effect::Resource("gpu0".into()));
    let metadata = Metadata::from(MetadataValue::Effects(effect_set_metadata(&effects)));
    let mut payload = None;
    metadata.visit_syntax(&mut |_, value| payload = Some(value.clone()));
    let expr = payload.unwrap();
    assert_eq!(
        chelis_deep::validate::find_raw_vocabulary_tag(core::slice::from_ref(&expr)),
        None
    );
    assert_eq!(expr.tag(), Some(DeepTag::Effects));
    let Expr::Node(node, _) = &expr else {
        panic!("structural effects node")
    };
    let resource = node
        .children_slice()
        .iter()
        .find(|child| child.tag() == Some(DeepTag::Resource))
        .unwrap();
    assert_eq!(resource.tag(), Some(DeepTag::Resource));
}

#[test]
fn effect_names_are_payload_not_vocabulary_tags() {
    let mut effects = EffectSet::new();
    effects.insert(Effect::Accum);
    let payload = effect_set_metadata(&effects);
    assert!(matches!(payload.values(), [EffectMember::Name(name)] if name.value() == "accum"));
}

#[test]
fn effect_annotation_reaches_registered_expressions_and_preserves_data() {
    let source = "(def {property_seed: (fn {} (params {}) (app {} (var {} debug) (lit {} 1))), custom: (fn {} (params {}) (app {} (var {} debug) (lit {} 1))), source: (original (fn {} (params {}) (app {} (var {} debug) (lit {} 1))))} f (lit {} 1))";
    let parsed = chelis_deep::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let Expr::Node(node, _) = &parsed[0] else {
        panic!("stamped declaration")
    };
    let unknown = Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
        head: "custom".into(),
        meta: node.meta().clone(),
        children: vec![],
        span: parsed[0].span(),
    }));
    for original in [parsed[0].clone(), unknown] {
        let rewritten = Analysis::default().annotate_effects(
            &original,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &mut Locals::new(),
        );
        let metadata = match &rewritten {
            Expr::Node(node, _) => node.meta(),
            Expr::UnknownForm(data) => &data.meta,
            _ => unreachable!(),
        };
        assert_eq!(metadata.extensions(), node.meta().extensions());
        let closure = metadata.property_seed().unwrap().expression();
        let Expr::Node(closure, _) = closure else {
            panic!("closure")
        };
        assert!(
            closure
                .meta()
                .values()
                .any(|v| matches!(v, MetadataValue::Effects(_))),
            "registered expression closure needs inferred effects"
        );
        assert_eq!(
            metadata.source(),
            node.meta().source(),
            "preserved source is data"
        );
    }
}
