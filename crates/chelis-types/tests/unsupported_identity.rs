use chelis_types::unsupported::{
    RejectionAuthorityKind, SpanRef, Stage, Unsupported, UnsupportedKind,
};

fn complete_trial_error() -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Construct("a non-literal window list".into()),
        "compiled-backend lowering",
        Stage::Lowering,
        chelis_types::unimplemented_rejection!(1058, "current wording"),
    )
    .with_span(SpanRef {
        offset: Some(17),
        len: Some(4),
        span_id: Some("surf:17..21".into()),
    })
    .with_supported_alternative("use integer literal window and stride lists")
}

#[test]
fn unsupported_stays_within_the_result_error_size_limit() {
    assert!(
        std::mem::size_of::<Unsupported>() < 128,
        "Unsupported is {} bytes and will trigger clippy::result_large_err",
        std::mem::size_of::<Unsupported>()
    );
}

#[test]
fn identity_carries_every_current_typed_obligation_without_hint_wording() {
    let identity = complete_trial_error().identity();
    assert_eq!(identity.brand, "unsupported:");
    assert_eq!(
        identity.what,
        UnsupportedKind::Construct("a non-literal window list".into())
    );
    assert_eq!(identity.context, "compiled-backend lowering");
    assert_eq!(identity.stage, Stage::Lowering);
    assert_eq!(
        identity.span,
        Some(Box::new(SpanRef {
            offset: Some(17),
            len: Some(4),
            span_id: Some("surf:17..21".into()),
        }))
    );
    assert_eq!(identity.disposition, RejectionAuthorityKind::Unimplemented);
    assert_eq!(identity.atom, None);
    assert_eq!(
        identity.tracking_issue.map(|issue| issue.number()),
        Some(1058)
    );
    assert_eq!(
        identity.supported_alternative.as_deref(),
        Some("use integer literal window and stride lists")
    );
}

#[test]
fn identity_mutations_fail_but_hint_wording_is_derived() {
    let base = complete_trial_error();
    let differently_worded = Unsupported::new(
        base.what.clone(),
        base.context.clone(),
        base.stage,
        chelis_types::unimplemented_rejection!(1058, "rewritten prose only"),
    )
    .with_span((**base.span.as_ref().expect("test span")).clone())
    .with_supported_alternative("use integer literal window and stride lists");
    assert_eq!(base.identity(), differently_worded.identity());
    assert_ne!(
        base.identity(),
        Unsupported::new(
            UnsupportedKind::Construct("a non-literal stride list".into()),
            base.context.clone(),
            base.stage,
            base.authority,
        )
        .with_span((**base.span.as_ref().expect("test span")).clone())
        .with_supported_alternative("use integer literal window and stride lists")
        .identity()
    );
    assert_ne!(
        base.identity(),
        Unsupported::new(
            base.what.clone(),
            "a different lowering context",
            base.stage,
            base.authority,
        )
        .with_span((**base.span.as_ref().expect("test span")).clone())
        .with_supported_alternative("use integer literal window and stride lists")
        .identity()
    );
    assert_ne!(
        base.identity(),
        Unsupported::new(
            base.what.clone(),
            base.context.clone(),
            Stage::Codegen("c"),
            base.authority,
        )
        .with_span((**base.span.as_ref().expect("test span")).clone())
        .with_supported_alternative("use integer literal window and stride lists")
        .identity()
    );
    assert_ne!(
        base.identity(),
        Unsupported::new(
            base.what.clone(),
            base.context.clone(),
            base.stage,
            base.authority,
        )
        .with_span(SpanRef {
            offset: Some(18),
            len: Some(4),
            span_id: Some("surf:18..22".into()),
        })
        .with_supported_alternative("use integer literal window and stride lists")
        .identity()
    );
    assert_ne!(
        base.identity(),
        Unsupported::new(
            base.what.clone(),
            base.context.clone(),
            base.stage,
            chelis_types::deliberate_rejection!("[04-TOT-2]", "different authority wording"),
        )
        .with_span((**base.span.as_ref().expect("test span")).clone())
        .with_supported_alternative("use integer literal window and stride lists")
        .identity()
    );
    assert_ne!(
        base.identity(),
        Unsupported::new(
            base.what.clone(),
            base.context.clone(),
            base.stage,
            base.authority,
        )
        .with_span((**base.span.as_ref().expect("test span")).clone())
        .with_supported_alternative("run under chelis eval")
        .identity()
    );

    let deliberate = |authority| {
        Unsupported::new(
            UnsupportedKind::Builtin("softmax".into()),
            "C host emission",
            Stage::Codegen("c"),
            authority,
        )
        .with_supported_alternative("run under chelis eval")
        .identity()
    };
    assert_ne!(
        deliberate(chelis_types::deliberate_rejection!(
            "[04-TOT-2]",
            "first wording"
        )),
        deliberate(chelis_types::deliberate_rejection!(
            "[04-TOT-3]",
            "second wording"
        )),
        "changing the deciding numbered atom must change identity"
    );

    let tracked = |authority| {
        Unsupported::new(
            UnsupportedKind::Construct("runtime-shaped window".into()),
            "compiled lowering",
            Stage::Lowering,
            authority,
        )
        .identity()
    };
    assert_ne!(
        tracked(chelis_types::unimplemented_rejection!(
            1058,
            "first wording"
        )),
        tracked(chelis_types::unimplemented_rejection!(
            729,
            "second wording"
        )),
        "changing tracking metadata must change the current trial identity"
    );
}
