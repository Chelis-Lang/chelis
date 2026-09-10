//! spec/10 §3.3: foreign byte coordinates travel independently of local access.
use chelis_compiler_api::schema::{DiagnosticSpan, Span};

#[test]
fn locations_preserve_full_u64_and_distinguish_points_from_empty_ranges() {
    let point = format!(r#"{{"span":"point","offset":{}}}"#, u64::MAX);
    let decoded: DiagnosticSpan = serde_json::from_str(&point).unwrap();
    let offset: u64 = decoded.offset();
    assert_eq!(offset, u64::MAX);
    assert_eq!(decoded.extent(), None);
    assert_eq!(serde_json::to_string(&decoded).unwrap(), point);
    let empty: DiagnosticSpan =
        serde_json::from_str(r#"{"span":"range","offset":0,"len":0}"#).unwrap();
    assert_eq!(empty.extent(), Some(0));
    for json in [
        r#"{"span":"point","offset":-1}"#,
        r#"{"span":"range","offset":0,"len":1.0}"#,
        r#"{"span":"point","offset":0,"len":0}"#,
    ] {
        assert!(serde_json::from_str::<DiagnosticSpan>(json).is_err());
    }
}

#[test]
fn local_source_access_checks_arithmetic_bounds_and_utf8() {
    let source = "aλz";
    assert_eq!(Span { offset: 1, len: 2 }.slice(source).unwrap(), "λ");
    assert_eq!(Span { offset: 4, len: 0 }.slice(source).unwrap(), "");
    for span in [
        Span {
            offset: u64::MAX,
            len: 1,
        },
        Span {
            offset: 1,
            len: u64::MAX,
        },
        Span { offset: 5, len: 0 },
        Span { offset: 1, len: 1 },
        Span { offset: 2, len: 1 },
    ] {
        assert!(span.slice(source).is_err(), "accepted {span:?}");
        // Foreign transport does not claim access to this local source.
        let json = serde_json::to_string(&span).unwrap();
        assert_eq!(serde_json::from_str::<Span>(&json).unwrap(), span);
    }
}
