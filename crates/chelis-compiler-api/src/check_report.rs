//! The canonical rendering of the `chelis check` report document
//! (chelis#886).
//!
//! [04-FIT-11] requires the report to be produced by serializing one typed
//! value: a hand-assembled document admits drift between the bytes a
//! consumer reads and the type that claims to describe them. Before this
//! module the CLI held two `format!` templates spelling the document's keys
//! -- one for the checker's report and one for the early-failure path -- so
//! the type existed and described nothing.
//!
//! The document is a *published* interface: reward surfaces, the hull
//! conformance input, and downstream shells read it. Adopting a typed
//! producer is therefore not licence to restyle the bytes, and this module
//! exists because no stock `serde_json` formatter reproduces the layout the
//! templates emitted. [`ReportFormatter`] does, exactly, so the defect is
//! repaired without a wire change.
//!
//! Three deviations from [`serde_json::ser::PrettyFormatter`] carry the
//! whole layout:
//!
//! 1. **Arrays are compact, and so is everything inside one.** The document
//!    breaks objects across lines but keeps `unresolved_names`, `errors` and
//!    `inferred_signatures` on a single line each.
//! 2. **`f64` is written with `Display`, not `ryu`.** `Display` prints an
//!    integral double as `1`; `serde_json` prints `1.0`. The templates used
//!    `format!` for `score` and `components`, so `"score": 1` is the shipped
//!    spelling and is asserted by substring in 17 files under
//!    `crates/chelis-cli/tests/`.
//!
//!    This applies to each fixed binary64 adapter in the document,
//!    including diagnostic severity. The adapters admit only finite values
//!    in their declared domains, and the JSON decoder preserves the bits
//!    of the emitted decimal. Integral severity uses the same document
//!    spelling; its endpoint and signed-zero controls are below.
//!
//! 3. **Compact separators carry no space** (`":"`, `","`), matching the
//!    `serde_json::to_string` that produced the per-error objects.

use std::io;

use serde::Serialize;
use serde_json::ser::Formatter;

use crate::schema::CheckResult;

impl CheckResult {
    /// Render the report as `chelis check` publishes it.
    ///
    /// This is the single producer [04-FIT-11] requires: every `chelis
    /// check` document, including the ones for failures that occur before
    /// the checker runs ([04-FIT-12]), is this function's output.
    pub fn to_report_json(&self) -> Result<String, serde_json::Error> {
        let mut bytes = Vec::new();
        let mut serializer =
            serde_json::Serializer::with_formatter(&mut bytes, ReportFormatter::default());
        self.serialize(&mut serializer)?;
        // `serde_json` writes UTF-8 by construction; it escapes anything it
        // cannot.
        Ok(String::from_utf8(bytes).expect("serde_json emits UTF-8"))
    }
}

/// The report document's layout, as a `serde_json` formatter.
#[derive(Default)]
struct ReportFormatter {
    /// Nesting depth of the enclosing objects, for the indent.
    object_depth: usize,
    /// Nesting depth of the enclosing arrays. Non-zero means compact.
    array_depth: usize,
    /// Whether the container being closed emitted at least one member, so an
    /// empty one closes as `{}` rather than over two lines. Mirrors
    /// `PrettyFormatter`'s own field, including its reuse across both
    /// container kinds: a container's close is always preceded by its own
    /// last `end_*_value`, so one flag suffices at any depth.
    has_value: bool,
}

impl ReportFormatter {
    /// Inside an array, the document is single-line.
    fn compact(&self) -> bool {
        self.array_depth > 0
    }

    fn indent<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        for _ in 0..self.object_depth {
            writer.write_all(b"  ")?;
        }
        Ok(())
    }
}

impl Formatter for ReportFormatter {
    /// `Display`, not the default `ryu`: see the module note.
    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        write!(writer, "{value}")
    }

    fn begin_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.array_depth += 1;
        self.has_value = false;
        writer.write_all(b"[")
    }

    fn end_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.array_depth -= 1;
        writer.write_all(b"]")
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b",")
        }
    }

    fn end_array_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.object_depth += 1;
        self.has_value = false;
        writer.write_all(b"{")
    }

    fn end_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.object_depth -= 1;
        if !self.compact() && self.has_value {
            writer.write_all(b"\n")?;
            self.indent(writer)?;
        }
        writer.write_all(b"}")
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b",")?;
        }
        if self.compact() {
            return Ok(());
        }
        writer.write_all(b"\n")?;
        self.indent(writer)
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(if self.compact() { b":" } else { b": " })
    }

    fn end_object_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.has_value = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::schema::numbers::{NonnegativeCount, UnitInterval};
    use crate::schema::{
        CheckResult, Diagnostic, WireCheckResult, WireInferredParameter, WireInferredSignature,
        WireInferredType,
    };
    use serde::Serialize;

    fn clean_report() -> CheckResult {
        CheckResult::try_from_fitness(&chelis_types::FitnessReport {
            score: 1.0,
            components: chelis_types::fitness::FitnessComponents {
                parse: 1.0,
                structure: 1.0,
                names: 1.0,
                types: 1.0,
            },
            typed_nodes: 4,
            untyped_nodes: 0,
            total_nodes: 4,
            unresolved_names: Vec::new(),
            errors: Vec::new(),
        })
        .expect("valid measured report")
    }

    fn signature() -> WireInferredSignature {
        let boolean = WireInferredType::Prim {
            name: "bool".into(),
        };
        WireInferredSignature {
            function: "f".into(),
            recursive_cycle: false,
            checked_signature: "bool".into(),
            display_signature: "bool".into(),
            checked_signature_structured: boolean.clone(),
            display_signature_structured: boolean.clone(),
            effect_row: Vec::new(),
            effect_row_display: Vec::new(),
            params: vec![WireInferredParameter {
                index: 0,
                name: "x".into(),
                written: false,
                inferred_read_only: false,
                checked_type: "bool".into(),
                display_type: "bool".into(),
                checked_type_structured: boolean.clone(),
                display_type_structured: boolean,
            }]
            .try_into()
            .expect("parameter is in its owning position"),
        }
    }

    /// The layout, in full. This is the published document, so the pin is
    /// the bytes and not a parsed value: object member order, the two-space
    /// indent, and the single-line arrays are all part of what a consumer
    /// reads.
    #[test]
    fn the_document_layout_is_pinned() {
        assert_eq!(
            clean_report().to_report_json().expect("render"),
            "{\n  \"score\": 1,\n  \"components\": {\n    \"parse\": 1,\n    \
             \"structure\": 1,\n    \"names\": 1,\n    \"types\": 1\n  },\n  \
             \"typed_nodes\": 4,\n  \"untyped_nodes\": 0,\n  \"total_nodes\": 4,\n  \
             \"unresolved_names\": [],\n  \"errors\": []\n}"
        );
    }

    /// The deviation that carries the most weight, stated on its own.
    ///
    /// `serde_json` writes an integral double as `1.0`. The templates this
    /// replaces used `format!`, so `1` is the shipped spelling, and the CLI
    /// corpus asserts the substring `"score": 1` in 17 files. A
    /// formatter swapped for a stock one passes every structural test and
    /// fails here.
    #[test]
    fn an_integral_double_keeps_its_format_spelling() {
        let rendered = clean_report().to_report_json().expect("render");
        assert!(
            rendered.contains("\"score\": 1,"),
            "an integral score is spelled `1`; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("\"score\": 1.0"),
            "`1.0` is `serde_json`'s spelling, not this document's"
        );
    }

    /// `severity` takes the document's spelling too, not `serde_json`'s.
    ///
    /// The bounded severity adapter includes both endpoints, so its integral
    /// values follow the same document spelling as scores and components.
    #[test]
    fn an_integral_severity_uses_the_document_spelling() {
        let mut report = clean_report();
        report.errors = vec![
            Diagnostic::try_from_check_error(&chelis_types::errors::CheckError {
                kind: chelis_types::errors::CheckErrorKind::Other,
                message: "probe".to_string(),
                severity: 1.0,
                expected: None,
                got: None,
                span_offset: None,
                span_id: None,
                suggestions: Vec::new(),
            })
            .expect("valid diagnostic"),
        ];
        let rendered = report.to_report_json().expect("render");
        assert!(rendered.contains("\"severity\":1}"), "{rendered}");
        assert!(!rendered.contains("\"severity\":1.0"), "{rendered}");
    }

    /// A fractional double is unchanged, so the deviation above is confined
    /// to the integral case rather than being a lossy re-spelling.
    #[test]
    fn a_fractional_double_is_unchanged() {
        let mut report = clean_report();
        report.score = UnitInterval::new(0.7299999999999999).unwrap();
        report.components.names = UnitInterval::new(0.75).unwrap();
        let rendered = report.to_report_json().expect("render");
        assert!(
            rendered.contains("\"score\": 0.7299999999999999,"),
            "{rendered}"
        );
        assert!(rendered.contains("\"names\": 0.75,\n"), "{rendered}");
        let parsed: WireCheckResult = serde_json::from_str(&rendered).expect("round-trip");
        assert_eq!(parsed.score.get(), 0.7299999999999999);
    }

    /// [04-FIT-18] admits signed zero and requires it preserved. `Display`
    /// and `ryu` agree on the sign; the point is that this formatter does
    /// not normalize it away.
    #[test]
    fn signed_zero_survives_the_formatter() {
        let mut report = clean_report();
        report.score = UnitInterval::new(-0.0).unwrap();
        let rendered = report.to_report_json().expect("render");
        assert!(rendered.contains("\"score\": -0,"), "{rendered}");
        let parsed: WireCheckResult = serde_json::from_str(&rendered).expect("round-trip");
        assert!(
            parsed.score.get().is_sign_negative(),
            "the sign is part of the value"
        );
    }

    /// Arrays are single-line, and so is everything nested inside one --
    /// including the objects the `errors` array carries. Without the
    /// nesting half of that rule a stock pretty formatter would break every
    /// diagnostic across seven lines.
    #[test]
    fn an_array_and_its_contents_stay_on_one_line() {
        let mut report = clean_report();
        report.unresolved_names = vec!["nope".to_string(), "also_nope".to_string()];
        report.inferred_signatures = Some(vec![signature()]);
        let rendered = report.to_report_json().expect("render");
        let signatures = serde_json::to_string(&vec![signature()]).unwrap();
        assert!(
            rendered.contains(&format!("\n  \"inferred_signatures\": {signatures},\n")),
            "typed nested signature stays compact; got:\n{rendered}",
        );
    }

    /// `inferred_signatures` is a member of the report's type, absent by
    /// omission rather than emitted as `null` ([04-FIT-13]). "Absent" and
    /// "requested but empty" are different documents.
    #[test]
    fn inferred_signatures_is_absent_when_not_requested() {
        let rendered = clean_report().to_report_json().expect("render");
        assert!(!rendered.contains("inferred_signatures"), "{rendered}");

        let mut requested = clean_report();
        requested.inferred_signatures = Some(Vec::new());
        let rendered = requested.to_report_json().expect("render");
        assert!(
            rendered.contains("\n  \"inferred_signatures\": [],\n  \"errors\""),
            "an empty request sits between `unresolved_names` and `errors`; got:\n{rendered}"
        );
    }

    /// An empty container closes on its own line, not across two. Guards the
    /// `has_value` bookkeeping the indent depends on.
    #[test]
    fn an_empty_object_closes_compactly() {
        let object = std::collections::BTreeMap::<String, String>::new();
        let mut bytes = Vec::new();
        vec![object]
            .serialize(&mut serde_json::Serializer::with_formatter(
                &mut bytes,
                super::ReportFormatter::default(),
            ))
            .unwrap();
        assert_eq!(bytes, b"[{}]");
    }

    /// The rendered document is readable by the consumer type. A layout the
    /// producer can write but the consumer cannot read would satisfy every
    /// byte pin above and still be broken.
    #[test]
    fn the_rendered_document_reads_back_through_the_consumer_type() {
        let mut report = clean_report();
        report.unresolved_names = vec!["nope".to_string()];
        report.inferred_signatures = Some(vec![signature()]);
        let rendered = report.to_report_json().expect("render");
        let parsed: WireCheckResult = serde_json::from_str(&rendered).expect("round-trip");
        assert_eq!(parsed.typed_nodes.get(), 4);
        assert_eq!(parsed.total_nodes.get(), 4);
        assert_eq!(parsed.unresolved_names, vec!["nope".to_string()]);
        assert_eq!(
            serde_json::to_value(parsed.inferred_signatures).unwrap(),
            serde_json::to_value(report.inferred_signatures).unwrap(),
            "the member survives the crossing rather than being dropped"
        );
        assert!(parsed.errors.is_empty());
    }
    #[test]
    fn emitted_report_decimals_preserve_their_exact_binary64_bits() {
        for bits in [0x3fb9_52b7_4674_9bc0, 0x8000_0000_0000_0000, 1] {
            let mut report = clean_report();
            report.score = UnitInterval::new(f64::from_bits(bits)).unwrap();
            let encoded = report.to_report_json().unwrap();
            let decoded: WireCheckResult = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded.score.get().to_bits(), bits, "{encoded}");
        }
    }

    #[test]
    fn document_rendering_rejects_inconsistent_counter_relationships() {
        let mut report = clean_report();
        report.typed_nodes = NonnegativeCount::new(5).unwrap();
        assert!(
            report
                .to_report_json()
                .unwrap_err()
                .to_string()
                .contains("inconsistent report counts")
        );
    }
}
