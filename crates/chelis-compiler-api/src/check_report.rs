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
//!    `format!`, so `"score": 1` is the shipped spelling and is asserted by
//!    substring across the CLI test corpus.
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
    use crate::schema::{CheckResult, FitnessComponents, WireCheckResult};

    fn clean_report() -> CheckResult {
        CheckResult {
            score: 1.0,
            components: FitnessComponents {
                parse: 1.0,
                structure: 1.0,
                names: 1.0,
                types: 1.0,
            },
            typed_nodes: 4,
            untyped_nodes: 0,
            total_nodes: 4,
            unresolved_names: Vec::new(),
            inferred_signatures: None,
            errors: Vec::new(),
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
    /// corpus asserts the substring `"score": 1` in eighteen files. A
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

    /// A fractional double is unchanged, so the deviation above is confined
    /// to the integral case rather than being a lossy re-spelling.
    #[test]
    fn a_fractional_double_is_unchanged() {
        let mut report = clean_report();
        report.score = 0.7299999999999999;
        report.components.names = 0.75;
        let rendered = report.to_report_json().expect("render");
        assert!(
            rendered.contains("\"score\": 0.7299999999999999,"),
            "{rendered}"
        );
        assert!(rendered.contains("\"names\": 0.75,\n"), "{rendered}");
        let parsed: WireCheckResult = serde_json::from_str(&rendered).expect("round-trip");
        assert_eq!(parsed.score, 0.7299999999999999);
    }

    /// [04-FIT-18] admits signed zero and requires it preserved. `Display`
    /// and `ryu` agree on the sign; the point is that this formatter does
    /// not normalize it away.
    #[test]
    fn signed_zero_survives_the_formatter() {
        let mut report = clean_report();
        report.score = -0.0;
        let rendered = report.to_report_json().expect("render");
        assert!(rendered.contains("\"score\": -0,"), "{rendered}");
        let parsed: WireCheckResult = serde_json::from_str(&rendered).expect("round-trip");
        assert!(
            parsed.score.is_sign_negative(),
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
        report.inferred_signatures = Some(serde_json::json!([
            {"function": "f", "params": [{"index": 0, "name": "x"}]}
        ]));
        let rendered = report.to_report_json().expect("render");
        assert!(
            rendered.contains("\n  \"unresolved_names\": [\"nope\",\"also_nope\"],\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains(
                "\n  \"inferred_signatures\": \
                 [{\"function\":\"f\",\"params\":[{\"index\":0,\"name\":\"x\"}]}],\n"
            ),
            "a nested object inside an array stays compact; got:\n{rendered}"
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
        requested.inferred_signatures = Some(serde_json::Value::Array(Vec::new()));
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
        let mut report = clean_report();
        report.inferred_signatures = Some(serde_json::json!({}));
        let rendered = report.to_report_json().expect("render");
        assert!(
            rendered.contains("\"inferred_signatures\": {},"),
            "{rendered}"
        );
    }

    /// The rendered document is readable by the consumer type. A layout the
    /// producer can write but the consumer cannot read would satisfy every
    /// byte pin above and still be broken.
    #[test]
    fn the_rendered_document_reads_back_through_the_consumer_type() {
        let mut report = clean_report();
        report.unresolved_names = vec!["nope".to_string()];
        report.inferred_signatures = Some(serde_json::json!([{"function": "f"}]));
        let rendered = report.to_report_json().expect("render");
        let parsed: WireCheckResult = serde_json::from_str(&rendered).expect("round-trip");
        assert_eq!(parsed.typed_nodes, 4);
        assert_eq!(parsed.total_nodes, 4);
        assert_eq!(parsed.unresolved_names, vec!["nope".to_string()]);
        assert_eq!(
            parsed.inferred_signatures,
            Some(serde_json::json!([{"function": "f"}])),
            "the member survives the crossing rather than being dropped"
        );
        assert!(parsed.errors.is_empty());
    }
}
