//! Prepare evaluation output before deciding which terminal path may emit it.
use chelis_compiler_api::schema::EvalResult;
use serde::Serialize;
use std::io::{self, Write};

fn serialize_eval_json_into<W: Write>(result: &EvalResult, writer: W) -> serde_json::Result<()> {
    let mut serializer = serde_json::Serializer::new(writer);
    result.serialize(serde_stacker::Serializer::new(&mut serializer))
}

pub(crate) fn serialize_eval_json(result: &EvalResult) -> serde_json::Result<String> {
    let mut bytes = Vec::new();
    serialize_eval_json_into(result, &mut bytes)?;
    Ok(String::from_utf8(bytes).expect("serde_json emits UTF-8"))
}

#[derive(Debug)]
pub(crate) struct NumericTrapCliError(pub(crate) String);

impl std::fmt::Display for NumericTrapCliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NumericTrapCliError {}

pub(crate) struct EvalOutput {
    stdout: Option<String>,
    stderr: Vec<String>,
    pub(crate) error: Option<String>,
    numeric_trap: bool,
}

impl EvalOutput {
    pub(crate) fn text(rendered: String) -> Self {
        if rendered.is_empty() {
            Self {
                stdout: None,
                stderr: vec![
                    "warning: input contains only def declarations; nothing to evaluate".into(),
                ],
                error: None,
                numeric_trap: false,
            }
        } else {
            Self {
                stdout: Some(rendered),
                stderr: vec![],
                error: None,
                numeric_trap: false,
            }
        }
    }

    pub(crate) fn json(rendered: String) -> Self {
        Self {
            stdout: Some(rendered),
            stderr: vec![],
            error: None,
            numeric_trap: false,
        }
    }

    pub(crate) fn failure(transcript: Vec<String>, error: String, json: bool) -> Self {
        Self::failure_with_trap_kind(transcript, error, json, false)
    }

    pub(crate) fn numeric_trap_failure(transcript: Vec<String>, error: String, json: bool) -> Self {
        Self::failure_with_trap_kind(transcript, error, json, true)
    }

    fn failure_with_trap_kind(
        transcript: Vec<String>,
        error: String,
        json: bool,
        numeric_trap: bool,
    ) -> Self {
        Self {
            stdout: (!json && !transcript.is_empty()).then(|| transcript.join("\n")),
            stderr: if json { transcript } else { vec![] },
            error: Some(error),
            numeric_trap,
        }
    }

    pub(crate) fn emit(self) -> Result<(), Box<dyn std::error::Error>> {
        self.emit_effects()?;
        match self.error {
            Some(error) if self.numeric_trap => Err(Box::new(NumericTrapCliError(error))),
            Some(error) => Err(error.into()),
            None => Ok(()),
        }
    }

    pub(crate) fn emit_effects(&self) -> io::Result<()> {
        self.write_to(&mut io::stdout().lock(), &mut io::stderr().lock())
    }

    fn write_to(&self, stdout: &mut dyn Write, stderr: &mut dyn Write) -> io::Result<()> {
        if let Some(rendered) = &self.stdout {
            writeln!(stdout, "{rendered}")?;
        }
        stdout.flush()?;
        for line in &self.stderr {
            writeln!(stderr, "{line}")?;
        }
        stderr.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_compiler_api::schema::{
        DictEntryValue, EXECUTION_VALUE_SCHEMA_VERSION, EvaluatedRoot, ExecutionValue,
        RootManifestResult,
    };
    use std::process::Command;

    const SERIALIZE_CHILD_CASE_ENV: &str = "CHELIS_DEEP_EXECUTION_VALUE_SERIALIZE_CHILD";

    fn wrap(case: &str, value: ExecutionValue) -> ExecutionValue {
        match case {
            "list" => ExecutionValue::List { value: vec![value] },
            "tuple" => ExecutionValue::Tuple { value: vec![value] },
            "adt" => ExecutionValue::Adt {
                ctor: "Link".to_owned(),
                fields: vec![value],
            },
            "dict" => ExecutionValue::Dict {
                entries: vec![DictEntryValue {
                    key: ExecutionValue::String {
                        value: "k".to_owned(),
                    },
                    value,
                }],
            },
            other => panic!("unknown container {other}"),
        }
    }

    fn result(value: ExecutionValue) -> EvalResult {
        EvalResult {
            schema_version: EXECUTION_VALUE_SCHEMA_VERSION,
            roots: vec![EvaluatedRoot {
                node_id: 0,
                name: Some("a".to_owned()),
                value,
                display: None,
            }],
            manifest: RootManifestResult::default(),
            transcript: Vec::new(),
        }
    }

    #[test]
    fn shallow_eval_json_keeps_the_derived_wire_shape() {
        for case in ["list", "tuple", "adt", "dict"] {
            let value = result(wrap(case, ExecutionValue::Unit));
            assert_eq!(
                serialize_eval_json(&value).expect("serialize with grown stack"),
                serde_json::to_string(&value).expect("serialize shallow result"),
                "{case} wire spelling changed"
            );
        }
    }

    #[test]
    fn serialization_write_failure_is_reported() {
        struct RefuseWrite;
        impl Write for RefuseWrite {
            fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("writer refused value"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let value = result(wrap("adt", ExecutionValue::Unit));
        let error =
            serialize_eval_json_into(&value, RefuseWrite).expect_err("writer failure propagates");
        assert!(error.is_io(), "unexpected error: {error}");
    }

    #[test]
    fn deep_serialize_child() {
        let Ok(case) = std::env::var(SERIALIZE_CHILD_CASE_ENV) else {
            return;
        };
        let mut value = ExecutionValue::Unit;
        for _ in 0..5_000 {
            value = wrap(&case, value);
        }
        let value = result(value);
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let json = serialize_eval_json(&value).expect("serialize deep result");
                assert!(json.contains(&format!(
                    "\"schema_version\":{EXECUTION_VALUE_SCHEMA_VERSION}"
                )));
                assert!(json.contains("\"type\":\"unit\""));
                assert!(json.len() > 5_000);
            })
            .expect("spawn small-stack serialization thread")
            .join()
            .expect("serialization complete");
    }

    #[test]
    fn every_result_container_serializes_on_a_small_stack() {
        for case in ["list", "tuple", "adt", "dict"] {
            let output = Command::new(std::env::current_exe().expect("test binary"))
                .args([
                    "--exact",
                    "eval_output::tests::deep_serialize_child",
                    "--nocapture",
                ])
                .env(SERIALIZE_CHILD_CASE_ENV, case)
                .output()
                .unwrap_or_else(|error| panic!("run {case} child: {error}"));
            assert!(
                output.status.success(),
                "{case} serialization aborted: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn failures_preserve_complete_lines_on_the_selected_channel() {
        for json in [false, true] {
            let output = EvalOutput::failure(
                vec!["first".into(), "second\ncontinued".into()],
                "stop".into(),
                json,
            );
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            output.write_to(&mut stdout, &mut stderr).unwrap();
            assert_eq!(
                if json { &stderr } else { &stdout },
                b"first\nsecond\ncontinued\n"
            );
            assert!(if json { &stdout } else { &stderr }.is_empty());
            assert_eq!(output.error.as_deref(), Some("stop"));
        }
    }

    #[test]
    fn failure_without_effects_does_not_fabricate_output() {
        for json in [false, true] {
            let output = EvalOutput::failure(vec![], "stop".into(), json);
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            output.write_to(&mut stdout, &mut stderr).unwrap();
            assert!(stdout.is_empty() && stderr.is_empty());
        }
    }

    #[test]
    fn successful_output_keeps_its_existing_text_or_json_shape() {
        for (output, expected_out, expected_err) in [
            (
                EvalOutput::text("first\nout = 7".into()),
                "first\nout = 7\n",
                "",
            ),
            (
                EvalOutput::json("{\"roots\":[]}".into()),
                "{\"roots\":[]}\n",
                "",
            ),
            (
                EvalOutput::text(String::new()),
                "",
                "warning: input contains only def declarations; nothing to evaluate\n",
            ),
        ] {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            output.write_to(&mut stdout, &mut stderr).unwrap();
            assert_eq!(stdout, expected_out.as_bytes());
            assert_eq!(stderr, expected_err.as_bytes());
            assert!(output.error.is_none());
        }
    }
}
