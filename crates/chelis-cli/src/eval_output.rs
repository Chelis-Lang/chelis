//! Prepare evaluation output before deciding which terminal path may emit it.
use std::io::{self, Write};

pub(crate) struct EvalOutput {
    stdout: Option<String>,
    stderr: Vec<String>,
    pub(crate) error: Option<String>,
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
            }
        } else {
            Self {
                stdout: Some(rendered),
                stderr: vec![],
                error: None,
            }
        }
    }

    pub(crate) fn json(rendered: String) -> Self {
        Self {
            stdout: Some(rendered),
            stderr: vec![],
            error: None,
        }
    }

    pub(crate) fn failure(transcript: Vec<String>, error: String, json: bool) -> Self {
        Self {
            stdout: (!json && !transcript.is_empty()).then(|| transcript.join("\n")),
            stderr: if json { transcript } else { vec![] },
            error: Some(error),
        }
    }

    pub(crate) fn emit(self) -> Result<(), Box<dyn std::error::Error>> {
        self.emit_effects()?;
        match self.error {
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
