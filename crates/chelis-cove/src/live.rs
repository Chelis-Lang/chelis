use std::collections::BTreeMap;
use std::fmt::Write as _;

use chelis_tide::compiler;
use chelis_tide::schema::{
    CheckRequest, CompileRequest, CompileTarget, DesugarRequest, Diagnostic, EvalRequest,
    ExecutionValue, LowerRequest, SourceKind, TensorValue, WireDimInfo, WireRiscOp,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveStage {
    Ready,
    Parse,
    Check,
}

#[derive(Debug, Clone)]
pub struct LiveAnalysis {
    pub stage: LiveStage,
    pub fitness: Option<f64>,
    pub diagnostics: Vec<Diagnostic>,
    pub deep_text: Option<String>,
}

impl LiveAnalysis {
    pub fn status_text(&self) -> String {
        match self.fitness {
            Some(score) => format!("Chelis: {score:.2}"),
            None => "Chelis: unavailable".to_string(),
        }
    }
}

pub fn analyze(source: &str) -> LiveAnalysis {
    let deep_text = match compiler::desugar(DesugarRequest {
        source: source.to_string(),
    })
    .map(|result| result.deep_text)
    {
        Ok(deep_text) => deep_text,
        Err(err) => {
            return LiveAnalysis {
                stage: LiveStage::Parse,
                fitness: None,
                diagnostics: err.errors,
                deep_text: None,
            };
        }
    };

    match compiler::check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
    }) {
        Ok(result) => LiveAnalysis {
            stage: if result.errors.is_empty() {
                LiveStage::Ready
            } else {
                LiveStage::Check
            },
            fitness: Some(result.score),
            diagnostics: result.errors,
            deep_text: Some(deep_text),
        },
        Err(err) => LiveAnalysis {
            stage: if err.stage == "parse" {
                LiveStage::Parse
            } else {
                LiveStage::Check
            },
            fitness: None,
            diagnostics: err.errors,
            deep_text: Some(deep_text),
        },
    }
}

pub fn compile_output(source: &str) -> String {
    match compiler::compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    }) {
        Ok(result) => {
            let mut out = String::new();
            let _ = writeln!(&mut out, "compile target: {:?}", result.target);
            let _ = writeln!(&mut out, "entry: {}", result.entry_name);
            let _ = writeln!(&mut out, "generated files:");
            for file in &result.files {
                let _ = writeln!(&mut out, "  {}", file.path);
            }
            if !result.compile_flags.is_empty() {
                let _ = writeln!(
                    &mut out,
                    "compile flags: {}",
                    result.compile_flags.join(" ")
                );
            }
            if !result.link_flags.is_empty() {
                let _ = writeln!(&mut out, "link flags: {}", result.link_flags.join(" "));
            }
            out
        }
        Err(err) => format_failure("compile", &err.errors),
    }
}

pub fn eval_output(source: &str) -> String {
    let bindings = match zero_bindings(source) {
        Ok(bindings) => bindings,
        Err(message) => return message,
    };
    match compiler::eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings,
    }) {
        Ok(result) => {
            let mut out = String::new();
            out.push_str("eval: zero-filled named bindings\n");
            for root in result.roots {
                let name = root
                    .name
                    .unwrap_or_else(|| format!("node_{}", root.node_id));
                let _ = writeln!(&mut out, "{name}: {}", render_execution_value(&root.value));
            }
            if out.is_empty() {
                "eval: no roots".to_string()
            } else {
                out
            }
        }
        Err(err) => format_failure("eval", &err.errors),
    }
}

pub fn diagnostics_text(analysis: &LiveAnalysis) -> String {
    let mut out = String::new();
    let stage = match analysis.stage {
        LiveStage::Ready => "ready",
        LiveStage::Parse => "parse",
        LiveStage::Check => "check",
    };
    let _ = writeln!(&mut out, "stage: {stage}");
    if let Some(score) = analysis.fitness {
        let _ = writeln!(&mut out, "fitness: {score:.2}");
    } else {
        let _ = writeln!(&mut out, "fitness: unavailable");
    }
    if analysis.diagnostics.is_empty() {
        out.push_str("diagnostics: none");
        return out;
    }

    out.push_str("diagnostics:\n");
    for diagnostic in &analysis.diagnostics {
        let _ = writeln!(&mut out, "- {}", diagnostic.message);
        if let Some(expected) = &diagnostic.expected {
            let _ = writeln!(&mut out, "  expected: {expected}");
        }
        if let Some(got) = &diagnostic.got {
            let _ = writeln!(&mut out, "  got: {got}");
        }
    }
    out
}

fn render_execution_value(value: &ExecutionValue) -> String {
    match value {
        ExecutionValue::Tensor { value } => {
            format!("shape={:?} data={:?}", value.shape, value.data)
        }
        ExecutionValue::Int64 { value } => value.to_string(),
        ExecutionValue::Float64 { value } => value.to_string(),
        ExecutionValue::Bool { value } => value.to_string(),
        ExecutionValue::String { value } => value.clone(),
        ExecutionValue::List { value: items } => format!(
            "[{}]",
            items
                .iter()
                .map(render_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Dict { entries } => format!(
            "dict({})",
            entries
                .iter()
                .map(|entry| format!(
                    "{}: {}",
                    render_execution_value(&entry.key),
                    render_execution_value(&entry.value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Tuple { value: items } => format!(
            "({})",
            items
                .iter()
                .map(render_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Adt { ctor, fields } if fields.is_empty() => ctor.clone(),
        ExecutionValue::Adt { ctor, fields } => format!(
            "{}({})",
            ctor,
            fields
                .iter()
                .map(render_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Unit => "()".to_string(),
    }
}

fn format_failure(stage: &str, diagnostics: &[Diagnostic]) -> String {
    let mut out = format!("{stage} failed\n");
    for diagnostic in diagnostics {
        let _ = writeln!(&mut out, "- {}", diagnostic.message);
    }
    out
}

fn zero_bindings(source: &str) -> Result<BTreeMap<String, TensorValue>, String> {
    let lower = compiler::lower(LowerRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
    })
    .map_err(|err| format_failure("eval", &err.errors))?;

    let mut bindings = BTreeMap::new();
    for node in lower.dag.nodes {
        let WireRiscOp::Load { name } = node.op else {
            continue;
        };
        let mut shape = Vec::new();
        for dim in node.output_type.dims {
            match dim {
                WireDimInfo::Lit { size } => shape.push(size),
                WireDimInfo::Named {
                    size: Some(size), ..
                } => shape.push(size),
                WireDimInfo::Named { name, size: None } => {
                    return Err(format!(
                        "eval failed\n- cannot auto-evaluate unresolved named dimension `{name}`"
                    ));
                }
            }
        }
        let len = shape.iter().product();
        bindings.insert(
            name,
            TensorValue {
                shape,
                data: vec![0.0; len],
            },
        );
    }

    Ok(bindings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_program_reports_fitness_and_deep() {
        let analysis = analyze("def f(x: tensor[n, f32]): tensor[n, f32] = x\n");
        assert_eq!(analysis.stage, LiveStage::Ready);
        assert_eq!(analysis.fitness, Some(1.0));
        assert!(
            analysis
                .deep_text
                .as_deref()
                .unwrap_or_default()
                .contains("(def")
        );
    }

    #[test]
    fn parse_error_is_reported_without_fitness() {
        let analysis = analyze("def f( = 1\n");
        assert_eq!(analysis.stage, LiveStage::Parse);
        assert_eq!(analysis.fitness, None);
        assert!(!analysis.diagnostics.is_empty());
    }

    #[test]
    fn type_error_is_reported_without_panicking() {
        let analysis = analyze("def f() = 1 + true\n");
        assert_eq!(analysis.stage, LiveStage::Check);
        assert!(analysis.fitness.is_some());
        assert!(!analysis.diagnostics.is_empty());
    }

    #[test]
    fn zero_eval_populates_named_loads() {
        let output =
            eval_output("x = (x : tensor[2, 3, f32])\ny = (relu(x) : tensor[2, 3, f32])\n");
        assert!(output.contains("zero-filled named bindings"));
        assert!(output.contains("shape=[2, 3]"));
    }
}
