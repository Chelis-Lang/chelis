use std::collections::HashMap;

use chelis_ir::dag::{Dag, NodeId};

pub struct PipelineResult {
    pub dag: Dag,
    pub deep_text: String,
    pub root_nodes: HashMap<String, NodeId>,
}

/// Parse Surf source through the canonical compiler-API pipeline.
pub fn compile_surf(source: &str) -> Result<PipelineResult, String> {
    let outcome =
        chelis_compiler_api::pipeline::run_source(chelis_compiler_api::pipeline::PipelineRequest {
            source_kind: chelis_compiler_api::schema::SourceKind::Surf,
            source,
            entry: None,
            goal: chelis_compiler_api::pipeline::PipelineGoal::Lower(
                chelis_compiler_api::pipeline::LoweringMode::Strict,
            ),
        })
        .map_err(e2e_pipeline_error)?;
    let chelis_compiler_api::pipeline::PipelineOutcome::Lowered(lowered) = outcome else {
        unreachable!("the lower goal returns only a lowered outcome")
    };
    let parts = lowered.into_parts();
    let deep_text = chelis_deep::printer::print_canonical(parts.checked.expanded_deep());
    let root_nodes = parts
        .named_roots
        .into_entries()
        .map(|(name, node)| (name.into_string(), node))
        .collect();

    Ok(PipelineResult {
        dag: parts.dag,
        deep_text,
        root_nodes,
    })
}

fn e2e_pipeline_error(rejection: chelis_compiler_api::pipeline::PipelineRejection) -> String {
    use chelis_compiler_api::pipeline::{PipelineRejection, PreparationError};

    match rejection {
        PipelineRejection::Preparation(PreparationError::SurfParse { error, .. }) => {
            format!("Surf parse error: {error}")
        }
        PipelineRejection::Preparation(PreparationError::Expansion(error)) => {
            format!("Macro expansion error: {error}")
        }
        PipelineRejection::Preparation(PreparationError::DeepParse(error)) => {
            format!("Deep parse error: {error}")
        }
        other => other.to_string(),
    }
}
