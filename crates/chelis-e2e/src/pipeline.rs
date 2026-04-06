use std::collections::HashMap;

use chelis_ir::dag::Dag;
use chelis_ir::dag::NodeId;
use chelis_surf::ast::Decl;

pub struct PipelineResult {
    pub dag: Dag,
    pub deep_text: String, // for debugging
    pub root_nodes: HashMap<String, NodeId>,
}

/// Parse Surf source through the full pipeline: parse -> desugar -> typecheck -> lower -> DAG
pub fn compile_surf(source: &str) -> Result<PipelineResult, String> {
    // 1. Parse Surf
    let decls =
        chelis_surf::parser::parse_str(source).map_err(|e| format!("Surf parse error: {e}"))?;
    let lowered_names = lowered_decl_names(&decls);

    // 2. Desugar to Deep
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let deep_text = chelis_deep::printer::print_canonical(&deep_exprs);

    // 3. Type check (Phase 0e)
    let checked = chelis_types::check_phase0e_program(&deep_exprs)
        .map_err(|r| format!("Type errors: {:?}", r.errors))?;

    // 4. Lower to RISC DAG
    let dag = chelis_ir::lower::lower_program(&checked);

    if dag.roots().len() != lowered_names.len() {
        return Err(format!(
            "lowered root count mismatch: expected {} named roots, got {}",
            lowered_names.len(),
            dag.roots().len()
        ));
    }

    let root_nodes = lowered_names
        .into_iter()
        .zip(dag.roots().iter().copied())
        .collect();

    Ok(PipelineResult {
        dag,
        deep_text,
        root_nodes,
    })
}

fn lowered_decl_names(decls: &[Decl]) -> Vec<String> {
    decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::FunDef { name, .. } | Decl::LetDef { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}
