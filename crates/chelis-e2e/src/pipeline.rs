use chelis_ir::dag::Dag;

pub struct PipelineResult {
    pub dag: Dag,
    pub deep_text: String, // for debugging
}

/// Parse Surf source through the full pipeline: parse -> desugar -> typecheck -> lower -> DAG
pub fn compile_surf(source: &str) -> Result<PipelineResult, String> {
    // 1. Parse Surf
    let decls =
        chelis_surf::parser::parse_str(source).map_err(|e| format!("Surf parse error: {e}"))?;

    // 2. Desugar to Deep
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let deep_text = chelis_deep::printer::print_canonical(&deep_exprs);

    // 3. Type check (Phase 0e)
    let checked = chelis_types::check_phase0e_program(&deep_exprs)
        .map_err(|r| format!("Type errors: {:?}", r.errors))?;

    // 4. Lower to RISC DAG
    let dag = chelis_ir::lower::lower_program(&checked);

    Ok(PipelineResult { dag, deep_text })
}
