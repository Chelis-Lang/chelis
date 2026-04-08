use std::collections::HashMap;

use chelis_deep::ast::Expr as DeepExpr;
use chelis_ir::dag::Dag;
use chelis_ir::dag::NodeId;
use chelis_surf::ast::Decl;

pub struct PipelineResult {
    pub dag: Dag,
    pub deep_text: String, // for debugging
    pub root_nodes: HashMap<String, NodeId>,
}

/// Parse Surf source through the full pipeline:
/// parse -> desugar -> typecheck -> effect check -> linearity check -> lower -> DAG
pub fn compile_surf(source: &str) -> Result<PipelineResult, String> {
    // 1. Parse Surf
    let decls =
        chelis_surf::parser::parse_str(source).map_err(|e| format!("Surf parse error: {e}"))?;

    // 2. Desugar to Deep
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let deep_exprs =
        chelis_macros::expand_program(&deep_exprs, &chelis_macros::ExpansionOptions::default())
            .map_err(|e| format!("Macro expansion error: {e}"))?
            .into_exprs();
    let deep_text = chelis_deep::printer::print_canonical(&deep_exprs);

    // 3. Type check (Phase 0e)
    let checked = chelis_types::check_phase0e_program(&deep_exprs)
        .map_err(|r| format!("Type errors: {:?}", r.errors))?;
    let lowered_names = lowered_root_names_from_decls(&decls, checked.type_env());
    let checked = chelis_effects::check_program(&checked).map_err(|errors| {
        errors
            .into_iter()
            .map(|e| e.message)
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    let checked = chelis_types::check_linearity(&checked).map_err(|errors| {
        errors
            .into_iter()
            .map(|e| e.message)
            .collect::<Vec<_>>()
            .join("; ")
    })?;

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

fn lowered_root_names_from_decls(
    decls: &[Decl],
    type_env: &HashMap<String, DeepExpr>,
) -> Vec<String> {
    let mut names = Vec::new();
    for decl in decls {
        collect_decl_root_names(decl, type_env, &mut names);
    }
    names
}

fn collect_decl_root_names(
    decl: &Decl,
    type_env: &HashMap<String, DeepExpr>,
    out: &mut Vec<String>,
) {
    match decl {
        Decl::FunDef { name, .. } | Decl::LetDef { name, .. } => {
            extend_root_names(name, type_env.get(name), out)
        }
        Decl::Module { decls, .. } => {
            for decl in decls {
                collect_decl_root_names(decl, type_env, out);
            }
        }
        _ => {}
    }
}

fn extend_root_names(name: &str, ty: Option<&DeepExpr>, out: &mut Vec<String>) {
    if let Some(DeepExpr::List(list, _)) = ty
        && let Some(DeepExpr::Atom(chelis_deep::ast::Atom::Symbol(tag), _)) = list.elements.first()
    {
        if tag == "t-fn" {
            extend_root_names(name, list.elements.last(), out);
            return;
        }
        if tag == "t-tuple" {
            for (index, child) in list.elements.iter().skip(2).enumerate() {
                extend_root_names(&format!("{name}.{index}"), Some(child), out);
            }
            return;
        }
    }
    out.push(name.to_string());
}
