//! Analysis helpers over lowered RISC DAGs.

use std::collections::{HashSet, VecDeque};

use chelis_types::types::Prim;
use serde::{Deserialize, Serialize};

use crate::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCopyCost {
    pub name: String,
    pub copy_count: usize,
    pub bytes_copied: Option<usize>,
    #[serde(skip)]
    pub byte_formula: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CopyCostSummary {
    pub functions: Vec<FunctionCopyCost>,
    pub total_copy_count: usize,
    pub total_bytes_copied: Option<usize>,
    #[serde(skip)]
    pub total_byte_formula: Option<String>,
}

pub fn analyze_copy_costs(dag: &Dag, roots: &[(String, NodeId)]) -> CopyCostSummary {
    let root_sets = roots
        .iter()
        .map(|(name, root)| (name.clone(), vec![*root]))
        .collect::<Vec<_>>();
    analyze_copy_costs_for_roots(dag, &root_sets)
}

pub fn analyze_copy_costs_for_roots(dag: &Dag, roots: &[(String, Vec<NodeId>)]) -> CopyCostSummary {
    let functions = roots
        .iter()
        .map(|(name, roots)| analyze_function_copy_cost_for_roots(dag, name, roots))
        .collect::<Vec<_>>();
    let total_copy_count = functions.iter().map(|function| function.copy_count).sum();
    let total_bytes_copied = functions.iter().try_fold(0usize, |acc, function| {
        function.bytes_copied.map(|bytes| acc.saturating_add(bytes))
    });
    let total_byte_formula = summarize_formula_terms(
        functions
            .iter()
            .filter_map(|function| function.byte_formula.clone())
            .collect(),
    );
    CopyCostSummary {
        functions,
        total_copy_count,
        total_bytes_copied,
        total_byte_formula,
    }
}

pub fn analyze_function_copy_cost(dag: &Dag, name: &str, root: NodeId) -> FunctionCopyCost {
    analyze_function_copy_cost_for_roots(dag, name, &[root])
}

pub fn analyze_function_copy_cost_for_roots(
    dag: &Dag,
    name: &str,
    roots: &[NodeId],
) -> FunctionCopyCost {
    let mut copy_count = 0usize;
    let mut concrete_total = 0usize;
    let mut all_concrete = true;
    let mut formula_terms = Vec::new();

    for id in reachable_nodes(dag, roots) {
        let Some(node) = dag.get(id) else {
            continue;
        };
        if !matches!(node.op, RiscOp::Copy) {
            continue;
        }
        copy_count += 1;
        match tensor_bytes(&node.output_type) {
            CopyBytes::Concrete(bytes) => {
                concrete_total = concrete_total.saturating_add(bytes);
                formula_terms.push(bytes.to_string());
            }
            CopyBytes::Formula(formula) => {
                all_concrete = false;
                formula_terms.push(formula);
            }
        }
    }

    FunctionCopyCost {
        name: name.to_string(),
        copy_count,
        bytes_copied: all_concrete.then_some(concrete_total),
        byte_formula: summarize_formula_terms(formula_terms),
    }
}

fn reachable_nodes(dag: &Dag, roots: &[NodeId]) -> Vec<NodeId> {
    let mut seen = HashSet::new();
    let mut stack = roots.iter().copied().collect::<VecDeque<_>>();
    let mut out = Vec::new();
    while let Some(id) = stack.pop_back() {
        if !seen.insert(id) {
            continue;
        }
        out.push(id);
        if let Some(node) = dag.get(id) {
            for input in &node.inputs {
                stack.push_back(*input);
            }
        }
    }
    out
}

enum CopyBytes {
    Concrete(usize),
    Formula(String),
}

fn tensor_bytes(ty: &TensorType) -> CopyBytes {
    let element_size = element_size_bytes(ty.precision);
    let mut concrete_elements = 1usize;
    let mut formula_factors = Vec::new();
    for dim in &ty.dims {
        match dim {
            DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => {
                concrete_elements = concrete_elements.saturating_mul(*size);
                formula_factors.push(size.to_string());
            }
            DimInfo::Named(name, None) => {
                formula_factors.push(name.clone());
            }
        }
    }

    if ty.dims.iter().all(dim_is_concrete) {
        return CopyBytes::Concrete(concrete_elements.saturating_mul(element_size));
    }

    if element_size != 1 || formula_factors.is_empty() {
        formula_factors.push(element_size.to_string());
    }
    CopyBytes::Formula(formula_factors.join(" * "))
}

fn dim_is_concrete(dim: &DimInfo) -> bool {
    matches!(dim, DimInfo::Lit(_) | DimInfo::Named(_, Some(_)))
}

fn element_size_bytes(prim: Prim) -> usize {
    match prim {
        Prim::F64 | Prim::Int64 => 8,
        Prim::F32 | Prim::Int32 => 4,
        Prim::F16 | Prim::Bf16 | Prim::Int16 => 2,
        Prim::Int8 | Prim::Bool => 1,
        // E2 (WS-A0 RT-1 fixup): per spec/04-type-system.md §1.1.1
        // f8e4m3 is deferred and the type checker rejects it
        // upstream. Computing a memory-cost classification for a
        // dtype that has no admitted backend would silently entrench
        // the deferred classification — panic rather than report 1.
        Prim::F8e4m3 => panic!(
            "f8e4m3 is deferred per spec/04-type-system.md §1.1.1 and \
             should have been rejected upstream"
        ),
        Prim::String => 8,
    }
}

fn summarize_formula_terms(terms: Vec<String>) -> Option<String> {
    let terms = terms
        .into_iter()
        .filter(|term| term != "0")
        .collect::<Vec<_>>();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" + "))
    }
}

#[cfg(test)]
mod tests {
    use chelis_types::types::Prim;

    use super::*;
    use crate::dag::{Dag, DimInfo, RiscOp, TensorType};

    fn tensor(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
        TensorType { dims, precision }
    }

    #[test]
    fn counts_only_copy_nodes_and_sums_concrete_bytes() {
        let mut dag = Dag::new();
        let ty = tensor(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32);
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let copy = dag.add_node(RiscOp::Copy, vec![x], ty.clone(), None);
        let out = dag.add_node(RiscOp::Neg, vec![copy], ty.clone(), None);
        dag.add_node(RiscOp::Drop, vec![x], ty, None);
        dag.add_root(out);

        let summary = analyze_copy_costs(&dag, &[("main".to_string(), out)]);

        assert_eq!(summary.total_copy_count, 1);
        assert_eq!(summary.total_bytes_copied, Some(24));
        assert_eq!(summary.functions[0].copy_count, 1);
        assert_eq!(summary.functions[0].bytes_copied, Some(24));
    }

    /// E2 (WS-A0 RT-1 fixup): `element_size_bytes` must panic on
    /// f8e4m3 with the §1.1.1 message rather than silently
    /// classifying it as a 1-byte dtype.
    #[test]
    #[should_panic(expected = "f8e4m3 is deferred per spec/04-type-system.md §1.1.1")]
    fn element_size_bytes_panics_on_f8e4m3_per_spec_1_1_1() {
        let _ = element_size_bytes(Prim::F8e4m3);
    }

    #[test]
    fn omits_bytes_when_copy_shape_has_symbolic_dimension() {
        let mut dag = Dag::new();
        let ty = tensor(
            vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(4)],
            Prim::F32,
        );
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let out = dag.add_node(RiscOp::Copy, vec![x], ty, None);
        dag.add_root(out);

        let summary = analyze_copy_costs(&dag, &[("main".to_string(), out)]);

        assert_eq!(summary.total_copy_count, 1);
        assert_eq!(summary.total_bytes_copied, None);
        assert_eq!(summary.functions[0].bytes_copied, None);
        assert_eq!(
            summary.functions[0].byte_formula.as_deref(),
            Some("batch * 4 * 4")
        );
    }
}
