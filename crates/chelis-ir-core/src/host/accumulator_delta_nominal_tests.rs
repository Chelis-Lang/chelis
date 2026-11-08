#![cfg(test)]

use super::*;

#[test]
fn named_nominal_proof_uses_the_exact_normalized_formal_kind() {
    fn find_fn(expr: &Expr) -> Option<&Expr> {
        if let Expr::MetaExpr(meta, _) = expr {
            return find_fn(&meta.expr);
        }
        let node = as_node(expr)?;
        if node.tag() == DeepTag::Fn {
            return Some(expr);
        }
        node.children_slice().iter().find_map(find_fn)
    }
    let source = "type Chain[n] =\n | End { v: tensor[n, f32] }\n | Link { v: tensor[n, f32], next: Chain[n] }\ndef step[n](acc: Chain[n], i: i64) -> Chain[n] = Link { v: insert(scalar_to_tensor(0.0f32), 0i32, add(i, 3i64)), next: acc }\n";
    let declarations =
        chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    let checked = chelis_types::check_ir_program(&deep).unwrap();
    let proof = crate::test_support::with_operation(|operation| {
        accumulator_construction_proof(
            checked.facts().exprs().iter().find_map(find_fn).unwrap(),
            checked.facts().adt_registry(),
            operation,
        )
        .unwrap()
    })
    .expect("named nominal construction has inherited evidence");
    let nominal = |arg| HostTypeTerm::Adt("Chain".into(), vec![arg]);
    let dim = |dim| HostTypeTerm::Dimension(crate::host_type_state::HostShapeSlot::Dim(dim));
    assert!(proof.validates_formal_type(&nominal(dim(DimInfo::Named("n".into(), None)))));
    for other in [
        HostTypeTerm::TypeVariable("n".into()),
        dim(DimInfo::Named("caller".into(), None)),
        dim(DimInfo::Lit(3)),
        HostTypeTerm::Unit,
    ] {
        assert!(
            !proof.validates_formal_type(&nominal(other)),
            "physical layout or equal extent is not the proof's logical obligation"
        );
    }
}

#[test]
fn recursive_delta_reuses_only_the_identical_root_obligation() {
    crate::test_support::with_operation(|operation| {
        fn find_fn(expr: &Expr) -> Option<&Expr> {
            if let Expr::MetaExpr(meta, _) = expr {
                return find_fn(&meta.expr);
            }
            let node = as_node(expr)?;
            if node.tag() == DeepTag::Fn {
                return Some(expr);
            }
            node.children_slice().iter().find_map(find_fn)
        }
        let source = "type Chain[n] =\n | End { v: tensor[n, f32] }\n | Link { v: tensor[n, f32], next: Chain[n] }\nout = fold(fn (acc: Chain[3], i: i64) -> Link { v: to_tensor([1.0f32, 2.0f32, 3.0f32]), next: acc }, End { v: to_tensor([1.0f32, 2.0f32, 3.0f32]) }, range(0i64, 2i64))\n";
        let declarations =
            chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
        let deep =
            chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
                .unwrap();
        let checked = chelis_types::check_ir_program(&deep).unwrap();
        let proof = accumulator_delta_proof(
            checked.facts().exprs().iter().find_map(find_fn).unwrap(),
            checked.facts().adt_registry(),
            operation,
        )
        .unwrap()
        .expect("recursive nominal component has identical obligation");
        assert!(
            matches!(proof.recipe(), AccumulatorDeltaRecipe::Constructor { fields, .. } if fields == &[AccumulatorDeltaRecipe::Fresh, AccumulatorDeltaRecipe::Inherited])
        );
        let field = |name: &str| HostExpr::new(HostExprKind::Var(name.into(), HostTypeTerm::Unit));
        let body = |tail: &str| {
            HostExpr::new(HostExprKind::AdtConstruct {
                ctor: "Link".into(),
                fields: vec![field("new_tensor"), field(tail)],
                ty: HostTypeTerm::Adt("Chain".into(), Vec::new()),
            })
        };
        assert!(proof.validates_body(&body("acc")));
        assert!(
            !proof.validates_body(&body("other")),
            "same type/shape is insufficient evidence"
        );
        let mut other = body("acc");
        if let HostExprKind::AdtConstruct { ty, .. } = &mut other.kind {
            *ty = HostTypeTerm::Adt("Other".into(), Vec::new());
        }
        assert!(
            !proof.validates_body(&other),
            "another nominal owner cannot reuse the constructor spelling"
        );
    });
}
