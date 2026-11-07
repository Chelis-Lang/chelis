use super::*;
#[test]
fn unrepresented_rank_and_precision_witnesses_retain_full_entry_checks() {
    fn find_fn(expr: &Expr) -> Option<&Expr> {
        if let Expr::MetaExpr(meta, _) = expr {
            return find_fn(&meta.expr);
        }
        let node = as_node(expr)?;
        if node.tag() == DeepTag::Fn
            && node
                .children_slice()
                .first()
                .and_then(as_node)
                .is_some_and(|params| {
                    params
                        .children_slice()
                        .first()
                        .and_then(param_name)
                        .as_deref()
                        == Some("acc")
                })
        {
            return Some(expr);
        }
        node.children_slice().iter().find_map(find_fn)
    }
    for (binders, element) in [
        ("n, p: Float", "tensor[n, p]"),
        ("s, n", "tensor[..s, n, f32]"),
    ] {
        let source = format!(
            "def run[{binders}](seed: List[{element}], steps: i64) -> i64 = len(fold(fn (acc: List[{element}], i: i64) -> append(acc, index(seed, 0i64)), seed, range(0i64, steps)))\n"
        );
        let declarations =
            chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack).unwrap();
        let deep =
            chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
                .unwrap();
        let checked = match chelis_types::check_ir_program(&deep) {
            Ok(checked) => checked,
            Err(errors) if element.contains("..") => {
                assert!(
                    errors
                        .errors
                        .iter()
                        .any(|error| error.message.contains("rank-polymorphic"))
                );
                continue;
            }
            Err(errors) => panic!("{errors:?}"),
        };
        crate::test_support::with_operation(|operation| {
            assert!(
                accumulator_delta_proof(
                    checked.facts().exprs().iter().find_map(find_fn).unwrap(),
                    checked.facts().adt_registry(),
                    operation,
                )
                .unwrap()
                .is_none(),
                "{source}"
            );
            assert!(
                accumulator_construction_proof(
                    checked.facts().exprs().iter().find_map(find_fn).unwrap(),
                    checked.facts().adt_registry(),
                    operation,
                )
                .unwrap()
                .is_none(),
                "{source}"
            );
        });
    }
}
