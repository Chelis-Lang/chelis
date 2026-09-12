//! Persistent semantic-label transport, separate from the immutable prototype oracle.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::errors::CheckErrorKind;
use chelis_types::{
    TypeEnv, build_compiled_library_context, build_compiled_library_context_with_base,
    check_ir_program, check_ir_with_context,
};

fn parse(source: &str) -> Vec<chelis_deep::Expr> {
    desugar_program(&parse_str(source).expect("fixture parses"))
}

#[test]
fn concrete_instantiations_and_value_aliases_keep_names_without_coupling_calls() {
    for reverse in [false, true] {
        let operation = if reverse {
            "mul(gain,x)"
        } else {
            "mul(x,gain)"
        };
        let library = format!(
            "def aligned[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> tensor[d,f32] = {operation}"
        );
        let consumer = "def use() = { f = aligned\n g = f\n a = g(to_tensor([1.0f32,2.0f32]),to_tensor([1.0f32,2.0f32]))\n b = g(to_tensor([1.0f32,2.0f32,3.0f32]),to_tensor([1.0f32,2.0f32,3.0f32]))\n (sum(a,fixed),sum(b,fixed)) }";
        check_ir_program(&parse(&format!("{library}\n{consumer}")))
            .expect("whole: independent calls and aliases");
        let (live, _) = build_compiled_library_context(&parse(&library)).unwrap();
        let bytes = bincode::serialize(&live).unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bytes).unwrap();
        for context in [&live, &decoded] {
            check_ir_with_context(context, &parse(consumer))
                .expect("context: independent calls and aliases");
        }
    }
}

/// §4.5.2 joins concrete ragged axes without erasing genuine binder/name
/// uniformity. A label-bearing ID may already have a concrete constraint.
#[test]
fn concrete_labelled_list_join_preserves_ragged_and_uniformity_boundaries() {
    let mut failures = Vec::new();
    for reverse_mul in [false, true] {
        let operation = if reverse_mul {
            "mul(gain,x)"
        } else {
            "mul(x,gain)"
        };
        let library = format!(
            "def aligned[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> tensor[d,f32] = {operation}"
        );
        let (live, _) = build_compiled_library_context(&parse(&library)).unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bincode::serialize(&live).unwrap()).unwrap();
        let extension = "def forwarding[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> tensor[d,f32] = aligned(x,gain)";
        let (layered, _) =
            build_compiled_library_context_with_base(&live, &parse(extension)).unwrap();
        for reverse_list in [false, true] {
            let literal2 = "to_tensor([1.0f32,2.0f32])";
            let literal3 = "to_tensor([1.0f32,2.0f32,3.0f32])";
            let labelled2 = format!("aligned({literal2},{literal2})");
            let labelled3 = format!("aligned({literal3},{literal3})");
            let list = |a: &str, b: &str| {
                if reverse_list {
                    format!("[{b},{a}]")
                } else {
                    format!("[{a},{b}]")
                }
            };
            let mixed = list(&labelled2, literal3);
            let ragged = list(&labelled2, &labelled3);
            let rigid_mixed = list("aligned(x,gain)", literal3);
            let rigid_distinct = list("aligned(x,gain)", "aligned(y,gain)");
            let same_concrete = list(&labelled2, &labelled2);
            let cases = [
                ("mixed concrete", true, format!("def use() = {mixed}")),
                ("independent concrete calls", true, format!("def use() = {ragged}")),
                ("explicit wildcard", true, format!("def use() -> List[tensor[*,f32]] = {ragged}")),
                ("named uniformity", false, format!("def use() -> List[tensor[fixed,f32]] = {ragged}")),
                ("rigid plus literal", false, format!("def use[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> List[tensor[d,f32]] = {rigid_mixed}")),
                ("distinct rigid", false, format!("def use[d,e](x: tensor[d,f32], y: tensor[e,f32], gain: tensor[fixed,f32]) -> List[tensor[d,f32]] = {rigid_distinct}")),
                ("same rigid", true, "def use[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> List[tensor[d,f32]] = [aligned(x,gain),aligned(x,gain)]".into()),
                ("equal labelled query", true, format!("def use() = match {same_concrete} with {{ | Cons(head,tail) => sum(head,fixed) | Nil => scalar_to_tensor(0.0f32) }}")),
                ("distinct named uniformity", false, format!("def use(a: tensor[other,f32], gain: tensor[fixed,f32]) -> List[tensor[fixed,f32]] = {}", list("a", "gain"))),
                ("rank mismatch", false, format!("def use() = {}", list(&labelled2, "to_tensor([[1.0f32,2.0f32]])"))),
            ];
            for (case, expected, source) in cases {
                for route in 0..4 {
                    let result = match route {
                        0 => check_ir_program(&parse(&format!("{library}\n{source}"))),
                        1 => check_ir_with_context(&live, &parse(&source)),
                        2 => check_ir_with_context(&decoded, &parse(&source)),
                        _ => check_ir_with_context(
                            &layered,
                            &parse(&source.replace("aligned(", "forwarding(")),
                        ),
                    };
                    let correct = if expected {
                        result.is_ok()
                    } else {
                        result.as_ref().err().is_some_and(|report| {
                            report.errors.iter().any(|error| {
                                matches!(error.kind, CheckErrorKind::DimensionMismatch)
                            })
                        })
                    };
                    if !correct {
                        failures.push(format!("{case}: reverse_mul={reverse_mul} reverse_list={reverse_list} route={route} expected={expected}: {:?}", result.err().map(|r| r.errors)));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} observations failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
