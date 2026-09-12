//! Persistent semantic-label transport, separate from the immutable prototype oracle.

/// Consumer families, not repeated witnesses of only one operation.
#[test]
fn dimension_observations_cover_arithmetic_comparison_and_propagation() {
    let cases = [
        (
            "diagonal cannot erase authored result identity",
            false,
            "def run[d](x: tensor[d,d,f32], g: tensor[row,row,f32]) -> tensor[17,f32] = diagonal(square(x,g),0i32,1i32)",
        ),
        (
            "stride identity cannot erase authored result identity",
            false,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32]) -> tensor[17,f32] = stride(tag(x,g),1i64)",
        ),
        (
            "pad identity cannot erase authored result identity",
            false,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32]) -> tensor[17,f32] = pad(tag(x,g),[[0i64,0i64]],0.0f32)",
        ),
        (
            "deferred gather retained name",
            true,
            "def apply_it[d,b](f: tensor[d,2,f32] -> b, t: tensor[d,2,f32]) -> b = f(t)\ndef run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int32]) = sum(apply_it(fn (v) -> { r = gather(v,i,-1i32)\n r },tag2(x,g)),row)",
        ),
        (
            "deferred gather false result",
            false,
            "def apply_it[d,b](f: tensor[d,2,f32] -> b, t: tensor[d,2,f32]) -> b = f(t)\ndef run[d,e](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int32]) -> tensor[e,1,f32] = apply_it(fn (v) -> { r = gather(v,i,-1i32)\n r },tag2(x,g))",
        ),
        (
            "deferred trace retained name",
            true,
            "def apply_it[d,b](f: tensor[d,2,2,f32] -> b, t: tensor[d,2,2,f32]) -> b = f(t)\ndef run[d](x: tensor[d,2,2,f32], g: tensor[batch,row,col,f32]) = sum(apply_it(fn (v) -> { r = trace(v,1i32,2i32)\n r },tag3(x,g)),batch)",
        ),
        (
            "deferred trace false result",
            false,
            "def apply_it[d,b](f: tensor[d,2,2,f32] -> b, t: tensor[d,2,2,f32]) -> b = f(t)\ndef run[d,e](x: tensor[d,2,2,f32], g: tensor[batch,row,col,f32]) -> tensor[e,f32] = apply_it(fn (v) -> { r = trace(v,1i32,2i32)\n r },tag3(x,g))",
        ),
        (
            "deferred scatter retained name",
            true,
            "def apply_it[d,b](f: tensor[d,2,f32] -> b, t: tensor[d,2,f32]) -> b = f(t)\ndef run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int32], u: tensor[d,1,f32]) = sum(apply_it(fn (v) -> { r = scatter(v,i,u,-1i32,\"add\")\n r },tag2(x,g)),row)",
        ),
        (
            "deferred scatter bad updates",
            false,
            "def apply_it[d,b](f: tensor[d,2,f32] -> b, t: tensor[d,2,f32]) -> b = f(t)\ndef run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int32], u: tensor[d,3,f32]) = sum(apply_it(fn (v) -> { r = scatter(v,i,u,-1i32,\"add\")\n r },tag2(x,g)),row)",
        ),
        (
            "deferred scatter_replace retained name",
            true,
            "def apply_it[d,b](f: tensor[d,2,f32] -> b, t: tensor[d,2,f32]) -> b = f(t)\ndef run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int32], u: tensor[d,1,f32]) = sum(apply_it(fn (v) -> { r = scatter_replace(v,i,u,-1i32)\n r },tag2(x,g)),row)",
        ),
        (
            "deferred scatter_replace bad updates",
            false,
            "def apply_it[d,b](f: tensor[d,2,f32] -> b, t: tensor[d,2,f32]) -> b = f(t)\ndef run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int32], u: tensor[d,3,f32]) = sum(apply_it(fn (v) -> { r = scatter_replace(v,i,u,-1i32)\n r },tag2(x,g)),row)",
        ),
        (
            "expand bare nonunit",
            false,
            "def run(x: tensor[4,f32]) = expand(x,0i32,3i64)",
        ),
        (
            "reshape bare bad product",
            false,
            "def run(x: tensor[4,f32]) = reshape(x,[3i64])",
        ),
        (
            "concat bare wrong sum",
            false,
            "def run(x: tensor[2,f32], y: tensor[2,f32]) -> tensor[5,f32] = concat([x,y],0i32)",
        ),
        (
            "window unknown runtime claim",
            true,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32]) -> tensor[2,f32] = reduce_window_sum(tag(x,g),[2i64],[1i64])",
        ),
        (
            "stride unknown runtime claim",
            true,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32]) -> tensor[3,f32] = stride(tag(x,g),2i64)",
        ),
        (
            "pad unknown runtime claim",
            true,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32]) -> tensor[6,f32] = pad(tag(x,g),[[0i64,1i64]],0.0f32)",
        ),
        (
            "diagonal bound identity query",
            true,
            "def run(x: tensor[2,2,f32], g: tensor[2,2,f32]) = sum(diagonal(square(x,g),0i32,1i32),row)",
        ),
        (
            "where bound contradiction",
            false,
            "def run(x: tensor[2,f32], g: tensor[2,f32], y: tensor[3,f32], h: tensor[3,f32], c: tensor[row,bool]) = where(c,tag(x,g),tag(y,h))",
        ),
        (
            "where bare literal not a name",
            false,
            "def run(x: tensor[2,f32], g: tensor[2,f32], y: tensor[2,f32], c: tensor[row,bool]) = where(c,tag(x,g),y)",
        ),
        (
            "clamp bound contradiction",
            false,
            "def run(x: tensor[2,f32], g: tensor[2,f32], y: tensor[3,f32], h: tensor[3,f32]) = clamp(tag(x,g),tag(y,h),tag(y,h))",
        ),
        (
            "gather retained name",
            true,
            "def run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int64]) = sum(gather(tag2(x,g),i,1i32),row)",
        ),
        (
            "trace retained name",
            true,
            "def run[d](x: tensor[d,2,2,f32], g: tensor[batch,row,col,f32]) = sum(trace(tag3(x,g),1i32,2i32),batch)",
        ),
        (
            "gather declared identity wrong",
            false,
            "def run[d,e](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int64]) -> tensor[e,1,f32] = gather(tag2(x,g),i,1i32)",
        ),
        (
            "trace declared identity wrong",
            false,
            "def run[d,e](x: tensor[d,2,2,f32], g: tensor[batch,row,col,f32]) -> tensor[e,f32] = trace(tag3(x,g),1i32,2i32)",
        ),
        (
            "unresolved gather lambda at binding rejects",
            false,
            "def run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[1,int32]) = { f = fn (v) -> gather(v,i,1i32)\n sum(f(tag2(x,g)),row) }",
        ),
        (
            "unresolved trace lambda at binding rejects",
            false,
            "def run[d](x: tensor[d,2,2,f32], g: tensor[batch,row,col,f32]) = { f = fn (v) -> trace(v,1i32,2i32)\n sum(f(tag3(x,g)),batch) }",
        ),
        (
            "scatter symbolic",
            true,
            "def run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[row,1,int64], u: tensor[row,1,f32]) -> tensor[d,2,f32] = scatter_elements(tag2(x,g),i,u,1i32)",
        ),
        (
            "scatter different name",
            false,
            "def run[d](x: tensor[d,2,f32], g: tensor[row,col,f32], i: tensor[other,1,int64], u: tensor[other,1,f32]) -> tensor[d,2,f32] = scatter_elements(tag2(x,g),i,u,1i32)",
        ),
        (
            "scatter known fits",
            true,
            "def run(x: tensor[4,2,f32], g: tensor[4,2,f32], i: tensor[3,1,int64], u: tensor[3,1,f32]) = scatter_elements(tag2(x,g),i,u,1i32)",
        ),
        (
            "scatter known overshoot",
            false,
            "def run(x: tensor[4,2,f32], g: tensor[4,2,f32], i: tensor[5,1,int64], u: tensor[5,1,f32]) = scatter_elements(tag2(x,g),i,u,1i32)",
        ),
        (
            "where symbolic",
            true,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32], c: tensor[row,bool]) -> tensor[d,f32] = where(c,tag(x,g),g)",
        ),
        (
            "where distinct name",
            false,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32], c: tensor[other,bool]) = where(c,tag(x,g),g)",
        ),
        (
            "clamp symbolic",
            true,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32]) -> tensor[d,f32] = clamp(tag(x,g),g,g)",
        ),
        (
            "clamp distinct name",
            false,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32], lo: tensor[other,f32]) = clamp(tag(x,g),lo,lo)",
        ),
        (
            "layer norm positive",
            true,
            "def run(x: tensor[2,4,f32], g: tensor[2,4,f32], w: tensor[col,f32]) = layer_norm(tag2(x,g),w,w,0.001f32)",
        ),
        (
            "layer norm zero",
            false,
            "def run(x: tensor[2,0,f32], g: tensor[2,0,f32], w: tensor[col,f32]) = layer_norm(tag2(x,g),w,w,0.001f32)",
        ),
        (
            "expand unit",
            true,
            "def run(x: tensor[1,f32], g: tensor[1,f32]) -> tensor[3,f32] = expand(tag(x,g),0i32,3i64)",
        ),
        (
            "expand named nonunit runtime claim",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) = expand(tag(x,g),0i32,3i64)",
        ),
        (
            "matmul unit batch",
            true,
            "def run(x: tensor[1,2,3,f32], g: tensor[1,2,3,f32], y: tensor[4,3,5,f32]) -> tensor[4,2,5,f32] = matmul(tag3(x,g),y)",
        ),
        (
            "matmul wrong batch",
            false,
            "def run(x: tensor[2,2,3,f32], g: tensor[2,2,3,f32], y: tensor[4,3,5,f32]) = matmul(tag3(x,g),y)",
        ),
        (
            "window extent",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) -> tensor[3,f32] = reduce_window_sum(tag(x,g),[2i64],[1i64])",
        ),
        (
            "window oversize",
            false,
            "def run(x: tensor[2,f32], g: tensor[2,f32]) = reduce_window_sum(tag(x,g),[3i64],[1i64])",
        ),
        (
            "window wrong extent",
            false,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) -> tensor[2,f32] = reduce_window_sum(tag(x,g),[2i64],[1i64])",
        ),
        (
            "shrink fits",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) -> tensor[3,f32] = shrink(tag(x,g),[[0i64,3i64]])",
        ),
        (
            "shrink overshoot",
            false,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) = shrink(tag(x,g),[[0i64,5i64]])",
        ),
        (
            "stride extent",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) -> tensor[2,f32] = stride(tag(x,g),2i64)",
        ),
        (
            "stride wrong extent",
            false,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) -> tensor[3,f32] = stride(tag(x,g),2i64)",
        ),
        (
            "stride identity query",
            true,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32]) = sum(stride(tag(x,g),1i64),row)",
        ),
        (
            "stride bound identity query",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) = sum(stride(tag(x,g),1i64),row)",
        ),
        (
            "pad extent",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) -> tensor[5,f32] = pad(tag(x,g),[[0i64,1i64]],0.0f32)",
        ),
        (
            "pad wrong extent",
            false,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) -> tensor[6,f32] = pad(tag(x,g),[[0i64,1i64]],0.0f32)",
        ),
        (
            "pad identity query",
            true,
            "def run[d](x: tensor[d,f32], g: tensor[row,f32]) = sum(pad(tag(x,g),[[0i64,0i64]],0.0f32),row)",
        ),
        (
            "pad bound identity query",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) = sum(pad(tag(x,g),[[0i64,0i64]],0.0f32),row)",
        ),
        (
            "reshape product",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) -> tensor[2,2,f32] = reshape(tag(x,g),[2i64,2i64])",
        ),
        (
            "reshape named input runtime product claim",
            true,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) = reshape(tag(x,g),[3i64])",
        ),
        (
            "reshape negative",
            false,
            "def run(x: tensor[4,f32], g: tensor[4,f32]) = reshape(tag(x,g),[-1i64])",
        ),
        (
            "concat sum",
            true,
            "def run(x: tensor[2,f32], g: tensor[2,f32]) -> tensor[4,f32] = concat([tag(x,g),tag(x,g)],0i32)",
        ),
        (
            "concat named input unknown sum runtime claim",
            true,
            "def run(x: tensor[2,f32], g: tensor[2,f32]) -> tensor[5,f32] = concat([tag(x,g),tag(x,g)],0i32)",
        ),
        (
            "diagonal selected name query",
            true,
            "def run[d](x: tensor[d,d,f32], g: tensor[row,row,f32]) = sum(diagonal(square(x,g),0i32,1i32),row)",
        ),
        (
            "diagonal minimum",
            true,
            "def run(x: tensor[2,4,f32], g: tensor[2,4,f32]) -> tensor[2,f32] = diagonal(tag2(x,g),0i32,1i32)",
        ),
        (
            "diagonal wrong minimum",
            false,
            "def run(x: tensor[2,4,f32], g: tensor[2,4,f32]) -> tensor[3,f32] = diagonal(tag2(x,g),0i32,1i32)",
        ),
    ];
    let mut failures = Vec::new();
    for reverse in [false, true] {
        let library = "def tag[d](x: tensor[d,f32], g: tensor[row,f32]) -> tensor[d,f32] = MUL\ndef tag2[d,e](x: tensor[d,e,f32], g: tensor[row,col,f32]) -> tensor[d,e,f32] = MUL\ndef tag3[d,e,f](x: tensor[d,e,f,f32], g: tensor[batch,row,col,f32]) -> tensor[d,e,f,f32] = MUL\ndef square[d](x: tensor[d,d,f32], g: tensor[row,row,f32]) -> tensor[d,d,f32] = MUL".replace("MUL", if reverse { "mul(g,x)" } else { "mul(x,g)" });
        let (live, _) = build_compiled_library_context(&parse(&library)).unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bincode::serialize(&live).unwrap()).unwrap();
        let (layered, _) = build_compiled_library_context_with_base(&live, &parse("def forward[d](x: tensor[d,f32], g: tensor[row,f32]) -> tensor[d,f32] = tag(x,g)\ndef forward2[d,e](x: tensor[d,e,f32], g: tensor[row,col,f32]) -> tensor[d,e,f32] = tag2(x,g)\ndef forward3[d,e,f](x: tensor[d,e,f,f32], g: tensor[batch,row,col,f32]) -> tensor[d,e,f,f32] = tag3(x,g)\ndef forward_square[d](x: tensor[d,d,f32], g: tensor[row,row,f32]) -> tensor[d,d,f32] = square(x,g)")).unwrap();
        for (name, expected, source) in cases {
            for route in 0..4 {
                let result = match route {
                    0 => check_ir_program(&parse(&format!("{library}\n{source}"))),
                    1 => check_ir_with_context(&live, &parse(source)),
                    2 => check_ir_with_context(&decoded, &parse(source)),
                    _ => check_ir_with_context(
                        &layered,
                        &parse(
                            &source
                                .replace("tag(", "forward(")
                                .replace("tag2(", "forward2(")
                                .replace("tag3(", "forward3(")
                                .replace("square(", "forward_square("),
                        ),
                    ),
                };
                let correct = if expected {
                    result.is_ok()
                } else {
                    result.as_ref().err().is_some_and(|r| {
                        r.errors.iter().any(|e| {
                            matches!(
                                e.kind,
                                CheckErrorKind::DimensionMismatch | CheckErrorKind::TypeMismatch
                            )
                        })
                    })
                };
                if !correct {
                    failures.push(format!(
                        "{name} reverse={reverse} route={route} expected={expected}: {:?}",
                        result.err().map(|r| r.errors)
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} consumer observations failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
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

#[test]
fn list_axis_classification_crosses_extent_label_rigidity_and_wildcard() {
    let mut failures = Vec::new();
    let mut observations = 0;
    for reverse_mul in [false, true] {
        let op = if reverse_mul {
            "mul(gain,x)"
        } else {
            "mul(x,gain)"
        };
        let library = format!(
            "def fixed_gain[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> tensor[d,f32] = {op}\n\
             def other_gain[d](x: tensor[d,f32], gain: tensor[other,f32]) -> tensor[d,f32] = {op}\n\
             def capture_fixed[d](x: tensor[d,f32], gain: tensor[fixed,f32]) = fn () -> fixed_gain(x,gain)\n\
             def capture_other[d](x: tensor[d,f32], gain: tensor[other,f32]) = fn () -> other_gain(x,gain)"
        );
        let (live, _) = build_compiled_library_context(&parse(&library)).unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bincode::serialize(&live).unwrap()).unwrap();
        let extension = "def forward_fixed[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> tensor[d,f32] = fixed_gain(x,gain)\n\
                         def forward_other[d](x: tensor[d,f32], gain: tensor[other,f32]) -> tensor[d,f32] = other_gain(x,gain)";
        let (layered, _) =
            build_compiled_library_context_with_base(&live, &parse(extension)).unwrap();
        let literal2 = "to_tensor([1.0f32,2.0f32])";
        let literal3 = "to_tensor([1.0f32,2.0f32,3.0f32])";
        let query = |list: &str| {
            format!(
                "match {list} with {{ | Cons(head,tail) => sum(head,fixed) | Nil => scalar_to_tensor(0.0f32) }}"
            )
        };
        for reverse_list in [false, true] {
            let list = |a: &str, b: &str| {
                if reverse_list {
                    format!("[{b},{a}]")
                } else {
                    format!("[{a},{b}]")
                }
            };
            let mut cases = Vec::new();
            for same_label in [false, true] {
                let name = if same_label { "fixed" } else { "other" };
                let callee = format!("{name}_gain");
                for equal_extent in [false, true] {
                    let second = if equal_extent { literal2 } else { literal3 };
                    let xs = list(
                        &format!("fixed_gain({literal2},{literal2})"),
                        &format!("{callee}({second},{second})"),
                    );
                    let uniform = same_label && equal_extent;
                    for (claim, expected, body) in [
                        ("bare", true, format!("def use() = {xs}")),
                        (
                            "wildcard",
                            true,
                            format!("def use() -> List[tensor[*,f32]] = {xs}"),
                        ),
                        (
                            "uniform",
                            uniform,
                            format!("def use() -> List[tensor[fixed,f32]] = {xs}"),
                        ),
                        ("query", uniform, format!("def use() = {}", query(&xs))),
                    ] {
                        cases.push((
                            format!("literal label={same_label} extent={equal_extent} {claim}"),
                            expected,
                            body,
                        ));
                    }
                }
                // Unknown extents of concrete named parameters are not authored
                // generic binders, even after a helper installs a private ID.
                let params = format!(
                    "x: tensor[fixed,f32], y: tensor[{name},f32], gx: tensor[fixed,f32], gy: tensor[{name},f32]"
                );
                let xs = list("fixed_gain(x,gx)", &format!("{callee}(y,gy)"));
                cases.push((
                    format!("named bare {name}"),
                    true,
                    format!("def use({params}) = {xs}"),
                ));
                cases.push((
                    format!("named uniform {name}"),
                    same_label,
                    format!("def use({params}) -> List[tensor[fixed,f32]] = {xs}"),
                ));
                cases.push((
                    format!("named query {name}"),
                    same_label,
                    format!("def use({params}) = {}", query(&xs)),
                ));
                cases.push((format!("published captures {name}"), true, format!("def use({params}) = {{ a = capture_fixed(x,gx)\n b = capture_{name}(y,gy)\n {} }}", list("a()", "b()"))));
                // Actual active authored binders must still unify, even with an
                // explicit wildcard result and even through returned closures.
                let generic = format!(
                    "x: tensor[d,f32], y: tensor[e,f32], gx: tensor[fixed,f32], gy: tensor[{name},f32]"
                );
                cases.push((
                    format!("distinct generic {name}"),
                    false,
                    format!("def use[d,e]({generic}) -> List[tensor[*,f32]] = {xs}"),
                ));
                cases.push((format!("generic captures {name}"), false, format!("def use[d,e]({generic}) = {{ a = capture_fixed(x,gx)\n b = capture_{name}(y,gy)\n {} }}", list("a()", "b()"))));
            }
            for known_extent in [false, true] {
                let params = if known_extent {
                    "w: tensor[*,f32]"
                } else {
                    "x: tensor[fixed,f32], gain: tensor[fixed,f32], w: tensor[*,f32]"
                };
                let named = if known_extent {
                    format!("fixed_gain({literal2},{literal2})")
                } else {
                    "fixed_gain(x,gain)".into()
                };
                let xs = list(&named, "w");
                cases.push((
                    format!("wildcard bare {known_extent}"),
                    true,
                    format!("def use({params}) = {xs}"),
                ));
                cases.push((
                    format!("wildcard uniform {known_extent}"),
                    !reverse_list,
                    format!("def use({params}) -> List[tensor[fixed,f32]] = {xs}"),
                ));
                cases.push((
                    format!("wildcard query {known_extent}"),
                    !reverse_list,
                    format!("def use({params}) = {}", query(&xs)),
                ));
            }
            for equal_extent in [false, true] {
                let plain = if equal_extent { literal2 } else { literal3 };
                let xs = list(&format!("fixed_gain({literal2},{literal2})"), plain);
                cases.push((
                    format!("name/literal bare {equal_extent}"),
                    true,
                    format!("def use() = {xs}"),
                ));
                cases.push((
                    format!("name/literal uniform {equal_extent}"),
                    false,
                    format!("def use() -> List[tensor[fixed,f32]] = {xs}"),
                ));
                cases.push((
                    format!("name/literal query {equal_extent}"),
                    false,
                    format!("def use() = {}", query(&xs)),
                ));
            }
            for (case, expected, source) in cases {
                for route in 0..4 {
                    let result = match route {
                        0 => check_ir_program(&parse(&format!("{library}\n{source}"))),
                        1 => check_ir_with_context(&live, &parse(&source)),
                        2 => check_ir_with_context(&decoded, &parse(&source)),
                        _ => check_ir_with_context(
                            &layered,
                            &parse(
                                &source
                                    .replace("fixed_gain(", "forward_fixed(")
                                    .replace("other_gain(", "forward_other("),
                            ),
                        ),
                    };
                    observations += 1;
                    let correct = if expected {
                        result.is_ok()
                    } else {
                        result.as_ref().err().is_some_and(|r| {
                            r.errors
                                .iter()
                                .any(|e| matches!(e.kind, CheckErrorKind::DimensionMismatch))
                        })
                    };
                    if !correct {
                        failures.push(format!("{case} mul={reverse_mul} list={reverse_list} route={route} expected={expected}: {:?}", result.err().map(|r| r.errors)));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} observations failed:\n{}",
        failures.len(),
        observations,
        failures.join("\n")
    );
}
