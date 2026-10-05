//! A bracket literal is a List unless `to_tensor` or a declared tensor type
//! converts it (spec/02-surf-syntax.md §P10b, spec/04-type-system.md §5.6).
use chelis_surf::{desugar::desugar_program, parser::parse_str};
fn deep(source: &str) -> String {
    desugar_program(&parse_str(source).unwrap())
        .unwrap()
        .iter()
        .map(chelis_deep::printer::print_expr)
        .collect::<Vec<_>>()
        .join("\n")
}
#[test]
fn bare_numeric_bindings_are_lists_and_to_tensor_makes_tensors() {
    for (literal, dtype) in [
        ("[1, 2, 3]", "i32"),
        ("[1.0, 2.0, 3.0]", "f32"),
        ("[-1i64, 2i64]", "i64"),
        ("[[1.0f16, 2.0f16], [3.0f16, 4.0f16]]", "f16"),
    ] {
        let bare = deep(&format!("result = {literal}\n"));
        assert!(!bare.contains("to_tensor"), "{literal}: {bare}");
        // spec/04 §5.6: an unsuffixed `to_tensor` element states its dtype
        // through the dtype argument.
        let explicit = deep(&format!("result = to_tensor({literal}, {dtype})\n"));
        assert!(explicit.contains("to_tensor)"), "{literal}: {explicit}");
    }
}
#[test]
fn explicit_list_context_and_nonnumeric_bindings_stay_lists() {
    for source in [
        "result: List[f32] = [1.0, 2.0]\n",
        "sig result: List[i32]\nresult = [1, 2]\n",
        "result = [true, false]\n",
        "result = [\"a\", \"b\"]\n",
        "result = []\n",
        "def one() -> i32 = 1\nresult = [one(), one()]\n",
    ] {
        let output = deep(source);
        assert!(!output.contains("(var {} to_tensor)"), "{source}: {output}");
    }
}
#[test]
fn local_bare_binding_stays_list_and_local_tensor_annotation_converts() {
    let inferred = deep("def sample() = {\n  xs = [1.0, 2.0]\n  xs\n}\n");
    assert!(!inferred.contains("(var {} to_tensor)"), "{inferred}");
    let explicit = deep("def sample() = {\n  xs: List[f32] = [1.0, 2.0]\n  xs\n}\n");
    assert!(!explicit.contains("(var {} to_tensor)"), "{explicit}");
    let declared = deep("def sample() = {\n  xs: tensor[2, f32] = [1.0, 2.0]\n  xs\n}\n");
    assert!(declared.contains("(var {} to_tensor)"), "{declared}");
}

#[test]
fn standalone_tensor_signature_supplies_the_declared_literal_context() {
    let output = deep("sig result: tensor[2, f64]\nresult = [1.0, 2.0]\n");
    assert!(output.contains("(var {} to_tensor)"), "{output}");
    assert!(output.contains("(t-prim {} f64)"), "{output}");
    assert!(!output.contains("(t-prim {} f32)"), "{output}");
}

/// spec/04 §8.6: `to_tensor` is reserved, so no lexical binding can capture
/// the conversion a declared tensor literal synthesizes, or any authored
/// call.
#[test]
fn a_lexical_binding_of_to_tensor_is_a_loud_refusal() {
    for source in [
        "def sample() = {\n to_tensor = fn (xs: List[i32]) -> xs\n xs: tensor[2, i32] = [1, 2]\n xs\n}\nresult = sample()\n",
        "def sample(to_tensor) = {\n xs: tensor[2, i32] = [1, 2]\n xs\n}\n",
        "def sample() = {\n (to_tensor, other) = (fn (xs: List[i32]) -> xs, 1)\n xs: tensor[2, i32] = [1, 2]\n xs\n}\n",
        "def sample(to_tensor: List[i32] -> tensor[2, i32]) -> tensor[2, i32] = [1, 2]\n",
        "sig sample: (List[i32] -> tensor[2, i32]) -> tensor[2, i32]\ndef sample(to_tensor) = [1, 2]\n",
        "def sample() = {\n to_tensor = fn (xs: List[i32]) -> xs\n xs: List[i32] = [1, 2]\n to_tensor(xs)\n}\nresult = sample()\n",
        "def sample() = {\n to_tensor = fn (xs: List[i32]) -> xs\n xs = [1, 2]\n to_tensor(xs)\n}\nresult = sample()\n",
    ] {
        let parsed = parse_str(source).unwrap();
        let error = desugar_program(&parsed).expect_err("`to_tensor` cannot be bound");
        assert!(
            error.to_string().contains("`to_tensor` is reserved"),
            "{error}"
        );
        assert!(error.span().is_some(), "{error}");
    }
}

/// A cast or a callee's parameter type never makes a bracket literal a tensor
/// or states its elements' dtype, so the argument of a `to_tensor` call there
/// is an ordinary `List` whose literals bind at their suffix or the dtype
/// argument (spec/04-type-system.md §5.6).
#[test]
fn a_to_tensor_argument_is_an_ordinary_list_whose_literals_keep_their_dtypes() {
    for source in [
        "result = cast(to_tensor([1.5, 2.5], f32), f64)\n",
        "def take(x: tensor[2, f64]) -> tensor[2, f64] = x\nresult = take(to_tensor([1.5, 2.5], f32))\n",
    ] {
        let deep = deep(source);
        assert!(deep.contains("(t-prim {} f32)} 1.5)"), "{source}{deep}");
        assert!(!deep.contains("(t-prim {} f64)} 1.5)"), "{source}{deep}");
    }
    // A suffix states the element dtype at the construction site.
    let suffixed = deep("result = cast(to_tensor([1.5f64, 2.5f64]), f64)\n");
    assert!(suffixed.contains("(t-prim {} f64)} 1.5)"), "{suffixed}");
}
