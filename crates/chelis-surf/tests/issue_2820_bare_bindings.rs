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
    for literal in [
        "[1, 2, 3]",
        "[1.0, 2.0, 3.0]",
        "[-1i64, 2i64]",
        "[[1.0f16, 2.0f16], [3.0f16, 4.0f16]]",
    ] {
        let bare = deep(&format!("result = {literal}\n"));
        assert!(!bare.contains("to_tensor"), "{literal}: {bare}");
        let explicit = deep(&format!("result = to_tensor({literal})\n"));
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

#[test]
fn lexical_tensor_constructor_capture_is_a_loud_refusal() {
    for source in [
        "def sample() = {\n to_tensor = fn (xs: List[i32]) -> xs\n xs: tensor[2, i32] = [1, 2]\n xs\n}\nresult = sample()\n",
        "def sample(to_tensor) = {\n xs: tensor[2, i32] = [1, 2]\n xs\n}\n",
        "def sample() = {\n (to_tensor, other) = (fn (xs: List[i32]) -> xs, 1)\n xs: tensor[2, i32] = [1, 2]\n xs\n}\n",
        // A declared tensor result converts a bare bracket-literal body too,
        // whether the result is inline or in a standalone `sig`.
        "def sample(to_tensor: List[i32] -> tensor[2, i32]) -> tensor[2, i32] = [1, 2]\n",
        "sig sample: (List[i32] -> tensor[2, i32]) -> tensor[2, i32]\ndef sample(to_tensor) = [1, 2]\n",
    ] {
        let parsed = parse_str(source).unwrap();
        let error = desugar_program(&parsed)
            .expect_err("a lexical callable cannot own a synthesized tensor conversion");
        assert!(error.to_string().contains("to_tensor"), "{error}");
        assert!(error.span().is_some(), "{error}");
    }
    // An authored call, an explicit List and an unannotated bracket literal
    // synthesize no conversion.
    for source in [
        "def sample() = {\n to_tensor = fn (xs: List[i32]) -> xs\n xs: List[i32] = [1, 2]\n to_tensor(xs)\n}\nresult = sample()\n",
        "def sample() = {\n to_tensor = fn (xs: List[i32]) -> xs\n xs = [1, 2]\n to_tensor(xs)\n}\nresult = sample()\n",
    ] {
        assert!(
            desugar_program(&parse_str(source).unwrap()).is_ok(),
            "{source}"
        );
    }
}

#[test]
fn a_shadowed_to_tensor_call_is_an_ordinary_call_that_adopts_nothing() {
    // The intrinsic call in a cast adopts the target dtype.
    let intrinsic = deep("result = cast(to_tensor([1.5, 2.5]), f64)\n");
    assert!(intrinsic.contains("(t-prim {} f64)} 1.5)"), "{intrinsic}");
    // A lexical `to_tensor` is the author's function: its List argument keeps
    // the literal default.
    let shadowed = deep(
        "def sample(to_tensor: List[f32] -> tensor[2, f32]) -> tensor[2, f64] = cast(to_tensor([1.5, 2.5]), f64)\n",
    );
    assert!(shadowed.contains("(t-prim {} f32)} 1.5)"), "{shadowed}");
    assert!(!shadowed.contains("(t-prim {} f64)} 1.5)"), "{shadowed}");
}
