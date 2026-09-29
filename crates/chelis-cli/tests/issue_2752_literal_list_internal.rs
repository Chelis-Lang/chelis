//! chelis#2752, spec/04 §4.7: internal C calls check literal tensor
//! extents inside List formals before the body, matching Eval.

mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

fn assert_both_trap(source: &str, eval_context: &str, native_context: &str) {
    for native in [false, true] {
        let (ok, output) = result_claims::run(source, native);
        assert!(!ok, "native={native}: {output}");
        let expected = if native { native_context } else { eval_context };
        assert!(
            output.contains(&format!("{expected}\nnumeric trap: domain in load at i64")),
            "native={native}: {output}"
        );
        assert!(!output.contains("out ="), "native={native}: {output}");
    }
}

#[test]
fn literal_list_internal_call_rejects_mismatch_and_accepts_match() {
    let prefix = "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
                  def f(xs: List[tensor[2, f32]]) -> i64 = 7i64\n";
    assert_both_trap(
        &format!("{prefix}out = f([hidden(to_tensor([1.0f32, 2.0f32, 3.0f32]))])\n"),
        "extent `2`: claimed = 2, xs[0] axis 0 = 3",
        "input `xs[0]` axis 0 expected 2, got 3",
    );
    for native in [false, true] {
        let (ok, output) = result_claims::run(
            &format!("{prefix}out = f([hidden(to_tensor([1.0f32, 2.0f32]))])\n"),
            native,
        );
        assert!(ok, "native={native}: {output}");
        assert!(output.contains("out = 7"), "native={native}: {output}");
    }
}

#[test]
fn nested_literal_list_checks_each_element_before_body() {
    let prefix = "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
                  def f(xss: List[List[tensor[2, f32]]]) -> i64 = 7i64\n";
    assert_both_trap(
        &format!(
            "{prefix}out = f([[hidden(to_tensor([1.0f32, 2.0f32]))], [], [hidden(to_tensor([3.0f32, 4.0f32, 5.0f32]))]])\n"
        ),
        "extent `2`: claimed = 2, xss[2][0] axis 0 = 3",
        "input `xss[2][0]` axis 0 expected 2, got 3",
    );
}

#[test]
fn earlier_literal_list_failure_precedes_later_named_list_failure() {
    let source = "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
                  def f[n](xs: List[tensor[2, f32]], ys: List[tensor[n, f32]]) -> i64 = 7i64\n\
                  out = f([hidden(to_tensor([1.0f32, 2.0f32, 3.0f32]))], [hidden(to_tensor([4.0f32, 5.0f32])), hidden(to_tensor([6.0f32, 7.0f32, 8.0f32]))])\n";
    assert_both_trap(
        source,
        "extent `2`: claimed = 2, xs[0] axis 0 = 3",
        "input `xs[0]` axis 0 expected 2, got 3",
    );
}

#[test]
fn retained_callable_checks_literal_list_formal() {
    let prefix = "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
                  def broad(xs: List[tensor[*, f32]]) -> i64 = len(xs)\n\
                  def invoke(f: List[tensor[2, f32]] -> i64, xs: List[tensor[*, f32]]) -> i64 = f(xs)\n";
    assert_both_trap(
        &format!("{prefix}out = invoke(broad, [hidden(to_tensor([1.0f32, 2.0f32, 3.0f32]))])\n"),
        "extent `2`: claimed = 2, arg0[0] axis 0 = 3",
        "input `arg0[0]` axis 0 expected 2, got 3",
    );
    for native in [false, true] {
        let (ok, output) = result_claims::run(
            &format!("{prefix}out = invoke(broad, [hidden(to_tensor([1.0f32, 2.0f32]))])\n"),
            native,
        );
        assert!(ok, "native={native}: {output}");
        assert!(output.contains("out = 1"), "native={native}: {output}");
    }
}

#[test]
fn inline_callback_checks_literal_list_formal() {
    assert_both_trap(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def outer(xss: List[List[tensor[*, f32]]]) -> List[i64] = map(fn (xs: List[tensor[2, f32]]) -> len(xs), xss)\n\
         out = outer([[hidden(to_tensor([1.0f32, 2.0f32, 3.0f32]))]])\n",
        "extent `2`: claimed = 2, xs[0] axis 0 = 3",
        "input `xs[0]` axis 0 expected 2, got 3",
    );
}

#[test]
fn literal_list_precedes_tensor_helper_even_when_unused() {
    let prefix = "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
                  def f(xs: List[tensor[2, f32]], y: tensor[3, f32]) -> tensor[3, f32] = y + y\n";
    let mismatched = format!(
        "{prefix}out = f([hidden(to_tensor([1.0f32, 2.0f32, 3.0f32]))], to_tensor([4.0f32, 5.0f32, 6.0f32]))\n"
    );
    assert_both_trap(
        &mismatched,
        "extent `2`: claimed = 2, xs[0] axis 0 = 3",
        "input `xs[0]` axis 0 expected 2, got 3",
    );
    let matching = format!(
        "{prefix}out = f([hidden(to_tensor([1.0f32, 2.0f32]))], to_tensor([4.0f32, 5.0f32, 6.0f32]))\n"
    );
    for native in [false, true] {
        let (ok, output) = result_claims::run(&matching, native);
        assert!(ok, "native={native}: {output}");
        assert!(
            output.contains("out = tensor(shape=[3], data=[8.0, 10.0, 12.0])"),
            "native={native}: {output}"
        );
    }
}
