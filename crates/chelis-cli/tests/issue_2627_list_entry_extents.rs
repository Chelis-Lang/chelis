//! chelis#2627, spec/04 §4.7: each tensor inside a List parameter contributes
//! its named extent witness at function entry, in signature and element order.
//! Empty Lists contribute no witness. Eval and native C must agree.

mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

fn assert_both_trap(source: &str, context: &str) {
    for native in [false, true] {
        let (ok, output) = result_claims::run(source, native);
        let lines = output
            .lines()
            .map(|line| line.trim_start_matches("error: "))
            .filter(|line| line.starts_with("extent `") || line.starts_with("numeric trap:"))
            .collect::<Vec<_>>();
        assert!(!ok, "native={native}: {output}");
        assert_eq!(
            lines,
            [context, "numeric trap: domain in load at i64"],
            "native={native}: {output}"
        );
        assert!(!output.contains("out ="), "native={native}: {output}");
    }
}

fn assert_both_value(source: &str, value: &str) {
    for native in [false, true] {
        let (ok, output) = result_claims::run(source, native);
        assert!(ok, "native={native}: {output}");
        assert!(
            output.lines().any(|line| line == value),
            "native={native}: {output}"
        );
        assert!(
            !output.contains("numeric trap:"),
            "native={native}: {output}"
        );
    }
}

#[test]
fn differing_elements_trap_even_when_body_selects_only_one() {
    assert_both_trap(
        "def f[n](xs: List[tensor[n, f32]]) -> tensor[n, f32] = index(xs, 1i64)\n\
         out = f([to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0])])\n",
        "extent `n`: xs[0] axis 0 = 2, xs[1] axis 0 = 3",
    );
}

#[test]
fn unused_list_still_checks_all_elements_at_entry() {
    assert_both_trap(
        "def f[n](xs: List[tensor[n, f32]]) -> i64 = 7i64\n\
         out = f([to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0])])\n",
        "extent `n`: xs[0] axis 0 = 2, xs[1] axis 0 = 3",
    );
}

#[test]
fn list_witness_precedes_a_later_direct_parameter() {
    assert_both_trap(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def f[n](xs: List[tensor[n, f32]], y: tensor[n, f32]) -> i64 = len(xs)\n\
         out = f([hidden(to_tensor([1.0, 2.0]))], hidden(to_tensor([4.0, 5.0, 6.0])))\n",
        "extent `n`: xs[0] axis 0 = 2, y axis 0 = 3",
    );
}

#[test]
fn direct_parameter_precedes_a_later_list_witness() {
    assert_both_trap(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def f[n](y: tensor[n, f32], xs: List[tensor[n, f32]]) -> i64 = len(xs)\n\
         out = f(hidden(to_tensor([1.0, 2.0])), [hidden(to_tensor([4.0, 5.0, 6.0]))])\n",
        "extent `n`: y axis 0 = 2, xs[0] axis 0 = 3",
    );
}

#[test]
fn list_only_witness_controls_a_declared_result_claim() {
    assert_both_trap(
        "def f[n](xs: List[tensor[n, f32]], t0: tensor[*, f32]) -> tensor[n, f32] = t0\n\
         out = f([to_tensor([1.0, 2.0])], to_tensor([4.0, 5.0, 6.0]))\n",
        "extent `n`: xs[0] axis 0 = 2, load axis 0 = 3",
    );
}

#[test]
fn list_only_witness_guards_a_helper_call_result() {
    assert_both_trap(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def f[n](xs: List[tensor[n, f32]], y: tensor[*, f32]) -> tensor[n, f32] = hidden(y)\n\
         out = f([hidden(to_tensor([1.0, 2.0]))], hidden(to_tensor([4.0, 5.0, 6.0])))\n",
        "extent `n`: xs[0] axis 0 = 2, load axis 0 = 3",
    );
    assert_both_value(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def f[n](xs: List[tensor[n, f32]], y: tensor[*, f32]) -> tensor[n, f32] = hidden(y)\n\
         out = f([hidden(to_tensor([1.0, 2.0]))], hidden(to_tensor([4.0, 5.0])))\n",
        "out = tensor(shape=[2], data=[4.0, 5.0])",
    );
}

#[test]
fn a_later_list_uses_the_first_nonempty_list_as_witness() {
    assert_both_trap(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def f[n](a: List[tensor[n, f32]], b: List[tensor[n, f32]]) -> i64 = len(a) + len(b)\n\
         out = f([], [hidden(to_tensor([1.0, 2.0])), hidden(to_tensor([4.0, 5.0, 6.0]))])\n",
        "extent `n`: b[0] axis 0 = 2, b[1] axis 0 = 3",
    );
}

#[test]
fn a_nested_list_uses_its_innermost_tensor_witnesses() {
    assert_both_trap(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def f[n](xss: List[List[tensor[n, f32]]]) -> i64 = len(xss)\n\
         out = f([[hidden(to_tensor([1.0, 2.0]))], [], [hidden(to_tensor([4.0, 5.0, 6.0]))]])\n",
        "extent `n`: xss[0][0] axis 0 = 2, xss[2][0] axis 0 = 3",
    );
}

#[test]
fn indirect_callable_checks_its_list_before_the_supplied_body() {
    assert_both_trap(
        "def broad(xs: List[tensor[*, f32]]) -> i64 = len(xs)\n\
         def invoke(f: List[tensor[seq, f32]] -> i64, xs: List[tensor[*, f32]]) -> i64 = f(xs)\n\
         out = invoke(broad, [to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0])])\n",
        "extent `seq`: arg0[0] axis 0 = 2, arg0[1] axis 0 = 3",
    );
}

#[test]
fn indirect_callable_orders_direct_and_list_witnesses() {
    let prefix = "def broad(y: tensor[*, f32], xs: List[tensor[*, f32]]) -> i64 = len(xs)\n\
         def invoke(f: tensor[seq, f32] -> List[tensor[seq, f32]] -> i64, y: tensor[*, f32], xs: List[tensor[*, f32]]) -> i64 = f(y, xs)\n";
    assert_both_trap(
        &format!(
            "{prefix}out = invoke(broad, to_tensor([1.0, 2.0]), [to_tensor([4.0, 5.0, 6.0])])\n"
        ),
        "extent `seq`: arg0 axis 0 = 2, arg1[0] axis 0 = 3",
    );
    assert_both_value(
        &format!(
            "{prefix}def empty() -> List[tensor[*, f32]] = []\n\
             out = invoke(broad, to_tensor([1.0, 2.0]), empty())\n"
        ),
        "out = 0",
    );
}

#[test]
fn indirect_callable_checks_nested_lists() {
    assert_both_trap(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def broad(xss: List[List[tensor[*, f32]]]) -> i64 = len(xss)\n\
         def invoke(f: List[List[tensor[seq, f32]]] -> i64, xss: List[List[tensor[*, f32]]]) -> i64 = f(xss)\n\
         out = invoke(broad, [[hidden(to_tensor([1.0, 2.0]))], [hidden(to_tensor([4.0, 5.0, 6.0]))]])\n",
        "extent `seq`: arg0[0][0] axis 0 = 2, arg0[1][0] axis 0 = 3",
    );
}

#[test]
fn inline_map_callback_checks_its_named_list_formal() {
    assert_both_trap(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def outer[n](xss: List[List[tensor[*, f32]]]) -> List[i64] = map(fn (xs: List[tensor[n, f32]]) -> len(xs), xss)\n\
         out = outer([[hidden(to_tensor([1.0, 2.0])), hidden(to_tensor([3.0, 4.0, 5.0]))]])\n",
        "extent `n`: xs[0] axis 0 = 2, xs[1] axis 0 = 3",
    );
    assert_both_value(
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def outer[n](xss: List[List[tensor[*, f32]]]) -> List[i64] = map(fn (xs: List[tensor[n, f32]]) -> len(xs), xss)\n\
         out = outer([[hidden(to_tensor([1.0, 2.0])), hidden(to_tensor([3.0, 4.0]))]])\n",
        "out = [2]",
    );
}

#[test]
fn agreeing_and_empty_lists_execute_without_inventing_a_witness() {
    assert_both_value(
        "def f[n](xs: List[tensor[n, f32]]) -> tensor[n, f32] = index(xs, 1i64)\n\
         out = f([to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0])])\n",
        "out = tensor(shape=[2], data=[4.0, 5.0])",
    );
    assert_both_value(
        "def f[n](xs: List[tensor[n, f32]], y: tensor[n, f32]) -> tensor[n, f32] = y\n\
         out = f([], to_tensor([4.0, 5.0]))\n",
        "out = tensor(shape=[2], data=[4.0, 5.0])",
    );
}
