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
fn exported_c_entry_keeps_named_list_ahead_of_later_list_failure() {
    use std::fs;

    let dir = tempfile::tempdir().expect("entry fixture");
    let source = dir.path().join("completion.ch");
    fs::write(
        &source,
        "def hidden(x: tensor[*, f32]) -> tensor[*, f32] = x\n\
         def f[n](xs: List[tensor[n, f32]], ys: List[tensor[2, f32]]) -> i64 = 7i64\n\
         out = f([hidden(to_tensor([1.0, 2.0]))], [hidden(to_tensor([3.0, 4.0]))])\n",
    )
    .expect("source");
    let out = dir.path().join("c");
    let built = assert_cmd::Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--allow-style-violations"])
        .arg(&source)
        .args(["--target", "c", "-o"])
        .arg(&out)
        .output()
        .expect("build");
    assert!(built.status.success(), "{built:?}");

    let run = |first: i64, second: i64, later: i64, name: &str| {
        let harness = format!(
            "#define main generated_main\n#include \"completion.c\"\n#undef main\n\
             int main(void) {{\n\
             chelis_value xs_items[2] = {{\n\
               chelis_value_take_tensor(chelis_alloc(1, (int64_t[]){{{first}}}, CHELIS_DTYPE_F32)),\n\
               chelis_value_take_tensor(chelis_alloc(1, (int64_t[]){{{second}}}, CHELIS_DTYPE_F32))\n\
             }};\n\
             chelis_list *xs = chelis_list_from_values(xs_items, 2);\n\
             chelis_value_release(xs_items[0]); chelis_value_release(xs_items[1]);\n\
             chelis_value ys_item = chelis_value_take_tensor(chelis_alloc(1, (int64_t[]){{{later}}}, CHELIS_DTYPE_F32));\n\
             chelis_list *ys = chelis_list_from_values(&ys_item, 1);\n\
             chelis_value_release(ys_item);\n\
             printf(\"out = %lld\\n\", (long long){}(xs, ys));\n\
             chelis_list_release(xs); chelis_list_release(ys);\n\
             return 0;\n}}\n",
            common::authored_c_symbol("f")
        );
        let file = format!("{name}.c");
        fs::write(out.join(&file), harness).expect("harness");
        assert!(
            common::link_generated(&out, &file, name).success(),
            "{name} link"
        );
        let result = std::process::Command::new(out.join(name))
            .output()
            .expect("execute");
        (
            result.status.success(),
            format!(
                "{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            ),
        )
    };

    let (ok, output) = run(2, 3, 3, "both_bad");
    assert!(!ok, "{output}");
    assert!(
        output.contains(
            "extent `n`: xs[0] axis 0 = 2, xs[1] axis 0 = 3\nnumeric trap: domain in load at i64"
        ),
        "{output}"
    );
    let (ok, output) = run(2, 2, 3, "later_bad");
    assert!(!ok, "{output}");
    assert!(
        output.contains(
            "input `ys[0]` axis 0 expected 2, got 3\nnumeric trap: domain in load at i64"
        ),
        "{output}"
    );
    let (ok, output) = run(2, 2, 2, "both_good");
    assert!(ok, "{output}");
    assert!(output.contains("out = 7"), "{output}");
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
