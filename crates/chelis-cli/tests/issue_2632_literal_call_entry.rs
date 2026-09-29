//! spec/04 §4.7: an authored literal tensor parameter is checked at each
//! invocation, including when its argument was produced by a runtime movement
//! operation and the callee is inlined into a tensor kernel.

mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

use result_claims::run;

fn source(bound: usize) -> String {
    format!(
        "def g(y: tensor[3, f32]) -> tensor[f32] = sum(y, 0i32)\n\
         def f[n](x: tensor[n, f32], k: i64) -> tensor[f32] = g(shrink(&x, [[0i64, k]]))\n\
         out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), {bound}i64)\n"
    )
}

#[test]
fn original_disagreeing_call_traps_before_g_computes_on_both_lanes() {
    for native in [false, true] {
        let (ok, output) = run(&source(2), native);
        assert!(!ok, "native={native}: {output}");
        assert!(
            output.contains("extent `3`: claimed = 3, y axis 0 = 2"),
            "native={native}: {output}"
        );
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in load at i64"),
            "native={native}: {output}"
        );
        assert!(!output.contains("out = 2.0"), "native={native}: {output}");
    }
}

#[test]
fn agreeing_runtime_argument_executes_on_both_lanes() {
    for native in [false, true] {
        let (ok, output) = run(&source(3), native);
        assert!(ok, "native={native}: {output}");
        assert!(output.contains("out = 3.0"), "native={native}: {output}");
        assert!(
            !output.contains("numeric trap:"),
            "native={native}: {output}"
        );
    }
}

#[test]
fn wildcard_parameter_does_not_claim_the_callers_observed_extent() {
    let source = source(2).replace("g(y: tensor[3, f32])", "g(y: tensor[*, f32])");
    for native in [false, true] {
        let (ok, output) = run(&source, native);
        assert!(ok, "native={native}: {output}");
        assert!(output.contains("out = 2.0"), "native={native}: {output}");
        assert!(
            !output.contains("numeric trap:"),
            "native={native}: {output}"
        );
    }
}

#[test]
fn original_shape_arithmetic_in_a_taken_if_arm_traps_on_both_lanes() {
    let source = "def g(y: tensor[3, f32]) -> tensor[f32] = sum(y, 0i32)\n\
                  def f(x: tensor[4, f32]) -> tensor[f32] = {\n\
                    s = tensor_to_scalar(sum(&x, 0i32))\n\
                    k = sub(shape(&x, 0i32), 2i64)\n\
                    if gt(s, -5.0f32) then g(shrink(&x, [[0i64, k]])) else sum(x, 0i32)\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))\n";
    for native in [false, true] {
        let (ok, output) = run(source, native);
        assert!(!ok, "native={native}: {output}");
        assert!(
            output.contains("extent `3`: claimed = 3, y axis 0 = 2"),
            "native={native}: {output}"
        );
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in load at i64"),
            "native={native}: {output}"
        );
        assert!(!output.contains("out = 2.0"), "native={native}: {output}");
    }
}

fn selected_source(taken: bool) -> String {
    format!(
        "def g(y: tensor[3, f32]) -> tensor[f32] ! {{ IO }} = {{\n\
           _ = print(\"body-ran\")\n\
           sum(y, 0i32)\n\
         }}\n\
         def f(x: tensor[4, f32], b: bool) -> tensor[f32] ! {{ IO }} = {{\n\
           k = sub(shape(&x, 0i32), 2i64)\n\
           if b then g(shrink(&x, [[0i64, k]])) else sum(x, 0i32)\n\
         }}\n\
         out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), {taken})\n"
    )
}

#[test]
fn taken_call_checks_its_literal_before_the_body_on_both_lanes() {
    for native in [false, true] {
        let (ok, output) = run(&selected_source(true), native);
        assert!(!ok, "native={native}: {output}");
        assert!(
            output.contains("extent `3`: claimed = 3, y axis 0 = 2")
                || output.contains("input `y` axis 0 expected 3, got 2"),
            "native={native}: {output}"
        );
        assert!(
            output.contains("numeric trap: domain in load at i64"),
            "native={native}: {output}"
        );
        assert!(!output.contains("body-ran"), "native={native}: {output}");
        assert!(!output.contains("out ="), "native={native}: {output}");
    }
}

#[test]
fn untaken_call_does_not_check_its_literal_on_either_lane() {
    for native in [false, true] {
        let (ok, output) = run(&selected_source(false), native);
        assert!(ok, "native={native}: {output}");
        assert!(output.contains("out = 4.0"), "native={native}: {output}");
        assert!(!output.contains("body-ran"), "native={native}: {output}");
        assert!(
            !output.contains("numeric trap:"),
            "native={native}: {output}"
        );
    }
}
