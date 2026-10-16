//! [04-INF-9], spec/06 §3: static builtin values retain operation identity.
mod common;
use assert_cmd::Command;
use common::build_and_run;

fn evaluate(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("builtin_vmap.ch");
    std::fs::write(&path, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn assert_native(source: &str, expected: &[&str], label: &str) {
    let eval = evaluate(source);
    for row in expected {
        assert!(eval.contains(row), "missing {row}: {eval}");
    }
    assert_eq!(build_and_run(source, label), eval);
}

#[test]
fn admitted_builtin_targets_preserve_direct_alias_capture_and_shadow_identity() {
    let source = r#"
f = neg
chain = f
x = to_tensor([[1.0f32,-2.0f32],[3.0f32,-4.0f32]])
direct = vmap(neg)(x)
aliased = vmap(chain)(x)
twin = vmap(fn (z: tensor[2,f32]) -> neg(z))(x)
def local(z: tensor[2,2,f32]) -> tensor[2,2,f32] = {
  target = abs
  vmap(target)(z)
}
def captured(z: tensor[2,2,f32]) -> tensor[2,2,f32] = {
  target = chain
  wrapper = fn (v: tensor[2,2,f32]) -> vmap(target)(v)
  wrapper(z)
}
def shadow(neg: (&tensor[2,f32] -> tensor[2,f32]), z: tensor[2,2,f32]) -> tensor[2,2,f32] = vmap(neg)(z)
positive = local(x)
negative = captured(x)
shadowed = shadow(abs,x)
"#;
    assert_native(
        source,
        &[
            "direct = tensor(shape=[2, 2], data=[-1.0, 2.0, -3.0, 4.0])",
            "aliased = tensor(shape=[2, 2], data=[-1.0, 2.0, -3.0, 4.0])",
            "positive = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])",
            "negative = tensor(shape=[2, 2], data=[-1.0, 2.0, -3.0, 4.0])",
            "shadowed = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])",
        ],
        "builtin_vmap_aliases",
    );
}

#[test]
fn float_predicate_targets_keep_bool_output_at_all_stored_widths() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (op, mask) in [
            ("is_nan", "[false, true, false, false]"),
            ("is_finite", "[true, false, false, true]"),
            ("is_infinite", "[false, false, true, false]"),
        ] {
            let source = format!(
                r#"
f = {op}
x = to_tensor([[1.0{dtype},div(0.0{dtype},0.0{dtype})],[div(1.0{dtype},0.0{dtype}),-2.0{dtype}]])
result = vmap(f)(x)
twin = vmap(fn (z: tensor[2,{dtype}]) -> {op}(z))(x)
"#
            );
            assert_native(
                &source,
                &[&format!("result = tensor(shape=[2, 2], data={mask})")],
                &format!("builtin_vmap_{op}_{dtype}"),
            );
        }
    }
}

#[test]
fn binary_and_comparison_builtin_targets_keep_distinct_result_precision() {
    let source = r#"
x = to_tensor([[1i32,2i32],[3i32,4i32]])
y = to_tensor([[4i32,2i32],[1i32,4i32]])
f = add
result = vmap(f)(x,y)
mask = vmap(fn (a: tensor[2,i32], b: tensor[2,i32]) -> eq(a,b))(x,y)
"#;
    assert_native(
        source,
        &[
            "result = tensor(shape=[2, 2], data=[5, 4, 4, 8])",
            "mask = tensor(shape=[2, 2], data=[false, true, false, true])",
        ],
        "builtin_vmap_binary",
    );
}

#[test]
fn static_builtin_vmap_preserves_nonzero_axes_scalar_rows_and_empty_batches() {
    let source = r#"
x = to_tensor([[1.0f64,-2.0f64,3.0f64],[-4.0f64,5.0f64,-6.0f64]])
result = vmap(neg,axis=1)(x)
scalar_rows = vmap(abs)(to_tensor([-1.0f64,2.0f64,-3.0f64]))
def empty(z: tensor[0,2,f32]) -> tensor[0,2,bool] = vmap(is_finite)(z)
empty_base: tensor[0,f32] = to_tensor([])
empty_result = empty(reshape(empty_base,[0i64,2i64]))
axis_input = to_tensor([[[1.0f64,2.0f64],[3.0f64,4.0f64],[5.0f64,6.0f64]],[[10.0f64,20.0f64],[30.0f64,40.0f64],[50.0f64,60.0f64]]])
axis_control = vmap(fn (z: tensor[2,2,f64]) -> sum(z,0),axis=1)(axis_input)
"#;
    assert_native(
        source,
        &[
            "result = tensor(shape=[2, 3], data=[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])",
            "scalar_rows = tensor(shape=[3], data=[1.0, 2.0, 3.0])",
            "empty_result = tensor(shape=[0, 2], data=[])",
            "axis_control = tensor(shape=[2, 3], data=[11.0, 33.0, 55.0, 22.0, 44.0, 66.0])",
        ],
        "builtin_vmap_axes",
    );
}

#[test]
fn invalid_builtin_vmap_calls_fail_checking_without_execution() {
    for source in [
        "x=to_tensor([1i32,2i32])\nresult=vmap(is_finite)(x)",
        "x=to_tensor([true,false])\nresult=vmap(neg)(x)",
        "emit=print",
        "x=to_tensor([1.0f32,2.0f32])\nresult=vmap(add)(x)",
        "x=to_tensor([1.0f32,2.0f32])\ny=to_tensor([1.0f64,2.0f64])\nresult=vmap(add)(x,y)",
        "x=to_tensor([[1i32,2i32],[3i32,4i32]])\ny=to_tensor([[1i32,2i32,3i32],[4i32,5i32,6i32]])\nresult=vmap(eq)(x,y)",
        "x=to_tensor([1.0f32,2.0f32])\nresult=vmap(neg,axis=2)(x)",
        "def axis(n:i32,x:tensor[2,f32])->tensor[2,f32]=vmap(neg,axis=n)(x)",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rejected.ch");
        std::fs::write(&path, source).unwrap();
        Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("check")
            .arg(path)
            .assert()
            .failure();
    }
}

#[test]
fn admitted_value_callbacks_keys_and_gradient_use_the_existing_operation_owners() {
    let source = r#"
def apply(h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[2,f32]) -> tensor[2,f32] = h(x)
def mapped[n](h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[n,2,f32]) -> tensor[n,2,f32] = vmap(h)(x)
def loss(x: tensor[2,f32]) -> f32 = tensor_to_scalar(sum(vmap(neg)(x),0))
x = to_tensor([-1.0f32,2.0f32])
ordinary = apply(abs,x)
batched = mapped(neg,to_tensor([[1.0f32,2.0f32],[3.0f32,4.0f32]]))
derivative = grad(loss)(x)
def pair() -> (key,key) = {
  seed = key_from_seed
  splitter = split_key
  splitter(seed(1i64))
}
keys = {
  seed = key_from_seed
  vmap(seed)(to_tensor([1i64,2i64]))
}
halves = pair()
"#;
    assert_native(
        source,
        &[
            "ordinary = tensor(shape=[2], data=[1.0, 2.0])",
            "batched = tensor(shape=[2, 2], data=[-1.0, -2.0, -3.0, -4.0])",
            "derivative = tensor(shape=[2], data=[-1.0, -1.0])",
        ],
        "builtin_vmap_callbacks_keys_grad",
    );
}

#[test]
fn callback_contracts_remain_narrower_than_the_supplied_builtin_value() {
    let source = r#"
def mapped[n](h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[n,*,f32]) -> tensor[n,2,f32] = vmap(h)(x)
result = mapped(neg,to_tensor([[1.0f32,2.0f32,3.0f32],[4.0f32,5.0f32,6.0f32]]))
"#;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("claim.ch");
    std::fs::write(&path, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("numeric trap: domain"),
        "{:?}",
        output
    );
    let out = dir.path().join("out");
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&out)
        .assert()
        .success();
    let native = std::process::Command::new(out.join("claim"))
        .output()
        .unwrap();
    assert!(!native.status.success());
    assert!(
        String::from_utf8_lossy(&native.stderr).contains("numeric trap: domain"),
        "{:?}",
        native
    );
}

#[test]
fn builtin_value_carriers_keep_eager_effects_and_unused_callable_parameter_work() {
    let source = r#"
unused = abs
f = neg
def data() -> tensor[2,2,f32] ! {IO} = {
  _ = print("operand-once")
  to_tensor([[1.0f32,2.0f32],[3.0f32,4.0f32]])
}
def keep_work(h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[2,f32]) -> tensor[2,f32] = neg(x)
x = data()
result = vmap(f)(x)
kept = keep_work(abs,to_tensor([1.0f32,2.0f32]))
"#;
    let evaluated = evaluate(source);
    assert_eq!(evaluated.matches("operand-once").count(), 1, "{evaluated}");
    assert!(
        evaluated.contains("kept = tensor(shape=[2], data=[-1.0, -2.0])"),
        "{evaluated}"
    );
    assert_eq!(
        build_and_run(source, "builtin_vmap_eager_and_unused"),
        evaluated
    );
}

#[test]
fn observable_returned_or_computed_callable_values_keep_typed_c_rejections() {
    for source in [
        "def expose() -> (&tensor[2,f32] -> tensor[2,f32]) = neg\nout = print(6i32)\n",
        "def choose(flag:bool) -> (&tensor[2,f32] -> tensor[2,f32]) = if flag then neg else abs\ntarget = choose(false)\nx = to_tensor([1.0f32,-2.0f32])\nresult = target(x)\n",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("observable_callable.ch");
        let out = dir.path().join("out");
        std::fs::write(&path, source).unwrap();
        let result = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("build")
            .arg("--emit-c")
            .arg(&path)
            .args(["--target", "c", "--output"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            !result.status.success(),
            "an unresolved observable callable cannot disappear with a successful build"
        );
        let diagnostic = String::from_utf8_lossy(&result.stderr);
        assert!(
            diagnostic.contains("unsupported:") && diagnostic.contains("function value"),
            "{diagnostic}"
        );
        assert!(!out.exists() || std::fs::read_dir(out).unwrap().next().is_none());
    }
}

#[test]
fn admitted_integer_builtin_targets_preserve_values_beyond_float_exactness() {
    let source = r#"
x = to_tensor([[9007199254740993i64,-9007199254740995i64],[17i64,-19i64]])
negative = vmap(neg,axis=1)(x)
f = abs
positive = vmap(f)(x)
twin = vmap(fn (row: tensor[2,i64]) -> neg(row),axis=1)(x)
"#;
    assert_native(
        source,
        &[
            "negative = tensor(shape=[2, 2], data=[-9007199254740993, 9007199254740995, -17, 19])",
            "positive = tensor(shape=[2, 2], data=[9007199254740993, 9007199254740995, 17, 19])",
        ],
        "builtin_vmap_exact_i64",
    );
}

#[test]
fn unresolved_transform_templates_keep_other_roots_and_refuse_a_standalone_entry() {
    let template = "def mapped(h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[2,2,f32]) -> tensor[2,2,f32] = vmap(h)(x)\n";
    let with_root = format!(
        "{template}x = to_tensor([[1.0f32,2.0f32],[3.0f32,4.0f32]])\nresult = vmap(neg)(x)\n"
    );
    assert_native(
        &with_root,
        &["result = tensor(shape=[2, 2], data=[-1.0, -2.0, -3.0, -4.0])"],
        "builtin_vmap_unused_template",
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("standalone.ch");
    let out = dir.path().join("out");
    std::fs::write(&path, format!("module Standalone\n{template}")).unwrap();
    let result = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg("--emit-c")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "a callable-valued standalone entry needs a typed capability failure"
    );
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("vmap"),
        "{:?}",
        result
    );
    assert!(!out.exists() || std::fs::read_dir(out).unwrap().next().is_none());
}

#[test]
fn integer_minimum_overflow_is_identical_for_builtin_and_typed_lambda_rows() {
    for operation in ["neg", "abs"] {
        for target in [
            operation.to_string(),
            format!("fn (row: tensor[i64]) -> {operation}(row)"),
        ] {
            let source = format!(
                "result = vmap({target})(to_tensor([sub(-9223372036854775807i64,1i64)]))\n"
            );
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("overflow.ch");
            let out = dir.path().join("out");
            std::fs::write(&path, &source).unwrap();
            let eval = Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["eval", "--file"])
                .arg(&path)
                .output()
                .unwrap();
            assert!(!eval.status.success());
            let expected = format!("numeric trap: overflow in {operation} at i64");
            assert!(
                String::from_utf8_lossy(&eval.stderr).contains(&expected),
                "{:?}",
                eval
            );
            Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .arg("build")
                .arg(&path)
                .args(["--target", "c", "--output"])
                .arg(&out)
                .assert()
                .success();
            let native = std::process::Command::new(out.join("overflow"))
                .output()
                .unwrap();
            assert!(!native.status.success());
            assert!(
                String::from_utf8_lossy(&native.stderr).contains(&expected),
                "{:?}",
                native
            );
        }
    }
}

#[test]
fn forwarded_transform_formals_specialize_only_when_their_actual_target_is_static() {
    let source = r#"
def inner[n](h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[n,2,f32]) -> tensor[n,2,f32] = vmap(h)(x)
def outer[n](h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[n,2,f32]) -> tensor[n,2,f32] = inner(h,x)
def unused(h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[2,2,f32]) -> tensor[2,2,f32] = inner(abs,x)
x = to_tensor([[1.0f32,-2.0f32],[3.0f32,-4.0f32]])
result = outer(neg,x)
static_target = unused(neg,x)
"#;
    assert_native(
        source,
        &[
            "result = tensor(shape=[2, 2], data=[-1.0, 2.0, -3.0, 4.0])",
            "static_target = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])",
        ],
        "builtin_vmap_forwarded_formal",
    );
}

#[test]
fn same_signature_shadowed_builtin_names_keep_both_supplied_identities() {
    let source = r#"
def mapped(neg: (&tensor[2,f32] -> tensor[2,f32]), abs: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[2,2,f32]) -> (tensor[2,2,f32],tensor[2,2,f32]) = (vmap(neg)(x),vmap(abs)(x))
x = to_tensor([[1.0f32,-2.0f32],[3.0f32,-4.0f32]])
result = mapped(abs,neg,x)
"#;
    assert_native(
        source,
        &[
            "result.0 = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])",
            "result.1 = tensor(shape=[2, 2], data=[-1.0, 2.0, -3.0, 4.0])",
        ],
        "builtin_vmap_two_shadows",
    );
}

#[test]
fn builtin_targets_keep_named_runtime_axes_and_narrow_nonzero_axis_formals() {
    let source = r#"
def mapped[b,n,p: Float](x: tensor[b,n,p]) -> tensor[b,n,bool] = vmap(is_finite,axis=1)(x)
def borrowed[b](h: (&tensor[2,f32] -> tensor[2,f32]), x: tensor[2,b,f32]) -> tensor[2,b,f32] = vmap(h,axis=1)(x)
x = to_tensor([[1.0f32,-2.0f32,3.0f32],[4.0f32,5.0f32,-6.0f32]])
mask = mapped(x)
result = borrowed(neg,x)
"#;
    assert_native(
        source,
        &[
            "mask = tensor(shape=[2, 3], data=[true, true, true, true, true, true])",
            "result = tensor(shape=[2, 3], data=[-1.0, 2.0, -3.0, -4.0, -5.0, 6.0])",
        ],
        "builtin_vmap_named_nonzero_formal",
    );
}

#[test]
fn admitted_host_builtin_values_keep_host_effects_and_pure_callback_results() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("data.txt");
    let source = format!(
        r#"
reader = read_file
writer = write_file
def invoke(h: (string -> string), path: string) -> string ! {{IO}} = h(path)
def apply(h: (&f32 -> f32), x: f32) -> f32 = h(x)
before = print("before")
written = writer({path:?},"content")
content = invoke(reader,{path:?})
x = to_tensor([1.0f32,-2.0f32])
printed = print(x)
positive = apply(abs,-1.0f32)
after = print("after")
"#,
        path = data.to_str().unwrap()
    );
    let evaluated = evaluate(&source);
    assert!(evaluated.find("before\n").unwrap() < evaluated.find("after\n").unwrap());
    assert_eq!(evaluated.matches("before\n").count(), 1);
    assert_eq!(evaluated.matches("after\n").count(), 1);
    assert!(evaluated.contains("content = content"));
    assert!(evaluated.contains("positive = 1.0"));
    assert_eq!(
        build_and_run(&source, "builtin_vmap_host_carriers"),
        evaluated
    );
}

#[test]
fn host_builtin_callback_failure_precedes_later_effects() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing.txt");
    let source = format!(
        r#"
reader = read_file
def invoke(h: (string -> string), path: string) -> string ! {{IO}} = h(path)
content = invoke(reader,{path:?})
after = print("must-not-run")
"#,
        path = missing.to_str().unwrap()
    );
    let file = directory.path().join("host_failure.ch");
    std::fs::write(&file, &source).unwrap();
    let evaluated = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(!evaluated.status.success());
    assert!(!String::from_utf8_lossy(&evaluated.stdout).contains("must-not-run"));
    let output = directory.path().join("out");
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&file)
        .args(["--target", "c", "--output"])
        .arg(&output)
        .assert()
        .success();
    let native = std::process::Command::new(output.join("host_failure"))
        .output()
        .unwrap();
    assert!(!native.status.success());
    assert!(!String::from_utf8_lossy(&native.stdout).contains("must-not-run"));
}

/// The row Load describes physical input; a narrower formal stays a claim.
#[test]
fn physical_rows_are_checked_against_typed_lambda_and_builtin_formals() {
    for axis in [0, 1] {
        for supplied_builtin in [false, true] {
            for width in [2, 3] {
                let (input, output_shape, formal) = if axis == 0 {
                    let row = vec!["-1.0f32"; width].join(",");
                    (
                        format!("[[{row}],[{row}],[{row}]]"),
                        "[3, 2]",
                        "tensor[3,*,f32]",
                    )
                } else {
                    let row = "[-1.0f32,-1.0f32,-1.0f32]";
                    (
                        format!("[{}]", vec![row; width].join(",")),
                        "[2, 3]",
                        "tensor[*,3,f32]",
                    )
                };
                let result_type = if axis == 0 {
                    "tensor[3,2,f32]"
                } else {
                    "tensor[2,3,f32]"
                };
                let (params, target, actual) = if supplied_builtin {
                    (
                        format!("h: (&tensor[2,f32] -> tensor[2,f32]), x: {formal}"),
                        "h",
                        "abs, ",
                    )
                } else {
                    (
                        format!("x: {formal}"),
                        "fn (row: &tensor[2,f32]) -> abs(row)",
                        "",
                    )
                };
                let axis_suffix = if axis == 0 { "" } else { ",axis=1" };
                let source = format!(
                    r#"
def mapped({params}) -> {result_type} = vmap({target}{axis_suffix})(x)
x=to_tensor({input})
result=mapped({actual}x)
after=print("after-row-claim")
"#
                );
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("physical_rows.ch");
                let out = dir.path().join("native");
                std::fs::write(&path, &source).unwrap();
                let evaluated = Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .args(["eval", "--file"])
                    .arg(&path)
                    .output()
                    .unwrap();
                Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .arg("build")
                    .arg(&path)
                    .args(["--target", "c", "--output"])
                    .arg(&out)
                    .assert()
                    .success();
                let native = std::process::Command::new(out.join("physical_rows"))
                    .output()
                    .unwrap();
                for output in [&evaluated, &native] {
                    let text = format!(
                        "{}{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                    if width == 2 {
                        assert!(
                            output.status.success(),
                            "axis={axis} builtin={supplied_builtin}: {text}"
                        );
                        assert!(text.contains(&format!("result = tensor(shape={output_shape}, data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])")),"{text}");
                        assert!(text.contains("after-row-claim"), "{text}");
                    } else {
                        assert!(!output.status.success(), "{text}");
                        assert!(
                            text.contains("numeric trap: domain in load at i64"),
                            "axis={axis} builtin={supplied_builtin}: {text}"
                        );
                        assert!(!text.contains("after-row-claim"), "{text}");
                    }
                }
                if width == 2 {
                    assert_eq!(native.stdout, evaluated.stdout);
                }
            }
        }
    }
}

#[test]
fn authored_local_callable_ascriptions_keep_alias_callback_and_shadow_execution() {
    let source = r#"
type Seed = i64 -> key
def chain() -> key = {
  first: Seed = key_from_seed
  second: Seed = first
  second(-1i64)
}
def mapped_keys() -> List[key] = {
  seed: i64 -> key = key_from_seed
  map(seed,[1i64,-1i64])
}
def mix(k: key, n: i64) -> key = k
def folded_keys() -> key = {
  mix: key -> i64 -> key = fold_in
  fold(mix,key_from_seed(7i64),[1i64,2i64])
}
def mapped_values(x: tensor[2,2,f32]) -> tensor[2,2,f32] = {
  target: (&tensor[2,f32] -> tensor[2,f32]) = abs
  vmap(target)(x)
}
chain_out = chain()
map_out = mapped_keys()
fold_out = folded_keys()
values = mapped_values(to_tensor([[1.0f32,-2.0f32],[3.0f32,-4.0f32]]))
"#;
    assert_native(
        source,
        &["values = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])"],
        "builtin_typed_local_aliases",
    );
    for rejected in [
        "def bad() -> key = { seed: f32 -> key = key_from_seed\n seed(1.0f32) }",
        "def bad() -> key = { seed: i64 -> i64 -> key = key_from_seed\n seed(1i64,2i64) }",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bad.ch");
        std::fs::write(&path, rejected).unwrap();
        Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("check")
            .arg(path)
            .assert()
            .failure();
    }
}

#[test]
fn authored_function_ascriptions_do_not_resolve_runtime_callable_choices() {
    let source = r#"
def selected(flag: bool, x: f32) -> bool = {
  f: (&f32 -> bool) = if flag then is_nan else is_finite
  f(x)
}
result = selected(false,1.0f32)
"#;
    assert!(evaluate(source).contains("result = true"));
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("choice.ch");
    std::fs::write(&path, source).unwrap();
    let result = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(directory.path().join("out"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("chelis#879"), "{error}");
    assert!(!error.contains("chelis#730"), "{error}");
}

fn canonical_contract_program(
    source: &str,
    label: &str,
) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("{label}.ch"));
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["fmt", "--inplace"])
        .arg(&path)
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["lint", "--check"])
        .arg(&path)
        .assert()
        .success();
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let feedback: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(feedback["score"].as_f64(), Some(1.0), "{feedback}");
    assert_eq!(feedback["errors"], serde_json::json!([]), "{feedback}");
    (dir, path)
}

fn assert_local_callable_contract_output(output: &std::process::Output, trap: Option<&str>) {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    match trap {
        Some(trap) => {
            assert!(!output.status.success(), "{text}");
            assert!(text.contains(trap), "{text}");
            assert!(!text.contains("after-local-claim"), "{text}");
        }
        None => {
            assert!(output.status.success(), "{text}");
            assert_eq!(text.matches("after-local-claim").count(), 1, "{text}");
        }
    }
}

/// [04-INF-9], §4.7: copying/capturing identity retains every authored entry claim.
#[test]
fn local_callable_claims_survive_alias_and_capture_before_mapped_primitive() {
    for route in ["direct", "chain", "captured", "widened", "named"] {
        for width in [2, 3] {
            let target = match route {
                "direct" | "named" => "vmap(f)(x)",
                "widened" => "g: (&tensor[*,f64] -> tensor[*,bool]) = f\nvmap(g)(x)",
                "chain" => "g = f\nh = g\nvmap(h)(x)",
                "captured" => "g = f\nwrapper = fn (z: tensor[1,*,f64]) -> vmap(g)(z)\nwrapper(x)",
                _ => unreachable!(),
            };
            let row = vec!["1.0f64"; width].join(",");
            let contract = if route == "named" {
                "Narrow"
            } else {
                "(&tensor[2,f64] -> tensor[2,bool])"
            };
            let source = format!(
                r#"
type Narrow = &tensor[2,f64] -> tensor[2,bool]
def run(x: tensor[1,*,f64]) -> tensor[1,*,bool] = {{
  f: {contract} = is_finite
  {target}
}}
result = run(to_tensor([[{row}]]))
after = print("after-local-claim")
"#
            );
            let (dir, path) = canonical_contract_program(&source, "local_callable_claim");
            let evaluated = Command::cargo_bin("chelis")
                .unwrap()
                .args(["eval", "--file"])
                .arg(&path)
                .output()
                .unwrap();
            let out = dir.path().join("native");
            Command::cargo_bin("chelis")
                .unwrap()
                .arg("build")
                .arg(&path)
                .args(["--target", "c", "--output"])
                .arg(&out)
                .assert()
                .success();
            let native = std::process::Command::new(out.join("local_callable_claim"))
                .output()
                .unwrap();
            let trap = (width == 3).then_some("numeric trap: domain in load at i64");
            for output in [&evaluated, &native] {
                assert_local_callable_contract_output(output, trap);
            }
            if width == 2 {
                assert_eq!(evaluated.stdout, native.stdout);
                assert!(
                    String::from_utf8_lossy(&evaluated.stdout)
                        .contains("result = tensor(shape=[1, 2], data=[true, true])")
                );
            }
        }
    }
}

/// [04-INF-9], §4.7: result obligations are independent of input obligations.
#[test]
fn local_callable_result_claims_survive_aliases_before_later_io() {
    for route in [
        "direct",
        "copy",
        "captured",
        "widened",
        "named",
        "cumulative",
    ] {
        for width in [2, 3] {
            let target = match route {
                "direct" | "named" => "vmap(f)(x)".to_owned(),
                "copy" => "g = copy(f)\nh = g\nvmap(h)(x)".to_owned(),
                "captured" => {
                    "g = copy(f)\nwrapper = fn (z: tensor[1,*,f64]) -> vmap(g)(z)\nwrapper(x)"
                        .to_owned()
                }
                "widened" => "g: (&tensor[*,f64] -> tensor[*,bool]) = f\nvmap(g)(x)".to_owned(),
                "cumulative" => {
                    format!("g: (&tensor[{width},f64] -> tensor[*,bool]) = f\nvmap(g)(x)")
                }
                _ => unreachable!(),
            };
            let row = vec!["1.0f64"; width].join(",");
            let contract = if route == "named" {
                "ResultOnly[2]"
            } else {
                "(&tensor[*,f64] -> tensor[2,bool])"
            };
            let source = format!(
                r#"
type ResultOnly[n] = &tensor[*,f64] -> tensor[n,bool]
def run(x: tensor[1,*,f64]) -> tensor[1,*,bool] = {{
  f: {contract} = is_finite
  {target}
}}
result = run(to_tensor([[{row}]]))
after = print("after-local-claim")
"#
            );
            let (dir, path) = canonical_contract_program(&source, "local_callable_result_claim");
            let evaluated = Command::cargo_bin("chelis")
                .unwrap()
                .args(["eval", "--file"])
                .arg(&path)
                .output()
                .unwrap();
            let out = dir.path().join("native");
            Command::cargo_bin("chelis")
                .unwrap()
                .arg("build")
                .arg(&path)
                .args(["--target", "c", "--output"])
                .arg(&out)
                .assert()
                .success();
            let native = std::process::Command::new(out.join("local_callable_result_claim"))
                .output()
                .unwrap();
            for output in [&evaluated, &native] {
                assert_local_callable_contract_output(
                    output,
                    (width == 3).then_some("numeric trap: domain in is_finite at i64"),
                );
            }
            if width == 2 {
                assert_eq!(evaluated.stdout, native.stdout);
                assert!(
                    String::from_utf8_lossy(&evaluated.stdout)
                        .contains("result = tensor(shape=[1, 2], data=[true, true])")
                );
            }
        }
    }
}

/// A same-shape result guard precedes its element kernel, not its input load.
#[test]
fn local_callable_result_claims_preserve_exact_integer_and_guard_priority() {
    for minimum in [false, true] {
        for width in [2, 3] {
            let value = if minimum {
                "-9223372036854775808i64"
            } else {
                "9007199254740993i64"
            };
            let row = vec![value; width].join(",");
            let source = format!(
                r#"
def run(x: tensor[1,*,i64]) -> tensor[1,*,i64] = {{
  f: (&tensor[*,i64] -> tensor[2,i64]) = abs
  g: (&tensor[*,i64] -> tensor[*,i64]) = copy(f)
  vmap(g)(x)
}}
result = run(to_tensor([[{row}]]))
after = print("after-local-claim")
"#
            );
            let (dir, path) = canonical_contract_program(&source, "local_callable_integer_result");
            let evaluated = Command::cargo_bin("chelis")
                .unwrap()
                .args(["eval", "--file"])
                .arg(&path)
                .output()
                .unwrap();
            let out = dir.path().join("native");
            Command::cargo_bin("chelis")
                .unwrap()
                .arg("build")
                .arg(&path)
                .args(["--target", "c", "--output"])
                .arg(&out)
                .assert()
                .success();
            let native = std::process::Command::new(out.join("local_callable_integer_result"))
                .output()
                .unwrap();
            let trap = if width == 3 {
                Some("numeric trap: domain in abs at i64")
            } else if minimum {
                Some("numeric trap: overflow in abs at i64")
            } else {
                None
            };
            for output in [&evaluated, &native] {
                assert_local_callable_contract_output(output, trap);
            }
            if trap.is_none() {
                assert_eq!(evaluated.stdout, native.stdout);
                assert!(String::from_utf8_lossy(&evaluated.stdout).contains(
                    "result = tensor(shape=[1, 2], data=[9007199254740993, 9007199254740993])"
                ));
            }
        }
    }
}

/// The IO evaluator must check the local entry contract before the later effect.
#[test]
fn local_callable_claims_guard_direct_host_invocation_before_io() {
    for width in [2, 3] {
        let row = vec!["1.0f64"; width].join(",");
        let source = format!(
            r#"
def run(x: tensor[*,f64]) -> tensor[*,bool] ! {{IO}} = {{
  f: (&tensor[2,f64] -> tensor[2,bool]) = is_finite
  y = f(x)
  _ = print("after-local-claim")
  y
}}
result = run(to_tensor([{row}]))
"#
        );
        let (dir, path) = canonical_contract_program(&source, "local_callable_io_claim");
        let evaluated = Command::cargo_bin("chelis")
            .unwrap()
            .args(["eval", "--file"])
            .arg(&path)
            .output()
            .unwrap();
        assert_local_callable_contract_output(
            &evaluated,
            (width == 3).then_some("numeric trap: domain in load at i64"),
        );
        let built = Command::cargo_bin("chelis")
            .unwrap()
            .arg("build")
            .arg(&path)
            .args(["--target", "c", "--output"])
            .arg(dir.path().join("out"))
            .output()
            .unwrap();
        assert!(built.status.success(), "{built:?}");
        let native = std::process::Command::new(dir.path().join("out/local_callable_io_claim"))
            .output()
            .unwrap();
        assert_local_callable_contract_output(
            &native,
            (width == 3).then_some("numeric trap: domain in load at i64"),
        );
        if width == 2 {
            assert_eq!(native.stdout, evaluated.stdout);
        }
    }
}

/// A false callable width fails at ingress before the primitive's own overflow.
#[test]
fn local_callable_claims_precede_signed_minimum_overflow() {
    for width in [2, 3] {
        let row = vec!["-9223372036854775808i64"; width].join(",");
        let source = format!(
            r#"
def run(x: tensor[1,*,i64]) -> tensor[1,*,i64] = {{
  f: (&tensor[2,i64] -> tensor[2,i64]) = abs
  g = f
  vmap(g)(x)
}}
result = run(to_tensor([[{row}]]))
after = print("after-local-claim")
"#
        );
        let (dir, path) = canonical_contract_program(&source, "local_callable_overflow_claim");
        let evaluated = Command::cargo_bin("chelis")
            .unwrap()
            .args(["eval", "--file"])
            .arg(&path)
            .output()
            .unwrap();
        let out = dir.path().join("native");
        Command::cargo_bin("chelis")
            .unwrap()
            .arg("build")
            .arg(&path)
            .args(["--target", "c", "--output"])
            .arg(&out)
            .assert()
            .success();
        let native = std::process::Command::new(out.join("local_callable_overflow_claim"))
            .output()
            .unwrap();
        let trap = if width == 3 {
            "numeric trap: domain in load at i64"
        } else {
            "numeric trap: overflow in abs at i64"
        };
        for output in [&evaluated, &native] {
            assert_local_callable_contract_output(output, Some(trap));
        }
    }
}

#[test]
fn local_callable_ascriptions_reject_dtype_family_and_result_precision_changes() {
    for source in [
        "def run(x: tensor[1,*,f32]) -> tensor[1,*,bool] = { f: (&tensor[2,f64] -> tensor[2,bool]) = is_finite\n vmap(f)(x) }",
        "def run(x: tensor[1,*,i64]) -> tensor[1,*,bool] = { f: (&tensor[2,i64] -> tensor[2,bool]) = is_finite\n vmap(f)(x) }",
        "def run(x: tensor[1,*,f64]) -> tensor[1,*,f64] = { f: (&tensor[2,f64] -> tensor[2,f64]) = is_finite\n vmap(f)(x) }",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid_local_callable.ch");
        std::fs::write(&path, source).unwrap();
        Command::cargo_bin("chelis")
            .unwrap()
            .args(["fmt", "--inplace"])
            .arg(&path)
            .assert()
            .success();
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .arg("check")
            .arg(&path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let feedback: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(feedback["components"]["parse"], 1);
        assert_eq!(feedback["components"]["names"], 1);
        assert!(
            feedback["components"]["types"].as_f64().unwrap() < 1.0,
            "{feedback}"
        );
        assert!(
            !feedback["errors"].as_array().unwrap().is_empty(),
            "{feedback}"
        );
    }
}

/// Independent invocation obligations cannot disappear at a static value boundary.
/// The wildcard outer signature deliberately cannot discharge either claim.
fn round2_assert_callable_contract(
    source: &str,
    label: &str,
    trap: Option<&str>,
    expected_result: Option<&str>,
    eval: bool,
    native: bool,
) {
    let (dir, path) = canonical_contract_program(source, label);
    let mut outputs = Vec::new();
    let mut lanes = Vec::new();
    if eval {
        lanes.push("Eval");
        outputs.push(
            Command::cargo_bin("chelis")
                .unwrap()
                .args(["eval", "--file"])
                .arg(&path)
                .output()
                .unwrap(),
        );
    }
    if native {
        lanes.push("native");
        let out = dir.path().join("native");
        Command::cargo_bin("chelis")
            .unwrap()
            .arg("build")
            .arg(&path)
            .args(["--target", "c", "--output"])
            .arg(&out)
            .assert()
            .success();
        outputs.push(
            std::process::Command::new(out.join(label))
                .output()
                .unwrap(),
        );
    }
    assert!(!outputs.is_empty(), "no lane selected");
    for (lane, output) in lanes.iter().zip(&outputs) {
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        match trap {
            Some(trap) => {
                assert!(!output.status.success(), "{label} {lane}: {text}");
                assert!(text.contains(trap), "{label} {lane}: {text}");
                assert!(!text.contains("after-probe"), "{label} {lane}: {text}");
            }
            None => {
                assert!(output.status.success(), "{label} {lane}: {text}");
                assert_eq!(
                    text.matches("after-probe").count(),
                    1,
                    "{label} {lane}: {text}"
                );
                assert!(
                    text.contains(expected_result.unwrap()),
                    "{label} {lane}: {text}"
                );
            }
        }
    }
    if trap.is_none() && outputs.len() == 2 {
        assert_eq!(outputs[0].stdout, outputs[1].stdout, "{label}");
    }
}

/// [04-INF-9], §4.7: a result-only formal still owns a producer obligation
/// when its input entry plan contains no guards. Axis1 maps physical rows
/// of length3; forcing axis0 instead would incorrectly produce rows of length2.
#[test]
fn whole_callable_formal_result_obligations_survive_inline_and_forwarded_specialization() {
    for forwarded in [false, true] {
        for width in [2, 3] {
            let outer = if forwarded {
                "def outer(h: &tensor[*,f32] -> tensor[*,bool], x: tensor[*,2,f32]) -> tensor[*,2,bool] = inner(h,x)"
            } else {
                ""
            };
            let caller = if forwarded { "outer" } else { "inner" };
            let rows = vec!["[1.0f32,2.0f32]"; width].join(",");
            let source = format!(
                r#"
def inner(h: &tensor[*,f32] -> tensor[2,bool], x: tensor[*,2,f32]) -> tensor[*,2,bool] = vmap(h,axis=1)(x)
{outer}
result = {caller}(is_finite,to_tensor([{rows}]))
after = print("after-probe")
"#
            );
            round2_assert_callable_contract(
                &source,
                "whole_callable_formal_result",
                (width == 3).then_some("numeric trap: domain in is_finite at i64"),
                Some("result = tensor(shape=[2, 2], data=[true, true, true, true])"),
                true,
                true,
            );
        }
    }
}

/// Negative parity: an input-only contract rejects before the operation;
/// the independent wildcard result cannot mask or authorize its entry.
#[test]
fn whole_callable_formal_input_obligations_remain_independent_from_results() {
    for forwarded in [false, true] {
        for width in [2, 3] {
            let outer = if forwarded {
                "def outer(h: &tensor[*,f32] -> tensor[*,bool], x: tensor[*,2,f32]) -> tensor[*,2,bool] = inner(h,x)"
            } else {
                ""
            };
            let caller = if forwarded { "outer" } else { "inner" };
            let rows = vec!["[1.0f32,2.0f32]"; width].join(",");
            let source = format!(
                r#"
def inner(h: &tensor[2,f32] -> tensor[*,bool], x: tensor[*,2,f32]) -> tensor[*,2,bool] = vmap(h,axis=1)(x)
{outer}
result = {caller}(is_finite,to_tensor([{rows}]))
after = print("after-probe")
"#
            );
            round2_assert_callable_contract(
                &source,
                "whole_callable_formal_input",
                (width == 3).then_some("numeric trap: domain in load at i64"),
                Some("result = tensor(shape=[2, 2], data=[true, true, true, true])"),
                true,
                true,
            );
        }
    }
}

/// Named callable formals have the same entire invocation contract.
/// These recorded routes support native execution; their Eval route loudly
/// refused in round2 and is deliberately not represented as an evaluated result.
#[test]
fn whole_callable_named_formals_retain_independent_native_obligations() {
    for result_only in [false, true] {
        for forwarded in [false, true] {
            for width in [2, 3] {
                let signature = if result_only {
                    "&tensor[*,f32] -> tensor[n,bool]"
                } else {
                    "&tensor[n,f32] -> tensor[*,bool]"
                };
                let outer = if forwarded {
                    "def outer(h: &tensor[*,f32] -> tensor[*,bool], x: tensor[*,2,f32]) -> tensor[*,2,bool] = inner(h,x)"
                } else {
                    ""
                };
                let caller = if forwarded { "outer" } else { "inner" };
                let rows = vec!["[1.0f32,2.0f32]"; width].join(",");
                let source = format!(
                    r#"
type Contract[n] = {signature}
def inner(h: Contract[2], x: tensor[*,2,f32]) -> tensor[*,2,bool] = vmap(h,axis=1)(x)
{outer}
result = {caller}(is_finite,to_tensor([{rows}]))
after = print("after-probe")
"#
                );
                let trap = if result_only {
                    "numeric trap: domain in is_finite at i64"
                } else {
                    "numeric trap: domain in load at i64"
                };
                round2_assert_callable_contract(
                    &source,
                    "whole_callable_named_formal",
                    (width == 3).then_some(trap),
                    Some("result = tensor(shape=[2, 2], data=[true, true, true, true])"),
                    false,
                    true,
                );
            }
        }
    }
}

/// Global resolution must inherit predecessor obligations, including through
/// Copy, widening and lexical capture, rather than reconstruct just identity.
/// Global C ownership refusals are not native qualification of these programs.
#[test]
fn whole_callable_global_aliases_retain_inherited_input_obligations_in_eval() {
    for route in ["direct", "alias", "copy", "widened", "captured"] {
        for width in [2, 3] {
            let body = match route {
                "direct" => "vmap(f)(x)",
                "alias" => "vmap(g)(x)",
                "copy" => "vmap(h)(x)",
                "widened" => "vmap(wide)(x)",
                "captured" => "{ wrapper = fn (z: tensor[1,*,f64]) -> vmap(g)(z)\n wrapper(x) }",
                _ => unreachable!(),
            };
            let row = vec!["1.0f64"; width].join(",");
            let source = format!(
                r#"
f: &tensor[2,f64] -> tensor[*,bool] = is_finite
g = f
h = copy(g)
wide: &tensor[*,f64] -> tensor[*,bool] = g
def run(x: tensor[1,*,f64]) -> tensor[1,*,bool] = {body}
result = run(to_tensor([[{row}]]))
after = print("after-probe")
"#
            );
            round2_assert_callable_contract(
                &source,
                "whole_callable_global_input",
                (width == 3).then_some("numeric trap: domain in load at i64"),
                Some("result = tensor(shape=[1, 2], data=[true, true])"),
                true,
                false,
            );
        }
    }
}

/// Result obligations are independent of entry; exact integers above2^53
/// exclude a Float funnel, and false result width must precede abs overflow.
#[test]
fn whole_callable_global_result_obligations_preserve_exact_integer_priority_in_eval() {
    for route in ["direct", "alias", "copy", "widened", "captured"] {
        for minimum in [false, true] {
            for width in [2, 3] {
                let body = match route {
                    "direct" => "vmap(f)(x)",
                    "alias" => "vmap(g)(x)",
                    "copy" => "vmap(h)(x)",
                    "widened" => "vmap(wide)(x)",
                    "captured" => {
                        "{ wrapper = fn (z: tensor[1,*,i64]) -> vmap(g)(z)\n wrapper(x) }"
                    }
                    _ => unreachable!(),
                };
                let value = if minimum {
                    "-9223372036854775808i64"
                } else {
                    "9007199254740993i64"
                };
                let row = vec![value; width].join(",");
                let source = format!(
                    r#"
f: &tensor[*,i64] -> tensor[2,i64] = abs
g = f
h = copy(g)
wide: &tensor[*,i64] -> tensor[*,i64] = g
def run(x: tensor[1,*,i64]) -> tensor[1,*,i64] = {body}
result = run(to_tensor([[{row}]]))
after = print("after-probe")
"#
                );
                let trap = if width == 3 {
                    Some("numeric trap: domain in abs at i64")
                } else if minimum {
                    Some("numeric trap: overflow in abs at i64")
                } else {
                    None
                };
                round2_assert_callable_contract(
                    &source,
                    "whole_callable_global_result",
                    trap,
                    Some(
                        "result = tensor(shape=[1, 2], data=[9007199254740993, 9007199254740993])",
                    ),
                    true,
                    false,
                );
            }
        }
    }
}

/// Every adapter adds its own complete contract rather than replacing that of
/// the supplied value: a valid new input cannot erase the earlier false result.
#[test]
fn whole_callable_global_cumulative_contracts_keep_independent_entry_and_result_in_eval() {
    for narrow_input in [false, true] {
        for width in [2, 3] {
            let (original, adapted, trap) = if narrow_input {
                (
                    "&tensor[2,f64] -> tensor[*,bool]",
                    "&tensor[*,f64] -> tensor[2,bool]".to_owned(),
                    "numeric trap: domain in load at i64",
                )
            } else {
                (
                    "&tensor[*,f64] -> tensor[2,bool]",
                    format!("&tensor[{width},f64] -> tensor[*,bool]"),
                    "numeric trap: domain in is_finite at i64",
                )
            };
            let row = vec!["1.0f64"; width].join(",");
            let source = format!(
                r#"
f: {original} = is_finite
g: {adapted} = f
h = copy(g)
def run(x: tensor[1,*,f64]) -> tensor[1,*,bool] = vmap(h)(x)
result = run(to_tensor([[{row}]]))
after = print("after-probe")
"#
            );
            round2_assert_callable_contract(
                &source,
                "whole_callable_global_cumulative",
                (width == 3).then_some(trap),
                Some("result = tensor(shape=[1, 2], data=[true, true])"),
                true,
                false,
            );
        }
    }
}

/// [04-INF-6], §4.7: independent result-only frames are actualized at each
/// f32/f64 call site; a false claim traps at abs before later caller IO.
#[test]
fn whole_callable_generic_formals_actualize_each_call_and_keep_result_priority() {
    for literal in [false, true] {
        for width in [2, 3] {
            let first_target = if literal {
                "fn (z: &tensor[*, f32]) -> abs(z)"
            } else {
                "abs"
            };
            let second_target = if literal {
                "fn (z: &tensor[*, f64]) -> abs(z)"
            } else {
                "abs"
            };
            let values = if width == 2 {
                "[16777217.0f64, -2.0f64]"
            } else {
                "[16777217.0f64, -2.0f64, 3.0f64]"
            };
            let source = format!(
                "def invoke[p: Float](f: (&tensor[*, p] -> tensor[2, p]), x: tensor[*, p]) -> tensor[*, p] = f(x)\n\
                 first = invoke({first_target}, to_tensor([-1.0f32, 2.0f32]))\n\
                 out = invoke({second_target}, to_tensor({values}))\n\
                 shown = print(\"after-probe\")\n"
            );
            let label = format!("generic_callable_{literal}_{width}");
            let expected = "out = tensor(shape=[2], data=[16777217.0, 2.0])";
            round2_assert_callable_contract(
                &source,
                &label,
                (width == 3).then_some("numeric trap: domain in abs at i64"),
                (width == 2).then_some(expected),
                true,
                // Plain-Fn native result-only signature authority is a
                // separately recorded inherited gap; builtin frames use the
                // whole-value owner exercised by this target.
                !literal,
            );
        }
    }
}

/// A specialized DAG sibling cannot supply static identity to a different
/// host caller whose transform target is selected at runtime.
#[test]
fn specialized_dag_entry_does_not_hide_a_remaining_runtime_transform_caller() {
    let source = r#"
def mapped(h: (&tensor[2, f32] -> tensor[2, bool]), x: tensor[2, 2, f32]) -> tensor[2, 2, bool] = vmap(h)(x)
def static_entry(x: tensor[2, 2, f32]) -> tensor[2, 2, bool] = mapped(is_finite, x)
def remaining(flag: bool, x: tensor[2, 2, f32]) -> tensor[2, 2, bool] = {
  h = if flag then is_nan else is_finite
  mapped(h, x)
}
"#;
    let (dir, path) = canonical_contract_program(source, "remaining_transform");
    let destination = dir.path().join("native");
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&destination)
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "runtime target was silently omitted: {error}"
    );
    // This control proves native refusal without an artifact. The existing
    // [05-HOST-1] rendering is not evidence of target-diagnostic conformance.
    assert!(
        !error.is_empty(),
        "native refusal must explain the remaining caller"
    );
    assert!(!error.contains("chelis#730"), "{error}");
    assert!(!destination.exists() || std::fs::read_dir(destination).unwrap().next().is_none());
}

#[test]
fn retained_host_builtin_frames_check_nested_entry_before_effects() {
    for width in [2, 3] {
        let elements = vec!["1.0f32"; width].join(",");
        let source = format!(
            "def invoke(h: (&List[tensor[2,f32]] -> i64), x: List[tensor[*,f32]]) -> i64 = h(x)\nresult=invoke(len,[to_tensor([{elements}])])\nafter=print(\"after-probe\")\n"
        );
        if width == 2 {
            assert_native(
                &source,
                &["result = 1", "after-probe"],
                "retained_host_entry_ok",
            );
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("retained_host_entry.ch");
        std::fs::write(&path, source).unwrap();
        let eval = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(!eval.status.success());
        assert!(!String::from_utf8_lossy(&eval.stdout).contains("after-probe"));
        let stderr = String::from_utf8_lossy(&eval.stderr);
        assert!(stderr.contains("numeric trap: domain"), "{stderr}");
        assert!(stderr.contains("arg0[0]"), "{stderr}");
        let out = dir.path().join("out");
        Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("build")
            .arg(&path)
            .args(["--target", "c", "--output"])
            .arg(&out)
            .assert()
            .success();
        let native = std::process::Command::new(out.join("retained_host_entry"))
            .output()
            .unwrap();
        assert!(!native.status.success(), "{native:?}");
        assert!(!String::from_utf8_lossy(&native.stdout).contains("after-probe"));
        assert!(
            String::from_utf8_lossy(&native.stderr).contains("numeric trap: domain"),
            "{native:?}"
        );
    }
}

#[test]
fn retained_builtin_result_extent_precedes_element_overflow() {
    for binding in [
        "f: (&tensor[*,i64] -> tensor[2,i64]) = abs",
        "local = abs\nf: (&tensor[*,i64] -> tensor[2,i64]) = local",
    ] {
        let source = format!(
            "def run(x: tensor[*,i64]) -> tensor[*,i64] ! {{IO}} = {{\n{binding}\ny=f(x)\n_=print(\"after-probe\")\ny\n}}\nresult=run(to_tensor([-9223372036854775808i64,-9223372036854775808i64,-9223372036854775808i64]))"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("retained_priority.ch");
        std::fs::write(&path, source).unwrap();
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("numeric trap: domain in abs at i64"),
            "{stderr}"
        );
        assert!(!stderr.contains("numeric trap: overflow"), "{stderr}");
        assert!(!String::from_utf8_lossy(&output.stdout).contains("after-probe"));
        let valid = std::fs::read_to_string(&path).unwrap().replace(
            "-9223372036854775808i64,-9223372036854775808i64,-9223372036854775808i64",
            "-9007199254740993i64,9007199254740995i64",
        );
        assert!(evaluate(&valid).contains("data=[9007199254740993, 9007199254740995]"));
    }
}

#[test]
fn builtin_value_selection_retains_the_actual_selected_producer() {
    for callee in ["index", "picker"] {
        let source = format!(
            r#"
picker = index
def run[n](x: tensor[n,4,f32]) -> tensor[3,f32] ! {{IO}} = {{
 _=print("before-probe")
 values: List[tensor[*,f32]] = [diagonal(x,0i32,1i32),cumsum(diagonal(x,0i32,1i32),0i32)]
 tail=skip(values,1i64)
 selected={callee}(tail,0i64)
 _=print("after-probe")
 selected
}}
result=run(to_tensor([[1.0f32,2.0f32,3.0f32,4.0f32],[1.0f32,2.0f32,3.0f32,4.0f32]]))
"#
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("retained_projection.ch");
        std::fs::write(&path, source).unwrap();
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("numeric trap: domain in cumsum at i64"),
            "{stderr}"
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("before-probe"), "{stdout}");
    }
}

#[test]
fn retained_host_builtin_result_frames_survive_selection_and_unused_callbacks() {
    for body in [
        "h(x,0i64)",
        "{ pair = (h,0i64)\n alias = copy(pair)\n (selected,n) = alias\n selected(x,n) }",
    ] {
        for (input, result, nesting) in [
            ("List[tensor[*,f32]]", "tensor[2,f32]", false),
            ("List[List[tensor[*,f32]]]", "List[tensor[2,f32]]", true),
        ] {
            for width in [2, 3] {
                let elements = vec!["1.0f32"; width].join(",");
                let value = if nesting {
                    format!("[[to_tensor([{elements}])]]")
                } else {
                    format!("[to_tensor([{elements}])]")
                };
                let outer_result = if nesting {
                    "List[tensor[*,f32]]"
                } else {
                    "tensor[*,f32]"
                };
                let source = if body == "h(x,0i64)" {
                    format!(
                        "def invoke(h: (&{input} -> i64 -> {result}), x: {input}) -> {outer_result} = {body}\nresult=invoke(index,{value})\nafter=print(\"after-probe\")"
                    )
                } else {
                    format!(
                        "def invoke(x: {input}) -> {outer_result} ! {{IO}} = {{\n _=print(\"before-probe\")\n h: (&{input} -> i64 -> {result}) = index\n {body}\n}}\nresult=invoke({value})\nafter=print(\"after-probe\")"
                    )
                };
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("retained_host_result.ch");
                std::fs::write(&path, &source).unwrap();
                let eval = Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .args(["eval", "--file"])
                    .arg(&path)
                    .output()
                    .unwrap();
                let out = dir.path().join("out");
                Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .arg("build")
                    .arg(&path)
                    .args(["--target", "c", "--output"])
                    .arg(&out)
                    .assert()
                    .success();
                let native = std::process::Command::new(out.join("retained_host_result"))
                    .output()
                    .unwrap();
                if width == 2 {
                    assert!(eval.status.success(), "{eval:?}");
                    assert!(native.status.success(), "{native:?}");
                    assert_eq!(native.stdout, eval.stdout);
                } else {
                    for output in [&eval, &native] {
                        assert!(!output.status.success(), "{output:?}");
                        assert!(
                            String::from_utf8_lossy(&output.stderr)
                                .contains("numeric trap: domain"),
                            "{output:?}"
                        );
                        assert!(!String::from_utf8_lossy(&output.stdout).contains("after-probe"));
                    }
                }
                let unused = format!(
                    "def invoke(h: (&{input} -> i64 -> {result}), x: {input}) -> i64 = len(x)\nresult=invoke(index,{value})"
                );
                assert_native(&unused, &["result = 1"], "unused_retained_host_result");
            }
        }
    }
}

#[test]
fn retained_builtin_control_preconditions_precede_result_extent() {
    for (operation, signature, controls) in [
        (
            "dropout",
            "key -> &tensor[*,f32] -> f32 -> tensor[2,f32]",
            "-1.0f32",
        ),
        (
            "uniform_like",
            "key -> &tensor[*,f32] -> f32 -> f32 -> tensor[2,f32]",
            "2.0f32,1.0f32",
        ),
    ] {
        let source = format!(
            "def run(x: tensor[*,f32]) -> tensor[*,f32] ! {{IO}} = {{\nf: ({signature}) = {operation}\ny=f(key_from_seed(42i64),x,{controls})\n_=print(\"after-probe\")\ny\n}}\nresult=run(to_tensor([1.0f32,2.0f32,3.0f32]))"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("retained_control.ch");
        std::fs::write(&path, source).unwrap();
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("numeric trap: domain in {operation}")),
            "{stderr}"
        );
        assert!(!stderr.contains("extent `2`"), "{stderr}");
        assert!(!String::from_utf8_lossy(&output.stdout).contains("after-probe"));
    }
}
