//! Chelis-Lang/chelis#3362: `grad` through `diagonal`, `trace`, `cumsum`, and
//! `einsum` follows each atom's stated adjoint ([05-OP-53], [05-OP-33],
//! [05-OP-51]) in eval and compiled C at every float dtype.
//!
//! Every `a_<case>` root is a transformed value and every `e_<case>` root is
//! the atom's adjoint (or forward value, for the `vmap` cases) computed by the
//! untransformed host kernels: the reverse inclusive scan is a host `cumsum`
//! of the reversed cotangent, the einsum adjoint is the host contraction of
//! the cotangent with the other operand in forward output-then-reduction
//! order, and the diagonal adjoint is the scattered literal. The printed
//! values are shortest round-trip spellings, so equal text is equal bits.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;

const CASES: &str = r#"def cs1(x: tensor[4, P], w: tensor[4, P]) -> tensor[P] = sum(mul(cumsum(x, 0i32), w), 0i32)
def cs2(x: tensor[2, 7, P], w: tensor[2, 7, P]) -> tensor[P] = sum(sum(mul(cumsum(x, -1i32), w), 1i32), 0i32)
def cs3(x: tensor[3, 1, P], w: tensor[3, 1, P]) -> tensor[P] = sum(sum(mul(cumsum(x, 1i32), w), 1i32), 0i32)
def cs4(x: tensor[3, 2, P], w: tensor[3, 2, P]) -> tensor[P] = sum(sum(mul(cumsum(x, 0i32), w), 1i32), 0i32)
def dg1(x: tensor[2, 2, P], w: tensor[2, P]) -> tensor[P] = sum(mul(diagonal(x, 0i32, 1i32), w), 0i32)
def dg2(x: tensor[2, 3, 4, P], w: tensor[3, 2, P]) -> tensor[P] = sum(sum(mul(diagonal(x, 2i32, 0i32), w), 1i32), 0i32)
def es1(a: tensor[2, 3, P], b: tensor[3, 2, P], w: tensor[2, 2, P]) -> tensor[P] = sum(sum(mul(einsum("ij,jk->ik", a, b), w), 1i32), 0i32)
def es2(a: tensor[2, 3, P], b: tensor[2, 4, P], w: tensor[2, 4, P]) -> tensor[P] = sum(sum(mul(einsum("ij,kl->il", a, b), w), 1i32), 0i32)
def es3(a: tensor[3, 3, P], b: tensor[3, P], w: tensor[3, P]) -> tensor[P] = sum(mul(einsum("ii,i->i", a, b), w), 0i32)
def es4(a: tensor[2, 3, P], b: tensor[3, P]) -> tensor[P] = einsum("ij,j->", a, b)
def es5(a: tensor[2, 3, P], b: tensor[3, 2, P]) -> tensor[P] = cast(sum(sum(einsum("ij,jk->ik", a, b, accumulator=f64), 1i32), 0i32), P)
def dgv(x: tensor[3, 4, P]) -> tensor[3, P] = diagonal(x, 1i32, 0i32)
def csv(x: tensor[7, P]) -> tensor[7, P] = cumsum(x, 0i32)
def esv(a: tensor[3, P], b: tensor[3, 2, P]) -> tensor[2, P] = einsum("j,jk->k", a, b)
def tr1(x: tensor[2, 3, 4, P], w: tensor[3, P]) -> tensor[P] = sum(mul(trace(x, 2i32, 0i32), w), 0i32)
def trv(x: tensor[3, 4, P]) -> tensor[P] = trace(x, 1i32, 0i32)
v4 = to_tensor([0.1P, 0.7P, 1.3P, 2.9P])
w4 = to_tensor([0.3P, -1.7P, 2.2P, 0.9P])
r4 = to_tensor([3i32, 2i32, 1i32, 0i32])
a_cs1 = grad(cs1, wrt=x)(v4, w4)
e_cs1 = gather(cumsum(gather(w4, r4, 0i32), 0i32), r4, 0i32)
v27 = to_tensor([[0.5P, -1.25P, 3.0P, 0.125P, -2.5P, 1.0P, 0.75P], [1.5P, 2.25P, -0.5P, 4.0P, 0.25P, -3.0P, 1.125P]])
w27 = to_tensor([[1.1P, -0.3P, 2.7P, 0.4P, -1.9P, 3.3P, 0.6P], [-2.1P, 0.8P, 1.7P, -0.2P, 2.9P, 0.05P, -1.4P]])
r7 = to_tensor([6i32, 5i32, 4i32, 3i32, 2i32, 1i32, 0i32])
a_cs2 = grad(cs2, wrt=x)(v27, w27)
e_cs2 = gather(cumsum(gather(w27, r7, 1i32), 1i32), r7, 1i32)
v31 = to_tensor([[1.0P], [-2.0P], [3.0P]])
w31 = to_tensor([[-0.0P], [0.5P], [-1.5P]])
a_cs3 = grad(cs3, wrt=x)(v31, w31)
e_cs3 = cumsum(w31, 1i32)
v32 = to_tensor([[1.0P, 2.0P], [3.0P, 4.0P], [5.0P, 6.0P]])
w32 = to_tensor([[0.7P, -0.1P], [1.9P, 2.3P], [-0.6P, 0.45P]])
r3 = to_tensor([2i32, 1i32, 0i32])
a_cs4 = grad(cs4, wrt=x)(v32, w32)
e_cs4 = gather(cumsum(gather(w32, r3, 0i32), 0i32), r3, 0i32)
a_dg1 = grad(dg1, wrt=x)(to_tensor([[1.0P, 2.0P], [3.0P, 4.0P]]), to_tensor([0.3P, -1.7P]))
e_dg1 = to_tensor([[0.3P, 0.0P], [0.0P, -1.7P]])
x234 = to_tensor([[[1.0P, 2.0P, 3.0P, 4.0P], [5.0P, 6.0P, 7.0P, 8.0P], [9.0P, 10.0P, 11.0P, 12.0P]], [[13.0P, 14.0P, 15.0P, 16.0P], [17.0P, 18.0P, 19.0P, 20.0P], [21.0P, 22.0P, 23.0P, 24.0P]]])
w32b = to_tensor([[0.5P, -1.5P], [2.5P, 0.25P], [-0.75P, 3.5P]])
a_dg2 = grad(dg2, wrt=x)(x234, w32b)
e_dg2 = to_tensor([[[0.5P, 0.0P, 0.0P, 0.0P], [2.5P, 0.0P, 0.0P, 0.0P], [-0.75P, 0.0P, 0.0P, 0.0P]], [[0.0P, -1.5P, 0.0P, 0.0P], [0.0P, 0.25P, 0.0P, 0.0P], [0.0P, 3.5P, 0.0P, 0.0P]]])
f_dg2 = diagonal(x234, 2i32, 0i32)
a23 = to_tensor([[1.5P, -2.0P, 0.25P], [3.0P, 0.5P, -1.0P]])
b32 = to_tensor([[0.1P, 2.0P], [-0.7P, 1.3P], [2.2P, -0.4P]])
w22 = to_tensor([[0.9P, -1.1P], [0.35P, 2.6P]])
a_es1a = grad(es1, wrt=a)(a23, b32, w22)
e_es1a = einsum("ik,jk->ij", w22, b32)
a_es1b = grad(es1, wrt=b)(a23, b32, w22)
e_es1b = einsum("ij,ik->jk", a23, w22)
b24 = to_tensor([[0.1P, 2.0P, -0.7P, 1.3P], [2.2P, -0.4P, 0.6P, 1.9P]])
w24 = to_tensor([[0.9P, -1.1P, 0.35P, 2.6P], [1.7P, 0.2P, -0.8P, 0.55P]])
a_es2a = grad(es2, wrt=a)(a23, b24, w24)
e_es2a = insert(einsum("il,kl->i", w24, b24), 1i32, 3i64)
a_es2b = grad(es2, wrt=b)(a23, b24, w24)
e_es2b = insert(einsum("ij,il->l", a23, w24), 0i32, 2i64)
a33 = to_tensor([[1.0P, 2.0P, 3.0P], [4.0P, 5.0P, 6.0P], [7.0P, 8.0P, 9.0P]])
b3 = to_tensor([0.5P, -1.5P, 2.5P])
w3 = to_tensor([1.25P, 0.75P, -2.0P])
eye = to_tensor([[true, false, false], [false, true, false], [false, false, true]])
z33 = to_tensor([[0.0P, 0.0P, 0.0P], [0.0P, 0.0P, 0.0P], [0.0P, 0.0P, 0.0P]])
a_es3a = grad(es3, wrt=a)(a33, b3, w3)
e_es3a = where(eye, insert(einsum("i,i->i", w3, b3), 1i32, 3i64), z33)
a_es3b = grad(es3, wrt=b)(a33, b3, w3)
e_es3b = einsum("i,i->i", w3, diagonal(a33, 0i32, 1i32))
a_es4 = grad(es4, wrt=b)(a23, b3)
e_es4 = sum(a23, 0i32)
o22 = to_tensor([[1.0P, 1.0P], [1.0P, 1.0P]])
a_es5 = grad(es5, wrt=a)(a23, b32)
e_es5 = cast(einsum("ik,jk->ij", o22, b32, accumulator=f64), P)
a_fdg = vmap(dgv)(x234)
e_fdg = diagonal(x234, 2i32, 1i32)
a_fcs = vmap(csv)(v27)
e_fcs = cumsum(v27, 1i32)
b232 = to_tensor([[[0.1P, 2.0P], [-0.7P, 1.3P], [2.2P, -0.4P]], [[1.9P, -0.6P], [0.3P, 0.8P], [-1.2P, 2.4P]]])
a_fes = vmap(esv)(a23, b232)
e_fes = einsum("zj,zjk->zk", a23, b232)
a_tr1 = grad(tr1, wrt=x)(x234, to_tensor([0.5P, -1.5P, 2.5P]))
e_tr1 = to_tensor([[[0.5P, 0.0P, 0.0P, 0.0P], [-1.5P, 0.0P, 0.0P, 0.0P], [2.5P, 0.0P, 0.0P, 0.0P]], [[0.0P, 0.5P, 0.0P, 0.0P], [0.0P, -1.5P, 0.0P, 0.0P], [0.0P, 2.5P, 0.0P, 0.0P]]])
a_ftr = vmap(trv)(x234)
e_ftr = trace(x234, 2i32, 1i32)
"#;

fn eval_text(reef: &std::path::Path, app: &std::path::Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef)
        .current_dir(app)
        .args(["eval", "--file", "src/main.ch"])
        .output()
        .unwrap()
}

fn root_pairs(stdout: &str) -> Vec<(String, String, String)> {
    let roots: Vec<(&str, &str)> = stdout
        .lines()
        .filter_map(|line| line.split_once(" = "))
        .collect();
    roots
        .iter()
        .filter_map(|(name, value)| {
            let case = name.strip_prefix("a_")?;
            let expected = roots
                .iter()
                .find(|(other, _)| other.strip_prefix("e_") == Some(case))
                .unwrap_or_else(|| panic!("no expected root for `{name}`"))
                .1;
            Some((case.to_string(), (*value).to_string(), expected.to_string()))
        })
        .collect()
}

#[test]
fn ordered_adjoints_match_the_atoms_in_eval_and_c_at_every_float_width() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let (_dir, reef, app) = common::make_app("issue-3362");
        let program = format!("module Demo.Main\n{}", CASES.replace('P', dtype));
        common::write_file(&app.join("src/main.ch"), &program);
        let output = eval_text(&reef, &app);
        assert!(output.status.success(), "{dtype}: {output:?}");
        let evaluated = String::from_utf8(output.stdout).unwrap();
        let pairs = root_pairs(&evaluated);
        assert_eq!(
            pairs.len(),
            19,
            "{dtype}: every case is present\n{evaluated}"
        );
        for (case, actual, expected) in &pairs {
            assert_eq!(actual, expected, "eval {dtype} `{case}`");
        }
        let native = common::build_and_run_app(&reef, &app, "main");
        assert_eq!(
            native, evaluated,
            "{dtype}: compiled C agrees with eval on every root"
        );
    }
}

fn scalar_root(stdout: &str, name: &str) -> f64 {
    let prefix = format!("{name} = ");
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{name}` root in:\n{stdout}"))
        .parse()
        .unwrap_or_else(|_| panic!("`{name}` is not a printed scalar in:\n{stdout}"))
}

/// Central differences at f64 for a nonlinear loss over each operation.
#[test]
fn ordered_adjoints_match_central_differences_at_f64() {
    let h = 1e-6;
    for (signature, body, shape, point, extra) in [
        (
            "x: tensor[4, f64]",
            "sum(mul(sin(cumsum(x, 0i32)), to_tensor([0.3f64, -1.7f64, 2.2f64, 0.9f64])), 0i32)",
            vec![4],
            vec![0.1, 0.7, -1.3, 2.9],
            "",
        ),
        (
            "x: tensor[2, 3, f64]",
            "sum(mul(sin(diagonal(x, 1i32, 0i32)), to_tensor([0.5f64, -1.5f64])), 0i32)",
            vec![2, 3],
            vec![0.2, -0.4, 1.1, 0.8, 0.3, -0.9],
            "",
        ),
        (
            "x: tensor[3, 3, f64]",
            "sin(trace(x, 0i32, 1i32))",
            vec![3, 3],
            vec![0.2, -0.4, 1.1, 0.8, 0.3, -0.9, 0.6, 0.05, -0.7],
            "",
        ),
        (
            "x: tensor[2, 3, f64]",
            "sum(sum(mul(sin(einsum(\"ij,kj->ik\", x, b)), to_tensor([[0.9f64, -1.1f64], [0.35f64, 2.6f64]])), 1i32), 0i32)",
            vec![2, 3],
            vec![1.5, -2.0, 0.25, 3.0, 0.5, -1.0],
            "b = to_tensor([[0.1f64, 2.0f64, -0.7f64], [1.3f64, 2.2f64, -0.4f64]])\n",
        ),
    ] {
        let literal = |values: &[f64]| -> String {
            let cell = |v: &f64| format!("{v:?}f64");
            if shape.len() == 1 {
                format!(
                    "[{}]",
                    values.iter().map(cell).collect::<Vec<_>>().join(", ")
                )
            } else {
                let rows: Vec<String> = values
                    .chunks(shape[1])
                    .map(|row| format!("[{}]", row.iter().map(cell).collect::<Vec<_>>().join(", ")))
                    .collect();
                format!("[{}]", rows.join(", "))
            }
        };
        let signature = if extra.is_empty() {
            signature.to_string()
        } else {
            format!("{signature}, b: tensor[2, 3, f64]")
        };
        let call_tail = if extra.is_empty() { "" } else { ", b" };
        let wrt = if extra.is_empty() { "" } else { ", wrt=x" };
        let mut program =
            format!("module Demo.Main\ndef loss({signature}) -> tensor[f64] = {body}\n{extra}");
        program.push_str(&format!(
            "g = grad(loss{wrt})(to_tensor({}){call_tail})\n",
            literal(&point)
        ));
        for i in 0..point.len() {
            for (tag, sign) in [("p", 1.0), ("m", -1.0)] {
                let mut moved = point.clone();
                moved[i] += sign * h;
                program.push_str(&format!(
                    "l{tag}{i} = loss(to_tensor({}){call_tail})\n",
                    literal(&moved)
                ));
            }
        }
        let (_dir, reef, app) = common::make_app("issue-3362-fd");
        common::write_file(&app.join("src/main.ch"), &program);
        let output = eval_text(&reef, &app);
        assert!(output.status.success(), "{body}: {output:?}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        let gradient = common::parse_tensor_data(&stdout, "g");
        assert_eq!(gradient.len(), point.len(), "{body}");
        for (i, analytic) in gradient.iter().enumerate() {
            let plus = scalar_root(&stdout, &format!("lp{i}"));
            let minus = scalar_root(&stdout, &format!("lm{i}"));
            let numeric = (plus - minus) / (2.0 * h);
            assert!(
                (analytic - numeric).abs() <= 1e-6 * (1.0 + numeric.abs()),
                "{body}: d/dx{i} analytic {analytic} vs central difference {numeric}"
            );
        }
    }
}

/// Integer forms are forward-only and bool is a type error ([05-OP-33]):
/// differentiating an integer result stays a structural rejection, and a
/// bool `diagonal` selection inside a differentiated body carries no
/// cotangent.
#[test]
fn integer_and_bool_forms_receive_no_adjoint() {
    for (op, call) in [
        ("cumsum", "sum(cumsum(x, 0i32), 0i32)"),
        ("diagonal", "sum(diagonal(x, 0i32, 1i32), 0i32)"),
        ("trace", "trace(x, 0i32, 1i32)"),
        ("einsum", "einsum(\"ij,ij->\", x, x)"),
    ] {
        let shape = if op == "cumsum" { "4" } else { "2, 2" };
        let value = if op == "cumsum" {
            "[1i32, 2i32, 3i32, 4i32]"
        } else {
            "[[1i32, 2i32], [3i32, 4i32]]"
        };
        let (_dir, reef, app) = common::make_app("issue-3362-int");
        common::write_file(
            &app.join("src/main.ch"),
            &format!(
                "module Demo.Main\ndef f(x: tensor[{shape}, i32]) -> tensor[i32] = {call}\ng = grad(f)(to_tensor({value}))\n"
            ),
        );
        let output = eval_text(&reef, &app);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success(),
            "{op}: integer grad must be rejected"
        );
        assert!(
            stderr.contains("grad requires a scalar floating output"),
            "{op}: {stderr}"
        );
        assert!(
            !stderr.contains("has no numeric IR lowering"),
            "{op}: {stderr}"
        );
    }
    let (_dir, reef, app) = common::make_app("issue-3362-bool");
    common::write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\ndef f(x: tensor[2, 2, f32]) -> tensor[f32] = sum(where(diagonal(gt(x, to_tensor([[0.0f32, 0.0f32], [0.0f32, 0.0f32]])), 0i32, 1i32), to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])), 0i32)\ng = grad(f)(to_tensor([[1.0f32, -2.0f32], [3.0f32, -4.0f32]]))\n",
    );
    let output = eval_text(&reef, &app);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        common::parse_tensor_data(&String::from_utf8(output.stdout).unwrap(), "g"),
        vec![0.0; 4]
    );
}

/// An integer form inside a differentiated body keeps its checked host
/// kernel, which the transform cannot consume; the rejection names the
/// issue that owns the missing integer graph rather than the generic
/// fallback.
#[test]
fn integer_forms_inside_a_differentiated_body_cite_their_issue() {
    let (_dir, reef, app) = common::make_app("issue-3362-int-body");
    common::write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\ndef f(x: tensor[3, f32], k: tensor[3, i32]) -> tensor[f32] = sum(mul(x, cast(cumsum(k, 0i32), f32)), 0i32)\ng = grad(f, wrt=x)(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([1i32, 2i32, 3i32]))\n",
    );
    let output = eval_text(&reef, &app);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stderr}");
    assert!(stderr.contains("chelis#3377"), "{stderr}");
    assert!(!stderr.contains("has no numeric IR lowering"), "{stderr}");
}

/// CPU seconds consumed by this process's waited-for children.
fn children_cpu_seconds() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` fills the struct it is handed and reports failure
    // through its return value, which is checked before the read.
    let usage = unsafe {
        assert_eq!(
            libc::getrusage(libc::RUSAGE_CHILDREN, usage.as_mut_ptr()),
            0
        );
        usage.assume_init()
    };
    let seconds = |time: libc::timeval| time.tv_sec as f64 + time.tv_usec as f64 / 1e6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

/// The cumsum graph is O(n) nodes, so lowering its adjoint must stay
/// near-linear in n. Every axis-source query once scanned the whole graph,
/// which made `grad` over a 1024-step cumsum take minutes (CPU grew about
/// 32-fold from n = 256 to n = 1024). Child CPU time, not wall time, so load
/// on the machine does not move the ratio.
#[test]
fn cumsum_adjoint_lowering_scales_near_linearly() {
    let mut cpu = Vec::new();
    for n in [256, 1024] {
        let (_dir, reef, app) = common::make_app("issue-3362-scale");
        common::write_file(
            &app.join("src/main.ch"),
            &format!(
                "module Demo.Main\ndef f(x: tensor[{n}, f32]) -> tensor[f32] = sum(cumsum(x, 0i32), 0i32)\ng = sum(grad(f)(insert(scalar_to_tensor(1.0f32), 0i32, {n}i64)), 0i32)\n"
            ),
        );
        let before = children_cpu_seconds();
        let output = eval_text(&reef, &app);
        cpu.push(children_cpu_seconds() - before);
        assert!(output.status.success(), "n={n}: {output:?}");
        let expected = n * (n + 1) / 2;
        assert_eq!(
            scalar_root(&String::from_utf8(output.stdout).unwrap(), "g"),
            expected as f64,
            "n={n}"
        );
    }
    assert!(
        cpu[1] < 10.0 * cpu[0],
        "4x the cumsum length took {:.2}s of CPU against {:.2}s: superlinear lowering",
        cpu[1],
        cpu[0]
    );
}
