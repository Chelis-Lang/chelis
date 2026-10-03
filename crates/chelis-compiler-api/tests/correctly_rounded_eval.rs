//! chelis#2957 K2: eval computes `tanh` as the correctly rounded [05-OP-46]
//! primitive and every section 3.3 activation as its pinned graph of
//! correctly rounded primitives, bit for bit against an independent reference
//! built from `chelis-crmath` (spec/design/correctly_rounded_math.md section 8,
//! tests 6-7). Also the `sin` adjoint through `cos` (chelis#2989), `gelu` at the
//! largest finite inputs (chelis#2997), the `tanh` adjoint, and the IEEE default
//! floating-point environment around evaluation (section 6).
use chelis_compiler_api::compiler::{compile, eval_selected, prepare_eval_in_context};
use chelis_compiler_api::schema::{
    CompileRequest, CompileResult, CompileTarget, EvalRequest, EvalResult, SourceKind, TensorValue,
};
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
use half::{bf16, f16};
use serde_json::json;
use std::collections::BTreeMap;

fn input_tensor(dtype: &str, input: &[u64]) -> TensorValue {
    let width = match dtype {
        "f64" => 16,
        "f32" => 8,
        _ => 4,
    };
    let bits: Vec<String> = input
        .iter()
        .map(|bits| format!("{bits:0width$x}"))
        .collect();
    TensorValue {
        shape: vec![input.len() as i64],
        data: serde_json::from_value(json!({"dtype": dtype, "bits": bits})).unwrap(),
    }
}

/// Evaluate `main(x)` over one rank-1 input given as storage bits, and return
/// the result's storage bits.
fn eval_main(source: &str, dtype: &str, input: &[u64]) -> Vec<u64> {
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::from([("x".to_string(), input_tensor(dtype, input))]),
        },
        &["main".to_string()],
    )
    .unwrap_or_else(|error| panic!("evaluate {source}: {error:?}"));
    main_bits(&result, dtype)
}

/// `main(x)` in a client of a package whose library module imports chelis-std.
fn eval_main_in_package(library: &str, client: &str, dtype: &str, input: &[u64]) -> Vec<u64> {
    // The package compile needs more stack than a test thread's default, as
    // the CLI's worker thread provides.
    let (library, client, dtype, input) = (
        library.to_string(),
        client.to_string(),
        dtype.to_string(),
        input.to_vec(),
    );
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || eval_package_on_this_thread(&library, &client, &dtype, &input))
        .unwrap()
        .join()
        .unwrap()
}

fn eval_package_on_this_thread(
    library: &str,
    client: &str,
    dtype: &str,
    input: &[u64],
) -> Vec<u64> {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("reef.toml"),
        format!(
            "[package]\nname = \"correct-rounding\"\nversion = \"0.1.0\"\n\
             compiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
        ),
    )
    .unwrap();
    std::fs::write(directory.path().join("src/values.ch"), library).unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let result = prepare_eval_in_context(&context, client)
        .unwrap_or_else(|error| panic!("compile {client}: {error:?}"))
        .eval_root(
            if input.is_empty() {
                BTreeMap::new()
            } else {
                BTreeMap::from([("x".to_string(), input_tensor(dtype, input))])
            },
            "main",
        )
        .unwrap_or_else(|error| panic!("evaluate {client}: {error:?}"));
    main_bits(&result, dtype)
}

fn main_bits(result: &EvalResult, dtype: &str) -> Vec<u64> {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("main"))
        .expect("main root");
    let value = serde_json::to_value(&root.value).unwrap();
    let data = &value["value"]["data"];
    assert_eq!(data["dtype"], dtype, "{value}");
    data["bits"]
        .as_array()
        .expect("tensor bits")
        .iter()
        .map(|bits| u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap())
        .collect()
}

fn unary_source(dtype: &str, n: usize, body: &str) -> String {
    format!("def main(x: tensor[{n}, {dtype}]) -> tensor[{n}, {dtype}] = {body}\n")
}

/// The #2952, #2959 and #2971 witnesses, the tanh cancellation points, the
/// section 8 test 7 gelu points, signed zeros, infinities, the largest
/// finite values, and a deterministic sample of [-20, 20].
fn inputs_f64(max: f64) -> Vec<f64> {
    let mut values = vec![
        0.0,
        -0.0,
        1e-8,
        -1e-8,
        1e-5,
        1e-3,
        -0.05580474063754082,
        -1.3359944820404053,
        -0.46123430132865906,
        -1.72889078,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -9.336,
        0.1,
        1.7,
        -2.3,
        3.9,
        44.0,
        -90.0,
        max,
        -max,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut state = 0x2957_u64;
    for _ in 0..512 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        values.push(unit * 40.0 - 20.0);
    }
    values
}

fn inputs_f32() -> Vec<f32> {
    inputs_f64(f64::from(f32::MAX))
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

// The section 3.3 graphs, written here independently of the compiler's one
// definition, each primitive rounded once at the operand dtype. Host
// arithmetic leaves an ISA-chosen NaN (x86's default NaN is negative), so a
// NaN result is finalized to [04-NUM-2]'s canonical quiet NaN as the spec
// requires; a canonical NaN in stays NaN through the later primitives, so
// finalizing each graph's result equals finalizing every primitive.
macro_rules! reference_graphs {
    ($module:ident, $t:ty, $canonical_nan:expr, $exp:path, $tanh:path) => {
        mod $module {
            fn canonical(x: $t) -> $t {
                if x.is_nan() {
                    <$t>::from_bits($canonical_nan)
                } else {
                    x
                }
            }

            pub fn sigmoid(x: $t) -> $t {
                canonical(1.0 / (1.0 + $exp(-x)))
            }

            pub fn silu(x: $t) -> $t {
                canonical(x * sigmoid(x))
            }

            pub fn gelu(x: $t) -> $t {
                let cubic = 0.044715_f64 as $t;
                let c = 0.797_884_560_802_865_4_f64 as $t;
                let u = c * (x + cubic * ((x * x) * x));
                canonical(x * sigmoid(2.0 * u))
            }

            pub fn tanh(x: $t) -> $t {
                $tanh(x)
            }

            pub fn tanh_adjoint(x: $t) -> $t {
                let y = $tanh(x);
                canonical(1.0 * (1.0 - y * y))
            }

            pub fn softmax(x: &[$t]) -> Vec<$t> {
                let max = x.iter().copied().fold(<$t>::NEG_INFINITY, <$t>::max);
                let exps: Vec<$t> = x.iter().map(|value| $exp(value - max)).collect();
                // The sum is section 2.3's adjacent-pair balanced tree.
                let mut level = exps.clone();
                while level.len() > 1 {
                    level = level
                        .chunks(2)
                        .map(|pair| {
                            if pair.len() == 2 {
                                pair[0] + pair[1]
                            } else {
                                pair[0]
                            }
                        })
                        .collect();
                }
                let sum = level[0];
                exps.iter().map(|value| canonical(value / sum)).collect()
            }

            // Std.Contracts.normal_cdf and its erf_approx, primitive by primitive.
            pub fn normal_cdf(x: $t) -> $t {
                let k = |value: f64| value as $t;
                let erf = |x: $t| -> $t {
                    let ax = if x < 0.0 { -x } else { x };
                    if ax < k(0.00001) {
                        // The library spells 2/sqrt(pi) as 1.1283791670955126.
                        return x * k(std::f64::consts::FRAC_2_SQRT_PI);
                    }
                    let t = 1.0 / (1.0 + k(0.3275911) * ax);
                    let poly = t
                        * (k(0.254829592)
                            + t * (k(-0.284496736)
                                + t * (k(1.421413741)
                                    + t * (k(-1.453152027) + t * k(1.061405429)))));
                    let y = 1.0 - poly * $exp(-(ax * ax));
                    if x < 0.0 { -y } else { y }
                };
                canonical(k(0.5) * (1.0 - erf(-(x * k(0.7071067811865475)))))
            }
        }
    };
}

reference_graphs!(ref32, f32, 0x7fc0_0000, chelis_crmath::exp_f32, chelis_crmath::tanh_f32);
reference_graphs!(
    ref64,
    f64,
    0x7ff8_0000_0000_0000,
    chelis_crmath::exp_f64,
    chelis_crmath::tanh_f64
);

/// An activation's surface name and its reference.
type Case<T> = (&'static str, fn(T) -> T);

fn assert_bits(op: &str, dtype: &str, inputs: &[u64], got: &[u64], want: &[u64]) {
    assert_eq!(got.len(), want.len());
    let mismatches: Vec<String> = inputs
        .iter()
        .zip(got.iter().zip(want))
        .filter(|(_, (got, want))| got != want)
        .take(8)
        .map(|(input, (got, want))| format!("x={input:#x}: eval {got:#x}, reference {want:#x}"))
        .collect();
    assert!(
        mismatches.is_empty(),
        "{dtype} {op} differs from the correctly rounded section 3.3 reference: {mismatches:?}"
    );
}

#[test]
fn f32_activations_match_the_correctly_rounded_reference_bit_for_bit() {
    let inputs = inputs_f32();
    let bits: Vec<u64> = inputs.iter().map(|x| u64::from(x.to_bits())).collect();
    let cases: [Case<f32>; 4] = [
        ("tanh", ref32::tanh),
        ("sigmoid", ref32::sigmoid),
        ("silu", ref32::silu),
        ("gelu", ref32::gelu),
    ];
    for (op, reference) in cases {
        let source = unary_source("f32", inputs.len(), &format!("{op}(x)"));
        let got = eval_main(&source, "f32", &bits);
        let want: Vec<u64> = inputs
            .iter()
            .map(|x| u64::from(reference(*x).to_bits()))
            .collect();
        assert_bits(op, "f32", &bits, &got, &want);
    }
}

#[test]
fn f64_activations_match_the_correctly_rounded_reference_bit_for_bit() {
    let inputs = inputs_f64(f64::MAX);
    let bits: Vec<u64> = inputs.iter().map(|x| x.to_bits()).collect();
    let cases: [Case<f64>; 4] = [
        ("tanh", ref64::tanh),
        ("sigmoid", ref64::sigmoid),
        ("silu", ref64::silu),
        ("gelu", ref64::gelu),
    ];
    for (op, reference) in cases {
        let source = unary_source("f64", inputs.len(), &format!("{op}(x)"));
        let got = eval_main(&source, "f64", &bits);
        let want: Vec<u64> = inputs.iter().map(|x| reference(*x).to_bits()).collect();
        assert_bits(op, "f64", &bits, &got, &want);
    }
}

#[test]
fn tanh_near_zero_is_the_hyperbolic_tangent_not_a_cancelled_sigmoid() {
    for x in [1e-8_f32, -1e-8, 1e-5, 1e-3] {
        let got = eval_main(
            &unary_source("f32", 1, "tanh(x)"),
            "f32",
            &[u64::from(x.to_bits())],
        );
        let got = f32::from_bits(got[0] as u32);
        assert_ne!(got, 0.0, "tanh({x:e}) cancelled to zero");
        assert_eq!(
            got.to_bits(),
            chelis_crmath::tanh_f32(x).to_bits(),
            "tanh({x:e})"
        );
    }
}

#[test]
fn softmax_matches_the_correctly_rounded_reference_bit_for_bit() {
    // The #2971 probe vector, then the #2952 witnesses.
    let probe: Vec<f32> = vec![
        0.1,
        1.7,
        -2.3,
        0.33,
        3.9,
        -0.77,
        2.2,
        1.05,
        -5.5,
        0.6,
        4.1,
        -1.9,
        2.75,
        0.01,
        -0.4,
        1.3,
        -1.335_994_5,
        -0.461_234_3,
        -1.728_890_8,
    ];
    let bits: Vec<u64> = probe.iter().map(|x| u64::from(x.to_bits())).collect();
    let got = eval_main(
        &unary_source("f32", probe.len(), "softmax(x, 0)"),
        "f32",
        &bits,
    );
    let want: Vec<u64> = ref32::softmax(&probe)
        .iter()
        .map(|x| u64::from(x.to_bits()))
        .collect();
    assert_bits("softmax", "f32", &bits, &got, &want);

    let wide: Vec<f64> = probe.iter().map(|x| f64::from(*x)).collect();
    let bits: Vec<u64> = wide.iter().map(|x| x.to_bits()).collect();
    let got = eval_main(
        &unary_source("f64", wide.len(), "softmax(x, 0)"),
        "f64",
        &bits,
    );
    let want: Vec<u64> = ref64::softmax(&wide).iter().map(|x| x.to_bits()).collect();
    assert_bits("softmax", "f64", &bits, &got, &want);
}

#[test]
fn normal_cdf_matches_the_correctly_rounded_reference_bit_for_bit() {
    for (dtype, inputs) in [
        (
            "f32",
            inputs_f32()
                .iter()
                .map(|x| f64::from(*x))
                .collect::<Vec<_>>(),
        ),
        ("f64", inputs_f64(f64::MAX)),
    ] {
        let finite: Vec<f64> = inputs
            .into_iter()
            .filter(|x| x.is_finite() && x.abs() < 40.0)
            // The witnesses lead the list, then a bounded prefix of the sample.
            .take(64)
            .collect();
        // The library function is scalar-generic, so each input is a literal
        // spelled at its shortest round-trip decimal.
        let literal = |x: f64| {
            let magnitude = if dtype == "f32" {
                format!("{:?}f32", (x as f32).abs())
            } else {
                format!("{:?}f64", x.abs())
            };
            if x.is_sign_negative() {
                format!("neg({magnitude})")
            } else {
                magnitude
            }
        };
        let calls: Vec<String> = finite
            .iter()
            .map(|x| format!("cdf({})", literal(*x)))
            .collect();
        let library = "module Probe.Values\nexport (cdf)\nimport Std.Contracts (normal_cdf)\n\
                       def cdf[p: Float](x: p) -> p = normal_cdf(x)\n";
        let client = format!(
            "module Probe.Client\nimport Probe.Values (cdf)\nmain = to_tensor([{}])\n",
            calls.join(", ")
        );
        let (bits, want): (Vec<u64>, Vec<u64>) = if dtype == "f32" {
            finite
                .iter()
                .map(|x| {
                    let x = *x as f32;
                    (
                        u64::from(x.to_bits()),
                        u64::from(ref32::normal_cdf(x).to_bits()),
                    )
                })
                .unzip()
        } else {
            finite
                .iter()
                .map(|x| (x.to_bits(), ref64::normal_cdf(*x).to_bits()))
                .unzip()
        };
        let got = eval_main_in_package(library, &client, dtype, &[]);
        assert_bits("normal_cdf", dtype, &bits, &got, &want);
    }
}

/// chelis#2997: every finite f16 input, against the pinned graph evaluated at
/// f16 (each primitive correctly rounded at f32, then finalized once).
#[test]
fn gelu_is_the_pinned_graph_on_every_finite_f16_input() {
    let r = |x: f32| f16::from_f32(x).to_f32();
    let k = |x: f64| f16::from_f64(x).to_f32();
    let sigmoid = |x: f32| r(1.0 / r(1.0 + chelis_crmath::exp_f16(f16::from_f32(-x)).to_f32()));
    let gelu = |x: f32| {
        let u = r(k(0.797_884_560_802_865_4) * r(x + r(k(0.044715) * r(r(x * x) * x))));
        r(x * sigmoid(r(2.0 * u)))
    };
    let inputs: Vec<u64> = (0..=u16::MAX)
        .filter(|bits| f16::from_bits(*bits).is_finite())
        .map(u64::from)
        .collect();
    let got = eval_main(
        &unary_source("f16", inputs.len(), "gelu(x)"),
        "f16",
        &inputs,
    );
    let want: Vec<u64> = inputs
        .iter()
        .map(|bits| u64::from(f16::from_f32(gelu(f16::from_bits(*bits as u16).to_f32())).to_bits()))
        .collect();
    assert_bits("gelu", "f16", &inputs, &got, &want);
    let max = inputs
        .iter()
        .position(|bits| *bits == u64::from(f16::MAX.to_bits()))
        .unwrap();
    assert_eq!(got[max], u64::from(f16::MAX.to_bits()), "gelu(f16::MAX)");
}

#[test]
fn gelu_at_the_largest_finite_input_is_the_input() {
    for (dtype, max) in [
        ("bf16", u64::from(bf16::MAX.to_bits())),
        ("f32", u64::from(f32::MAX.to_bits())),
        ("f64", f64::MAX.to_bits()),
    ] {
        let got = eval_main(&unary_source(dtype, 1, "gelu(x)"), dtype, &[max]);
        assert_eq!(got, vec![max], "{dtype} gelu(max finite) must be the input");
    }
}

fn grad_source(dtype: &str, n: usize, forward: &str) -> String {
    format!(
        "def f(x: tensor[{n}, {dtype}]) -> tensor[{dtype}] = sum({forward}(x), 0i32)\n\
         def main(x: tensor[{n}, {dtype}]) -> tensor[{n}, {dtype}] = grad(f)(x)\n"
    )
}

/// chelis#2989: the `sin` adjoint is the lane's own `cos`, bit for bit,
/// including large magnitudes where `sin(x + pi/2)` loses the shift.
#[test]
fn sin_adjoint_is_the_cos_primitive_bit_for_bit() {
    let cases: [(&str, Vec<u64>); 4] = [
        (
            "f32",
            [0.7_f32, 1.3, 100.0, 500.0, 1e4, f32::MAX, 0.0, -0.0, -3.5]
                .iter()
                .map(|x| u64::from(x.to_bits()))
                .collect(),
        ),
        (
            "f64",
            [0.7_f64, 1.3, 100.0, 500.0, 1e4, 1e300, 0.0, -0.0]
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        ),
        (
            "f16",
            [0.7_f32, 1.3, 100.0, 500.0, 65504.0, 0.0, -0.0]
                .iter()
                .map(|x| u64::from(f16::from_f32(*x).to_bits()))
                .collect(),
        ),
        (
            "bf16",
            [0.7_f32, 1.3, 100.0, 500.0, bf16::MAX.to_f32(), 0.0, -0.0]
                .iter()
                .map(|x| u64::from(bf16::from_f32(*x).to_bits()))
                .collect(),
        ),
    ];
    for (dtype, bits) in cases {
        let forward = eval_main(&unary_source(dtype, bits.len(), "cos(x)"), dtype, &bits);
        let adjoint = eval_main(&grad_source(dtype, bits.len(), "sin"), dtype, &bits);
        assert_bits("grad(sin) vs cos", dtype, &bits, &adjoint, &forward);
    }
}

/// [05-OP-46]: tanh's adjoint is `g * (1 - y * y)` with `y = tanh(x)`.
#[test]
fn tanh_adjoint_is_one_minus_tanh_squared() {
    let inputs = inputs_f32();
    let bits: Vec<u64> = inputs.iter().map(|x| u64::from(x.to_bits())).collect();
    let got = eval_main(&grad_source("f32", inputs.len(), "tanh"), "f32", &bits);
    let want: Vec<u64> = inputs
        .iter()
        .map(|x| u64::from(ref32::tanh_adjoint(*x).to_bits()))
        .collect();
    assert_bits("grad(tanh)", "f32", &bits, &got, &want);

    let inputs = inputs_f64(f64::MAX);
    let bits: Vec<u64> = inputs.iter().map(|x| x.to_bits()).collect();
    let got = eval_main(&grad_source("f64", inputs.len(), "tanh"), "f64", &bits);
    let want: Vec<u64> = inputs
        .iter()
        .map(|x| ref64::tanh_adjoint(*x).to_bits())
        .collect();
    assert_bits("grad(tanh)", "f64", &bits, &got, &want);
}

#[cfg(target_arch = "aarch64")]
mod control {
    // FPCR: FZ (bit 24) flushes subnormals; RMode (bits 22-23) 0b01 rounds up.
    pub const HOSTILE: u64 = (1 << 24) | (0b01 << 22);

    pub fn read() -> u64 {
        let fpcr: u64;
        // SAFETY: reading FPCR has no side effect.
        unsafe { std::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack)) };
        fpcr
    }

    pub fn write(fpcr: u64) {
        // SAFETY: only floating-point control bits are written.
        unsafe { std::arch::asm!("msr fpcr, {}", in(reg) fpcr, options(nostack)) };
    }
}

#[cfg(target_arch = "x86_64")]
mod control {
    // MXCSR: FTZ (bit 15), DAZ (bit 6), and rounding up (bits 13-14 = 0b10).
    pub const HOSTILE: u64 = 0x1f80 | 0x8000 | 0x0040 | 0x4000;
    /// Bits 6-15: DAZ, the exception masks, rounding control, and FTZ. Bits
    /// 0-5 are sticky status flags, not environment: IEEE status flags are
    /// unobservable ([05-OP-46]), and any float operation the test's own code
    /// runs outside a pinned call may raise one (inexact, typically). FPCR on
    /// aarch64 holds no status, so that comparison stays whole-register.
    const CONTROL_BITS: u32 = 0xffc0;

    /// The control field of MXCSR.
    pub fn read() -> u64 {
        let mut mxcsr: u32 = 0;
        // SAFETY: `stmxcsr` stores four bytes to the given live local.
        unsafe { std::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack)) };
        u64::from(mxcsr & CONTROL_BITS)
    }

    pub fn write(mxcsr: u64) {
        let mxcsr = mxcsr as u32;
        // SAFETY: `ldmxcsr` loads four bytes from the given live local.
        unsafe { std::arch::asm!("ldmxcsr [{}]", in(reg) &mxcsr, options(nostack, readonly)) };
    }
}

/// Section 6: an embedding host's flush-to-zero and rounding mode do not
/// reach eval's results, and the host's state is restored afterwards.
#[test]
fn eval_pins_the_ieee_default_and_restores_the_host_environment() {
    // A subnormal product, an inexact quotient, and a transcendental.
    let inputs: Vec<u64> = [1e-38_f32, 3.0, 0.5]
        .iter()
        .map(|x| u64::from(x.to_bits()))
        .collect();
    let source = unary_source(
        "f32",
        3,
        "add(mul(x, to_tensor([0.01f32, 0.33f32, 1.0f32])), tanh(x))",
    );
    let default = eval_main(&source, "f32", &inputs);
    let saved = control::read();
    control::write(control::HOSTILE);
    let hostile = eval_main(&source, "f32", &inputs);
    let after = control::read();
    control::write(saved);
    assert_eq!(
        after,
        control::HOSTILE,
        "eval must restore the host's control state"
    );
    assert_eq!(
        hostile, default,
        "the host's FP environment changed eval's results"
    );
}

/// Section 6 at compile time: literal finalization and constant folding run
/// under the IEEE default too, so a host's flush-to-zero and rounding mode do
/// not change the generated C (a rounded-up literal, a flushed folded `sub`),
/// and the host's state is restored afterwards.
#[test]
fn compile_pins_the_ieee_default_and_restores_the_host_environment() {
    let request = || CompileRequest {
        source_kind: SourceKind::Surf,
        source: "y = exp(-100.0f32)\nz = sub(1.1754944e-38f32, 1.0e-38f32)\n".to_string(),
        target: CompileTarget::C,
        entry_name: None,
    };
    let emitted = |result: CompileResult| {
        result
            .files
            .into_iter()
            .map(|file| (file.path, file.contents))
            .collect::<Vec<_>>()
    };
    let default = emitted(compile(request()).expect("compile"));
    let saved = control::read();
    control::write(control::HOSTILE);
    let hostile = compile(request());
    let after = control::read();
    control::write(saved);
    assert_eq!(
        after,
        control::HOSTILE,
        "compile must restore the host's control state"
    );
    assert_eq!(
        emitted(hostile.expect("compile")),
        default,
        "the host's FP environment changed the generated C"
    );
}

/// Sanity for the reference itself: the sample covers inputs where each graph
/// is not trivially saturated.
#[test]
fn reference_inputs_cover_both_saturated_and_unsaturated_regions() {
    let inputs = inputs_f32();
    assert!(
        inputs
            .iter()
            .any(|x| ref32::sigmoid(*x) > 0.1 && ref32::sigmoid(*x) < 0.9)
    );
    assert!(
        inputs
            .iter()
            .any(|x| ref32::gelu(*x) == *x && x.is_finite() && *x > 1e30)
    );
}
