//! Tests for the tiered validated binder generator (RFC D-STARVE).

use super::*;

fn deep_of(surf: &str) -> Vec<Expr> {
    let decls = chelis_surf::parser::parse_str(surf).expect("parse surf");
    chelis_surf::desugar::desugar_program(&decls)
}

fn source_of(surf: &str) -> String {
    chelis_deep::printer::print_canonical(&deep_of(surf))
}

const PROB: &str = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
";

#[test]
fn rejection_sampling_produces_a_valid_probability() {
    let exprs = deep_of(PROB);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let mut rng = GenRng::new(0);
    let got = generate_binder(
        inv,
        &ConstEnv::new(),
        &source_of(PROB),
        &[],
        &mut rng,
        0.01,
        1000,
    )
    .expect("a [0,1] band has ~10% acceptance, never starves");
    assert_eq!(got.method, GenMethod::Rejection);
    let v = got.env.get("p.value").copied().unwrap();
    assert!((0.0..=1.0).contains(&v), "validated in [0,1], got {v}");
}

#[test]
fn rejection_sampling_is_deterministic_under_fixed_seed() {
    let exprs = deep_of(PROB);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let src = source_of(PROB);
    let mut r1 = GenRng::new(7);
    let mut r2 = GenRng::new(7);
    let a = generate_binder(inv, &ConstEnv::new(), &src, &[], &mut r1, 0.01, 1000).unwrap();
    let b = generate_binder(inv, &ConstEnv::new(), &src, &[], &mut r2, 0.01, 1000).unwrap();
    assert_eq!(a.env, b.env, "same seed => same sample");
}

const EQ_PROB: &str = "module Stats.Eq
@opaque
@invariant(p) p.value == 0.5
type Exact =
  | Exact { value: f32 }
def make(x: f32) -> Exact = Exact { value: x }
";

#[test]
fn exact_equality_invariant_starves_with_equality_atoms_shape() {
    let exprs = deep_of(EQ_PROB);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let mut rng = GenRng::new(0);
    // No producers wired here; both tiers starve.
    let diag = generate_binder(
        inv,
        &ConstEnv::new(),
        &source_of(EQ_PROB),
        &[],
        &mut rng,
        0.01,
        500,
    )
    .expect_err("exact == over a float field starves by design");
    assert_eq!(diag.type_name, "Exact");
    assert_eq!(diag.shape, PredShape::EqualityAtoms);
    assert!(diag.recommended_route.contains("Tier B"));
    assert!(diag.message().contains("equality-atoms"));
}

#[test]
fn band_shape_is_classified() {
    let surf = "module Stats.Band
@opaque
@invariant(p) p.value >= 0.4 && p.value <= 0.6
type Band =
  | Band { value: f32 }
def make(x: f32) -> Band = Band { value: x }
";
    let exprs = deep_of(surf);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let shape = classify_pred_shape(inv, &ConstEnv::new());
    match shape {
        PredShape::BandWidth(Some(w)) => assert!((w - 0.2).abs() < 1e-6, "band width 0.2, got {w}"),
        other => panic!("expected band-width, got {other:?}"),
    }
}

#[test]
fn min_rate_zero_disables_floor_short_circuit() {
    // With floor 0.0 the rejection tier uses the full budget (no early
    // break); a satisfiable band still succeeds.
    let exprs = deep_of(PROB);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let mut rng = GenRng::new(1);
    let got = generate_binder(
        inv,
        &ConstEnv::new(),
        &source_of(PROB),
        &[],
        &mut rng,
        0.0,
        1000,
    )
    .expect("floor 0.0 still generates for a satisfiable band");
    assert_eq!(got.method, GenMethod::Rejection);
}

const SIMPLEX: &str = "module Stats.Simplex
export (make_simplex)
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.01
def make_simplex(a: f32, b: f32, c: f32) -> Simplex =
  { s = abs(a) + abs(b) + abs(c) + 0.001;
    Simplex { weights: to_tensor([abs(a) / s, abs(b) / s, (abs(c) + 0.001) / s]) } }
";

#[test]
fn simplex_band_rejection_starves_then_constructor_serves() {
    // sum(weights)==1 band is measure-near-zero under independent
    // component sampling; rejection starves. A `normalize` producer that
    // divides by the sum lands in the band, so constructor-based
    // generation serves it WITHOUT starving (the D-STARVE acceptance
    // probe).
    let exprs = deep_of(SIMPLEX);
    let inv = &collect_opaque_invariants(&exprs)[0];
    let mut consts = ConstEnv::new();
    consts.insert("eps".to_string(), 0.01);
    let producers = vec![GenProducer {
        name: "make_simplex".to_string(),
        param_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        param_kinds: vec![
            GenParamKind::Scalar("f32".to_string()),
            GenParamKind::Scalar("f32".to_string()),
            GenParamKind::Scalar("f32".to_string()),
        ],
        option_wrapped: false,
    }];
    let mut rng = GenRng::new(0);
    let got = generate_binder(
        inv,
        &consts,
        &source_of(SIMPLEX),
        &producers,
        &mut rng,
        0.01,
        400,
    )
    .expect("constructor-based generation serves the simplex band without starving");
    assert_eq!(
        got.method,
        GenMethod::Constructor,
        "the equality-band binder is served by the producer, not rejection"
    );
    let sum: f64 = (0..3)
        .map(|i| got.env.get(&format!("p.weights.{i}")).copied().unwrap())
        .sum();
    assert!(
        (sum - 1.0).abs() <= 0.01 + 1e-6,
        "validated in band, sum={sum}"
    );
}
