#!/usr/bin/env python3
"""Check target arithmetic and native host admission as separate obligations."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile

import no_std_core_support as support
from no_std_core_support import ROOT, OracleFailure, compile_rejected, consumer, portable, run

CRATE = "chelis-certificate-core"
BASELINE = "724e30db000c106633031c9d5b02cf3cbd39337d"
BASELINE_REPOSITORY = "https://github.com/Chelis-Lang/chelis.git"
BASELINE_FILES = (
    "Cargo.lock", "crates/chelis-prove/Cargo.toml",
    "crates/chelis-prove/src/clarabel_sos/poly.rs",
    "crates/chelis-prove/src/clarabel_sos/exact.rs",
    "crates/chelis-prove/src/clarabel_sos/linalg.rs",
    "crates/chelis-prove/src/clarabel_sos/engine.rs",
    "crates/chelis-prove/src/clarabel_sos/encode.rs",
)
FORBIDDEN = (
    "chelis-prove", "chelis-types", "chelis-surf", "chelis-deep",
    "chelis-compiler-api", "chelis-pipeline-core", "clarabel", "openblas-src",
    "blas", "lapack", "blas-sys", "lapack-sys", "accelerate-src",
    "cvc5-rs", "z3", "z3-sys", "carcara", "rug", "gmp-mpfr-sys",
)
PORTABLE_PACKAGES = {
    "chelis-no-std-consumer", CRATE, "num-bigint", "num-rational",
    "num-integer", "num-traits",
}
CORE_TESTS = (
    "identities_and_normalization_are_exact",
    "exact_psd_includes_singular_and_excludes_tiny_negative_pivots",
    "errors_preserve_block_order_and_symmetry_precedence",
    "arithmetic_success_does_not_validate_multiplier_or_domain",
    "repair_preserves_free_coordinates_and_pivot_order",
    "constructors_and_repair_preserve_shape_assertions",
)
HOST_TESTS = (
    "canonical_production_layouts_preserve_exact_admission",
    "evidence_preserves_exact_values_and_block_order",
    "goal_domain_and_strictness_cannot_gain_authority",
    "negative_altered_and_swapped_multipliers_cannot_gain_authority",
    "missing_extra_and_wrong_size_blocks_cannot_gain_authority",
    "public_errors_preserve_host_labels",
    "native_clarabel_production_parity",
)
RATIONAL_DEPENDENCIES = '''
num-bigint = { version = "0.4", default-features = false }
num-rational = { version = "0.4", default-features = false, features = ["num-bigint"] }
'''

PORTABLE_SOURCE = r'''
#![no_std]
extern crate alloc;
use alloc::vec;
use chelis_certificate_core::exact::{RatSymMatrix, SosBlock, SosCertificate};
use chelis_certificate_core::linalg::{RatMatrix, project_onto_affine};
use chelis_certificate_core::poly::RationalPoly;
pub fn exact_check() -> bool {
    let one = RationalPoly::from_int_coeffs(&[1]).coeff(0);
    let repaired = project_onto_affine(
        &RatMatrix::from_data(1, 2, vec![one.clone(), one.clone()]),
        1, &[one.clone()],
    ).unwrap();
    SosCertificate {
        poly: RationalPoly::constant(one.clone()), lo: one.clone(), hi: one,
        blocks: vec![SosBlock {
            gram: RatSymMatrix::from_rows(1, repaired),
            multiplier: RationalPoly::from_int_coeffs(&[1]),
        }],
    }.verify().is_ok()
}
'''

ARITHMETIC_SOURCE = r'''
use __ARITHMETIC__::exact::{CertError, RatSymMatrix, SosBlock, SosCertificate};
use __ARITHMETIC__::linalg::{ProjectError, RatMatrix, project_onto_affine, rref};
use __ARITHMETIC__::poly::RationalPoly;
use num_bigint::BigInt;
use num_rational::BigRational;
use serde_json::{json, Value};

#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn q(n: i64) -> BigRational { BigRational::from_integer(BigInt::from(n)) }
fn rat(value: &BigRational) -> String { format!("{}/{}", value.numer(), value.denom()) }
fn rationals(values: &[BigRational]) -> Vec<String> { values.iter().map(rat).collect() }
fn matrix(n: usize, entries: &[i64]) -> RatSymMatrix {
    RatSymMatrix::from_rows(n, entries.iter().map(|&n| q(n)).collect())
}
fn cert(gram: RatSymMatrix) -> SosCertificate {
    SosCertificate {
        poly: gram.quadratic_form_poly(), lo: q(-1), hi: q(1),
        blocks: vec![SosBlock { gram, multiplier: RationalPoly::from_int_coeffs(&[1]) }],
    }
}
fn identities() -> Value {
    let mut candidate = cert(matrix(2, &[1, 0, 0, 1]));
    assert_eq!(candidate.poly, RationalPoly::from_int_coeffs(&[1, 0, 1, 0]));
    let valid = candidate.verify();
    assert_eq!(valid, Ok(()));
    let normalized = rationals(candidate.poly.coeffs());
    candidate.poly = RationalPoly::from_int_coeffs(&[1, 1, 1]);
    let mismatch = candidate.verify();
    assert_eq!(mismatch, Err(CertError::PolynomialMismatch));
    let huge = BigInt::from(1u8) << 256usize;
    let a = BigRational::new(&huge + 1, &huge + 3);
    let b = BigRational::new(&huge + 5, &huge + 7);
    let poly = RationalPoly::from_coeffs(vec![a.clone(), b.clone()]);
    let square = poly.mul(&poly);
    assert_eq!(square.eval(&q(2)), (&a + &b * q(2)).pow(2));
    let distinct = poly != poly.add(&RationalPoly::constant(BigRational::new(BigInt::from(1), huge)));
    assert!(distinct);
    let zero = RationalPoly::from_int_coeffs(&[0, 0]);
    assert_eq!(zero.degree(), None);
    assert!(zero.coeffs().is_empty());
    let mut negative = cert(matrix(1, &[1]));
    negative.poly = RationalPoly::from_int_coeffs(&[-1]);
    negative.blocks[0].multiplier = negative.poly.clone();
    negative.lo = q(2);
    negative.hi = q(-2);
    let arithmetic_only = negative.verify();
    assert_eq!(arithmetic_only, Ok(()));
    json!({
        "normalized": normalized, "valid": format!("{valid:?}"),
        "mismatch": format!("{mismatch:?}"), "huge_coefficients": rationals(poly.coeffs()),
        "square": rationals(square.coeffs()), "evaluation": rat(&square.eval(&q(2))),
        "tiny_delta_is_distinct": distinct, "zero_coefficients": rationals(zero.coeffs()),
        "zero_degree": zero.degree(), "arithmetic_only": format!("{arithmetic_only:?}")
    })
}
fn matrices() -> Value {
    let singular = matrix(2, &[1, 1, 1, 1]);
    let psd = [singular.is_psd(), matrix(2, &[0, 1, 1, 0]).is_psd(), matrix(2, &[1, 2, 2, 1]).is_psd()];
    assert_eq!(psd, [true, false, false]);
    assert_eq!(singular.quadratic_form_poly(), RationalPoly::from_int_coeffs(&[1, 2, 1]));
    let tiny = -BigRational::new(BigInt::from(1), BigInt::from(1u8) << 1024usize);
    let tiny_psd = RatSymMatrix::from_rows(1, vec![tiny.clone()]).is_psd();
    assert!(!tiny_psd);
    let mut malformed = cert(matrix(2, &[-1, 1, 0, -1]));
    malformed.blocks.push(SosBlock {
        gram: matrix(1, &[-1]), multiplier: RationalPoly::from_int_coeffs(&[1]),
    });
    let asymmetric = malformed.verify();
    assert_eq!(asymmetric, Err(CertError::NotSymmetric { block: 0 }));
    malformed.blocks[0].gram = matrix(1, &[1]);
    let later_negative = malformed.verify();
    assert_eq!(later_negative, Err(CertError::NotPsd { block: 1 }));
    json!({
        "psd": psd, "singular_form": rationals(singular.quadratic_form_poly().coeffs()),
        "tiny_pivot": rat(&tiny), "tiny_psd": tiny_psd,
        "errors": [format!("{asymmetric:?}"), format!("{later_negative:?}")]
    })
}
fn repairs() -> Value {
    let aug = RatMatrix::from_data(2, 4, [0, 1, 1, 3, 1, 1, 0, 4].map(q).to_vec());
    let reduced = rref(&aug, 3);
    assert_eq!(reduced.pivots, vec![(0, 0), (1, 1)]);
    assert_eq!(reduced.free_columns(), vec![2]);
    let repaired = project_onto_affine(&aug, 3, &[q(9), q(9), q(2)]).unwrap();
    assert_eq!(repaired, vec![q(3), q(1), q(2)]);
    let rows: Vec<Vec<String>> = (0..reduced.reduced.rows()).map(|i|
        (0..reduced.reduced.cols()).map(|j| rat(reduced.reduced.at(i, j))).collect()
    ).collect();
    let inconsistent = RatMatrix::from_data(2, 2, [1, 1, 1, 2].map(q).to_vec());
    let error = project_onto_affine(&inconsistent, 1, &[q(0)]);
    assert_eq!(error, Err(ProjectError::Inconsistent));
    let negative = RatMatrix::from_data(1, 2, [1, -1].map(q).to_vec());
    let negative_repair = project_onto_affine(&negative, 1, &[q(4)]).unwrap();
    let verification = cert(RatSymMatrix::from_rows(1, negative_repair.clone())).verify();
    assert_eq!(verification, Err(CertError::NotPsd { block: 0 }));
    let aug = RatMatrix::zeros(1, 2);
    let assertions = [
        std::panic::catch_unwind(|| RatSymMatrix::from_rows(2, vec![q(1)])).is_err(),
        std::panic::catch_unwind(|| RatMatrix::from_data(2, 2, vec![q(1)])).is_err(),
        std::panic::catch_unwind(|| rref(&aug, 2)).is_err(),
        std::panic::catch_unwind(|| project_onto_affine(&aug, 1, &[])).is_err(),
    ];
    assert_eq!(assertions, [true; 4]);
    json!({
        "pivots": reduced.pivots, "free": reduced.free_columns(), "rref": rows,
        "solution": rationals(&repaired), "inconsistent": format!("{error:?}"),
        "negative_solution": rationals(&negative_repair),
        "negative_verification": format!("{verification:?}"), "shape_assertions": assertions
    })
}
fn main() {
    println!("CERTIFICATE_ARITHMETIC {}", json!({
        "identities": identities(), "matrices": matrices(), "repairs": repairs()
    }));
}
'''

HOST_SOURCE = r'''
use __ARITHMETIC__::exact::{RatSymMatrix, SosBlock, SosCertificate};
use __ARITHMETIC__::poly::RationalPoly;
use chelis_prove::clarabel_sos::encode::SosLayout;
use chelis_prove::clarabel_sos::extract::PolyOnInterval;
use chelis_prove::clarabel_sos::{ClarabelSosEngine, SosProposer};
use chelis_prove::discharge::{Discharge, DischargeEngine, Goal};
use chelis_prove::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};
use chelis_prove::tier_b::SmtProperty;
use num_bigint::BigInt;
use num_rational::BigRational;
use serde_json::{json, Map, Value};

#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn q(n: i64) -> BigRational { BigRational::from_integer(BigInt::from(n)) }
fn goal(coeffs: &[i64], lo: i64, hi: i64, strict: bool) -> Goal {
    let x = || SmtExpr::Var("x".into());
    let cmp = |op, lhs, rhs| SmtExpr::Cmp(op, Box::new(lhs), Box::new(rhs));
    let mut body = SmtExpr::RealLit(0.0);
    for &coefficient in coeffs.iter().rev() {
        body = SmtExpr::Arith(ArithOp::Add,
            Box::new(SmtExpr::Arith(ArithOp::Mul, Box::new(body), Box::new(x()))),
            Box::new(SmtExpr::RealLit(coefficient as f64)));
    }
    Goal::smt(SmtProperty {
        variables: vec![("x".into(), SmtSort::Real)],
        preconditions: vec![
            cmp(CmpOp::Ge, x(), SmtExpr::RealLit(lo as f64)),
            cmp(CmpOp::Le, x(), SmtExpr::RealLit(hi as f64)),
        ],
        postcondition: cmp(if strict { CmpOp::Gt } else { CmpOp::Ge }, body, SmtExpr::RealLit(0.0)),
    })
}
fn canonical(coeffs: &[i64], lo: i64, hi: i64, vars: &[i64]) -> SosCertificate {
    let poly = RationalPoly::from_int_coeffs(coeffs);
    let layout = SosLayout::for_goal(poly.degree().unwrap_or(0), &q(lo), &q(hi));
    layout.certificate_from_vars(&vars.iter().map(|&n| q(n)).collect::<Vec<_>>(), &poly, &q(lo), &q(hi))
}
struct Candidate(SosCertificate);
impl SosProposer for Candidate {
    fn propose(&self, _: &PolyOnInterval) -> Option<SosCertificate> { Some(self.0.clone()) }
}
fn record(discharge: Discharge) -> Value {
    json!({
        "result": format!("{:?}", discharge.result()),
        "soundness": format!("{:?}", discharge.soundness()),
        "qualifiers": discharge.qualifier_set().iter().map(|q| format!("{q:?}")).collect::<Vec<_>>(),
        "evidence": discharge.evidence(),
    })
}
fn admit(cert: SosCertificate, request: &Goal) -> Value {
    record(ClarabelSosEngine::with_proposer(Candidate(cert)).discharge(request, 5_000))
}
fn main() {
    let mut canonical_cases = Map::new();
    for (name, coeffs, lo, hi, vars) in [
        ("constant", vec![1], -1, 1, vec![1]),
        ("even", vec![1, 0, 1], -1, 1, vec![1, 0, 1, 0]),
        ("odd", vec![0, 1], 0, 1, vec![1, 0]),
        ("cubic", vec![0, 0, 0, 1], 0, 1, vec![0, 0, 1, 0, 0, 0]),
        ("zero", vec![0], -1, 1, vec![0]),
        ("equal_endpoints", vec![0, 0, 1], 0, 0, vec![0, 0, 1, 0]),
        ("singular", vec![1, 2, 1], -1, 1, vec![1, 1, 1, 0]),
    ] {
        let cert = canonical(&coeffs, lo, hi, &vars);
        assert_eq!(cert.verify(), Ok(()));
        canonical_cases.insert(name.into(), admit(cert, &goal(&coeffs, lo, hi, false)));
    }
    let cert = canonical(&[0, 0, 1], -1, 1, &[0, 0, 1, 0]);
    let mut binding = Map::new();
    for (name, coeffs, lo, hi, strict) in [
        ("wrong_goal", vec![-1, 0, 1], -1, 1, false),
        ("wrong_domain", vec![0, 0, 1], -1, 2, false),
        ("strict_same", vec![0, 0, 1], -1, 1, true),
        ("strict_other", vec![1, 0, 1], -1, 1, true),
    ] {
        binding.insert(name.into(), admit(cert.clone(), &goal(&coeffs, lo, hi, strict)));
    }
    let reversed = goal(&[1], 2, -2, false);
    assert!(!ClarabelSosEngine::unwired().fitness(&reversed));
    binding.insert("reversed".into(), record(ClarabelSosEngine::unwired().discharge(&reversed, 5_000)));
    let mut errors = Map::new();
    for (entries, coeffs, label) in [
        ([1, 1, 0, 1], vec![1, 1, 1], "not_symmetric"),
        ([1, 2, 2, 1], vec![1, 4, 1], "not_psd"),
        ([1, 0, 0, 1], vec![0, 0, 1], "polynomial_mismatch"),
    ] {
        let mut cert = canonical(&coeffs, -1, 1, &[0, 0, 1, 0]);
        cert.blocks[0].gram = RatSymMatrix::from_rows(2, entries.map(q).to_vec());
        errors.insert(label.into(), admit(cert, &goal(&coeffs, -1, 1, false)));
    }
    let mut layout_repair = Map::new();
    let negative = SosCertificate {
        poly: RationalPoly::from_int_coeffs(&[-1]), lo: q(0), hi: q(1),
        blocks: vec![SosBlock {
            gram: RatSymMatrix::from_rows(1, vec![q(1)]),
            multiplier: RationalPoly::from_int_coeffs(&[-1]),
        }],
    };
    let mut altered = canonical(&[0, 0, 1], -1, 1, &[0, 0, 1, 0]);
    altered.blocks[1].multiplier = RationalPoly::from_int_coeffs(&[-999]);
    let mut swapped = canonical(&[0, 1], 0, 1, &[1, 0]);
    swapped.blocks.swap(0, 1);
    let mut missing = canonical(&[0, 0, 1], -1, 1, &[0, 0, 1, 0]);
    missing.blocks.pop();
    let mut extra = canonical(&[1], 0, 1, &[1]);
    extra.blocks.push(SosBlock {
        gram: RatSymMatrix::from_rows(1, vec![q(0)]), multiplier: RationalPoly::from_int_coeffs(&[1]),
    });
    let mut wrong_size = canonical(&[1], 0, 1, &[1]);
    wrong_size.blocks[0].gram = RatSymMatrix::from_rows(2, vec![q(1), q(0), q(0), q(0)]);
    for (name, cert, request) in [
        ("negative", negative, goal(&[-1], 0, 1, false)),
        ("altered", altered, goal(&[0, 0, 1], -1, 1, false)),
        ("swapped", swapped, goal(&[0, 1], 0, 1, false)),
        ("missing", missing, goal(&[0, 0, 1], -1, 1, false)),
        ("extra", extra, goal(&[1], 0, 1, false)),
        ("wrong_size", wrong_size, goal(&[1], 0, 1, false)),
    ] {
        let arithmetic = cert.verify();
        assert_eq!(arithmetic, Ok(()));
        layout_repair.insert(name.into(), json!({
            "arithmetic": format!("{arithmetic:?}"), "admission": admit(cert, &request)
        }));
    }
    let engine = ClarabelSosEngine::clarabel();
    let mut production = Map::new();
    for (name, coeffs, lo, hi, strict) in [
        ("constant", vec![1], -1, 1, false),
        ("even", vec![1, 0, 1], -1, 1, false),
        ("odd", vec![0, 1], 0, 1, false),
        ("equal_endpoints", vec![0, 0, 1], 0, 0, false),
        ("zero", vec![0], -1, 1, false),
        ("negative", vec![-1], -1, 1, false),
        ("strict", vec![1, 0, 1], -1, 1, true),
    ] {
        production.insert(name.into(), record(engine.discharge(&goal(&coeffs, lo, hi, strict), 5_000)));
    }
    println!("CERTIFICATE_HOST {}", json!({
        "canonical": canonical_cases, "binding": binding, "errors": errors,
        "layout_repair": layout_repair, "production": production
    }));
}
'''


def fraction_string(value: Fraction | int) -> str:
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def expected_arithmetic() -> dict:
    huge = 1 << 256
    a, b = Fraction(huge + 1, huge + 3), Fraction(huge + 5, huge + 7)
    return {
        "identities": {
            "normalized": ["1/1", "0/1", "1/1"], "valid": "Ok(())",
            "mismatch": "Err(PolynomialMismatch)",
            "huge_coefficients": [fraction_string(a), fraction_string(b)],
            "square": [fraction_string(a * a), fraction_string(2 * a * b), fraction_string(b * b)],
            "evaluation": fraction_string((a + 2 * b) ** 2),
            "tiny_delta_is_distinct": True, "zero_coefficients": [], "zero_degree": None,
            "arithmetic_only": "Ok(())",
        },
        "matrices": {
            "psd": [True, False, False], "singular_form": ["1/1", "2/1", "1/1"],
            "tiny_pivot": fraction_string(Fraction(-1, 1 << 1024)), "tiny_psd": False,
            "errors": ["Err(NotSymmetric { block: 0 })", "Err(NotPsd { block: 1 })"],
        },
        "repairs": {
            "pivots": [[0, 0], [1, 1]], "free": [2],
            "rref": [["1/1", "0/1", "-1/1", "1/1"], ["0/1", "1/1", "1/1", "3/1"]],
            "solution": ["3/1", "1/1", "2/1"], "inconsistent": "Err(Inconsistent)",
            "negative_solution": ["-1/1"], "negative_verification": "Err(NotPsd { block: 0 })",
            "shape_assertions": [True, True, True, True],
        },
    }


CANONICAL_CASES = {
    "constant", "even", "odd", "cubic", "zero", "equal_endpoints", "singular",
}
BINDING_CASES = {"wrong_goal", "wrong_domain", "strict_same", "strict_other", "reversed"}
ERROR_CASES = {"not_symmetric", "not_psd", "polynomial_mismatch"}
REPAIR_CASES = {"negative", "altered", "swapped", "missing", "extra", "wrong_size"}
PRODUCTION_PROOFS = {"constant", "even", "odd", "equal_endpoints"}
PRODUCTION_NONPROOFS = {"zero", "negative", "strict"}


def expected_certificates() -> dict:
    def certificate(poly, lo, hi, blocks):
        return {
            "poly": [fraction_string(n) for n in poly],
            "lo": fraction_string(lo), "hi": fraction_string(hi),
            "blocks": [
                {"gram": [[fraction_string(n) for n in row] for row in gram],
                 "multiplier": [fraction_string(n) for n in multiplier]}
                for gram, multiplier in blocks
            ],
        }

    even = [([[1, 0], [0, 1]], [1]), ([[0]], [1, 0, -1])]
    square = [([[0, 0], [0, 1]], [1]), ([[0]], [1, 0, -1])]
    return {
        "constant": certificate([1], -1, 1, [([[1]], [1])]),
        "even": certificate([1, 0, 1], -1, 1, even),
        "odd": certificate([0, 1], 0, 1, [([[1]], [0, 1]), ([[0]], [1, -1])]),
        "cubic": certificate([0, 0, 0, 1], 0, 1, [
            ([[0, 0], [0, 1]], [0, 1]), ([[0, 0], [0, 0]], [1, -1]),
        ]),
        "zero": certificate([], -1, 1, [([[0]], [1])]),
        "equal_endpoints": certificate([0, 0, 1], 0, 0, [
            ([[0, 0], [0, 1]], [1]), ([[0]], [0, 0, -1]),
        ]),
        "singular": certificate([1, 2, 1], -1, 1, [
            ([[1, 1], [1, 1]], [1]), ([[0]], [1, 0, -1]),
        ]),
        "repair_negative": certificate([-1], 0, 1, [([[1]], [-1])]),
        "repair_altered": certificate([0, 0, 1], -1, 1, [
            (square[0][0], [1]), ([[0]], [-999]),
        ]),
        "repair_swapped": certificate([0, 1], 0, 1, [([[0]], [1, -1]), ([[1]], [0, 1])]),
        "repair_missing": certificate([0, 0, 1], -1, 1, [square[0]]),
        "repair_extra": certificate([1], 0, 1, [([[1]], [1]), ([[0]], [1])]),
        "repair_wrong_size": certificate([1], 0, 1, [([[1, 0], [0, 0]], [1])]),
    }


def fields(value: object, names: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != names:
        raise OracleFailure(f"incomplete {label} evidence")
    return value


def parse_record(stdout: str, prefix: str) -> dict:
    records = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(records) != 1:
        raise OracleFailure(f"missing or repeated structured evidence: {prefix.strip()}")
    try:
        record = json.loads(records[0], object_pairs_hook=unique_fields)
    except (ValueError, TypeError) as error:
        raise OracleFailure(f"invalid structured evidence: {error}") from error
    if not isinstance(record, dict):
        raise OracleFailure("structured evidence must be an object")
    return record


def unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"repeated field: {key}")
        result[key] = value
    return result


def validate_arithmetic(record: dict) -> None:
    if json.dumps(record, sort_keys=True) != json.dumps(expected_arithmetic(), sort_keys=True):
        raise OracleFailure("incomplete or incorrect exact arithmetic evidence")


def validate_rational(value: object) -> None:
    if not isinstance(value, str) or value.count("/") != 1:
        raise OracleFailure("missing exact rational evidence")
    try:
        if fraction_string(Fraction(value)) != value:
            raise ValueError("noncanonical rational")
    except (ValueError, ZeroDivisionError) as error:
        raise OracleFailure("invalid exact rational evidence") from error


def validate_certificate(value: object) -> None:
    certificate = fields(value, {"lo", "hi", "poly", "blocks"}, "certificate")
    validate_rational(certificate["lo"])
    validate_rational(certificate["hi"])
    coefficients = certificate["poly"]
    if not isinstance(coefficients, list):
        raise OracleFailure("missing exact polynomial evidence")
    for coefficient in coefficients:
        validate_rational(coefficient)
    if coefficients and coefficients[-1] == "0/1":
        raise OracleFailure("untrimmed polynomial evidence")
    blocks = certificate["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise OracleFailure("missing exact block evidence")
    for value in blocks:
        block = fields(value, {"gram", "multiplier"}, "block")
        gram = block["gram"]
        if not isinstance(gram, list) or not gram:
            raise OracleFailure("missing exact Gram evidence")
        for row in gram:
            if not isinstance(row, list) or len(row) != len(gram):
                raise OracleFailure("incomplete exact Gram evidence")
            for entry in row:
                validate_rational(entry)
        multiplier = block["multiplier"]
        if not isinstance(multiplier, list) or not multiplier:
            raise OracleFailure("missing exact multiplier evidence")
        for entry in multiplier:
            validate_rational(entry)
        if multiplier[-1] == "0/1":
            raise OracleFailure("untrimmed multiplier evidence")


def validate_discharge(value: object, *, proved: bool) -> None:
    discharge = fields(value, {"result", "soundness", "qualifiers", "evidence"}, "discharge")
    expected = ("Proved", "Exact", ["CertificateBearing"]) if proved else ("Unknown", "Untrusted", [])
    if (discharge["result"], discharge["soundness"], discharge["qualifiers"]) != expected:
        raise OracleFailure("incorrect host authority evidence")
    evidence = fields(discharge["evidence"], {"engine", "backend"}, "engine attribution")
    if evidence["engine"] != "clarabel_sos":
        raise OracleFailure("incorrect engine attribution evidence")
    if proved:
        backend = fields(evidence["backend"], {"outcome", "strict", "certificate"}, "proof")
        if backend["outcome"] != "proved" or backend["strict"] is not False:
            raise OracleFailure("incorrect canonical proof evidence")
        validate_certificate(backend["certificate"])
    else:
        backend = fields(evidence["backend"], {"outcome", "reason", "detail"}, "nonproof")
        if backend["outcome"] != "unknown" or not isinstance(backend["reason"], str) or not backend["reason"]:
            raise OracleFailure("missing nonproof reason evidence")
        if backend["detail"] is not None and not isinstance(backend["detail"], dict):
            raise OracleFailure("invalid nonproof detail evidence")


def validate_host(record: dict, *, candidate: bool) -> None:
    fields(record, {"canonical", "binding", "errors", "layout_repair", "production"}, "host")
    for section, names in (
        ("canonical", CANONICAL_CASES), ("binding", BINDING_CASES),
        ("errors", ERROR_CASES), ("layout_repair", REPAIR_CASES),
        ("production", PRODUCTION_PROOFS | PRODUCTION_NONPROOFS),
    ):
        fields(record[section], names, section)
    certificates = expected_certificates()
    for name, value in record["canonical"].items():
        validate_discharge(value, proved=True)
        if value["evidence"]["backend"]["certificate"] != certificates[name]:
            raise OracleFailure(f"incomplete or incorrect canonical certificate evidence: {name}")
    for value in record["binding"].values():
        validate_discharge(value, proved=False)
    for label, value in record["errors"].items():
        validate_discharge(value, proved=False)
        if value["evidence"]["backend"]["detail"] != {"cert_error": label}:
            raise OracleFailure("incorrect public error label evidence")
    for name, value in record["production"].items():
        validate_discharge(value, proved=name in PRODUCTION_PROOFS)
        if name in PRODUCTION_PROOFS:
            certificate = value["evidence"]["backend"]["certificate"]
            if any(certificate[field] != certificates[name][field] for field in ("poly", "lo", "hi")):
                raise OracleFailure(f"incorrect production goal evidence: {name}")
    for name, value in record["layout_repair"].items():
        fields(value, {"arithmetic", "admission"}, "layout repair")
        if value["arithmetic"] != "Ok(())":
            raise OracleFailure("layout repair changed arithmetic verification")
        validate_discharge(value["admission"], proved=not candidate)
        backend = value["admission"]["evidence"]["backend"]
        if candidate:
            if backend != {
                "outcome": "unknown",
                "reason": "certificate does not match the canonical interval layout",
                "detail": {"cert_matches_goal_layout": False},
            }:
                raise OracleFailure("layout rejection lacks the designed repair evidence")
        elif backend["certificate"] != certificates["repair_" + name]:
            raise OracleFailure("incomplete original layout admission evidence")


def compare_snapshots(baseline: dict, candidate: dict) -> None:
    for record, current in ((baseline, False), (candidate, True)):
        fields(record, {"arithmetic", "host"}, "snapshot")
        validate_arithmetic(record["arithmetic"])
        validate_host(record["host"], candidate=current)
    if baseline["arithmetic"] != candidate["arithmetic"]:
        raise OracleFailure("exact arithmetic differs from the immutable base")
    for section in ("canonical", "binding", "errors", "production"):
        if baseline["host"][section] != candidate["host"][section]:
            raise OracleFailure(f"{section} values, outcomes, or exact evidence differ from the immutable base")


def checkout_baseline(directory: Path) -> Path:
    directory = directory.resolve()
    run(("git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(directory)))
    present = support._execute(("git", "cat-file", "-e", f"{BASELINE}^{{commit}}"), cwd=directory)
    if present.returncode != 0:
        shallow = run(("git", "rev-parse", "--is-shallow-repository"), cwd=ROOT)
        if shallow.stdout.strip() != "true":
            raise OracleFailure("immutable baseline commit is absent from a complete clone")
        run(("git", "fetch", "--quiet", "--no-tags", "--depth=1", BASELINE_REPOSITORY, BASELINE), cwd=directory)
    run(("git", "checkout", "--quiet", "--detach", BASELINE), cwd=directory)
    actual = run(("git", "rev-parse", "HEAD"), cwd=directory)
    if actual.stdout.strip() != BASELINE:
        raise OracleFailure("baseline revision differs from the required commit")
    for path in BASELINE_FILES:
        if not (directory / path).is_file():
            raise OracleFailure(f"missing original baseline evidence: {path}")
    return directory


def native_features() -> tuple[str, ...]:
    if sys.platform == "linux":
        return ("ci-openblas-system",)
    if sys.platform == "darwin":
        return ("clarabel",)
    raise OracleFailure("native certificate parity needs Linux or macOS")


def dependency(workspace: Path, name: str) -> str:
    return f'{name} = {{ path = {json.dumps(str(workspace / "crates" / name))}, default-features = false }}\n'


def snapshot(workspace: Path, *, candidate: bool) -> dict:
    arithmetic_owner = "chelis_certificate_core" if candidate else "chelis_prove::clarabel_sos"
    extra = RATIONAL_DEPENDENCIES + 'serde_json = "1"\n'
    arithmetic_result = consumer(
        workspace / "crates" / (CRATE if candidate else "chelis-prove"),
        ARITHMETIC_SOURCE.replace("__ARITHMETIC__", arithmetic_owner),
        features=() if candidate else native_features(), dependencies=extra,
    )
    host_result = consumer(
        workspace / "crates/chelis-prove",
        HOST_SOURCE.replace("__ARITHMETIC__", arithmetic_owner),
        features=native_features(),
        dependencies=extra + (dependency(workspace, CRATE) if candidate else ""),
    )
    return {
        "arithmetic": parse_record(arithmetic_result.stdout, "CERTIFICATE_ARITHMETIC "),
        "host": parse_record(host_result.stdout, "CERTIFICATE_HOST "),
    }


def target_graphs() -> None:
    for defaults in (False, True):
        names = portable(CRATE, PORTABLE_SOURCE, default_features=defaults,
                         dependencies=RATIONAL_DEPENDENCIES, forbidden=FORBIDDEN)
        if names != PORTABLE_PACKAGES:
            raise OracleFailure(f"unexpected portable dependency closure: {sorted(names)}")
        print(f"Target compilation: thumbv7em-none-eabi, default-features={str(defaults).lower()}.", flush=True)


def negative_controls() -> None:
    for module in ("exact", "poly", "linalg"):
        compile_rejected(
            "chelis-prove", f"use chelis_prove::clarabel_sos::{module};\n",
            features=native_features(), expected=("E0432", module),
        )


def native_suites() -> None:
    run(("cargo", "nextest", "run", "--locked", "--no-fail-fast", "-p", CRATE,
         "--all-targets", "--ignore-default-filter"),
        required_tests=tuple(f"{CRATE}::exact_contract${name}" for name in CORE_TESTS))
    features = ",".join(f"chelis-prove/{feature}" for feature in native_features())
    run(("cargo", "nextest", "run", "--locked", "--no-fail-fast", "-p", "chelis-prove",
         "--features", features, "--ignore-default-filter",
         "-E", "test(clarabel_sos::) | binary(certificate_core_contract)"),
        required_tests=tuple(f"chelis-prove::certificate_core_contract${name}" for name in HOST_TESTS))


def main() -> int:
    native_features()
    target_graphs()
    negative_controls()
    with tempfile.TemporaryDirectory(prefix="chelis-certificate-baseline-") as temporary:
        baseline = checkout_baseline(Path(temporary) / "baseline")
        previous = snapshot(baseline, candidate=False)
        current = snapshot(ROOT, candidate=True)
        compare_snapshots(previous, current)
        print("CERTIFICATE_COMPARISON " + json.dumps({
            "baseline": BASELINE, "before": previous, "after": current,
            "intentional_admission_differences": sorted(REPAIR_CASES),
        }, sort_keys=True), flush=True)
    native_suites()
    run((sys.executable, "scripts/test_no_std_core_support.py"))
    from test_certificate_core_oracle import MANDATORY_TESTS
    result = run((sys.executable, "scripts/test_certificate_core_oracle.py"))
    if parse_record(result.stdout, "CERTIFICATE_ORACLE_TESTS ") != {"passed": sorted(MANDATORY_TESTS), "skipped": 0}:
        raise OracleFailure("incomplete oracle failure-control test evidence")
    print("certificate core oracle: PASS", flush=True)
    print("Native host evidence: external allocator consumer, frozen-base parity, and complete mandatory suites.", flush=True)
    print("No device execution, allocator capacity, or whole-pipeline portability claim.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleFailure, OSError, ValueError) as error:
        print(f"certificate core oracle: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
