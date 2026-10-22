use chelis_certificate_core::exact::{CertError, RatSymMatrix, SosBlock, SosCertificate};
use chelis_certificate_core::linalg::{ProjectError, RatMatrix, project_onto_affine, rref};
use chelis_certificate_core::poly::RationalPoly;
use num_bigint::BigInt;
use num_rational::BigRational;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

fn matrix(n: usize, entries: &[i64]) -> RatSymMatrix {
    RatSymMatrix::from_rows(n, entries.iter().map(|&n| q(n)).collect())
}

fn certificate(gram: RatSymMatrix) -> SosCertificate {
    SosCertificate {
        poly: gram.quadratic_form_poly(),
        lo: q(-1),
        hi: q(1),
        blocks: vec![SosBlock {
            gram,
            multiplier: RationalPoly::from_int_coeffs(&[1]),
        }],
    }
}

#[test]
fn identities_and_normalization_are_exact() {
    let mut cert = certificate(matrix(2, &[1, 0, 0, 1]));
    assert_eq!(cert.poly, RationalPoly::from_int_coeffs(&[1, 0, 1, 0, 0]));
    assert_eq!(cert.verify(), Ok(()));
    cert.poly = RationalPoly::from_int_coeffs(&[1, 1, 1]);
    assert_eq!(cert.verify(), Err(CertError::PolynomialMismatch));
    let zero = RationalPoly::from_int_coeffs(&[0, 0]);
    assert_eq!(zero.degree(), None);
    assert!(zero.coeffs().is_empty());
    let huge = BigInt::from(1u8) << 256usize;
    let a = BigRational::new(&huge + 1, &huge + 3);
    let b = BigRational::new(&huge + 5, &huge + 7);
    let poly = RationalPoly::from_coeffs(vec![a.clone(), b.clone()]);
    assert_eq!(poly.mul(&poly).eval(&q(2)), (&a + &b * q(2)).pow(2));
    let delta = BigRational::new(BigInt::from(1), huge);
    assert_ne!(poly, poly.add(&RationalPoly::constant(delta)));
}

#[test]
fn exact_psd_includes_singular_and_excludes_tiny_negative_pivots() {
    let singular = matrix(2, &[1, 1, 1, 1]);
    assert!(singular.is_psd());
    assert_eq!(
        singular.quadratic_form_poly(),
        RationalPoly::from_int_coeffs(&[1, 2, 1])
    );
    assert!(!matrix(2, &[0, 1, 1, 0]).is_psd());
    assert!(!matrix(2, &[1, 2, 2, 1]).is_psd());
    let tiny = -BigRational::new(BigInt::from(1), BigInt::from(1u8) << 1024usize);
    assert!(!RatSymMatrix::from_rows(1, vec![tiny]).is_psd());
}

#[test]
fn errors_preserve_block_order_and_symmetry_precedence() {
    let mut cert = certificate(matrix(2, &[-1, 1, 0, -1]));
    cert.blocks.push(SosBlock {
        gram: matrix(1, &[-1]),
        multiplier: RationalPoly::from_int_coeffs(&[1]),
    });
    assert_eq!(cert.verify(), Err(CertError::NotSymmetric { block: 0 }));
    cert.blocks[0].gram = matrix(1, &[1]);
    assert_eq!(cert.verify(), Err(CertError::NotPsd { block: 1 }));
}

#[test]
fn arithmetic_success_does_not_validate_multiplier_or_domain() {
    let mut cert = certificate(matrix(1, &[1]));
    cert.poly = RationalPoly::from_int_coeffs(&[-1]);
    cert.blocks[0].multiplier = cert.poly.clone();
    cert.lo = q(2);
    cert.hi = q(-2);
    assert_eq!(cert.verify(), Ok(()));
}

#[test]
fn repair_preserves_free_coordinates_and_pivot_order() {
    let aug = RatMatrix::from_data(2, 4, [0, 1, 1, 3, 1, 1, 0, 4].map(q).to_vec());
    let reduced = rref(&aug, 3);
    assert_eq!(reduced.pivots, vec![(0, 0), (1, 1)]);
    assert_eq!(reduced.free_columns(), vec![2]);
    assert_eq!(
        project_onto_affine(&aug, 3, &[q(9), q(9), q(2)]),
        Ok(vec![q(3), q(1), q(2)])
    );
    let inconsistent = RatMatrix::from_data(2, 2, [1, 1, 1, 2].map(q).to_vec());
    assert_eq!(
        project_onto_affine(&inconsistent, 1, &[q(0)]),
        Err(ProjectError::Inconsistent)
    );
    let negative = RatMatrix::from_data(1, 2, [1, -1].map(q).to_vec());
    let repaired = project_onto_affine(&negative, 1, &[q(4)]).unwrap();
    assert_eq!(
        certificate(RatSymMatrix::from_rows(1, repaired)).verify(),
        Err(CertError::NotPsd { block: 0 })
    );
}

#[test]
fn constructors_and_repair_preserve_shape_assertions() {
    assert!(std::panic::catch_unwind(|| RatSymMatrix::from_rows(2, vec![q(1)])).is_err());
    assert!(std::panic::catch_unwind(|| RatMatrix::from_data(2, 2, vec![q(1)])).is_err());
    let aug = RatMatrix::zeros(1, 2);
    assert!(std::panic::catch_unwind(|| rref(&aug, 2)).is_err());
    assert!(std::panic::catch_unwind(|| project_onto_affine(&aug, 1, &[])).is_err());
}
