use chelis_clarabel_provider::{Cone, DenseMatrix, DenseVector, Problem, Settings, Status, solve};

fn vector(values: &[f64]) -> DenseVector {
    DenseVector::new(values.to_vec()).unwrap()
}

fn matrix(rows: usize, cols: usize, values: &[f64]) -> DenseMatrix {
    DenseMatrix::new(rows, cols, values.to_vec()).unwrap()
}

#[test]
fn unconstrained_qp_returns_primal_and_status() {
    let problem = Problem::new(
        matrix(1, 1, &[2.0]),
        vector(&[-4.0]),
        matrix(0, 1, &[]),
        vector(&[]),
        vec![],
    )
    .unwrap();
    let result = solve(&problem, &Settings::default()).unwrap();
    assert_eq!(result.status(), Status::Solved);
    assert!((result.primal().as_slice()[0] - 2.0).abs() < 1e-7);
    assert!(result.dual().as_slice().is_empty());
    assert!(result.slack().as_slice().is_empty());
}

#[test]
fn equality_and_inequality_qps_solve_with_distinct_cones() {
    let equality = Problem::new(
        matrix(1, 1, &[2.0]),
        vector(&[0.0]),
        matrix(1, 1, &[1.0]),
        vector(&[1.0]),
        vec![Cone::zero(1).unwrap()],
    )
    .unwrap();
    let eq_result = solve(&equality, &Settings::default()).unwrap();
    assert_eq!(eq_result.status(), Status::Solved);
    assert!((eq_result.primal().as_slice()[0] - 1.0).abs() < 1e-7);

    let inequality = Problem::new(
        matrix(1, 1, &[2.0]),
        vector(&[0.0]),
        matrix(1, 1, &[-1.0]),
        vector(&[-1.0]),
        vec![Cone::nonnegative(1).unwrap()],
    )
    .unwrap();
    let ineq_result = solve(&inequality, &Settings::default()).unwrap();
    assert_eq!(ineq_result.status(), Status::Solved);
    assert!((ineq_result.primal().as_slice()[0] - 1.0).abs() < 1e-6);
    assert!(ineq_result.slack().as_slice()[0].abs() < 1e-6);
}

#[test]
fn rejects_shape_cone_and_finite_data_errors_before_solver() {
    assert!(DenseMatrix::new(2, 2, vec![1.0]).is_err());
    assert!(DenseVector::new(vec![f64::NAN]).is_err());
    assert!(Cone::power(0.0).is_err());
    assert!(Cone::second_order(1).is_err());

    let bad_q = Problem::new(
        matrix(1, 1, &[1.0]),
        vector(&[1.0, 2.0]),
        matrix(0, 1, &[]),
        vector(&[]),
        vec![],
    );
    assert!(bad_q.is_err());

    let bad_cones = Problem::new(
        matrix(1, 1, &[1.0]),
        vector(&[0.0]),
        matrix(1, 1, &[1.0]),
        vector(&[0.0]),
        vec![],
    );
    assert!(bad_cones.is_err());

    let asymmetric = Problem::new(
        matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]),
        vector(&[0.0, 0.0]),
        matrix(0, 2, &[]),
        vector(&[]),
        vec![],
    );
    assert!(asymmetric.is_err());
}

#[test]
fn non_psd_cone_families_are_representable_and_validated() {
    assert_eq!(Cone::second_order(4).unwrap().dimension(), 4);
    assert_eq!(Cone::exponential().dimension(), 3);
    assert_eq!(Cone::power(0.4).unwrap().dimension(), 3);
    assert_eq!(
        Cone::generalized_power(vec![0.25, 0.75], 2)
            .unwrap()
            .dimension(),
        4
    );
    assert!(Cone::generalized_power(vec![0.25, 0.5], 2).is_err());
    assert!(Cone::generalized_power(vec![0.0, 1.0], 2).is_err());
}

#[test]
fn all_nonlinear_cone_families_solve_an_interior_feasibility_problem() {
    let cases = [
        (Cone::second_order(3).unwrap(), vec![2.0, 0.0, 0.0]),
        (Cone::exponential(), vec![0.0, 1.0, 2.0]),
        (Cone::power(0.5).unwrap(), vec![1.0, 1.0, 0.0]),
        (
            Cone::generalized_power(vec![0.5, 0.5], 1).unwrap(),
            vec![1.0, 1.0, 0.0],
        ),
    ];
    for (cone, b) in cases {
        let problem = Problem::new(
            matrix(1, 1, &[2.0]),
            vector(&[0.0]),
            matrix(b.len(), 1, &vec![0.0; b.len()]),
            vector(&b),
            vec![cone],
        )
        .unwrap();
        let result = solve(&problem, &Settings::default()).unwrap();
        assert_eq!(result.status(), Status::Solved);
        assert!(result.primal().as_slice()[0].abs() < 1e-6);
    }
}

#[test]
fn iteration_limit_is_a_status_without_an_optimality_claim() {
    let problem = Problem::new(
        matrix(1, 1, &[2.0]),
        vector(&[0.0]),
        matrix(1, 1, &[-1.0]),
        vector(&[-1.0]),
        vec![Cone::nonnegative(1).unwrap()],
    )
    .unwrap();
    let settings = Settings::new(0, 1e-8, 1e-8, 1e-8).unwrap();
    let result = solve(&problem, &settings).unwrap();
    assert_eq!(result.status(), Status::MaxIterations);
}
