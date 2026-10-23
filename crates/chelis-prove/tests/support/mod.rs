//! Every integration binary installs one exact libtest worker entry.

pub fn isolate() {
    chelis_prove::worker::test_support::enable_isolation("support::solver_worker");
}

#[allow(dead_code)]
pub fn tagged_f64(value: f64) -> chelis_numeric_core::dtype_semantics::ScalarValue {
    chelis_numeric_core::dtype_semantics::scalar_from_f64(
        "test box bound",
        chelis_numeric_core::prim::Prim::F64,
        value,
    )
    .expect("f64 is a valid tagged test bound")
}

#[allow(dead_code)]
pub fn tagged_dims(
    dims: Vec<(String, f64, f64)>,
) -> Vec<(
    String,
    chelis_numeric_core::dtype_semantics::ScalarValue,
    chelis_numeric_core::dtype_semantics::ScalarValue,
)> {
    dims.into_iter()
        .map(|(name, lo, hi)| (name, tagged_f64(lo), tagged_f64(hi)))
        .collect()
}

#[test]
fn solver_worker() {
    chelis_prove::worker::test_support::run_worker_if_requested("support::solver_worker");
}
