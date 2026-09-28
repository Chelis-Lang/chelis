//! Every integration binary installs one exact libtest worker entry.

pub fn isolate() {
    chelis_prove::worker::test_support::enable_isolation("support::solver_worker");
}

#[test]
fn solver_worker() {
    chelis_prove::worker::test_support::run_worker_if_requested("support::solver_worker");
}
