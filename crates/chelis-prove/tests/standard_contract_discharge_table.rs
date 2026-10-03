//! chelis#2957: the committed standard-contract discharge table matches a
//! recomputation of every row, byte for byte. Recomputing runs every fuzz
//! discharge at every width (about 34 s in a debug build), so this target runs
//! nightly; `contracts::tests::chelis_2957_discharge_table_canary_*` recomputes
//! one row per pull request.

mod support;

use std::path::Path;

use chelis_prove::{
    STANDARD_CONTRACT_DISCHARGE_TABLE_PATH, committed_standard_contract_discharge_table,
    recompute_standard_contract_discharge_table,
};

#[test]
fn standard_contract_discharge_table_matches_recomputation() {
    crate::support::isolate();
    let recomputed =
        recompute_standard_contract_discharge_table(&chelis_std_bundle::EMBEDDED_RUNTIME).unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(STANDARD_CONTRACT_DISCHARGE_TABLE_PATH);
    if std::env::var("CHELIS_PROVE_DISCHARGE_TABLE_WRITE").as_deref() == Ok("1") {
        std::fs::write(&path, &recomputed).unwrap();
        return;
    }
    assert!(
        recomputed == committed_standard_contract_discharge_table(),
        "{} is stale; rerun this test with CHELIS_PROVE_DISCHARGE_TABLE_WRITE=1 and \
         review the verdict changes",
        path.display()
    );
}
