//! A MOCK `chelis-beacon` binary for the [`crate::beacon_shim`] tests
//! (chelis#439). It stands in for the real out-of-tree verifier so every §5
//! mapping row of `docs/design/beacon_subprocess_shim.md` can be exercised in
//! CI without a real Beacon.
//!
//! It accepts the same CLI the shim drives — `dispatch --request -` (stdin) or
//! `dispatch --request <path>` (temp file) — reads the request (to prove the
//! shim wrote a well-formed one when asked), and emits a canned `CheckReport`
//! JSON selected by the `MOCK_BEACON_SCENARIO` env var. It is a test fixture,
//! not a product binary; it links no chelis crate and contains no verifier
//! logic.
//!
//! Scenarios (`MOCK_BEACON_SCENARIO`):
//!
//! - `proved`                       -> `{"verdict":"proved"}` (oracle-verified)
//! - `proved_oracle_unverified`     -> `{"verdict":"proved_oracle_unverified"}`
//! - `refuted_verified`             -> `{"verdict":"refuted","oracle_verified":true,...}`
//! - `refuted_unverified`           -> `{"verdict":"refuted","oracle_verified":false,...}`
//! - `refuted_no_flag`              -> `{"verdict":"refuted",...}` (no flag)
//! - `nonzero_exit`                 -> stderr + exit 3
//! - `unparseable`                  -> non-JSON stdout, exit 0
//! - `hang`                         -> sleep far past any test timeout (hard-kill arm)
//! - `echo_request`                 -> echo the received request back on stdout (request-shape arm)
//! - default / unset                -> `proved`
//!
//! The request the shim sends is read from stdin or the `--request <path>` file;
//! `echo_request` writes it straight back so a test can assert the exact base64
//! and `expected_dag_sha256` the shim emitted.

use std::io::Read;

fn read_request(args: &[String]) -> String {
    // Find `--request <value>`; `-` means stdin, anything else is a file path.
    let mut request_arg = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--request" {
            request_arg = it.next().cloned();
            break;
        }
    }
    match request_arg.as_deref() {
        Some("-") | None => {
            let mut buf = String::new();
            let _ = std::io::stdin().read_to_string(&mut buf);
            buf
        }
        Some(path) => std::fs::read_to_string(path).unwrap_or_default(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scenario = std::env::var("MOCK_BEACON_SCENARIO").unwrap_or_else(|_| "proved".to_string());

    // `hang_no_drain` deliberately does NOT read its input: it models a child
    // that has not yet started draining stdin, so a large stdin write would
    // block the parent's `write_all` on a full pipe. The shim's large-request
    // auto-fallback to the temp-file transport is what keeps the hard kill
    // reachable; this scenario exists to prove that. All other scenarios drain
    // the request first (a normal child reads its input before responding).
    if scenario == "hang_no_drain" {
        std::thread::sleep(std::time::Duration::from_secs(600));
        return;
    }

    // Drain the request so the shim's stdin write (small-request path) or its
    // temp-file write (large-request path) is consumed before we respond.
    let request = read_request(&args);

    match scenario.as_str() {
        "proved" => {
            print!(r#"{{"verdict":"proved","oracle_verified":true}}"#);
        }
        "large_report" => {
            println!("{}", serde_json::json!({"verdict":"proved","oracle_verified":true,
                "evidence":{"tree_payload":"x".repeat(256 * 1024)}}));
        }
        "proved_oracle_unverified" => {
            print!(r#"{{"verdict":"proved_oracle_unverified"}}"#);
        }
        "proved_no_flag" => {
            // A `proved` verdict with NO oracle_verified field: the normal
            // verified case (the discriminator is the proved_oracle_unverified
            // token, not the flag), so the shim accepts it as SoundApproximate.
            print!(r#"{{"verdict":"proved"}}"#);
        }
        "proved_oracle_false" => {
            // Self-contradictory: a `proved` verdict that explicitly claims its
            // oracle did NOT verify. The shim must fail this closed (the MED).
            print!(r#"{{"verdict":"proved","oracle_verified":false}}"#);
        }
        "refuted_verified" => {
            print!(
                r#"{{"verdict":"refuted","oracle_verified":true,"counterexample":{{"s":42.0}}}}"#
            );
        }
        "refuted_unverified" => {
            print!(
                r#"{{"verdict":"refuted","oracle_verified":false,"counterexample":{{"s":42.0}}}}"#
            );
        }
        "refuted_no_flag" => {
            print!(r#"{{"verdict":"refuted","counterexample":{{"s":42.0}}}}"#);
        }
        "nonzero_exit" => {
            eprint!("mock beacon: simulated verifier failure");
            std::process::exit(3);
        }
        "unparseable" => {
            print!("this is not json at all <<<");
        }
        "hang" => {
            // Sleep far beyond any test timeout so the shim's hard-kill fires
            // (this scenario DOES drain its input first).
            std::thread::sleep(std::time::Duration::from_secs(600));
        }
        "echo_request" => {
            // Echo the received request so a test can assert its exact shape.
            print!("{request}");
        }
        "crash" => {
            // Simulate a process abort (e.g., OOM kill, SIGABRT from a C
            // library assertion). The shim must produce an honest
            // `TierBResult::Error` with a reason naming the signal, not a
            // hang or a silent pass. (chelis_plan Task 2.)
            std::process::abort();
        }
        "partial_output" => {
            // Emit partial (incomplete) JSON, then exit. Simulates a child
            // that crashes mid-output (e.g., stack overflow while serializing
            // the CheckReport). The shim must produce an honest unparseable-
            // report error, not a hang. (chelis_plan Task 2.)
            print!(r#"{{"verdict":"prov"#);
            std::process::exit(0);
        }
        other => {
            eprint!("mock beacon: unknown scenario `{other}`");
            std::process::exit(2);
        }
    }
}
