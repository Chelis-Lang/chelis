//! Beacon subprocess shim (chelis#439): a transport-only [`DischargeEngine`]
//! that routes a [`GoalShape::BoxRange`] goal to an out-of-tree `chelis-beacon`
//! verifier binary.
//!
//! Beacon's verifier logic stays OUT of tree. This shim only:
//!
//! 1. transports the goal (its box/range bounds + the serialized exact-version `WireDag` v6
//!    bytes, inline base64) to the pinned binary;
//! 2. enforces the discharge `timeout_ms` as a HARD subprocess kill;
//! 3. maps the returned `CheckReport` JSON to a [`Discharge`], fail-closed, via
//!    the integrity gate [`Discharge::new`].
//!
//! The full contract is `docs/design/beacon_subprocess_shim.md`; the frozen
//! surfaces it consumes without changing are `docs/design/phase2_seam_contract.md`.
//!
//! # The byte flow (content-addressed store, no frozen-surface change)
//!
//! The frozen [`DischargeEngine::discharge`] signature gives the shim only the
//! [`Goal`], whose [`IrHandle`] carries `dag_hash` + `root_index` and NO bytes.
//! The serialized exact-version `WireDag` v6 bytes the shim must transport live in
//! [`crate::graph_extract::ExtractedGoal::wire_dag_bytes`], which never enters
//! the `Goal`. A [`WireDagByteStore`] bridges the two: the dispatch site (the
//! caller that runs the WI-3 producer and registers the shim) populates the
//! store with `dag_hash -> wire_dag_bytes` at extraction time and passes it into
//! [`BeaconShim::new`]; the shim looks the bytes up by `goal.ir.dag_hash()`.
//! This changes nothing about [`IrHandle`], [`Goal`], the trait, or the
//! registry.
//!
//! # Solver-free
//!
//! The shim is TRANSPORT ONLY: no solver linkage, no cvc5-named symbol. It ships
//! in the DEFAULT (non-smt) build and keeps it solver-free (the
//! `check_is_solver_free_on_the_corpus` gate). Its deps (`base64`,
//! `wait-timeout`, `std::process`, `serde_json`, `sha2`) are transport
//! utilities.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use sha2::{Digest, Sha256};
use wait_timeout::ChildExt as _;

use crate::discharge::{
    Discharge, DischargeEngine, Goal, GoalShape, IntervalBox, OutputRange, Qualifier, QualifierSet,
    Soundness,
};
use crate::tier_b::TierBResult;

/// The schema version of the request the shim emits. Beacon pins against this.
const REQUEST_SCHEMA_VERSION: u32 = 2;

/// The request-size ceiling above which the [`RequestTransport::Stdin`] path
/// auto-falls-back to [`RequestTransport::TempFile`] to stay deadlock-safe.
///
/// Writing a request larger than the OS pipe buffer (commonly 64 KiB on Linux)
/// to a child that has not yet started DRAINING stdin blocks `write_all` on the
/// full pipe forever: `wait_timeout` is never reached, so the hard-kill
/// guarantee (`docs/design/beacon_subprocess_shim.md` §6) silently fails and the
/// call hangs for the child's whole lifetime. A WireDag artifact base64s to well
/// past this for non-trivial programs, so this is a real path, not a corner.
///
/// 32 KiB is comfortably under the typical 64 KiB pipe buffer, so a request at
/// or below it fits in the buffer and `write_all` returns without the child
/// having read a byte; anything larger uses the temp-file transport, which the
/// red team proved deadlock-safe at all sizes (the child reads a file, the
/// parent writes no pipe). The threshold is a safety floor, not a tuning knob:
/// it only decides stdin-vs-tempfile, never the soundness mapping.
const STDIN_REQUEST_MAX_BYTES: usize = 32 * 1024;

/// The environment variable the shim falls back to for the `chelis-beacon`
/// binary path when no explicit path is supplied at construction (Q2). There is
/// NO walk-up filesystem detection: an explicit path or this env var, or the
/// shim does not fit.
pub const BEACON_BIN_ENV: &str = "CHELIS_BEACON_BIN";

/// Which Beacon oracle lane the subprocess shim asks the out-of-tree binary to
/// use. The default intentionally preserves the #439 request contract:
/// `oracle: null` means "Beacon's verified default" (currently Arb box). The
/// zonotope mode is an explicit selector and must only be enabled by a caller
/// that has checked the binary's protocol capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BeaconOracleMode {
    /// Emit `oracle: null`.
    #[default]
    VerifiedDefaultArbBox,
    /// Emit `oracle: "zonotope_verified"`.
    VerifiedZonotope,
}

impl BeaconOracleMode {
    fn request_value(self) -> serde_json::Value {
        match self {
            Self::VerifiedDefaultArbBox => serde_json::Value::Null,
            Self::VerifiedZonotope => serde_json::json!("zonotope_verified"),
        }
    }
}

/// A content-addressed store mapping an exact-version `WireDag` v6 artifact's content hash
/// (lowercase-hex sha256, the same key the WI-3 producer computes) to the EXACT
/// serialized bytes.
///
/// Dispatch-site-owned: the caller that runs the WI-3 graph-extraction producer
/// populates this from [`crate::graph_extract::ExtractedGoal`] at extraction
/// time and passes it into [`BeaconShim::new`]. It is `Arc`-shared internally so
/// the shim closes over a cheap clone. This is the bridge that lets the shim get
/// the bytes WITHOUT changing the frozen [`IrHandle`] (which carries only the
/// hash) — see the module docs.
#[derive(Debug, Clone, Default)]
pub struct WireDagByteStore {
    inner: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl WireDagByteStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert the exact serialized `WireDag` v6 bytes under their content hash.
    /// The key MUST be the lowercase-hex sha256 of `bytes` (the producer's
    /// `ExtractedGoal::dag_hash`); the shim recomputes and re-checks the digest
    /// before trusting any report, so a mis-keyed insert fails closed rather
    /// than transporting the wrong artifact.
    pub fn insert(&self, dag_hash: impl Into<String>, bytes: Vec<u8>) {
        self.inner
            .lock()
            .expect("WireDagByteStore mutex is never poisoned (no panic under the lock)")
            .insert(dag_hash.into(), bytes);
    }

    /// Populate directly from an [`crate::graph_extract::ExtractedGoal`],
    /// keying its `wire_dag_bytes` by its `dag_hash`. This is the dispatch-site
    /// convenience the producer path uses; it keeps the key/bytes pairing the
    /// producer already computed, so the recompute guard never trips on a
    /// correctly-populated store.
    pub fn insert_extracted(&self, extracted: &crate::graph_extract::ExtractedGoal) {
        self.insert(extracted.dag_hash.clone(), extracted.wire_dag_bytes.clone());
    }

    /// The bytes for `dag_hash`, if present. Returns an owned copy so the lock
    /// is not held across the subprocess call.
    fn get(&self, dag_hash: &str) -> Option<Vec<u8>> {
        self.inner
            .lock()
            .expect("WireDagByteStore mutex is never poisoned (no panic under the lock)")
            .get(dag_hash)
            .cloned()
    }
}

/// Whether the shim sends the request on the child's stdin (`--request -`) or
/// via a temp file (`--request <path>`). Stdin is the default; the temp-file arm
/// exists so a binary that cannot read stdin (or a very large request) still has
/// a path, and so the temp-file invocation is itself testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestTransport {
    /// `chelis-beacon dispatch --request -`, request on stdin.
    #[default]
    Stdin,
    /// `chelis-beacon dispatch --request <tempfile>`, request in a temp file.
    TempFile,
}

/// The in-tree transport-only [`DischargeEngine`] for [`GoalShape::BoxRange`]
/// goals. Shells to the pinned `chelis-beacon` binary; maps its `CheckReport`
/// to a [`Discharge`] fail-closed. See the module docs and
/// `docs/design/beacon_subprocess_shim.md`.
#[derive(Debug, Clone)]
pub struct BeaconShim {
    /// The resolved `chelis-beacon` binary path.
    binary: PathBuf,
    /// The content-addressed byte store the dispatch site populated.
    store: WireDagByteStore,
    /// How the request reaches the child (stdin by default).
    transport: RequestTransport,
    /// Which verified Beacon lane to request from the subprocess.
    oracle_mode: BeaconOracleMode,
}

impl BeaconShim {
    /// Construct a shim with an EXPLICIT binary path (Q2, the primary discovery
    /// route). The store is the dispatch-site-owned [`WireDagByteStore`] keyed by
    /// `dag_hash`.
    pub fn new(binary: impl Into<PathBuf>, store: WireDagByteStore) -> Self {
        Self {
            binary: binary.into(),
            store,
            transport: RequestTransport::default(),
            oracle_mode: BeaconOracleMode::default(),
        }
    }

    /// Construct a shim, discovering the binary path (Q2): an explicit path is
    /// not given here, so the `CHELIS_BEACON_BIN` env var is the only source. If
    /// it is unset, returns `None` — the shim is NOT registered and a `BoxRange`
    /// goal takes the existing no-fit path (`Unsupported`), never a crash.
    ///
    /// Use [`BeaconShim::new`] when the dispatch site has an explicit path; use
    /// this when discovery is entirely env-driven.
    pub fn from_env(store: WireDagByteStore) -> Option<Self> {
        let path = std::env::var_os(BEACON_BIN_ENV)?;
        if path.is_empty() {
            return None;
        }
        Some(Self::new(path, store))
    }

    /// Set the request transport (stdin vs temp file). Stdin is the default for
    /// small requests; see [`Self::effective_transport`] for the large-request
    /// auto-fallback.
    pub fn with_transport(mut self, transport: RequestTransport) -> Self {
        self.transport = transport;
        self
    }

    /// Select the Beacon oracle lane requested from the subprocess. This only
    /// changes the request JSON; the report-to-discharge mapping remains the
    /// same fail-closed #439 CheckReport mapping.
    pub fn with_oracle_mode(mut self, oracle_mode: BeaconOracleMode) -> Self {
        self.oracle_mode = oracle_mode;
        self
    }

    /// The transport actually used for a request of `request_len` bytes.
    ///
    /// Deadlock-safety floor (the red-team HIGH): a [`RequestTransport::Stdin`]
    /// request larger than [`STDIN_REQUEST_MAX_BYTES`] would block `write_all`
    /// on a full pipe against a non-draining child and never reach the
    /// hard-kill, so it auto-falls-back to the proven-deadlock-safe
    /// [`RequestTransport::TempFile`]. A small Stdin request stays on stdin; an
    /// explicitly-`TempFile` shim always uses the temp file. This is a transport
    /// decision ONLY; it never touches the soundness mapping.
    fn effective_transport(&self, request_len: usize) -> RequestTransport {
        match self.transport {
            RequestTransport::Stdin if request_len > STDIN_REQUEST_MAX_BYTES => {
                RequestTransport::TempFile
            }
            other => other,
        }
    }

    /// Build the request JSON for a box/range goal. Q5/Q6: `wire_dag_v6_base64`
    /// is base64 of the EXACT `bytes` (no parse/reformat between the WI-3 bytes
    /// and the base64), and `expected_dag_sha256` is the handle's `dag_hash`.
    fn build_request(
        dag_hash: &str,
        root_index: u64,
        inputs: &IntervalBox,
        output: &OutputRange,
        bytes: &[u8],
        oracle: BeaconOracleMode,
    ) -> serde_json::Value {
        let input_dims: Vec<serde_json::Value> = inputs
            .dims
            .iter()
            .map(|(name, lo, hi)| serde_json::json!({ "name": name, "lo": lo, "hi": hi }))
            .collect();
        serde_json::json!({
            "schema_version": REQUEST_SCHEMA_VERSION,
            "wire_dag_v6_base64": BASE64.encode(bytes),
            "expected_dag_sha256": dag_hash,
            "root_index": root_index,
            "inputs": input_dims,
            "output": {
                "output": output.output,
                "lo": output.lo,
                "hi": output.hi,
            },
            "oracle": oracle.request_value(),
            "split_max_depth": serde_json::Value::Null,
            "split_max_boxes": serde_json::Value::Null,
        })
    }

    /// Spawn `chelis-beacon`, feed it `request_bytes`, enforce `timeout_ms` as a
    /// HARD kill, and return the raw subprocess outcome. Any spawn/wait/IO error
    /// or a timeout is surfaced as a fail-closed [`SubprocessOutcome::Failed`];
    /// this function never panics on a misbehaving child.
    fn run_beacon(&self, request_bytes: &[u8], timeout_ms: u64) -> SubprocessOutcome {
        // The temp file (if any) must outlive the child, so it is bound here.
        let mut command = Command::new(&self.binary);
        command.arg("dispatch");

        // Auto-fall-back to the temp-file transport for a request too large to
        // write to a stdin pipe without risking a full-buffer deadlock (the
        // red-team HIGH). The decision is made ONCE here, off the request size.
        let transport = self.effective_transport(request_bytes.len());

        let _tempfile_guard = match transport {
            RequestTransport::Stdin => {
                command.arg("--request").arg("-");
                command.stdin(Stdio::piped());
                None
            }
            RequestTransport::TempFile => {
                let mut file = match tempfile::NamedTempFile::new() {
                    Ok(f) => f,
                    Err(err) => {
                        return SubprocessOutcome::Failed {
                            reason: format!("could not create request temp file: {err}"),
                            stderr: String::new(),
                        };
                    }
                };
                if let Err(err) = file.write_all(request_bytes) {
                    return SubprocessOutcome::Failed {
                        reason: format!("could not write request temp file: {err}"),
                        stderr: String::new(),
                    };
                }
                command.arg("--request").arg(file.path());
                command.stdin(Stdio::null());
                Some(file)
            }
        };
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(err) => {
                return SubprocessOutcome::Failed {
                    reason: format!("could not spawn `{}`: {err}", self.binary.display()),
                    stderr: String::new(),
                };
            }
        };

        // Feed the request on stdin, then drop the handle so the child sees EOF.
        // Only the (small-request) stdin transport writes a pipe here; the
        // large-request fallback already wrote the temp file above.
        if transport == RequestTransport::Stdin
            && let Some(mut stdin) = child.stdin.take()
        {
            let write_result = stdin.write_all(request_bytes);
            // Drop the handle explicitly so the child sees EOF before we wait.
            drop(stdin);
            if let Err(err) = write_result {
                // Best-effort kill; the child may have already exited.
                let _ = child.kill();
                let _ = child.wait();
                return SubprocessOutcome::Failed {
                    reason: format!("could not write request to beacon stdin: {err}"),
                    stderr: String::new(),
                };
            }
        }

        // HARD kill at the timeout (Q3): wait_timeout returns None on expiry.
        let timeout = Duration::from_millis(timeout_ms);
        match child.wait_timeout(timeout) {
            Ok(Some(status)) => {
                // Exited within the budget; collect its output.
                let output = match child.wait_with_output() {
                    Ok(output) => output,
                    Err(err) => {
                        return SubprocessOutcome::Failed {
                            reason: format!("could not collect beacon output: {err}"),
                            stderr: String::new(),
                        };
                    }
                };
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                if status.success() {
                    SubprocessOutcome::Exited { stdout, stderr }
                } else {
                    SubprocessOutcome::Failed {
                        reason: format!(
                            "beacon exited with nonzero status {}",
                            status
                                .code()
                                .map_or_else(|| "signal".to_string(), |c| c.to_string())
                        ),
                        stderr,
                    }
                }
            }
            Ok(None) => {
                // Timed out: hard-kill, reap, and fail closed (never a pass).
                let _ = child.kill();
                let _ = child.wait();
                SubprocessOutcome::TimedOut
            }
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                SubprocessOutcome::Failed {
                    reason: format!("error waiting on beacon: {err}"),
                    stderr: String::new(),
                }
            }
        }
    }
}

/// The raw outcome of a `chelis-beacon` invocation, before mapping to a
/// [`Discharge`]. Kept separate so the subprocess mechanics and the soundness
/// mapping are independently testable.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SubprocessOutcome {
    /// The binary exited successfully; carries its stdout (the `CheckReport`
    /// JSON) and stderr.
    Exited { stdout: String, stderr: String },
    /// The binary was hard-killed at the timeout. Fail-closed.
    TimedOut,
    /// Spawn failure, nonzero exit, or an IO error. Fail-closed; carries a
    /// reason and any captured stderr for the discharge evidence.
    Failed { reason: String, stderr: String },
}

/// The verdict Beacon reports for a goal. Mirrors the §5 mapping rows.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReportVerdict {
    /// Oracle-verified proof: the output range holds (sound over-approximation).
    Proved,
    /// A proof whose soundness oracle did NOT verify. The integrity guard: it is
    /// `Untrusted`, NOT laundered into any proof qualifier.
    ProvedOracleUnverified,
    /// The output range does not hold; a counterexample is supplied. Its
    /// soundness depends on `oracle_verified` (mirrors the proved guard).
    Refuted,
}

/// The `CheckReport` JSON Beacon emits on stdout. Unknown fields are ignored so
/// Beacon can add fields without breaking the shim; the fields the soundness
/// mapping keys off are pinned here.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
struct CheckReport {
    /// The verdict (proved / proved_oracle_unverified / refuted).
    verdict: ReportVerdict,
    /// Whether Beacon's soundness oracle verified this result. SYMMETRIC across
    /// proofs and refutations: `proved` already implies verified, but a
    /// `refuted` verdict carries it explicitly so the shim can mirror the proof
    /// guard. ABSENT (`None`) defaults to UNVERIFIED — conservative.
    #[serde(default)]
    oracle_verified: Option<bool>,
    /// The counterexample model for a `refuted` verdict (any JSON shape).
    #[serde(default)]
    counterexample: serde_json::Value,
    /// Beacon-side evidence from the real binary. The shim preserves this under
    /// `beacon_evidence` on successful mappings so live e2e tests can assert
    /// the exact bytes and root the binary consumed, not only what Chelis sent.
    #[serde(default)]
    evidence: serde_json::Value,
}

impl DischargeEngine for BeaconShim {
    fn name(&self) -> &'static str {
        "beacon"
    }

    fn fitness(&self, goal: &Goal) -> bool {
        // Beacon's native form is the box/range goal; the shim claims exactly
        // that shape and nothing else.
        matches!(goal.shape, GoalShape::BoxRange { .. })
    }

    fn discharge(&self, goal: &Goal, timeout_ms: u64) -> Discharge {
        let GoalShape::BoxRange { inputs, output } = &goal.shape else {
            // fitness() gates this; a non-fitting goal handed here anyway must
            // not fabricate a proof.
            return untrusted_error(
                "beacon shim cannot discharge a non-BoxRange goal shape",
                serde_json::json!({ "engine": "beacon", "error": "wrong_goal_shape" }),
            );
        };

        // The handle must address a serialized artifact (dag_hash + root_index).
        let (Some(dag_hash), Some(root_index)) = (goal.ir.dag_hash(), goal.ir.root_index()) else {
            return untrusted_error(
                "beacon shim requires a populated IrHandle (dag_hash + root_index)",
                serde_json::json!({ "engine": "beacon", "error": "unpopulated_ir_handle" }),
            );
        };

        // Fetch the EXACT bytes the producer hashed. A store miss is fail-closed
        // (distinct from a missing binary, which is no-fit via non-registration).
        let Some(bytes) = self.store.get(dag_hash) else {
            return untrusted_error(
                "beacon shim: no WireDag bytes in the store for this goal's dag_hash",
                serde_json::json!({
                    "engine": "beacon",
                    "error": "byte_store_miss",
                    "dag_hash": dag_hash,
                }),
            );
        };

        // Defense-in-depth: recompute sha256 over the bytes we are about to send
        // and assert it equals the handle's dag_hash. This round-trips by
        // construction (the store key IS the producer's hash); a mismatch means
        // a corrupt / mis-keyed store and fails closed.
        let recomputed = sha256_hex(&bytes);
        if recomputed != dag_hash {
            return untrusted_error(
                "beacon shim: wire_dag hash mismatch (store bytes do not hash to the handle's dag_hash)",
                serde_json::json!({
                    "engine": "beacon",
                    "error": "wire_dag_hash_mismatch",
                    "expected_dag_sha256": dag_hash,
                    "actual_dag_sha256": recomputed,
                }),
            );
        }

        let request = Self::build_request(
            dag_hash,
            root_index,
            inputs,
            output,
            &bytes,
            self.oracle_mode,
        );
        let request_bytes = serde_json::to_vec(&request)
            .expect("a serde_json::Value built from owned data serializes");

        match self.run_beacon(&request_bytes, timeout_ms) {
            SubprocessOutcome::Exited { stdout, stderr } => map_report(&stdout, &stderr),
            SubprocessOutcome::TimedOut => untrusted_error(
                "beacon timeout",
                serde_json::json!({ "engine": "beacon", "error": "timeout" }),
            ),
            SubprocessOutcome::Failed { reason, stderr } => untrusted_error(
                &reason,
                serde_json::json!({ "engine": "beacon", "error": reason, "stderr": stderr }),
            ),
        }
    }
}

/// Parse the `CheckReport` JSON and map it to a [`Discharge`] per §5. An
/// unparseable report is fail-closed (`Untrusted` + `Error`, stderr in
/// evidence); a parseable report maps by verdict and the `oracle_verified`
/// flag.
fn map_report(stdout: &str, stderr: &str) -> Discharge {
    let report: CheckReport = match serde_json::from_str(stdout) {
        Ok(report) => report,
        Err(err) => {
            return untrusted_error(
                "beacon report is not parseable as a CheckReport",
                serde_json::json!({
                    "engine": "beacon",
                    "error": "unparseable_report",
                    "parse_error": err.to_string(),
                    "stdout": stdout,
                    "stderr": stderr,
                }),
            );
        }
    };

    match report.verdict {
        // Oracle-verified proof: a sound over-approximation, never an exact
        // proof. SoundApproximate + SoundOverApproximation -> sound_approximate.
        //
        // Defense-in-depth (the red-team MED): the proof-side discriminator is
        // the `proved_oracle_unverified` TOKEN, but a `proved` verdict carrying
        // an explicit `oracle_verified: false` is SELF-CONTRADICTORY. Rather than
        // let the asymmetry stand (a regressed Beacon emitting proved+false would
        // read as SoundApproximate), fail closed to Untrusted, symmetric with the
        // refuted arm. An absent flag on a `proved` verdict is the normal case
        // (`proved` already means verified) and stays SoundApproximate.
        ReportVerdict::Proved if report.oracle_verified == Some(false) => untrusted_error(
            "beacon reported `proved` with oracle_verified=false (self-contradictory); failing closed",
            serde_json::json!({
                "engine": "beacon",
                "verdict": "proved",
                "oracle_verified": false,
                "error": "contradictory_proved_unverified",
            }),
        ),
        ReportVerdict::Proved => Discharge::new(
            Soundness::SoundApproximate,
            QualifierSet::from_iter_kinds([Qualifier::SoundOverApproximation]),
            TierBResult::Proved,
            serde_json::json!({
                "engine": "beacon",
                "verdict": "proved",
                "beacon_evidence": report.evidence,
            }),
        )
        .unwrap_or_else(internal_error_discharge),

        // Integrity guard: a proof whose oracle did not verify is Untrusted with
        // NO proof qualifier, so it cannot read as proven-modulo-anything.
        ReportVerdict::ProvedOracleUnverified => untrusted_error(
            "beacon reported proved_oracle_unverified: the soundness oracle did not verify the bound",
            serde_json::json!({ "engine": "beacon", "verdict": "proved_oracle_unverified" }),
        ),

        // The goal did not hold (Disproved either way). The SOUNDNESS mirrors the
        // proved guard on the same oracle_verified flag: a verified refutation is
        // SoundApproximate + SoundOverApproximation; an unverified or un-flagged
        // refutation defaults to Untrusted (conservative).
        ReportVerdict::Refuted => {
            let verified = report.oracle_verified.unwrap_or(false);
            if verified {
                Discharge::new(
                    Soundness::SoundApproximate,
                    QualifierSet::from_iter_kinds([Qualifier::SoundOverApproximation]),
                    TierBResult::Disproved(report.counterexample.clone()),
                    serde_json::json!({
                        "engine": "beacon",
                        "verdict": "refuted",
                        "oracle_verified": true,
                        "beacon_evidence": report.evidence,
                    }),
                )
                .unwrap_or_else(internal_error_discharge)
            } else {
                // Untrusted refutation: the result is still Disproved, but the
                // soundness is the bottom of the lattice and it carries NO
                // qualifier (a fuzz/empirical-style untrusted disproof). Built
                // through Discharge::new with an empty set, so it is always valid.
                Discharge::new(
                    Soundness::Untrusted,
                    QualifierSet::new(),
                    TierBResult::Disproved(report.counterexample.clone()),
                    serde_json::json!({
                        "engine": "beacon",
                        "verdict": "refuted",
                        "oracle_verified": false,
                        "beacon_evidence": report.evidence,
                    }),
                )
                .unwrap_or_else(internal_error_discharge)
            }
        }
    }
}

/// A fail-closed `Untrusted` + `TierBResult::Error` discharge with an empty
/// qualifier set. Always valid (an untrusted/empty discharge passes the
/// integrity invariant), so the `.expect` never fires.
fn untrusted_error(reason: &str, evidence: serde_json::Value) -> Discharge {
    Discharge::new(
        Soundness::Untrusted,
        QualifierSet::new(),
        TierBResult::Error(reason.to_string()),
        evidence,
    )
    .expect("untrusted discharge with an empty qualifier set is always valid")
}

/// The last-resort discharge if `Discharge::new` somehow rejects a mapping the
/// soundness map should permit. It never should (the proved/refuted rows pair
/// `SoundOverApproximation` with its `SoundApproximate` floor), but surfacing
/// the constructor error as a non-proof discharge keeps the seam total rather
/// than panicking.
fn internal_error_discharge(err: crate::discharge::DischargeError) -> Discharge {
    untrusted_error(
        "beacon shim internal error building discharge",
        serde_json::json!({ "engine": "beacon", "internal_error": err.to_string() }),
    )
}

/// Lowercase-hex sha256 of `bytes` (the same digest the WI-3 producer computes).
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(hex, "{byte:02x}").expect("writing to a String never fails");
    }
    hex
}
