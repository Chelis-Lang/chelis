//! Subprocess-based contract prover that calls the `chelis-beacon` binary's
//! `contract` command to discharge certified-envelope assumptions.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use wait_timeout::ChildExt;

use crate::beacon_shim::BEACON_BIN_ENV;
use crate::composition::{AssumptionDischarge, DischargeMethod};

/// A subprocess-based contract prover that invokes the `chelis-beacon contract`
/// command and maps its JSON response to an [`AssumptionDischarge`].
#[derive(Debug, Clone)]
pub struct BeaconContractProver {
    binary: PathBuf,
    timeout_ms: u64,
}

impl BeaconContractProver {
    /// Default subprocess timeout in milliseconds.
    const DEFAULT_TIMEOUT_MS: u64 = 10_000;

    /// Construct a prover with an explicit binary path.
    pub fn new(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            timeout_ms: Self::DEFAULT_TIMEOUT_MS,
        }
    }

    /// Discover the binary via the `CHELIS_BEACON_BIN` environment variable.
    /// Returns `None` if the variable is unset or empty.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var_os(BEACON_BIN_ENV)?;
        if path.is_empty() {
            return None;
        }
        Some(Self::new(path))
    }

    /// Attempt to prove a contract by invoking `chelis-beacon contract --query -`
    /// with the contract id on stdin. Returns `Some(AssumptionDischarge)` if the
    /// response indicates a certified-envelope proof, `None` otherwise.
    ///
    /// Returns `None` (honest fallback to fuzz) on ANY failure: spawn error,
    /// timeout, non-zero exit, parse error, unexpected verdict. This is
    /// intentionally silent — a misconfigured `CHELIS_BEACON_BIN` degrades to
    /// fuzz rather than crashing the prover.
    ///
    /// # Pipe buffer assumption
    ///
    /// Stdout is read AFTER `wait_timeout` returns. This is safe because the
    /// Beacon contract response is always a small JSON object (~1KB), well under
    /// the OS pipe buffer (~64KB on Linux). If Beacon ever produces responses
    /// larger than the pipe buffer, the child would block and this would timeout.
    pub fn prove_contract(&self, contract_id: &str) -> Option<AssumptionDischarge> {
        // The domain [-300, 300] is the committed erf envelope's full coverage.
        // For structural proofs (monotonicity, reflection), Beacon ignores the
        // domain entirely — the proof is universal. For the range contract, the
        // envelope covers [-300, 300] which maps to input |x| < 300*sqrt(2) ≈ 424.
        let request = serde_json::json!({
            "contract_id": contract_id,
            "domain": [{"name": "x", "lo": -300.0, "hi": 300.0}],
        });
        let request_bytes = serde_json::to_vec(&request).ok()?;

        let mut child = Command::new(&self.binary)
            .args(["contract", "--query", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;

        // Write the request to stdin and explicitly close the handle so the
        // child sees EOF immediately.
        {
            let mut stdin = child.stdin.take()?;
            if stdin.write_all(&request_bytes).is_err() {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            drop(stdin); // Explicit EOF delivery
        }

        // Wait with timeout; hard-kill on expiry.
        let timeout = Duration::from_millis(self.timeout_ms);
        let status = match child.wait_timeout(timeout) {
            Ok(Some(status)) => status,
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        };

        if !status.success() {
            return None;
        }

        // Read stdout. The child has exited and the response is small (~1KB),
        // so the pipe buffer was never full and the data is available.
        let stdout = {
            use std::io::Read;
            let mut buf = Vec::new();
            if let Some(mut out) = child.stdout.take() {
                let _ = out.read_to_end(&mut buf);
            }
            buf
        };

        let response: serde_json::Value = serde_json::from_slice(&stdout).ok()?;

        let verdict = response.get("verdict")?.as_str()?;
        let guarantee_class = response.get("guarantee_class")?.as_str()?;

        if verdict == "proved" && guarantee_class == "certified_envelope" {
            let mut evidence = response.clone();
            evidence["status"] = serde_json::json!("proved");
            Some(AssumptionDischarge::new(
                DischargeMethod::CertifiedEnvelope,
                evidence,
            ))
        } else {
            None
        }
    }
}
