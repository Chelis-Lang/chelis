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
    pub fn prove_contract(&self, contract_id: &str) -> Option<AssumptionDischarge> {
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

        // Write the request to stdin and close the handle so the child sees EOF.
        if let Some(mut stdin) = child.stdin.take() {
            if stdin.write_all(&request_bytes).is_err() {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
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

        // Read stdout. The child has exited, so reading is safe.
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
