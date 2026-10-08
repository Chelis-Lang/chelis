//! Subprocess-based contract prover that calls the `chelis-beacon` binary's
//! `contract` command to discharge certified-envelope assumptions.

use std::path::PathBuf;
use std::time::Duration;

use chelis_types::unsupported::Unsupported;

use crate::beacon_shim::BEACON_BIN_ENV;
use crate::beacon_supervisor::{self, Input, Invocation, Outcome, ProcessOps, SystemOps};
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

    #[cfg(test)]
    pub(crate) fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
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
    /// with the contract id on stdin. Returns a certified-envelope proof only
    /// when Beacon completed and returned one. A subprocess failure is typed
    /// degradation, distinct from an ordinary completed "no proof" response.
    ///
    pub fn prove_contract(
        &self,
        contract_id: &str,
    ) -> Result<Option<AssumptionDischarge>, Unsupported> {
        self.prove_contract_with_ops(contract_id, &SystemOps)
    }

    pub(crate) fn prove_contract_with_ops(
        &self,
        contract_id: &str,
        ops: &impl ProcessOps,
    ) -> Result<Option<AssumptionDischarge>, Unsupported> {
        // The domain [-300, 300] is the committed erf envelope's full coverage.
        // For structural proofs (monotonicity, reflection), Beacon ignores the
        // domain entirely — the proof is universal. For the range contract, the
        // envelope covers [-300, 300] which maps to input |x| < 300*sqrt(2) ≈ 424.
        let request = serde_json::json!({
            "contract_id": contract_id,
            "domain": [{"name": "x", "lo": -300.0, "hi": 300.0}],
        });
        let request_bytes = serde_json::to_vec(&request).expect("owned JSON request serializes");
        let args = ["contract".into(), "--query".into(), "-".into()];
        let stdout = match beacon_supervisor::run_with_ops(
            Invocation {
                binary: &self.binary,
                args: &args,
                input: Input::Stdin(&request_bytes),
                timeout: Duration::from_millis(self.timeout_ms),
            },
            ops,
        ) {
            Outcome::Completed { status, stdout, .. } if status.success() => stdout,
            Outcome::Completed { status, .. } => {
                return Err(beacon_supervisor::abnormal_exit(status));
            }
            Outcome::TimedOut => return Err(beacon_supervisor::timeout_unsupported()),
            Outcome::Degraded(reason) => return Err(reason),
        };

        let response: serde_json::Value = serde_json::from_slice(&stdout)
            .map_err(|_| beacon_supervisor::protocol_unsupported("invalid JSON response"))?;
        let invalid = || beacon_supervisor::protocol_unsupported("invalid contract response");
        if response
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            != Some(1)
            || response
                .get("contract_id")
                .and_then(serde_json::Value::as_str)
                != Some(contract_id)
        {
            return Err(invalid());
        }
        let verdict = response
            .get("verdict")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(invalid)?;
        let guarantee_class = response
            .get("guarantee_class")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(invalid)?;
        let soundness = response
            .get("soundness")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(invalid)?;
        let expected_qualifiers: &[&str] = match (verdict, guarantee_class, soundness) {
            ("proved", "certified_envelope", "sound_approximate") => {
                &["sound_over_approximation", "special_function_certified"]
            }
            ("proved", "sound_over_approximation", "sound_approximate") => {
                &["sound_over_approximation"]
            }
            (
                "proved_oracle_unverified" | "unknown" | "unsupported" | "invalid",
                "untrusted",
                "untrusted",
            ) => &[],
            _ => return Err(invalid()),
        };
        let qualifiers = response
            .get("discharge_qualifiers")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(invalid)?;
        if qualifiers.len() != expected_qualifiers.len()
            || expected_qualifiers.iter().any(|expected| {
                qualifiers
                    .iter()
                    .filter(|value| value.as_str() == Some(expected))
                    .count()
                    != 1
            })
        {
            return Err(invalid());
        }
        if guarantee_class != "certified_envelope" {
            return Ok(None);
        }
        let mut evidence = response;
        evidence["status"] = serde_json::json!("proved");
        Ok(Some(AssumptionDischarge::new(
            DischargeMethod::CertifiedEnvelope,
            evidence,
        )))
    }
}
