//! WI-15 sum-of-squares certificate engine (behind the `clarabel` feature).
//!
//! Proves `p(x) >= 0` for `x in [a, b]` (univariate polynomial, bounded
//! interval) by producing an EXACT RATIONAL sum-of-squares certificate that an
//! independent checker re-verifies exactly, then discharging
//! [`crate::discharge::Qualifier::CertificateBearing`] at
//! [`crate::discharge::Soundness::Exact`] -- but ONLY when that exact
//! certificate verifies. A float SDP solution alone is never a proof.
//!
//! See `docs/design/clarabel_sos_engine.md` for the full design.
//!
//! Module layout:
//!
//! - [`poly`]: exact univariate polynomials over `BigRational` (the arithmetic
//!   the certificate identity is checked in).
//! - [`exact`]: the soundness-critical verifier -- exact PSD test (rational
//!   LDL^T) plus the certificate's polynomial-identity check. Independent of the
//!   float SDP proposer's precision: a bad Gram is rejected here regardless of
//!   how it was produced.
//! - [`encode`] + [`linalg`]: the Markov-Lukacs SoS-to-SDP layout, the exact
//!   coefficient-matching linear system, and the Peyrl-Parrilo exact rational
//!   projection. Pure `BigRational`, shared by the float proposer and the repair.
//! - [`propose`]: the Clarabel float SDP proposer (the only BLAS-linked step;
//!   present only on targets with the `sdp` backend wired). It PROPOSES a
//!   candidate; the engine re-verifies it exactly before trusting it.
//! - [`engine`]: the [`ClarabelSosEngine`] `DischargeEngine` -- the honesty gate
//!   that maps a verified candidate to `CertificateBearing@Exact` and anything
//!   else to an honest `Unknown@Untrusted`.

pub mod encode;
pub mod engine;
pub mod exact;
pub mod extract;
pub mod linalg;
pub mod poly;
// The Clarabel float SDP proposer needs the `sdp` BLAS backend, which is wired
// only for the targets that have it (Linux: OpenBLAS, macOS: Accelerate). On
// any other target the exact core + the UnwiredProposer engine still build.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod propose;

pub use engine::{ClarabelSosEngine, SosProposer, UnwiredProposer};
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub use propose::ClarabelProposer;
