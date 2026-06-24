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
//!
//! The float Markov-Lukacs SDP proposer (Clarabel) and the Peyrl-Parrilo
//! rational repair land on top of this core once the BLAS-linkage decision for
//! Clarabel's PSD cone is pinned (escalated to the orchestrator). The exact core
//! is fully testable without any solver.

pub mod encode;
pub mod engine;
pub mod exact;
pub mod extract;
pub mod linalg;
pub mod poly;

pub use engine::{ClarabelSosEngine, SosProposer, UnwiredProposer};
