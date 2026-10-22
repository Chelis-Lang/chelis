//! Host sum-of-squares proof admission behind the `clarabel` feature.
//!
//! `chelis-certificate-core` owns exact polynomials, matrices, certificates,
//! and affine repair. Arithmetic verification alone grants no proof authority.
//! This host binds each certificate to the requested polynomial, endpoints,
//! and canonical Markov-Lukacs interval layout before proof admission.
//! A floating-point solver result alone is not a proof.
//!
//! - [`encode`] owns interval layouts and coefficient constraints.
//! - [`extract`] recognizes the supported polynomial interval goals.
//! - [`propose`] uses Clarabel and exact affine repair to propose certificates.
//! - [`engine`] checks goal binding, arithmetic, and canonical layout.
//!
//! An admitted certificate yields `CertificateBearing@Exact`.
//! A failed check yields `Unknown@Untrusted` with an empty qualifier set.
//! See `docs/design/clarabel_sos_engine.md` for the host design.

pub mod encode;
pub mod engine;
pub mod extract;
// The Clarabel float SDP proposer needs the `sdp` BLAS backend, which is wired
// only for the targets that have it (Linux: OpenBLAS, macOS: Accelerate). On
// any other target the exact core + the UnwiredProposer engine still build.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod propose;

pub use engine::{ClarabelSosEngine, SosProposer, UnwiredProposer};
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub use propose::ClarabelProposer;
