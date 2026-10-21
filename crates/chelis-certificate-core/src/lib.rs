#![no_std]

//! Exact rational polynomial, matrix, certificate, and affine repair operations.
//!
//! This crate needs an allocator for arbitrary-precision values and collections.
//! It has no `std` feature. All dependency defaults are disabled.
//! Host feature unification can enable dependency `std` features in a host graph.
//! An isolated target build supplies separate portable compilation evidence.
//! Target compilation does not establish device execution or allocator suitability.
//!
//! [`exact::SosCertificate::verify`] checks symmetry, PSD, and an exact identity.
//! Success grants no proof authority. It does not validate the requested goal,
//! interval order, interval multipliers, strictness, or provenance.
//! The host owns these checks in `chelis_prove::clarabel_sos::engine`.
//! The host also owns solver layouts, proposal conversion, and evidence encoding.
//! See `spec/design/verification_stack_master_plan.md` for proof integration ownership.
//!
//! Public types move directly from the old host `poly`, `exact`, and `linalg`
//! modules. No host compatibility import remains.

extern crate alloc;

pub mod exact;
pub mod linalg;
pub mod poly;
