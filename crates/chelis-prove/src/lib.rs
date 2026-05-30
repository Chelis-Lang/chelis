//! Three-tier property verification dispatcher.
//!
//! This crate implements the specification-consumption layer of the Chelis
//! trust stack. It takes `@property` declarations and dispatches them through:
//!
//! - **Tier A:** Type system validation (dimension, effect, linearity discharge)
//! - **Tier B:** SMT solving via cvc5 (nonlinear real arithmetic)
//! - **Tier C:** Randomized fuzz testing (existing `chelis prove` logic)
//!
//! See `docs/trust-stack-verification.md` for architectural framing.

pub mod artifact;
pub mod dispatch;
pub mod from_property_spec;
pub mod inlineability;
pub mod solver;
pub mod tier_a;
pub mod tier_b;
pub mod tier_c;

pub use artifact::{ProofArtifact, ProofStatus, ProofTier};
pub use dispatch::{DispatchOptions, dispatch_property};
pub use from_property_spec::{PropertySpecInput, to_dispatch_amenability, to_smt_property};
pub use inlineability::{Fuzzability, Inlineability, classify_fuzzability, classify_inlineability};
pub use tier_b::{SmtProperty, solve_property};
