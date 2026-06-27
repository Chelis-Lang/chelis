//! Named transformations for the proof orchestrator.
//!
//! Each transformation implements [`crate::transformation::Transformation`] and
//! must pass the soundness harness before shipping.

pub mod abstract_subterm;
pub mod goal_split;
