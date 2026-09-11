//! Same-compilation observations, not a certificate or numerical semantics.
//!
//! Projection happens at the actual emission boundary. Only successful artifact
//! construction can return the projection, paired with that exact artifact.

use crate::compiler::{CompiledExecutionArtifact, Result};
use crate::schema::{GeneralKind, stage_error};

/// A successful artifact and the caller's projection of its actual compilation.
/// Private fields prevent replacement while retaining this pairing. The
/// projection remains unchecked caller data, not a compiler-correctness proof.
#[derive(Debug)]
pub struct TracedCompilation<T> {
    artifact: CompiledExecutionArtifact,
    projection: T,
}

impl<T> TracedCompilation<T> {
    pub fn artifact(&self) -> &CompiledExecutionArtifact {
        &self.artifact
    }

    pub fn projection(&self) -> &T {
        &self.projection
    }

    /// Consume the pairing when the caller needs to own the separate products.
    pub fn into_parts(self) -> (CompiledExecutionArtifact, T) {
        (self.artifact, self.projection)
    }
}

/// Counts emission events independently of projection contents. Saturation at
/// two retains duplicate evidence without overflow or executing projection twice.
pub(crate) struct Capture<T> {
    count: u8,
    value: Option<T>,
}

impl<T> Capture<T> {
    pub(crate) fn new() -> Self {
        Self { count: 0, value: None }
    }

    pub(crate) fn observe(&mut self, project: impl FnOnce() -> T) {
        self.count = self.count.saturating_add(1).min(2);
        if self.count == 1 {
            self.value = Some(project());
        }
    }

    fn finish_pair<A>(self, artifact: Result<A>) -> Result<(A, T)> {
        // An earlier observation never masks a later compilation failure.
        let artifact = artifact?;
        if self.count != 1 {
            return Err(stage_error(
                "observation",
                "compilation trace requires exactly one selected emission",
                GeneralKind::CompileError,
            ));
        }
        let projection = self.value.ok_or_else(|| stage_error(
            "observation",
            "compilation trace is missing its selected projection",
            GeneralKind::CompileError,
        ))?;
        Ok((artifact, projection))
    }

    pub(crate) fn finish(
        self,
        artifact: Result<CompiledExecutionArtifact>,
    ) -> Result<TracedCompilation<T>> {
        self.finish_pair(artifact)
            .map(|(artifact, projection)| TracedCompilation { artifact, projection })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_and_duplicate_emissions_cannot_produce_a_pair() {
        assert_eq!(Capture::<()>::new().finish_pair(Ok(())).unwrap_err().stage, "observation");
        let mut duplicate = Capture::new();
        duplicate.observe(|| "first");
        duplicate.observe(|| panic!("a duplicate must not execute projection"));
        assert_eq!(duplicate.finish_pair(Ok(())).unwrap_err().stage, "observation");
    }

    #[test]
    fn compiler_failure_wins_over_every_observation_state() {
        for count in 0..=2 {
            let mut capture = Capture::new();
            for _ in 0..count {
                capture.observe(|| "apparently valid");
            }
            let failure: Result<()> = Err(stage_error("codegen", "later failure", GeneralKind::CompileError));
            let error = capture.finish_pair(failure).unwrap_err();
            assert_eq!(error.stage, "codegen");
            assert!(error.errors.iter().any(|d| d.message == "later failure"));
        }
    }

    #[test]
    fn one_projection_is_paired_with_the_actual_success() {
        let mut capture = Capture::new();
        capture.observe(|| "selected payload");
        assert_eq!(capture.finish_pair(Ok("actual artifact")).unwrap(), ("actual artifact", "selected payload"));
    }
}
