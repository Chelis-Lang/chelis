//! Shared producer/consumer report admission under [04-FIT-18].

use super::numbers::{NonnegativeCount, UnitInterval};
use super::{CheckResult, Diagnostic, FitnessComponents, WireCheckResult, WireDiagnostic};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ReportWire<D> {
    score: UnitInterval,
    components: FitnessComponents,
    typed_nodes: NonnegativeCount,
    untyped_nodes: NonnegativeCount,
    total_nodes: NonnegativeCount,
    unresolved_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inferred_signatures: Option<Vec<super::WireInferredSignature>>,
    errors: Vec<D>,
}

fn validate_counters(
    typed: NonnegativeCount,
    untyped: NonnegativeCount,
    total: NonnegativeCount,
) -> Result<(), String> {
    if typed.get() > total.get() || untyped.get() != total.get() - typed.get() {
        return Err("inconsistent report counts: typed_nodes must not exceed total_nodes and untyped_nodes must equal total_nodes - typed_nodes".to_string());
    }
    Ok(())
}

impl CheckResult {
    pub fn validate(&self) -> Result<(), String> {
        validate_counters(self.typed_nodes, self.untyped_nodes, self.total_nodes)
    }

    pub fn try_from_fitness(report: &chelis_types::FitnessReport) -> Result<Self, String> {
        let result = Self {
            score: UnitInterval::new(report.score)?,
            components: FitnessComponents {
                parse: UnitInterval::new(report.components.parse)?,
                structure: UnitInterval::new(report.components.structure)?,
                names: UnitInterval::new(report.components.names)?,
                types: UnitInterval::new(report.components.types)?,
            },
            typed_nodes: report.typed_nodes.try_into()?,
            untyped_nodes: report.untyped_nodes.try_into()?,
            total_nodes: report.total_nodes.try_into()?,
            unresolved_names: report.unresolved_names.clone(),
            inferred_signatures: None,
            errors: report
                .errors
                .iter()
                .map(Diagnostic::try_from_check_error)
                .collect::<Result<_, _>>()?,
        };
        result.validate()?;
        Ok(result)
    }
}

impl Serialize for CheckResult {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        ReportWire {
            score: self.score,
            components: self.components.clone(),
            typed_nodes: self.typed_nodes,
            untyped_nodes: self.untyped_nodes,
            total_nodes: self.total_nodes,
            unresolved_names: self.unresolved_names.clone(),
            errors: self.errors.clone(),
            inferred_signatures: self.inferred_signatures.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WireCheckResult {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ReportWire::<WireDiagnostic>::deserialize(deserializer)?;
        validate_counters(wire.typed_nodes, wire.untyped_nodes, wire.total_nodes)
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            score: wire.score,
            components: wire.components,
            typed_nodes: wire.typed_nodes,
            untyped_nodes: wire.untyped_nodes,
            total_nodes: wire.total_nodes,
            unresolved_names: wire.unresolved_names,
            errors: wire.errors,
            inferred_signatures: wire.inferred_signatures,
        })
    }
}
