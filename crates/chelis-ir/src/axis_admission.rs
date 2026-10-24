//! Host diagnostic stages for portable graph extent checks.

use chelis_dag_core::axis_sources::{self, AxisSource, AxisSourceError};
use chelis_dag_core::dag::{Dag, DagNode};
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};

/// Retain graph extent facts and attach the host stage and authority.
pub fn axis_source_rejection(error: AxisSourceError, stage: Stage) -> Unsupported {
    let authority = if error.sourceless {
        chelis_types::unimplemented_rejection!(1277, AxisSourceError::SOURCELESS_HINT)
    } else {
        chelis_types::unimplemented_rejection!(1277, AxisSourceError::MAPPING_HINT)
    };
    Unsupported::new(
        UnsupportedKind::Construct(error.what),
        error.context,
        stage,
        authority,
    )
}

/// Check every output axis with the diagnostic stage of the host lane.
pub fn check_axis_sources(dag: &Dag, stage: Stage) -> Result<(), Unsupported> {
    axis_sources::check_axis_sources(dag).map_err(|error| axis_source_rejection(error, stage))
}

/// Check one node's physical axis sources with its host diagnostic stage.
pub fn check_node_axis_sources(
    dag: &Dag,
    node: &DagNode,
    sources: &[AxisSource],
    stage: Stage,
) -> Result<(), Unsupported> {
    axis_sources::check_node_axis_sources(dag, node, sources)
        .map_err(|error| axis_source_rejection(error, stage))
}

/// Check rendered dimension names with the diagnostic stage of the host lane.
pub fn check_rendered_dim_origins(dag: &Dag, stage: Stage) -> Result<(), Unsupported> {
    axis_sources::check_rendered_dim_origins(dag)
        .map_err(|error| axis_source_rejection(error, stage))
}
