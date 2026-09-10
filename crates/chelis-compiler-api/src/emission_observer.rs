//! Opt-in native views of actual code-generation inputs. Not a certificate.
//!
//! These borrowed views introduce no serialization format or numeric conversion.
//! An observation can precede a later compilation error; callers must also check
//! the enclosing compilation result. The callback cannot mutate selected payloads.

use chelis_ir::dag::Dag;
use chelis_ir::host::ConcreteHostProgram;
use chelis_ir::ownership::{VerifiedDagView, VerifiedHostEmission};
use chelis_types::manifest::ManifestedProgram;

/// The selected payload, after ownership verification and before code generation.
pub enum SelectedEmission<'a> {
    Dag {
        /// Selected entry/whole-program graph before specialization and fusion.
        /// This is not interchangeable with a library-level lowering snapshot.
        unfused: &'a Dag,
        selected: VerifiedDagView<'a>,
    },
    /// Nested tensor helpers are available through the existing verified cursors.
    /// There is no single standalone unfused graph for this host payload.
    Host(VerifiedHostEmission<'a>),
}

pub struct EmissionObservation<'a> {
    /// The actual checked source, manifest, and target for this compilation.
    /// Its presence is provenance, not a proof of source/graph correspondence.
    pub program: &'a ManifestedProgram,
    /// The actual initial host lowering, before entry projection and backend
    /// preparation. This may include functions absent from `selected` and from
    /// the emitted artifact (for example, a tuple gradient's scalar loss).
    /// It is not ownership-verified or a pre-AD graph. Consumers must check
    /// their own source/selected-payload correspondence; this is provenance,
    /// not permission to treat an unselected function as emitted code.
    pub lowered_host: Option<&'a ConcreteHostProgram>,
    pub selected: SelectedEmission<'a>,
}

pub(crate) type Observer<'a> = &'a mut dyn FnMut(EmissionObservation<'_>);

pub(crate) fn observe(
    observer: &mut Option<Observer<'_>>,
    program: &ManifestedProgram,
    lowered_host: Option<&ConcreteHostProgram>,
    selected: SelectedEmission<'_>,
) {
    if let Some(observer) = observer {
        observer(EmissionObservation {
            program,
            lowered_host,
            selected,
        });
    }
}
