use crate::resources::{Operation, Resources};
use chelis_syntax_host::NativeStack;

pub(crate) fn with_operation<R>(run: impl FnOnce(&Operation<'_, NativeStack>) -> R) -> R {
    Resources::new(&NativeStack).run(run)
}
