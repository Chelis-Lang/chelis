//! Operation-local stack limits and the first-failure gate.
use super::*;
use crate::resources::Operation;

pub(super) const STACK_RED_ZONE_BYTES: usize = 128 * 1024;
pub(super) const FALLBACK_MAX_DEPTH: usize = 20;

pub(super) struct RecursionDepthGuard<'a>(&'a Cell<usize>);
impl<'a> RecursionDepthGuard<'a> {
    pub(super) fn enter(operation: &'a Operation<'_>) -> Self {
        operation.depth.set(operation.depth.get() + 1);
        Self(&operation.depth)
    }
}
impl Drop for RecursionDepthGuard<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get().saturating_sub(1));
    }
}

pub(super) struct StackExhaustionScope<'a> {
    failure: &'a RefCell<Option<(String, Option<String>)>>,
    depth: &'a Cell<usize>,
}
impl<'a> StackExhaustionScope<'a> {
    pub(super) fn enter(operation: &'a Operation<'_>) -> Self {
        if operation.stack_scope_depth.get() == 0 {
            *operation.stack_failure.borrow_mut() = None;
        }
        operation
            .stack_scope_depth
            .set(operation.stack_scope_depth.get() + 1);
        Self {
            failure: &operation.stack_failure,
            depth: &operation.stack_scope_depth,
        }
    }
    pub(super) fn drain_into(&self, errors: &mut impl DiagnosticOutput) {
        if let Some((site, span)) = self.failure.borrow_mut().take() {
            errors.push(stack_depth_error(&site, span.as_deref()));
        }
    }
}
impl Drop for StackExhaustionScope<'_> {
    fn drop(&mut self) {
        self.depth.set(self.depth.get().saturating_sub(1));
    }
}

pub(super) fn stack_guard_tripped<'a>(
    site: &str,
    span_id: impl FnOnce() -> Option<&'a str>,
    operation: &Operation<'_>,
) -> bool {
    let exceeded = match (operation.remaining)() {
        Some(remaining) => remaining < STACK_RED_ZONE_BYTES,
        None => operation.depth.get() > FALLBACK_MAX_DEPTH,
    };
    if exceeded {
        let span = span_id();
        let mut failure = operation.stack_failure.borrow_mut();
        if failure.is_none() {
            *failure = Some((site.to_string(), span.map(str::to_string)));
        }
    }
    exceeded
}

macro_rules! stack_guard {
    ($operation:expr, $site:expr, $expr:expr, $bail:expr) => {
        let _stack_depth_guard = RecursionDepthGuard::enter($operation);
        if stack_guard_tripped($site, || $expr.span_id(), $operation) {
            return $bail;
        }
    };
    ($operation:expr, $site:expr, $expr:expr) => {
        let _stack_depth_guard = RecursionDepthGuard::enter($operation);
        if stack_guard_tripped($site, || $expr.span_id(), $operation) {
            return;
        }
    };
}

pub(crate) fn stack_depth_error(site: &str, span_id: Option<&str>) -> CheckError {
    let where_clause = match span_id {
        Some(id) if !id.is_empty() => format!(" near source span `{id}`"),
        _ => String::new(),
    };
    CheckError::new(
        CheckErrorKind::Other,
        format!(
            "type-checker stack budget exhausted while analyzing deeply-nested \
             input in `{site}`{where_clause}: the program nests deeper than the \
             checker can analyze on the available native stack. The recursion is \
             finite (this is a checker limitation, not necessarily a type error); \
             see WI-1 / spec/design/verification_stack_master_plan.md \u{00A7}4.1."
        ),
        vec![
            "Deeply-nested application chains (often from reef-linking many \
             exports into one program) can exhaust the recursive checker's stack \
             budget; reduce the nesting or split the program. An iterative / \
             stack-growing inference follow-up (stacker::maybe_grow) is tracked to \
             lift this for legitimately-deep programs."
                .to_string(),
        ],
    )
}
