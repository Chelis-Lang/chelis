//! Checked-builtin inference-route ownership and fail-loud diagnostics.

use super::*;

pub(super) fn checked_inference_rule(name: Option<&str>) -> Option<builtins::BuiltinInferenceRule> {
    name.and_then(|name| {
        builtins::builtin_decl(name).and_then(|decl| match decl.inference {
            builtins::InferenceDisposition::Checked(rule) => Some(rule),
            builtins::InferenceDisposition::GenericAccepted { .. } => None,
        })
    })
}

pub(super) fn reject_unregistered_checked_route(
    name: Option<&str>,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    let name = name?;
    let decl = builtins::builtin_decl(name)?;
    let builtins::InferenceDisposition::Checked(rule) = decl.inference else {
        return None;
    };
    if builtins::has_registered_inference_route(name, rule) {
        return None;
    }
    Some(report(
        errors,
        CheckError::new(
            CheckErrorKind::Other,
            format!(
                "internal: builtin `{name}` declares checked inference `{rule:?}` but has no registered dispatcher route"
            ),
            vec![
                "Add the semantic inference route in the same change as the builtin declaration, or explicitly declare GenericAccepted with a reviewed reason"
                    .to_string(),
            ],
        ),
    ))
}

pub(super) fn unobserved_checked_route_diagnostic(
    name: &str,
    rule: builtins::BuiltinInferenceRule,
) -> CheckError {
    CheckError::new(
        CheckErrorKind::Other,
        format!(
            "internal: builtin `{name}` declares checked inference `{rule:?}` but its application reached no semantic dispatcher route"
        ),
        vec![
            "Restore the builtin's semantic inference arm, or explicitly declare GenericAccepted with a reviewed reason"
                .to_string(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unobserved_checked_route_is_a_loud_internal_error() {
        let error = unobserved_checked_route_diagnostic(
            "probe_unarmed_op",
            builtins::BuiltinInferenceRule::Specialized,
        );
        assert!(matches!(error.kind, CheckErrorKind::Other));
        assert_eq!(
            error.message,
            "internal: builtin `probe_unarmed_op` declares checked inference `Specialized` but its application reached no semantic dispatcher route"
        );
    }
}
