//! Shared final-authority classifier for every capacity-census family.

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SurfaceDescriptor {
    pub family: String,
    pub kind: String,
    pub id: String,
    pub flags: Vec<String>,
}

impl SurfaceDescriptor {
    pub fn new(family: &str, kind: &str, id: &str, flags: &[&str]) -> Self {
        Self {
            family: family.to_string(),
            kind: kind.to_string(),
            id: id.to_string(),
            flags: flags.iter().map(|flag| (*flag).to_string()).collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticSurfaceDescriptor {
    pub family: &'static str,
    pub kind: &'static str,
    pub id: &'static str,
    pub flags: &'static [&'static str],
}

impl StaticSurfaceDescriptor {
    pub const fn new(
        family: &'static str,
        kind: &'static str,
        id: &'static str,
        flags: &'static [&'static str],
    ) -> Self {
        Self {
            family,
            kind,
            id,
            flags,
        }
    }

    fn matches(self, surface: &SurfaceDescriptor) -> bool {
        surface.family == self.family
            && surface.kind == self.kind
            && surface.id == self.id
            && surface
                .flags
                .iter()
                .map(String::as_str)
                .eq(self.flags.iter().copied())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericOperationRegistration {
    pub surface: StaticSurfaceDescriptor,
    pub atom: &'static str,
    pub authority_anchor: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct AuthorityRegistries<'a> {
    pub nonnumeric: &'a [StaticSurfaceDescriptor],
    pub tagged_transports: &'a [StaticSurfaceDescriptor],
    pub numeric_operations: &'a [NumericOperationRegistration],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalAuthority<'a> {
    Nonnumeric,
    TaggedTransport,
    NumericOperation { atom: &'a str },
}

pub fn classify_final_authority<'a>(
    surface: &SurfaceDescriptor,
    registries: AuthorityRegistries<'a>,
    numbered_spec: &str,
) -> Result<FinalAuthority<'a>, String> {
    let mut matches = Vec::new();

    for registration in registries.nonnumeric {
        if registration.matches(surface) {
            if !surface.flags.is_empty() {
                return Err(format!(
                    "NONNUMERIC registration carries numeric flags {:?}: [{}:{}] {}",
                    surface.flags, surface.family, surface.kind, surface.id
                ));
            }
            matches.push(FinalAuthority::Nonnumeric);
        }
    }
    for registration in registries.tagged_transports {
        if registration.matches(surface) {
            matches.push(FinalAuthority::TaggedTransport);
        }
    }
    for registration in registries.numeric_operations {
        if registration.surface.matches(surface) {
            validate_numeric_registration(*registration, numbered_spec).map_err(|problem| {
                format!(
                    "INVALID NUMERIC OPERATION REGISTRATION for [{}:{}] {}: {problem}",
                    surface.family, surface.kind, surface.id
                )
            })?;
            matches.push(FinalAuthority::NumericOperation {
                atom: registration.atom,
            });
        }
    }

    match matches.as_slice() {
        [authority] => Ok(*authority),
        [] => Err(format!(
            "descriptor has zero final matches; every discovered row requires exactly one final authority class: [{}:{}] {} {:?}",
            surface.family, surface.kind, surface.id, surface.flags
        )),
        _ => Err(format!(
            "descriptor has multiple final authority classes ({matches:?}); every discovered row requires exactly one: [{}:{}] {} {:?}",
            surface.family, surface.kind, surface.id, surface.flags
        )),
    }
}

fn validate_numeric_registration(
    registration: NumericOperationRegistration,
    numbered_spec: &str,
) -> Result<(), String> {
    let Some(atom_body) = registration
        .atom
        .strip_prefix('[')
        .and_then(|atom| atom.strip_suffix(']'))
    else {
        return Err(format!(
            "malformed atom `{}` (expected `[05-OP-N]`)",
            registration.atom
        ));
    };
    let parts: Vec<&str> = atom_body.split('-').collect();
    if parts.len() != 3
        || parts[0] != "05"
        || parts[1] != "OP"
        || parts[2].is_empty()
        || !parts[2].chars().all(|character| character.is_ascii_digit())
    {
        return Err(format!(
            "wrong atom grammar/group `{}` (expected `[05-OP-N]`)",
            registration.atom
        ));
    }
    if registration.authority_anchor.trim().is_empty() {
        return Err("authority anchor is empty".to_string());
    }

    let definition_prefix = format!("> **{}**", registration.atom);
    let mut lines = numbered_spec.lines();
    let Some(definition_line) =
        lines.find(|line| line.trim_start().starts_with(&definition_prefix))
    else {
        return Err(format!(
            "atom `{}` does not exist as a normative `> **[05-OP-N]**` definition",
            registration.atom
        ));
    };
    let mut definition = definition_line.trim_start().to_string();
    for line in lines {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('>') {
            break;
        }
        if trimmed.starts_with("> **[") {
            break;
        }
        definition.push('\n');
        definition.push_str(trimmed);
    }
    if !definition.contains(registration.authority_anchor) {
        return Err(format!(
            "atom `{}` is unrelated: its normative definition does not contain authority anchor `{}`",
            registration.atom, registration.authority_anchor
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAMILY: &str = "synthetic";
    const PLAIN: StaticSurfaceDescriptor =
        StaticSurfaceDescriptor::new(FAMILY, "callable", "plain", &[]);
    const TAGGED: StaticSurfaceDescriptor =
        StaticSurfaceDescriptor::new(FAMILY, "transport", "tagged", &["tagged-transport"]);
    const NUMERIC: StaticSurfaceDescriptor =
        StaticSurfaceDescriptor::new(FAMILY, "callable", "numeric", &["numeric-op"]);
    const NUMERIC_REGISTRATION: NumericOperationRegistration = NumericOperationRegistration {
        surface: NUMERIC,
        atom: "[05-OP-7]",
        authority_anchor: "numeric(signature)",
    };
    const SPEC: &str = "> **[05-OP-7]** `numeric(signature)` is exact.\n> More semantics.\n\n> **[05-OP-8]** `other()` is unrelated.\n";

    fn registries<'a>(
        nonnumeric: &'a [StaticSurfaceDescriptor],
        tagged: &'a [StaticSurfaceDescriptor],
        numeric: &'a [NumericOperationRegistration],
    ) -> AuthorityRegistries<'a> {
        AuthorityRegistries {
            nonnumeric,
            tagged_transports: tagged,
            numeric_operations: numeric,
        }
    }

    #[test]
    fn each_exact_final_class_is_recognized() {
        assert_eq!(
            classify_final_authority(
                &SurfaceDescriptor::new(FAMILY, "callable", "plain", &[]),
                registries(&[PLAIN], &[], &[]),
                SPEC,
            ),
            Ok(FinalAuthority::Nonnumeric)
        );
        assert_eq!(
            classify_final_authority(
                &SurfaceDescriptor::new(FAMILY, "transport", "tagged", &["tagged-transport"],),
                registries(&[], &[TAGGED], &[]),
                SPEC,
            ),
            Ok(FinalAuthority::TaggedTransport)
        );
        assert_eq!(
            classify_final_authority(
                &SurfaceDescriptor::new(FAMILY, "callable", "numeric", &["numeric-op"]),
                registries(&[], &[], &[NUMERIC_REGISTRATION]),
                SPEC,
            ),
            Ok(FinalAuthority::NumericOperation { atom: "[05-OP-7]" })
        );
    }

    #[test]
    fn zero_or_multiple_final_classes_fail() {
        let surface = SurfaceDescriptor::new(FAMILY, "callable", "plain", &[]);
        let zero = classify_final_authority(&surface, registries(&[], &[], &[]), SPEC)
            .expect_err("an unregistered descriptor has no authority");
        assert!(zero.contains("exactly one final authority class"), "{zero}");

        let multiple =
            classify_final_authority(&surface, registries(&[PLAIN], &[PLAIN], &[]), SPEC)
                .expect_err("one descriptor cannot carry two final classes");
        assert!(
            multiple.contains("multiple final authority classes"),
            "{multiple}"
        );
    }

    #[test]
    fn nonnumeric_registration_cannot_hide_numeric_flags() {
        let numeric_plain =
            StaticSurfaceDescriptor::new(FAMILY, "callable", "numeric-plain", &["numeric-op"]);
        let surface = SurfaceDescriptor::new(FAMILY, "callable", "numeric-plain", &["numeric-op"]);
        let error =
            classify_final_authority(&surface, registries(&[numeric_plain], &[], &[]), SPEC)
                .expect_err("nonnumeric authority requires structurally nonnumeric flags");
        assert!(
            error.contains("NONNUMERIC registration carries numeric flags"),
            "{error}"
        );
    }

    #[test]
    fn tagged_transport_registration_is_exact_in_every_descriptor_field() {
        for surface in [
            SurfaceDescriptor::new("other", "transport", "tagged", &["tagged-transport"]),
            SurfaceDescriptor::new(FAMILY, "other", "tagged", &["tagged-transport"]),
            SurfaceDescriptor::new(FAMILY, "transport", "successor", &["tagged-transport"]),
            SurfaceDescriptor::new(FAMILY, "transport", "tagged", &[]),
        ] {
            let error = classify_final_authority(&surface, registries(&[], &[TAGGED], &[]), SPEC)
                .expect_err("tagged transport resemblance is not authority");
            assert!(
                error.contains("exactly one final authority class"),
                "{error}"
            );
        }
    }

    #[test]
    fn numeric_registration_requires_exact_normative_atom_and_anchor() {
        for registration in [
            NumericOperationRegistration {
                atom: "05-OP-7",
                ..NUMERIC_REGISTRATION
            },
            NumericOperationRegistration {
                atom: "[05-OBS-1]",
                ..NUMERIC_REGISTRATION
            },
            NumericOperationRegistration {
                atom: "[05-OP-999]",
                ..NUMERIC_REGISTRATION
            },
            NumericOperationRegistration {
                atom: "[05-OP-8]",
                ..NUMERIC_REGISTRATION
            },
        ] {
            let surface = SurfaceDescriptor::new(FAMILY, "callable", "numeric", &["numeric-op"]);
            assert!(
                classify_final_authority(&surface, registries(&[], &[], &[registration]), SPEC,)
                    .is_err(),
                "invalid or unrelated registration passed: {registration:?}"
            );
        }
    }

    #[test]
    fn duplicate_numeric_registration_is_not_one_authority() {
        let surface = SurfaceDescriptor::new(FAMILY, "callable", "numeric", &["numeric-op"]);
        let error = classify_final_authority(
            &surface,
            registries(&[], &[], &[NUMERIC_REGISTRATION, NUMERIC_REGISTRATION]),
            SPEC,
        )
        .expect_err("a duplicated mapping is ambiguous");
        assert!(
            error.contains("multiple final authority classes"),
            "{error}"
        );
    }
}
