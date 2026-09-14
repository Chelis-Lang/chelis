use bincode::Options;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub const SHELL_MAGIC: &[u8; 8] = b"CHELCHB\0";
pub const SHELL_FORMAT_VERSION: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellPackage {
    pub format_version: u32,
    pub package: PackageId,
    pub compiler: String,
    pub modules: Vec<ShellModule>,
    pub dependencies: Vec<PackageId>,
    pub archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageId {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellModule {
    pub module: String,
    pub exports: Vec<ShellSymbol>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellSymbol {
    pub name: String,
    pub kind: SymbolKind,
    /// Canonical serialized type for a value export. Values always carry
    /// `Some` (including signature-only exports with `has_body == false`);
    /// type, macro, and dimension exports may use `None`. An empty string is
    /// never a representation.
    pub type_repr: Option<String>,
    pub type_variable_restrictions: Vec<TypeVariableRestriction>,
    pub effects: Vec<String>,
    pub has_body: bool,
}

/// A canonical quantified-type-variable domain carried across public package
/// metadata. `variable` names the alpha-canonical identity in `type_repr`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TypeVariableRestriction {
    pub variable: String,
    pub domain: TypeVariableDomain,
}

/// The §5.9 primitive dtype bounds and §3.1 inferred operation-value
/// restrictions from `spec/04-type-system.md`, kept distinct in published
/// metadata. A new semantic domain requires a spec and shell format change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeVariableDomain {
    /// §5.9 `Float`.
    ActiveFloat,
    /// §5.9 `Int`.
    ActiveInt,
    /// §5.9 `Numeric`.
    ActiveNumeric,
    /// An inferred operation operand is a float scalar or tensor, not a dtype.
    FloatValue,
    /// An inferred operation operand is a signed integer scalar or tensor.
    IntValue,
    /// An inferred operation operand is a numeric scalar or tensor.
    NumericValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolKind {
    Value,
    Type,
    Macro,
    Dim,
}

pub fn encode_shell(shell: &ShellPackage) -> Result<Vec<u8>, bincode::Error> {
    validate_shell(shell)?;
    let payload = bincode::serialize(shell)?;
    let mut encoded = Vec::with_capacity(SHELL_MAGIC.len() + 4 + payload.len());
    encoded.extend_from_slice(SHELL_MAGIC);
    encoded.extend_from_slice(&SHELL_FORMAT_VERSION.to_le_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

pub fn decode_shell(bytes: &[u8]) -> Result<ShellPackage, bincode::Error> {
    if !bytes.starts_with(SHELL_MAGIC) {
        return Err(validation_error("unsupported predecessor shell format"));
    }
    let version_bytes: [u8; 4] = bytes
        .get(SHELL_MAGIC.len()..SHELL_MAGIC.len() + 4)
        .ok_or_else(|| validation_error("truncated shell format version"))?
        .try_into()
        .expect("the checked slice has exactly four bytes");
    let version = u32::from_le_bytes(version_bytes);
    if version != SHELL_FORMAT_VERSION {
        return Err(validation_error(&format!(
            "shell format version {version} is unsupported; expected {SHELL_FORMAT_VERSION}"
        )));
    }
    let payload = &bytes[SHELL_MAGIC.len() + 4..];
    let shell: ShellPackage = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(payload)?;
    validate_shell(&shell)?;
    let canonical = encode_shell(&shell)?;
    if canonical != bytes {
        return Err(validation_error(
            "shell bytes are not the canonical encoding of the decoded payload",
        ));
    }
    Ok(shell)
}

pub fn write_shell(path: &Path, shell: &ShellPackage) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = encode_shell(shell)?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn read_shell(path: &Path) -> Result<ShellPackage, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(decode_shell(&bytes)?)
}

/// Validate invariants carried by the complete CHB payload.
///
/// A CHB is read at a package-install trust boundary, so fields that are
/// not needed to locate the paired archive still need validation. This
/// rejects malformed metadata, ambiguous duplicate entries, and ordering
/// that cannot have been emitted by [`encode_shell`] from Reef's canonical
/// shell builder.
pub fn validate_shell(shell: &ShellPackage) -> Result<(), bincode::Error> {
    if shell.format_version != SHELL_FORMAT_VERSION {
        return Err(validation_error(&format!(
            "payload format version {} is unsupported; expected {SHELL_FORMAT_VERSION}",
            shell.format_version
        )));
    }
    validate_nonempty_trimmed("package.name", &shell.package.name)?;
    validate_nonempty_trimmed("package.version", &shell.package.version)?;
    validate_compiler_pin(&shell.compiler)?;
    if !is_lowercase_hex_sha256(&shell.archive_sha256) {
        return Err(validation_error(
            "archive_sha256 must be exactly 64 lowercase hexadecimal characters",
        ));
    }

    validate_strict_order(
        "modules",
        shell.modules.iter().map(|module| module.module.as_str()),
    )?;
    for module in &shell.modules {
        validate_nonempty_trimmed("module name", &module.module)?;
        validate_strict_order(
            &format!("exports for module `{}`", module.module),
            module.exports.iter().map(|symbol| symbol.name.as_str()),
        )?;
        for symbol in &module.exports {
            validate_nonempty_trimmed("export name", &symbol.name)?;
            let type_variables = match &symbol.type_repr {
                Some(type_repr) => Some(canonical_type_variables(type_repr)?),
                None if symbol.kind == SymbolKind::Value => {
                    return Err(validation_error(&format!(
                        "value export `{}` in module `{}` requires a type representation",
                        symbol.name, module.module
                    )));
                }
                None => None,
            };
            validate_type_variable_restrictions(
                &format!(
                    "type-variable restrictions for export `{}` in module `{}`",
                    symbol.name, module.module
                ),
                &symbol.type_variable_restrictions,
            )?;
            if !symbol.type_variable_restrictions.is_empty() && symbol.type_repr.is_none() {
                return Err(validation_error(&format!(
                    "export `{}` in module `{}` carries type-variable restrictions without a type representation",
                    symbol.name, module.module
                )));
            }
            if let Some(type_variables) = type_variables {
                for restriction in &symbol.type_variable_restrictions {
                    let index = canonical_type_variable_index(&restriction.variable)?;
                    if !type_variables.contains(&index) {
                        return Err(validation_error(&format!(
                            "export `{}` in module `{}` restricts `{}` but that canonical variable is absent from its type representation",
                            symbol.name, module.module, restriction.variable
                        )));
                    }
                }
            }
            for effect in &symbol.effects {
                validate_nonempty_trimmed("effect name", effect)?;
            }
        }
    }

    validate_strict_order(
        "dependencies",
        shell
            .dependencies
            .iter()
            .map(|dependency| dependency.name.as_str()),
    )?;
    for dependency in &shell.dependencies {
        validate_nonempty_trimmed("dependency name", &dependency.name)?;
        if !dependency.version.is_empty() && dependency.version.trim() != dependency.version {
            return Err(validation_error(&format!(
                "dependency `{}` version must not have leading or trailing whitespace",
                dependency.name
            )));
        }
    }
    Ok(())
}

fn validate_nonempty_trimmed(label: &str, value: &str) -> Result<(), bincode::Error> {
    if value.is_empty() || value.trim() != value {
        return Err(validation_error(&format!(
            "{label} must be non-empty without leading or trailing whitespace"
        )));
    }
    Ok(())
}

fn validate_type_variable_restrictions(
    label: &str,
    restrictions: &[TypeVariableRestriction],
) -> Result<(), bincode::Error> {
    let mut previous = None;
    for restriction in restrictions {
        let index = canonical_type_variable_index(&restriction.variable)?;
        if previous.is_some_and(|prior| prior >= index) {
            return Err(validation_error(&format!(
                "{label} must be strictly sorted by canonical index with no duplicates"
            )));
        }
        previous = Some(index);
    }
    Ok(())
}

fn canonical_type_variable_index(value: &str) -> Result<u32, bincode::Error> {
    let Some(index) = value.strip_prefix('t') else {
        return Err(validation_error(
            "restricted type-variable identities must use canonical `t<index>` spelling",
        ));
    };
    if index.is_empty()
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || (index.len() > 1 && index.starts_with('0'))
    {
        return Err(validation_error(
            "restricted type-variable identities must use canonical `t<index>` spelling",
        ));
    }
    index.parse().map_err(|_| {
        validation_error("restricted type-variable identity index exceeds the u32 range")
    })
}

fn canonical_type_variables(type_repr: &str) -> Result<BTreeSet<u32>, bincode::Error> {
    let expression = chelis_deep::parse_and_stamp_type(type_repr).map_err(|error| {
        validation_error(&format!("type representation is not valid Deep: {error}"))
    })?;

    fn visit(
        expr: &chelis_deep::Expr,
        variables: &mut BTreeSet<u32>,
    ) -> Result<(), bincode::Error> {
        use chelis_deep::{Atom, DeepTag, Expr};

        match expr {
            Expr::Atom(..) => {}
            Expr::Node(node, _) => {
                if node.tag() == DeepTag::TVar {
                    let Some(Expr::Atom(Atom::Name(name), _)) = node.children_slice().first()
                    else {
                        return Err(validation_error(
                            "t-var type representation is missing its canonical identity",
                        ));
                    };
                    variables.insert(canonical_type_variable_index(name)?);
                }
                for child in node.children_slice() {
                    visit(child, variables)?;
                }
            }
            Expr::Map(..)
            | Expr::MetaExpr(..)
            | Expr::List(..)
            | Expr::BareList(..)
            | Expr::UnknownForm(..) => {
                return Err(validation_error(
                    "type representation contains a non-type carrier",
                ));
            }
        }
        Ok(())
    }

    let mut variables = BTreeSet::new();
    visit(&expression, &mut variables)?;
    for (expected, actual) in variables.iter().copied().enumerate() {
        if actual != expected as u32 {
            return Err(validation_error(
                "type representation variables must be contiguous from `t0`",
            ));
        }
    }
    Ok(variables)
}

fn validate_compiler_pin(pin: &str) -> Result<(), bincode::Error> {
    let Some(version) = pin.strip_prefix('=') else {
        return Err(validation_error(
            "compiler must be an exact pin beginning with `=`",
        ));
    };
    semver::Version::parse(version).map_err(|error| {
        validation_error(&format!(
            "compiler must be an exact `=<semver>` pin: {error}"
        ))
    })?;
    Ok(())
}

fn validate_strict_order<'a>(
    label: &str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), bincode::Error> {
    let mut previous: Option<&str> = None;
    for value in values {
        if previous.is_some_and(|prior| prior >= value) {
            return Err(validation_error(&format!(
                "{label} must be strictly sorted with no duplicates"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn is_lowercase_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validation_error(message: &str) -> bincode::Error {
    Box::new(bincode::ErrorKind::Custom(format!(
        "invalid shell envelope: {message}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_roundtrips() {
        let shell = fixture_shell();

        let bytes = encode_shell(&shell).expect("encode shell");
        let decoded = decode_shell(&bytes).expect("decode shell");
        assert_eq!(decoded, shell);
    }

    #[test]
    fn shell_decode_rejects_trailing_bytes() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes.extend_from_slice(b"junk");

        assert!(
            decode_shell(&bytes).is_err(),
            "CHB decoding must consume the complete input"
        );
    }

    #[test]
    fn shell_decode_rejects_every_truncation() {
        let bytes = encode_shell(&fixture_shell()).expect("encode shell");

        for end in 0..bytes.len() {
            assert!(
                decode_shell(&bytes[..end]).is_err(),
                "truncation at byte {end} unexpectedly decoded"
            );
        }
    }

    #[test]
    fn shell_encoding_has_an_explicit_v3_envelope() {
        let bytes = encode_shell(&fixture_shell()).expect("encode shell");

        assert_eq!(&bytes[..8], b"CHELCHB\0");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            SHELL_FORMAT_VERSION
        );
        // v4 carries operation-value domains separately from authored dtype bounds.
        assert_eq!(SHELL_FORMAT_VERSION, 4);
    }

    #[test]
    fn shell_decode_rejects_the_unversioned_predecessor_layout() {
        let predecessor = bincode::serialize(&fixture_shell()).expect("encode old shell payload");

        let error = decode_shell(&predecessor).expect_err("unversioned CHB must be rejected");
        assert!(
            error
                .to_string()
                .contains("unsupported predecessor shell format"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn shell_decode_rejects_unknown_format_versions() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes[8..12].copy_from_slice(&99_u32.to_le_bytes());

        let error = decode_shell(&bytes).expect_err("unknown CHB version must be rejected");
        assert_eq!(
            error.to_string(),
            "invalid shell envelope: shell format version 99 is unsupported; expected 4"
        );
    }

    #[test]
    fn operation_value_domains_round_trip_without_becoming_dtype_bounds() {
        for domain in [
            TypeVariableDomain::FloatValue,
            TypeVariableDomain::IntValue,
            TypeVariableDomain::NumericValue,
        ] {
            let mut shell = fixture_shell();
            let symbol = &mut shell.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t0))".into());
            symbol.type_variable_restrictions = vec![TypeVariableRestriction {
                variable: "t0".into(),
                domain,
            }];
            let encoded = encode_shell(&shell).unwrap();
            assert_eq!(decode_shell(&encoded).unwrap(), shell);
            let mut old_version = encoded;
            old_version[8..12].copy_from_slice(&3_u32.to_le_bytes());
            assert!(decode_shell(&old_version).is_err());
        }
    }

    #[test]
    fn shell_validation_rejects_a_restriction_detached_from_its_type_variable() {
        let mut shell = fixture_shell();
        shell.modules[0].exports[0].type_variable_restrictions = vec![TypeVariableRestriction {
            variable: "t1".to_string(),
            domain: TypeVariableDomain::ActiveFloat,
        }];

        let error = validate_shell(&shell).expect_err("detached restriction must be rejected");
        assert!(
            error.to_string().contains(
                "restricts `t1` but that canonical variable is absent from its type representation"
            ),
            "unexpected diagnostic: {error}"
        );

        let mut metadata_only = fixture_shell();
        let symbol = &mut metadata_only.modules[0].exports[0];
        symbol.type_repr = Some("(t-prim {type: (t-var {} t0)} f32)".to_string());
        symbol.type_variable_restrictions = vec![TypeVariableRestriction {
            variable: "t0".to_string(),
            domain: TypeVariableDomain::ActiveFloat,
        }];
        let error = validate_shell(&metadata_only)
            .expect_err("a metadata-only variable is not part of the quantified type");
        assert!(
            error.to_string().contains(
                "restricts `t0` but that canonical variable is absent from its type representation"
            ),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn shell_validation_rejects_malformed_type_representations_for_every_scheme_shape() {
        let malformed = [
            ("invalid syntax", "(t-unit {}"),
            ("multiple expressions", "(t-unit {}) (t-unit {})"),
            ("unknown root", "(not-a-type {} f32)"),
            ("bare atom", "f32"),
            ("bare list", "()"),
            ("bare map", "{type: (t-prim {} f32)}"),
            (
                "legacy metadata-expression wrapper",
                "^{:doc \"legacy\"} (t-prim {} f32)",
            ),
            ("runtime root", "(lit {type: (t-var {} t0)} 1)"),
            (
                "nested runtime node",
                "(t-fn {} (lit {type: (t-var {} t0)} 1) (t-var {} t0))",
            ),
            (
                "tensor namespace swap",
                "(t-tensor {} (t-var {} t0) (d-rank {} r0))",
            ),
        ];

        for has_body in [false, true] {
            for restricted in [false, true] {
                for (label, source) in malformed {
                    let mut shell = fixture_shell();
                    let symbol = &mut shell.modules[0].exports[0];
                    symbol.has_body = has_body;
                    symbol.type_repr = Some(source.to_string());
                    symbol.type_variable_restrictions = restricted
                        .then(|| TypeVariableRestriction {
                            variable: "t0".to_string(),
                            domain: TypeVariableDomain::ActiveFloat,
                        })
                        .into_iter()
                        .collect();

                    let shape = format!("{label}, has_body={has_body}, restricted={restricted}");
                    assert!(
                        validate_shell(&shell).is_err(),
                        "validate_shell admitted {shape}"
                    );
                    assert!(
                        encode_shell(&shell).is_err(),
                        "encode_shell admitted {shape}"
                    );
                    assert!(
                        decode_shell(&encode_unvalidated(&shell)).is_err(),
                        "decode_shell admitted {shape}"
                    );
                }
            }
        }
    }

    #[test]
    fn shell_validation_requires_value_signatures_and_accepts_valid_authored_fallbacks() {
        for has_body in [false, true] {
            for restricted in [false, true] {
                let mut shell = fixture_shell();
                let symbol = &mut shell.modules[0].exports[0];
                symbol.has_body = has_body;
                symbol.type_repr = Some(if restricted {
                    "(t-fn {} (t-var {} t0) (t-var {} t0))".to_string()
                } else {
                    "(t-fn {} (t-prim {} f32) (t-prim {} f32))".to_string()
                });
                symbol.type_variable_restrictions = restricted
                    .then(|| TypeVariableRestriction {
                        variable: "t0".to_string(),
                        domain: TypeVariableDomain::ActiveFloat,
                    })
                    .into_iter()
                    .collect();

                validate_shell(&shell).unwrap_or_else(|error| {
                    panic!(
                        "valid signature rejected for has_body={has_body}, restricted={restricted}: {error}"
                    )
                });
                decode_shell(&encode_shell(&shell).expect("encode valid shell"))
                    .expect("decode valid shell");
            }
        }

        for has_body in [false, true] {
            let mut shell = fixture_shell();
            let symbol = &mut shell.modules[0].exports[0];
            symbol.has_body = has_body;
            symbol.type_repr = None;
            assert!(
                validate_shell(&shell).is_err(),
                "value export without a type representation was admitted (has_body={has_body})"
            );
        }

        for kind in [SymbolKind::Type, SymbolKind::Macro, SymbolKind::Dim] {
            let mut shell = fixture_shell();
            let symbol = &mut shell.modules[0].exports[0];
            symbol.kind = kind;
            symbol.type_repr = None;
            symbol.has_body = false;
            validate_shell(&shell).unwrap_or_else(|error| {
                panic!("{kind:?} export without a value type failed: {error}")
            });
        }
    }

    fn encode_unvalidated(shell: &ShellPackage) -> Vec<u8> {
        let payload = bincode::serialize(shell).expect("serialize malformed test payload");
        let mut bytes = Vec::with_capacity(SHELL_MAGIC.len() + 4 + payload.len());
        bytes.extend_from_slice(SHELL_MAGIC);
        bytes.extend_from_slice(&SHELL_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn fixture_shell() -> ShellPackage {
        ShellPackage {
            format_version: SHELL_FORMAT_VERSION,
            package: PackageId {
                name: "chelis-std".to_string(),
                version: "0.1.0".to_string(),
            },
            compiler: "=0.1.21".to_string(),
            modules: vec![ShellModule {
                module: "App.Demo".to_string(),
                exports: vec![ShellSymbol {
                    name: "forward".to_string(),
                    kind: SymbolKind::Value,
                    type_repr: Some("(t-fn {} (t-prim {} f32) (t-prim {} f32))".to_string()),
                    type_variable_restrictions: Vec::new(),
                    effects: Vec::new(),
                    has_body: true,
                }],
            }],
            dependencies: Vec::new(),
            archive_sha256: "d".repeat(64),
        }
    }
}
