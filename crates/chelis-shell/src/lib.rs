use bincode::Options;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub const SHELL_MAGIC: &[u8; 8] = b"CHELCHB\0";
// v4 carries #2071's authored and operation-value type-variable domains. v5
// adds checked collection-operation relations. Positional bincode cannot read
// either change as an absent field without changing the programs an export
// admits, so both require exact version rejection.
/// Bumped to 6 for chelis#2443: [`TypeVariableDomain`] gained an
/// `ActiveSet` variant, so a shell published by this compiler can carry a
/// domain a version-5 reader cannot decode.
// #3130: Surf stages carry explicit syntax and Deep 0.20 has no Pipe.
pub const SHELL_FORMAT_VERSION: u32 = 7;

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
    /// Published description of this function value's checked
    /// collection-operation relations. This ledger contributes to CHB package
    /// identity; it is not a serialized checker-reuse environment. Every
    /// canonical variable must also occur in `type_repr`; newly authored
    /// wrappers may not publish hidden body-inferred predicates ([04-INF-9]).
    pub collection_obligations: Vec<CollectionObligation>,
    pub effects: Vec<String>,
    pub has_body: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CollectionObligation {
    Len {
        operand: String,
        result: String,
    },
    Index {
        list: String,
        index: String,
        result: String,
    },
    Append {
        list: String,
        value: String,
        result: String,
    },
    Concat {
        lhs: String,
        rhs: String,
        result: String,
    },
    KeyFromSeed {
        operand: String,
        result: String,
    },
    SplitKey {
        operand: String,
        result: String,
    },
    SplitKeys {
        operand: String,
        count: String,
        result: String,
    },
    FoldIn {
        operand: String,
        index: String,
        result: String,
    },
}

impl CollectionObligation {
    pub fn builtin(&self) -> &'static str {
        match self {
            Self::Len { .. } => "len",
            Self::Index { .. } => "index",
            Self::Append { .. } => "append",
            Self::Concat { .. } => "concat",
            Self::KeyFromSeed { .. } => "key_from_seed",
            Self::SplitKey { .. } => "split_key",
            Self::SplitKeys { .. } => "split_keys",
            Self::FoldIn { .. } => "fold_in",
        }
    }

    pub fn types(&self) -> Vec<&str> {
        match self {
            Self::Len { operand, result } => vec![operand, result],
            Self::Index {
                list,
                index,
                result,
            } => vec![list, index, result],
            Self::Append {
                list,
                value,
                result,
            } => vec![list, value, result],
            Self::Concat { lhs, rhs, result } => vec![lhs, rhs, result],
            Self::KeyFromSeed { operand, result } | Self::SplitKey { operand, result } => {
                vec![operand, result]
            }
            Self::SplitKeys {
                operand,
                count,
                result,
            } => vec![operand, count, result],
            Self::FoldIn {
                operand,
                index,
                result,
            } => vec![operand, index, result],
        }
    }
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
// Not `Copy`: §5.9's set form publishes its member spellings, so the domain
// carries a `Vec`. A published format names its dtypes rather than encoding
// them as bits, which a shell consumer would have to decode.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
    /// §5.9's explicit dtype set, as the member spellings it admits in §1.1
    /// declaration order. Unlike a family, this does not widen when §1.1
    /// activates a dtype, so the members are published rather than a name.
    ActiveSet(Vec<String>),
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
            "shell format version {version} is unsupported; expected {SHELL_FORMAT_VERSION}; {}regenerate the shell package",
            if version == 6 {
                "Deep 0.20 removes pipe nodes; "
            } else {
                ""
            }
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
            "payload format version {} is unsupported; expected {SHELL_FORMAT_VERSION}; {}regenerate the shell package",
            shell.format_version,
            if shell.format_version == 6 {
                "Deep 0.20 removes pipe nodes; "
            } else {
                ""
            }
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
            validate_collection_obligations_for_type_repr(
                symbol.type_repr.as_deref(),
                &symbol.collection_obligations,
            )
            .map_err(|message| {
                validation_error(&format!(
                    "collection obligations for export `{}` in module `{}` {message}",
                    symbol.name, module.module
                ))
            })?;
            let variables = match &symbol.type_repr {
                Some(type_repr) => Some(canonical_variables(type_repr, true)?),
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
            if let Some(variables) = variables {
                for restriction in &symbol.type_variable_restrictions {
                    let index = canonical_type_variable_index(&restriction.variable)?;
                    if !variables.types.contains(&index) {
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

#[derive(Default)]
struct CanonicalVariables {
    types: BTreeSet<u32>,
    dimensions: BTreeSet<u32>,
    ranks: BTreeSet<u32>,
}

fn canonical_variables(
    type_repr: &str,
    require_contiguous: bool,
) -> Result<CanonicalVariables, bincode::Error> {
    let expression = chelis_deep::parse_and_stamp_type(type_repr).map_err(|error| {
        validation_error(&format!("type representation is not valid Deep: {error}"))
    })?;

    fn visit(
        expr: &chelis_deep::Expr,
        variables: &mut CanonicalVariables,
    ) -> Result<(), bincode::Error> {
        use chelis_deep::{Atom, DeepTag, Expr};

        match expr {
            Expr::Atom(..) => {}
            Expr::Node(node, _) => {
                let namespace = match node.tag() {
                    DeepTag::TVar => Some(("t-var", &mut variables.types, "t")),
                    DeepTag::DVar => Some(("d-var", &mut variables.dimensions, "d")),
                    DeepTag::DRank => Some(("d-rank", &mut variables.ranks, "r")),
                    _ => None,
                };
                if let Some((tag, namespace, prefix)) = namespace {
                    let Some(Expr::Atom(Atom::Name(name), _)) = node.children_slice().first()
                    else {
                        return Err(validation_error(&format!(
                            "{tag} type representation is missing its canonical identity"
                        )));
                    };
                    namespace.insert(canonical_variable_index(name, prefix, tag)?);
                }
                for child in node.children_slice() {
                    visit(child, variables)?;
                }
            }
            Expr::Map(..) | Expr::MetaExpr(..) | Expr::BareList(..) | Expr::UnknownForm(..) => {
                return Err(validation_error(
                    "type representation contains a non-type carrier",
                ));
            }
        }
        Ok(())
    }

    let mut variables = CanonicalVariables::default();
    visit(&expression, &mut variables)?;
    if require_contiguous {
        for (namespace, prefix, indices) in [
            ("type", "t", &variables.types),
            ("dimension", "d", &variables.dimensions),
            ("rank", "r", &variables.ranks),
        ] {
            for (expected, actual) in indices.iter().copied().enumerate() {
                if actual != expected as u32 {
                    return Err(validation_error(&format!(
                        "type representation {namespace} variables must be contiguous from `{prefix}0`"
                    )));
                }
            }
        }
    }
    Ok(variables)
}

fn canonical_variable_index(value: &str, prefix: &str, tag: &str) -> Result<u32, bincode::Error> {
    let Some(index) = value.strip_prefix(prefix) else {
        return Err(validation_error(&format!(
            "{tag} identities must use canonical `{prefix}<index>` spelling"
        )));
    };
    if index.is_empty()
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || (index.len() > 1 && index.starts_with('0'))
    {
        return Err(validation_error(&format!(
            "{tag} identities must use canonical `{prefix}<index>` spelling"
        )));
    }
    index
        .parse()
        .map_err(|_| validation_error(&format!("{tag} identity index exceeds the u32 range")))
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

/// Validate one published collection-contract ledger against its callable
/// type representation.
///
/// Both CHB validation and public Reef-schema deserialization use this
/// boundary so canonical Deep parsing, ledger identity, and carried-variable
/// membership cannot drift between package formats.
pub fn validate_collection_obligations_for_type_repr(
    type_repr: Option<&str>,
    obligations: &[CollectionObligation],
) -> Result<(), String> {
    if obligations.is_empty() {
        return Ok(());
    }

    validate_collection_obligation_ledger(obligations)?;
    let type_repr =
        type_repr.ok_or_else(|| "require a function type representation".to_string())?;
    let expression = chelis_deep::parse_and_stamp_type(type_repr).map_err(|error| {
        format!("require a function type representation that is valid Deep: {error}")
    })?;
    if !matches!(
        &expression,
        chelis_deep::Expr::Node(node, _) if node.tag() == chelis_deep::DeepTag::TFn
    ) {
        return Err("require a function type representation".to_string());
    }

    let variables =
        canonical_variables(type_repr, true).map_err(|error| validation_message(&error))?;
    for obligation in obligations {
        for carried in obligation.types() {
            let carried_variables =
                canonical_variables(carried, false).map_err(|error| validation_message(&error))?;
            for (prefix, carried_indices, declared_indices) in [
                ("t", &carried_variables.types, &variables.types),
                ("d", &carried_variables.dimensions, &variables.dimensions),
                ("r", &carried_variables.ranks, &variables.ranks),
            ] {
                for index in carried_indices {
                    if !declared_indices.contains(index) {
                        return Err(format!(
                            "contain a `{}` contract variable `{prefix}{index}` absent from its type representation",
                            obligation.builtin()
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validate the canonical identity of one published collection-contract
/// ledger.
///
/// Parsing alone is insufficient at an artifact trust boundary: whitespace
/// variants describe the same Deep type while producing distinct CHB bytes.
/// Exact canonical rendering plus strict order makes semantic identity and
/// artifact identity agree.
pub fn validate_collection_obligation_ledger(
    obligations: &[CollectionObligation],
) -> Result<(), String> {
    let mut previous: Option<(&str, Vec<&str>)> = None;
    for obligation in obligations {
        for carried in obligation.types() {
            let expression = chelis_deep::parse_and_stamp_type(carried).map_err(|error| {
                format!("contain a type representation that is not valid Deep: {error}")
            })?;
            let canonical = chelis_deep::printer::print_expr(&expression);
            if canonical != carried {
                return Err(format!(
                    "must use exact canonical Deep rendering; got `{carried}`, canonical form is `{canonical}`"
                ));
            }
        }
        let key = (obligation.builtin(), obligation.types());
        if previous.as_ref().is_some_and(|prior| prior >= &key) {
            return Err("must be strictly sorted with no duplicates".to_string());
        }
        previous = Some(key);
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

fn validation_message(error: &bincode::Error) -> String {
    let rendered = error.to_string();
    rendered
        .strip_prefix("invalid shell envelope: ")
        .unwrap_or(&rendered)
        .to_string()
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
    fn shell_encoding_has_an_explicit_v7_envelope() {
        let bytes = encode_shell(&fixture_shell()).expect("encode shell");

        assert_eq!(&bytes[..8], b"CHELCHB\0");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            SHELL_FORMAT_VERSION
        );
        assert_eq!(SHELL_FORMAT_VERSION, 7);
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
            "invalid shell envelope: shell format version 99 is unsupported; expected 7; regenerate the shell package"
        );
    }

    #[test]
    fn shell_decode_rejects_the_pre_collection_contract_version() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes[8..12].copy_from_slice(&4_u32.to_le_bytes());

        let error = decode_shell(&bytes).expect_err("CHB v4 must not decode without relations");
        assert_eq!(
            error.to_string(),
            "invalid shell envelope: shell format version 4 is unsupported; expected 7; regenerate the shell package"
        );
    }

    /// chelis#2443 took the envelope to 6, because `TypeVariableDomain` gained
    /// spec/04 §5.9's explicit dtype set and a v5 reader cannot decode it.
    /// A v5 shell must therefore be refused rather than read as if the domain
    /// vocabulary were unchanged.
    #[test]
    fn shell_decode_rejects_the_pre_dtype_set_domain_version() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes[8..12].copy_from_slice(&5_u32.to_le_bytes());

        let error = decode_shell(&bytes).expect_err("CHB v5 predates the dtype-set domain");
        assert_eq!(
            error.to_string(),
            "invalid shell envelope: shell format version 5 is unsupported; expected 7; regenerate the shell package"
        );
    }

    #[test]
    fn shell_decode_rejects_the_pre_pipe_normalization_version_before_payload_decode() {
        let mut bytes = b"CHELCHB\0".to_vec();
        bytes.extend_from_slice(&6_u32.to_le_bytes());
        let error =
            decode_shell(&bytes).expect_err("v6 must be refused before reading its payload");
        assert_eq!(
            error.to_string(),
            "invalid shell envelope: shell format version 6 is unsupported; expected 7; Deep 0.20 removes pipe nodes; regenerate the shell package"
        );
    }

    #[test]
    fn checked_collection_contracts_round_trip_and_change_identity() {
        let mut obligated = fixture_shell();
        let symbol = &mut obligated.modules[0].exports[0];
        symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
        symbol.collection_obligations = vec![CollectionObligation::Len {
            operand: "(t-var {} t0)".into(),
            result: "(t-var {} t1)".into(),
        }];

        let encoded = encode_shell(&obligated).expect("checked contract encodes");
        assert_eq!(decode_shell(&encoded).unwrap(), obligated);
        assert_ne!(encoded, encode_shell(&fixture_shell()).unwrap());
    }

    #[test]
    fn key_operation_contracts_round_trip_and_change_identity() {
        for obligation in [
            CollectionObligation::KeyFromSeed {
                operand: "(t-var {} t0)".into(),
                result: "(t-var {} t1)".into(),
            },
            CollectionObligation::SplitKey {
                operand: "(t-var {} t0)".into(),
                result: "(t-var {} t1)".into(),
            },
            CollectionObligation::SplitKeys {
                operand: "(t-var {} t0)".into(),
                count: "(t-prim {} i64)".into(),
                result: "(t-var {} t1)".into(),
            },
            CollectionObligation::FoldIn {
                operand: "(t-var {} t0)".into(),
                index: "(t-var {} t0)".into(),
                result: "(t-var {} t1)".into(),
            },
        ] {
            let mut obligated = fixture_shell();
            let symbol = &mut obligated.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
            symbol.collection_obligations = vec![obligation];
            let encoded = encode_shell(&obligated).unwrap();
            assert_eq!(decode_shell(&encoded).unwrap(), obligated);
            assert_ne!(encoded, encode_shell(&fixture_shell()).unwrap());
        }
    }

    #[test]
    fn checked_collection_contract_ledger_is_strictly_canonical() {
        let append = CollectionObligation::Append {
            list: "(t-var {} t0)".into(),
            value: "(t-var {} t1)".into(),
            result: "(t-var {} t0)".into(),
        };
        let len = CollectionObligation::Len {
            operand: "(t-var {} t0)".into(),
            result: "(t-var {} t1)".into(),
        };

        let mut canonical = fixture_shell();
        let symbol = &mut canonical.modules[0].exports[0];
        symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
        symbol.collection_obligations = vec![append.clone(), len.clone()];
        let encoded = encode_shell(&canonical).expect("canonical relation ledger must encode");
        assert_eq!(
            decode_shell(&encoded).expect("canonical relation ledger must decode"),
            canonical
        );

        for (label, obligations) in [
            ("duplicate", vec![len.clone(), len.clone()]),
            ("noncanonical order", vec![len.clone(), append.clone()]),
        ] {
            let mut invalid = fixture_shell();
            let symbol = &mut invalid.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
            symbol.collection_obligations = obligations;

            let error = validate_shell(&invalid)
                .expect_err("noncanonical relation ledger must fail validation");
            assert!(
                error.to_string().contains("strictly sorted"),
                "unexpected {label} validation diagnostic: {error}"
            );
            assert!(
                encode_shell(&invalid).is_err(),
                "{label} relation ledger must not encode"
            );
            assert!(
                decode_shell(&encode_unvalidated(&invalid)).is_err(),
                "{label} relation ledger must not decode"
            );
        }

        for (label, mutate) in [
            ("operand whitespace", 0usize),
            ("result whitespace", 1usize),
        ] {
            let mut invalid = fixture_shell();
            let symbol = &mut invalid.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
            let mut noncanonical = len.clone();
            if let CollectionObligation::Len { operand, result } = &mut noncanonical {
                if mutate == 0 {
                    *operand = "(t-var   {} t0)".into();
                } else {
                    *result = "(t-var {}   t1)".into();
                }
            }
            symbol.collection_obligations = vec![noncanonical];

            let error =
                validate_shell(&invalid).expect_err("noncanonical Deep must fail validation");
            assert!(
                error.to_string().contains("canonical Deep"),
                "unexpected {label} validation diagnostic: {error}"
            );
            assert!(
                encode_shell(&invalid).is_err(),
                "{label} relation ledger must not encode"
            );
            assert!(
                decode_shell(&encode_unvalidated(&invalid)).is_err(),
                "{label} relation ledger must not decode"
            );
        }
    }

    #[test]
    fn checked_collection_contracts_reject_hidden_or_malformed_variables() {
        let mut hidden = fixture_shell();
        hidden.modules[0].exports[0].collection_obligations = vec![CollectionObligation::Len {
            operand: "(t-var {} t0)".into(),
            result: "(t-prim {} i64)".into(),
        }];
        assert!(
            validate_shell(&hidden)
                .unwrap_err()
                .to_string()
                .contains("absent from its type representation")
        );

        let mut malformed = fixture_shell();
        malformed.modules[0].exports[0].collection_obligations = vec![CollectionObligation::Len {
            operand: "List[t0]".into(),
            result: "(t-prim {} i64)".into(),
        }];
        assert!(
            validate_shell(&malformed)
                .unwrap_err()
                .to_string()
                .contains("not valid Deep")
        );

        for (operand, variable) in [
            (
                "(t-tensor {} (d-var {} d1) (d-rank {} r0) (t-prim {} f32))",
                "`d1`",
            ),
            (
                "(t-tensor {} (d-var {} d0) (d-rank {} r1) (t-prim {} f32))",
                "`r1`",
            ),
        ] {
            let mut hidden_shape = fixture_shell();
            let symbol = &mut hidden_shape.modules[0].exports[0];
            symbol.type_repr = Some(
                "(t-fn {} (t-tensor {} (d-var {} d0) (d-rank {} r0) (t-prim {} f32)) (t-prim {} i64))"
                    .into(),
            );
            symbol.collection_obligations = vec![CollectionObligation::Len {
                operand: operand.into(),
                result: "(t-prim {} i64)".into(),
            }];
            let error = validate_shell(&hidden_shape).unwrap_err();
            assert!(
                error.to_string().contains(variable),
                "unexpected hidden-variable diagnostic: {error}"
            );
        }
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
                    collection_obligations: Vec::new(),
                    effects: Vec::new(),
                    has_body: true,
                }],
            }],
            dependencies: Vec::new(),
            archive_sha256: "d".repeat(64),
        }
    }
}
