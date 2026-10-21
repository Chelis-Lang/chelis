//! Portable CHB metadata and complete validation.
//!
//! Callers supply stack services for Deep type validation.
//! The host owns binary codecs, files, archive hashes, and installation.
//! A valid metadata value does not prove archive integrity or publisher identity.
//! Executable consumers need an allocator and enough stack resources.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use chelis_deep::parser::StampOrParseError;
use chelis_stack_services::StackServices;
use core::fmt;
use serde::{Deserialize, Serialize};
// v4 carries #2071's authored and operation-value type-variable domains. v5
// adds checked collection-operation relations. Positional bincode cannot read
// either change as an absent field without changing the programs an export
// admits, so both require exact version rejection.
/// Bumped to 6 for chelis#2443: [`TypeVariableDomain`] gained an
/// `ActiveSet` variant, so a shell published by this compiler can carry a
/// domain a version-5 reader cannot decode.
// #3130: Surf stages carry explicit syntax and Deep 0.20 has no Pipe.
// Float classification adds a checked predicate relation to exported schemes.
pub const SHELL_FORMAT_VERSION: u32 = 8;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatClassifyKind {
    Nan,
    Finite,
    Infinite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CollectionObligation {
    FloatClassify {
        kind: FloatClassifyKind,
        operand: String,
        result: String,
    },
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
            Self::FloatClassify { kind, .. } => match kind {
                FloatClassifyKind::Nan => "is_nan",
                FloatClassifyKind::Finite => "is_finite",
                FloatClassifyKind::Infinite => "is_infinite",
            },
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
            Self::FloatClassify {
                operand, result, ..
            }
            | Self::Len { operand, result } => vec![operand, result],
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

/// A text field with a nonempty, edge-whitespace constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellTextField {
    PackageName,
    PackageVersion,
    ModuleName,
    ExportName,
    EffectName,
    DependencyName,
}

impl fmt::Display for ShellTextField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PackageName => "package.name",
            Self::PackageVersion => "package.version",
            Self::ModuleName => "module name",
            Self::ExportName => "export name",
            Self::EffectName => "effect name",
            Self::DependencyName => "dependency name",
        })
    }
}

/// A collection with strict name order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellCollection {
    Modules,
    Exports { module: String },
    Dependencies,
}

impl fmt::Display for ShellCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Modules => f.write_str("modules"),
            Self::Exports { module } => write!(f, "exports for module `{module}`"),
            Self::Dependencies => f.write_str("dependencies"),
        }
    }
}

/// An invalid canonical restriction identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestrictionIdentityError {
    Spelling,
    IndexOverflow,
}

impl fmt::Display for RestrictionIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Spelling => {
                "restricted type-variable identities must use canonical `t<index>` spelling"
            }
            Self::IndexOverflow => "restricted type-variable identity index exceeds the u32 range",
        })
    }
}

impl core::error::Error for RestrictionIdentityError {}

/// A failure in a published Deep type representation.
#[derive(Debug)]
pub enum TypeRepresentationError {
    InvalidDeep {
        source: Box<StampOrParseError>,
    },
    MissingCanonicalIdentity {
        tag: &'static str,
    },
    IdentitySpelling {
        tag: &'static str,
        prefix: &'static str,
    },
    IdentityOverflow {
        tag: &'static str,
    },
    NonTypeCarrier,
    Noncontiguous {
        namespace: &'static str,
        prefix: &'static str,
    },
}

impl fmt::Display for TypeRepresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDeep { source } => {
                write!(f, "type representation is not valid Deep: {source}")
            }
            Self::MissingCanonicalIdentity { tag } => write!(
                f,
                "{tag} type representation is missing its canonical identity"
            ),
            Self::IdentitySpelling { tag, prefix } => write!(
                f,
                "{tag} identities must use canonical `{prefix}<index>` spelling"
            ),
            Self::IdentityOverflow { tag } => {
                write!(f, "{tag} identity index exceeds the u32 range")
            }
            Self::NonTypeCarrier => f.write_str("type representation contains a non-type carrier"),
            Self::Noncontiguous { namespace, prefix } => write!(
                f,
                "type representation {namespace} variables must be contiguous from `{prefix}0`"
            ),
        }
    }
}

impl core::error::Error for TypeRepresentationError {}

/// A failure in a published collection-contract ledger.
#[derive(Debug)]
pub enum CollectionValidationError {
    InvalidCarriedDeep {
        source: Box<StampOrParseError>,
    },
    NoncanonicalType {
        supplied: String,
        canonical: String,
    },
    StrictOrder,
    MissingFunctionType,
    InvalidFunctionDeep {
        source: Box<StampOrParseError>,
    },
    TypeRepresentation {
        source: TypeRepresentationError,
    },
    MissingContractVariable {
        builtin: &'static str,
        prefix: &'static str,
        index: u32,
    },
}

impl fmt::Display for CollectionValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCarriedDeep { source } => write!(
                f,
                "contain a type representation that is not valid Deep: {source}"
            ),
            Self::NoncanonicalType {
                supplied,
                canonical,
            } => write!(
                f,
                "must use exact canonical Deep rendering; got `{supplied}`, canonical form is `{canonical}`"
            ),
            Self::StrictOrder => f.write_str("must be strictly sorted with no duplicates"),
            Self::MissingFunctionType => f.write_str("require a function type representation"),
            Self::InvalidFunctionDeep { source } => write!(
                f,
                "require a function type representation that is valid Deep: {source}"
            ),
            Self::TypeRepresentation { source } => source.fmt(f),
            Self::MissingContractVariable {
                builtin,
                prefix,
                index,
            } => write!(
                f,
                "contain a `{builtin}` contract variable `{prefix}{index}` absent from its type representation"
            ),
        }
    }
}

impl core::error::Error for CollectionValidationError {}

/// A typed failure in the complete CHB metadata value.
#[derive(Debug)]
pub enum ShellValidationError {
    UnsupportedFormatVersion {
        version: u32,
    },
    InvalidText {
        field: ShellTextField,
    },
    CompilerPinPrefix,
    CompilerPinVersion {
        source: semver::Error,
    },
    ArchiveSha256,
    StrictOrder {
        collection: ShellCollection,
    },
    CollectionObligations {
        module: String,
        symbol: String,
        source: CollectionValidationError,
    },
    TypeRepresentation {
        source: TypeRepresentationError,
    },
    MissingValueType {
        module: String,
        symbol: String,
    },
    RestrictionIdentity {
        source: RestrictionIdentityError,
    },
    RestrictionOrder {
        module: String,
        symbol: String,
    },
    RestrictionsWithoutType {
        module: String,
        symbol: String,
    },
    RestrictionAbsentFromType {
        module: String,
        symbol: String,
        variable: String,
    },
    DependencyVersionWhitespace {
        dependency: String,
    },
}

impl fmt::Display for ShellValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid shell envelope: ")?;
        match self {
            Self::UnsupportedFormatVersion { version } => write!(
                f,
                "payload format version {version} is unsupported; expected {SHELL_FORMAT_VERSION}; {}regenerate the shell package",
                if *version == 6 {
                    "Deep 0.20 removes pipe nodes; "
                } else {
                    ""
                },
            ),
            Self::InvalidText { field } => write!(
                f,
                "{field} must be non-empty without leading or trailing whitespace"
            ),
            Self::CompilerPinPrefix => {
                f.write_str("compiler must be an exact pin beginning with `=`")
            }
            Self::CompilerPinVersion { source } => {
                write!(f, "compiler must be an exact `=<semver>` pin: {source}")
            }
            Self::ArchiveSha256 => {
                f.write_str("archive_sha256 must be exactly 64 lowercase hexadecimal characters")
            }
            Self::StrictOrder { collection } => {
                write!(f, "{collection} must be strictly sorted with no duplicates")
            }
            Self::CollectionObligations {
                module,
                symbol,
                source,
            } => write!(
                f,
                "collection obligations for export `{symbol}` in module `{module}` {source}"
            ),
            Self::TypeRepresentation { source } => source.fmt(f),
            Self::MissingValueType { module, symbol } => write!(
                f,
                "value export `{symbol}` in module `{module}` requires a type representation"
            ),
            Self::RestrictionIdentity { source } => source.fmt(f),
            Self::RestrictionOrder { module, symbol } => write!(
                f,
                "type-variable restrictions for export `{symbol}` in module `{module}` must be strictly sorted by canonical index with no duplicates"
            ),
            Self::RestrictionsWithoutType { module, symbol } => write!(
                f,
                "export `{symbol}` in module `{module}` carries type-variable restrictions without a type representation"
            ),
            Self::RestrictionAbsentFromType {
                module,
                symbol,
                variable,
            } => write!(
                f,
                "export `{symbol}` in module `{module}` restricts `{variable}` but that canonical variable is absent from its type representation"
            ),
            Self::DependencyVersionWhitespace { dependency } => write!(
                f,
                "dependency `{dependency}` version must not have leading or trailing whitespace"
            ),
        }
    }
}

impl core::error::Error for ShellValidationError {}

/// Validate every metadata field without mutation.
///
/// The caller supplies stack services for the portable Deep parser.
/// This operation does not read bytes, files, archives, or registries.
pub fn validate_shell(
    shell: &ShellPackage,
    stack: &impl StackServices,
) -> Result<(), ShellValidationError> {
    if shell.format_version != SHELL_FORMAT_VERSION {
        return Err(ShellValidationError::UnsupportedFormatVersion {
            version: shell.format_version,
        });
    }
    validate_text(ShellTextField::PackageName, &shell.package.name)?;
    validate_text(ShellTextField::PackageVersion, &shell.package.version)?;
    let version = shell
        .compiler
        .strip_prefix('=')
        .ok_or(ShellValidationError::CompilerPinPrefix)?;
    semver::Version::parse(version)
        .map_err(|source| ShellValidationError::CompilerPinVersion { source })?;
    if !is_lowercase_hex_sha256(&shell.archive_sha256) {
        return Err(ShellValidationError::ArchiveSha256);
    }
    validate_strict_order(
        shell.modules.iter().map(|module| module.module.as_str()),
        || ShellCollection::Modules,
    )?;
    for module in &shell.modules {
        validate_text(ShellTextField::ModuleName, &module.module)?;
        validate_strict_order(
            module.exports.iter().map(|symbol| symbol.name.as_str()),
            || ShellCollection::Exports {
                module: module.module.clone(),
            },
        )?;
        for symbol in &module.exports {
            validate_text(ShellTextField::ExportName, &symbol.name)?;
            validate_collection_obligations_for_type_repr(
                symbol.type_repr.as_deref(),
                &symbol.collection_obligations,
                stack,
            )
            .map_err(|source| ShellValidationError::CollectionObligations {
                module: module.module.clone(),
                symbol: symbol.name.clone(),
                source,
            })?;
            let variables = match &symbol.type_repr {
                Some(type_repr) => Some(
                    canonical_variables(type_repr, true, stack)
                        .map_err(|source| ShellValidationError::TypeRepresentation { source })?,
                ),
                None if symbol.kind == SymbolKind::Value => {
                    return Err(ShellValidationError::MissingValueType {
                        module: module.module.clone(),
                        symbol: symbol.name.clone(),
                    });
                }
                None => None,
            };
            let mut previous = None;
            for restriction in &symbol.type_variable_restrictions {
                let index = restriction_index(&restriction.variable)
                    .map_err(|source| ShellValidationError::RestrictionIdentity { source })?;
                if previous.is_some_and(|prior| prior >= index) {
                    return Err(ShellValidationError::RestrictionOrder {
                        module: module.module.clone(),
                        symbol: symbol.name.clone(),
                    });
                }
                previous = Some(index);
            }
            if !symbol.type_variable_restrictions.is_empty() && symbol.type_repr.is_none() {
                return Err(ShellValidationError::RestrictionsWithoutType {
                    module: module.module.clone(),
                    symbol: symbol.name.clone(),
                });
            }
            if let Some(variables) = variables {
                for restriction in &symbol.type_variable_restrictions {
                    let index = restriction_index(&restriction.variable)
                        .map_err(|source| ShellValidationError::RestrictionIdentity { source })?;
                    if !variables.types.contains(&index) {
                        return Err(ShellValidationError::RestrictionAbsentFromType {
                            module: module.module.clone(),
                            symbol: symbol.name.clone(),
                            variable: restriction.variable.clone(),
                        });
                    }
                }
            }
            for effect in &symbol.effects {
                validate_text(ShellTextField::EffectName, effect)?;
            }
        }
    }
    validate_strict_order(
        shell
            .dependencies
            .iter()
            .map(|dependency| dependency.name.as_str()),
        || ShellCollection::Dependencies,
    )?;
    for dependency in &shell.dependencies {
        validate_text(ShellTextField::DependencyName, &dependency.name)?;
        if !dependency.version.is_empty() && dependency.version.trim() != dependency.version {
            return Err(ShellValidationError::DependencyVersionWhitespace {
                dependency: dependency.name.clone(),
            });
        }
    }
    Ok(())
}

fn validate_text(field: ShellTextField, value: &str) -> Result<(), ShellValidationError> {
    if value.is_empty() || value.trim() != value {
        return Err(ShellValidationError::InvalidText { field });
    }
    Ok(())
}

fn validate_strict_order<'a>(
    values: impl Iterator<Item = &'a str>,
    collection: impl FnOnce() -> ShellCollection,
) -> Result<(), ShellValidationError> {
    let mut previous = None;
    for value in values {
        if previous.is_some_and(|prior| prior >= value) {
            return Err(ShellValidationError::StrictOrder {
                collection: collection(),
            });
        }
        previous = Some(value);
    }
    Ok(())
}

fn restriction_index(value: &str) -> Result<u32, RestrictionIdentityError> {
    let Some(index) = value.strip_prefix('t') else {
        return Err(RestrictionIdentityError::Spelling);
    };
    if index.is_empty()
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || (index.len() > 1 && index.starts_with('0'))
    {
        return Err(RestrictionIdentityError::Spelling);
    }
    index
        .parse()
        .map_err(|_| RestrictionIdentityError::IndexOverflow)
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
    stack: &impl StackServices,
) -> Result<CanonicalVariables, TypeRepresentationError> {
    let expression = chelis_deep::parse_and_stamp_type(type_repr, stack).map_err(|source| {
        TypeRepresentationError::InvalidDeep {
            source: Box::new(source),
        }
    })?;
    expression_variables(&expression, require_contiguous)
}

fn expression_variables(
    expression: &chelis_deep::Expr,
    require_contiguous: bool,
) -> Result<CanonicalVariables, TypeRepresentationError> {
    fn visit(
        expr: &chelis_deep::Expr,
        variables: &mut CanonicalVariables,
    ) -> Result<(), TypeRepresentationError> {
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
                        return Err(TypeRepresentationError::MissingCanonicalIdentity { tag });
                    };
                    namespace.insert(canonical_variable_index(name, prefix, tag)?);
                }
                for child in node.children_slice() {
                    visit(child, variables)?;
                }
            }
            Expr::Map(..) | Expr::MetaExpr(..) | Expr::BareList(..) | Expr::UnknownForm(..) => {
                return Err(TypeRepresentationError::NonTypeCarrier);
            }
        }
        Ok(())
    }
    let mut variables = CanonicalVariables::default();
    visit(expression, &mut variables)?;
    if require_contiguous {
        for (namespace, prefix, indices) in [
            ("type", "t", &variables.types),
            ("dimension", "d", &variables.dimensions),
            ("rank", "r", &variables.ranks),
        ] {
            for (expected, actual) in indices.iter().copied().enumerate() {
                if actual != expected as u32 {
                    return Err(TypeRepresentationError::Noncontiguous { namespace, prefix });
                }
            }
        }
    }
    Ok(variables)
}

fn canonical_variable_index(
    value: &str,
    prefix: &'static str,
    tag: &'static str,
) -> Result<u32, TypeRepresentationError> {
    let Some(index) = value.strip_prefix(prefix) else {
        return Err(TypeRepresentationError::IdentitySpelling { tag, prefix });
    };
    if index.is_empty()
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || (index.len() > 1 && index.starts_with('0'))
    {
        return Err(TypeRepresentationError::IdentitySpelling { tag, prefix });
    }
    index
        .parse()
        .map_err(|_| TypeRepresentationError::IdentityOverflow { tag })
}

/// Validate a collection ledger against its callable type representation.
pub fn validate_collection_obligations_for_type_repr(
    type_repr: Option<&str>,
    obligations: &[CollectionObligation],
    stack: &impl StackServices,
) -> Result<(), CollectionValidationError> {
    if obligations.is_empty() {
        return Ok(());
    }
    validate_collection_obligation_ledger(obligations, stack)?;
    let type_repr = type_repr.ok_or(CollectionValidationError::MissingFunctionType)?;
    let expression = chelis_deep::parse_and_stamp_type(type_repr, stack).map_err(|source| {
        CollectionValidationError::InvalidFunctionDeep {
            source: Box::new(source),
        }
    })?;
    if !matches!(&expression, chelis_deep::Expr::Node(node, _) if node.tag() == chelis_deep::DeepTag::TFn)
    {
        return Err(CollectionValidationError::MissingFunctionType);
    }
    let variables = expression_variables(&expression, true)
        .map_err(|source| CollectionValidationError::TypeRepresentation { source })?;
    for obligation in obligations {
        for carried in obligation.types() {
            let carried_variables = canonical_variables(carried, false, stack)
                .map_err(|source| CollectionValidationError::TypeRepresentation { source })?;
            for (prefix, carried_indices, declared_indices) in [
                ("t", &carried_variables.types, &variables.types),
                ("d", &carried_variables.dimensions, &variables.dimensions),
                ("r", &carried_variables.ranks, &variables.ranks),
            ] {
                for index in carried_indices {
                    if !declared_indices.contains(index) {
                        return Err(CollectionValidationError::MissingContractVariable {
                            builtin: obligation.builtin(),
                            prefix,
                            index: *index,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validate the canonical identity and order of a collection ledger.
pub fn validate_collection_obligation_ledger(
    obligations: &[CollectionObligation],
    stack: &impl StackServices,
) -> Result<(), CollectionValidationError> {
    let mut previous: Option<(&str, Vec<&str>)> = None;
    for obligation in obligations {
        let types = obligation.types();
        for carried in &types {
            let expression =
                chelis_deep::parse_and_stamp_type(carried, stack).map_err(|source| {
                    CollectionValidationError::InvalidCarriedDeep {
                        source: Box::new(source),
                    }
                })?;
            let canonical = chelis_deep::printer::print_expr(&expression);
            if canonical != *carried {
                return Err(CollectionValidationError::NoncanonicalType {
                    supplied: (*carried).into(),
                    canonical,
                });
            }
        }
        let key = (obligation.builtin(), types);
        if previous.as_ref().is_some_and(|prior| prior >= &key) {
            return Err(CollectionValidationError::StrictOrder);
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
