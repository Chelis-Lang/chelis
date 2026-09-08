//! Thread-local opacity context for checker-enforced opaque types
//! (RFC D-CHECK, spec/design/opaque_invariants_rfc.md).
//!
//! Enforcement runs INSIDE inference (post-annotation passes are
//! unsound: type stamps are dropped on Error-typed nodes and `var`s),
//! so the inference hooks need ambient access to:
//!
//! - the current module key and current top-level decl name,
//! - per-module export sets and a top-level binding -> module map
//!   (the sixth rejection, RT-0 C2),
//! - preformatted "exported producers with signatures" strings for
//!   the D-CHECK error contract.
//!
//! The opaque map itself lives on [`crate::adt::AdtDef`]
//! (`opaque` + `defining_module`), which persists through the
//! compiled-context caches; this context carries only the per-run
//! program-shape data. Unlike declaration-local type binders and signatures,
//! opacity is intentionally a whole-check contextual policy; binder/type
//! resolution state remains explicit on `Env` and annotation contexts.
//!
//! When no context is installed (e.g. annotation passes that re-run
//! `infer_top_level` for stamping), every hook is a no-op, so the
//! drivers that install it remain the single source of violations.

use chelis_unord::UnordSet;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::adt::AdtRegistry;
use crate::errors::{CheckError, CheckErrorKind};
use crate::session::DiagnosticSink;
use crate::types::Type;

/// Program-shape opacity metadata. Accumulated per check phase and
/// persisted on `TypeEnvInner` so the stacked library/new-code paths
/// (and their bincode caches) keep the defining module's export
/// information visible when new code is checked against a prebuilt
/// context.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct OpacityModuleMeta {
    /// Lexical per-module export sets: module key -> exported names.
    /// Only lexical `(export ...)` nodes populate this; the reef
    /// package linker historically stripped `Export` decls during
    /// rewrite (survey section 2), and modules absent here are never
    /// the source of a sixth-rejection flag (see `bindings`).
    pub exports: BTreeMap<String, BTreeSet<String>>,
    /// Top-level binding name -> defining module key, for bindings
    /// declared inside lexical module wrappers. Used by the sixth
    /// rejection; names absent from this map are never flagged
    /// (fail-open for unattributable names, so package-linked
    /// exported producers stay callable).
    pub bindings: BTreeMap<String, String>,
    /// Opaque ADT name -> formatted producer entries
    /// ("name: sig"), unioned across phases, for error text.
    pub producer_entries: BTreeMap<String, BTreeSet<String>>,
}

impl OpacityModuleMeta {
    /// Union `other` into `self` (set union per key; `other` wins on
    /// binding collisions, which only happen for redeclared names).
    pub(crate) fn merge_from(&mut self, other: &OpacityModuleMeta) {
        for (module, names) in &other.exports {
            self.exports
                .entry(module.clone())
                .or_default()
                .extend(names.iter().cloned());
        }
        for (name, module) in &other.bindings {
            self.bindings.insert(name.clone(), module.clone());
        }
        for (adt, entries) in &other.producer_entries {
            self.producer_entries
                .entry(adt.clone())
                .or_default()
                .extend(entries.iter().cloned());
        }
    }
}

/// Per-run opacity context installed by the inference drivers after
/// declaration collection.
#[derive(Debug, Default, Clone)]
pub(crate) struct OpacityContextData {
    /// Module key the item currently being inferred belongs to.
    pub current_module: Option<String>,
    /// Name of the top-level decl currently being inferred (message
    /// location context per the D-CHECK error contract).
    pub current_decl: Option<String>,
    /// Program-shape metadata (exports, bindings, producer text).
    pub meta: OpacityModuleMeta,
}

impl OpacityContextData {
    pub(crate) fn from_meta(meta: OpacityModuleMeta) -> Self {
        Self {
            current_module: None,
            current_decl: None,
            meta,
        }
    }
}

thread_local! {
    static OPACITY_CONTEXT: RefCell<Option<OpacityContextData>> = const { RefCell::new(None) };
    /// RT-1 F2-bypass / RFC v5: in-process provenance flag, TRUE only
    /// while checking decls produced by the reef LINKER (which is the
    /// sole legitimate producer of the `Pkg__`/`pkg__` internal-name
    /// format). FALSE by default, so raw `.ch`/`.dp` ingestion rejects
    /// the linker name format as a forged module identity. Set by an
    /// install-guard at the link boundary; the linker feeds linked
    /// Deep to the checker in-process, so the flag is never lost to
    /// serialization.
    static LINKED_PROGRAM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Install the linked-program provenance flag for the duration of the
/// returned guard (RFC v5). Call this at the reef link boundary before
/// running the checker over linker-mangled decls; nested installs are
/// saved and restored. See [`linked_program`].
pub fn install_linked_program_guard() -> LinkedProgramGuard {
    let previous = LINKED_PROGRAM.with(|cell| cell.replace(true));
    LinkedProgramGuard { previous }
}

pub struct LinkedProgramGuard {
    previous: bool,
}

impl Drop for LinkedProgramGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        LINKED_PROGRAM.with(|cell| cell.set(previous));
    }
}

/// Whether the current check is running over reef-linked decls
/// (RFC v5). When FALSE, the linker name format is rejected as a
/// forged declaration; when TRUE, the linker's own output is accepted.
pub(crate) fn linked_program() -> bool {
    LINKED_PROGRAM.with(std::cell::Cell::get)
}

/// Install `data` for the duration of the returned guard, restoring
/// the previous value (typically `None`) on drop so nested or
/// re-entrant inference passes do not leak state.
pub(crate) fn install_opacity_context(data: OpacityContextData) -> OpacityGuard {
    let previous = OPACITY_CONTEXT.with(|cell| cell.borrow_mut().replace(data));
    OpacityGuard { previous }
}

pub(crate) struct OpacityGuard {
    previous: Option<OpacityContextData>,
}

impl Drop for OpacityGuard {
    fn drop(&mut self) {
        let restored = self.previous.take();
        OPACITY_CONTEXT.with(|cell| *cell.borrow_mut() = restored);
    }
}

/// Update the per-item position (module key + decl name) as the
/// driver loop advances. No-op when no context is installed.
pub(crate) fn set_current_item(module: Option<String>, decl: Option<String>) {
    OPACITY_CONTEXT.with(|cell| {
        if let Some(data) = cell.borrow_mut().as_mut() {
            data.current_module = module;
            data.current_decl = decl;
        }
    });
}

/// Run `f` against the installed context, or return `None` when no
/// context is installed (enforcement disabled for this pass).
pub(crate) fn with_context<R>(f: impl FnOnce(&OpacityContextData) -> R) -> Option<R> {
    OPACITY_CONTEXT.with(|cell| cell.borrow().as_ref().map(f))
}

// ── Inference hooks (RFC D-CHECK rejection set) ──────────────────

/// Core rejection: `adt_name` was constructed/inspected via `action`
/// in the current module. Pushes an `OpaqueTypeViolation` and returns
/// true when the ADT is opaque and the current module is not its
/// defining module. No-op without an installed context (annotation
/// passes) so the drivers stay the single source of violations.
pub(crate) fn check_opaque_use(
    action: OpaqueAction,
    adt_name: &str,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let Some(def) = adt_reg.lookup(adt_name) else {
        return false;
    };
    if !def.opaque {
        return false;
    }
    // A `None` defining module already carried the named-module
    // declaration error; do not cascade per-use violations onto it.
    let Some(defining) = def.defining_module.as_deref() else {
        return false;
    };
    let error = with_context(|ctx| {
        if ctx.current_module.as_deref() == Some(defining) {
            None
        } else {
            Some(violation_error(ctx, action, adt_name, defining))
        }
    })
    .flatten();
    match error {
        Some(error) => {
            errors.push(error);
            true
        }
        None => false,
    }
}

/// Bare-constructor-reference rejection (the constructor binding
/// itself is hidden): fires when `name` resolves to a constructor of
/// an out-of-module opaque ADT. Positional applications instantiate their
/// constructor directly from the registry and therefore do not route their
/// callee through this bare-reference hook.
pub(crate) fn check_ctor_reference(
    name: &str,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let Some((adt_name, _variant)) = adt_reg
        .lookup_variant(name)
        .or_else(|| adt_reg.lookup_variant_terminal_unique(name))
    else {
        return false;
    };
    let adt_name = adt_name.to_string();
    check_opaque_use(OpaqueAction::CtorReference, &adt_name, adt_reg, errors)
}

/// Terminal segment of a binding name: the tail after the last reef
/// internal-name `__` separator or `.` module separator. Bare lexical
/// names have no separator, so the terminal is the name itself.
fn terminal_segment(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

/// Resolve a reference name to its canonical binding key in `bindings`
/// (RT-1 F1): reef leaves un-imported cross-module references at their
/// bare terminal name while the opacity metadata is keyed by the
/// internal (mangled) name, so an exact lookup misses. Mirror
/// inference's `lookup_terminal_unique`: prefer an exact match, then
/// fall back to the UNIQUE binding whose terminal segment equals the
/// reference's terminal segment. Terminal ambiguity yields `None`
/// (fail-open) -- which matches inference, since an ambiguous bare
/// reference does not resolve cleanly there either.
fn resolve_binding_key<'a>(name: &str, bindings: &'a BTreeMap<String, String>) -> Option<&'a str> {
    if let Some((key, _)) = bindings.get_key_value(name) {
        return Some(key.as_str());
    }
    let target = terminal_segment(name);
    let mut matches = bindings
        .keys()
        .filter(|key| terminal_segment(key) == target);
    let first = matches.next()?;
    matches.next().is_none().then_some(first.as_str())
}

/// The sixth rejection (RT-0 C2, broadened in RFC v3): a reference to
/// an unexported binding of an opaque-defining module whose signature
/// mentions the opaque type. `ty` is the reference's resolved type
/// (declared or inferred scheme instantiation). Names without module
/// attribution in the context metadata are never flagged (fail-open
/// for unattributable names; see `OpacityModuleMeta::bindings`).
pub(crate) fn check_unexported_reference(
    name: &str,
    ty: &Type,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let error = with_context(|ctx| {
        // RT-1 F1: canonicalize the reference to its binding key
        // (handles reef bare un-imported references), then attribute.
        let binding_key = resolve_binding_key(name, &ctx.meta.bindings)?;
        let def_mod = ctx.meta.bindings.get(binding_key)?;
        if ctx.current_module.as_deref() == Some(def_mod.as_str()) {
            return None;
        }
        if ctx
            .meta
            .exports
            .get(def_mod)
            .is_some_and(|set| set.contains(binding_key))
        {
            return None;
        }
        // Deterministic order: smallest opaque type name first.
        let mut opaque_types: Vec<&str> = adt_reg
            .defs
            .values()
            .filter(|def| def.opaque && def.defining_module.as_deref() == Some(def_mod.as_str()))
            .map(|def| def.name.as_str())
            .collect();
        opaque_types.sort_unstable();
        for type_name in opaque_types {
            if type_mentions_adt(ty, type_name, adt_reg) {
                return Some(unexported_reference_error(
                    ctx,
                    binding_key,
                    type_name,
                    def_mod,
                ));
            }
        }
        None
    })
    .flatten();
    match error {
        Some(error) => {
            errors.push(error);
            true
        }
        None => false,
    }
}

// ── Module identity (D-CHECK `module_key`) ───────────────────────

/// The module key for a top-level item: the lexical wrapper key wins;
/// otherwise the reef internal-name stem of the item's own name;
/// otherwise `None` (top-level outside any module).
pub(crate) fn module_key_for_item(
    lexical: Option<&str>,
    item_name: Option<&str>,
) -> Option<String> {
    if let Some(key) = lexical {
        return Some(key.to_string());
    }
    item_name.and_then(reef_module_stem)
}

/// Parse the reef package-linker internal-name encoding
/// (`Pkg__<pkg>__<Module>__<Name>` and its lowercase `pkg__` twin,
/// `chelis-reef::internal_name`) into a module key: the stem between
/// the marker and the trailing name, with `__` separators rendered as
/// `.` (e.g. `Pkg__opq__Demo__Types__Probability` -> `opq.Demo.Types`).
pub(crate) fn reef_module_stem(name: &str) -> Option<String> {
    let stem = name
        .strip_prefix("Pkg__")
        .or_else(|| name.strip_prefix("pkg__"))?;
    let (module_part, _terminal) = stem.rsplit_once("__")?;
    if module_part.is_empty() {
        return None;
    }
    Some(module_part.replace("__", "."))
}

// ── De-mangle reef internal names for display (RT-1 F3) ──────────

/// Best-effort de-mangle of a reef internal identifier for display in
/// a violation message (RFC v4c). The package linker rewrites names to
/// `Pkg__<pkg>__<Module>__<Name>` (or the lowercase `pkg__` twin); the
/// user-facing name is the trailing `<Name>` segment. Lexical
/// (non-reef) identifiers carry no marker prefix and pass through
/// unchanged.
///
/// Public so the eval value renderer (chelis-compiler-api) shows the
/// user-facing constructor name rather than the internal mangled form,
/// matching the de-mangling already applied to diagnostics (chelis#399).
pub fn demangle_ident(name: &str) -> String {
    if name.starts_with("Pkg__") || name.starts_with("pkg__") {
        terminal_segment(name).to_string()
    } else {
        name.to_string()
    }
}

/// Best-effort de-mangle of a module key for display (RFC v4c). A reef
/// module key is `<package>.<Module.Path>` where the module path
/// segments are PascalCase (TypeIdent, spec §3.1) and the package
/// segments are lowercase; drop the leading package segments. A
/// lexical module key is fully lowercased by the Surf desugar
/// (`module Stats.Prob` -> `stats.prob`) and is shown as-is, so a key
/// with no PascalCase segment is treated as lexical.
pub(crate) fn demangle_module(key: &str) -> String {
    let segments: Vec<&str> = key.split('.').collect();
    let has_pascal = segments
        .iter()
        .any(|seg| seg.chars().next().is_some_and(|c| c.is_ascii_uppercase()));
    if !has_pascal {
        return key.to_string();
    }
    let module: Vec<&str> = segments
        .iter()
        .copied()
        .skip_while(|seg| seg.chars().next().is_none_or(|c| !c.is_ascii_uppercase()))
        .collect();
    if module.is_empty() {
        key.to_string()
    } else {
        module.join(".")
    }
}

/// De-mangle every ADT name inside a type for display (RT-1 F3), so a
/// producer signature renders `(f32) -> Probability` rather than
/// `(f32) -> Pkg__opq__Demo__Types__Probability`.
pub(crate) fn demangle_type(ty: &Type) -> Type {
    match ty {
        Type::Adt(name, args) => Type::Adt(
            demangle_ident(name),
            args.iter().map(demangle_type).collect(),
        ),
        Type::KindedAdt(name, args) => Type::KindedAdt(
            demangle_ident(name),
            args.iter()
                .map(|argument| match argument {
                    crate::types::NominalArg::Type(ty) => {
                        crate::types::NominalArg::Type(demangle_type(ty))
                    }
                    crate::types::NominalArg::Dimension(dim) => {
                        crate::types::NominalArg::Dimension(dim.clone())
                    }
                })
                .collect(),
        ),
        Type::Fn(args, ret) => Type::Fn(
            args.iter().map(demangle_type).collect(),
            Box::new(demangle_type(ret)),
        ),
        Type::Tuple(items) => Type::Tuple(items.iter().map(demangle_type).collect()),
        Type::Ref(inner) => Type::Ref(Box::new(demangle_type(inner))),
        other => other.clone(),
    }
}

// ── Mentions-T containment (RFC v3 sixth rejection) ──────────────

/// True when `ty` mentions the ADT named `target`, with containment
/// chased through named type definitions in the registry: an
/// unexported `helper() -> WrapRec` where the non-opaque `WrapRec`
/// carries a `Probability` field mentions `Probability`.
pub(crate) fn type_mentions_adt(ty: &Type, target: &str, adt_reg: &AdtRegistry) -> bool {
    let mut seen = UnordSet::new();
    mentions_inner(ty, target, adt_reg, &mut seen)
}

fn mentions_inner(
    ty: &Type,
    target: &str,
    adt_reg: &AdtRegistry,
    seen: &mut UnordSet<String>,
) -> bool {
    match ty {
        Type::Adt(name, args) => {
            if name == target {
                return true;
            }
            if args
                .iter()
                .any(|a| mentions_inner(a, target, adt_reg, seen))
            {
                return true;
            }
            if !seen.insert(name.clone()) {
                return false;
            }
            adt_reg.lookup(name).is_some_and(|def| {
                def.variants.iter().any(|variant| {
                    variant
                        .fields
                        .iter()
                        .any(|(_, fty)| mentions_inner(fty, target, adt_reg, seen))
                })
            })
        }
        Type::KindedAdt(name, args) => {
            if name == target {
                return true;
            }
            if args.iter().any(|argument| {
                argument
                    .as_type()
                    .is_some_and(|ty| mentions_inner(ty, target, adt_reg, seen))
            }) {
                return true;
            }
            if !seen.insert(name.clone()) {
                return false;
            }
            adt_reg.lookup(name).is_some_and(|def| {
                def.variants.iter().any(|variant| {
                    variant
                        .fields
                        .iter()
                        .any(|(_, fty)| mentions_inner(fty, target, adt_reg, seen))
                })
            })
        }
        Type::Fn(args, ret) => {
            args.iter()
                .any(|a| mentions_inner(a, target, adt_reg, seen))
                || mentions_inner(ret, target, adt_reg, seen)
        }
        Type::Ref(inner) => mentions_inner(inner, target, adt_reg, seen),
        Type::Tuple(items) => items
            .iter()
            .any(|t| mentions_inner(t, target, adt_reg, seen)),
        Type::Var(_) | Type::Prim(_) | Type::Tensor(_, _) | Type::Unit | Type::Error(_) => false,
    }
}

// ── D-CHECK error contract ───────────────────────────────────────

/// The action vocabulary for the pinned violation message shape (see
/// `crates/chelis-types/tests/opaque_types.rs` for the byte-exact
/// contract tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpaqueAction {
    RecordConstruction,
    CtorApplication,
    CtorReference,
    PatRecord,
    PatCtor,
    FieldAccess,
    RecordUpdate,
    CastInto,
    CastOut,
    LitForge,
}

impl OpaqueAction {
    fn phrase(self) -> &'static str {
        match self {
            OpaqueAction::RecordConstruction => "record construction of",
            OpaqueAction::CtorApplication => "constructor application of",
            OpaqueAction::CtorReference => "constructor reference to",
            OpaqueAction::PatRecord => "record pattern match on",
            OpaqueAction::PatCtor => "constructor pattern match on",
            OpaqueAction::FieldAccess => "field access on",
            OpaqueAction::RecordUpdate => "record update of",
            OpaqueAction::CastInto => "cast into",
            OpaqueAction::CastOut => "cast out of",
            OpaqueAction::LitForge => "literal ascription to",
        }
    }
}

fn location_context(data: &OpacityContextData) -> String {
    match &data.current_decl {
        Some(decl) => format!("in def `{}`", demangle_ident(decl)),
        None => "at top level".to_string(),
    }
}

fn producers_for(data: &OpacityContextData, type_name: &str) -> String {
    match data.meta.producer_entries.get(type_name) {
        Some(entries) if !entries.is_empty() => {
            entries.iter().cloned().collect::<Vec<_>>().join(", ")
        }
        _ => "none".to_string(),
    }
}

/// Build the pinned violation error for a construction/inspection
/// rejection (everything except the sixth rejection).
pub(crate) fn violation_error(
    data: &OpacityContextData,
    action: OpaqueAction,
    type_name: &str,
    defining_module: &str,
) -> CheckError {
    // RT-1 F3: de-mangle reef internal names for display. The
    // producer-entry text is stored de-mangled at build time.
    let producers = producers_for(data, type_name);
    let type_name = demangle_ident(type_name);
    let defining_module = demangle_module(defining_module);
    CheckError::new(
        CheckErrorKind::OpaqueTypeViolation,
        format!(
            "{}: {} opaque type `{}` outside its defining module `{}`; exported producers \
             of `{}`: {}",
            location_context(data),
            action.phrase(),
            type_name,
            defining_module,
            defining_module,
            producers
        ),
        vec![format!(
            "obtain `{type_name}` values through the exported producers of `{defining_module}`"
        )],
    )
}

/// Build the pinned violation error for the sixth rejection: an
/// out-of-module reference to an unexported binding of the defining
/// module whose signature mentions the opaque type.
pub(crate) fn unexported_reference_error(
    data: &OpacityContextData,
    binding: &str,
    type_name: &str,
    defining_module: &str,
) -> CheckError {
    // RT-1 F3: de-mangle reef internal names for display.
    let producers = producers_for(data, type_name);
    let binding = demangle_ident(binding);
    let type_name = demangle_ident(type_name);
    let defining_module = demangle_module(defining_module);
    CheckError::new(
        CheckErrorKind::OpaqueTypeViolation,
        format!(
            "{}: reference to unexported binding `{}` of module `{}` whose signature \
             mentions opaque type `{}`; exported producers of `{}`: {}",
            location_context(data),
            binding,
            defining_module,
            type_name,
            defining_module,
            producers
        ),
        vec![format!(
            "obtain `{type_name}` values through the exported producers of `{defining_module}`"
        )],
    )
}

/// Build the fail-closed violation for a deferred `access` /
/// `record-update` whose target type was NEVER pinned by the end of
/// the def (let-generalization makes an unannotated accessor lambda
/// polymorphic, so a later caller instantiates fresh variables and
/// the original target can launder an opaque value through it).
/// Mirrors the deferred-borrow ledger's never-pinned rejection; only
/// emitted when the check unit declares at least one opaque type.
pub(crate) fn unresolved_target_error(
    data: &OpacityContextData,
    action: OpaqueAction,
) -> CheckError {
    CheckError::new(
        CheckErrorKind::OpaqueTypeViolation,
        format!(
            "{}: {} an unresolved target type cannot be verified against opaque type \
             boundaries; annotate the target so the checker can resolve it",
            location_context(data),
            action.phrase(),
        ),
        vec!["add a type annotation to the accessed parameter or binding".to_string()],
    )
}

/// True when `name` matches the reef linker's internal-name format
/// (`Pkg__<pkg>__<Module>__<Name>` / lowercase twin) -- i.e. it
/// parses as a module-stem-bearing mangled name. RFC v5: this format
/// is the linker's PRIVATE output; a hand-authored program using it
/// forges module identity through the stem channel.
///
/// Public so the reef package linker shares this single definition of
/// the reserved-name predicate (CR-7); reef rejects the format on
/// user-authored entry/test decls (RFC v6) using exactly this rule.
pub fn is_linker_format_name(name: &str) -> bool {
    reef_module_stem(name).is_some()
}

/// Build the declaration error for a forged reef linker name in a
/// program not produced by the linker (RFC v5).
pub(crate) fn forged_linker_name_error(name: &str) -> CheckError {
    CheckError::new(
        CheckErrorKind::ReservedLinkerName,
        format!(
            "`{name}` uses the reef package-linker's reserved internal-name format \
             (`Pkg__`/`pkg__`...), which only the linker may produce; rename the \
             declaration"
        ),
        vec![
            "the `Pkg__pkg__Module__Name` format is the linker's private output; \
             hand-authored Surf and Deep must use ordinary identifiers"
                .to_string(),
        ],
    )
}

/// Build the declaration error for `@opaque` outside a named module
/// (D-CHECK, RT-0 M6).
pub(crate) fn unmoduled_opaque_error(type_name: &str) -> CheckError {
    CheckError::new(
        CheckErrorKind::OpaqueTypeViolation,
        format!("@opaque type `{type_name}` requires a named enclosing module"),
        vec![
            "wrap the declaration in a named `module` so the enforcement boundary is unambiguous"
                .to_string(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reef_stem_parses_both_marker_cases() {
        assert_eq!(
            reef_module_stem("Pkg__opq__Demo__Types__Probability"),
            Some("opq.Demo.Types".to_string())
        );
        assert_eq!(
            reef_module_stem("pkg__opq__Demo__Types__probability"),
            Some("opq.Demo.Types".to_string())
        );
    }

    #[test]
    fn reef_stem_rejects_unmangled_names() {
        assert_eq!(reef_module_stem("Probability"), None);
        assert_eq!(reef_module_stem("probability"), None);
        assert_eq!(reef_module_stem("Pkg__lonely"), None);
    }

    #[test]
    fn module_key_prefers_lexical_wrapper() {
        assert_eq!(
            module_key_for_item(
                Some("stats.prob"),
                Some("Pkg__opq__Demo__Types__Probability")
            ),
            Some("stats.prob".to_string())
        );
        assert_eq!(
            module_key_for_item(None, Some("Pkg__opq__Demo__Types__Probability")),
            Some("opq.Demo.Types".to_string())
        );
        assert_eq!(module_key_for_item(None, Some("plain_name")), None);
        assert_eq!(module_key_for_item(None, None), None);
    }

    #[test]
    fn demangle_ident_strips_marker_else_passthrough() {
        assert_eq!(
            demangle_ident("Pkg__opq__Demo__Types__Probability"),
            "Probability"
        );
        assert_eq!(
            demangle_ident("pkg__opq__Demo__Types__raw_make"),
            "raw_make"
        );
        // Lexical (unmangled) identifiers pass through unchanged.
        assert_eq!(demangle_ident("Probability"), "Probability");
        assert_eq!(demangle_ident("raw_make"), "raw_make");
    }

    #[test]
    fn demangle_module_strips_package_for_reef_keeps_lexical() {
        // Reef key: drop the lowercase package segment(s), keep the
        // PascalCase module path.
        assert_eq!(demangle_module("opq.Demo.Types"), "Demo.Types");
        assert_eq!(demangle_module("chelis.std.Std.Test"), "Std.Test");
        // Lexical (Surf-lowercased) key has no PascalCase segment and
        // is shown as-is, so existing byte-exact messages are stable.
        assert_eq!(demangle_module("stats.prob"), "stats.prob");
    }

    #[test]
    fn demangle_type_rewrites_adt_names_in_signatures() {
        use crate::types::Prim;
        let ty = Type::Fn(
            vec![Type::Prim(Prim::F32)],
            Box::new(Type::Adt(
                "Pkg__opq__Demo__Types__Probability".to_string(),
                vec![],
            )),
        );
        assert_eq!(demangle_type(&ty).to_string(), "(f32) -> Probability");
    }

    #[test]
    fn is_linker_format_name_matches_only_full_mangled_names() {
        // RFC v5: the linker format requires the marker prefix AND a
        // non-empty module stem before the terminal.
        assert!(is_linker_format_name("Pkg__foo__Secret"));
        assert!(is_linker_format_name("pkg__foo__forge"));
        assert!(is_linker_format_name(
            "Pkg__chelis__std__Std__Decimal__RoundingMode"
        ));
        // Marker prefix but no stem -> not a complete mangled name.
        assert!(!is_linker_format_name("Pkg__lonely"));
        // Ordinary user identifiers pass through (CR-7 borderline:
        // single-underscore names are not the double-underscore
        // linker format).
        assert!(!is_linker_format_name("Probability"));
        assert!(!is_linker_format_name("forge"));
        assert!(!is_linker_format_name("my_pkg_thing"));
        assert!(!is_linker_format_name("pkg_count"));
    }

    #[test]
    fn linked_program_flag_defaults_false_and_guard_scopes() {
        assert!(!linked_program(), "default provenance is not-linked");
        {
            let _guard = install_linked_program_guard();
            assert!(linked_program(), "guard sets the linked flag");
            {
                let _nested = install_linked_program_guard();
                assert!(linked_program());
            }
            assert!(linked_program(), "nested guard restores TRUE on drop");
        }
        assert!(!linked_program(), "guard restores FALSE on drop");
    }
}
