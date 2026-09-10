//! Pinned rustc-private discovery, compiled directly by capacity_census_wire_calls.
//! This is a receipt primitive, not an assertion that any discovered codec is safe.
#![feature(rustc_private)]
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_span;
extern crate rustc_type_ir;

use rustc_driver::{Callbacks, Compilation};
use rustc_hir::def::DefKind;
use rustc_interface::interface;
use rustc_middle::mir::{self, visit::Visitor};
use rustc_middle::ty::{self, GenericArgsRef, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::Span;
use rustc_span::def_id::{CRATE_DEF_INDEX, DefId};
use std::collections::{BTreeSet, VecDeque};

// DefId deliberately has no Ord. Use its compiler-assigned crate/index pair;
// diagnostic spelling and hash iteration never define traversal order.
#[derive(Clone, Copy, Eq, PartialEq)]
struct OrderedDef(DefId);
impl Ord for OrderedDef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0.krate.as_u32(), self.0.index.as_u32())
            .cmp(&(other.0.krate.as_u32(), other.0.index.as_u32()))
    }
}
impl PartialOrd for OrderedDef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
#[derive(Default)]
struct DefSet(BTreeSet<OrderedDef>);
impl DefSet {
    fn new() -> Self {
        Self::default()
    }
    fn insert(&mut self, id: DefId) -> bool {
        self.0.insert(OrderedDef(id))
    }
    fn contains(&self, id: &DefId) -> bool {
        self.0.contains(&OrderedDef(*id))
    }
    fn iter(&self) -> impl Iterator<Item = &DefId> {
        self.0.iter().map(|id| &id.0)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl FromIterator<DefId> for DefSet {
    fn from_iter<T: IntoIterator<Item = DefId>>(iter: T) -> Self {
        Self(iter.into_iter().map(OrderedDef).collect())
    }
}

// No dependency on a second serde build inside the compiler process.
fn quoted(s: impl AsRef<str>) -> String {
    let mut out = String::from("\"");
    for c in s.as_ref().chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn array(values: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", values.into_iter().collect::<Vec<_>>().join(","))
}
fn object(values: &[(&str, String)]) -> String {
    format!(
        "{{{}}}",
        values
            .iter()
            .map(|(k, v)| format!("{}:{v}", quoted(k)))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn definition(tcx: TyCtxt<'_>, id: DefId) -> String {
    object(&[
        (
            "def_id",
            quoted(format!("{}:{}", id.krate.as_u32(), id.index.as_u32())),
        ),
        ("crate", quoted(tcx.crate_name(id.krate).as_str())),
        (
            "item_name",
            tcx.opt_item_name(id)
                .map(|name| quoted(name.as_str()))
                .unwrap_or_else(|| "null".into()),
        ),
        (
            "stable_crate_id",
            quoted(format!("{:016x}", tcx.stable_crate_id(id.krate))),
        ),
        (
            "path",
            quoted(tcx.def_path(id).to_string_no_crate_verbose()),
        ),
        (
            "def_path_hash",
            quoted(format!("{:?}", tcx.def_path_hash(id))),
        ),
    ])
}
fn location(tcx: TyCtxt<'_>, span: Span) -> String {
    object(&[
        (
            "span",
            quoted(tcx.sess.source_map().span_to_diagnostic_string(span)),
        ),
        (
            "expansion",
            quoted(format!("{:?}", span.ctxt().outer_expn_data())),
        ),
    ])
}

struct Identities {
    serialize: DefId,
    serializer: DefId,
    dynamic: DefSet,
    decoder_traits: DefSet,
    schema_trait: Option<DefId>,
    deserialize: Option<DefId>,
    conversion_traits: Vec<DefId>,
}
fn external(
    tcx: TyCtxt<'_>,
    name: &str,
    path: &[&str],
    kind: DefKind,
) -> Result<Option<DefId>, String> {
    let crates: Vec<_> = tcx
        .crates(())
        .iter()
        .copied()
        .filter(|c| tcx.crate_name(*c).as_str() == name)
        .collect();
    if crates.len() > 1 {
        return Err(format!("ambiguous {name} defining crate"));
    }
    let Some(krate) = crates.first() else {
        return Ok(None);
    };
    let mut current = DefId {
        krate: *krate,
        index: CRATE_DEF_INDEX,
    };
    for (index, segment) in path.iter().enumerate() {
        let found: DefSet = tcx
            .module_children(current)
            .iter()
            .filter(|child| child.ident.name.as_str() == *segment)
            .filter_map(|child| child.res.opt_def_id())
            .filter(|id| {
                tcx.def_kind(*id)
                    == if index + 1 == path.len() {
                        kind
                    } else {
                        DefKind::Mod
                    }
            })
            .collect();
        if found.len() != 1 {
            return Err(format!("missing or ambiguous {name}::{path:?}"));
        }
        current = *found.iter().next().unwrap();
    }
    Ok(Some(current))
}
impl Identities {
    fn read(tcx: TyCtxt<'_>) -> Result<Self, String> {
        let serialize = external(tcx, "serde_core", &["ser", "Serialize"], DefKind::Trait)?
            .ok_or("missing serde_core")?;
        let serializer = external(tcx, "serde_core", &["ser", "Serializer"], DefKind::Trait)?
            .ok_or("missing serde_core")?;
        if tcx.def_kind(serialize) != DefKind::Trait || tcx.def_kind(serializer) != DefKind::Trait {
            return Err("serde identity is not a compiler trait".into());
        }
        let mut dynamic = DefSet::new();
        let decoder_module = external(tcx, "serde_core", &["de"], DefKind::Mod)?
            .ok_or("missing serde decoder module")?;
        let decoder_traits = tcx
            .module_children(decoder_module)
            .iter()
            .filter_map(|child| child.res.opt_def_id())
            .filter(|id| {
                tcx.def_kind(*id) == DefKind::Trait && tcx.opt_parent(*id) == Some(decoder_module)
            })
            .collect();
        // Semantic carrier identities, not encoder entrypoint names. All aliases
        // and opaque returns normalize to these same compiler definitions.
        for (path, kind) in [
            (&["Value"][..], DefKind::Enum),
            (&["Number"][..], DefKind::Struct),
        ] {
            if let Some(id) = external(tcx, "serde_json", path, kind)? {
                dynamic.insert(id);
            }
        }
        // RawValue is feature-dependent; query exports without requiring it.
        if let Some(krate) = tcx
            .crates(())
            .iter()
            .find(|c| tcx.crate_name(**c).as_str() == "serde_json")
        {
            let root = DefId {
                krate: *krate,
                index: CRATE_DEF_INDEX,
            };
            for child in tcx.module_children(root) {
                if child.ident.name.as_str() == "value" {
                    for value in tcx.module_children(child.res.def_id()) {
                        if value.ident.name.as_str() == "RawValue" {
                            dynamic.insert(value.res.def_id());
                        }
                    }
                }
            }
        }
        let compiler_json = std::env::var("WIRE_CALL_SCOPE").as_deref() == Ok("compiler-json");
        let mut conversion_traits = Vec::new();
        if compiler_json {
            for name in ["IntoPyObject", "FromPyObject"] {
                conversion_traits.push(
                    external(tcx, "pyo3", &[name], DefKind::Trait)?
                        .ok_or("missing defining PyO3 conversion trait")?,
                );
            }
        }
        Ok(Self {
            serialize,
            serializer,
            dynamic,
            decoder_traits,
            schema_trait: external(tcx, "schemars", &["JsonSchema"], DefKind::Trait)?,
            deserialize: if compiler_json {
                Some(
                    external(tcx, "serde_core", &["de", "Deserialize"], DefKind::Trait)?
                        .ok_or("missing defining Deserialize trait")?,
                )
            } else {
                None
            },
            conversion_traits,
        })
    }
}

fn traits<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: DefId,
    args: GenericArgsRef<'tcx>,
) -> Vec<ty::TraitPredicate<'tcx>> {
    let predicates = tcx.predicates_of(def).instantiate(tcx, args).predicates;
    rustc_type_ir::elaborate::elaborate(tcx, predicates.into_iter().map(|p| p.skip_normalization()))
        .filter_map(|p| p.as_trait_clause().map(|p| p.skip_binder()))
        .collect()
}
fn dynamic(identities: &Identities, value: Ty<'_>) -> bool {
    value.walk().any(|arg| {
        arg.as_type().is_some_and(
            |t| matches!(t.kind(), ty::Adt(def, _) if identities.dynamic.contains(&def.did())),
        )
    })
}
fn transport<'tcx>(tcx: TyCtxt<'tcx>, value: Ty<'tcx>, identities: &Identities) -> bool {
    if dynamic(identities, value) {
        return true;
    }
    value.walk().any(|arg| {
        arg.as_type().is_some_and(|t| match t.kind() {
            ty::Str => true,
            ty::Adt(def, args) => {
                tcx.lang_items().string() == Some(def.did())
                    || (tcx.is_diagnostic_item(rustc_span::sym::Vec, def.did())
                        && args.type_at(0) == tcx.types.u8)
            }
            ty::Slice(element) | ty::Array(element, _) => *element == tcx.types.u8,
            _ => false,
        })
    })
}
fn shape<'tcx>(tcx: TyCtxt<'tcx>, value: Ty<'tcx>, depth: usize) -> String {
    let unsupported =
        |reason: &str| object(&[("tag", quoted("unsupported")), ("reason", quoted(reason))]);
    if depth > 128 {
        return unsupported("type argument nesting limit");
    }
    let primitive = |name: &str| object(&[("tag", quoted("primitive")), ("name", quoted(name))]);
    match value.kind() {
        ty::Bool => primitive("bool"),
        ty::Char => primitive("char"),
        ty::Str => primitive("str"),
        ty::Int(kind) => primitive(kind.name_str()),
        ty::Uint(kind) => primitive(kind.name_str()),
        ty::Float(kind) => primitive(kind.name_str()),
        ty::Ref(_, inner, mutable) => object(&[
            ("tag", quoted("reference")),
            (
                "mutable",
                (*mutable == rustc_hir::Mutability::Mut).to_string(),
            ),
            ("inner", shape(tcx, *inner, depth + 1)),
        ]),
        ty::Slice(element) => object(&[
            ("tag", quoted("slice")),
            ("element", shape(tcx, *element, depth + 1)),
        ]),
        ty::Array(element, length) => match length.try_to_target_usize(tcx) {
            Some(length) => object(&[
                ("tag", quoted("array")),
                ("element", shape(tcx, *element, depth + 1)),
                ("length", length.to_string()),
            ]),
            None => unsupported("array length is not a compiler literal"),
        },
        ty::Tuple(elements) => object(&[
            ("tag", quoted("tuple")),
            (
                "elements",
                array(elements.iter().map(|t| shape(tcx, t, depth + 1))),
            ),
        ]),
        ty::Adt(def, args) => object(&[
            ("tag", quoted("nominal")),
            ("definition", definition(tcx, def.did())),
            (
                "arguments",
                array(args.iter().filter_map(|arg| match arg.kind() {
                    ty::GenericArgKind::Type(t) => Some(shape(tcx, t, depth + 1)),
                    // Region identities cannot change a serialized value shape.
                    ty::GenericArgKind::Lifetime(_) => None,
                    ty::GenericArgKind::Const(value) => {
                        Some(match value.try_to_target_usize(tcx) {
                            Some(value) => {
                                object(&[("tag", quoted("const")), ("value", value.to_string())])
                            }
                            None => unsupported("nonliteral or non-usize nominal const argument"),
                        })
                    }
                })),
            ),
        ]),
        ty::Param(parameter) => object(&[
            ("tag", quoted("parameter")),
            ("index", parameter.index.to_string()),
        ]),
        ty::Alias(_, _) => object(&[("tag", quoted("projection"))]),
        _ => unsupported("non-carrier compiler type"),
    }
}
fn typ<'tcx>(tcx: TyCtxt<'tcx>, value: Ty<'tcx>, identities: &Identities) -> String {
    let nominal = if let ty::Adt(def, _) = value.kind() {
        definition(tcx, def.did())
    } else {
        "null".into()
    };
    // Every nominal under reference/container/tuple/alias arguments carries its
    // own canonical identity. Display text is diagnostic, never type authority.
    let nodes = value.walk().filter_map(|arg| arg.as_type()).map(|t| {
        let definition = match t.kind() {
            ty::Adt(def, _) => definition(tcx, def.did()),
            ty::Alias(_, alias) => definition(
                tcx,
                match alias.kind {
                    ty::AliasTyKind::Projection { def_id }
                    | ty::AliasTyKind::Inherent { def_id }
                    | ty::AliasTyKind::Opaque { def_id }
                    | ty::AliasTyKind::Free { def_id } => def_id,
                },
            ),
            ty::FnDef(def, _) | ty::Closure(def, _) | ty::Coroutine(def, _) => {
                definition(tcx, *def)
            }
            _ => "null".into(),
        };
        object(&[
            ("text", quoted(t.to_string())),
            ("kind", quoted(format!("{:?}", t.kind()))),
            ("definition", definition),
        ])
    });
    object(&[
        ("shape", shape(tcx, value, 0)),
        ("text", quoted(value.to_string())),
        ("nominal", nominal),
        ("nodes", array(nodes)),
        ("dynamic", dynamic(identities, value).to_string()),
        ("open", value.has_non_region_param().to_string()),
    ])
}

// Native-binding collection is intentionally independent of serde.  Its type
// records keep the same compiler-derived shape and identity vocabulary while
// omitting the wire-only dynamic-carrier classification.
fn native_typ<'tcx>(tcx: TyCtxt<'tcx>, value: Ty<'tcx>) -> String {
    let nominal = if let ty::Adt(def, _) = value.kind() {
        definition(tcx, def.did())
    } else {
        "null".into()
    };
    let nodes = value.walk().filter_map(|arg| arg.as_type()).map(|t| {
        let definition = match t.kind() {
            ty::Adt(def, _) => definition(tcx, def.did()),
            ty::Alias(_, alias) => definition(
                tcx,
                match alias.kind {
                    ty::AliasTyKind::Projection { def_id }
                    | ty::AliasTyKind::Inherent { def_id }
                    | ty::AliasTyKind::Opaque { def_id }
                    | ty::AliasTyKind::Free { def_id } => def_id,
                },
            ),
            ty::FnDef(def, _) | ty::Closure(def, _) | ty::Coroutine(def, _) => {
                definition(tcx, *def)
            }
            _ => "null".into(),
        };
        object(&[
            ("text", quoted(t.to_string())),
            ("kind", quoted(format!("{:?}", t.kind()))),
            ("definition", definition),
        ])
    });
    object(&[
        ("shape", shape(tcx, value, 0)),
        ("text", quoted(value.to_string())),
        ("nominal", nominal),
        ("nodes", array(nodes)),
        ("open", value.has_non_region_param().to_string()),
    ])
}

fn native_generic_arg<'tcx>(tcx: TyCtxt<'tcx>, arg: ty::GenericArg<'tcx>) -> String {
    match arg.kind() {
        ty::GenericArgKind::Type(value) => {
            object(&[("kind", quoted("type")), ("type", native_typ(tcx, value))])
        }
        ty::GenericArgKind::Lifetime(value) => object(&[
            ("kind", quoted("lifetime")),
            ("text", quoted(format!("{value:?}"))),
        ]),
        ty::GenericArgKind::Const(value) => object(&[
            ("kind", quoted("const")),
            ("text", quoted(format!("{value:?}"))),
        ]),
    }
}

fn native_implementation<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    substitutions: GenericArgsRef<'tcx>,
) -> String {
    // An anonymous const in an impl's trait arguments has the impl as its
    // syntactic parent but does not inherit its generic arguments. Only real
    // associated bodies carry an implementation owner and its substitutions.
    if !matches!(
        tcx.def_kind(owner),
        DefKind::AssocFn | DefKind::AssocConst { .. }
    ) {
        return "null".into();
    }
    let Some(id) = tcx.opt_parent(owner) else {
        return "null".into();
    };
    if !matches!(tcx.def_kind(id), DefKind::Impl { .. }) {
        return "null".into();
    }
    if let Some(trait_ref) = tcx.impl_opt_trait_ref(id) {
        let trait_ref = trait_ref
            .instantiate(tcx, substitutions)
            .skip_normalization();
        object(&[
            ("definition", definition(tcx, id)),
            ("trait", definition(tcx, trait_ref.def_id)),
            ("self_type", native_typ(tcx, trait_ref.self_ty())),
        ])
    } else {
        let self_type = tcx
            .type_of(id)
            .instantiate(tcx, substitutions)
            .skip_normalization();
        object(&[
            ("definition", definition(tcx, id)),
            ("trait", "null".into()),
            ("self_type", native_typ(tcx, self_type)),
        ])
    }
}

fn native_caller<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    substitutions: GenericArgsRef<'tcx>,
) -> String {
    object(&[
        ("definition", definition(tcx, owner)),
        ("kind", quoted(format!("{:?}", tcx.def_kind(owner)))),
        ("ancestors", native_ancestors(tcx, owner)),
        (
            "implementation",
            native_implementation(tcx, owner, substitutions),
        ),
        (
            "substitutions",
            array(substitutions.iter().map(|arg| native_generic_arg(tcx, arg))),
        ),
        (
            "open_type_or_const",
            substitutions.has_non_region_param().to_string(),
        ),
    ])
}

fn native_ancestors(tcx: TyCtxt<'_>, owner: DefId) -> String {
    let mut ancestors = Vec::new();
    let mut parent = tcx.opt_parent(owner);
    while let Some(id) = parent {
        ancestors.push(definition(tcx, id));
        parent = tcx.opt_parent(id);
    }
    array(ancestors)
}

fn native_place<'tcx>(tcx: TyCtxt<'tcx>, place: mir::Place<'tcx>, value: Ty<'tcx>) -> String {
    object(&[
        ("id", quoted(format!("{place:?}"))),
        ("local", place.local.as_usize().to_string()),
        (
            "projection",
            array(
                place
                    .projection
                    .iter()
                    .map(|part| quoted(format!("{part:?}"))),
            ),
        ),
        ("type", native_typ(tcx, value)),
    ])
}

fn native_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    operand: &mir::Operand<'tcx>,
    value: Ty<'tcx>,
) -> String {
    let (kind, place) = match operand {
        mir::Operand::Copy(place) => ("copy", native_place(tcx, *place, value)),
        mir::Operand::Move(place) => ("move", native_place(tcx, *place, value)),
        mir::Operand::Constant(_) => ("constant", "null".into()),
        mir::Operand::RuntimeChecks(_) => ("runtime-checks", "null".into()),
    };
    object(&[
        ("kind", quoted(kind)),
        ("place", place),
        ("type", native_typ(tcx, value)),
    ])
}

fn native_terminator_kind(kind: &mir::TerminatorKind<'_>) -> &'static str {
    match kind {
        mir::TerminatorKind::Goto { .. } => "Goto",
        mir::TerminatorKind::SwitchInt { .. } => "SwitchInt",
        mir::TerminatorKind::UnwindResume => "UnwindResume",
        mir::TerminatorKind::UnwindTerminate(_) => "UnwindTerminate",
        mir::TerminatorKind::Return => "Return",
        mir::TerminatorKind::Unreachable => "Unreachable",
        mir::TerminatorKind::Drop { .. } => "Drop",
        mir::TerminatorKind::Call { .. } => "Call",
        mir::TerminatorKind::TailCall { .. } => "TailCall",
        mir::TerminatorKind::Assert { .. } => "Assert",
        mir::TerminatorKind::Yield { .. } => "Yield",
        mir::TerminatorKind::CoroutineDrop => "CoroutineDrop",
        mir::TerminatorKind::FalseEdge { .. } => "FalseEdge",
        mir::TerminatorKind::FalseUnwind { .. } => "FalseUnwind",
        mir::TerminatorKind::InlineAsm { .. } => "InlineAsm",
    }
}

fn native_rvalue_kind(value: &mir::Rvalue<'_>) -> &'static str {
    match value {
        mir::Rvalue::Use(..) => "Use",
        mir::Rvalue::Repeat(..) => "Repeat",
        mir::Rvalue::Ref(..) => "Ref",
        mir::Rvalue::ThreadLocalRef(..) => "ThreadLocalRef",
        mir::Rvalue::RawPtr(..) => "RawPtr",
        mir::Rvalue::Cast(..) => "Cast",
        mir::Rvalue::BinaryOp(..) => "BinaryOp",
        mir::Rvalue::UnaryOp(..) => "UnaryOp",
        mir::Rvalue::Discriminant(..) => "Discriminant",
        mir::Rvalue::Aggregate(..) => "Aggregate",
        mir::Rvalue::CopyForDeref(..) => "CopyForDeref",
        mir::Rvalue::WrapUnsafeBinder(..) => "WrapUnsafeBinder",
        mir::Rvalue::Reborrow(..) => "Reborrow",
    }
}

struct NativeRvaluePlaces<'tcx> {
    places: Vec<mir::Place<'tcx>>,
}
impl<'tcx> Visitor<'tcx> for NativeRvaluePlaces<'tcx> {
    fn visit_place(
        &mut self,
        place: &mir::Place<'tcx>,
        context: mir::visit::PlaceContext,
        location: mir::Location,
    ) {
        self.places.push(*place);
        self.super_place(place, context, location);
    }
}

struct NativeDiscovery<'tcx> {
    tcx: TyCtxt<'tcx>,
    bodies: BTreeSet<String>,
    calls: BTreeSet<String>,
    aggregates: BTreeSet<String>,
    flows: BTreeSet<String>,
    errors: BTreeSet<String>,
    queue: VecDeque<(DefId, GenericArgsRef<'tcx>)>,
    seen: Vec<(DefId, GenericArgsRef<'tcx>)>,
    owners: DefSet,
}
impl<'tcx> NativeDiscovery<'tcx> {
    fn normalize(&mut self, owner: DefId, args: GenericArgsRef<'tcx>, value: Ty<'tcx>) -> Ty<'tcx> {
        let env = if args.has_non_region_param() {
            ty::TypingEnv::post_analysis(self.tcx, owner)
        } else {
            ty::TypingEnv::fully_monomorphized()
        };
        match self.tcx.try_instantiate_and_normalize_erasing_regions(
            args,
            env,
            ty::EarlyBinder::bind(self.tcx, value),
        ) {
            Ok(value) => value,
            Err(error) => {
                self.errors.insert(format!(
                    "unresolved native type in {}: {error:?}",
                    self.tcx.def_path_str(owner)
                ));
                value
            }
        }
    }

    fn enqueue(&mut self, def: DefId, args: GenericArgsRef<'tcx>) {
        if def.is_local() && self.tcx.is_mir_available(def) && !args.has_non_region_param() {
            self.queue.push_back((def, args));
        }
    }

    fn body(&mut self, owner: DefId, substitutions: GenericArgsRef<'tcx>) {
        let tcx = self.tcx;
        let body = match tcx.def_kind(owner) {
            DefKind::Const { .. }
            | DefKind::AssocConst { .. }
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::Static { .. } => tcx.mir_for_ctfe(owner),
            DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::SyntheticCoroutineBody => {
                tcx.optimized_mir(owner)
            }
            kind => {
                self.errors.insert(format!(
                    "unsupported native body kind {kind:?}: {}",
                    tcx.def_path_str(owner)
                ));
                return;
            }
        };
        let caller = native_caller(tcx, owner, substitutions);
        let formal_inputs: Vec<_> = (1..=body.arg_count)
            .map(|index| {
                let local = mir::Local::from_usize(index);
                let ty = self.normalize(owner, substitutions, body.local_decls[local].ty);
                object(&[("local", index.to_string()), ("type", native_typ(tcx, ty))])
            })
            .collect();
        let result = self.normalize(owner, substitutions, body.return_ty());
        let blocks = body.basic_blocks.iter_enumerated().map(|(block, data)| {
            let term = data.terminator();
            object(&[
                ("block", block.as_usize().to_string()),
                ("cleanup", data.is_cleanup.to_string()),
                ("statements", data.statements.len().to_string()),
                ("terminator", quoted(native_terminator_kind(&term.kind))),
                (
                    "successors",
                    array(term.successors().map(|next| next.as_usize().to_string())),
                ),
                ("source", location(tcx, term.source_info.span)),
            ])
        });
        self.bodies.insert(object(&[
            ("definition", definition(tcx, owner)),
            ("kind", quoted(format!("{:?}", tcx.def_kind(owner)))),
            ("ancestors", native_ancestors(tcx, owner)),
            (
                "implementation",
                native_implementation(tcx, owner, substitutions),
            ),
            (
                "substitutions",
                array(substitutions.iter().map(|arg| native_generic_arg(tcx, arg))),
            ),
            (
                "open_type_or_const",
                substitutions.has_non_region_param().to_string(),
            ),
            ("formal_inputs", array(formal_inputs)),
            ("formal_result", native_typ(tcx, result)),
            ("source", location(tcx, body.span)),
            ("blocks", array(blocks)),
        ]));

        for (block, data) in body.basic_blocks.iter_enumerated() {
            for (statement_index, statement) in data.statements.iter().enumerate() {
                let mir::StatementKind::Assign(assignment) = &statement.kind else {
                    continue;
                };
                let (destination, value) = &**assignment;
                let mir_location = mir::Location {
                    block,
                    statement_index,
                };
                let mut source_places = NativeRvaluePlaces { places: Vec::new() };
                source_places.visit_rvalue(value, mir_location);
                let mut sources = BTreeSet::new();
                for place in source_places.places {
                    let value =
                        self.normalize(owner, substitutions, place.ty(&body.local_decls, tcx).ty);
                    sources.insert(native_place(tcx, place, value));
                }
                let destination_type = self.normalize(
                    owner,
                    substitutions,
                    destination.ty(&body.local_decls, tcx).ty,
                );
                self.flows.insert(object(&[
                    ("caller", caller.clone()),
                    ("block", block.as_usize().to_string()),
                    ("statement", statement_index.to_string()),
                    ("source", location(tcx, statement.source_info.span)),
                    (
                        "destination",
                        native_place(tcx, *destination, destination_type),
                    ),
                    ("rvalue", quoted(native_rvalue_kind(value))),
                    ("sources", array(sources)),
                ]));
                if let mir::Rvalue::Aggregate(kind, operands) = value
                    && let mir::AggregateKind::Adt(def, variant, args, _, active_field) = &**kind
                {
                    let adt = tcx.adt_def(*def);
                    if variant.as_usize() >= adt.variants().len() {
                        self.errors.insert(format!(
                            "invalid native ADT aggregate variant in {} at bb{}[{}]",
                            tcx.def_path_str(owner),
                            block.as_usize(),
                            statement_index,
                        ));
                        continue;
                    }
                    let variant_fields = &adt.variant(*variant).fields;
                    let field_operands: Vec<_> = match active_field {
                        Some(field) => {
                            if !adt.is_union()
                                || operands.len() != 1
                                || field.as_usize() >= variant_fields.len()
                            {
                                self.errors.insert(format!(
                                    "invalid native union aggregate shape in {} at bb{}[{}]",
                                    tcx.def_path_str(owner),
                                    block.as_usize(),
                                    statement_index,
                                ));
                                continue;
                            }
                            let Some(operand) = operands.iter().next() else {
                                self.errors.insert(format!(
                                    "missing native union aggregate operand in {} at bb{}[{}]",
                                    tcx.def_path_str(owner),
                                    block.as_usize(),
                                    statement_index,
                                ));
                                continue;
                            };
                            vec![(&variant_fields[*field], operand)]
                        }
                        None => {
                            if adt.is_union() || variant_fields.len() != operands.len() {
                                self.errors.insert(format!(
                                    "invalid native ADT aggregate shape in {} at bb{}[{}]",
                                    tcx.def_path_str(owner),
                                    block.as_usize(),
                                    statement_index,
                                ));
                                continue;
                            }
                            variant_fields.iter().zip(operands.iter()).collect()
                        }
                    };
                    let mut fields = Vec::new();
                    for (field, operand) in field_operands {
                        let value = self.normalize(
                            owner,
                            substitutions,
                            operand.ty(&body.local_decls, tcx),
                        );
                        fields.push(object(&[
                            ("definition", definition(tcx, field.did)),
                            ("operand", native_operand(tcx, operand, value)),
                        ]));
                    }
                    let mut aggregate_args = Vec::new();
                    for arg in args.iter() {
                        let instantiated = ty::EarlyBinder::bind(tcx, arg)
                            .instantiate(tcx, substitutions)
                            .skip_normalization();
                        aggregate_args.push(native_generic_arg(tcx, instantiated));
                    }
                    self.aggregates.insert(object(&[
                        ("caller", caller.clone()),
                        ("definition", definition(tcx, *def)),
                        ("variant", quoted(adt.variant(*variant).name.as_str())),
                        (
                            "variant_definition",
                            definition(tcx, adt.variant(*variant).def_id),
                        ),
                        ("arguments", array(aggregate_args)),
                        (
                            "destination",
                            native_place(tcx, *destination, destination_type),
                        ),
                        ("fields", array(fields)),
                        ("block", block.as_usize().to_string()),
                        ("statement", statement_index.to_string()),
                        ("source", location(tcx, statement.source_info.span)),
                    ]));
                }
            }
            let term = data.terminator();
            let mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                ..
            } = &term.kind
            else {
                continue;
            };
            let function = self.normalize(owner, substitutions, func.ty(&body.local_decls, tcx));
            let mut callee = "null".into();
            let mut formal_inputs = "null".into();
            let mut formal_result = "null".into();
            let mut kind = "indirect";
            if function.is_fn() {
                let signature = function.fn_sig(tcx).skip_binder();
                formal_inputs =
                    array(signature.inputs().iter().map(|value| {
                        native_typ(tcx, self.normalize(owner, substitutions, *value))
                    }));
                formal_result = native_typ(
                    tcx,
                    self.normalize(owner, substitutions, signature.output()),
                );
            }
            if let ty::FnDef(def, call_args) = function.kind() {
                self.enqueue(*def, call_args);
                let env = if substitutions.has_non_region_param() {
                    ty::TypingEnv::post_analysis(tcx, owner)
                } else {
                    ty::TypingEnv::fully_monomorphized()
                };
                let resolution = match ty::Instance::try_resolve(tcx, env, *def, call_args) {
                    Ok(Some(instance)) => {
                        self.enqueue(instance.def_id(), instance.args);
                        let instance_kind = match instance.def {
                            ty::InstanceKind::Virtual(..) => "virtual",
                            ty::InstanceKind::Shim(ty::ShimKind::FnPtr(..)) => "fn-pointer-shim",
                            _ => "resolved",
                        };
                        kind = instance_kind;
                        object(&[
                            ("definition", definition(tcx, instance.def_id())),
                            (
                                "substitutions",
                                array(instance.args.iter().map(|arg| native_generic_arg(tcx, arg))),
                            ),
                        ])
                    }
                    Ok(None) => {
                        kind = "unresolved";
                        "null".into()
                    }
                    Err(error) => {
                        kind = "unresolved";
                        self.errors.insert(format!(
                            "native call resolution failed in {}: {error:?}",
                            tcx.def_path_str(owner)
                        ));
                        "null".into()
                    }
                };
                callee = object(&[
                    ("definition", definition(tcx, *def)),
                    (
                        "substitutions",
                        array(call_args.iter().map(|arg| native_generic_arg(tcx, arg))),
                    ),
                    ("resolved", resolution),
                ]);
            }
            let mut arguments = Vec::new();
            for arg in args {
                let value =
                    self.normalize(owner, substitutions, arg.node.ty(&body.local_decls, tcx));
                arguments.push(native_operand(tcx, &arg.node, value));
            }
            let destination_type = self.normalize(
                owner,
                substitutions,
                destination.ty(&body.local_decls, tcx).ty,
            );
            let mut call_sources = BTreeSet::new();
            for arg in args {
                let place = match &arg.node {
                    mir::Operand::Copy(place) | mir::Operand::Move(place) => Some(*place),
                    mir::Operand::Constant(_) | mir::Operand::RuntimeChecks(_) => None,
                };
                if let Some(place) = place {
                    let value =
                        self.normalize(owner, substitutions, place.ty(&body.local_decls, tcx).ty);
                    call_sources.insert(native_place(tcx, place, value));
                }
            }
            self.flows.insert(object(&[
                ("caller", caller.clone()),
                ("block", block.as_usize().to_string()),
                ("statement", data.statements.len().to_string()),
                ("source", location(tcx, term.source_info.span)),
                (
                    "destination",
                    native_place(tcx, *destination, destination_type),
                ),
                ("rvalue", quoted("call-result")),
                ("sources", array(call_sources)),
            ]));
            self.calls.insert(object(&[
                ("caller", caller.clone()),
                ("kind", quoted(kind)),
                ("callee", callee),
                ("callable_type", native_typ(tcx, function)),
                ("arguments", array(arguments)),
                ("formal_inputs", formal_inputs),
                ("formal_result", formal_result),
                (
                    "destination",
                    native_place(tcx, *destination, destination_type),
                ),
                (
                    "target",
                    target
                        .map(|block| block.as_usize().to_string())
                        .unwrap_or_else(|| "null".into()),
                ),
                ("unwind", quoted(format!("{unwind:?}"))),
                ("block", block.as_usize().to_string()),
                ("statement", data.statements.len().to_string()),
                ("source", location(tcx, term.source_info.span)),
            ]));
        }
    }

    fn run(mut self) -> String {
        let tcx = self.tcx;
        let owners: Vec<_> = self.owners.iter().copied().collect();
        for owner in &owners {
            let args = ty::GenericArgs::identity_for_item(tcx, *owner);
            self.body(*owner, args);
            if !args.has_non_region_param() {
                self.seen.push((*owner, args));
            }
        }
        while let Some((owner, args)) = self.queue.pop_front() {
            if self.seen.contains(&(owner, args)) {
                continue;
            }
            self.seen.push((owner, args));
            if self.seen.len() > 100_000 {
                self.errors
                    .insert("native local instance expansion limit exceeded".into());
                break;
            }
            self.body(owner, args);
        }
        object(&[
            ("format", "3".into()),
            ("scope", quoted("native-bindings")),
            ("compiler", quoted(env!("WIRE_DRIVER_COMPILER"))),
            (
                "crate",
                definition(tcx, rustc_span::def_id::CRATE_DEF_ID.to_def_id()),
            ),
            ("bodies", array(self.bodies)),
            ("calls", array(self.calls)),
            ("aggregates", array(self.aggregates)),
            ("flows", array(self.flows)),
            ("errors", array(self.errors.iter().map(quoted))),
        ])
    }
}

struct FunctionValues<'tcx> {
    values: Vec<Ty<'tcx>>,
    erased: Vec<(Ty<'tcx>, Span)>,
}
impl<'tcx> Visitor<'tcx> for FunctionValues<'tcx> {
    fn visit_operand(&mut self, operand: &mir::Operand<'tcx>, location: mir::Location) {
        if let mir::Operand::Constant(constant) = operand {
            self.values.push(constant.const_.ty());
        }
        self.super_operand(operand, location);
    }
    fn visit_rvalue(&mut self, value: &mir::Rvalue<'tcx>, location: mir::Location) {
        if let mir::Rvalue::Cast(_, mir::Operand::Constant(constant), target) = value
            && matches!(target.kind(), ty::FnPtr(..))
        {
            self.erased.push((constant.const_.ty(), constant.span));
        }
        self.super_rvalue(value, location);
    }
}

struct Discovery<'tcx> {
    tcx: TyCtxt<'tcx>,
    identities: Identities,
    calls: BTreeSet<String>,
    returns: BTreeSet<String>,
    codecs: BTreeSet<String>,
    schemas: BTreeSet<String>,
    errors: BTreeSet<String>,
    queue: VecDeque<(DefId, GenericArgsRef<'tcx>)>,
    seen: Vec<(DefId, GenericArgsRef<'tcx>)>,
    relevant: DefSet,
    concrete: DefSet,
    owners: DefSet,
    unresolved: DefSet,
    readers: BTreeSet<String>,
}
impl<'tcx> Discovery<'tcx> {
    fn returned_dynamic(&mut self, owner: DefId, root: Ty<'tcx>) -> bool {
        let mut pending = vec![root];
        let mut seen = Vec::new();
        let env = ty::TypingEnv::post_analysis(self.tcx, owner);
        while let Some(value) = pending.pop() {
            if seen.contains(&value) {
                continue;
            }
            seen.push(value);
            if seen.len() > 10_000 {
                self.errors.insert(format!(
                    "returned carrier expansion limit in {}",
                    self.tcx.def_path_str(owner)
                ));
                return false;
            }
            if dynamic(&self.identities, value) {
                return true;
            }
            for argument in value.walk() {
                if let Some(ty) = argument.as_type()
                    && let ty::Adt(def, args) = ty.kind()
                {
                    for field in def.all_fields() {
                        match self
                            .tcx
                            .try_normalize_erasing_regions(env, field.ty(self.tcx, args))
                        {
                            Ok(field) => pending.push(field),
                            Err(error) => {
                                self.errors.insert(format!(
                                    "unresolved returned field in {}: {error:?}",
                                    self.tcx.def_path_str(owner)
                                ));
                            }
                        }
                    }
                }
            }
        }
        false
    }
    fn enqueue(&mut self, def: DefId, args: GenericArgsRef<'tcx>) {
        if self.owners.contains(&def) && !args.has_non_region_param() {
            self.queue.push_back((def, args));
        }
    }
    fn normalize(
        &mut self,
        owner: DefId,
        args: GenericArgsRef<'tcx>,
        value: Ty<'tcx>,
    ) -> Option<Ty<'tcx>> {
        let env = if args.has_non_region_param() {
            ty::TypingEnv::post_analysis(self.tcx, owner)
        } else {
            ty::TypingEnv::fully_monomorphized()
        };
        match self.tcx.try_instantiate_and_normalize_erasing_regions(
            args,
            env,
            ty::EarlyBinder::bind(self.tcx, value),
        ) {
            Ok(value) => Some(value),
            Err(error) => {
                self.errors.insert(format!(
                    "unresolved type in {}: {error:?}",
                    self.tcx.def_path_str(owner)
                ));
                None
            }
        }
    }
    fn body(&mut self, owner: DefId, substitutions: GenericArgsRef<'tcx>, template: bool) {
        let tcx = self.tcx;
        let body = match tcx.def_kind(owner) {
            DefKind::Const { .. }
            | DefKind::AssocConst { .. }
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::Static { .. } => tcx.mir_for_ctfe(owner),
            DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::SyntheticCoroutineBody => {
                tcx.optimized_mir(owner)
            }
            kind => {
                self.errors.insert(format!(
                    "unsupported compiled body kind {kind:?}: {}",
                    tcx.def_path_str(owner)
                ));
                return;
            }
        };
        let owner_traits = traits(tcx, owner, substitutions);
        let serializer_parameters: Vec<_> = owner_traits
            .iter()
            .filter(|p| {
                p.def_id() == self.identities.serializer && p.self_ty().has_non_region_param()
            })
            .map(|p| p.self_ty())
            .collect();
        let mut implementation = "null".into();
        // Nested item definitions do not inherit an enclosing method's generic
        // arguments. Only the method's direct impl is its implementation owner.
        if let Some(id) = tcx.opt_parent(owner)
            && let DefKind::Impl { .. } = tcx.def_kind(id)
        {
            if let Some(trait_ref) = tcx.impl_opt_trait_ref(id) {
                let trait_ref = trait_ref
                    .instantiate(tcx, substitutions)
                    .skip_normalization();
                implementation = object(&[
                    ("trait", definition(tcx, trait_ref.def_id)),
                    ("self_type", typ(tcx, trait_ref.self_ty(), &self.identities)),
                ]);
            } else {
                let self_type = tcx
                    .type_of(id)
                    .instantiate(tcx, substitutions)
                    .skip_normalization();
                implementation = object(&[
                    ("trait", "null".into()),
                    ("self_type", typ(tcx, self_type, &self.identities)),
                ]);
            }
        }
        let mut ancestors = Vec::new();
        let mut ancestor = tcx.opt_parent(owner);
        while let Some(id) = ancestor {
            ancestors.push(definition(tcx, id));
            ancestor = tcx.opt_parent(id);
        }
        let caller = object(&[
            ("definition", definition(tcx, owner)),
            ("substitutions", quoted(format!("{substitutions:?}"))),
            ("implementation", implementation),
            ("ancestors", array(ancestors)),
        ]);
        if let Some(result) = self.normalize(owner, substitutions, body.return_ty())
            && self.returned_dynamic(owner, result)
        {
            self.returns.insert(object(&[
                ("caller", caller.clone()),
                ("type", typ(tcx, result, &self.identities)),
                ("source", location(tcx, body.span)),
            ]));
        }
        let mut values = FunctionValues {
            values: Vec::new(),
            erased: Vec::new(),
        };
        values.visit_body(body);
        values
            .values
            .extend(body.local_decls.iter().map(|local| local.ty));
        for value in values.values {
            let Some(value) = self.normalize(owner, substitutions, value) else {
                continue;
            };
            for arg in value.walk() {
                if let Some(t) = arg.as_type()
                    && let ty::FnDef(def, args) | ty::Closure(def, args) | ty::Coroutine(def, args) =
                        t.kind()
                {
                    self.enqueue(*def, args);
                }
            }
        }
        for (value, span) in values.erased {
            let Some(value) = self.normalize(owner, substitutions, value) else {
                continue;
            };
            if let ty::FnDef(def, args) = value.kind()
                && traits(tcx, *def, args).iter().any(|p| {
                    p.def_id() == self.identities.serialize
                        || p.def_id() == self.identities.serializer
                })
            {
                self.errors.insert(format!(
                    "erased serialization function value at {}: {}",
                    tcx.sess.source_map().span_to_diagnostic_string(span),
                    tcx.def_path_str(*def)
                ));
            }
        }
        for block in body.basic_blocks.iter() {
            let term = block.terminator();
            let mir::TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } = &term.kind
            else {
                continue;
            };
            let Some(function) =
                self.normalize(owner, substitutions, func.ty(&body.local_decls, tcx))
            else {
                continue;
            };
            let Some(result) = self.normalize(
                owner,
                substitutions,
                destination.ty(&body.local_decls, tcx).ty,
            ) else {
                continue;
            };
            let source = location(tcx, term.source_info.span);
            let argument_types: Vec<_> = args
                .iter()
                .filter_map(|arg| {
                    self.normalize(owner, substitutions, arg.node.ty(&body.local_decls, tcx))
                })
                .collect();
            let possible_transport = transport(tcx, result, &self.identities)
                || argument_types.iter().any(|t| {
                    matches!(t.kind(), ty::Ref(_, _, rustc_hir::Mutability::Mut))
                        && transport(tcx, *t, &self.identities)
                });
            let ty::FnDef(def, generics) = function.kind() else {
                if possible_transport {
                    self.errors
                        .insert(format!("indirect encoder call at {source}: {function}"));
                }
                continue;
            };
            self.enqueue(*def, generics);
            if !def.is_local() {
                for argument in &argument_types {
                    for element in argument.walk() {
                        if let Some(t) = element.as_type()
                            && let ty::FnDef(callback, callback_args) = t.kind()
                            && traits(tcx, *callback, callback_args).iter().any(|p| {
                                p.def_id() == self.identities.serialize
                                    || p.def_id() == self.identities.serializer
                            })
                        {
                            self.errors.insert(format!("serialization function passed to external callback at {source}: {}", tcx.def_path_str(*callback)));
                        }
                    }
                }
            }
            if possible_transport
                && tcx.opt_parent(*def).is_some_and(|p| {
                    tcx.def_kind(p) == DefKind::Trait
                        && !self.identities.decoder_traits.contains(&p)
                })
            {
                if tcx.opt_parent(*def) == self.identities.schema_trait {
                    self.schemas.insert(object(&[
                        ("caller", caller.clone()),
                        ("callee", definition(tcx, *def)),
                        ("substitutions", quoted(format!("{generics:?}"))),
                        ("source", source.clone()),
                        ("result", typ(tcx, result, &self.identities)),
                    ]));
                } else {
                    if !template {
                        self.concrete.insert(owner);
                    }
                    // A known impl such as Result<T,E>'s Try::branch can resolve
                    // even while T/E remain parameters; ask rustc instead of
                    // equating every generic argument with dynamic dispatch.
                    let env = if template {
                        ty::TypingEnv::post_analysis(tcx, owner)
                    } else {
                        ty::TypingEnv::fully_monomorphized()
                    };
                    match ty::Instance::try_resolve(tcx, env, *def, generics) {
                        Ok(Some(instance))
                            if !matches!(
                                instance.def,
                                ty::InstanceKind::Virtual(..)
                                    | ty::InstanceKind::Shim(ty::ShimKind::FnPtr(..))
                            ) =>
                        {
                            self.enqueue(instance.def_id(), instance.args)
                        }
                        _ if template => {
                            self.relevant.insert(owner);
                        }
                        _ => {
                            self.errors.insert(format!(
                                "unresolved trait encoder at {source}: {function}"
                            ));
                        }
                    }
                }
            }
            let obligations = traits(tcx, *def, generics);
            let mut payloads = Vec::new();
            let mut serializers = Vec::new();
            let mut read_payloads = Vec::new();
            for predicate in obligations {
                if predicate.def_id() == self.identities.serialize {
                    payloads.push(predicate.self_ty());
                }
                if predicate.def_id() == self.identities.serializer {
                    serializers.push(predicate.self_ty());
                }
                if Some(predicate.def_id()) == self.identities.deserialize {
                    read_payloads.push(predicate.self_ty());
                }
            }
            if !read_payloads.is_empty() {
                read_payloads.sort_by_key(|t| t.to_string());
                read_payloads.dedup();
                if read_payloads.iter().any(|t| t.has_non_region_param()) {
                    self.errors.insert(format!(
                        "open compiler JSON reader in {}",
                        tcx.def_path_str(owner)
                    ));
                }
                self.readers.insert(object(&[
                    ("caller", caller.clone()),
                    ("callee", definition(tcx, *def)),
                    ("source", source.clone()),
                    (
                        "payloads",
                        array(read_payloads.iter().map(|t| typ(tcx, *t, &self.identities))),
                    ),
                    (
                        "arguments",
                        array(
                            argument_types
                                .iter()
                                .map(|t| typ(tcx, *t, &self.identities)),
                        ),
                    ),
                    ("result", typ(tcx, result, &self.identities)),
                ]));
            }
            if payloads.is_empty() && serializers.is_empty() {
                continue;
            }
            self.relevant.insert(owner);
            if !template {
                self.concrete.insert(owner);
            }
            // A surrounding Serialize impl never grants admission to an
            // independent concrete encoder invocation in its body.
            let parameterized_codec = generics.types().any(|t| {
                t.walk().any(|arg| {
                    arg.as_type()
                        .is_some_and(|t| serializer_parameters.contains(&t))
                })
            });
            if template
                && !parameterized_codec
                && payloads
                    .iter()
                    .chain(&serializers)
                    .any(|t| t.has_non_region_param())
            {
                self.unresolved.insert(owner);
                continue;
            }
            payloads.sort_by_key(|t| t.to_string());
            payloads.dedup();
            let record = object(&[
                ("caller", caller.clone()),
                ("callee", definition(tcx, *def)),
                ("substitutions", quoted(format!("{generics:?}"))),
                ("source", source),
                (
                    "payloads",
                    array(payloads.iter().map(|t| typ(tcx, *t, &self.identities))),
                ),
                (
                    "serializers",
                    array(serializers.iter().map(|t| typ(tcx, *t, &self.identities))),
                ),
                (
                    "arguments",
                    array(
                        argument_types
                            .iter()
                            .map(|t| typ(tcx, *t, &self.identities)),
                    ),
                ),
                ("result", typ(tcx, result, &self.identities)),
                ("local_callee", def.is_local().to_string()),
            ]);
            if parameterized_codec {
                self.codecs.insert(record);
            } else {
                if payloads
                    .iter()
                    .chain(&serializers)
                    .any(|t| t.has_non_region_param())
                {
                    self.errors.insert(format!(
                        "open serialization payload in {}",
                        tcx.def_path_str(owner)
                    ));
                }
                self.calls.insert(record);
            }
        }
    }
    fn run(mut self) -> String {
        let tcx = self.tcx;
        let mut owners: Vec<_> = self.owners.iter().copied().collect();
        owners.sort_by_key(|id| id.index.as_u32());
        for owner in &owners {
            let args = ty::GenericArgs::identity_for_item(tcx, *owner);
            if args.has_non_region_param() {
                self.body(*owner, args, true);
            } else {
                self.enqueue(*owner, args);
            }
        }
        while let Some((def, args)) = self.queue.pop_front() {
            if self.seen.contains(&(def, args)) {
                continue;
            }
            self.seen.push((def, args));
            if self.seen.len() > 100_000 {
                self.errors
                    .insert("local instance expansion limit exceeded".into());
                break;
            }
            self.body(def, args, false);
        }
        for owner in self.relevant.iter() {
            let args = ty::GenericArgs::identity_for_item(tcx, *owner);
            let codec = traits(tcx, *owner, args)
                .iter()
                .any(|p| p.def_id() == self.identities.serializer);
            if (!codec || self.unresolved.contains(owner))
                && args.has_non_region_param()
                && (!self.concrete.contains(owner)
                    || tcx
                        .effective_visibilities(())
                        .is_exported(owner.expect_local()))
            {
                self.errors.insert(format!(
                    "open generic publication body {}",
                    tcx.def_path_str(*owner)
                ));
            }
        }
        let compiler_json = self.identities.deserialize.is_some();
        let mut result = object(&[
            ("format", if compiler_json { "2" } else { "1" }.into()),
            ("compiler", quoted(env!("WIRE_DRIVER_COMPILER"))),
            (
                "crate",
                definition(tcx, rustc_span::def_id::CRATE_DEF_ID.to_def_id()),
            ),
            (
                "serialize_trait",
                definition(tcx, self.identities.serialize),
            ),
            (
                "serializer_trait",
                definition(tcx, self.identities.serializer),
            ),
            (
                "schema_trait",
                self.identities
                    .schema_trait
                    .map(|id| definition(tcx, id))
                    .unwrap_or_else(|| "null".into()),
            ),
            (
                "decoder_traits",
                array(
                    self.identities
                        .decoder_traits
                        .iter()
                        .map(|id| definition(tcx, *id))
                        .collect::<BTreeSet<_>>(),
                ),
            ),
            (
                "dynamic_carriers",
                array(
                    self.identities
                        .dynamic
                        .iter()
                        .map(|id| definition(tcx, *id))
                        .collect::<BTreeSet<_>>(),
                ),
            ),
            (
                "bodies",
                array(owners.iter().map(|id| definition(tcx, *id))),
            ),
            ("instances", self.seen.len().to_string()),
            ("calls", array(self.calls)),
            ("codec_calls", array(self.codecs)),
            ("schema_calls", array(self.schemas)),
            ("dynamic_returns", array(self.returns)),
            ("errors", array(self.errors.iter().map(quoted))),
        ]);
        if let Some(deserialize) = self.identities.deserialize {
            assert_eq!(result.pop(), Some('}'));
            result.push_str(&format!(
                ",\"scope\":\"compiler-json\",\"deserialize_trait\":{},\"conversion_traits\":{},\"reader_calls\":{}}}",
                definition(tcx, deserialize),
                array(self.identities.conversion_traits.iter().map(|id| definition(tcx, *id))),
                array(self.readers),
            ));
        }
        result
    }
}

struct Probe {
    command: Vec<String>,
}
impl Callbacks for Probe {
    fn after_analysis<'tcx>(&mut self, _: &interface::Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        let native = std::env::var("WIRE_CALL_SCOPE").as_deref() == Ok("native-bindings");
        let mut result = if native {
            NativeDiscovery {
                tcx,
                bodies: BTreeSet::new(),
                calls: BTreeSet::new(),
                aggregates: BTreeSet::new(),
                flows: BTreeSet::new(),
                errors: BTreeSet::new(),
                queue: VecDeque::new(),
                seen: Vec::new(),
                owners: tcx
                    .hir_body_owners()
                    .map(|owner| owner.to_def_id())
                    .filter(|owner| tcx.is_mir_available(*owner))
                    .collect(),
            }
            .run()
        } else {
            let identities =
                Identities::read(tcx).unwrap_or_else(|error| panic!("wire call identity: {error}"));
            let compiler_json = identities.deserialize.is_some();
            Discovery {
                tcx,
                identities,
                calls: BTreeSet::new(),
                returns: BTreeSet::new(),
                codecs: BTreeSet::new(),
                schemas: BTreeSet::new(),
                errors: BTreeSet::new(),
                queue: VecDeque::new(),
                seen: Vec::new(),
                relevant: DefSet::new(),
                concrete: DefSet::new(),
                unresolved: DefSet::new(),
                readers: BTreeSet::new(),
                owners: tcx
                    .hir_body_owners()
                    .map(|owner| owner.to_def_id())
                    .filter(|owner| {
                        if !compiler_json {
                            return true;
                        }
                        let mut current = tcx.opt_parent(*owner);
                        while let Some(id) = current {
                            if tcx.def_kind(id) == DefKind::Mod
                                && tcx.opt_parent(id)
                                    == Some(rustc_span::def_id::CRATE_DEF_ID.to_def_id())
                                && tcx
                                    .opt_item_name(id)
                                    .is_some_and(|name| name.as_str() == "compiler_json")
                            {
                                return true;
                            }
                            current = tcx.opt_parent(id);
                        }
                        false
                    })
                    .collect(),
            }
            .run()
        };
        let path = std::env::var_os("WIRE_CALL_REPORT").expect("WIRE_CALL_REPORT is required");
        assert_eq!(result.pop(), Some('}'));
        result.push_str(&format!(
            ",\"rustc_command\":{},\"inputs\":{}}}",
            array(self.command.iter().map(quoted)),
            array(
                tcx.sess
                    .source_map()
                    .files()
                    .iter()
                    .filter(|file| file.src.is_some())
                    .map(|file| {
                        let path = match &file.name {
                            rustc_span::FileName::Real(real) => real
                                .local_path()
                                .map(|p| quoted(p.to_string_lossy()))
                                .unwrap_or_else(|| "null".into()),
                            _ => "null".into(),
                        };
                        object(&[
                            ("name", quoted(format!("{:?}", file.name))),
                            ("path", path),
                            ("hash", quoted(file.src_hash.to_string())),
                        ])
                    })
            )
        ));
        std::fs::write(path, result).expect("write compiler call receipt");
        Compilation::Stop
    }
}
fn main() {
    let mut args: Vec<_> = std::env::args().collect();
    // Cargo's RUSTC_WRAPPER passes the real compiler as argv[1]. Non-target
    // crates retain that compiler; only the explicitly selected crate is probed.
    if let Ok(target) = std::env::var("WIRE_CALL_CRATE") {
        let actual = args
            .windows(2)
            .find(|w| w[0] == "--crate-name")
            .map(|w| w[1].clone());
        let rustc = args.remove(1);
        if actual.as_deref() != Some(target.as_str()) {
            let status = std::process::Command::new(rustc)
                .args(&args[1..])
                .status()
                .expect("delegate rustc");
            std::process::exit(status.code().unwrap_or(1));
        }
    }
    rustc_driver::run_compiler(
        &args,
        &mut Probe {
            command: args.clone(),
        },
    );
}
