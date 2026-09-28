//! Lower checked host code to the ownership-explicit control-flow form.
//!
//! Lowering assigns every operand one of the three closed dispositions before
//! verification. Phase 2 places terminal operations at scope exit; Phase 3
//! may move those terminals to proven last uses without changing ownership.

use std::collections::{BTreeMap, BTreeSet};

use chelis_deep::DeepTag;
use chelis_deep::ast::{Atom, Expr};
use chelis_types::CheckedProgram;
use chelis_types::manifest::RootManifest;
use chelis_types::types::{Lane, Prim};

use crate::dag::{DimInfo, TensorType};
use crate::host::{
    ConcreteHostCallback, ConcreteHostCallbackKind, ConcreteHostExpr, ConcreteHostExprKind,
    ConcreteHostFunction, ConcreteHostMatchArm, ConcreteHostProgram, HostBinding, HostDisplayRoot,
    HostExpr, HostExprKind, HostFunctionOrigin, HostTensorHelper,
};
use crate::host_type_state::ConcreteHostType;

use super::classify::{ClassifyError, NonHeapKind, Placement, ValueClass, classify, render_type};
use super::error::OwnershipError;
use super::ir::{
    ApplyKind, Block, BlockId, BlockParam, CallableBody, Edge, EdgeId, HostSiteAction,
    HostSiteBuilder, HostSiteId, HostSiteKind, Op, OpId, Operand, Operation, OperationRole,
    OperationSchema, OwnerId, OwnerInfo, OwnerOrigin, OwnershipProgram, ParamMode, ScheduleState,
    Terminator, Unit, UnitId, UnitKind,
};

const ROOTS_UNIT: &str = "roots";

#[derive(Clone)]
struct ParamSpec {
    mode: ParamMode,
    ty: ConcreteHostType,
    type_pattern: FormalTypePattern,
    callback_modes: Option<Vec<ParamMode>>,
}

struct DeclaredParam<'a> {
    mode: ParamMode,
    callback_modes: Option<Vec<ParamMode>>,
    checked_type: &'a Expr,
}

#[derive(Clone)]
struct Signature {
    params: Vec<ParamSpec>,
}

#[derive(Clone)]
enum FormalTypePattern {
    Exact,
    Function(Vec<Self>, Box<Self>),
    Adt(Vec<Self>),
    List(Box<Self>),
    Dict(Box<Self>, Box<Self>),
    Tuple(Vec<Self>),
    Tensor(Vec<FormalDimension>),
    Option(Box<Self>),
}

#[derive(Clone)]
enum FormalDimension {
    Quantified(String),
    Nominal,
}

impl FormalTypePattern {
    fn from_checked(formal: &ConcreteHostType, checked: &Expr) -> Self {
        let Some((tag, children)) = tag_and_children(checked) else {
            return Self::nominal(formal);
        };
        if tag == DeepTag::TRef {
            return children
                .first()
                .map(|inner| Self::from_checked(formal, inner))
                .unwrap_or_else(|| Self::nominal(formal));
        }
        match (formal, tag) {
            (ConcreteHostType::Function(params, ret), DeepTag::TFn) => {
                let Some((checked_ret, checked_params)) = children.split_last() else {
                    return Self::nominal(formal);
                };
                if params.len() != checked_params.len() {
                    return Self::nominal(formal);
                }
                Self::Function(
                    params
                        .iter()
                        .zip(checked_params)
                        .map(|(formal, checked)| Self::from_checked(formal, checked))
                        .collect(),
                    Box::new(Self::from_checked(ret, checked_ret)),
                )
            }
            (ConcreteHostType::Adt(_, args), DeepTag::TAdt) => {
                let Some(checked_args) = children.get(1..) else {
                    return Self::nominal(formal);
                };
                if args.len() != checked_args.len() {
                    return Self::nominal(formal);
                }
                Self::Adt(
                    args.iter()
                        .zip(checked_args)
                        .map(|(formal, checked)| Self::from_checked(formal, checked))
                        .collect(),
                )
            }
            (ConcreteHostType::List(inner), DeepTag::TAdt)
            | (ConcreteHostType::Option(inner), DeepTag::TAdt) => children
                .get(1)
                .map(|checked| {
                    let inner = Box::new(Self::from_checked(inner, checked));
                    if matches!(formal, ConcreteHostType::List(_)) {
                        Self::List(inner)
                    } else {
                        Self::Option(inner)
                    }
                })
                .unwrap_or_else(|| Self::nominal(formal)),
            (ConcreteHostType::Dict(key, value), DeepTag::TAdt) => {
                match (children.get(1), children.get(2)) {
                    (Some(checked_key), Some(checked_value)) => Self::Dict(
                        Box::new(Self::from_checked(key, checked_key)),
                        Box::new(Self::from_checked(value, checked_value)),
                    ),
                    _ => Self::nominal(formal),
                }
            }
            (ConcreteHostType::Tuple(items), DeepTag::TTuple) if items.len() == children.len() => {
                Self::Tuple(
                    items
                        .iter()
                        .zip(children)
                        .map(|(formal, checked)| Self::from_checked(formal, checked))
                        .collect(),
                )
            }
            (ConcreteHostType::Tensor(tensor), DeepTag::TTensor) => {
                let Some((_, checked_dims)) = children.split_last() else {
                    return Self::nominal(formal);
                };
                if tensor.dims.len() != checked_dims.len() {
                    return Self::nominal(formal);
                }
                Self::Tensor(
                    checked_dims
                        .iter()
                        .map(|dimension| {
                            dimension_variable_key(dimension)
                                .map(FormalDimension::Quantified)
                                .unwrap_or(FormalDimension::Nominal)
                        })
                        .collect(),
                )
            }
            _ => Self::nominal(formal),
        }
    }

    fn nominal(formal: &ConcreteHostType) -> Self {
        match formal {
            ConcreteHostType::Function(params, ret) => Self::Function(
                params.iter().map(Self::nominal).collect(),
                Box::new(Self::nominal(ret)),
            ),
            ConcreteHostType::Adt(_, args) => Self::Adt(args.iter().map(Self::nominal).collect()),
            ConcreteHostType::List(inner) => Self::List(Box::new(Self::nominal(inner))),
            ConcreteHostType::Dict(key, value) => {
                Self::Dict(Box::new(Self::nominal(key)), Box::new(Self::nominal(value)))
            }
            ConcreteHostType::Tuple(items) => {
                Self::Tuple(items.iter().map(Self::nominal).collect())
            }
            ConcreteHostType::Tensor(tensor) => {
                Self::Tensor(vec![FormalDimension::Nominal; tensor.dims.len()])
            }
            ConcreteHostType::Option(inner) => Self::Option(Box::new(Self::nominal(inner))),
            ConcreteHostType::Scalar(_)
            | ConcreteHostType::KeyBuiltinCallable(_)
            | ConcreteHostType::MappedFile
            | ConcreteHostType::Unit => Self::Exact,
        }
    }
}

fn dimension_variable_key(expr: &Expr) -> Option<String> {
    let (DeepTag::DVar, children) = tag_and_children(expr)? else {
        return None;
    };
    match children.first().map(strip_meta) {
        Some(Expr::Atom(Atom::Name(name), _)) => Some(name.clone()),
        _ => None,
    }
}

#[derive(Default)]
struct CallTypeInstantiation {
    dimensions: BTreeMap<String, DimInfo>,
}

struct DirectCallArgument<'a> {
    function: &'a str,
    index: usize,
    expr: &'a ConcreteHostExpr,
    pattern: &'a FormalTypePattern,
    formal: &'a ConcreteHostType,
    checked_slot: &'a ConcreteHostType,
}

impl CallTypeInstantiation {
    /// Check one checker-owned call slot against the resolved callable formal.
    /// A direct callable may retain universally quantified dimension names in
    /// its body signature even though the call site has instantiated them to
    /// literals or other named dimensions ([04-TY] section 4.4). All other
    /// structure, including scalar precision and tensor rank, stays exact.
    fn admits(
        &mut self,
        pattern: &FormalTypePattern,
        formal: &ConcreteHostType,
        actual: &ConcreteHostType,
    ) -> bool {
        match (pattern, formal, actual) {
            (FormalTypePattern::Exact, _, _) => formal == actual,
            (
                FormalTypePattern::Function(pattern_params, pattern_ret),
                ConcreteHostType::Function(formal_params, formal_ret),
                ConcreteHostType::Function(actual_params, actual_ret),
            ) => {
                pattern_params.len() == formal_params.len()
                    && formal_params.len() == actual_params.len()
                    && pattern_params
                        .iter()
                        .zip(formal_params)
                        .zip(actual_params)
                        .all(|((pattern, formal), actual)| self.admits(pattern, formal, actual))
                    && self.admits(pattern_ret, formal_ret, actual_ret)
            }
            (
                FormalTypePattern::Adt(pattern_args),
                ConcreteHostType::Adt(formal_name, formal_args),
                ConcreteHostType::Adt(actual_name, actual_args),
            ) => {
                formal_name == actual_name
                    && pattern_args.len() == formal_args.len()
                    && formal_args.len() == actual_args.len()
                    && pattern_args
                        .iter()
                        .zip(formal_args)
                        .zip(actual_args)
                        .all(|((pattern, formal), actual)| self.admits(pattern, formal, actual))
            }
            (
                FormalTypePattern::List(pattern),
                ConcreteHostType::List(formal),
                ConcreteHostType::List(actual),
            )
            | (
                FormalTypePattern::Option(pattern),
                ConcreteHostType::Option(formal),
                ConcreteHostType::Option(actual),
            ) => self.admits(pattern, formal, actual),
            (
                FormalTypePattern::Dict(key_pattern, value_pattern),
                ConcreteHostType::Dict(formal_key, formal_value),
                ConcreteHostType::Dict(actual_key, actual_value),
            ) => {
                self.admits(key_pattern, formal_key, actual_key)
                    && self.admits(value_pattern, formal_value, actual_value)
            }
            (
                FormalTypePattern::Tuple(patterns),
                ConcreteHostType::Tuple(formal),
                ConcreteHostType::Tuple(actual),
            ) => {
                patterns.len() == formal.len()
                    && formal.len() == actual.len()
                    && patterns
                        .iter()
                        .zip(formal)
                        .zip(actual)
                        .all(|((pattern, formal), actual)| self.admits(pattern, formal, actual))
            }
            (
                FormalTypePattern::Tensor(dimensions),
                ConcreteHostType::Tensor(formal),
                ConcreteHostType::Tensor(actual),
            ) => self.admits_tensor(dimensions, formal, actual),
            _ => false,
        }
    }

    fn admits_tensor(
        &mut self,
        dimensions: &[FormalDimension],
        formal: &TensorType,
        actual: &TensorType,
    ) -> bool {
        formal.precision == actual.precision
            && dimensions.len() == formal.dims.len()
            && formal.dims.len() == actual.dims.len()
            && dimensions
                .iter()
                .zip(&formal.dims)
                .zip(&actual.dims)
                .all(|((pattern, formal), actual)| self.admits_dimension(pattern, formal, actual))
    }

    fn admits_dimension(
        &mut self,
        pattern: &FormalDimension,
        formal: &DimInfo,
        actual: &DimInfo,
    ) -> bool {
        match pattern {
            FormalDimension::Quantified(key) => {
                if formal == actual {
                    return true;
                }
                match self.dimensions.get(key) {
                    Some(bound) => dimensions_compatible(bound, actual),
                    None => {
                        self.dimensions.insert(key.clone(), actual.clone());
                        true
                    }
                }
            }
            FormalDimension::Nominal => nominal_dimension_accepts(formal, actual),
        }
    }
}

fn nominal_dimension_accepts(formal: &DimInfo, actual: &DimInfo) -> bool {
    match (formal, actual) {
        (DimInfo::Named(name, None), _) if name.is_empty() || name == "*" => true,
        (DimInfo::Named(formal, None), DimInfo::Named(actual, _)) => formal == actual,
        (DimInfo::Named(_, None), DimInfo::Lit(_)) => true,
        _ => dimensions_compatible(formal, actual),
    }
}

fn dimensions_compatible(left: &DimInfo, right: &DimInfo) -> bool {
    match (left, right) {
        (DimInfo::Lit(left), DimInfo::Lit(right)) => left == right,
        (DimInfo::Named(left, left_size), DimInfo::Named(right, right_size)) => {
            left == right
                && match (left_size, right_size) {
                    (Some(left), Some(right)) => left == right,
                    _ => true,
                }
        }
        (DimInfo::Lit(left), DimInfo::Named(_, Some(right)))
        | (DimInfo::Named(_, Some(left)), DimInfo::Lit(right)) => left == right,
        (DimInfo::Lit(_), DimInfo::Named(_, None)) | (DimInfo::Named(_, None), DimInfo::Lit(_)) => {
            true
        }
    }
}

pub(super) fn lower(
    checked: &CheckedProgram,
    host: &ConcreteHostProgram,
    manifest: &RootManifest,
    root_bindings: &[Option<String>],
    sites: &mut HostSiteBuilder,
) -> Result<OwnershipProgram, OwnershipError> {
    let signatures = build_signatures(checked, host)?;
    let ctx = Context {
        host,
        signatures,
        global_names: host
            .globals
            .iter()
            .map(|binding| binding.name.clone())
            .collect(),
        function_names: host
            .functions
            .iter()
            .map(|function| function.name.clone())
            .collect(),
        function_units: host
            .functions
            .iter()
            .enumerate()
            .map(|(index, function)| (function.name.clone(), UnitId((index + 1) as u32)))
            .collect(),
    };
    let mut units = vec![lower_roots(&ctx, manifest, root_bindings, sites, 0)?];
    for (index, function) in host.functions.iter().enumerate() {
        units.push(lower_function(&ctx, function, sites, index + 1)?);
    }
    Ok(OwnershipProgram { units })
}

fn build_signatures(
    checked: &CheckedProgram,
    host: &ConcreteHostProgram,
) -> Result<BTreeMap<String, Signature>, OwnershipError> {
    let mut signatures = BTreeMap::new();
    for function in &host.functions {
        let declarations = declared_params(checked, &function.name, function.params.len())
            .ok_or_else(|| OwnershipError::MissingSignature {
                function: function.name.clone(),
            })?;
        let params = function
            .params
            .iter()
            .zip(declarations)
            .map(|(param, declared)| ParamSpec {
                mode: declared.mode,
                ty: param.ty.clone(),
                type_pattern: if function.origin == HostFunctionOrigin::Authored {
                    FormalTypePattern::from_checked(&param.ty, declared.checked_type)
                } else {
                    FormalTypePattern::nominal(&param.ty)
                },
                callback_modes: declared.callback_modes,
            })
            .collect();
        signatures.insert(function.name.clone(), Signature { params });
    }
    Ok(signatures)
}

/// Read ownership modes and dimension-variable provenance from the checked
/// function type. A monomorphized host specialization uses the authored
/// generic signature.
fn declared_params<'a>(
    checked: &'a CheckedProgram,
    name: &str,
    arity: usize,
) -> Option<Vec<DeclaredParam<'a>>> {
    let generic_name = match name.split_once("__mono_") {
        Some((generic, _)) => generic,
        None => name,
    };
    let ty = checked
        .type_env()
        .get(name)
        .or_else(|| checked.type_env().get(generic_name))?;
    let (params, _) = fn_type_parts(ty)?;
    if params.len() != arity {
        return None;
    }
    params
        .into_iter()
        .map(|param| {
            let (mode, callback_modes) = param_mode_of(param)?;
            Some(DeclaredParam {
                mode,
                callback_modes,
                checked_type: param,
            })
        })
        .collect()
}

fn strip_meta(expr: &Expr) -> &Expr {
    match expr {
        Expr::MetaExpr(meta, _) => strip_meta(&meta.expr),
        other => other,
    }
}

fn tag_and_children(expr: &Expr) -> Option<(DeepTag, &[Expr])> {
    match strip_meta(expr) {
        Expr::Node(node, _) => Some((node.tag(), node.children_slice())),
        _ => None,
    }
}

fn fn_type_parts(expr: &Expr) -> Option<(Vec<&Expr>, &Expr)> {
    let (tag, children) = tag_and_children(expr)?;
    if tag != DeepTag::TFn {
        return None;
    }
    let (ret, params) = children.split_last()?;
    Some((params.iter().collect(), ret))
}

fn param_mode_of(param: &Expr) -> Option<(ParamMode, Option<Vec<ParamMode>>)> {
    match tag_and_children(param) {
        Some((DeepTag::TRef, _)) => Some((ParamMode::Borrowed, None)),
        Some((DeepTag::TFn, children)) => {
            let (_, params) = children.split_last()?;
            let modes = params
                .iter()
                .map(|param| param_mode_of(param).map(|(mode, _)| mode))
                .collect::<Option<Vec<_>>>()?;
            Some((ParamMode::Owned, Some(modes)))
        }
        _ => Some((ParamMode::Owned, None)),
    }
}

struct Context<'a> {
    host: &'a ConcreteHostProgram,
    signatures: BTreeMap<String, Signature>,
    global_names: BTreeSet<String>,
    function_names: BTreeSet<String>,
    function_units: BTreeMap<String, UnitId>,
}

#[derive(Clone, Copy)]
enum Place {
    Owner(OwnerId),
    Callback(OwnerId),
}

#[derive(Clone)]
enum Value {
    Fresh(OwnerId),
    Named(OwnerId),
    Callback(OwnerId),
    FunctionRef { name: String, ty: ConcreteHostType },
}

struct Scope {
    owners: Vec<OwnerId>,
    names: BTreeMap<String, Place>,
}

struct BlockBuilder {
    id: BlockId,
    params: Vec<BlockParam>,
    ops: Vec<Operation>,
    terminator: Option<Terminator>,
}

#[derive(Clone, Copy)]
struct AdapterBlocks {
    entry: BlockId,
    body: BlockId,
}

struct UnitLowerer<'a, 'sites> {
    ctx: &'a Context<'a>,
    unit_name: String,
    helpers: &'a [HostTensorHelper],
    blocks: Vec<BlockBuilder>,
    owners: BTreeMap<OwnerId, OwnerInfo>,
    owner_depth: BTreeMap<OwnerId, usize>,
    callback_modes: BTreeMap<OwnerId, Vec<ParamMode>>,
    next_owner: u32,
    next_operation: u32,
    next_edge: u32,
    recorded_edges: BTreeSet<EdgeId>,
    current: BlockId,
    scopes: Vec<Scope>,
    moved: BTreeSet<OwnerId>,
    adapter: Option<AdapterBlocks>,
    captures: BTreeMap<String, OwnerId>,
    sites: &'sites mut HostSiteBuilder,
    unit_index: usize,
    active_site: Option<HostSiteId>,
    /// Span of the host expression currently being lowered, saved and restored
    /// around each expression exactly as `active_site` is. Owners minted while
    /// it is set record it (chelis#2122).
    active_span: Option<String>,
}

impl<'a, 'sites> UnitLowerer<'a, 'sites> {
    fn new(
        ctx: &'a Context<'a>,
        unit_name: &str,
        helpers: &'a [HostTensorHelper],
        sites: &'sites mut HostSiteBuilder,
        unit_index: usize,
    ) -> Self {
        Self {
            ctx,
            unit_name: unit_name.to_string(),
            helpers,
            blocks: Vec::new(),
            owners: BTreeMap::new(),
            owner_depth: BTreeMap::new(),
            callback_modes: BTreeMap::new(),
            next_owner: 0,
            next_operation: 0,
            next_edge: 0,
            recorded_edges: BTreeSet::new(),
            current: BlockId(0),
            scopes: Vec::new(),
            moved: BTreeSet::new(),
            adapter: None,
            captures: BTreeMap::new(),
            sites,
            unit_index,
            active_site: None,
            active_span: None,
        }
    }

    fn invariant(&self, detail: impl Into<String>) -> OwnershipError {
        OwnershipError::LoweringInvariant {
            unit: self.unit_name.clone(),
            detail: detail.into(),
        }
    }

    fn new_block(&mut self, params: Vec<BlockParam>) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BlockBuilder {
            id,
            params,
            ops: Vec::new(),
            terminator: None,
        });
        id
    }

    fn with_site<T>(
        &mut self,
        kind: HostSiteKind,
        action: impl FnOnce(&mut Self) -> Result<T, OwnershipError>,
    ) -> Result<T, OwnershipError> {
        let site = self.sites.add(self.unit_index, kind);
        let previous = self.active_site.replace(site);
        let result = action(self);
        self.active_site = previous;
        result
    }

    fn emit(&mut self, op: Op) {
        self.emit_with_role(op, OperationRole::Semantic);
    }

    fn emit_scope_exit(&mut self, op: Op) {
        self.emit_with_role(op, OperationRole::ProvisionalScopeExit);
    }

    fn emit_with_role(&mut self, op: Op, role: OperationRole) {
        let block = self.current;
        let operation = OpId(self.next_operation);
        self.next_operation = self
            .next_operation
            .checked_add(1)
            .expect("ownership operation census exceeds u32");
        let site = self
            .active_site
            .expect("every ownership operation is emitted inside a host site");
        self.sites.record(
            site,
            HostSiteAction::Operation {
                unit: self.unit_index,
                block,
                operation,
            },
        );
        self.blocks[block.0 as usize].ops.push(Operation {
            id: operation,
            role,
            kind: op,
        });
    }

    fn record_edge(&mut self, kind: HostSiteKind, target: BlockId) {
        let edge = self.blocks[self.current.0 as usize]
            .terminator
            .as_ref()
            .into_iter()
            .flat_map(Terminator::edges)
            .find(|edge| edge.target == target && !self.recorded_edges.contains(&edge.id))
            .map(|edge| edge.id)
            .expect("recorded control edge must exist on the current terminator");
        self.recorded_edges.insert(edge);
        let site = self.sites.add(self.unit_index, kind);
        self.sites.record(
            site,
            HostSiteAction::ControlEdge {
                unit: self.unit_index,
                edge,
                source: self.current,
                target,
            },
        );
    }

    fn set_terminator(&mut self, mut terminator: Terminator) -> Result<(), OwnershipError> {
        let index = self.current.0 as usize;
        if self.blocks[index].terminator.is_some() {
            return Err(self.invariant(format!("block b{} terminated twice", self.current.0)));
        }
        for edge in terminator.edges_mut() {
            edge.id = EdgeId(self.next_edge);
            self.next_edge = self
                .next_edge
                .checked_add(1)
                .expect("ownership edge census exceeds u32");
        }
        self.blocks[index].terminator = Some(terminator);
        let site = self
            .active_site
            .expect("every ownership terminator is emitted inside a host site");
        self.sites.record(
            site,
            HostSiteAction::Terminator {
                unit: self.unit_index,
                block: self.current,
            },
        );
        Ok(())
    }

    fn depth(&self) -> usize {
        match self.scopes.as_slice() {
            [] => 0,
            [_, nested @ ..] => nested.len(),
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope {
            owners: Vec::new(),
            names: BTreeMap::new(),
        });
    }

    fn scope_mut(&mut self) -> Result<&mut Scope, OwnershipError> {
        let error = self.invariant("no open scope");
        self.scopes.last_mut().ok_or(error)
    }

    fn classify_or_reject(
        &self,
        ty: &ConcreteHostType,
        placement: Placement,
        name: Option<&str>,
    ) -> Result<ValueClass, OwnershipError> {
        classify(ty, placement).map_err(|error| match error {
            ClassifyError::FirstClassFunction => OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name: match name {
                    Some(name) => name.to_string(),
                    None => render_type(ty),
                },
            },
            ClassifyError::FunctionContainer => OwnershipError::FunctionContainer {
                unit: self.unit_name.clone(),
                ty: render_type(ty),
            },
        })
    }

    fn mint(
        &mut self,
        ty: &ConcreteHostType,
        placement: Placement,
        origin: OwnerOrigin,
        names: Vec<String>,
    ) -> Result<OwnerId, OwnershipError> {
        let class = self.classify_or_reject(ty, placement, names.first().map(String::as_str))?;
        let id = OwnerId(self.next_owner);
        self.next_owner += 1;
        self.owners.insert(
            id,
            OwnerInfo {
                ty: ty.clone(),
                class,
                placement,
                origin,
                names,
                span_id: self.active_span.clone(),
            },
        );
        self.owner_depth.insert(id, self.depth());
        Ok(id)
    }

    fn info(&self, owner: OwnerId) -> Result<&OwnerInfo, OwnershipError> {
        self.owners
            .get(&owner)
            .ok_or_else(|| self.invariant(format!("owner %{} has no metadata", owner.0)))
    }

    fn owner_label(&self, owner: OwnerId) -> Result<String, OwnershipError> {
        let info = self.info(owner)?;
        Ok(match info.names.first() {
            Some(name) => name.clone(),
            None => render_type(&info.ty),
        })
    }

    fn register(&mut self, owner: OwnerId) -> Result<(), OwnershipError> {
        self.scope_mut()?.owners.push(owner);
        Ok(())
    }

    fn name_owner(&mut self, owner: OwnerId, name: &str) -> Result<(), OwnershipError> {
        if let Some(info) = self.owners.get_mut(&owner) {
            info.names.push(name.to_string());
        }
        self.scope_mut()?
            .names
            .insert(name.to_string(), Place::Owner(owner));
        Ok(())
    }

    fn lookup(&self, name: &str) -> Option<Place> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.names.get(name).copied())
    }

    fn copy(&mut self, source: OwnerId) -> Result<OwnerId, OwnershipError> {
        let info = self.info(source)?;
        let ty = info.ty.clone();
        let placement = info.placement;
        let dest = self.mint(&ty, placement, OwnerOrigin::Owned, Vec::new())?;
        self.emit(Op::Copy {
            dest,
            source: Operand::clone_(source),
        });
        Ok(dest)
    }

    fn consume(&mut self, value: Value, tail: Option<usize>) -> Result<Operand, OwnershipError> {
        match value {
            Value::Fresh(owner) => {
                self.moved.insert(owner);
                Ok(Operand::move_(owner))
            }
            Value::Named(owner) => {
                let info = self.info(owner)?;
                if info.class.is_callback() {
                    return Err(OwnershipError::FirstClassFunctionValue {
                        unit: self.unit_name.clone(),
                        name: self.owner_label(owner)?,
                    });
                }
                let depth = self.owner_depth.get(&owner).copied().ok_or_else(|| {
                    self.invariant(format!("owner %{} has no scope depth", owner.0))
                })?;
                // A by-value Copy scalar owns no resource, so a use duplicates
                // it rather than transferring it. Moving the original owner
                // would consume it, wrongly leaving a second occurrence of the
                // same named scalar in one call (`f3(x, x)` in tail position)
                // reading a dead owner (chelis#2068). Duplicating instead keeps
                // the original live for every slot. This is scoped strictly to
                // Copy scalars: a heap value (tensor, string, container, ...)
                // still moves on its proven last use, so a genuine
                // use-after-move of an owned resource stays rejected.
                let is_copy_scalar =
                    matches!(info.class, ValueClass::NonHeap(NonHeapKind::Scalar(_)));
                let movable = info.origin == OwnerOrigin::Owned
                    && !is_copy_scalar
                    && tail.is_some_and(|scope| depth >= scope);
                if movable {
                    self.moved.insert(owner);
                    Ok(Operand::move_(owner))
                } else {
                    let copy = self.copy(owner)?;
                    self.moved.insert(copy);
                    Ok(Operand::move_(copy))
                }
            }
            Value::Callback(owner) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name: self.owner_label(owner)?,
            }),
            Value::FunctionRef { name, ty } => {
                let owner = self.materialize_function_ref(&name, &ty)?;
                self.moved.insert(owner);
                Ok(Operand::move_(owner))
            }
        }
    }

    fn borrow(&mut self, value: Value) -> Result<Operand, OwnershipError> {
        match value {
            Value::Fresh(owner) => {
                // A fresh value projected out of a nested scope carries the
                // moved marker that suppressed that inner scope's terminal.
                // Borrowing it in the enclosing scope transfers liveness to
                // that scope just as binding it does, so its eventual terminal
                // remains scheduled after the borrow.
                self.moved.remove(&owner);
                self.register(owner)?;
                Ok(Operand::borrow(owner))
            }
            Value::Named(owner) | Value::Callback(owner) => Ok(Operand::borrow(owner)),
            Value::FunctionRef { name, ty } => {
                let owner = self.materialize_function_ref(&name, &ty)?;
                self.register(owner)?;
                Ok(Operand::borrow(owner))
            }
        }
    }

    fn bind(&mut self, name: &str, value: Value) -> Result<(), OwnershipError> {
        match value {
            Value::Fresh(owner) => {
                // A fresh value returned from a nested scope was marked moved
                // while crossing that scope edge. Binding it in the enclosing
                // scope establishes the new live owner there; leaving the
                // historical marker set would suppress its eventual terminal.
                self.moved.remove(&owner);
                self.register(owner)?;
                self.name_owner(owner, name)?;
                // Only producers with recorded callback provenance enter
                // callback scope. Ordinary returned function values retain
                // their owner until the C ABI rejects them (chelis#879).
                if self.callback_modes.contains_key(&owner) {
                    self.scope_mut()?
                        .names
                        .insert(name.to_string(), Place::Callback(owner));
                }
                Ok(())
            }
            Value::Named(owner) => {
                self.name_owner(owner, name)?;
                if self.callback_modes.contains_key(&owner) {
                    self.scope_mut()?
                        .names
                        .insert(name.to_string(), Place::Callback(owner));
                }
                Ok(())
            }
            Value::Callback(owner) => {
                self.scope_mut()?
                    .names
                    .insert(name.to_string(), Place::Callback(owner));
                Ok(())
            }
            Value::FunctionRef { name: function, ty } => {
                let owner = self.materialize_function_ref(&function, &ty)?;
                self.register(owner)?;
                self.name_owner(owner, name)
            }
        }
    }

    fn materialize_function_ref(
        &mut self,
        name: &str,
        ty: &ConcreteHostType,
    ) -> Result<OwnerId, OwnershipError> {
        let owner = self.mint(
            ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![name.to_string()],
        )?;
        self.emit(Op::Define {
            dest: owner,
            label: format!("function_ref:{name}"),
        });
        Ok(owner)
    }

    /// Add Phase 2 scope-exit terminals. Heap values use Drop; nonheap
    /// identities use a consuming discard so loop back-edges have exact live
    /// sets as well.
    fn exit_scope(&mut self) -> Result<(), OwnershipError> {
        let scope = self
            .scopes
            .pop()
            .ok_or_else(|| self.invariant("no scope to exit"))?;
        for owner in scope.owners.into_iter().rev() {
            if self.moved.contains(&owner) || self.info(owner)?.origin != OwnerOrigin::Owned {
                continue;
            }
            if self.info(owner)?.class.is_heap() {
                self.emit_scope_exit(Op::Drop {
                    owner: Operand::move_(owner),
                });
            } else {
                self.emit_scope_exit(Op::Discard { owner });
            }
            self.moved.insert(owner);
        }
        Ok(())
    }

    fn resolve_out(&mut self, value: Value, scope_depth: usize) -> Value {
        match value {
            Value::Named(owner)
                if self
                    .owner_depth
                    .get(&owner)
                    .is_some_and(|depth| *depth >= scope_depth)
                    && self
                        .owners
                        .get(&owner)
                        .is_some_and(|info| info.origin == OwnerOrigin::Owned) =>
            {
                self.emit(Op::Project {
                    source: Operand::borrow(owner),
                });
                self.moved.insert(owner);
                Value::Fresh(owner)
            }
            other => other,
        }
    }

    fn finish(
        self,
        kind: UnitKind,
        callable_body: Option<CallableBody>,
        entry: BlockId,
    ) -> Result<Unit, OwnershipError> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for builder in self.blocks {
            let terminator =
                builder
                    .terminator
                    .ok_or_else(|| OwnershipError::LoweringInvariant {
                        unit: self.unit_name.clone(),
                        detail: format!("block b{} has no terminator", builder.id.0),
                    })?;
            blocks.push(Block {
                id: builder.id,
                params: builder.params,
                ops: builder.ops,
                terminator,
            });
        }
        Ok(Unit {
            id: UnitId(self.unit_index as u32),
            name: self.unit_name,
            kind,
            schedule: ScheduleState::Phase2ScopeExit,
            callable_body,
            entry,
            blocks,
            owners: self.owners,
        })
    }

    fn lower_expr(
        &mut self,
        expr: &ConcreteHostExpr,
        tail: Option<usize>,
    ) -> Result<Value, OwnershipError> {
        self.with_expr_span(expr, |lowerer| {
            lowerer.with_site(HostSiteKind::Expression, |lowerer| {
                lowerer.lower_expr_at_site(expr, tail)
            })
        })
    }

    /// Record `expr`'s span for the owners minted while lowering it, saved and
    /// restored exactly as `active_site` is. A span-less node keeps the nearest
    /// enclosing span rather than clearing it: the enclosing source region
    /// still locates the owner, and host lowering leaves `span_id` empty on
    /// plenty of interior nodes (chelis#2122).
    ///
    /// Every path that lowers an expression goes through here: `lower_expr`,
    /// and `lower_direct_call_argument`, which reaches `lower_expr_at_site`
    /// without passing through it and can also return early for a raw literal.
    ///
    /// One owner class is knowingly outside it: `materialize_function_ref`
    /// mints from a `Value::FunctionRef` that outlives the expression scope
    /// that produced it, so such an owner takes the enclosing region (for
    /// example the whole tuple in `(identity, 1i64)`). Narrowing it would mean
    /// carrying a span on the `Value`, tracked as chelis#2319.
    fn with_expr_span<T>(
        &mut self,
        expr: &ConcreteHostExpr,
        action: impl FnOnce(&mut Self) -> Result<T, OwnershipError>,
    ) -> Result<T, OwnershipError> {
        let previous = self.active_span.clone();
        if expr.span_id.is_some() {
            self.active_span = expr.span_id.clone();
        }
        let result = action(self);
        self.active_span = previous;
        result
    }

    fn lower_expr_at_site(
        &mut self,
        expr: &ConcreteHostExpr,
        tail: Option<usize>,
    ) -> Result<Value, OwnershipError> {
        match &expr.kind {
            ConcreteHostExprKind::ResultClaimScope { body, .. } => self.lower_expr(body, tail),
            ConcreteHostExprKind::FormalIngress { value, .. } => self.lower_expr(value, tail),
            ConcreteHostExprKind::Int(value) => {
                self.define(&ConcreteHostType::Int64, format!("literal {value}"))
            }
            ConcreteHostExprKind::Float(value) => {
                self.define(&ConcreteHostType::Float64, format!("literal {value}"))
            }
            ConcreteHostExprKind::Bool(value) => {
                self.define(&ConcreteHostType::Bool, format!("literal {value}"))
            }
            ConcreteHostExprKind::String(value) => {
                self.define(&ConcreteHostType::String, format!("string {value:?}"))
            }
            ConcreteHostExprKind::Unit => self.define(&ConcreteHostType::Unit, "unit".to_string()),
            ConcreteHostExprKind::List(items, ty) => self.aggregate("list", items, ty),
            ConcreteHostExprKind::Tuple(items, ty) => self.aggregate("tuple", items, ty),
            ConcreteHostExprKind::AdtConstruct { ctor, fields, ty } => {
                self.aggregate(&format!("adt:{ctor}"), fields, ty)
            }
            ConcreteHostExprKind::Var(name, ty) => self.lower_var(name, ty),
            ConcreteHostExprKind::Call {
                function,
                args,
                arg_tys,
                ty,
            } => {
                let direct_specs = if crate::host::is_host_unresolved_marker(function)
                    || matches!(self.lookup(function), Some(Place::Callback(_)))
                {
                    None
                } else {
                    self.ctx
                        .signatures
                        .get(function)
                        .map(|signature| signature.params.clone())
                };
                let values = if let Some(specs) = direct_specs {
                    if args.len() != specs.len() {
                        return Err(OwnershipError::CallArityMismatch {
                            unit: self.unit_name.clone(),
                            callee: function.clone(),
                            supplied: args.len(),
                            declared: specs.len(),
                        });
                    }
                    if arg_tys.len() != specs.len() {
                        return Err(OwnershipError::CallArityMismatch {
                            unit: self.unit_name.clone(),
                            callee: function.clone(),
                            supplied: arg_tys.len(),
                            declared: specs.len(),
                        });
                    }
                    let mut values = Vec::with_capacity(args.len());
                    let mut instantiation = CallTypeInstantiation::default();
                    for (index, ((arg, arg_ty), spec)) in
                        args.iter().zip(arg_tys).zip(&specs).enumerate()
                    {
                        if !instantiation.admits(&spec.type_pattern, &spec.ty, arg_ty) {
                            return Err(OwnershipError::CallArgumentType {
                                unit: self.unit_name.clone(),
                                callee: function.clone(),
                                argument: index,
                                expected: render_type(&spec.ty),
                                actual: render_type(arg_ty),
                            });
                        }
                        values.push(self.lower_direct_call_argument(
                            DirectCallArgument {
                                function,
                                index,
                                expr: arg,
                                pattern: &spec.type_pattern,
                                formal: &spec.ty,
                                checked_slot: arg_ty,
                            },
                            &mut instantiation,
                        )?);
                    }
                    values
                } else {
                    args.iter()
                        .map(|arg| {
                            self.with_site(HostSiteKind::Argument, |lowerer| {
                                lowerer.lower_expr(arg, None)
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                self.lower_call(function, values, ty, tail)
            }
            ConcreteHostExprKind::SignatureEntry { plan, args } => {
                if plan.observations().nodes().len() != args.len() {
                    return Err(OwnershipError::CallArityMismatch {
                        unit: self.unit_name.clone(),
                        callee: "signature entry".into(),
                        supplied: args.len(),
                        declared: plan.observations().nodes().len(),
                    });
                }
                let mut operands = Vec::with_capacity(args.len());
                for (index, arg) in args.iter().enumerate() {
                    let actual = expr_type(arg);
                    if !matches!(actual, ConcreteHostType::Tensor(_)) {
                        return Err(OwnershipError::CallArgumentType {
                            unit: self.unit_name.clone(),
                            callee: "signature entry".into(),
                            argument: index,
                            expected: "tensor observation".into(),
                            actual: render_type(&actual),
                        });
                    }
                    let value = self.with_site(HostSiteKind::Argument, |lowerer| {
                        lowerer.lower_expr(arg, None)
                    })?;
                    operands.push(self.borrow(value)?);
                }
                self.apply(
                    &ConcreteHostType::Unit,
                    "signature entry".into(),
                    vec![super::ir::OwnershipUse::Borrow; operands.len()],
                    operands,
                )
            }
            ConcreteHostExprKind::Builtin { name, args, ty } => {
                if name == "copy" && args.len() == 1 {
                    let value = self.with_site(HostSiteKind::Argument, |lowerer| {
                        lowerer.lower_expr(&args[0], None)
                    })?;
                    let source = self.borrow(value)?;
                    let dest = self.copy(source.owner)?;
                    return Ok(Value::Fresh(dest));
                }
                if name == "debug" && args.len() == 1 {
                    let value = self.with_site(HostSiteKind::Argument, |lowerer| {
                        lowerer.lower_expr(&args[0], None)
                    })?;
                    let operand = self.borrow(value)?;
                    self.emit(Op::Apply {
                        dest: None,
                        label: format!("builtin:{name}"),
                        kind: ApplyKind::Intrinsic,
                        schema: OperationSchema::new(vec![super::ir::OwnershipUse::Borrow], None),
                        args: vec![operand],
                    });
                    // `debug` observes and returns the same logical value. It
                    // therefore differs from source `copy`, which lowers to
                    // `Op::Copy` and mints a fresh owner above.
                    return Ok(Value::Named(operand.owner));
                }
                let mut operands = Vec::with_capacity(args.len());
                let use_ = if name == "Some" {
                    super::ir::OwnershipUse::Move
                } else {
                    super::ir::OwnershipUse::Borrow
                };
                for arg in args {
                    let value = self.with_site(HostSiteKind::Argument, |lowerer| {
                        lowerer.lower_expr(arg, None)
                    })?;
                    operands.push(match use_ {
                        super::ir::OwnershipUse::Move => self.consume(value, None)?,
                        super::ir::OwnershipUse::Borrow => self.borrow(value)?,
                        super::ir::OwnershipUse::Clone => {
                            unreachable!("lowering chooses clone by consuming a named owner")
                        }
                    });
                }
                let value = self.apply(
                    ty,
                    format!("builtin:{name}"),
                    vec![use_; operands.len()],
                    operands,
                )?;
                // A closed key-builtin value has the checked operation's
                // affine argument modes. Record them at its producer so an
                // unrelated function value cannot acquire callback status
                // merely by sharing its concrete function type.
                if args.is_empty()
                    && crate::host_type_state::KeyBuiltinCallable::from_symbol(name).is_some()
                    && let ConcreteHostType::Function(params, _) = ty
                    && let Value::Fresh(owner) = &value
                {
                    self.callback_modes
                        .insert(*owner, vec![ParamMode::Owned; params.len()]);
                }
                Ok(value)
            }
            ConcreteHostExprKind::AdtFieldAccess {
                base,
                field_index,
                ty,
            } => {
                let base = self.lower_expr(base, None)?;
                let base = self.borrow(base)?;
                self.apply(
                    ty,
                    format!("field:{field_index}"),
                    vec![super::ir::OwnershipUse::Borrow],
                    vec![base],
                )
            }
            ConcreteHostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ty,
            } => self.lower_if(cond, then_expr, else_expr, ty),
            ConcreteHostExprKind::MatchOption {
                scrutinee,
                bind_name,
                some_expr,
                none_expr,
                ty,
            } => self.lower_match_option(scrutinee, bind_name, some_expr, none_expr, ty),
            ConcreteHostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ty,
            } => self.lower_match_adt(scrutinee, arms, default_expr.as_deref(), ty),
            ConcreteHostExprKind::Let { bindings, body, .. }
            | ConcreteHostExprKind::RetainedInvocation { bindings, body, .. } => {
                self.push_scope();
                let depth = self.depth();
                for binding in bindings {
                    self.with_site(HostSiteKind::Binding, |lowerer| {
                        let value = lowerer.lower_expr(&binding.value, None)?;
                        lowerer.bind(&binding.name, value)
                    })?;
                }
                let body_tail = Some(tail.map_or(depth, |outer| outer.min(depth)));
                let value = self.lower_expr(body, body_tail)?;
                let value = self.resolve_out(value, depth);
                self.exit_scope()?;
                Ok(value)
            }
            ConcreteHostExprKind::Map { callback, list, ty } => self.lower_map(callback, list, ty),
            ConcreteHostExprKind::Fold {
                callback,
                init,
                list,
                ty,
            } => self.lower_fold(callback, init, list, ty),
            ConcreteHostExprKind::TensorCall { helper, args, ty } => {
                self.lower_tensor_call(*helper, args, ty, tail)
            }
            ConcreteHostExprKind::Filter { callback, list, ty } => {
                self.lower_filter_like("filter_step", callback, list, ty)
            }
            ConcreteHostExprKind::Scan {
                callback,
                init,
                list,
                ty,
            } => self.lower_scan(callback, init, list, ty),
            ConcreteHostExprKind::Partition { callback, list, ty } => {
                self.lower_filter_like("partition_step", callback, list, ty)
            }
            ConcreteHostExprKind::FlatMap { callback, list, ty } => {
                self.lower_flat_map(callback, list, ty)
            }
        }
    }

    fn define(&mut self, ty: &ConcreteHostType, label: String) -> Result<Value, OwnershipError> {
        let dest = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.emit(Op::Define { dest, label });
        Ok(Value::Fresh(dest))
    }

    fn apply(
        &mut self,
        ty: &ConcreteHostType,
        label: String,
        expected: Vec<super::ir::OwnershipUse>,
        args: Vec<Operand>,
    ) -> Result<Value, OwnershipError> {
        self.apply_kind(ty, label, ApplyKind::Intrinsic, expected, args)
    }

    fn apply_kind(
        &mut self,
        ty: &ConcreteHostType,
        label: String,
        kind: ApplyKind,
        expected: Vec<super::ir::OwnershipUse>,
        args: Vec<Operand>,
    ) -> Result<Value, OwnershipError> {
        let dest = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let class = self.info(dest)?.class;
        let schema = OperationSchema::new(expected, Some(class));
        self.emit(Op::Apply {
            dest: Some(dest),
            label,
            kind,
            schema,
            args,
        });
        Ok(Value::Fresh(dest))
    }

    fn aggregate(
        &mut self,
        label: &str,
        items: &[ConcreteHostExpr],
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        self.classify_or_reject(ty, Placement::Value, None)?;
        let mut operands = Vec::with_capacity(items.len());
        for item in items {
            let value = self.with_site(HostSiteKind::Argument, |lowerer| {
                lowerer.lower_expr(item, None)
            })?;
            operands.push(self.consume(value, None)?);
        }
        self.apply(
            ty,
            label.to_string(),
            vec![super::ir::OwnershipUse::Move; operands.len()],
            operands,
        )
    }

    fn lower_var(&mut self, name: &str, ty: &ConcreteHostType) -> Result<Value, OwnershipError> {
        if let Some(place) = self.lookup(name) {
            return Ok(match place {
                Place::Owner(owner) => Value::Named(owner),
                Place::Callback(owner) => Value::Callback(owner),
            });
        }
        if name == "Nil" {
            return self.define(ty, "empty_list".to_string());
        }
        if name == "None" {
            return self.define(ty, "none".to_string());
        }
        if self.unit_name != ROOTS_UNIT && self.ctx.global_names.contains(name) {
            return self.capture(name);
        }
        if self.ctx.function_names.contains(name) {
            return Ok(Value::FunctionRef {
                name: name.to_string(),
                ty: ty.clone(),
            });
        }
        Err(OwnershipError::UnboundName {
            unit: self.unit_name.clone(),
            name: name.to_string(),
        })
    }

    /// Lazily extend the function's artifact adapter when a top-level binding
    /// is captured. Entry and body identities are distinct external borrows.
    fn capture(&mut self, name: &str) -> Result<Value, OwnershipError> {
        if let Some(owner) = self.captures.get(name) {
            return Ok(Value::Named(*owner));
        }
        let ty = self
            .ctx
            .host
            .globals
            .iter()
            .find(|binding| binding.name == name)
            .map(|binding| binding.ty.clone())
            .ok_or_else(|| OwnershipError::UnboundName {
                unit: self.unit_name.clone(),
                name: name.to_string(),
            })?;
        let adapter = self
            .adapter
            .ok_or_else(|| self.invariant("root unit cannot capture a global"))?;
        let entry_owner = self.mint(
            &ty,
            Placement::Value,
            OwnerOrigin::ExternalBorrow,
            Vec::new(),
        )?;
        let body_owner = self.mint(
            &ty,
            Placement::Value,
            OwnerOrigin::ExternalBorrow,
            vec![name.to_string()],
        )?;
        self.owner_depth.insert(entry_owner, 0);
        self.owner_depth.insert(body_owner, 0);
        self.blocks[adapter.entry.0 as usize]
            .params
            .push(BlockParam {
                owner: entry_owner,
                mode: ParamMode::EntryBorrow,
            });
        self.blocks[adapter.body.0 as usize]
            .params
            .push(BlockParam {
                owner: body_owner,
                mode: ParamMode::Borrowed,
            });
        match self.blocks[adapter.entry.0 as usize].terminator.as_mut() {
            Some(Terminator::Jump(edge)) => edge.args.push(Operand::borrow(entry_owner)),
            _ => return Err(self.invariant("function adapter has no jump to its body")),
        }
        let no_scope = self.invariant("capture has no function scope");
        self.scopes
            .first_mut()
            .ok_or(no_scope)?
            .names
            .insert(name.to_string(), Place::Owner(body_owner));
        self.captures.insert(name.to_string(), body_owner);
        Ok(Value::Named(body_owner))
    }

    /// Resolve a variable through the ownership environment instead of
    /// trusting its host-expression annotation. Call-site monomorphization can
    /// legitimately leave the generic spelling on that annotation, while the
    /// bound owner retains the concrete checked type.
    fn direct_call_argument_type(
        &self,
        expr: &ConcreteHostExpr,
    ) -> Result<ConcreteHostType, OwnershipError> {
        if let ConcreteHostExprKind::Var(name, _) = &expr.kind {
            if let Some(place) = self.lookup(name) {
                let owner = match place {
                    Place::Owner(owner) | Place::Callback(owner) => owner,
                };
                return Ok(self.info(owner)?.ty.clone());
            }
            if self.unit_name != ROOTS_UNIT
                && let Some(binding) = self
                    .ctx
                    .host
                    .globals
                    .iter()
                    .find(|binding| binding.name == *name)
            {
                return Ok(binding.ty.clone());
            }
        }
        Ok(match &expr.kind {
            ConcreteHostExprKind::Int(_) => ConcreteHostType::Scalar(Prim::Int32),
            ConcreteHostExprKind::Float(_) => ConcreteHostType::Scalar(Prim::F32),
            _ => expr_type(expr),
        })
    }

    /// Restore the checker-owned call-slot type that the concrete host IR's raw
    /// `Int`/`Float` lexical carriers do not retain. The slot has already been
    /// checked against the resolved callee formal. Every other expression
    /// carries its own concrete type and must be one consistent instantiation
    /// of that formal instead of being retagged at the call boundary.
    fn lower_direct_call_argument(
        &mut self,
        argument: DirectCallArgument<'_>,
        instantiation: &mut CallTypeInstantiation,
    ) -> Result<Value, OwnershipError> {
        let actual = self.direct_call_argument_type(argument.expr)?;
        self.with_site(HostSiteKind::Argument, |lowerer| {
            lowerer.with_site(HostSiteKind::Expression, |lowerer| {
                // The span covers BOTH ways out of this body: the raw-literal
                // admission below returns without reaching `lower_expr_at_site`
                // (chelis#2122 red team round 2).
                lowerer.with_expr_span(argument.expr, |lowerer| {
                    let raw_literal_allowed = match (&argument.expr.kind, argument.checked_slot) {
                        (ConcreteHostExprKind::Int(_), ConcreteHostType::Scalar(prim)) => {
                            prim.is_integer() || prim.is_float()
                        }
                        (ConcreteHostExprKind::Float(_), ConcreteHostType::Scalar(prim)) => {
                            prim.is_float()
                        }
                        _ => false,
                    };
                    if raw_literal_allowed {
                        let label = match &argument.expr.kind {
                            ConcreteHostExprKind::Int(value) => format!("literal {value}"),
                            ConcreteHostExprKind::Float(value) => format!("literal {value}"),
                            _ => unreachable!("raw literal admission is exhaustive"),
                        };
                        return lowerer.define(argument.checked_slot, label);
                    }

                    if !instantiation.admits(argument.pattern, argument.formal, &actual) {
                        return Err(OwnershipError::CallArgumentType {
                            unit: lowerer.unit_name.clone(),
                            callee: argument.function.to_string(),
                            argument: argument.index,
                            expected: render_type(argument.formal),
                            actual: render_type(&actual),
                        });
                    }
                    lowerer.lower_expr_at_site(argument.expr, None)
                })
            })
        })
    }

    fn lower_call(
        &mut self,
        function: &str,
        values: Vec<Value>,
        ty: &ConcreteHostType,
        tail: Option<usize>,
    ) -> Result<Value, OwnershipError> {
        // These two unspellable placeholders are typed negative evidence for
        // the target capability boundary. Ownership still has to account for
        // their exact argument/result payload so the pre-emission verifier is
        // total, but it must not turn any ordinary unknown callee into an
        // admissible call. The C ABI projection rejects the retained marker
        // before a raw call can be emitted.
        if crate::host::is_host_unresolved_marker(function) {
            let mut args = Vec::with_capacity(values.len());
            for value in values {
                args.push(self.borrow(value)?);
            }
            return self.apply(
                ty,
                "unresolved_call_placeholder".to_string(),
                vec![super::ir::OwnershipUse::Borrow; args.len()],
                args,
            );
        }
        let (label, kind, specs) = match self.lookup(function) {
            Some(Place::Callback(owner)) => {
                let params = match &self.info(owner)?.ty {
                    ConcreteHostType::Function(params, _) => params.clone(),
                    other => {
                        return Err(self.invariant(format!(
                            "callback '{function}' has non-function type '{}'",
                            render_type(other)
                        )));
                    }
                };
                let modes = self.callback_modes.get(&owner).cloned().ok_or_else(|| {
                    self.invariant(format!("callback %{owner:?} has no parameter modes"))
                })?;
                if modes.len() != params.len() {
                    return Err(self.invariant(format!(
                        "callback %{owner:?} has {} parameter modes for {} parameters",
                        modes.len(),
                        params.len()
                    )));
                }
                let specs = params
                    .into_iter()
                    .zip(modes)
                    .map(|(ty, mode)| ParamSpec {
                        mode,
                        type_pattern: FormalTypePattern::nominal(&ty),
                        ty,
                        callback_modes: None,
                    })
                    .collect();
                (
                    format!("call_callback:%{}", owner.0),
                    ApplyKind::IndirectCall,
                    specs,
                )
            }
            Some(Place::Owner(_)) | None => match self.ctx.signatures.get(function) {
                Some(signature) => {
                    let callee =
                        self.ctx
                            .function_units
                            .get(function)
                            .copied()
                            .ok_or_else(|| {
                                self.invariant(format!(
                                    "resolved function `{function}` has no structural unit identity"
                                ))
                            })?;
                    (
                        format!("call:{function}"),
                        ApplyKind::DirectCall { callee },
                        signature.params.clone(),
                    )
                }
                None => {
                    return Err(OwnershipError::UnknownCallee {
                        unit: self.unit_name.clone(),
                        callee: function.to_string(),
                    });
                }
            },
        };
        if values.len() != specs.len() {
            return Err(OwnershipError::CallArityMismatch {
                unit: self.unit_name.clone(),
                callee: function.to_string(),
                supplied: values.len(),
                declared: specs.len(),
            });
        }
        let mut args = Vec::with_capacity(values.len());
        let mut expected = Vec::with_capacity(values.len());
        for (value, spec) in values.into_iter().zip(&specs) {
            if matches!(spec.ty, ConcreteHostType::Function(_, _)) {
                let operand = match value {
                    Value::Callback(owner) => Operand::borrow(owner),
                    Value::FunctionRef { name, .. } => {
                        let owner = self.mint(
                            &spec.ty,
                            Placement::Parameter,
                            OwnerOrigin::Owned,
                            vec![name.clone()],
                        )?;
                        self.emit(Op::Define {
                            dest: owner,
                            label: format!("function_ref:{name}"),
                        });
                        self.moved.insert(owner);
                        Operand::move_(owner)
                    }
                    other => self.consume(other, None)?,
                };
                expected.push(operand.use_);
                args.push(operand);
                continue;
            }
            let mode = spec.mode.use_();
            expected.push(mode);
            args.push(match mode {
                super::ir::OwnershipUse::Move => self.consume(value, tail)?,
                super::ir::OwnershipUse::Borrow => self.borrow(value)?,
                super::ir::OwnershipUse::Clone => unreachable!("parameter modes never clone"),
            });
        }
        self.apply_kind(ty, label, kind, expected, args)
    }

    fn lower_join_arm(
        &mut self,
        expr: &ConcreteHostExpr,
        join: BlockId,
    ) -> Result<(), OwnershipError> {
        let depth = self.depth();
        let value = self.lower_expr(expr, Some(depth))?;
        let result = self.consume(value, Some(depth))?;
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: join,
            args: vec![result],
            terminals: Vec::new(),
        }))
    }

    fn lower_if(
        &mut self,
        cond: &ConcreteHostExpr,
        then_expr: &ConcreteHostExpr,
        else_expr: &ConcreteHostExpr,
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        let cond = self.lower_expr(cond, None)?;
        let cond = self.borrow(cond)?;
        let join_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let then_block = self.new_block(Vec::new());
        let else_block = self.new_block(Vec::new());
        let join = self.new_block(vec![BlockParam {
            owner: join_owner,
            mode: ParamMode::Owned,
        }]);
        self.set_terminator(Terminator::Branch {
            condition: cond,
            then_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: then_block,
                args: Vec::new(),
                terminals: Vec::new(),
            },
            else_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: else_block,
                args: Vec::new(),
                terminals: Vec::new(),
            },
        })?;
        self.record_edge(HostSiteKind::BranchEdge, then_block);
        self.record_edge(HostSiteKind::BranchEdge, else_block);

        // `moved` is a single set for the whole unit, but a branch has two
        // paths. Consuming an enclosing owner on one arm marks it moved
        // everywhere, so the enclosing scope exit emits no release for the arm
        // that did NOT consume it, and that owner reaches the join live on one
        // path only. Lower each arm from the same snapshot and reconcile, so
        // every owner that survives one arm and not the other carries its
        // release on the arm that dropped it (chelis#2477).
        let moved_before = self.moved.clone();
        let outer_owners: BTreeSet<OwnerId> = self.owners.keys().copied().collect();

        self.current = then_block;
        self.push_scope();
        self.lower_join_arm(then_expr, join)?;
        let moved_after_then = self.moved.clone();
        // An arm with nested control flow ends in a different block than the
        // one it started in, and the release belongs on the block that jumps
        // to the join. Emitting into the arm's entry block would place it
        // before that block's own terminator, so a later borrow inside the arm
        // would read a released owner.
        let then_end = self.current;

        self.moved = moved_before;
        self.current = else_block;
        self.push_scope();
        self.lower_join_arm(else_expr, join)?;
        let moved_after_else = self.moved.clone();
        let else_end = self.current;

        self.moved = moved_after_then
            .union(&moved_after_else)
            .copied()
            .collect::<BTreeSet<_>>();
        self.release_on_arm(
            then_end,
            &moved_after_else,
            &moved_after_then,
            &outer_owners,
        )?;
        self.release_on_arm(
            else_end,
            &moved_after_then,
            &moved_after_else,
            &outer_owners,
        )?;

        self.current = join;
        Ok(Value::Fresh(join_owner))
    }

    /// Release, in `arm`'s block, every pre-existing owner that the sibling
    /// arm consumed and this one did not. Restricted to owners that existed
    /// before the branch: anything minted inside an arm does not exist on the
    /// other path and its own scope exit already covers it.
    ///
    /// These are emitted with `ProvisionalScopeExit`, exactly like a scope
    /// exit, so `last_use::schedule` re-places them with every other
    /// provisional release rather than treating them as fixed.
    fn release_on_arm(
        &mut self,
        arm: BlockId,
        moved_by_sibling: &BTreeSet<OwnerId>,
        moved_by_arm: &BTreeSet<OwnerId>,
        outer_owners: &BTreeSet<OwnerId>,
    ) -> Result<(), OwnershipError> {
        let owed: Vec<OwnerId> = moved_by_sibling
            .difference(moved_by_arm)
            .copied()
            .filter(|owner| outer_owners.contains(owner))
            .filter(|owner| {
                self.owners
                    .get(owner)
                    .is_some_and(|info| info.origin == OwnerOrigin::Owned)
            })
            .collect();
        let restore = self.current;
        self.current = arm;
        for owner in owed {
            let heap = self
                .owners
                .get(&owner)
                .is_some_and(|info| info.class.is_heap());
            if heap {
                self.emit_scope_exit(Op::Drop {
                    owner: Operand::move_(owner),
                });
            } else {
                self.emit_scope_exit(Op::Discard { owner });
            }
        }
        self.current = restore;
        Ok(())
    }

    fn lower_match_option(
        &mut self,
        scrutinee: &ConcreteHostExpr,
        bind_name: &str,
        some_expr: &ConcreteHostExpr,
        none_expr: &ConcreteHostExpr,
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        let payload_ty = match expr_type(scrutinee) {
            ConcreteHostType::Option(inner) => *inner,
            other => {
                return Err(OwnershipError::MatchScrutineeNotOption {
                    unit: self.unit_name.clone(),
                    ty: render_type(&other),
                });
            }
        };
        let scrutinee = self.lower_expr(scrutinee, None)?;
        let scrutinee = self.borrow(scrutinee)?;
        let some = self.new_block(Vec::new());
        let none = self.new_block(Vec::new());
        let join_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let join = self.new_block(vec![BlockParam {
            owner: join_owner,
            mode: ParamMode::Owned,
        }]);
        self.set_terminator(Terminator::Match {
            scrutinee,
            arms: vec![
                Edge {
                    id: EdgeId::UNASSIGNED,
                    target: some,
                    args: Vec::new(),
                    terminals: Vec::new(),
                },
                Edge {
                    id: EdgeId::UNASSIGNED,
                    target: none,
                    args: Vec::new(),
                    terminals: Vec::new(),
                },
            ],
        })?;
        self.record_edge(HostSiteKind::MatchArm, some);
        self.record_edge(HostSiteKind::MatchArm, none);
        self.current = some;
        self.push_scope();
        let payload = self.mint(
            &payload_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![bind_name.to_string()],
        )?;
        self.sites.add(self.unit_index, HostSiteKind::Binding);
        self.emit(Op::Apply {
            dest: Some(payload),
            label: "option_payload".to_string(),
            kind: ApplyKind::Intrinsic,
            schema: OperationSchema::new(
                vec![super::ir::OwnershipUse::Borrow],
                Some(self.info(payload)?.class),
            ),
            args: vec![scrutinee],
        });
        self.register(payload)?;
        self.scope_mut()?
            .names
            .insert(bind_name.to_string(), Place::Owner(payload));
        self.lower_join_arm(some_expr, join)?;
        self.current = none;
        self.push_scope();
        self.lower_join_arm(none_expr, join)?;
        self.current = join;
        Ok(Value::Fresh(join_owner))
    }

    fn lower_match_adt(
        &mut self,
        scrutinee: &ConcreteHostExpr,
        arms: &[ConcreteHostMatchArm],
        default_expr: Option<&ConcreteHostExpr>,
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        let scrutinee = self.lower_expr(scrutinee, None)?;
        let scrutinee = self.borrow(scrutinee)?;
        let join_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let join = self.new_block(vec![BlockParam {
            owner: join_owner,
            mode: ParamMode::Owned,
        }]);
        let arm_blocks = (0..arms.len())
            .map(|_| self.new_block(Vec::new()))
            .collect::<Vec<_>>();
        let default_block = default_expr.map(|_| self.new_block(Vec::new()));
        let mut edges = arm_blocks
            .iter()
            .map(|target| Edge {
                id: EdgeId::UNASSIGNED,
                target: *target,
                args: Vec::new(),
                terminals: Vec::new(),
            })
            .collect::<Vec<_>>();
        if let Some(target) = default_block {
            edges.push(Edge {
                id: EdgeId::UNASSIGNED,
                target,
                args: Vec::new(),
                terminals: Vec::new(),
            });
        }
        if edges.is_empty() {
            return Err(self.invariant("ADT match has no arms"));
        }
        self.set_terminator(Terminator::Match {
            scrutinee,
            arms: edges,
        })?;
        for target in arm_blocks.iter().copied().chain(default_block) {
            self.record_edge(HostSiteKind::MatchArm, target);
        }
        for (arm, block) in arms.iter().zip(arm_blocks) {
            self.current = block;
            self.push_scope();
            for binding in &arm.bindings {
                self.sites.add(self.unit_index, HostSiteKind::Binding);
                let owner = self.mint(
                    &binding.ty,
                    Placement::Value,
                    OwnerOrigin::Owned,
                    vec![binding.name.clone()],
                )?;
                self.emit(Op::Apply {
                    dest: Some(owner),
                    label: format!("adt_payload:{}:{}", arm.ctor, binding.field_index),
                    kind: ApplyKind::Intrinsic,
                    schema: OperationSchema::new(
                        vec![super::ir::OwnershipUse::Borrow],
                        Some(self.info(owner)?.class),
                    ),
                    args: vec![scrutinee],
                });
                self.register(owner)?;
                self.scope_mut()?
                    .names
                    .insert(binding.name.clone(), Place::Owner(owner));
            }
            self.lower_join_arm(&arm.expr, join)?;
        }
        if let (Some(expr), Some(block)) = (default_expr, default_block) {
            self.current = block;
            self.push_scope();
            self.lower_join_arm(expr, join)?;
        }
        self.current = join;
        Ok(Value::Fresh(join_owner))
    }

    fn callback_param_names(callback: &ConcreteHostCallback) -> Vec<String> {
        match &callback.kind {
            ConcreteHostCallbackKind::Named { params, .. }
            | ConcreteHostCallbackKind::Inline { params, .. } => {
                params.iter().map(|param| param.name.clone()).collect()
            }
        }
    }

    fn lower_callback_body(
        &mut self,
        callback: &ConcreteHostCallback,
        params: &[OwnerId],
    ) -> Result<Value, OwnershipError> {
        let depth = self.depth();
        match &callback.kind {
            ConcreteHostCallbackKind::Inline { body, .. } => self.lower_expr(body, Some(depth)),
            ConcreteHostCallbackKind::Named { function, .. } => {
                let values = params.iter().map(|owner| Value::Named(*owner)).collect();
                self.lower_call(function, values, &callback.ret_ty, Some(depth))
            }
        }
    }

    fn list_element_type(
        &self,
        list: &ConcreteHostExpr,
    ) -> Result<ConcreteHostType, OwnershipError> {
        match expr_type(list) {
            ConcreteHostType::List(inner) => Ok(*inner),
            other => Err(OwnershipError::LoopListNotList {
                unit: self.unit_name.clone(),
                ty: render_type(&other),
            }),
        }
    }

    fn lower_fold(
        &mut self,
        callback: &ConcreteHostCallback,
        init: &ConcreteHostExpr,
        list: &ConcreteHostExpr,
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        self.classify_or_reject(ty, Placement::Value, None)?;
        let names = Self::callback_param_names(callback);
        if names.len() != 2 {
            return Err(self.invariant(format!(
                "fold callback declares {} parameters, expected 2",
                names.len()
            )));
        }
        let element_ty = self.list_element_type(list)?;
        let init = self.lower_expr(init, None)?;
        let init = self.consume(init, None)?;
        let list = self.lower_expr(list, None)?;
        let list = self.borrow(list)?;
        let header_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let header = self.new_block(vec![BlockParam {
            owner: header_owner,
            mode: ParamMode::Owned,
        }]);
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: header,
            args: vec![init],
            terminals: Vec::new(),
        }))?;
        self.push_scope();
        self.sites.add(self.unit_index, HostSiteKind::Binding);
        let acc = self.mint(
            ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![names[0].clone()],
        )?;
        self.register(acc)?;
        self.scope_mut()?
            .names
            .insert(names[0].clone(), Place::Owner(acc));
        let body = self.new_block(vec![BlockParam {
            owner: acc,
            mode: ParamMode::Owned,
        }]);
        let exit_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let exit = self.new_block(vec![BlockParam {
            owner: exit_owner,
            mode: ParamMode::Owned,
        }]);
        self.current = header;
        self.set_terminator(Terminator::Loop {
            list,
            body_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: body,
                args: vec![Operand::move_(header_owner)],
                terminals: Vec::new(),
            },
            exit_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: exit,
                args: vec![Operand::move_(header_owner)],
                terminals: Vec::new(),
            },
        })?;
        self.record_edge(HostSiteKind::LoopEdge, body);
        self.record_edge(HostSiteKind::LoopEdge, exit);
        self.current = body;
        let element = self.mint(
            &element_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![names[1].clone()],
        )?;
        self.sites.add(self.unit_index, HostSiteKind::Binding);
        self.emit(Op::LoopItem {
            dest: element,
            list,
        });
        self.register(element)?;
        self.scope_mut()?
            .names
            .insert(names[1].clone(), Place::Owner(element));
        let depth = self.depth();
        let next = self.lower_callback_body(callback, &[acc, element])?;
        let next = self.consume(next, Some(depth))?;
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: header,
            args: vec![next],
            terminals: Vec::new(),
        }))?;
        self.current = exit;
        Ok(Value::Fresh(exit_owner))
    }

    fn lower_map(
        &mut self,
        callback: &ConcreteHostCallback,
        list: &ConcreteHostExpr,
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        self.lower_map_step(callback, list, ty, "list_push")
    }

    fn lower_map_step(
        &mut self,
        callback: &ConcreteHostCallback,
        list: &ConcreteHostExpr,
        ty: &ConcreteHostType,
        step: &str,
    ) -> Result<Value, OwnershipError> {
        self.classify_or_reject(ty, Placement::Value, None)?;
        let names = Self::callback_param_names(callback);
        if names.len() != 1 {
            return Err(self.invariant(format!(
                "map callback declares {} parameters, expected 1",
                names.len()
            )));
        }
        let element_ty = self.list_element_type(list)?;
        let list = self.lower_expr(list, None)?;
        let list = self.borrow(list)?;
        let seed = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.emit(Op::Define {
            dest: seed,
            label: "empty_list".to_string(),
        });
        self.moved.insert(seed);
        let header_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let header = self.new_block(vec![BlockParam {
            owner: header_owner,
            mode: ParamMode::Owned,
        }]);
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: header,
            args: vec![Operand::move_(seed)],
            terminals: Vec::new(),
        }))?;
        self.push_scope();
        let acc = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.register(acc)?;
        let body = self.new_block(vec![BlockParam {
            owner: acc,
            mode: ParamMode::Owned,
        }]);
        let exit_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let exit = self.new_block(vec![BlockParam {
            owner: exit_owner,
            mode: ParamMode::Owned,
        }]);
        self.current = header;
        self.set_terminator(Terminator::Loop {
            list,
            body_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: body,
                args: vec![Operand::move_(header_owner)],
                terminals: Vec::new(),
            },
            exit_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: exit,
                args: vec![Operand::move_(header_owner)],
                terminals: Vec::new(),
            },
        })?;
        self.record_edge(HostSiteKind::LoopEdge, body);
        self.record_edge(HostSiteKind::LoopEdge, exit);
        self.current = body;
        let element = self.mint(
            &element_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![names[0].clone()],
        )?;
        self.sites.add(self.unit_index, HostSiteKind::Binding);
        self.emit(Op::LoopItem {
            dest: element,
            list,
        });
        self.register(element)?;
        self.scope_mut()?
            .names
            .insert(names[0].clone(), Place::Owner(element));
        let depth = self.depth();
        let item = self.lower_callback_body(callback, &[element])?;
        let item = self.consume(item, Some(depth))?;
        let carried = self.consume(Value::Named(acc), Some(depth))?;
        let next = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.emit(Op::Apply {
            dest: Some(next),
            label: step.to_string(),
            kind: ApplyKind::Intrinsic,
            schema: OperationSchema::new(
                vec![super::ir::OwnershipUse::Move, super::ir::OwnershipUse::Move],
                Some(self.info(next)?.class),
            ),
            args: vec![carried, item],
        });
        self.moved.insert(next);
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: header,
            args: vec![Operand::move_(next)],
            terminals: Vec::new(),
        }))?;
        self.current = exit;
        Ok(Value::Fresh(exit_owner))
    }

    fn lower_filter_like(
        &mut self,
        step: &str,
        callback: &ConcreteHostCallback,
        list: &ConcreteHostExpr,
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        self.classify_or_reject(ty, Placement::Value, None)?;
        let names = Self::callback_param_names(callback);
        if names.len() != 1 {
            return Err(self.invariant(format!(
                "{step} callback declares {} parameters, expected 1",
                names.len()
            )));
        }
        let element_ty = self.list_element_type(list)?;
        let list = self.lower_expr(list, None)?;
        let list = self.borrow(list)?;
        let seed = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.emit(Op::Define {
            dest: seed,
            label: format!("{step}:empty"),
        });
        self.moved.insert(seed);
        let header_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let header = self.new_block(vec![BlockParam {
            owner: header_owner,
            mode: ParamMode::Owned,
        }]);
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: header,
            args: vec![Operand::move_(seed)],
            terminals: Vec::new(),
        }))?;
        self.push_scope();
        let acc = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.register(acc)?;
        let body = self.new_block(vec![BlockParam {
            owner: acc,
            mode: ParamMode::Owned,
        }]);
        let exit_owner = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let exit = self.new_block(vec![BlockParam {
            owner: exit_owner,
            mode: ParamMode::Owned,
        }]);
        self.current = header;
        self.set_terminator(Terminator::Loop {
            list,
            body_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: body,
                args: vec![Operand::move_(header_owner)],
                terminals: Vec::new(),
            },
            exit_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: exit,
                args: vec![Operand::move_(header_owner)],
                terminals: Vec::new(),
            },
        })?;
        self.record_edge(HostSiteKind::LoopEdge, body);
        self.record_edge(HostSiteKind::LoopEdge, exit);
        self.current = body;
        let element = self.mint(
            &element_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![names[0].clone()],
        )?;
        self.sites.add(self.unit_index, HostSiteKind::Binding);
        self.emit(Op::LoopItem {
            dest: element,
            list,
        });
        self.register(element)?;
        self.scope_mut()?
            .names
            .insert(names[0].clone(), Place::Owner(element));
        let depth = self.depth();
        // The step keeps the item after the predicate reads it, so the
        // predicate runs one scope deeper than the item: no use of the item
        // there is its last, and a consuming use, such as a by-value call to a
        // named definition, receives its own copy (chelis#2577).
        self.push_scope();
        let predicate_depth = self.depth();
        let predicate = self.lower_callback_body(callback, &[element])?;
        let predicate = self.consume(predicate, Some(predicate_depth))?;
        self.exit_scope()?;
        let item = self.consume(Value::Named(element), Some(depth))?;
        let carried = self.consume(Value::Named(acc), Some(depth))?;
        let next = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let next_class = self.info(next)?.class;
        self.emit(Op::Apply {
            dest: Some(next),
            label: step.to_string(),
            kind: ApplyKind::Intrinsic,
            schema: OperationSchema::new(
                vec![
                    super::ir::OwnershipUse::Move,
                    super::ir::OwnershipUse::Move,
                    super::ir::OwnershipUse::Move,
                ],
                Some(next_class),
            ),
            args: vec![carried, item, predicate],
        });
        self.moved.insert(next);
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: header,
            args: vec![Operand::move_(next)],
            terminals: Vec::new(),
        }))?;
        self.current = exit;
        Ok(Value::Fresh(exit_owner))
    }

    fn lower_flat_map(
        &mut self,
        callback: &ConcreteHostCallback,
        list: &ConcreteHostExpr,
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        // FlatMap has the same carried list owner as Map; the callback result
        // is itself a list and the step consumes it through concatenation.
        self.lower_map_step(callback, list, ty, "list_extend")
    }

    fn lower_scan(
        &mut self,
        callback: &ConcreteHostCallback,
        init: &ConcreteHostExpr,
        list: &ConcreteHostExpr,
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        self.classify_or_reject(ty, Placement::Value, None)?;
        let names = Self::callback_param_names(callback);
        if names.len() != 2 {
            return Err(self.invariant(format!(
                "scan callback declares {} parameters, expected 2",
                names.len()
            )));
        }
        let element_ty = self.list_element_type(list)?;
        let state_ty = expr_type(init);
        let init = self.lower_expr(init, None)?;
        let init = self.consume(init, None)?;
        let list = self.lower_expr(list, None)?;
        let list = self.borrow(list)?;
        let output_seed = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.emit(Op::Define {
            dest: output_seed,
            label: "scan:empty".to_string(),
        });
        self.moved.insert(output_seed);
        let header_state =
            self.mint(&state_ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let header_output = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let header = self.new_block(vec![
            BlockParam {
                owner: header_state,
                mode: ParamMode::Owned,
            },
            BlockParam {
                owner: header_output,
                mode: ParamMode::Owned,
            },
        ]);
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: header,
            args: vec![init, Operand::move_(output_seed)],
            terminals: Vec::new(),
        }))?;
        self.push_scope();
        let body_state = self.mint(
            &state_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![names[0].clone()],
        )?;
        self.sites.add(self.unit_index, HostSiteKind::Binding);
        let body_output = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.register(body_state)?;
        self.register(body_output)?;
        let body = self.new_block(vec![
            BlockParam {
                owner: body_state,
                mode: ParamMode::Owned,
            },
            BlockParam {
                owner: body_output,
                mode: ParamMode::Owned,
            },
        ]);
        let exit_state = self.mint(&state_ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let exit_output = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let exit = self.new_block(vec![
            BlockParam {
                owner: exit_state,
                mode: ParamMode::Owned,
            },
            BlockParam {
                owner: exit_output,
                mode: ParamMode::Owned,
            },
        ]);
        self.current = header;
        self.set_terminator(Terminator::Loop {
            list,
            body_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: body,
                args: vec![Operand::move_(header_state), Operand::move_(header_output)],
                terminals: Vec::new(),
            },
            exit_edge: Edge {
                id: EdgeId::UNASSIGNED,
                target: exit,
                args: vec![Operand::move_(header_state), Operand::move_(header_output)],
                terminals: Vec::new(),
            },
        })?;
        self.record_edge(HostSiteKind::LoopEdge, body);
        self.record_edge(HostSiteKind::LoopEdge, exit);
        self.current = body;
        self.scope_mut()?
            .names
            .insert(names[0].clone(), Place::Owner(body_state));
        let element = self.mint(
            &element_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![names[1].clone()],
        )?;
        self.sites.add(self.unit_index, HostSiteKind::Binding);
        self.emit(Op::LoopItem {
            dest: element,
            list,
        });
        self.register(element)?;
        self.scope_mut()?
            .names
            .insert(names[1].clone(), Place::Owner(element));
        let depth = self.depth();
        let next_state = self.lower_callback_body(callback, &[body_state, element])?;
        let next_state = self.consume(next_state, Some(depth))?;
        let output_item = self.copy(next_state.owner)?;
        self.moved.insert(output_item);
        let carried_output = self.consume(Value::Named(body_output), Some(depth))?;
        let next_output = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        let output_class = self.info(next_output)?.class;
        self.emit(Op::Apply {
            dest: Some(next_output),
            label: "list_push".to_string(),
            kind: ApplyKind::Intrinsic,
            schema: OperationSchema::new(
                vec![super::ir::OwnershipUse::Move, super::ir::OwnershipUse::Move],
                Some(output_class),
            ),
            args: vec![carried_output, Operand::move_(output_item)],
        });
        self.moved.insert(next_output);
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump(Edge {
            id: EdgeId::UNASSIGNED,
            target: header,
            args: vec![next_state, Operand::move_(next_output)],
            terminals: Vec::new(),
        }))?;
        self.current = exit;
        if self.info(exit_state)?.class.is_heap() {
            self.emit(Op::Drop {
                owner: Operand::move_(exit_state),
            });
        } else {
            self.emit(Op::Discard { owner: exit_state });
        }
        self.moved.insert(exit_state);
        Ok(Value::Fresh(exit_output))
    }

    fn lower_tensor_call(
        &mut self,
        helper: usize,
        args: &[ConcreteHostExpr],
        ty: &ConcreteHostType,
        tail: Option<usize>,
    ) -> Result<Value, OwnershipError> {
        let Some(helper_ref) = self.helpers.get(helper) else {
            return Err(self.invariant(format!("tensor helper {helper} does not exist")));
        };
        if helper_ref.identity_input().is_some() && args.len() == 1 {
            return self.lower_expr(&args[0], tail);
        }
        let mut operands = Vec::with_capacity(args.len());
        for arg in args {
            let value = self.with_site(HostSiteKind::Argument, |lowerer| {
                lowerer.lower_expr(arg, None)
            })?;
            operands.push(self.borrow(value)?);
        }
        self.apply(
            ty,
            format!("tensor:{}", helper_ref.name),
            vec![super::ir::OwnershipUse::Borrow; operands.len()],
            operands,
        )
    }
}

fn expr_type(expr: &ConcreteHostExpr) -> ConcreteHostType {
    match &expr.kind {
        ConcreteHostExprKind::Int(_) => ConcreteHostType::Int64,
        ConcreteHostExprKind::Float(_) => ConcreteHostType::Float64,
        ConcreteHostExprKind::Bool(_) => ConcreteHostType::Bool,
        ConcreteHostExprKind::String(_) => ConcreteHostType::String,
        ConcreteHostExprKind::Unit | ConcreteHostExprKind::SignatureEntry { .. } => {
            ConcreteHostType::Unit
        }
        ConcreteHostExprKind::List(_, ty)
        | ConcreteHostExprKind::Tuple(_, ty)
        | ConcreteHostExprKind::Var(_, ty)
        | ConcreteHostExprKind::Call { ty, .. }
        | ConcreteHostExprKind::Builtin { ty, .. }
        | ConcreteHostExprKind::AdtConstruct { ty, .. }
        | ConcreteHostExprKind::AdtFieldAccess { ty, .. }
        | ConcreteHostExprKind::If { ty, .. }
        | ConcreteHostExprKind::MatchOption { ty, .. }
        | ConcreteHostExprKind::MatchAdt { ty, .. }
        | ConcreteHostExprKind::Let { ty, .. }
        | ConcreteHostExprKind::RetainedInvocation { ty, .. }
        | ConcreteHostExprKind::Map { ty, .. }
        | ConcreteHostExprKind::Filter { ty, .. }
        | ConcreteHostExprKind::Fold { ty, .. }
        | ConcreteHostExprKind::Scan { ty, .. }
        | ConcreteHostExprKind::Partition { ty, .. }
        | ConcreteHostExprKind::FlatMap { ty, .. }
        | ConcreteHostExprKind::TensorCall { ty, .. }
        | ConcreteHostExprKind::ResultClaimScope { ty, .. }
        | ConcreteHostExprKind::FormalIngress { ty, .. } => ty.clone(),
    }
}

/// Consume the manifest into the exact host payload. Callable roots are
/// represented by compiler-owned observation bindings before site identities
/// are assigned, so ownership sinks never rediscover them by name.
pub(super) fn materialize_manifest_roots(
    host: &mut ConcreteHostProgram,
    manifest: &RootManifest,
) -> Result<Vec<Option<String>>, OwnershipError> {
    let mut callable_bindings = BTreeMap::<String, String>::new();
    let mut result = Vec::with_capacity(manifest.entries.len());
    for (manifest_index, root) in manifest.entries.iter().enumerate() {
        if root.lane != Lane::Host {
            result.push(None);
            continue;
        }
        let expected_display = display_root(root);
        if let Some(binding) = host.globals.iter().find(|binding| {
            binding.display_roots.iter().any(|display| {
                display.name == expected_display.name && display.path == expected_display.path
            })
        }) {
            callable_bindings
                .entry(root.def_name.clone())
                .or_insert_with(|| binding.name.clone());
            result.push(Some(binding.name.clone()));
            continue;
        }
        if let Some(binding) = host
            .globals
            .iter_mut()
            .find(|binding| binding.name == root.def_name)
        {
            binding.display_name = None;
            binding.display_roots.push(expected_display);
            result.push(Some(binding.name.clone()));
            continue;
        }
        if let Some(binding_name) = callable_bindings.get(&root.def_name) {
            let binding = host
                .globals
                .iter_mut()
                .find(|binding| binding.name == *binding_name)
                .expect("materialized callable binding remains in payload");
            binding.display_roots.push(display_root(root));
            result.push(Some(binding_name.clone()));
            continue;
        }
        let Some(function) = host
            .functions
            .iter()
            .find(|function| function.name == root.def_name)
        else {
            return Err(OwnershipError::ManifestRootWithoutBinding {
                root: root.name.clone(),
                def_name: root.def_name.clone(),
            });
        };
        if !function.params.is_empty() {
            return Err(OwnershipError::ManifestRootWithoutBinding {
                root: root.name.clone(),
                def_name: root.def_name.clone(),
            });
        }
        let function_name = function.name.clone();
        let ty = function.ret_ty.clone();
        let mut binding_name = format!("__chelis_manifest_observation_{manifest_index}");
        while host
            .globals
            .iter()
            .any(|binding| binding.name == binding_name)
            || host
                .functions
                .iter()
                .any(|function| function.name == binding_name)
        {
            binding_name.push('_');
        }
        host.globals.push(HostBinding {
            name: binding_name.clone(),
            display_name: None,
            display_roots: vec![display_root(root)],
            ty: ty.clone(),
            value: HostExpr::new(HostExprKind::Call {
                function: function_name,
                args: Vec::new(),
                arg_tys: Vec::new(),
                ty,
            }),
        });
        callable_bindings.insert(root.def_name.clone(), binding_name.clone());
        result.push(Some(binding_name));
    }
    Ok(result)
}

fn display_root(root: &chelis_types::manifest::RootEntry) -> HostDisplayRoot {
    let short_def = if chelis_types::is_linker_format_name(&root.def_name) {
        chelis_types::demangle_ident(&root.def_name)
    } else {
        root.def_name.clone()
    };
    let suffix = root
        .name
        .strip_prefix(root.def_name.as_str())
        .map_or("", |suffix| suffix);
    HostDisplayRoot {
        name: format!("{short_def}{suffix}"),
        path: root.path.clone(),
    }
}

fn lower_roots(
    ctx: &Context<'_>,
    manifest: &RootManifest,
    root_bindings: &[Option<String>],
    sites: &mut HostSiteBuilder,
    unit_index: usize,
) -> Result<Unit, OwnershipError> {
    if manifest.entries.len() != root_bindings.len() {
        return Err(OwnershipError::LoweringInvariant {
            unit: ROOTS_UNIT.to_string(),
            detail: "manifest/root-binding plan length mismatch".to_string(),
        });
    }
    let mut lowerer = UnitLowerer::new(
        ctx,
        ROOTS_UNIT,
        &ctx.host.global_tensor_helpers,
        sites,
        unit_index,
    );
    let entry = lowerer.new_block(Vec::new());
    lowerer.current = entry;
    lowerer.push_scope();
    for binding in &ctx.host.globals {
        lowerer.with_site(HostSiteKind::Binding, |lowerer| {
            let value = lowerer.lower_expr(&binding.value, None)?;
            lowerer.bind(&binding.name, value)
        })?;
    }
    let mut sinks = Vec::new();
    for (manifest_index, (root, binding_name)) in
        manifest.entries.iter().zip(root_bindings).enumerate()
    {
        let Some(binding_name) = binding_name else {
            continue;
        };
        match lowerer.lookup(binding_name) {
            Some(Place::Owner(owner)) => sinks.push((manifest_index, root, owner)),
            Some(Place::Callback(_)) | None => {
                return Err(OwnershipError::ManifestRootWithoutBinding {
                    root: root.name.clone(),
                    def_name: root.def_name.clone(),
                });
            }
        }
    }
    for (manifest_index, root, owner) in &sinks {
        lowerer.with_site(HostSiteKind::ManifestRoot, |lowerer| {
            let owner = lowerer.copy(*owner)?;
            lowerer.moved.insert(owner);
            lowerer.sites.record(
                lowerer.active_site.expect("root site"),
                HostSiteAction::Root {
                    unit: lowerer.unit_index,
                    manifest_index: Some(*manifest_index),
                    owner,
                },
            );
            lowerer.emit(Op::RootConsume {
                root: root.name.clone(),
                owner: Operand::move_(owner),
            });
            Ok(())
        })?;
    }
    let mut unmanifested_displays = Vec::new();
    for binding in &ctx.host.globals {
        for display in &binding.display_roots {
            let manifested = manifest.entries.iter().any(|root| {
                let selected = binding.name == root.def_name
                    || matches!(
                        &binding.value.kind,
                        ConcreteHostExprKind::Call { function, args, .. }
                            if function == &root.def_name && args.is_empty()
                    );
                root.lane == Lane::Host && selected && display_root(root) == *display
            });
            if !manifested {
                unmanifested_displays.push((binding.name.clone(), display.name.clone()));
            }
        }
        if binding.display_roots.is_empty()
            && let Some(display_name) = &binding.display_name
        {
            unmanifested_displays.push((binding.name.clone(), display_name.clone()));
        }
    }
    for (binding_name, display_name) in unmanifested_displays {
        let Some(Place::Owner(source)) = lowerer.lookup(&binding_name) else {
            return Err(OwnershipError::ManifestRootWithoutBinding {
                root: display_name.clone(),
                def_name: binding_name,
            });
        };
        lowerer.with_site(HostSiteKind::ManifestRoot, |lowerer| {
            let owner = lowerer.copy(source)?;
            lowerer.moved.insert(owner);
            lowerer.sites.record(
                lowerer.active_site.expect("display root site"),
                HostSiteAction::Root {
                    unit: lowerer.unit_index,
                    manifest_index: None,
                    owner,
                },
            );
            lowerer.emit(Op::RootConsume {
                root: display_name.clone(),
                owner: Operand::move_(owner),
            });
            Ok(())
        })?;
    }
    lowerer.with_site(HostSiteKind::FunctionReturn, |lowerer| {
        lowerer.exit_scope()?;
        lowerer.set_terminator(Terminator::Exit)
    })?;
    lowerer.finish(UnitKind::Roots, None, entry)
}

fn lower_function(
    ctx: &Context<'_>,
    function: &ConcreteHostFunction,
    sites: &mut HostSiteBuilder,
    unit_index: usize,
) -> Result<Unit, OwnershipError> {
    let signature =
        ctx.signatures
            .get(&function.name)
            .ok_or_else(|| OwnershipError::MissingSignature {
                function: function.name.clone(),
            })?;
    let mut lowerer = UnitLowerer::new(
        ctx,
        &function.name,
        &function.tensor_helpers,
        sites,
        unit_index,
    );
    let authored = function.origin == HostFunctionOrigin::Authored;
    let mut entry_params = Vec::with_capacity(function.params.len());
    let mut body_params = Vec::with_capacity(function.params.len());
    for (param, spec) in function.params.iter().zip(&signature.params) {
        lowerer.sites.add(lowerer.unit_index, HostSiteKind::Binding);
        let class =
            lowerer.classify_or_reject(&spec.ty, Placement::Parameter, Some(&param.name))?;
        let heap = class.is_heap();
        if authored {
            let entry_owner = lowerer.mint(
                &spec.ty,
                Placement::Parameter,
                if heap {
                    OwnerOrigin::ExternalBorrow
                } else {
                    OwnerOrigin::Owned
                },
                Vec::new(),
            )?;
            entry_params.push(BlockParam {
                owner: entry_owner,
                mode: if heap {
                    ParamMode::EntryBorrow
                } else {
                    ParamMode::Owned
                },
            });
        }
        let body_mode = if heap { spec.mode } else { ParamMode::Owned };
        let body_origin = match body_mode {
            ParamMode::Owned => OwnerOrigin::Owned,
            // A borrowed formal is external to the function unit whether the
            // function has an authored ABI adapter or is a translation-unit
            // internal specialization. Its borrow spans every control-flow
            // block in that invocation. Shorter-lived projections use an
            // explicit `BorrowedFrom` provenance edge.
            ParamMode::Borrowed | ParamMode::EntryBorrow => OwnerOrigin::ExternalBorrow,
        };
        let body_owner = lowerer.mint(
            &spec.ty,
            Placement::Parameter,
            body_origin,
            vec![param.name.clone()],
        )?;
        if let Some(modes) = &spec.callback_modes {
            lowerer.callback_modes.insert(body_owner, modes.clone());
        }
        body_params.push(BlockParam {
            owner: body_owner,
            mode: body_mode,
        });
    }
    let (entry, body) = if authored {
        let entry = lowerer.new_block(entry_params.clone());
        let body = lowerer.new_block(body_params.clone());
        lowerer.adapter = Some(AdapterBlocks { entry, body });
        lowerer.current = entry;
        lowerer.with_site(HostSiteKind::FunctionEntry, |lowerer| {
            let mut args = Vec::with_capacity(entry_params.len());
            for (entry_param, spec) in entry_params.iter().zip(&signature.params) {
                lowerer
                    .sites
                    .add(lowerer.unit_index, HostSiteKind::Argument);
                args.push(match (entry_param.mode, spec.mode) {
                    (ParamMode::EntryBorrow, ParamMode::Owned) => {
                        let copy = lowerer.copy(entry_param.owner)?;
                        Operand::move_(copy)
                    }
                    (ParamMode::EntryBorrow, ParamMode::Borrowed | ParamMode::EntryBorrow) => {
                        Operand::borrow(entry_param.owner)
                    }
                    (ParamMode::Owned | ParamMode::Borrowed, _) => {
                        Operand::move_(entry_param.owner)
                    }
                });
            }
            lowerer.set_terminator(Terminator::Jump(Edge {
                id: EdgeId::UNASSIGNED,
                target: body,
                args,
                terminals: Vec::new(),
            }))
        })?;
        (entry, body)
    } else {
        let body = lowerer.new_block(body_params.clone());
        lowerer
            .sites
            .add(lowerer.unit_index, HostSiteKind::FunctionEntry);
        (body, body)
    };
    lowerer.current = body;
    lowerer.push_scope();
    for (param, body_param) in function.params.iter().zip(&body_params) {
        let owner = body_param.owner;
        let (is_callback, is_owned) = {
            let info = lowerer.info(owner)?;
            (info.class.is_callback(), info.origin == OwnerOrigin::Owned)
        };
        let place = if is_callback {
            Place::Callback(owner)
        } else {
            Place::Owner(owner)
        };
        if is_owned {
            lowerer.register(owner)?;
        }
        lowerer.owner_depth.insert(owner, 0);
        lowerer.scope_mut()?.names.insert(param.name.clone(), place);
    }
    lowerer.with_site(HostSiteKind::FunctionReturn, |lowerer| {
        let value = lowerer.lower_expr(&function.body, Some(0))?;
        let result = lowerer.consume(value, Some(0))?;
        lowerer.exit_scope()?;
        lowerer.set_terminator(Terminator::Return { result })
    })?;
    lowerer.finish(UnitKind::Function, Some(CallableBody::new(body)), entry)
}

#[cfg(test)]
mod call_type_instantiation_tests {
    use super::*;

    fn tensor(dims: Vec<DimInfo>, precision: Prim) -> ConcreteHostType {
        ConcreteHostType::Tensor(TensorType { dims, precision })
    }

    #[test]
    fn callable_dimension_variable_accepts_one_consistent_call_site_instantiation() {
        let formal = tensor(vec![DimInfo::Named("n".into(), None)], Prim::F32);
        let literal_four = tensor(vec![DimInfo::Lit(4)], Prim::F32);
        let literal_five = tensor(vec![DimInfo::Lit(5)], Prim::F32);
        let named_batch = tensor(vec![DimInfo::Named("batch".into(), None)], Prim::F32);
        let pattern = FormalTypePattern::Tensor(vec![FormalDimension::Quantified("d0".into())]);
        let mut instantiation = CallTypeInstantiation::default();

        assert!(instantiation.admits(&pattern, &formal, &formal));
        assert!(instantiation.admits(&pattern, &formal, &literal_four));
        assert!(instantiation.admits(&pattern, &formal, &literal_four));
        assert!(!instantiation.admits(&pattern, &formal, &literal_five));
        assert!(CallTypeInstantiation::default().admits(&pattern, &formal, &named_batch));
    }

    #[test]
    fn nominal_dimensions_keep_names_while_accepting_literal_extents_and_wildcards() {
        let pattern = FormalTypePattern::Tensor(vec![FormalDimension::Nominal]);
        let batch = tensor(vec![DimInfo::Named("batch".into(), None)], Prim::F32);
        let same_batch = tensor(vec![DimInfo::Named("batch".into(), None)], Prim::F32);
        let seq = tensor(vec![DimInfo::Named("seq".into(), None)], Prim::F32);
        let literal_four = tensor(vec![DimInfo::Lit(4)], Prim::F32);
        let wildcard = tensor(vec![DimInfo::Named("*".into(), None)], Prim::F32);

        assert!(CallTypeInstantiation::default().admits(&pattern, &batch, &same_batch));
        assert!(CallTypeInstantiation::default().admits(&pattern, &batch, &literal_four));
        assert!(!CallTypeInstantiation::default().admits(&pattern, &batch, &seq));
        assert!(CallTypeInstantiation::default().admits(&pattern, &wildcard, &seq));
    }

    #[test]
    fn callable_dimension_instantiation_does_not_weaken_literals_rank_or_precision() {
        let formal = tensor(vec![DimInfo::Named("n".into(), None)], Prim::F32);
        let quantified = FormalTypePattern::Tensor(vec![FormalDimension::Quantified("d0".into())]);
        let wrong_rank = tensor(vec![DimInfo::Lit(4), DimInfo::Lit(1)], Prim::F32);
        let wrong_precision = tensor(vec![DimInfo::Lit(4)], Prim::F64);
        let literal_four = tensor(vec![DimInfo::Lit(4)], Prim::F32);
        let literal_five = tensor(vec![DimInfo::Lit(5)], Prim::F32);
        let nominal = FormalTypePattern::Tensor(vec![FormalDimension::Nominal]);

        assert!(!CallTypeInstantiation::default().admits(&quantified, &formal, &wrong_rank));
        assert!(!CallTypeInstantiation::default().admits(&quantified, &formal, &wrong_precision));
        assert!(!CallTypeInstantiation::default().admits(&nominal, &literal_four, &literal_five));
    }
}
