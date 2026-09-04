//! Lower checked host code to the ownership-explicit control-flow form.
//!
//! Lowering assigns every operand one of the three closed dispositions before
//! verification. Phase 2 places terminal operations at scope exit; Phase 3
//! may move those terminals to proven last uses without changing ownership.

use std::collections::{BTreeMap, BTreeSet};

use chelis_deep::DeepTag;
use chelis_deep::ast::Expr;
use chelis_types::CheckedProgram;
use chelis_types::manifest::RootManifest;
use chelis_types::types::Lane;

use crate::dag::RiscOp;
use crate::host::{
    ConcreteHostCallback, ConcreteHostCallbackKind, ConcreteHostExpr, ConcreteHostExprKind,
    ConcreteHostFunction, ConcreteHostMatchArm, ConcreteHostProgram, HostTensorHelper,
};
use crate::host_type_state::ConcreteHostType;

use super::classify::{ClassifyError, Placement, ValueClass, classify, render_type};
use super::error::OwnershipError;
use super::ir::{
    Block, BlockId, BlockParam, Edge, Op, Operand, OwnerId, OwnerInfo, OwnerOrigin,
    OwnershipProgram, ParamMode, Terminator, Unit, UnitKind,
};

const ROOTS_UNIT: &str = "roots";

#[derive(Clone)]
struct ParamSpec {
    mode: ParamMode,
    ty: ConcreteHostType,
    callback_modes: Option<Vec<ParamMode>>,
}

#[derive(Clone)]
struct Signature {
    params: Vec<ParamSpec>,
}

pub(super) fn lower(
    checked: &CheckedProgram,
    host: &ConcreteHostProgram,
    manifest: &RootManifest,
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
    };
    let mut units = vec![lower_roots(&ctx, manifest)?];
    for function in &host.functions {
        units.push(lower_function(&ctx, function)?);
    }
    Ok(OwnershipProgram { units })
}

fn build_signatures(
    checked: &CheckedProgram,
    host: &ConcreteHostProgram,
) -> Result<BTreeMap<String, Signature>, OwnershipError> {
    let mut signatures = BTreeMap::new();
    for function in &host.functions {
        let modes = declared_param_modes(checked, &function.name, function.params.len())
            .ok_or_else(|| OwnershipError::MissingSignature {
                function: function.name.clone(),
            })?;
        let params = function
            .params
            .iter()
            .zip(modes)
            .map(|(param, (mode, callback_modes))| ParamSpec {
                mode,
                ty: param.ty.clone(),
                callback_modes,
            })
            .collect();
        signatures.insert(function.name.clone(), Signature { params });
    }
    Ok(signatures)
}

/// Read owned/borrowed parameter modes from the checked function type. A
/// monomorphized host specialization uses the authored generic signature.
fn declared_param_modes(
    checked: &CheckedProgram,
    name: &str,
    arity: usize,
) -> Option<Vec<(ParamMode, Option<Vec<ParamMode>>)>> {
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
    params.into_iter().map(param_mode_of).collect()
}

fn strip_meta(expr: &Expr) -> &Expr {
    match expr {
        Expr::MetaExpr(meta, _) => strip_meta(&meta.expr),
        other => other,
    }
}

fn tag_and_children(expr: &Expr) -> Option<(DeepTag, &[Expr])> {
    match strip_meta(expr) {
        Expr::List(list, _) => {
            let tag = list.tag()?;
            Some((tag, list.elements.get(2..)?))
        }
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
    FunctionRef(String),
}

struct Scope {
    owners: Vec<OwnerId>,
    names: BTreeMap<String, Place>,
}

struct BlockBuilder {
    id: BlockId,
    params: Vec<BlockParam>,
    ops: Vec<Op>,
    terminator: Option<Terminator>,
}

#[derive(Clone, Copy)]
struct AdapterBlocks {
    entry: BlockId,
    body: BlockId,
}

struct UnitLowerer<'a> {
    ctx: &'a Context<'a>,
    unit_name: String,
    helpers: &'a [HostTensorHelper],
    blocks: Vec<BlockBuilder>,
    owners: BTreeMap<OwnerId, OwnerInfo>,
    owner_depth: BTreeMap<OwnerId, usize>,
    callback_modes: BTreeMap<OwnerId, Vec<ParamMode>>,
    next_owner: u32,
    current: BlockId,
    scopes: Vec<Scope>,
    moved: BTreeSet<OwnerId>,
    adapter: Option<AdapterBlocks>,
    captures: BTreeMap<String, OwnerId>,
}

impl<'a> UnitLowerer<'a> {
    fn new(ctx: &'a Context<'a>, unit_name: &str, helpers: &'a [HostTensorHelper]) -> Self {
        Self {
            ctx,
            unit_name: unit_name.to_string(),
            helpers,
            blocks: Vec::new(),
            owners: BTreeMap::new(),
            owner_depth: BTreeMap::new(),
            callback_modes: BTreeMap::new(),
            next_owner: 0,
            current: BlockId(0),
            scopes: Vec::new(),
            moved: BTreeSet::new(),
            adapter: None,
            captures: BTreeMap::new(),
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

    fn emit(&mut self, op: Op) {
        self.blocks[self.current.0 as usize].ops.push(op);
    }

    fn set_terminator(&mut self, terminator: Terminator) -> Result<(), OwnershipError> {
        let index = self.current.0 as usize;
        if self.blocks[index].terminator.is_some() {
            return Err(self.invariant(format!("block b{} terminated twice", self.current.0)));
        }
        self.blocks[index].terminator = Some(terminator);
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
                let movable =
                    info.origin == OwnerOrigin::Owned && tail.is_some_and(|scope| depth >= scope);
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
            Value::FunctionRef(name) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name,
            }),
        }
    }

    fn borrow(&mut self, value: Value) -> Result<Operand, OwnershipError> {
        match value {
            Value::Fresh(owner) => {
                self.register(owner)?;
                Ok(Operand::borrow(owner))
            }
            Value::Named(owner) | Value::Callback(owner) => Ok(Operand::borrow(owner)),
            Value::FunctionRef(name) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name,
            }),
        }
    }

    fn bind(&mut self, name: &str, value: Value) -> Result<(), OwnershipError> {
        match value {
            Value::Fresh(owner) => {
                self.register(owner)?;
                self.name_owner(owner, name)
            }
            Value::Named(owner) => self.name_owner(owner, name),
            Value::Callback(owner) => {
                self.scope_mut()?
                    .names
                    .insert(name.to_string(), Place::Callback(owner));
                Ok(())
            }
            Value::FunctionRef(function) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name: function,
            }),
        }
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
                self.emit(Op::Drop {
                    owner: Operand::move_(owner),
                });
            } else {
                self.emit(Op::Apply {
                    dest: None,
                    label: "discard".to_string(),
                    args: vec![Operand::move_(owner)],
                });
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
                self.moved.insert(owner);
                Value::Fresh(owner)
            }
            other => other,
        }
    }

    fn finish(self, kind: UnitKind, entry: BlockId) -> Result<Unit, OwnershipError> {
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
            name: self.unit_name,
            kind,
            entry,
            blocks,
            owners: self.owners,
        })
    }

    fn unlowered(&self, variant: &str) -> OwnershipError {
        OwnershipError::UnloweredExprKind {
            unit: self.unit_name.clone(),
            variant: variant.to_string(),
        }
    }

    fn lower_expr(
        &mut self,
        expr: &ConcreteHostExpr,
        tail: Option<usize>,
    ) -> Result<Value, OwnershipError> {
        match &expr.kind {
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
                function, args, ty, ..
            } => {
                let values = args
                    .iter()
                    .map(|arg| self.lower_expr(arg, None))
                    .collect::<Result<Vec<_>, _>>()?;
                self.lower_call(function, values, ty, tail)
            }
            ConcreteHostExprKind::Builtin { name, args, ty } => {
                let mut operands = Vec::with_capacity(args.len());
                for arg in args {
                    let value = self.lower_expr(arg, None)?;
                    operands.push(self.borrow(value)?);
                }
                self.apply(ty, format!("builtin:{name}"), operands)
            }
            ConcreteHostExprKind::AdtFieldAccess {
                base,
                field_index,
                ty,
            } => {
                let base = self.lower_expr(base, None)?;
                let base = self.borrow(base)?;
                self.apply(ty, format!("field:{field_index}"), vec![base])
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
            ConcreteHostExprKind::Let { bindings, body, .. } => {
                self.push_scope();
                let depth = self.depth();
                for binding in bindings {
                    let value = self.lower_expr(&binding.value, None)?;
                    self.bind(&binding.name, value)?;
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
            ConcreteHostExprKind::Filter { .. } => Err(self.unlowered("Filter")),
            ConcreteHostExprKind::Scan { .. } => Err(self.unlowered("Scan")),
            ConcreteHostExprKind::Partition { .. } => Err(self.unlowered("Partition")),
            ConcreteHostExprKind::FlatMap { .. } => Err(self.unlowered("FlatMap")),
            ConcreteHostExprKind::WithSeed { .. } => Err(self.unlowered("WithSeed")),
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
        args: Vec<Operand>,
    ) -> Result<Value, OwnershipError> {
        let dest = self.mint(ty, Placement::Value, OwnerOrigin::Owned, Vec::new())?;
        self.emit(Op::Apply {
            dest: Some(dest),
            label,
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
            let value = self.lower_expr(item, None)?;
            operands.push(self.consume(value, None)?);
        }
        self.apply(ty, label.to_string(), operands)
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
            return Ok(Value::FunctionRef(name.to_string()));
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

    fn lower_call(
        &mut self,
        function: &str,
        values: Vec<Value>,
        ty: &ConcreteHostType,
        tail: Option<usize>,
    ) -> Result<Value, OwnershipError> {
        let (label, specs) = match self.lookup(function) {
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
                        ty,
                        callback_modes: None,
                    })
                    .collect();
                (format!("call_callback:%{}", owner.0), specs)
            }
            Some(Place::Owner(_)) | None => match self.ctx.signatures.get(function) {
                Some(signature) => (format!("call:{function}"), signature.params.clone()),
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
        for (value, spec) in values.into_iter().zip(&specs) {
            if matches!(spec.ty, ConcreteHostType::Function(_, _)) {
                let operand = match value {
                    Value::Callback(owner) => Operand::borrow(owner),
                    Value::FunctionRef(name) => {
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
                args.push(operand);
                continue;
            }
            args.push(match spec.mode {
                ParamMode::Owned => self.consume(value, tail)?,
                ParamMode::Borrowed | ParamMode::EntryBorrow => self.borrow(value)?,
            });
        }
        self.apply(ty, label, args)
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
            target: join,
            args: vec![result],
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
                target: then_block,
                args: Vec::new(),
            },
            else_edge: Edge {
                target: else_block,
                args: Vec::new(),
            },
        })?;
        self.current = then_block;
        self.push_scope();
        self.lower_join_arm(then_expr, join)?;
        self.current = else_block;
        self.push_scope();
        self.lower_join_arm(else_expr, join)?;
        self.current = join;
        Ok(Value::Fresh(join_owner))
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
                    target: some,
                    args: Vec::new(),
                },
                Edge {
                    target: none,
                    args: Vec::new(),
                },
            ],
        })?;
        self.current = some;
        self.push_scope();
        let payload = self.mint(
            &payload_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![bind_name.to_string()],
        )?;
        self.emit(Op::Apply {
            dest: Some(payload),
            label: "option_payload".to_string(),
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
                target: *target,
                args: Vec::new(),
            })
            .collect::<Vec<_>>();
        if let Some(target) = default_block {
            edges.push(Edge {
                target,
                args: Vec::new(),
            });
        }
        if edges.is_empty() {
            return Err(self.invariant("ADT match has no arms"));
        }
        self.set_terminator(Terminator::Match {
            scrutinee,
            arms: edges,
        })?;
        for (arm, block) in arms.iter().zip(arm_blocks) {
            self.current = block;
            self.push_scope();
            for binding in &arm.bindings {
                let owner = self.mint(
                    &binding.ty,
                    Placement::Value,
                    OwnerOrigin::Owned,
                    vec![binding.name.clone()],
                )?;
                self.emit(Op::Apply {
                    dest: Some(owner),
                    label: format!("adt_payload:{}:{}", arm.ctor, binding.field_index),
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
            target: header,
            args: vec![init],
        }))?;
        self.push_scope();
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
                target: body,
                args: vec![Operand::move_(header_owner)],
            },
            exit_edge: Edge {
                target: exit,
                args: vec![Operand::move_(header_owner)],
            },
        })?;
        self.current = body;
        let element = self.mint(
            &element_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![names[1].clone()],
        )?;
        self.emit(Op::Apply {
            dest: Some(element),
            label: "loop_item".to_string(),
            args: vec![list],
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
            target: header,
            args: vec![next],
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
            target: header,
            args: vec![Operand::move_(seed)],
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
                target: body,
                args: vec![Operand::move_(header_owner)],
            },
            exit_edge: Edge {
                target: exit,
                args: vec![Operand::move_(header_owner)],
            },
        })?;
        self.current = body;
        let element = self.mint(
            &element_ty,
            Placement::Value,
            OwnerOrigin::Owned,
            vec![names[0].clone()],
        )?;
        self.emit(Op::Apply {
            dest: Some(element),
            label: "loop_item".to_string(),
            args: vec![list],
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
            label: "list_push".to_string(),
            args: vec![carried, item],
        });
        self.moved.insert(next);
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump(Edge {
            target: header,
            args: vec![Operand::move_(next)],
        }))?;
        self.current = exit;
        Ok(Value::Fresh(exit_owner))
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
        if is_identity_helper(helper_ref) && args.len() == 1 {
            return self.lower_expr(&args[0], tail);
        }
        let mut operands = Vec::with_capacity(args.len());
        for arg in args {
            let value = self.lower_expr(arg, None)?;
            operands.push(self.borrow(value)?);
        }
        self.apply(ty, format!("tensor:{}", helper_ref.name), operands)
    }
}

fn is_identity_helper(helper: &HostTensorHelper) -> bool {
    if helper.dag.roots().len() != 1 || helper.inputs.len() != 1 {
        return false;
    }
    let Some(node) = helper.dag.get(helper.dag.roots()[0]) else {
        return false;
    };
    match &node.op {
        RiscOp::Load { name } => {
            node.output_type == helper.output && helper.inputs[0].name == *name
        }
        _ => false,
    }
}

fn expr_type(expr: &ConcreteHostExpr) -> ConcreteHostType {
    match &expr.kind {
        ConcreteHostExprKind::Int(_) => ConcreteHostType::Int64,
        ConcreteHostExprKind::Float(_) => ConcreteHostType::Float64,
        ConcreteHostExprKind::Bool(_) => ConcreteHostType::Bool,
        ConcreteHostExprKind::String(_) => ConcreteHostType::String,
        ConcreteHostExprKind::Unit => ConcreteHostType::Unit,
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
        | ConcreteHostExprKind::Map { ty, .. }
        | ConcreteHostExprKind::Filter { ty, .. }
        | ConcreteHostExprKind::Fold { ty, .. }
        | ConcreteHostExprKind::Scan { ty, .. }
        | ConcreteHostExprKind::Partition { ty, .. }
        | ConcreteHostExprKind::FlatMap { ty, .. }
        | ConcreteHostExprKind::WithSeed { ty, .. }
        | ConcreteHostExprKind::TensorCall { ty, .. } => ty.clone(),
    }
}

fn lower_roots(ctx: &Context<'_>, manifest: &RootManifest) -> Result<Unit, OwnershipError> {
    let mut lowerer = UnitLowerer::new(ctx, ROOTS_UNIT, &ctx.host.global_tensor_helpers);
    let entry = lowerer.new_block(Vec::new());
    lowerer.current = entry;
    lowerer.push_scope();
    for binding in &ctx.host.globals {
        let value = lowerer.lower_expr(&binding.value, None)?;
        lowerer.bind(&binding.name, value)?;
    }
    let mut sinks = Vec::new();
    for root in &manifest.entries {
        if root.lane != Lane::Host || ctx.function_names.contains(&root.def_name) {
            continue;
        }
        match lowerer.lookup(&root.def_name) {
            Some(Place::Owner(owner)) => sinks.push((root, owner)),
            Some(Place::Callback(_)) | None => {
                return Err(OwnershipError::ManifestRootWithoutBinding {
                    root: root.name.clone(),
                    def_name: root.def_name.clone(),
                });
            }
        }
    }
    for (index, (root, owner)) in sinks.iter().enumerate() {
        let owner = *owner;
        let later = sinks[index + 1..].iter().any(|(_, other)| *other == owner);
        let owner = if later { lowerer.copy(owner)? } else { owner };
        lowerer.moved.insert(owner);
        lowerer.emit(Op::RootConsume {
            root: root.name.clone(),
            owner: Operand::move_(owner),
        });
    }
    lowerer.exit_scope()?;
    lowerer.set_terminator(Terminator::Exit)?;
    lowerer.finish(UnitKind::Roots, entry)
}

fn lower_function(
    ctx: &Context<'_>,
    function: &ConcreteHostFunction,
) -> Result<Unit, OwnershipError> {
    let signature =
        ctx.signatures
            .get(&function.name)
            .ok_or_else(|| OwnershipError::MissingSignature {
                function: function.name.clone(),
            })?;
    let mut lowerer = UnitLowerer::new(ctx, &function.name, &function.tensor_helpers);
    let mut entry_params = Vec::with_capacity(function.params.len());
    let mut body_params = Vec::with_capacity(function.params.len());
    for (param, spec) in function.params.iter().zip(&signature.params) {
        let class =
            lowerer.classify_or_reject(&spec.ty, Placement::Parameter, Some(&param.name))?;
        let heap = class.is_heap();
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
        let body_origin = if heap && spec.mode == ParamMode::Borrowed {
            OwnerOrigin::ExternalBorrow
        } else {
            OwnerOrigin::Owned
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
            mode: if heap { spec.mode } else { ParamMode::Owned },
        });
    }
    let entry = lowerer.new_block(entry_params.clone());
    let body = lowerer.new_block(body_params.clone());
    lowerer.adapter = Some(AdapterBlocks { entry, body });
    lowerer.current = entry;
    let mut args = Vec::with_capacity(entry_params.len());
    for (entry_param, spec) in entry_params.iter().zip(&signature.params) {
        args.push(match (entry_param.mode, spec.mode) {
            (ParamMode::EntryBorrow, ParamMode::Owned) => {
                let copy = lowerer.copy(entry_param.owner)?;
                Operand::move_(copy)
            }
            (ParamMode::EntryBorrow, ParamMode::Borrowed | ParamMode::EntryBorrow) => {
                Operand::borrow(entry_param.owner)
            }
            (ParamMode::Owned | ParamMode::Borrowed, _) => Operand::move_(entry_param.owner),
        });
    }
    lowerer.set_terminator(Terminator::Jump(Edge { target: body, args }))?;
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
    let value = lowerer.lower_expr(&function.body, Some(0))?;
    let result = lowerer.consume(value, Some(0))?;
    lowerer.exit_scope()?;
    lowerer.set_terminator(Terminator::Return { result })?;
    lowerer.finish(UnitKind::Function, entry)
}
