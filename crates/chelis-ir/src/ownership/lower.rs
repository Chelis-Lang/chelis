//! Ownership lowering: from the checked, concretely typed host program to
//! the explicit ownership form.
//!
//! The lowering decides every use disposition structurally:
//!
//! - a fresh (unnamed) value is moved into its consumer;
//! - a named owner is copied before a consuming use, except a bare name in
//!   tail position of a scope the owner belongs to, which moves it;
//! - a borrowed handle (borrowed formal, entry borrow, captured top-level
//!   owner) is only ever borrowed or copied;
//! - every owner a scope introduces and does not move is dropped when the
//!   scope exits (the Phase 2 terminal placement; last-use placement is
//!   Phase 3 work).
//!
//! Joins (`if`, `match`) and loops (`fold`, `map`) receive owned block
//! parameters; the root unit ends with one consuming sink per manifest
//! entry, in manifest order, and a drop for every other top-level owner.

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
    AggregateShape, Block, BlockId, BlockParam, Callee, Capture, HelperRef, HelperScope, Literal,
    MatchArm, MatchPattern, Op, Operand, OwnerId, OwnerInfo, ParamMode, ParamSpec, Program,
    ROOTS_UNIT, Signature, Terminator, Unit, UnitKind,
};

pub(crate) fn lower(
    checked: &CheckedProgram,
    host: &ConcreteHostProgram,
    manifest: &RootManifest,
) -> Result<Program, OwnershipError> {
    let signatures = build_signatures(checked, host)?;
    let ctx = Ctx {
        host,
        signatures: &signatures,
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
    Ok(Program { signatures, units })
}

// ── Signatures ──────────────────────────────────────────────────────────

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
        signatures.insert(
            function.name.clone(),
            Signature {
                params,
                ret: function.ret_ty.clone(),
            },
        );
    }
    Ok(signatures)
}

/// The declared parameter modes of a host function, read from the checked
/// program's type environment: a `t-ref` parameter is borrowed, every other
/// parameter is owned ([04-LIN-4]). A monomorphized specialization reads
/// its generic definition's signature.
fn declared_param_modes(
    checked: &CheckedProgram,
    name: &str,
    arity: usize,
) -> Option<Vec<(ParamMode, Option<Vec<ParamMode>>)>> {
    let generic_name = name.split("__mono_").next().unwrap_or(name);
    let ty = checked
        .type_env()
        .get(name)
        .or_else(|| checked.type_env().get(generic_name))?;
    let (params, _ret) = fn_type_parts(ty)?;
    if params.len() != arity {
        return None;
    }
    Some(params.into_iter().map(param_mode_of).collect())
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
            let children = list.elements.get(2..).unwrap_or(&[]);
            Some((tag, children))
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

fn param_mode_of(param: &Expr) -> (ParamMode, Option<Vec<ParamMode>>) {
    match tag_and_children(param) {
        Some((DeepTag::TRef, _)) => (ParamMode::Borrowed, None),
        Some((DeepTag::TFn, children)) => {
            let inner = children
                .split_last()
                .map(|(_, params)| {
                    params
                        .iter()
                        .map(|p| param_mode_of(p).0)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            (ParamMode::Owned, Some(inner))
        }
        _ => (ParamMode::Owned, None),
    }
}

// ── Per-unit lowering state ─────────────────────────────────────────────

struct Ctx<'a> {
    host: &'a ConcreteHostProgram,
    signatures: &'a BTreeMap<String, Signature>,
    global_names: BTreeSet<String>,
    function_names: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy)]
enum Place {
    Owner(OwnerId),
    Callback(OwnerId),
}

/// The result of lowering one expression.
#[derive(Debug, Clone)]
enum Value {
    /// An owner nobody holds yet: the consumer moves it, or registers it in
    /// the current scope when it only borrows it.
    Fresh(OwnerId),
    /// A reference to an owner some scope already holds.
    Named(OwnerId),
    /// A contextual-callback parameter in scope.
    Callback(OwnerId),
    /// A host function named as a value; representable only as a callback
    /// argument.
    FunctionRef(String),
}

struct Scope {
    owners: Vec<OwnerId>,
    names: BTreeMap<String, Place>,
}

struct BlockBuilder {
    id: BlockId,
    label: String,
    params: Vec<BlockParam>,
    ops: Vec<Op>,
    terminator: Option<Terminator>,
}

struct UnitLowerer<'a> {
    ctx: &'a Ctx<'a>,
    unit_name: String,
    helper_scope: HelperScope,
    helpers: &'a [HostTensorHelper],
    blocks: Vec<BlockBuilder>,
    owners: BTreeMap<OwnerId, OwnerInfo>,
    owner_depth: BTreeMap<OwnerId, usize>,
    callback_modes: BTreeMap<OwnerId, Vec<ParamMode>>,
    next_owner: u32,
    current: BlockId,
    scopes: Vec<Scope>,
    moved: BTreeSet<OwnerId>,
    captures: Vec<Capture>,
}

impl<'a> UnitLowerer<'a> {
    fn new(
        ctx: &'a Ctx<'a>,
        unit_name: &str,
        helper_scope: HelperScope,
        helpers: &'a [HostTensorHelper],
    ) -> Self {
        Self {
            ctx,
            unit_name: unit_name.to_string(),
            helper_scope,
            helpers,
            blocks: Vec::new(),
            owners: BTreeMap::new(),
            owner_depth: BTreeMap::new(),
            callback_modes: BTreeMap::new(),
            next_owner: 0,
            current: BlockId(0),
            scopes: Vec::new(),
            moved: BTreeSet::new(),
            captures: Vec::new(),
        }
    }

    fn invariant(&self, detail: impl Into<String>) -> OwnershipError {
        OwnershipError::LoweringInvariant {
            unit: self.unit_name.clone(),
            detail: detail.into(),
        }
    }

    fn new_block(&mut self, label: &str, params: Vec<BlockParam>) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BlockBuilder {
            id,
            label: label.to_string(),
            params,
            ops: Vec::new(),
            terminator: None,
        });
        id
    }

    fn emit(&mut self, op: Op) {
        let index = self.current.0 as usize;
        self.blocks[index].ops.push(op);
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
        self.scopes.len().saturating_sub(1)
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
                name: name.unwrap_or("<value>").to_string(),
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
        names: Vec<String>,
        handle: bool,
    ) -> Result<OwnerId, OwnershipError> {
        let class = self.classify_or_reject(ty, placement, names.first().map(String::as_str))?;
        let id = OwnerId(self.next_owner);
        self.next_owner += 1;
        self.owners.insert(
            id,
            OwnerInfo {
                ty: ty.clone(),
                class,
                names,
                handle,
            },
        );
        self.owner_depth.insert(id, self.depth());
        Ok(id)
    }

    fn info(&self, owner: OwnerId) -> Result<&OwnerInfo, OwnershipError> {
        self.owners
            .get(&owner)
            .ok_or_else(|| self.invariant(format!("owner %{} has no record", owner.0)))
    }

    fn is_heap(&self, owner: OwnerId) -> Result<bool, OwnershipError> {
        Ok(self.info(owner)?.class.is_heap())
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
        let ty = self.info(source)?.ty.clone();
        let dest = self.mint(&ty, Placement::Value, Vec::new(), false)?;
        self.emit(Op::Copy {
            dest,
            source: Operand::clone_(source),
        });
        Ok(dest)
    }

    /// Consume a value into an owner-taking position. `tail` names the
    /// shallowest scope depth whose owners may be moved because every scope
    /// at or below it ends right after this expression.
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
                        name: info.names.first().cloned().unwrap_or_default(),
                    });
                }
                if !info.class.is_heap() {
                    return Ok(Operand::move_(owner));
                }
                let depth = self.owner_depth.get(&owner).copied().unwrap_or(0);
                let movable = !info.handle && tail.is_some_and(|k| depth >= k);
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
                name: self
                    .info(owner)?
                    .names
                    .first()
                    .cloned()
                    .unwrap_or_default(),
            }),
            Value::FunctionRef(name) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name,
            }),
        }
    }

    /// Borrow a value for the extent of one operation. A fresh value is
    /// registered in the current scope so its terminal drop is placed at
    /// the scope's exit.
    fn borrow(&mut self, value: Value) -> Result<Operand, OwnershipError> {
        match value {
            Value::Fresh(owner) => {
                self.register(owner)?;
                Ok(Operand::borrow(owner))
            }
            Value::Named(owner) => Ok(Operand::borrow(owner)),
            Value::Callback(owner) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name: self
                    .info(owner)?
                    .names
                    .first()
                    .cloned()
                    .unwrap_or_default(),
            }),
            Value::FunctionRef(name) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name,
            }),
        }
    }

    /// Bind a source name to a value. A fresh value becomes a named owner of
    /// the current scope; a named value becomes a display alias of its
    /// owner ([04-LIN-3]: rebinding never mints an owner).
    fn bind(&mut self, name: &str, value: Value) -> Result<(), OwnershipError> {
        match value {
            Value::Fresh(owner) => {
                self.register(owner)?;
                self.name_owner(owner, name)
            }
            Value::Named(owner) => self.name_owner(owner, name),
            Value::Callback(owner) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name: self
                    .info(owner)?
                    .names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| name.to_string()),
            }),
            Value::FunctionRef(function) => Err(OwnershipError::FirstClassFunctionValue {
                unit: self.unit_name.clone(),
                name: function,
            }),
        }
    }

    /// Close the innermost scope: every heap owner it introduced and did not
    /// move receives its terminal drop here, in reverse introduction order.
    fn exit_scope(&mut self) -> Result<(), OwnershipError> {
        let scope = self.scopes.pop().ok_or_else(|| self.invariant("no scope to exit"))?;
        for owner in scope.owners.into_iter().rev() {
            if self.moved.contains(&owner) {
                continue;
            }
            let info = self.info(owner)?;
            if !info.class.is_heap() || info.handle {
                continue;
            }
            self.emit(Op::Drop {
                owner: Operand::move_(owner),
            });
            self.moved.insert(owner);
        }
        Ok(())
    }

    /// A value leaving a scope: an owner the exiting scope holds becomes
    /// fresh for the enclosing consumer instead of being dropped.
    fn resolve_out(&mut self, value: Value, scope_depth: usize) -> Value {
        match value {
            Value::Named(owner)
                if self
                    .owner_depth
                    .get(&owner)
                    .is_some_and(|depth| *depth >= scope_depth)
                    && !self.owners.get(&owner).is_some_and(|info| info.handle) =>
            {
                self.moved.insert(owner);
                Value::Fresh(owner)
            }
            other => other,
        }
    }

    fn finish(self, kind: UnitKind, entry: BlockId, body: BlockId) -> Result<Unit, OwnershipError> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for builder in self.blocks {
            let terminator = builder.terminator.ok_or_else(|| OwnershipError::LoweringInvariant {
                unit: self.unit_name.clone(),
                detail: format!("block b{} has no terminator", builder.id.0),
            })?;
            blocks.push(Block {
                id: builder.id,
                label: builder.label,
                params: builder.params,
                ops: builder.ops,
                terminator,
            });
        }
        Ok(Unit {
            name: self.unit_name,
            kind,
            entry,
            body,
            captures: self.captures,
            blocks,
            owners: self.owners,
        })
    }

    // ── Expressions ────────────────────────────────────────────────────

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
                self.literal(Literal::Int(*value), &ConcreteHostType::Int64)
            }
            ConcreteHostExprKind::Float(value) => {
                self.literal(Literal::Float(*value), &ConcreteHostType::Float64)
            }
            ConcreteHostExprKind::Bool(value) => {
                self.literal(Literal::Bool(*value), &ConcreteHostType::Bool)
            }
            ConcreteHostExprKind::String(value) => {
                self.literal(Literal::String(value.clone()), &ConcreteHostType::String)
            }
            ConcreteHostExprKind::Unit => self.literal(Literal::Unit, &ConcreteHostType::Unit),
            ConcreteHostExprKind::List(items, ty) => {
                self.aggregate(AggregateShape::List, items, ty)
            }
            ConcreteHostExprKind::Tuple(items, ty) => {
                self.aggregate(AggregateShape::Tuple, items, ty)
            }
            ConcreteHostExprKind::AdtConstruct { ctor, fields, ty } => {
                self.aggregate(AggregateShape::Adt { ctor: ctor.clone() }, fields, ty)
            }
            ConcreteHostExprKind::Var(name, ty) => self.lower_var(name, ty),
            ConcreteHostExprKind::Call {
                function, args, ty, ..
            } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.lower_expr(arg, None)?);
                }
                self.lower_call(function, values, ty, None)
            }
            ConcreteHostExprKind::Builtin { name, args, ty } => {
                let mut operands = Vec::with_capacity(args.len());
                for arg in args {
                    let value = self.lower_expr(arg, None)?;
                    operands.push(self.borrow(value)?);
                }
                let dest = self.mint(ty, Placement::Value, Vec::new(), false)?;
                self.emit(Op::Builtin {
                    dest,
                    name: name.clone(),
                    args: operands,
                });
                Ok(Value::Fresh(dest))
            }
            ConcreteHostExprKind::AdtFieldAccess {
                base,
                field_index,
                ty,
            } => {
                let base = self.lower_expr(base, None)?;
                let base = self.borrow(base)?;
                let dest = self.mint(ty, Placement::Value, Vec::new(), false)?;
                self.emit(Op::FieldAccess {
                    dest,
                    base,
                    field_index: *field_index,
                });
                Ok(Value::Fresh(dest))
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
                let body_tail = Some(tail.map_or(depth, |k| k.min(depth)));
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

    fn literal(&mut self, value: Literal, ty: &ConcreteHostType) -> Result<Value, OwnershipError> {
        let dest = self.mint(ty, Placement::Value, Vec::new(), false)?;
        self.emit(Op::Literal { dest, value });
        Ok(Value::Fresh(dest))
    }

    fn aggregate(
        &mut self,
        shape: AggregateShape,
        items: &[ConcreteHostExpr],
        ty: &ConcreteHostType,
    ) -> Result<Value, OwnershipError> {
        // Classify the aggregate before its items so a function container
        // is reported as the container it is (chelis#879).
        self.classify_or_reject(ty, Placement::Value, None)?;
        let mut operands = Vec::with_capacity(items.len());
        for item in items {
            let value = self.lower_expr(item, None)?;
            operands.push(self.consume(value, None)?);
        }
        let dest = self.mint(ty, Placement::Value, Vec::new(), false)?;
        self.emit(Op::Aggregate {
            dest,
            shape,
            items: operands,
        });
        Ok(Value::Fresh(dest))
    }

    fn lower_var(&mut self, name: &str, ty: &ConcreteHostType) -> Result<Value, OwnershipError> {
        if let Some(place) = self.lookup(name) {
            return Ok(match place {
                Place::Owner(owner) => Value::Named(owner),
                Place::Callback(owner) => Value::Callback(owner),
            });
        }
        if name == "Nil" {
            return self.literal(Literal::EmptyList, ty);
        }
        if name == "None" {
            return self.literal(Literal::None, ty);
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

    /// A top-level binding read inside a function body: a borrowed handle
    /// on the root unit's owner, minted once per function.
    fn capture(&mut self, name: &str) -> Result<Value, OwnershipError> {
        if let Some(capture) = self.captures.iter().find(|capture| capture.name == name) {
            return Ok(Value::Named(capture.owner));
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
        let owner = self.mint(&ty, Placement::Value, vec![name.to_string()], true)?;
        self.owner_depth.insert(owner, 0);
        self.captures.push(Capture {
            name: name.to_string(),
            owner,
        });
        if let Some(scope) = self.scopes.first_mut() {
            scope.names.insert(name.to_string(), Place::Owner(owner));
        }
        Ok(Value::Named(owner))
    }

    fn lower_call(
        &mut self,
        function: &str,
        values: Vec<Value>,
        ty: &ConcreteHostType,
        tail: Option<usize>,
    ) -> Result<Value, OwnershipError> {
        let (callee, specs) = match self.lookup(function) {
            Some(Place::Callback(owner)) => {
                let params = match &self.info(owner)?.ty {
                    ConcreteHostType::Function(params, _) => params.clone(),
                    other => {
                        return Err(self.invariant(format!(
                            "callback `{function}` has non-function type `{}`",
                            render_type(other)
                        )));
                    }
                };
                let modes = self.callback_modes.get(&owner).cloned();
                let specs = params
                    .into_iter()
                    .enumerate()
                    .map(|(index, ty)| ParamSpec {
                        mode: modes
                            .as_ref()
                            .and_then(|modes| modes.get(index).copied())
                            .unwrap_or(ParamMode::Owned),
                        ty,
                        callback_modes: None,
                    })
                    .collect::<Vec<_>>();
                (Callee::Callback(owner), specs)
            }
            Some(Place::Owner(_)) | None => match self.ctx.signatures.get(function) {
                Some(signature) => (
                    Callee::Function(function.to_string()),
                    signature.params.clone(),
                ),
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
        let mut operands = Vec::with_capacity(values.len());
        for (value, spec) in values.into_iter().zip(&specs) {
            let operand = if matches!(spec.ty, ConcreteHostType::Function(_, _)) {
                match value {
                    Value::Callback(owner) => Operand::move_(owner),
                    Value::FunctionRef(name) => {
                        let dest = self.mint(&spec.ty, Placement::Parameter, Vec::new(), false)?;
                        self.emit(Op::FunctionRef {
                            dest,
                            function: name,
                        });
                        Operand::move_(dest)
                    }
                    other => self.consume(other, None)?,
                }
            } else {
                match spec.mode {
                    ParamMode::Owned => self.consume(value, tail)?,
                    ParamMode::Borrowed | ParamMode::EntryBorrow => self.borrow(value)?,
                }
            };
            operands.push(operand);
        }
        let dest = self.mint(ty, Placement::Value, Vec::new(), false)?;
        self.emit(Op::Call {
            dest,
            callee,
            args: operands,
        });
        Ok(Value::Fresh(dest))
    }

    /// Lower one arm of a join: its own scope, its result moved into the
    /// join parameter, its owners dropped before the edge.
    fn lower_arm(
        &mut self,
        block: BlockId,
        expr: &ConcreteHostExpr,
        join: BlockId,
    ) -> Result<(), OwnershipError> {
        self.current = block;
        let depth = self.depth();
        let value = self.lower_expr(expr, Some(depth))?;
        let result = self.consume(value, Some(depth))?;
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump {
            target: join,
            args: vec![result],
        })
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
        let join_owner = self.mint(ty, Placement::Value, Vec::new(), false)?;
        let then_block = self.new_block("then", Vec::new());
        let else_block = self.new_block("else", Vec::new());
        let join = self.new_block(
            "join",
            vec![BlockParam {
                owner: join_owner,
                mode: ParamMode::Owned,
            }],
        );
        self.set_terminator(Terminator::Branch {
            cond,
            then_block,
            else_block,
        })?;
        self.push_scope();
        self.lower_arm(then_block, then_expr, join)?;
        self.push_scope();
        self.lower_arm(else_block, else_expr, join)?;
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
        let join_owner = self.mint(ty, Placement::Value, Vec::new(), false)?;
        let join = self.new_block(
            "join",
            vec![BlockParam {
                owner: join_owner,
                mode: ParamMode::Owned,
            }],
        );
        // The Some arm's binder is a fresh owner of the arm scope.
        self.push_scope();
        let payload = self.mint(
            &payload_ty,
            Placement::Value,
            vec![bind_name.to_string()],
            false,
        )?;
        self.register(payload)?;
        self.scope_mut()?
            .names
            .insert(bind_name.to_string(), Place::Owner(payload));
        let some_block = self.new_block(
            "some",
            vec![BlockParam {
                owner: payload,
                mode: ParamMode::Owned,
            }],
        );
        let none_block = self.new_block("none", Vec::new());
        self.set_terminator(Terminator::Match {
            scrutinee,
            arms: vec![
                MatchArm {
                    pattern: MatchPattern::Some,
                    target: some_block,
                },
                MatchArm {
                    pattern: MatchPattern::None,
                    target: none_block,
                },
            ],
        })?;
        self.lower_arm(some_block, some_expr, join)?;
        self.push_scope();
        self.lower_arm(none_block, none_expr, join)?;
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
        let join_owner = self.mint(ty, Placement::Value, Vec::new(), false)?;
        let join = self.new_block(
            "join",
            vec![BlockParam {
                owner: join_owner,
                mode: ParamMode::Owned,
            }],
        );
        let mut edges = Vec::with_capacity(arms.len() + 1);
        let mut pending: Vec<(BlockId, &ConcreteHostExpr)> = Vec::new();
        // Each arm's binders are fresh owners of that arm's scope. The
        // scopes are opened here and closed by `lower_arm`, in order.
        let mut arm_scopes: Vec<Scope> = Vec::new();
        for arm in arms {
            let mut scope = Scope {
                owners: Vec::new(),
                names: BTreeMap::new(),
            };
            let mut params = Vec::with_capacity(arm.bindings.len());
            for binding in &arm.bindings {
                let owner = self.mint(
                    &binding.ty,
                    Placement::Value,
                    vec![binding.name.clone()],
                    false,
                )?;
                // Binder depth is the arm scope, one below the current one.
                self.owner_depth.insert(owner, self.depth() + 1);
                scope.owners.push(owner);
                scope.names.insert(binding.name.clone(), Place::Owner(owner));
                params.push(BlockParam {
                    owner,
                    mode: ParamMode::Owned,
                });
            }
            let block = self.new_block(&format!("arm:{}", arm.ctor), params);
            edges.push(MatchArm {
                pattern: MatchPattern::Ctor(arm.ctor.clone()),
                target: block,
            });
            pending.push((block, &arm.expr));
            arm_scopes.push(scope);
        }
        if let Some(default) = default_expr {
            let block = self.new_block("default", Vec::new());
            edges.push(MatchArm {
                pattern: MatchPattern::Default,
                target: block,
            });
            pending.push((block, default));
            arm_scopes.push(Scope {
                owners: Vec::new(),
                names: BTreeMap::new(),
            });
        }
        self.set_terminator(Terminator::Match {
            scrutinee,
            arms: edges,
        })?;
        for ((block, expr), scope) in pending.into_iter().zip(arm_scopes) {
            self.scopes.push(scope);
            self.lower_arm(block, expr, join)?;
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

    /// Lower a loop body's callback against already-minted parameter
    /// owners, in the loop body scope, at the body's tail.
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

    fn list_element_type(&self, list: &ConcreteHostExpr) -> Result<ConcreteHostType, OwnershipError> {
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

        let header_param = self.mint(ty, Placement::Value, Vec::new(), false)?;
        let header = self.new_block(
            "loop_header",
            vec![BlockParam {
                owner: header_param,
                mode: ParamMode::Owned,
            }],
        );
        self.set_terminator(Terminator::Jump {
            target: header,
            args: vec![init],
        })?;
        let exit_param = self.mint(ty, Placement::Value, Vec::new(), false)?;
        let exit = self.new_block(
            "loop_exit",
            vec![BlockParam {
                owner: exit_param,
                mode: ParamMode::Owned,
            }],
        );

        self.push_scope();
        let acc = self.mint(ty, Placement::Value, vec![names[0].clone()], false)?;
        let element = self.mint(&element_ty, Placement::Value, vec![names[1].clone()], false)?;
        for (owner, name) in [(acc, &names[0]), (element, &names[1])] {
            self.register(owner)?;
            self.scope_mut()?
                .names
                .insert(name.clone(), Place::Owner(owner));
        }
        let body = self.new_block(
            "loop_body",
            vec![
                BlockParam {
                    owner: acc,
                    mode: ParamMode::Owned,
                },
                BlockParam {
                    owner: element,
                    mode: ParamMode::Owned,
                },
            ],
        );
        self.current = header;
        self.set_terminator(Terminator::Loop {
            list,
            carried: Operand::move_(header_param),
            body,
            exit,
        })?;
        self.current = body;
        let depth = self.depth();
        let next = self.lower_callback_body(callback, &[acc, element])?;
        let next = self.consume(next, Some(depth))?;
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump {
            target: header,
            args: vec![next],
        })?;
        self.current = exit;
        Ok(Value::Fresh(exit_param))
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
        let seed = self.mint(ty, Placement::Value, Vec::new(), false)?;
        self.emit(Op::Literal {
            dest: seed,
            value: Literal::EmptyList,
        });
        self.moved.insert(seed);

        let header_param = self.mint(ty, Placement::Value, Vec::new(), false)?;
        let header = self.new_block(
            "loop_header",
            vec![BlockParam {
                owner: header_param,
                mode: ParamMode::Owned,
            }],
        );
        self.set_terminator(Terminator::Jump {
            target: header,
            args: vec![Operand::move_(seed)],
        })?;
        let exit_param = self.mint(ty, Placement::Value, Vec::new(), false)?;
        let exit = self.new_block(
            "loop_exit",
            vec![BlockParam {
                owner: exit_param,
                mode: ParamMode::Owned,
            }],
        );

        self.push_scope();
        let acc = self.mint(ty, Placement::Value, Vec::new(), false)?;
        let element = self.mint(&element_ty, Placement::Value, vec![names[0].clone()], false)?;
        self.register(acc)?;
        self.register(element)?;
        self.scope_mut()?
            .names
            .insert(names[0].clone(), Place::Owner(element));
        let body = self.new_block(
            "loop_body",
            vec![
                BlockParam {
                    owner: acc,
                    mode: ParamMode::Owned,
                },
                BlockParam {
                    owner: element,
                    mode: ParamMode::Owned,
                },
            ],
        );
        self.current = header;
        self.set_terminator(Terminator::Loop {
            list,
            carried: Operand::move_(header_param),
            body,
            exit,
        })?;
        self.current = body;
        let depth = self.depth();
        let item = self.lower_callback_body(callback, &[element])?;
        let item = self.consume(item, Some(depth))?;
        let carried = self.consume(Value::Named(acc), Some(depth))?;
        let next = self.mint(ty, Placement::Value, Vec::new(), false)?;
        self.emit(Op::ListPush {
            dest: next,
            list: carried,
            item,
        });
        self.moved.insert(next);
        self.exit_scope()?;
        self.set_terminator(Terminator::Jump {
            target: header,
            args: vec![Operand::move_(next)],
        })?;
        self.current = exit;
        Ok(Value::Fresh(exit_param))
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
        // An identity helper is the erased `&x` borrow: it hands back its
        // input rather than allocating, so the call denotes the argument's
        // own owner and no fresh owner is minted for it.
        if is_identity_helper(helper_ref) && args.len() == 1 {
            return self.lower_expr(&args[0], tail);
        }
        let name = helper_ref.name.clone();
        let mut operands = Vec::with_capacity(args.len());
        for arg in args {
            let value = self.lower_expr(arg, None)?;
            operands.push(self.borrow(value)?);
        }
        let dest = self.mint(ty, Placement::Value, Vec::new(), false)?;
        self.emit(Op::TensorCall {
            dest,
            helper: HelperRef {
                scope: self.helper_scope.clone(),
                index: helper,
                name,
            },
            args: operands,
        });
        Ok(Value::Fresh(dest))
    }
}

/// Mirrors the backend's identity-helper recognition: a helper whose only
/// root is a load of its single input, typed exactly as its output.
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

/// The concrete type a host expression produces, mirroring the term-level
/// `host_expr_type` used by host lowering.
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

// ── Units ───────────────────────────────────────────────────────────────

fn lower_roots(ctx: &Ctx<'_>, manifest: &RootManifest) -> Result<Unit, OwnershipError> {
    let mut l = UnitLowerer::new(
        ctx,
        ROOTS_UNIT,
        HelperScope::Global,
        &ctx.host.global_tensor_helpers,
    );
    let entry = l.new_block("", Vec::new());
    l.current = entry;
    l.push_scope();
    for binding in &ctx.host.globals {
        let value = l.lower_expr(&binding.value, None)?;
        l.bind(&binding.name, value)?;
    }

    // One consuming sink per host-lane manifest entry, in manifest order
    // ([04-LIN-6]). Tensor-lane entries are observed by the DAG lane and a
    // function's manifest entry observes no value.
    let mut sinks = Vec::new();
    for entry in &manifest.entries {
        if entry.lane != Lane::Host || ctx.function_names.contains(&entry.def_name) {
            continue;
        }
        match l.lookup(&entry.def_name) {
            Some(Place::Owner(owner)) => sinks.push((entry, owner)),
            Some(Place::Callback(_)) | None => {
                return Err(OwnershipError::ManifestRootWithoutBinding {
                    root: entry.name.clone(),
                    def_name: entry.def_name.clone(),
                });
            }
        }
    }
    for (index, (entry, owner)) in sinks.iter().enumerate() {
        let owner = *owner;
        let later_root = sinks[index + 1..].iter().any(|(_, other)| *other == owner);
        let operand = if l.is_heap(owner)? && later_root {
            let copy = l.copy(owner)?;
            l.moved.insert(copy);
            Operand::move_(copy)
        } else {
            if l.is_heap(owner)? {
                l.moved.insert(owner);
            }
            Operand::move_(owner)
        };
        l.emit(Op::RootConsume {
            root: entry.name.clone(),
            path: entry.path.clone(),
            owner: operand,
        });
    }
    l.exit_scope()?;
    l.set_terminator(Terminator::Exit)?;
    l.finish(UnitKind::Roots, entry, entry)
}

fn lower_function(ctx: &Ctx<'_>, function: &ConcreteHostFunction) -> Result<Unit, OwnershipError> {
    let signature = ctx
        .signatures
        .get(&function.name)
        .ok_or_else(|| OwnershipError::MissingSignature {
            function: function.name.clone(),
        })?;
    let mut l = UnitLowerer::new(
        ctx,
        &function.name,
        HelperScope::Function(function.name.clone()),
        &function.tensor_helpers,
    );

    // The artifact-boundary adapter ([04-LIN-7]): heap arguments arrive as
    // entry borrows; an owned formal receives a copy, a borrowed formal
    // borrows the entry value directly, a nonheap value passes by value.
    let mut entry_params = Vec::with_capacity(function.params.len());
    for spec in &signature.params {
        let class = l.classify_or_reject(&spec.ty, Placement::Parameter, None)?;
        let handle = class.is_heap();
        let owner = l.mint(&spec.ty, Placement::Parameter, Vec::new(), handle)?;
        entry_params.push(BlockParam {
            owner,
            mode: if handle {
                ParamMode::EntryBorrow
            } else {
                ParamMode::Owned
            },
        });
    }
    let entry = l.new_block("entry", entry_params.clone());

    let mut body_params = Vec::with_capacity(function.params.len());
    for (param, spec) in function.params.iter().zip(&signature.params) {
        let class = l.classify_or_reject(&spec.ty, Placement::Parameter, Some(&param.name))?;
        let handle = class.is_heap() && spec.mode == ParamMode::Borrowed;
        let owner = l.mint(&spec.ty, Placement::Parameter, vec![param.name.clone()], handle)?;
        if let Some(modes) = &spec.callback_modes {
            l.callback_modes.insert(owner, modes.clone());
        }
        body_params.push(BlockParam {
            owner,
            mode: if class.is_heap() {
                spec.mode
            } else {
                ParamMode::Owned
            },
        });
    }
    let body = l.new_block("body", body_params.clone());

    l.current = entry;
    let mut args = Vec::with_capacity(function.params.len());
    for (entry_param, spec) in entry_params.iter().zip(&signature.params) {
        let operand = match (entry_param.mode, spec.mode) {
            (ParamMode::EntryBorrow, ParamMode::Owned) => {
                let copy = l.copy(entry_param.owner)?;
                Operand::move_(copy)
            }
            (ParamMode::EntryBorrow, ParamMode::Borrowed | ParamMode::EntryBorrow) => {
                Operand::borrow(entry_param.owner)
            }
            (ParamMode::Owned | ParamMode::Borrowed, _) => Operand::move_(entry_param.owner),
        };
        args.push(operand);
    }
    l.set_terminator(Terminator::Jump { target: body, args })?;

    l.current = body;
    l.push_scope();
    for (param, body_param) in function.params.iter().zip(&body_params) {
        let owner = body_param.owner;
        let info = l.info(owner)?;
        let place = if info.class.is_callback() {
            Place::Callback(owner)
        } else {
            Place::Owner(owner)
        };
        if info.class.is_heap() && !info.handle {
            l.register(owner)?;
        }
        l.owner_depth.insert(owner, 0);
        l.scope_mut()?.names.insert(param.name.clone(), place);
    }
    let value = l.lower_expr(&function.body, Some(0))?;
    let result = l.consume(value, Some(0))?;
    l.exit_scope()?;
    l.set_terminator(Terminator::Return { result })?;
    l.finish(UnitKind::Function, entry, body)
}
