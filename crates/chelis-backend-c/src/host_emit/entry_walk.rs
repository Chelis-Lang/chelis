//! chelis#2506, [04-NUM-11]: the tensors a parameter carries inside another
//! value are validated at the host entry before the body reads them.
//!
//! A tensor at a fixed tuple position is an ordinary signature observation:
//! it joins the entry plan under its parameter path (`p.1`), so its null,
//! dtype, rank, literal-extent and named-extent checks are the ones a tensor
//! parameter gets. A tensor whose presence or count depends on the value, one
//! inside an ADT, list, option or dictionary, is reached by a walker: one C
//! function per carried type, run over the value once for null, dtype and
//! rank and once for literal extents, with the same check renderers.
//!
//! A List element's named extent joins an invocation-local witness, including
//! the first nonempty element. ADT dimension arguments have no host
//! representation, so a walk through an ADT cannot assert that name's
//! identity.
//!
//! A walk costs the size of the value. An ordinary aggregate walk runs at
//! the exported entry. A claimed List walk runs in the body or retained call
//! boundary, since each invocation needs its own extent witness.

use super::*;
use chelis_ir::host::{
    EntryPattern, HostAdtLayout, HostFunctionOrigin, HostTensorInput, SignatureEntryPlan,
};

/// One tensor the entry plan observes: a tensor parameter, or a tensor at a
/// fixed tuple position inside a parameter.
pub(super) struct EntryObservation {
    pub param: usize,
    /// The parameter path, `p.1.0`; a tensor parameter's own name.
    pub path: String,
    /// The tuple components from the parameter to the tensor.
    pub steps: Vec<usize>,
    pub ty: TensorType,
}

/// Every tensor at a fixed tuple position of every parameter, in signature
/// order. Direct tensor obligations use the authored contract: a specialized
/// call's ABI can have wildcard axes even when its callee binds them.
pub(super) fn entry_observations(function: &HostFunction) -> Vec<EntryObservation> {
    fn collect(
        ty: &HostAbiType,
        param: usize,
        path: String,
        steps: Vec<usize>,
        out: &mut Vec<EntryObservation>,
    ) {
        match ty {
            HostAbiType::Tensor(ty) => out.push(EntryObservation {
                param,
                path,
                steps,
                ty: ty.clone(),
            }),
            HostAbiType::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    let mut steps = steps.clone();
                    steps.push(index);
                    collect(item, param, format!("{path}.{index}"), steps, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for (index, param) in function.params.iter().enumerate() {
        let declared = function
            .entry_contract
            .formals()
            .get(index)
            .filter(|formal| formal.name() == param.name)
            .and_then(|formal| match formal.pattern() {
                EntryPattern::Tensor(ty) => Some(ty),
                _ => None,
            });
        collect(
            declared.unwrap_or(&param.ty),
            index,
            param.name.clone(),
            Vec::new(),
            &mut out,
        );
    }
    out
}

/// The complete signature owns entry order; helper partitioning does not.
pub(super) fn function_entry_plan(function: &HostFunction) -> SignatureEntryPlan {
    SignatureEntryPlan::new(entry_observations(function).into_iter().map(|observation| {
        HostTensorInput {
            name: observation.path,
            ty: observation.ty,
        }
    }))
}

/// The plan's observations as host expressions, for the domination proof: a
/// tensor parameter is its variable, and a nested tensor has no variable, so
/// no helper comparison is discharged on its account.
pub(super) fn entry_plan_args(function: &HostFunction) -> Vec<HostExpr> {
    entry_observations(function)
        .into_iter()
        .map(|observation| {
            if observation.steps.is_empty() {
                let param = &function.params[observation.param];
                HostExpr::new(HostExprKind::Var(param.name.clone(), param.ty.clone()))
            } else {
                HostExpr::new(HostExprKind::Unit)
            }
        })
        .collect()
}

/// The C work one parameter adds to its function's entry, in signature order.
#[derive(Default)]
pub(super) struct ParamEntryWork {
    /// Container null checks and the reads that fetch nested values.
    pub fetch: Vec<String>,
    /// Walker calls for the null, dtype and rank pass.
    pub metadata: Vec<String>,
    /// Walker calls for the literal-extent pass.
    pub extents: Vec<String>,
    /// Fixed observations and walks in the parameter's structural order.
    pub ordered_extents: Vec<ExtentStep>,
}

pub(super) enum ExtentStep {
    Fixed(usize),
    Walk(String),
}

/// A function entry's nested-value work.
pub(super) struct FunctionEntryWork {
    /// The C expression of each plan observation.
    pub args: Vec<String>,
    /// The parameter each plan observation belongs to.
    pub owners: Vec<usize>,
    pub params: Vec<ParamEntryWork>,
    /// Names whose first witness may be a List element at this invocation.
    pub named_list_binders: Vec<String>,
    /// Releases of the values `fetch` read, after every check.
    pub release: Vec<String>,
}

impl FunctionEntryWork {
    /// Whether any parameter's value is walked.
    pub fn walks(&self) -> bool {
        self.params.iter().any(|param| !param.metadata.is_empty())
    }
}

/// A function's entry work: its body's, and its exported entry's when it has
/// one.
pub(super) struct EntryWork {
    pub body: FunctionEntryWork,
    pub exported: Option<FunctionEntryWork>,
    /// A List claim runs at every owned-body call and at exported entry.
    pub extent_at_body: bool,
}

#[derive(Clone, Copy)]
enum EntryWalkMode {
    FixedOnly,
    ClaimedLists,
    All,
}

/// A List preserves its element tensor's named axes through its type. ADTs
/// do not retain that binder identity, so this walk crosses only List layers.
pub(super) fn list_named_dims(ty: &HostAbiType, out: &mut Vec<String>) {
    match ty {
        HostAbiType::List(item) => list_named_dims(item, out),
        HostAbiType::Tensor(tensor) => {
            for dim in &tensor.dims {
                if let DimInfo::Named(name, _) = dim
                    && name != "*"
                    && !out.contains(name)
                {
                    out.push(name.clone());
                }
            }
        }
        _ => {}
    }
}

fn tensor_has_extent_claim(tensor: &TensorType) -> bool {
    tensor.dims.iter().any(|dim| {
        matches!(dim, DimInfo::Lit(_)) || matches!(dim, DimInfo::Named(name, _) if name != "*")
    })
}

pub(super) fn entry_type_has_extent_claim(ty: &HostAbiType) -> bool {
    match ty {
        HostAbiType::Tensor(tensor) => tensor_has_extent_claim(tensor),
        HostAbiType::List(inner) => entry_type_has_extent_claim(inner),
        _ => false,
    }
}

pub(super) fn entry_pattern_has_extent_claim(pattern: &EntryPattern<HostAbiType>) -> bool {
    match pattern {
        EntryPattern::Tensor(ty) => entry_type_has_extent_claim(ty),
        EntryPattern::List(inner) => entry_pattern_has_extent_claim(inner),
        EntryPattern::Other => false,
    }
}

/// Rebuild the List walker shape from the signature that owns its checks.
/// Call-site ABI axes may be wildcards and cannot supply this walker's guards.
fn claimed_list_type(pattern: &EntryPattern<HostAbiType>) -> Option<HostAbiType> {
    match pattern {
        EntryPattern::Tensor(HostAbiType::Tensor(tensor)) => {
            Some(HostAbiType::Tensor(tensor.clone()))
        }
        EntryPattern::List(inner) => Some(HostAbiType::List(Box::new(claimed_list_type(inner)?))),
        _ => None,
    }
}

pub(super) fn entry_pattern_matches_type(
    pattern: &EntryPattern<HostAbiType>,
    ty: &HostAbiType,
) -> bool {
    match (pattern, ty) {
        (EntryPattern::Tensor(HostAbiType::Tensor(declared)), HostAbiType::Tensor(actual)) => {
            declared.dims.len() == actual.dims.len()
                && declared.precision == actual.precision
                && declared
                    .dims
                    .iter()
                    .zip(&actual.dims)
                    .all(|(claim, observed)| match claim {
                        DimInfo::Lit(required) => {
                            matches!(observed, DimInfo::Lit(actual) if actual == required)
                        }
                        DimInfo::Named(name, _) if name != "*" => {
                            matches!(observed, DimInfo::Named(actual, _) if actual == name)
                        }
                        _ => true,
                    })
        }
        (EntryPattern::List(inner), HostAbiType::List(actual)) => {
            entry_pattern_matches_type(inner, actual)
        }
        (EntryPattern::Other, _) => true,
        _ => false,
    }
}

fn mono_entry_pattern_matches_abi(pattern: &EntryPattern<HostAbiType>, ty: &HostAbiType) -> bool {
    match (pattern, ty) {
        (EntryPattern::Tensor(HostAbiType::Tensor(declared)), HostAbiType::Tensor(actual)) => {
            declared.dims.len() == actual.dims.len()
                && declared.precision == actual.precision
                && declared
                    .dims
                    .iter()
                    .zip(&actual.dims)
                    .all(|(claim, observed)| {
                        !matches!((claim, observed), (DimInfo::Lit(a), DimInfo::Lit(b)) if a != b)
                    })
        }
        (EntryPattern::List(inner), HostAbiType::List(actual)) => {
            mono_entry_pattern_matches_abi(inner, actual)
        }
        (EntryPattern::Other, _) => true,
        _ => false,
    }
}

fn validate_entry_contract(function: &HostFunction) -> Result<(), Unsupported> {
    let contract = &function.entry_contract;
    if contract.formals().is_empty() {
        // Manually assembled helper fixtures may not carry authored entry
        // metadata, but a List extent claim can never take that route.
        if !function.params.iter().any(|param| {
            matches!(param.ty, HostAbiType::List(_)) && entry_type_has_extent_claim(&param.ty)
        }) {
            return Ok(());
        }
        return Err(invalid_abi_shape(
            "List entry has no projected signature contract".into(),
            "signature entry",
        ));
    }
    if contract.formals().len() != function.params.len()
        || !contract
            .formals()
            .iter()
            .zip(&function.params)
            .all(|(formal, param)| {
                formal.name() == param.name
                    && if function.origin == HostFunctionOrigin::Monomorphized {
                        mono_entry_pattern_matches_abi(formal.pattern(), &param.ty)
                    } else {
                        entry_pattern_matches_type(formal.pattern(), &param.ty)
                    }
            })
    {
        return Err(invalid_abi_shape(
            "signature entry contract lost formal order or List shape".into(),
            "signature entry",
        ));
    }
    let mut projected_names = Vec::new();
    for formal in contract.formals() {
        if let EntryPattern::List(_) = formal.pattern()
            && let Some(ty) = claimed_list_type(formal.pattern())
        {
            list_named_dims(&ty, &mut projected_names);
        }
    }
    if projected_names != contract.named_list_binders() {
        return Err(invalid_abi_shape(
            "signature entry contract lost a List binder".into(),
            "signature entry",
        ));
    }
    Ok(())
}

/// A tensor-carrying ADT field: its index, path segment and walker.
type WalkedField = (usize, String, usize);

/// One carried type a walker validates.
enum Shape {
    Tensor(TensorType),
    Tuple(Vec<(usize, usize)>),
    /// Per constructor with a tensor-carrying field: its tag and those fields.
    Adt(Vec<(String, Vec<WalkedField>)>),
    List(usize),
    Option(usize),
    /// A dictionary's values.
    Dict(usize),
}

/// The walkers a program's parameters need, one per carried type.
pub(super) struct EntryWalkers<'a> {
    layouts: &'a [HostAdtLayout<HostAbiType>],
    types: Vec<HostAbiType>,
    shapes: Vec<Option<Shape>>,
    carrying: Vec<HostAbiType>,
}

fn c_pointer_type(ty: &HostAbiType) -> &'static str {
    match ty {
        HostAbiType::Tensor(_) => "const chelis_tensor *",
        HostAbiType::Tuple(_) => "const chelis_tuple *",
        HostAbiType::Adt(_, _) => "const chelis_adt *",
        HostAbiType::List(_) => "const chelis_list *",
        HostAbiType::Option(_) => "const chelis_option *",
        HostAbiType::Dict(_, _) => "const chelis_dict *",
        _ => unreachable!("only a tensor-carrying type has a walker"),
    }
}

fn borrow_value(ty: &HostAbiType, value: &str) -> String {
    let kind = match ty {
        HostAbiType::Tensor(_) => "tensor",
        HostAbiType::Tuple(_) => "tuple",
        HostAbiType::Adt(_, _) => "adt",
        HostAbiType::List(_) => "list",
        HostAbiType::Option(_) => "option",
        HostAbiType::Dict(_, _) => "dict",
        _ => unreachable!("only a tensor-carrying type is borrowed"),
    };
    format!("chelis_{kind}_borrow_value({value})")
}

/// Whether `function` has an exported C entry.
fn has_exported_entry(function: &HostFunction) -> bool {
    function.origin == HostFunctionOrigin::Authored
}

/// Whether `ty` holds a tensor at a fixed tuple position.
fn observes(ty: &HostAbiType) -> bool {
    match ty {
        HostAbiType::Tensor(_) => true,
        HostAbiType::Tuple(items) => items.iter().any(observes),
        _ => false,
    }
}

impl<'a> EntryWalkers<'a> {
    pub fn new(program: &'a HostProgram) -> Result<Self, Unsupported> {
        let mut walkers = Self {
            layouts: &program.adt_layouts,
            types: Vec::new(),
            shapes: Vec::new(),
            carrying: Vec::new(),
        };
        let mut roots = Vec::new();
        for function in &program.functions {
            if has_exported_entry(function) {
                roots.extend(function.params.iter().map(|param| param.ty.clone()));
            }
            for formal in function.entry_contract.formals() {
                if matches!(formal.pattern(), EntryPattern::List(_))
                    && entry_pattern_has_extent_claim(formal.pattern())
                {
                    roots.push(claimed_list_type(formal.pattern()).ok_or_else(|| {
                        invalid_abi_shape(
                            "claimed List has no representable C entry shape".into(),
                            "signature entry",
                        )
                    })?);
                }
            }
        }
        walkers.carrying = walkers.carrying_types(roots.into_iter())?;
        Ok(walkers)
    }

    fn layout(&self, ty: &HostAbiType) -> Result<&'a HostAdtLayout<HostAbiType>, Unsupported> {
        self.layouts
            .iter()
            .find(|layout| layout.ty == *ty)
            .ok_or_else(|| {
                invalid_abi_shape(
                    format!("parameter type {ty:?} has no constructor layout to validate"),
                    "C host entry validation",
                )
            })
    }

    fn components(&self, ty: &HostAbiType) -> Result<Vec<HostAbiType>, Unsupported> {
        Ok(match ty {
            HostAbiType::Tuple(items) => items.clone(),
            HostAbiType::List(inner) | HostAbiType::Option(inner) => vec![(**inner).clone()],
            HostAbiType::Dict(_, value) => vec![(**value).clone()],
            HostAbiType::Adt(_, _) => self
                .layout(ty)?
                .constructors
                .iter()
                .flat_map(|constructor| constructor.fields.iter().map(|field| field.ty.clone()))
                .collect(),
            _ => Vec::new(),
        })
    }

    /// Every type reachable from `roots` that carries a tensor value, as the
    /// least fixed point over the component graph: a cycle alone carries
    /// nothing.
    fn carrying_types(
        &self,
        roots: impl Iterator<Item = HostAbiType>,
    ) -> Result<Vec<HostAbiType>, Unsupported> {
        let mut reached: Vec<(HostAbiType, Vec<HostAbiType>)> = Vec::new();
        let mut pending = roots.collect::<Vec<_>>();
        while let Some(ty) = pending.pop() {
            if reached.iter().any(|(seen, _)| *seen == ty) {
                continue;
            }
            let components = self.components(&ty)?;
            pending.extend(components.iter().cloned());
            reached.push((ty, components));
        }
        let mut carrying = reached
            .iter()
            .filter(|(ty, _)| matches!(ty, HostAbiType::Tensor(_)))
            .map(|(ty, _)| ty.clone())
            .collect::<Vec<_>>();
        loop {
            let before = carrying.len();
            for (ty, components) in &reached {
                if !carrying.contains(ty) && components.iter().any(|item| carrying.contains(item)) {
                    carrying.push(ty.clone());
                }
            }
            if carrying.len() == before {
                return Ok(carrying);
            }
        }
    }

    fn carries(&self, ty: &HostAbiType) -> bool {
        self.carrying.contains(ty)
    }

    /// Whether an entry does work on a value of `ty`: an exported entry
    /// (`walk`) checks every tensor it carries, a body only the tensors at
    /// fixed tuple positions.
    fn has_work(&self, ty: &HostAbiType, walk: bool) -> bool {
        if walk { self.carries(ty) } else { observes(ty) }
    }

    /// The walker for `ty`, registering it and every walker it calls.
    fn walker(&mut self, ty: &HostAbiType) -> Result<usize, Unsupported> {
        if let Some(id) = self.types.iter().position(|seen| seen == ty) {
            return Ok(id);
        }
        let id = self.types.len();
        self.types.push(ty.clone());
        self.shapes.push(None);
        let shape = match ty {
            HostAbiType::Tensor(tensor) => Shape::Tensor(tensor.clone()),
            HostAbiType::Tuple(items) => {
                let mut parts = Vec::new();
                for (index, item) in items.iter().enumerate() {
                    if self.carries(item) {
                        parts.push((index, self.walker(item)?));
                    }
                }
                Shape::Tuple(parts)
            }
            HostAbiType::Adt(_, _) => {
                let layout = self.layout(ty)?;
                let several = layout.constructors.len() > 1;
                let mut constructors = Vec::new();
                for constructor in &layout.constructors {
                    let mut fields = Vec::new();
                    for (index, field) in constructor.fields.iter().enumerate() {
                        if !self.carries(&field.ty) {
                            continue;
                        }
                        let field_name = field.name.clone().unwrap_or_else(|| index.to_string());
                        let segment = if several {
                            format!(
                                ".{}.{field_name}",
                                stored_constructor_name(ty, &constructor.name)
                            )
                        } else {
                            format!(".{field_name}")
                        };
                        fields.push((index, segment, self.walker(&field.ty)?));
                    }
                    if !fields.is_empty() {
                        constructors.push((constructor.name.clone(), fields));
                    }
                }
                Shape::Adt(constructors)
            }
            HostAbiType::List(inner) => Shape::List(self.walker(inner)?),
            HostAbiType::Option(inner) => Shape::Option(self.walker(inner)?),
            HostAbiType::Dict(_, value) => Shape::Dict(self.walker(value)?),
            _ => unreachable!("only a tensor-carrying type has a walker"),
        };
        self.shapes[id] = Some(shape);
        Ok(id)
    }

    /// The entry work of one function's body and of its exported entry.
    pub fn entry_work(&mut self, function: &HostFunction) -> Result<EntryWork, Unsupported> {
        validate_entry_contract(function)?;
        let named_list_binders = function.entry_contract.named_list_binders().to_vec();
        let claimed_lists = function
            .entry_contract
            .formals()
            .iter()
            .map(|formal| {
                matches!(formal.pattern(), EntryPattern::List(_))
                    && entry_pattern_has_extent_claim(formal.pattern())
            })
            .collect::<Vec<_>>();
        let extent_at_body = claimed_lists.iter().any(|claimed| *claimed);
        // The exported entry checks every aggregate in signature order.
        // Internal calls retain the established List-only admission boundary.
        let exported = if has_exported_entry(function) {
            Some(self.function_work(function, EntryWalkMode::All, named_list_binders.clone())?)
        } else {
            None
        };
        Ok(EntryWork {
            body: self.function_work(
                function,
                if extent_at_body {
                    EntryWalkMode::ClaimedLists
                } else {
                    EntryWalkMode::FixedOnly
                },
                named_list_binders,
            )?,
            exported,
            extent_at_body,
        })
    }

    /// The entry work of one function: fetch every tensor at a fixed tuple
    /// position and, for its exported entry (`walk`), call a walker on every
    /// other tensor-carrying value.
    fn function_work(
        &mut self,
        function: &HostFunction,
        mode: EntryWalkMode,
        named_list_binders: Vec<String>,
    ) -> Result<FunctionEntryWork, Unsupported> {
        let mut work = FunctionEntryWork {
            args: Vec::new(),
            owners: Vec::new(),
            params: (0..function.params.len())
                .map(|_| ParamEntryWork::default())
                .collect(),
            named_list_binders,
            release: Vec::new(),
        };
        let mut serial = 0usize;
        for (index, param) in function.params.iter().enumerate() {
            let claimed_list = function
                .entry_contract
                .formals()
                .get(index)
                .filter(|formal| {
                    matches!(formal.pattern(), EntryPattern::List(_))
                        && entry_pattern_has_extent_claim(formal.pattern())
                });
            let walk = match mode {
                EntryWalkMode::FixedOnly => false,
                EntryWalkMode::ClaimedLists => claimed_list.is_some(),
                EntryWalkMode::All => true,
            };
            let contract_ty = if walk {
                claimed_list
                    .map(|formal| {
                        claimed_list_type(formal.pattern()).ok_or_else(|| {
                            invalid_abi_shape(
                                "claimed List has no representable C entry shape".into(),
                                "signature entry",
                            )
                        })
                    })
                    .transpose()?
            } else {
                None
            };
            let ty = contract_ty.as_ref().unwrap_or(&param.ty);
            if contract_ty.is_some() && !self.carries(ty) {
                return Err(invalid_abi_shape(
                    "C walker inventory omitted a claimed List entry".into(),
                    "signature entry",
                ));
            }
            self.param_work(
                ty,
                c_ident(&param.name).into_owned(),
                param.name.clone(),
                index,
                walk,
                &mut serial,
                &mut work,
            )?;
        }
        Ok(work)
    }

    #[allow(clippy::too_many_arguments)]
    fn param_work(
        &mut self,
        ty: &HostAbiType,
        value: String,
        path: String,
        param: usize,
        walk: bool,
        serial: &mut usize,
        work: &mut FunctionEntryWork,
    ) -> Result<(), Unsupported> {
        match ty {
            HostAbiType::Tensor(_) => {
                work.params[param]
                    .ordered_extents
                    .push(ExtentStep::Fixed(work.args.len()));
                work.args.push(value);
                work.owners.push(param);
            }
            HostAbiType::Tuple(items) if self.has_work(ty, walk) => {
                let label = chelis_ir::span_sanitize::sanitize_for_format_string(&path);
                work.params[param].fetch.push(format!(
                    "if ({value} == NULL) {{ fprintf(stderr, \"input `{label}` is NULL\\n\"); abort(); }}"
                ));
                for (index, item) in items.iter().enumerate() {
                    let item_path = format!("{path}.{index}");
                    if !self.has_work(item, walk) {
                        continue;
                    }
                    let id = *serial;
                    *serial += 1;
                    let held = format!("__chelis_entry_value_{id}");
                    let typed = format!("__chelis_entry_item_{id}");
                    work.params[param].fetch.push(format!(
                        "chelis_value {held} = chelis_tuple_get({value}, {index});"
                    ));
                    work.params[param].fetch.push(format!(
                        "{}{typed} = {};",
                        c_pointer_type(item),
                        borrow_value(item, &held)
                    ));
                    work.release.push(format!("chelis_value_release({held});"));
                    self.param_work(item, typed, item_path, param, walk, serial, work)?;
                }
            }
            _ if walk && self.carries(ty) => {
                let walker = self.walker(ty)?;
                let id = *serial;
                *serial += 1;
                let root = format!("__chelis_entry_path_{id}");
                work.params[param].fetch.push(format!(
                    "const __chelis_entry_path {root} = {{ NULL, {}, 0 }};",
                    c_utf8_byte_literal(&path)
                ));
                let states =
                    if matches!(ty, HostAbiType::List(_)) && !work.named_list_binders.is_empty() {
                        "__chelis_entry_named_states"
                    } else {
                        "NULL"
                    };
                let state_count = work.named_list_binders.len();
                let call = |pass| {
                    format!(
                        "__chelis_entry_walk_{walker}({value}, &{root}, {pass}, {states}, {state_count});"
                    )
                };
                work.params[param].metadata.push(call(0));
                work.params[param].extents.push(call(1));
                work.params[param]
                    .ordered_extents
                    .push(ExtentStep::Walk(call(1)));
            }
            _ => {}
        }
        Ok(())
    }

    /// The path type, its printer, and every registered walker.
    pub fn render(&self, out: &mut Vec<String>) {
        out.push(
            r#"
typedef struct __chelis_entry_named_state {
    const char *key;
    const char *claim;
    int seen;
    int64_t value;
    int axis;
    char path[512];
} __chelis_entry_named_state;

static inline void __chelis_entry_named_observe(__chelis_entry_named_state *states, size_t count,
        const char *key, const char *path, int axis, int64_t value) {
    if (states == NULL) return;
    for (size_t i = 0; i < count; ++i) {
        __chelis_entry_named_state *state = &states[i];
        if (strcmp(state->key, key) != 0) continue;
        if (!state->seen) {
            state->seen = 1;
            state->value = value;
            state->axis = axis;
            snprintf(state->path, sizeof state->path, "%s", path);
        } else if (state->value != value) {
            fprintf(stderr, "extent `%s`: %s axis %d = %lld, %s axis %d = %lld\n",
                state->claim, state->path, state->axis, (long long)state->value,
                path, axis, (long long)value);
            chelis_numeric_trap("numeric trap: domain in load at i64");
        }
        return;
    }
}
"#
            .to_string(),
        );
        // A path segment's list index is an `i64` extent position.
        let index_type = CEmitter::prim_elem_type(Prim::Int64);
        out.push(format!(
            "typedef struct __chelis_entry_path {{\n    const struct __chelis_entry_path *parent;\n    const char *segment;\n    {index_type} index;\n}} __chelis_entry_path;"
        ));
        out.push(
            r#"

/* `used` counts the whole path's length, including what did not fit. */
static inline void __chelis_entry_path_append(const __chelis_entry_path *path, char *buffer, size_t size, size_t *used) {
    if (path->parent != NULL) __chelis_entry_path_append(path->parent, buffer, size, used);
    size_t room = *used < size ? size - *used : 0;
    char *at = room > 0 ? buffer + *used : NULL;
    int written = path->segment != NULL
        ? snprintf(at, room, "%s", path->segment)
        : snprintf(at, room, "[%lld]", (long long)path->index);
    if (written > 0) *used += (size_t)written;
}

/* Renders only on a failing check, into storage the trap that follows ends. */
static inline const char *__chelis_entry_path_text(const __chelis_entry_path *path) {
    static const char marker[] = "...(truncated)";
    static char buffer[512];
    size_t used = 0;
    buffer[0] = '\0';
    __chelis_entry_path_append(path, buffer, sizeof buffer, &used);
    if (used >= sizeof buffer) {
        /* A path that did not fit ends in the marker, placed at a UTF-8 lead byte. */
        size_t end = sizeof buffer - sizeof marker;
        while (end > 0 && (buffer[end] & 0xC0) == 0x80) --end;
        memcpy(buffer + end, marker, sizeof marker);
    }
    return buffer;
}
"#
            .to_string(),
        );
        if self.types.is_empty() {
            return;
        }
        for (id, ty) in self.types.iter().enumerate() {
            out.push(format!(
                "static void __chelis_entry_walk_{id}({}value, const __chelis_entry_path *path, int extents, __chelis_entry_named_state *states, size_t count);",
                c_pointer_type(ty)
            ));
        }
        for (id, ty) in self.types.iter().enumerate() {
            out.push(String::new());
            out.push(format!(
                "static void __chelis_entry_walk_{id}({}value, const __chelis_entry_path *path, int extents, __chelis_entry_named_state *states, size_t count) {{",
                c_pointer_type(ty)
            ));
            let label = TensorLabel::path("path");
            if !matches!(self.shapes[id], Some(Shape::Tensor(_))) {
                out.push("    if (!extents && value == NULL) {".to_string());
                out.push(format!(
                    "        fprintf(stderr, \"{} is NULL\\n\"{});",
                    label.format, label.args
                ));
                out.push("        abort();".to_string());
                out.push("    }".to_string());
            }
            let child = |segment: String,
                         fetch: String,
                         item: &HostAbiType,
                         walker: usize,
                         state_arg: &str| {
                [
                    "    {".to_string(),
                    format!("        chelis_value item = {fetch};"),
                    format!("        const __chelis_entry_path segment = {segment};"),
                    format!(
                        "        __chelis_entry_walk_{walker}({}, &segment, extents, {state_arg}, count);",
                        borrow_value(item, "item")
                    ),
                    "        chelis_value_release(item);".to_string(),
                    "    }".to_string(),
                ]
            };
            match self.shapes[id]
                .as_ref()
                .expect("walker shape is registered")
            {
                Shape::Tensor(tensor) => {
                    out.push("    if (!extents) {".to_string());
                    out.extend(
                        tensor_metadata_checks("value", &label, tensor)
                            .into_iter()
                            .map(|line| format!("        {line}")),
                    );
                    out.push("        return;".to_string());
                    out.push("    }".to_string());
                    for (axis, dim) in tensor.dims.iter().enumerate() {
                        match dim {
                            DimInfo::Lit(required) => out.extend(
                                literal_extent_check("value", &label, axis, *required)
                                    .into_iter()
                                    .map(|line| format!("    {line}")),
                            ),
                            DimInfo::Named(name, _) if name != "*" => {
                                out.push(format!(
                                    "    if (states != NULL) __chelis_entry_named_observe(states, count, {}, __chelis_entry_path_text(path), {axis}, chelis_tensor_shape(value, {axis}));",
                                    c_string_literal(name)
                                ));
                            }
                            _ => {}
                        }
                    }
                }
                Shape::Tuple(parts) => {
                    let HostAbiType::Tuple(items) = ty else {
                        unreachable!("tuple shape")
                    };
                    for (index, walker) in parts {
                        out.extend(child(
                            format!("{{ path, \".{index}\", 0 }}"),
                            format!("chelis_tuple_get(value, {index})"),
                            &items[*index],
                            *walker,
                            "NULL",
                        ));
                    }
                }
                Shape::Adt(constructors) => {
                    let layout = self.layout(ty).expect("walker layout is registered");
                    out.push("    chelis_string tag = chelis_adt_get_tag(value);".to_string());
                    for (position, (name, fields)) in constructors.iter().enumerate() {
                        let keyword = if position == 0 { "if" } else { "} else if" };
                        out.push(format!(
                            "    {keyword} (chelis_host_string_eq_cstr(tag, {})) {{",
                            c_utf8_byte_literal(&stored_constructor_name(ty, name))
                        ));
                        let constructor = layout
                            .constructors
                            .iter()
                            .find(|constructor| constructor.name == *name)
                            .expect("walked constructor is in its layout");
                        for (index, segment, walker) in fields {
                            out.extend(
                                child(
                                    format!("{{ path, {}, 0 }}", c_utf8_byte_literal(segment)),
                                    format!("chelis_adt_get_field(value, {index})"),
                                    &constructor.fields[*index].ty,
                                    *walker,
                                    "NULL",
                                )
                                .into_iter()
                                .map(|line| format!("    {line}")),
                            );
                        }
                    }
                    if !constructors.is_empty() {
                        out.push("    }".to_string());
                    }
                    out.push("    chelis_string_release(tag);".to_string());
                }
                Shape::List(walker) => {
                    let HostAbiType::List(item) = ty else {
                        unreachable!("list shape")
                    };
                    out.push(format!(
                        "    for ({index_type} index = 0; index < chelis_list_len(value); ++index) {{"
                    ));
                    out.extend(
                        child(
                            "{ path, NULL, index }".to_string(),
                            "chelis_list_index(value, index)".to_string(),
                            item,
                            *walker,
                            "states",
                        )
                        .into_iter()
                        .map(|line| format!("    {line}")),
                    );
                    out.push("    }".to_string());
                }
                Shape::Option(walker) => {
                    let HostAbiType::Option(item) = ty else {
                        unreachable!("option shape")
                    };
                    out.push("    if (chelis_option_is_some(value)) {".to_string());
                    out.extend(
                        child(
                            format!("{{ path, {}, 0 }}", c_utf8_byte_literal(".Some.value")),
                            "chelis_option_unwrap(value)".to_string(),
                            item,
                            *walker,
                            "NULL",
                        )
                        .into_iter()
                        .map(|line| format!("    {line}")),
                    );
                    out.push("    }".to_string());
                }
                Shape::Dict(walker) => {
                    let HostAbiType::Dict(_, item) = ty else {
                        unreachable!("dictionary shape")
                    };
                    out.push("    chelis_list *values = chelis_dict_values(value);".to_string());
                    out.push(format!(
                        "    for ({index_type} index = 0; index < chelis_list_len(values); ++index) {{"
                    ));
                    out.extend(
                        child(
                            "{ path, NULL, index }".to_string(),
                            "chelis_list_index(values, index)".to_string(),
                            item,
                            *walker,
                            "NULL",
                        )
                        .into_iter()
                        .map(|line| format!("    {line}")),
                    );
                    out.push("    }".to_string());
                    out.push("    chelis_list_release(values);".to_string());
                }
            }
            out.push("}".to_string());
        }
        out.push(String::new());
    }
}

/// How a check names the tensor it reads: a format fragment and the
/// arguments that fragment consumes.
pub(super) struct TensorLabel {
    pub format: String,
    pub args: String,
}

impl TensorLabel {
    /// A tensor named at compile time, `input `x``.
    pub fn fixed(name: &str) -> Self {
        let name = chelis_ir::span_sanitize::sanitize_for_format_string(name);
        Self {
            format: format!("input `{name}`"),
            args: String::new(),
        }
    }

    /// A tensor named by a runtime path, rendered only when a check fails.
    fn path(path: &str) -> Self {
        Self {
            format: "input `%s`".to_string(),
            args: format!(", __chelis_entry_path_text({path})"),
        }
    }
}

/// A retained callback's formal List is checked at its inlined boundary.
/// These loops use the same metadata and extent renderers as a function entry.
pub(super) fn retained_list_pass(
    ty: &HostAbiType,
    actual: &str,
    name: &str,
    extents: bool,
    state_count: usize,
) -> Vec<String> {
    fn walk(
        ty: &HostAbiType,
        value: &str,
        path: &str,
        depth: usize,
        extents: bool,
        state_count: usize,
        out: &mut Vec<String>,
    ) {
        match ty {
            HostAbiType::List(item) => {
                if !extents {
                    out.push(format!(
                        "if ({value} == NULL) {{ fprintf(stderr, \"input `%s` is NULL\\n\", __chelis_entry_path_text({path})); abort(); }}"
                    ));
                }
                let index = format!("__chelis_list_index_{depth}");
                let element = format!("__chelis_list_element_{depth}");
                let child = format!("__chelis_list_child_{depth}");
                let child_path = format!("__chelis_list_path_{depth}");
                out.push(format!(
                    "for (int64_t {index} = 0; {index} < chelis_list_len({value}); ++{index}) {{"
                ));
                out.push(format!(
                    "    chelis_value {element} = chelis_list_index({value}, {index});"
                ));
                out.push(format!(
                    "    {}{child} = {};",
                    c_pointer_type(item),
                    borrow_value(item, &element)
                ));
                out.push(format!(
                    "    const __chelis_entry_path {child_path} = {{ {path}, NULL, {index} }};"
                ));
                let mut inner = Vec::new();
                walk(
                    item,
                    &child,
                    &format!("&{child_path}"),
                    depth + 1,
                    extents,
                    state_count,
                    &mut inner,
                );
                out.extend(inner.into_iter().map(|line| format!("    {line}")));
                out.push(format!("    chelis_value_release({element});"));
                out.push("}".into());
            }
            HostAbiType::Tensor(tensor) => {
                let label = TensorLabel::path(path);
                if extents {
                    for (axis, dim) in tensor.dims.iter().enumerate() {
                        match dim {
                            DimInfo::Lit(required) => {
                                out.extend(literal_extent_check(value, &label, axis, *required));
                            }
                            DimInfo::Named(key, _) if key != "*" => {
                                out.push(format!(
                                    "__chelis_entry_named_observe(__chelis_entry_named_states, {state_count}, {}, __chelis_entry_path_text({path}), {axis}, chelis_tensor_shape({value}, {axis}));",
                                    c_string_literal(key)
                                ));
                            }
                            _ => {}
                        }
                    }
                } else {
                    out.extend(tensor_metadata_checks(value, &label, tensor));
                }
            }
            _ => unreachable!("retained List entry contains only List layers and a tensor"),
        }
    }
    let mut out = vec![
        "{".into(),
        format!(
            "const __chelis_entry_path __chelis_list_root = {{ NULL, {}, 0 }};",
            c_utf8_byte_literal(name)
        ),
    ];
    walk(
        ty,
        actual,
        "&__chelis_list_root",
        0,
        extents,
        state_count,
        &mut out,
    );
    out.push("}".into());
    out
}

/// The null, dtype and rank checks every entry tensor gets, in that order.
pub(super) fn tensor_metadata_checks(
    actual: &str,
    label: &TensorLabel,
    ty: &TensorType,
) -> Vec<String> {
    let TensorLabel { format, args } = label;
    let rank = ty.dims.len();
    let mut lines = vec![format!(
        "if ({actual} == NULL) {{ fprintf(stderr, \"{format} is NULL\\n\"{args}); abort(); }}"
    )];
    lines.extend(CEmitter::entry_dtype_guard(actual, format, args, ty));
    lines.push(format!(
        "if (chelis_tensor_rank({actual}) != {rank}) {{ fprintf(stderr, \"{format} expected rank {rank}, got %d\\n\"{args}, chelis_tensor_rank({actual})); abort(); }}"
    ));
    lines
}

/// A section 4.7 literal-extent check of one entry tensor axis.
pub(super) fn literal_extent_check(
    actual: &str,
    label: &TensorLabel,
    axis: usize,
    required: usize,
) -> Vec<String> {
    let TensorLabel { format, args } = label;
    let observed = format!("chelis_tensor_shape({actual}, {axis})");
    vec![
        format!("if ({observed} != {required}) {{"),
        format!(
            "    fprintf(stderr, \"{format} axis {axis} expected {required}, got %lld\\n\"{args}, (long long)({observed}));"
        ),
        "    chelis_numeric_trap(\"numeric trap: domain in load at i64\");".to_string(),
        "}".to_string(),
    ]
}

#[cfg(test)]
mod entry_contract_tests {
    use super::*;
    use chelis_ir::host::{EntryContract, HostFunctionOrigin, HostParam};
    use chelis_ir::host_type_state::HostTypeTerm;
    use chelis_types::types::Prim;

    fn function() -> HostFunction {
        let literal = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let named = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let declared = [
            HostParam {
                name: "unused".into(),
                ty: HostTypeTerm::Unit,
            },
            HostParam {
                name: "xs".into(),
                ty: HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(literal.clone()))),
            },
            HostParam {
                name: "xss".into(),
                ty: HostTypeTerm::List(Box::new(HostTypeTerm::List(Box::new(
                    HostTypeTerm::Tensor(named.clone()),
                )))),
            },
        ];
        let entry_contract = EntryContract::from_params(&declared)
            .try_map_tensor(|_, ty| match ty {
                HostTypeTerm::Tensor(tensor) => Ok::<_, ()>(HostAbiType::Tensor(tensor.clone())),
                _ => Err(()),
            })
            .unwrap();
        HostFunction {
            helper_result_claim_axes: Vec::new(),
            name: "f".into(),
            entry_contract,
            params: [
                HostParam {
                    name: "unused".into(),
                    ty: HostAbiType::Unit,
                },
                HostParam {
                    name: "xs".into(),
                    ty: HostAbiType::List(Box::new(HostAbiType::Tensor(literal))),
                },
                HostParam {
                    name: "xss".into(),
                    ty: HostAbiType::List(Box::new(HostAbiType::List(Box::new(
                        HostAbiType::Tensor(named),
                    )))),
                },
            ]
            .into(),
            ret_ty: HostAbiType::Unit,
            body: HostExpr::new(HostExprKind::Unit),
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }
    }

    #[test]
    fn projection_requires_all_claimed_list_formals_in_order() {
        let mut function = function();
        assert!(validate_entry_contract(&function).is_ok());

        function.params.swap(1, 2);
        assert!(validate_entry_contract(&function).is_err());
        function.params.swap(1, 2);

        let HostAbiType::List(inner) = &mut function.params[1].ty else {
            panic!("literal List")
        };
        let HostAbiType::Tensor(tensor) = inner.as_mut() else {
            panic!("literal tensor")
        };
        tensor.dims[0] = DimInfo::Lit(3);
        assert!(validate_entry_contract(&function).is_err());
    }

    #[test]
    fn projection_rejects_a_missing_literal_only_contract() {
        let mut function = function();
        function.params.pop();
        function.entry_contract = EntryContract::default();
        assert!(validate_entry_contract(&function).is_err());
    }

    #[test]
    fn monomorphized_contract_keeps_claims_when_abi_axes_are_wildcards() {
        fn list_tensor(function: &mut HostFunction) -> &mut TensorType {
            let HostAbiType::List(inner) = &mut function.params[1].ty else {
                panic!("literal List")
            };
            let HostAbiType::Tensor(tensor) = inner.as_mut() else {
                panic!("literal tensor")
            };
            tensor
        }
        let mut function = function();
        function.origin = HostFunctionOrigin::Monomorphized;
        list_tensor(&mut function).dims[0] = DimInfo::Named("*".into(), None);
        assert!(validate_entry_contract(&function).is_ok());

        list_tensor(&mut function).precision = Prim::F64;
        assert!(validate_entry_contract(&function).is_err());
        list_tensor(&mut function).precision = Prim::F32;
        list_tensor(&mut function)
            .dims
            .push(DimInfo::Named("*".into(), None));
        assert!(validate_entry_contract(&function).is_err());
    }
}
