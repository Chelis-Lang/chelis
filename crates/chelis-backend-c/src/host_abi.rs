//! Resolved C-host ABI vocabulary for chelis#730 Phase 2.
//!
//! Logical host types are resolved in `chelis-ir` without consulting a
//! backend.  This module is the target-specific boundary: it accepts only a
//! [`ConcreteHostType`] and either produces an exact C-host representation or
//! returns the shared structured [`Unsupported`] diagnostic.  No unresolved
//! type term can be represented here, and no negative decision selects an
//! alternate ABI type.
//!
//! The ABI vocabulary and its constructor stay private to this backend:
//!
//! ```compile_fail
//! use chelis_backend_c::host_abi::HostAbiType;
//! ```

use chelis_ir::ConcreteHostType;
use chelis_ir::host::{
    ConcreteHostBinding, ConcreteHostCallback, ConcreteHostCallbackKind, ConcreteHostExpr,
    ConcreteHostExprKind, ConcreteHostParam, HostBinding, HostCallback, HostCallbackKind, HostExpr,
    HostExprKind, HostFunction, HostMatchArm, HostParam, HostPatternBinding, HostProgram,
    HostTensorHelper,
};
use chelis_ir::host_type_state::KeyBuiltinCallable;
use chelis_ir::ownership::{
    HostSiteId, HostSiteKind, VerifiedHostAction, VerifiedHostEmission, VerifiedHostFunctionView,
    VerifiedHostSiteActionKind, VerifiedHostTensorHelperView,
};
use chelis_types::types::Prim;
use chelis_types::unsupported::{RejectionAuthority, Stage, Unsupported, UnsupportedKind};
use chelis_unord::UnordSet;

/// A host value whose complete logical type has an implemented C ABI.
///
/// Kept crate-private so callers cannot manufacture a purportedly-resolved
/// ABI value.  Construction is exclusively through [`Self::try_from_concrete`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum HostAbiType {
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    BFloat16,
    Float32,
    Float64,
    Bool,
    String,
    /// A scalar random key ([05-OP-69]..[05-OP-72]): the C value
    /// `chelis_key`, `typedef struct { uint64_t bits; } chelis_key;`, whose
    /// `bits` are the key's 64 bits. It is never an integer: it has no
    /// arithmetic, cast or comparison (spec/04 section 1.1).
    Key,
    /// A typed C function pointer for a declared callback or a resolved
    /// compiler key-builtin value.
    ///
    /// This is deliberately not a general value representation. Projection
    /// admits a local binding only when its initializer is a resolved key
    /// builtin or an already admitted callable alias. Ordinary function
    /// results and container elements have no callback-value ABI; the closed
    /// key-builtin carrier below is a separate internal representation.
    Callback(Vec<HostAbiType>, Box<HostAbiType>),
    /// A closed, unspecialized operation. Its identity is fixed in the type;
    /// the private C value is an inert witness until a checked call selects
    /// one concrete callback signature.
    KeyBuiltinCallable(KeyBuiltinCallable),
    Adt(String, Vec<HostAbiType>),
    List(Box<HostAbiType>),
    Dict(Box<HostAbiType>, Box<HostAbiType>),
    Tuple(Vec<HostAbiType>),
    Tensor(chelis_ir::TensorType),
    Option(Box<HostAbiType>),
    MappedFile,
    Unit,
}

pub(crate) type HostAbiProgram = HostProgram<HostAbiType>;
pub(crate) type HostAbiBinding = HostBinding<HostAbiType>;
pub(crate) type HostAbiFunction = HostFunction<HostAbiType>;
pub(crate) type HostAbiParam = HostParam<HostAbiType>;
pub(crate) type HostAbiCallback = HostCallback<HostAbiType>;
pub(crate) type HostAbiCallbackKind = HostCallbackKind<HostAbiType>;
pub(crate) type HostAbiExpr = HostExpr<HostAbiType>;
pub(crate) type HostAbiExprKind = HostExprKind<HostAbiType>;
pub(crate) type HostAbiMatchArm = HostMatchArm<HostAbiType>;

/// Private ABI projection paired with the exact verified payload/site map it
/// was derived from. The projection may change type representation, but it
/// cannot detach the nested verified DAG children or their structural site
/// identities.
pub(crate) struct ProjectedHostProgram<'a> {
    program: HostAbiProgram,
    emission: VerifiedHostEmission<'a>,
    sites: Vec<ProjectedHostSite<'a>>,
    root_sites: Vec<ProjectedHostSite<'a>>,
    function_sites: Vec<Vec<ProjectedHostSite<'a>>>,
    function_owner_bindings: Vec<Vec<(chelis_ir::ownership::VerifiedOwnerId, String)>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ProjectedHostSite<'a> {
    pub(crate) id: HostSiteId,
    pub(crate) kind: HostSiteKind,
    pub(crate) actions: Vec<VerifiedHostSiteActionKind>,
    pub(crate) directives: Vec<VerifiedHostAction<'a>>,
}

impl<'a> ProjectedHostProgram<'a> {
    pub(crate) fn program(&self) -> &HostAbiProgram {
        &self.program
    }

    pub(crate) fn manifest(&self) -> &chelis_types::manifest::RootManifest {
        self.emission.manifest()
    }

    pub(crate) fn global_tensor_helper(
        &self,
        index: usize,
    ) -> Option<VerifiedHostTensorHelperView<'a>> {
        self.emission.global_tensor_helper(index)
    }

    pub(crate) fn function_tensor_helper(
        &self,
        function: usize,
        helper: usize,
    ) -> Option<VerifiedHostTensorHelperView<'a>> {
        self.emission.function(function)?.tensor_helper(helper)
    }

    pub(crate) fn sites(&self) -> &[ProjectedHostSite<'a>] {
        &self.sites
    }

    pub(crate) fn root_sites(&self) -> &[ProjectedHostSite<'a>] {
        &self.root_sites
    }

    pub(crate) fn function_sites(&self, index: usize) -> Option<&[ProjectedHostSite<'a>]> {
        self.function_sites.get(index).map(Vec::as_slice)
    }

    pub(crate) fn function_owner_bindings(
        &self,
        index: usize,
    ) -> Option<&[(chelis_ir::ownership::VerifiedOwnerId, String)]> {
        self.function_owner_bindings.get(index).map(Vec::as_slice)
    }
}

impl HostAbiType {
    /// Resolve the private pre-Table-B C-host capability adapter.
    ///
    /// This match is intentionally exhaustive over the logical vocabulary.
    /// Table B (chelis#729 Phase 4) will replace the decisions without
    /// changing this fallible boundary.
    pub(crate) fn try_from_concrete(ty: &ConcreteHostType) -> Result<Self, Unsupported> {
        Ok(match ty {
            ConcreteHostType::Scalar(Prim::Int8) => Self::Int8,
            ConcreteHostType::Scalar(Prim::Int16) => Self::Int16,
            ConcreteHostType::Scalar(Prim::Int32) => Self::Int32,
            ConcreteHostType::Scalar(Prim::Int64) => Self::Int64,
            ConcreteHostType::Scalar(Prim::F16) => Self::Float16,
            ConcreteHostType::Scalar(Prim::Bf16) => Self::BFloat16,
            ConcreteHostType::Scalar(Prim::F32) => Self::Float32,
            ConcreteHostType::Scalar(Prim::F64) => Self::Float64,
            ConcreteHostType::Scalar(Prim::Bool) => Self::Bool,
            ConcreteHostType::Scalar(Prim::String) => Self::String,
            // f8e4m3 is deferred and inadmissible in the active language per
            // spec/04-type-system.md section 1.1.1.  It still has a `Prim`
            // identity so this boundary can reject it precisely.
            ConcreteHostType::Scalar(Prim::F8e4m3) => {
                return Err(rejected_dtype(
                    Prim::F8e4m3,
                    chelis_types::deliberate_rejection!(
                        "[04-DTYPE-1]",
                        "f8e4m3 is reserved but not active; use an active dtype"
                    ),
                ));
            }
            ConcreteHostType::Scalar(Prim::Key) => Self::Key,
            ConcreteHostType::Function(_, _) => {
                return Err(unsupported_function_value(ty, "C host ABI value selection"));
            }
            ConcreteHostType::KeyBuiltinCallable(op) => Self::KeyBuiltinCallable(*op),
            ConcreteHostType::Adt(name, args) => Self::Adt(
                name.clone(),
                args.iter()
                    .map(Self::try_from_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ConcreteHostType::List(inner) => Self::List(Box::new(Self::try_from_concrete(inner)?)),
            ConcreteHostType::Dict(key, value) => Self::Dict(
                Box::new(Self::try_from_concrete(key)?),
                Box::new(Self::try_from_concrete(value)?),
            ),
            ConcreteHostType::Tuple(items) => Self::Tuple(
                items
                    .iter()
                    .map(Self::try_from_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ConcreteHostType::Tensor(tensor) => {
                if tensor.precision == Prim::F8e4m3 {
                    return Err(rejected_dtype(
                        Prim::F8e4m3,
                        chelis_types::deliberate_rejection!(
                            "[04-DTYPE-1]",
                            "f8e4m3 is reserved but not active; use an active dtype"
                        ),
                    ));
                }
                Self::Tensor(tensor.clone())
            }
            ConcreteHostType::Option(inner) => {
                Self::Option(Box::new(Self::try_from_concrete(inner)?))
            }
            ConcreteHostType::MappedFile => Self::MappedFile,
            ConcreteHostType::Unit => Self::Unit,
        })
    }

    /// Return the standalone C type spelling for value ABIs.
    ///
    /// A callback has no standalone spelling: C function-pointer syntax
    /// requires the identifier inside the declarator.  Keeping that state out
    /// of this method removes the former `Fn -> void *` erasure path.
    pub(crate) fn c_type_name(&self) -> Option<&'static str> {
        match self {
            Self::Int8 => Some("int8_t"),
            Self::Int16 => Some("int16_t"),
            Self::Int32 => Some("int32_t"),
            Self::Int64 => Some("int64_t"),
            Self::Float16 | Self::BFloat16 => Some("uint16_t"),
            Self::Float32 => Some("float"),
            Self::Float64 => Some("double"),
            Self::Bool => Some("bool"),
            Self::String => Some("chelis_string"),
            Self::Key => Some("chelis_key"),
            Self::Callback(_, _) => None,
            Self::KeyBuiltinCallable(_) => Some("chelis_key_callable"),
            Self::Adt(_, _) => Some("chelis_adt*"),
            Self::List(_) => Some("chelis_list*"),
            Self::Dict(_, _) => Some("chelis_dict*"),
            Self::Tuple(_) => Some("chelis_tuple*"),
            Self::Tensor(_) => Some("chelis_tensor*"),
            Self::MappedFile => Some("chelis_mapped_file*"),
            Self::Option(_) => Some("chelis_option*"),
            Self::Unit => Some("int"),
        }
    }

    /// Return the C value a local of this ABI holds once the owner it
    /// aliased has been released, so the alias never carries a freed handle.
    ///
    /// `None` for an ABI whose C value carries no handle. The match names
    /// every variant so a new heap ABI must choose its cleared spelling: a
    /// `string` is a struct, and a pointer `NULL` does not convert to it.
    pub(crate) fn c_released_value(&self) -> Option<&'static str> {
        match self {
            Self::Int8
            | Self::Int16
            | Self::Int32
            | Self::Int64
            | Self::Float16
            | Self::BFloat16
            | Self::Float32
            | Self::Float64
            | Self::Bool
            // A `chelis_key` is its 64 bits by value and owns no heap handle.
            | Self::Key
            | Self::Unit
            | Self::Callback(_, _) => None,
            Self::KeyBuiltinCallable(_) => None,
            Self::String => Some("(chelis_string){ NULL }"),
            Self::Adt(_, _)
            | Self::List(_)
            | Self::Dict(_, _)
            | Self::Tuple(_)
            | Self::Tensor(_)
            | Self::MappedFile
            | Self::Option(_) => Some("NULL"),
        }
    }

    fn try_callback_signature(ty: &ConcreteHostType) -> Result<Self, Unsupported> {
        let ConcreteHostType::Function(params, ret) = ty else {
            return Err(invalid_callback_shape(format!(
                "callback position carries non-function type {ty:?}"
            )));
        };
        Ok(Self::Callback(
            params
                .iter()
                .map(Self::try_from_concrete)
                .collect::<Result<Vec<_>, _>>()?,
            Box::new(Self::try_from_concrete(ret)?),
        ))
    }
}

pub(crate) fn project_program(
    emission: VerifiedHostEmission<'_>,
) -> Result<ProjectedHostProgram<'_>, Unsupported> {
    let adt_layouts = emission.adt_layouts();
    let declared_callbacks = (0..emission.function_count())
        .filter_map(|index| emission.function(index))
        .map(|function| function.name().to_string())
        .collect::<UnordSet<_>>();
    let program = HostAbiProgram {
        globals: emission
            .globals()
            .iter()
            .cloned()
            .map(|binding| project_binding(binding, &declared_callbacks))
            .collect::<Result<Vec<_>, _>>()?,
        global_tensor_helpers: (0..emission.global_tensor_helper_count())
            .map(|index| {
                helper_metadata(
                    emission
                        .global_tensor_helper(index)
                        .expect("verified global helper census"),
                )
            })
            .collect(),
        functions: (0..emission.function_count())
            .filter_map(|index| emission.function(index))
            .map(|function| project_function(function, &declared_callbacks, adt_layouts))
            .collect::<Result<Vec<_>, _>>()?,
        summary_rejections: emission.summary_rejections().to_vec(),
        adt_layouts: emission
            .adt_layouts()
            .iter()
            .map(project_adt_layout)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect(),
    };
    let sites = emission.sites().map(project_site).collect();
    let root_sites = emission.root_sites().map(project_site).collect();
    let function_sites = (0..emission.function_count())
        .map(|index| {
            emission
                .function(index)
                .expect("verified function census")
                .sites()
                .map(project_site)
                .collect()
        })
        .collect();
    let function_owner_bindings = (0..emission.function_count())
        .map(|index| {
            emission
                .function(index)
                .expect("verified function census")
                .body_bindings()
                .into_iter()
                .map(|binding| (binding.owner().id(), binding.name().to_string()))
                .collect()
        })
        .collect();
    Ok(ProjectedHostProgram {
        program,
        emission,
        sites,
        root_sites,
        function_sites,
        function_owner_bindings,
    })
}

/// Project one parameter ADT layout for entry validation.
///
/// A layout whose ADT type has no C representation is dropped, since no
/// parameter can carry it. A field that holds no tensor value, such as a
/// function, is walked past, so it projects as `Unit` when its own type has
/// no C representation; a field that holds a tensor must project exactly.
fn project_adt_layout(
    layout: &chelis_ir::host::HostAdtLayout<ConcreteHostType>,
) -> Result<Option<chelis_ir::host::HostAdtLayout<HostAbiType>>, Unsupported> {
    fn carries_tensor(ty: &ConcreteHostType) -> bool {
        match ty {
            ConcreteHostType::Tensor(_) => true,
            ConcreteHostType::Adt(_, items) | ConcreteHostType::Tuple(items) => {
                items.iter().any(carries_tensor)
            }
            ConcreteHostType::List(inner) | ConcreteHostType::Option(inner) => {
                carries_tensor(inner)
            }
            ConcreteHostType::Dict(key, value) => carries_tensor(key) || carries_tensor(value),
            _ => false,
        }
    }
    let Ok(ty) = HostAbiType::try_from_concrete(&layout.ty) else {
        return Ok(None);
    };
    let constructors = layout
        .constructors
        .iter()
        .map(|constructor| {
            let fields = constructor
                .fields
                .iter()
                .map(|field| {
                    let ty = match HostAbiType::try_from_concrete(&field.ty) {
                        Ok(ty) => ty,
                        Err(_) if !carries_tensor(&field.ty) => HostAbiType::Unit,
                        Err(error) => return Err(error),
                    };
                    Ok(chelis_ir::host::HostAdtField {
                        name: field.name.clone(),
                        ty,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(chelis_ir::host::HostAdtConstructorLayout {
                name: constructor.name.clone(),
                fields,
            })
        })
        .collect::<Result<Vec<_>, Unsupported>>()?;
    Ok(Some(chelis_ir::host::HostAdtLayout { ty, constructors }))
}

fn project_site<'a>(site: chelis_ir::ownership::VerifiedHostSiteView<'a>) -> ProjectedHostSite<'a> {
    ProjectedHostSite {
        id: site.id(),
        kind: site.kind(),
        actions: site.action_kinds().collect(),
        directives: site.actions().collect(),
    }
}

fn helper_metadata(helper: VerifiedHostTensorHelperView<'_>) -> HostTensorHelper {
    let mut dag = chelis_ir::dag::Dag::new();
    let verified = helper.dag();
    if verified.roots().len() == 1
        && helper.inputs().len() == 1
        && let Some(node) = verified.get(verified.roots()[0])
        && let chelis_ir::dag::RiscOp::Load { name } = &node.op
        && node.output_type == *helper.output()
        && helper.inputs().iter().any(|input| input.name == *name)
    {
        // The metadata graph restates the helper's identity Load, so it
        // belongs to a declaration of the helper's name.
        let decl = dag.declare(verified.declaration(node.owner.decl).name.clone());
        let root = dag.add_node(
            decl,
            chelis_ir::dag::RiscOp::Load { name: name.clone() },
            Vec::new(),
            node.output_type.clone(),
            node.span_id.clone(),
        );
        dag.add_root(root);
    }
    HostTensorHelper {
        name: helper.name().to_string(),
        dag,
        inputs: helper.inputs().to_vec(),
        output: helper.output().clone(),
        specialization: helper.specialization().cloned(),
        summary_rejection: helper.summary_rejection().cloned(),
    }
}

pub(crate) fn project_binding(
    binding: ConcreteHostBinding,
    allowed_callbacks: &UnordSet<String>,
) -> Result<HostAbiBinding, Unsupported> {
    Ok(HostAbiBinding {
        name: binding.name,
        display_name: binding.display_name,
        display_roots: binding.display_roots,
        ty: if matches!(binding.ty, ConcreteHostType::Function(_, _)) {
            HostAbiType::try_callback_signature(&binding.ty)?
        } else {
            HostAbiType::try_from_concrete(&binding.ty)?
        },
        value: project_expr(binding.value, allowed_callbacks)?,
    })
}

fn project_function(
    function: VerifiedHostFunctionView<'_>,
    declared_callbacks: &UnordSet<String>,
    adt_layouts: &[chelis_ir::host::HostAdtLayout<ConcreteHostType>],
) -> Result<HostAbiFunction, Unsupported> {
    // Authored functions are externally declared even if unused by the
    // program's main. A closed builtin witness has no public callable ABI:
    // exporting it inside a tuple, option, collection or ADT would publish
    // an inert value and lose the operation selected by the checker.
    if function.origin() == chelis_ir::host::HostFunctionOrigin::Authored
        && public_type_contains_key_callable(function.ret_ty(), adt_layouts)
    {
        return Err(unsupported_function_value(
            function.ret_ty(),
            "C host public function result",
        ));
    }
    if function.origin() == chelis_ir::host::HostFunctionOrigin::Authored {
        for param in function.params() {
            if public_type_contains_key_callable(&param.ty, adt_layouts) {
                return Err(unsupported_function_value(
                    &param.ty,
                    "C host public function parameter",
                ));
            }
        }
    }
    let mut allowed_callbacks = declared_callbacks.clone();
    for param in function.params() {
        if matches!(param.ty, ConcreteHostType::Function(_, _)) {
            allowed_callbacks.insert(param.name.clone());
        }
    }
    Ok(HostAbiFunction {
        helper_result_claim_axes: function.helper_result_claim_axes().to_vec(),
        name: function.name().to_string(),
        entry_contract: function
            .entry_contract()
            .try_map_tensor(|_, tensor| HostAbiType::try_from_concrete(tensor))?,
        params: function
            .params()
            .iter()
            .cloned()
            .map(project_function_param)
            .collect::<Result<Vec<_>, _>>()?,
        ret_ty: HostAbiType::try_from_concrete(function.ret_ty())?,
        body: project_expr(function.body().clone(), &allowed_callbacks)?,
        tensor_helpers: (0..function.tensor_helper_count())
            .map(|index| {
                helper_metadata(
                    function
                        .tensor_helper(index)
                        .expect("verified function helper census"),
                )
            })
            .collect(),
        origin: function.origin(),
        specialization: function.specialization().cloned(),
        summary_rejections: function.summary_rejections().to_vec(),
    })
}

/// Trace both structural children and nominal fields reachable from a
/// published type. Local values may use the private carrier; only an
/// authored function declaration crosses this boundary.
pub(crate) fn public_type_contains_key_callable(
    ty: &ConcreteHostType,
    layouts: &[chelis_ir::host::HostAdtLayout<ConcreteHostType>],
) -> bool {
    fn contains(
        ty: &ConcreteHostType,
        layouts: &[chelis_ir::host::HostAdtLayout<ConcreteHostType>],
        visited: &mut UnordSet<String>,
    ) -> bool {
        match ty {
            ConcreteHostType::KeyBuiltinCallable(_) => true,
            ConcreteHostType::Function(params, ret) => {
                params.iter().any(|param| contains(param, layouts, visited))
                    || contains(ret, layouts, visited)
            }
            ConcreteHostType::Adt(name, args) => {
                if args.iter().any(|arg| contains(arg, layouts, visited)) {
                    return true;
                }
                if !visited.insert(name.clone()) {
                    return false;
                }
                let found = layouts
                    .iter()
                    .filter(|layout| {
                        matches!(&layout.ty, ConcreteHostType::Adt(layout_name, _) if layout_name == name)
                    })
                    .flat_map(|layout| &layout.constructors)
                    .flat_map(|constructor| &constructor.fields)
                    .any(|field| contains(&field.ty, layouts, visited));
                visited.remove(name);
                found
            }
            ConcreteHostType::List(inner) | ConcreteHostType::Option(inner) => {
                contains(inner, layouts, visited)
            }
            ConcreteHostType::Dict(key, value) => {
                contains(key, layouts, visited) || contains(value, layouts, visited)
            }
            ConcreteHostType::Tuple(items) => {
                items.iter().any(|item| contains(item, layouts, visited))
            }
            ConcreteHostType::Scalar(_)
            | ConcreteHostType::Tensor(_)
            | ConcreteHostType::MappedFile
            | ConcreteHostType::Unit => false,
        }
    }
    contains(ty, layouts, &mut UnordSet::new())
}

fn project_value_param(param: ConcreteHostParam) -> Result<HostAbiParam, Unsupported> {
    Ok(HostAbiParam {
        name: param.name,
        ty: HostAbiType::try_from_concrete(&param.ty)?,
    })
}

fn project_function_param(param: ConcreteHostParam) -> Result<HostAbiParam, Unsupported> {
    let ty = match &param.ty {
        ConcreteHostType::Function(_, _) => HostAbiType::try_callback_signature(&param.ty)?,
        _ => HostAbiType::try_from_concrete(&param.ty)?,
    };
    Ok(HostAbiParam {
        name: param.name,
        ty,
    })
}

fn project_callback(
    callback: ConcreteHostCallback,
    allowed_callbacks: &UnordSet<String>,
) -> Result<HostAbiCallback, Unsupported> {
    let kind = match callback.kind {
        ConcreteHostCallbackKind::Named { function, params } => HostAbiCallbackKind::Named {
            function: if allowed_callbacks.contains(&function) {
                function
            } else {
                return Err(unsupported_function_symbol(&function));
            },
            params: params
                .into_iter()
                .map(project_value_param)
                .collect::<Result<Vec<_>, _>>()?,
        },
        ConcreteHostCallbackKind::Inline { params, body } => HostAbiCallbackKind::Inline {
            params: params
                .into_iter()
                .map(project_value_param)
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(project_expr(*body, allowed_callbacks)?),
        },
    };
    Ok(HostAbiCallback {
        kind,
        ret_ty: HostAbiType::try_from_concrete(&callback.ret_ty)?,
    })
}

fn project_expr(
    expr: ConcreteHostExpr,
    allowed_callbacks: &UnordSet<String>,
) -> Result<HostAbiExpr, Unsupported> {
    let kind = match expr.kind {
        ConcreteHostExprKind::ResultClaimScope { plan, body, ty } => {
            HostAbiExprKind::ResultClaimScope {
                plan,
                body: Box::new(project_expr(*body, allowed_callbacks)?),
                ty: HostAbiType::try_from_concrete(&ty)?,
            }
        }
        ConcreteHostExprKind::FormalIngress { value, ty } => HostAbiExprKind::FormalIngress {
            value: Box::new(project_expr(*value, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::ExtentSites { value, sites, ty } => HostAbiExprKind::ExtentSites {
            value: Box::new(project_expr(*value, allowed_callbacks)?),
            sites,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Int(value) => HostAbiExprKind::Int(value),
        ConcreteHostExprKind::Float(value) => HostAbiExprKind::Float(value),
        ConcreteHostExprKind::Bool(value) => HostAbiExprKind::Bool(value),
        ConcreteHostExprKind::String(value) => HostAbiExprKind::String(value),
        ConcreteHostExprKind::List(items, ty) => HostAbiExprKind::List(
            items
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
                .collect::<Result<Vec<_>, _>>()?,
            HostAbiType::try_from_concrete(&ty)?,
        ),
        ConcreteHostExprKind::Tuple(items, ty) => HostAbiExprKind::Tuple(
            items
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
                .collect::<Result<Vec<_>, _>>()?,
            HostAbiType::try_from_concrete(&ty)?,
        ),
        ConcreteHostExprKind::Var(name, ty) => HostAbiExprKind::Var(
            name.clone(),
            if matches!(ty, ConcreteHostType::Function(_, _)) && allowed_callbacks.contains(&name) {
                HostAbiType::try_callback_signature(&ty)?
            } else {
                HostAbiType::try_from_concrete(&ty)?
            },
        ),
        ConcreteHostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            if chelis_ir::host::is_host_unresolved_marker(&function) {
                return Err(unsupported_callable_use(&function));
            }
            if args.len() != arg_tys.len() {
                return Err(invalid_callback_shape(format!(
                    "call `{function}` has {} arguments but {} checked argument types",
                    args.len(),
                    arg_tys.len()
                )));
            }
            let (args, arg_tys) = args
                .into_iter()
                .zip(arg_tys.iter())
                .map(|(arg, arg_ty)| {
                    if matches!(arg_ty, ConcreteHostType::Function(_, _)) {
                        Ok((
                            project_callback_argument(arg, arg_ty, allowed_callbacks)?,
                            HostAbiType::try_callback_signature(arg_ty)?,
                        ))
                    } else {
                        Ok((
                            project_expr(arg, allowed_callbacks)?,
                            HostAbiType::try_from_concrete(arg_ty)?,
                        ))
                    }
                })
                .collect::<Result<Vec<_>, Unsupported>>()?
                .into_iter()
                .unzip();
            HostAbiExprKind::Call {
                function,
                args,
                arg_tys,
                ty: HostAbiType::try_from_concrete(&ty)?,
            }
        }
        ConcreteHostExprKind::SignatureEntry {
            contract,
            plan,
            args,
            positions,
            lists,
        } => HostAbiExprKind::SignatureEntry {
            contract: contract
                .try_map_tensor(|_, tensor| HostAbiType::try_from_concrete(tensor))?,
            plan,
            args: args
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
                .collect::<Result<Vec<_>, _>>()?,
            positions,
            lists: lists
                .into_iter()
                .map(|entry| {
                    Ok(chelis_ir::host::HostListEntry {
                        position: entry.position,
                        name: entry.name,
                        ty: HostAbiType::try_from_concrete(&entry.ty)?,
                        value: project_expr(entry.value, allowed_callbacks)?,
                    })
                })
                .collect::<Result<Vec<_>, Unsupported>>()?,
        },
        ConcreteHostExprKind::Builtin { name, args, ty } => {
            if chelis_ir::host::is_host_unresolved_marker(&name) {
                return Err(unsupported_callable_use(&name));
            }
            let abi_ty = if args.is_empty()
                && matches!(
                    name.as_str(),
                    "key_from_seed" | "split_key" | "split_keys" | "fold_in"
                )
                && matches!(ty, ConcreteHostType::Function(_, _))
            {
                HostAbiType::try_callback_signature(&ty)?
            } else {
                HostAbiType::try_from_concrete(&ty)?
            };
            HostAbiExprKind::Builtin {
                name,
                args: args
                    .into_iter()
                    .map(|expr| project_expr(expr, allowed_callbacks))
                    .collect::<Result<Vec<_>, _>>()?,
                ty: abi_ty,
            }
        }
        ConcreteHostExprKind::AdtConstruct { ctor, fields, ty } => HostAbiExprKind::AdtConstruct {
            ctor,
            fields: fields
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
                .collect::<Result<Vec<_>, _>>()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::AdtFieldAccess {
            base,
            field_index,
            ty,
        } => HostAbiExprKind::AdtFieldAccess {
            base: Box::new(project_expr(*base, allowed_callbacks)?),
            field_index,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => HostAbiExprKind::If {
            cond: Box::new(project_expr(*cond, allowed_callbacks)?),
            then_expr: Box::new(project_expr(*then_expr, allowed_callbacks)?),
            else_expr: Box::new(project_expr(*else_expr, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => HostAbiExprKind::MatchOption {
            scrutinee: Box::new(project_expr(*scrutinee, allowed_callbacks)?),
            bind_name,
            some_expr: Box::new(project_expr(*some_expr, allowed_callbacks)?),
            none_expr: Box::new(project_expr(*none_expr, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => HostAbiExprKind::MatchAdt {
            scrutinee: Box::new(project_expr(*scrutinee, allowed_callbacks)?),
            arms: arms
                .into_iter()
                .map(|arm| {
                    Ok(HostMatchArm {
                        ctor: arm.ctor,
                        bindings: arm
                            .bindings
                            .into_iter()
                            .map(|binding| {
                                Ok(HostPatternBinding {
                                    name: binding.name,
                                    ty: HostAbiType::try_from_concrete(&binding.ty)?,
                                    field_index: binding.field_index,
                                })
                            })
                            .collect::<Result<Vec<_>, Unsupported>>()?,
                        expr: project_expr(arm.expr, allowed_callbacks)?,
                    })
                })
                .collect::<Result<Vec<_>, Unsupported>>()?,
            default_expr: default_expr
                .map(|expr| project_expr(*expr, allowed_callbacks).map(Box::new))
                .transpose()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Let { bindings, body, ty } => {
            let mut visible = allowed_callbacks.clone();
            let mut projected = Vec::with_capacity(bindings.len());
            for binding in bindings {
                let admitted = matches!(binding.ty, ConcreteHostType::Function(_, _))
                    && match &binding.value.kind {
                        ConcreteHostExprKind::Builtin { name, args, .. } => {
                            args.is_empty()
                                && matches!(
                                    name.as_str(),
                                    "key_from_seed" | "split_key" | "split_keys" | "fold_in"
                                )
                        }
                        ConcreteHostExprKind::Var(name, _) => visible.contains(name),
                        _ => false,
                    };
                let projected_binding = project_binding(binding, &visible)?;
                if admitted {
                    visible.insert(projected_binding.name.clone());
                }
                projected.push(projected_binding);
            }
            HostAbiExprKind::Let {
                bindings: projected,
                body: Box::new(project_expr(*body, &visible)?),
                ty: HostAbiType::try_from_concrete(&ty)?,
            }
        }
        ConcreteHostExprKind::RetainedInvocation { bindings, body, ty } => {
            HostAbiExprKind::RetainedInvocation {
                bindings: bindings
                    .into_iter()
                    .map(|binding| project_binding(binding, allowed_callbacks))
                    .collect::<Result<Vec<_>, _>>()?,
                body: Box::new(project_expr(*body, allowed_callbacks)?),
                ty: HostAbiType::try_from_concrete(&ty)?,
            }
        }
        ConcreteHostExprKind::Map { callback, list, ty } => HostAbiExprKind::Map {
            callback: project_callback(callback, allowed_callbacks)?,
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Filter { callback, list, ty } => HostAbiExprKind::Filter {
            callback: project_callback(callback, allowed_callbacks)?,
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        } => HostAbiExprKind::Fold {
            callback: project_callback(callback, allowed_callbacks)?,
            init: Box::new(project_expr(*init, allowed_callbacks)?),
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        } => HostAbiExprKind::Scan {
            callback: project_callback(callback, allowed_callbacks)?,
            init: Box::new(project_expr(*init, allowed_callbacks)?),
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Partition { callback, list, ty } => HostAbiExprKind::Partition {
            callback: project_callback(callback, allowed_callbacks)?,
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::FlatMap { callback, list, ty } => HostAbiExprKind::FlatMap {
            callback: project_callback(callback, allowed_callbacks)?,
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::TensorCall { helper, args, ty } => HostAbiExprKind::TensorCall {
            helper,
            args: args
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
                .collect::<Result<Vec<_>, _>>()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Unit => HostAbiExprKind::Unit,
    };
    Ok(HostAbiExpr {
        kind,
        span_id: expr.span_id,
        merged_spans: expr.merged_spans,
    })
}

fn project_callback_argument(
    expr: ConcreteHostExpr,
    expected: &ConcreteHostType,
    allowed_callbacks: &UnordSet<String>,
) -> Result<HostAbiExpr, Unsupported> {
    let ConcreteHostExprKind::Var(name, actual) = expr.kind else {
        return Err(unsupported_function_value(
            expected,
            "C host callback argument selection",
        ));
    };
    if let ConcreteHostType::KeyBuiltinCallable(op) = actual {
        return Ok(HostAbiExpr {
            kind: HostAbiExprKind::Builtin {
                name: op.symbol().to_string(),
                args: Vec::new(),
                ty: HostAbiType::try_callback_signature(expected)?,
            },
            span_id: expr.span_id,
            merged_spans: expr.merged_spans,
        });
    }
    if &actual != expected {
        return Err(invalid_callback_shape(format!(
            "callback `{name}` has type {actual:?}, expected {expected:?}"
        )));
    }
    if !allowed_callbacks.contains(&name) {
        return Err(unsupported_function_symbol(&name));
    }
    Ok(HostAbiExpr {
        kind: HostAbiExprKind::Var(name, HostAbiType::try_callback_signature(expected)?),
        span_id: expr.span_id,
        merged_spans: expr.merged_spans,
    })
}

fn rejected_dtype(precision: Prim, authority: RejectionAuthority) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Dtype(precision.name().to_string()),
        "C host ABI selection",
        Stage::Codegen("c"),
        authority,
    )
}

fn unsupported_function_value(ty: &ConcreteHostType, context: &'static str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::HostAbi(format!("function value `{ty:?}`")),
        context,
        Stage::Codegen("c"),
        chelis_types::unimplemented_rejection!(
            879,
            "the C host backend supports typed callback parameters and direct statically-known \
             callback arguments, but no first-class function-value ABI; specialize the call or \
             reject the containing construct"
        ),
    )
}

fn unsupported_function_symbol(name: &str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::HostAbi(format!("function value `{name}`")),
        "C host callback argument selection",
        Stage::Codegen("c"),
        chelis_types::unimplemented_rejection!(
            879,
            "only a declared function symbol or an in-scope typed callback parameter can cross \
             this boundary; dynamic function values have no C host ABI"
        ),
    )
}

/// The frozen rejection for an internal unresolved-callee marker. The
/// marker spelling never enters the diagnostic; the payload names the
/// semantic class instead, per marker kind.
fn unsupported_callable_use(marker: &str) -> Unsupported {
    if marker == chelis_ir::host::HOST_UNRESOLVED_TRANSFORM_MARKER {
        return Unsupported::new(
            UnsupportedKind::HostAbi("unresolved `grad`/`vmap` transform application".to_string()),
            "C host ABI callable-use projection",
            Stage::Codegen("c"),
            chelis_types::unimplemented_rejection!(
                879,
                "general C-host transformed function values are not implemented; rewrite the \
                 differentiated body to pure tensor ops (sum, add, mul, einsum) or run under \
                 `chelis eval`"
            ),
        );
    }
    Unsupported::new(
        UnsupportedKind::HostAbi("unresolved function value".to_string()),
        "C host ABI callable-use projection",
        Stage::Codegen("c"),
        chelis_types::unimplemented_rejection!(
            879,
            "the host lowerer did not resolve this application to a declared function symbol or \
             typed callback parameter; unresolved callables have no raw C call target"
        ),
    )
}

fn invalid_callback_shape(detail: String) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Construct(detail),
        "C host callback ABI projection",
        Stage::Codegen("c"),
        chelis_types::deliberate_rejection!(
            "[04-TOT-2]",
            "checked callable metadata and the resolved host program disagree; no fallback \
             callable representation is permitted"
        ),
    )
}
