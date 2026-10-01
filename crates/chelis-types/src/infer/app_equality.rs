//! chelis#2587: [05-OP-36]'s recursive equality domain.
//!
//! `eq` and `neq` admit unit and two `List`, tuple, `Dict`, `Option`, or ADT
//! values of one static type whose reachable fields are recursively
//! equality-comparable. Functions and resource handles are not. Unification
//! already holds the two operands to one static type; this module decides
//! whether that type's reachable structure is comparable. A `key` is refused
//! separately, by the key rule of spec/04 section 1.1 that linearity enforces
//! for every key-carrying operand.
//!
//! A data type's fields are summarized once per definition rather than
//! instantiated per use: each summary records the function or handle the
//! definition's own fields reach and the type parameters whose arguments they
//! reach. Walking an instance then descends only into its arguments, so a
//! definition that recurses through an ever larger instantiation of itself
//! still terminates.

use super::*;
use crate::adt::AdtDef;

/// The verdict [05-OP-36]'s domain gives one operand type.
pub(super) enum EqualityDomain {
    /// Every reachable field is equality-comparable.
    Admitted,
    /// A reachable field's type is a variable no dtype family restricts, so the
    /// verdict waits for it to bind.
    Awaits,
    /// The first function or resource handle the type reaches.
    Rejected(Type),
}

/// The language's resource-handle types: `mmap_file` returns the only one.
fn is_resource_handle(name: &str) -> bool {
    name == "MappedFile"
}

/// What a walk over one type reached.
#[derive(Default)]
struct Reach {
    incomparable: Option<Type>,
    variables: Vec<TypeVar>,
}

/// What one data type definition's own fields reach, for any arguments.
#[derive(Clone, Default, PartialEq)]
struct Summary {
    incomparable: Option<Type>,
    /// Positions in the definition's parameter list whose argument a field
    /// reaches.
    parameters: BTreeSet<usize>,
}

fn type_arguments(name: &str, args: &[Type]) -> (String, Vec<NominalArg>) {
    (
        name.to_string(),
        args.iter().cloned().map(NominalArg::Type).collect(),
    )
}

/// Walk `ty` without descending into a definition: a registered data type
/// contributes its summary, and an unregistered one (`Dict`) every argument.
fn reach(ty: &Type, summaries: &BTreeMap<String, Summary>) -> Reach {
    let mut reached = Reach::default();
    let mut pending = vec![ty.clone()];
    while let Some(ty) = pending.pop() {
        if reached.incomparable.is_some() {
            break;
        }
        let (name, args) = match ty {
            Type::Fn(..) => {
                reached.incomparable = Some(ty);
                continue;
            }
            Type::Adt(ref name, _) if is_resource_handle(name) => {
                reached.incomparable = Some(ty);
                continue;
            }
            Type::Adt(name, args) => type_arguments(&name, &args),
            Type::KindedAdt(name, args) => (name, args),
            Type::Tuple(items) => {
                pending.extend(items.into_iter().rev());
                continue;
            }
            Type::Ref(inner) => {
                pending.push(*inner);
                continue;
            }
            Type::Var(var) => {
                reached.variables.push(var);
                continue;
            }
            Type::Prim(_) | Type::Tensor(..) | Type::Unit | Type::Error(_) => continue,
        };
        let type_argument = |position: usize| match args.get(position) {
            Some(NominalArg::Type(argument)) => Some(argument.clone()),
            Some(NominalArg::Dimension(_)) | None => None,
        };
        match summaries.get(&name) {
            Some(summary) => {
                if let Some(incomparable) = &summary.incomparable {
                    reached.incomparable = Some(incomparable.clone());
                } else {
                    pending.extend(
                        summary
                            .parameters
                            .iter()
                            .rev()
                            .filter_map(|p| type_argument(*p)),
                    );
                }
            }
            None => pending.extend((0..args.len()).rev().filter_map(type_argument)),
        }
    }
    reached
}

/// The data type names `ty` reaches by name, through arguments and fields.
fn reachable_definitions<'a>(ty: &Type, adt_reg: &'a AdtRegistry) -> BTreeMap<String, &'a AdtDef> {
    let mut definitions = BTreeMap::new();
    let mut pending = vec![ty.clone()];
    while let Some(ty) = pending.pop() {
        match ty {
            Type::Adt(name, args) => {
                pending.extend(args);
                if let Some(def) = adt_reg.lookup(&name)
                    && !definitions.contains_key(&name)
                {
                    pending.extend(
                        def.variants
                            .iter()
                            .flat_map(|v| v.fields.iter().map(|(_, f)| f.clone())),
                    );
                    definitions.insert(name, def);
                }
            }
            Type::KindedAdt(name, args) => {
                pending.extend(args.into_iter().filter_map(|arg| match arg {
                    NominalArg::Type(argument) => Some(argument),
                    NominalArg::Dimension(_) => None,
                }));
                if let Some(def) = adt_reg.lookup(&name)
                    && !definitions.contains_key(&name)
                {
                    pending.extend(
                        def.variants
                            .iter()
                            .flat_map(|v| v.fields.iter().map(|(_, f)| f.clone())),
                    );
                    definitions.insert(name, def);
                }
            }
            Type::Tuple(items) => pending.extend(items),
            Type::Ref(inner) => pending.push(*inner),
            // A function is incomparable whatever it mentions.
            Type::Fn(..)
            | Type::Prim(_)
            | Type::Tensor(..)
            | Type::Var(_)
            | Type::Unit
            | Type::Error(_) => {}
        }
    }
    definitions
}

/// Summarize every definition `ty` reaches, as the least fixed point over
/// them: a summary only grows, and each is bounded by its definition.
fn summaries(ty: &Type, adt_reg: &AdtRegistry) -> BTreeMap<String, Summary> {
    let definitions = reachable_definitions(ty, adt_reg);
    let mut summaries: BTreeMap<String, Summary> = definitions
        .keys()
        .filter(|name| !is_resource_handle(name))
        .map(|name| (name.clone(), Summary::default()))
        .collect();
    loop {
        let mut grew = false;
        for (name, def) in &definitions {
            if is_resource_handle(name) {
                continue;
            }
            let mut next = summaries[name].clone();
            for (_, field) in def.variants.iter().flat_map(|variant| &variant.fields) {
                if next.incomparable.is_some() {
                    break;
                }
                let reached = reach(field, &summaries);
                next.incomparable = reached.incomparable;
                for var in reached.variables {
                    if let Some(position) = def
                        .param_args
                        .iter()
                        .position(|parameter| *parameter == NominalArg::Type(Type::Var(var)))
                    {
                        next.parameters.insert(position);
                    }
                }
            }
            if next != summaries[name] {
                summaries.insert(name.clone(), next);
                grew = true;
            }
        }
        if !grew {
            return summaries;
        }
    }
}

/// Decide [05-OP-36]'s recursive equality domain for one operand type.
///
/// A variable reached through a field is admitted when a dtype family
/// restricts it, since every member is a comparable scalar, and otherwise
/// leaves the verdict waiting on it.
pub(super) fn equality_domain(ty: &Type, subst: &Subst, adt_reg: &AdtRegistry) -> EqualityDomain {
    let ty = subst.apply(ty);
    let reached = reach(&ty, &summaries(&ty, adt_reg));
    if let Some(incomparable) = reached.incomparable {
        return EqualityDomain::Rejected(incomparable);
    }
    if reached
        .variables
        .iter()
        .any(|var| subst.tvar_restriction(*var).is_none())
    {
        return EqualityDomain::Awaits;
    }
    EqualityDomain::Admitted
}

/// Whether a suspended `eq` or `neq` call still waits on a field type: the
/// condition [`equality_domain`] suspended it on, so the call resumes on
/// exactly that condition.
pub(super) fn equality_call_awaits_binding(
    func_name: &str,
    arg_tys: &[Type],
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> bool {
    matches!(func_name, "eq" | "neq")
        && arg_tys
            .iter()
            .any(|ty| matches!(equality_domain(ty, subst, adt_reg), EqualityDomain::Awaits))
}

/// [05-OP-36]'s diagnostic for an operand that reaches a function or a
/// resource handle.
pub(super) fn equality_domain_rejection(
    fname: &str,
    operand: &Type,
    incomparable: &Type,
) -> (CheckErrorKind, String, Vec<String>) {
    let what = if matches!(incomparable, Type::Fn(..)) {
        "function"
    } else {
        "resource handle"
    };
    (
        CheckErrorKind::TypeMismatch,
        format!(
            "`{fname}` does not admit an operand of type `{operand}`: it reaches the {what} \
             type `{incomparable}`, and spec/05-risc-primitives.md [05-OP-36] makes functions \
             and resource handles not equality-comparable"
        ),
        vec![
            "Compare the fields that carry data instead: [05-OP-36] admits unit, scalars, \
             strings, tensors, and `List`, tuple, `Dict`, `Option`, or data-type values whose \
             reachable fields are all equality-comparable."
                .to_string(),
        ],
    )
}
