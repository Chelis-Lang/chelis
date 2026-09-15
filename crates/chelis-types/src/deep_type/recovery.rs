//! Failed type syntax retains usable structure, but never becomes a resolved type.

use super::*;
use crate::errors::propagate;

/// An invalid signature has a real failure witness and a private body-check frame.
/// Only the frame contains partial structure. Public name lookup retains the error.
#[derive(Debug, Clone)]
pub(crate) struct RejectedSignatureType {
    pub(crate) ty: Type,
    pub(crate) witness: ErrorWitness,
}

impl RejectedSignatureType {
    pub(crate) fn from_resolved(ty: Type, witness: ErrorWitness) -> Self {
        Self { ty, witness }
    }
}

pub(super) enum TypeResolution {
    Resolved(Type),
    Rejected { ty: Type, witness: ErrorWitness },
}

impl TypeResolution {
    fn failed(witness: ErrorWitness) -> Self {
        Self::Rejected {
            ty: propagate(&witness),
            witness,
        }
    }

    fn from_result(result: Result<Type, ErrorWitness>) -> Self {
        match result {
            Ok(ty) => Self::Resolved(ty),
            Err(witness) => Self::failed(witness),
        }
    }

    pub(super) fn into_result(self) -> Result<Type, ErrorWitness> {
        match self {
            Self::Resolved(ty) => Ok(ty),
            Self::Rejected { witness, .. } => Err(witness),
        }
    }

    fn into_parts(self) -> (Type, Option<ErrorWitness>) {
        match self {
            Self::Resolved(ty) => (ty, None),
            Self::Rejected { ty, witness } => (ty, Some(witness)),
        }
    }

    fn compose(parts: Vec<Self>, constructor: impl FnOnce(Vec<Type>) -> Type) -> Self {
        let mut types = Vec::with_capacity(parts.len());
        let mut failure = None;
        for part in parts {
            let (ty, witness) = part.into_parts();
            types.push(ty);
            if failure.is_none() {
                failure = witness;
            }
        }
        let ty = constructor(types);
        match failure {
            Some(witness) => Self::Rejected { ty, witness },
            None => Self::Resolved(ty),
        }
    }
}

impl DeepTypeResolver<'_, '_, '_> {
    pub(crate) fn resolve_signature(
        &mut self,
        expr: &deep::Expr,
    ) -> Result<ResolvedDeepType, RejectedSignatureType> {
        self.begin_resolution(expr);
        let (form, tag, children) = match self.type_form_expr(expr) {
            Ok(parts) => parts,
            Err(witness) => {
                return Err(RejectedSignatureType::from_resolved(
                    propagate(&witness),
                    witness,
                ));
            }
        };
        match self.recover_parts(form, tag, children) {
            TypeResolution::Resolved(ty) => Ok(ResolvedDeepType(ty)),
            TypeResolution::Rejected { ty, witness } => Err(RejectedSignatureType { ty, witness }),
        }
    }

    pub(super) fn recover_type(&mut self, expr: &deep::Expr) -> TypeResolution {
        self.enter_expr(expr);
        match self.type_form_expr(expr) {
            Ok((form, tag, children)) => self.recover_parts(form, tag, children),
            Err(witness) => TypeResolution::failed(witness),
        }
    }

    pub(super) fn recover_parts(
        &mut self,
        form: Option<DeepTag>,
        tag: &str,
        children: &[deep::Expr],
    ) -> TypeResolution {
        match form {
            Some(DeepTag::TFn) => {
                if children.is_empty() {
                    return TypeResolution::failed(self.malformed(format!(
                        "malformed `t-fn` in {}: expected at least a return type",
                        self.use_site.label()
                    )));
                }
                let parts = children
                    .iter()
                    .map(|child| self.recover_type(child))
                    .collect();
                TypeResolution::compose(parts, |mut parts| {
                    let result = parts.pop().expect("a function type has a result child");
                    Type::Fn(parts, Box::new(result))
                })
            }
            Some(DeepTag::TTuple) => {
                let parts = children
                    .iter()
                    .map(|child| self.recover_type(child))
                    .collect();
                TypeResolution::compose(parts, Type::Tuple)
            }
            Some(DeepTag::TRef) => {
                if let Err(witness) = self.exact_arity(tag, children, 1) {
                    return TypeResolution::failed(witness);
                }
                TypeResolution::compose(vec![self.recover_type(&children[0])], |mut parts| {
                    Type::Ref(Box::new(parts.pop().expect("a reference has one child")))
                })
            }
            Some(DeepTag::TTensor) => self.recover_tensor(children),
            Some(DeepTag::TAdt) => self.recover_nominal(children),
            _ => TypeResolution::from_result(self.resolve_atomic_type(form, tag, children)),
        }
    }

    fn recover_tensor(&mut self, children: &[deep::Expr]) -> TypeResolution {
        if children.is_empty() {
            return TypeResolution::failed(self.malformed(format!(
                "malformed `t-tensor` in {}: expected dimensions followed by a precision",
                self.use_site.label()
            )));
        }
        let dimensions: Vec<_> = children[..children.len() - 1]
            .iter()
            .map(|child| self.resolve_dim(child))
            .collect();
        let outer = std::mem::replace(&mut self.resolving_tensor_precision, true);
        let precision = self.recover_type(&children[children.len() - 1]);
        self.resolving_tensor_precision = outer;
        let precision = match precision {
            TypeResolution::Resolved(Type::Prim(prim)) => Ok(TensorPrec::Concrete(prim)),
            TypeResolution::Resolved(Type::Var(var)) => Ok(TensorPrec::Var(var)),
            TypeResolution::Resolved(other) => Err(self.type_error(format!(
                "tensor precision in {} must be `t-prim` or a legal `t-var`, got `{other}`",
                self.use_site.label()
            ))),
            TypeResolution::Rejected { witness, .. } => Err(witness),
        };
        // Dimensions have no error constructor. Never replace a failed one with
        // a wildcard or a fresh variable just to construct a tensor frame.
        let dimensions = dimensions.into_iter().collect::<Result<Vec<_>, _>>();
        match (dimensions, precision) {
            (Ok(dimensions), Ok(precision)) => {
                TypeResolution::Resolved(Type::Tensor(dimensions, precision))
            }
            (Err(witness), _) | (_, Err(witness)) => TypeResolution::failed(witness),
        }
    }

    fn recover_nominal(&mut self, children: &[deep::Expr]) -> TypeResolution {
        let Some(name) = children.first().and_then(symbol_name) else {
            return TypeResolution::failed(self.malformed(format!(
                "malformed `t-adt` in {}: expected a nominal type name followed by type arguments",
                self.use_site.label()
            )));
        };
        let Some(kinds) = self.headers.param_kinds(name) else {
            return TypeResolution::failed(self.type_error(format!(
                "unknown nominal type `{name}` in {}",
                self.use_site.label()
            )));
        };
        let actual = children.len() - 1;
        if actual != kinds.len() {
            return TypeResolution::failed(self.type_error(format!(
                "nominal type `{name}` in {} expects {} argument(s), got {actual}",
                self.use_site.label(),
                kinds.len()
            )));
        }
        let mut arguments = Vec::with_capacity(actual);
        let mut failure = None;
        let mut incomplete = false;
        for (index, (child, kind)) in children[1..].iter().zip(kinds).enumerate() {
            self.enter_expr(child);
            let argument = (|| -> Result<_, ErrorWitness> {
                let (child_tag, _, parts) = self.type_form_expr(child)?;
                match kind {
                    NominalParamKind::Type => {
                        if matches!(
                            child_tag,
                            Some(DeepTag::DName | DeepTag::DVar | DeepTag::DLit | DeepTag::DRank)
                        ) {
                            return Err(self.type_error(format!(
                                "nominal type `{name}` argument {} expects a type, got a dimension",
                                index + 1
                            )));
                        }
                        let (ty, witness) = self.recover_type(child).into_parts();
                        if failure.is_none() {
                            failure = witness;
                        }
                        Ok(NominalArg::Type(ty))
                    }
                    NominalParamKind::Dimension => {
                        let dim = match child_tag {
                            Some(DeepTag::DName | DeepTag::DVar | DeepTag::DLit) => {
                                self.resolve_dim(child)?
                            }
                            Some(DeepTag::TVar) => {
                                let name = self.one_symbol("t-var", parts)?;
                                Dim::Var(self.resolve_dim_var(name)?)
                            }
                            _ => return Err(self.type_error(format!(
                                "nominal type `{name}` argument {} expects a dimension, got a type",
                                index + 1
                            ))),
                        };
                        Ok(NominalArg::Dimension(dim))
                    }
                }
            })();
            match argument {
                Ok(argument) => arguments.push(argument),
                Err(witness) => {
                    incomplete = true;
                    if failure.is_none() {
                        failure = Some(witness);
                    }
                }
            }
        }
        if incomplete {
            return TypeResolution::failed(
                failure.expect("an incomplete argument reported a failure"),
            );
        }
        let ty = if kinds.contains(&NominalParamKind::Dimension) {
            Type::KindedAdt(name.to_string(), arguments)
        } else {
            Type::Adt(
                name.to_string(),
                arguments
                    .into_iter()
                    .map(|argument| match argument {
                        NominalArg::Type(ty) => ty,
                        NominalArg::Dimension(_) => {
                            unreachable!("type-only header produced a dimension argument")
                        }
                    })
                    .collect(),
            )
        };
        match failure {
            Some(witness) => TypeResolution::Rejected { ty, witness },
            None => TypeResolution::Resolved(ty),
        }
    }
}
