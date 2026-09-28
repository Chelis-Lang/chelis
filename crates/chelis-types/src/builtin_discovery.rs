//! Target-independent builtin case discovery. No backend capability cells.
use crate::builtins::{
    BUILTINS, BuiltinCapabilityDecl, BuiltinDecl, BuiltinSemanticDomain as Domain,
    BuiltinSiblingCaseId as Case,
};
use crate::types::{Prim, Type, TypeVarRestriction};
use crate::unify::Subst;
use std::collections::BTreeSet;

/// A concrete identity or a finite selector awaiting its operand's type.
/// The symbolic variant is not an identity and cannot enter the registry.
#[derive(Clone, Debug, PartialEq)]
pub enum BuiltinCaseSelection {
    Resolved(String),
    ByOperand(BuiltinCaseObligation),
}

/// An exact builtin selector applied to existing checker-owned operand types.
/// Private fields prevent callers from constructing an unchecked obligation.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltinCaseObligation {
    builtin: &'static str,
    arguments: Vec<Type>,
}

impl BuiltinCaseSelection {
    /// Apply the owner's ordinary substitution, preserving any still-symbolic
    /// dependency. Each concrete instantiation selects one declared identity.
    pub fn resolve(&self, subst: &Subst) -> Result<Self, String> {
        match self {
            Self::Resolved(_) => Ok(self.clone()),
            Self::ByOperand(obligation) => crate::builtins::builtin_decl(obligation.builtin)
                .ok_or("symbolic selector has no builtin declaration")?
                .semantic_selection(&obligation.arguments, subst),
        }
    }
}

impl BuiltinCapabilityDecl {
    /// Check the finite declaration before it can contribute discovery rows.
    pub fn validate(&self) -> Result<(), String> {
        let domains: BTreeSet<_> = self.domains.iter().copied().collect();
        if domains.is_empty() || domains.len() != self.domains.len() {
            return Err("empty or duplicate semantic domain".into());
        }
        let mut cases = BTreeSet::new();
        for case in self.sibling_cases {
            if case.domain == Domain::Numeric
                || !domains.contains(&case.domain)
                || !cases.insert(case.case)
            {
                return Err("duplicate sibling case or undeclared sibling domain".into());
            }
        }
        for domain in [Domain::Container, Domain::Boundary] {
            if domains.contains(&domain) != self.sibling_cases.iter().any(|c| c.domain == domain) {
                return Err("selected sibling domain has no exact case".into());
            }
        }
        Ok(())
    }
}

impl BuiltinDecl {
    /// Resolve a checked operand shape to its declared semantic case.
    /// Classification does not grant language acceptance: rejected shapes
    /// retain exact identities governed by the same operation atom.
    pub fn semantic_case(&self, arguments: &[Type]) -> Result<String, String> {
        match self.semantic_selection(arguments, &Subst::new())? {
            BuiltinCaseSelection::Resolved(identity) => Ok(identity),
            BuiltinCaseSelection::ByOperand(_) => {
                Err("semantic case requires operand substitution".into())
            }
        }
    }

    /// Select using the checked operand types and their actual dtype bounds.
    pub fn semantic_selection(
        &self,
        arguments: &[Type],
        subst: &Subst,
    ) -> Result<BuiltinCaseSelection, String> {
        self.capability.validate()?;
        let arguments: Vec<_> = arguments.iter().map(|ty| subst.apply(ty)).collect();
        let operand = |index| -> Result<&Type, String> {
            let mut ty = arguments
                .get(index)
                .ok_or("application has no governing operand")?;
            while let Type::Ref(inner) = ty {
                ty = inner;
            }
            Ok(ty)
        };
        let governing = match self.name {
            "to_string" | "len" | "eq" | "neq" => Some(0),
            "concat" => Some(1),
            _ => None,
        };
        let scalar_bound = if let Some(index) = governing {
            match operand(index)? {
                Type::Var(variable) => match subst.tvar_restriction(*variable) {
                    Some(
                        TypeVarRestriction::ActiveFloat
                        | TypeVarRestriction::ActiveInt
                        | TypeVarRestriction::ActiveNumeric,
                    ) => true,
                    None
                    | Some(
                        TypeVarRestriction::FloatValue
                        | TypeVarRestriction::IntValue
                        | TypeVarRestriction::NumericValue,
                    ) => {
                        return Ok(BuiltinCaseSelection::ByOperand(BuiltinCaseObligation {
                            builtin: self.name,
                            arguments,
                        }));
                    }
                },
                Type::Error(_) => return Err("error operand has no semantic case".into()),
                _ => false,
            }
        } else {
            false
        };
        let case = match self.name {
            "to_string" if scalar_bound => Some(Case::ToStringScalar),
            "eq" | "neq" if scalar_bound => None,
            "to_string" => Some(match operand(0)? {
                Type::Prim(_) => Case::ToStringScalar,
                Type::Tensor(..) => Case::ToStringTensor,
                Type::Adt(name, _) if name == "List" => Case::ToStringList,
                Type::Adt(name, _) if name == "Dict" => Case::ToStringDict,
                Type::Adt(name, _) if name == "Option" => Case::ToStringOption,
                Type::Adt(..) | Type::KindedAdt(..) => Case::ToStringAdt,
                Type::Tuple(_) => Case::ToStringTuple,
                Type::Unit => Case::ToStringUnit,
                Type::Fn(..) => Case::ToStringFunction,
                Type::Var(_) | Type::Error(_) => {
                    return Err("observation requires a resolved value kind".into());
                }
                Type::Ref(_) => unreachable!("references were stripped"),
            }),
            "len" => Some(match operand(0)? {
                Type::Adt(name, _) if name == "List" => Case::LenList,
                Type::Adt(name, _) if name == "Dict" => Case::LenDict,
                _ => return Err("len requires List or Dict".into()),
            }),
            "concat" => Some(match operand(1)? {
                Type::Prim(Prim::Int32) => Case::ConcatTensors,
                Type::Adt(name, _) if name == "List" => Case::ConcatList,
                _ => return Err("concat requires List or i32 axis".into()),
            }),
            "eq" | "neq" => match operand(0)? {
                Type::Tensor(..) | Type::Prim(Prim::Bool) => None,
                Type::Prim(prim) if prim.is_numeric() => None,
                Type::Var(_) | Type::Error(_) => {
                    return Err("equality requires a checked value kind".into());
                }
                _ => Some(if self.name == "eq" {
                    Case::EqRecursive
                } else {
                    Case::NeqRecursive
                }),
            },
            _ => {
                if self.capability.domains == [Domain::Numeric] {
                    None
                } else if self.capability.sibling_cases.len() == 1 {
                    Some(self.capability.sibling_cases[0].case)
                } else {
                    return Err(format!("{} has no exact case selector", self.name));
                }
            }
        };
        match case {
            None if self.capability.domains.contains(&Domain::Numeric) => Ok(
                BuiltinCaseSelection::Resolved(format!("Numeric:{}:TableA", self.name)),
            ),
            Some(case) => {
                let matches: Vec<_> = self
                    .capability
                    .sibling_cases
                    .iter()
                    .filter(|c| c.case == case)
                    .collect();
                if matches.len() != 1 {
                    return Err(format!(
                        "{} has missing or overlapping case {case:?}",
                        self.name
                    ));
                }
                Ok(BuiltinCaseSelection::Resolved(format!(
                    "{:?}:{}:{case:?}",
                    matches[0].domain, self.name
                )))
            }
            None => Err(format!("{} selects undeclared Numeric domain", self.name)),
        }
    }
}

/// Independently discover every declaration, rejecting duplicates before union.
pub fn builtin_semantic_identities() -> Result<Vec<String>, String> {
    let mut names = BTreeSet::new();
    let mut rows = BTreeSet::new();
    for builtin in BUILTINS {
        if !names.insert(builtin.name) {
            return Err(format!("duplicate builtin {}", builtin.name));
        }
        builtin
            .capability
            .validate()
            .map_err(|e| format!("{}: {e}", builtin.name))?;
        if builtin.capability.domains.contains(&Domain::Numeric) {
            rows.insert(format!("Numeric:{}:TableA", builtin.name));
        }
        for case in builtin.capability.sibling_cases {
            if !rows.insert(format!(
                "{:?}:{}:{:?}",
                case.domain, builtin.name, case.case
            )) {
                return Err(format!("duplicate builtin case {}", builtin.name));
            }
        }
    }
    Ok(rows.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TypeVar;

    #[test]
    fn operation_value_bounds_do_not_choose_a_scalar_case_before_binding() {
        for bound in [
            TypeVarRestriction::FloatValue,
            TypeVarRestriction::IntValue,
            TypeVarRestriction::NumericValue,
        ] {
            let variable = TypeVar(1234);
            let mut subst = Subst::new();
            subst.narrow_tvar_restriction(variable, bound).unwrap();
            let builtin = crate::builtins::builtin_decl("to_string").unwrap();
            assert!(matches!(
                builtin
                    .semantic_selection(&[Type::Var(variable)], &subst)
                    .unwrap(),
                BuiltinCaseSelection::ByOperand(_)
            ));
            let dtype = if bound == TypeVarRestriction::IntValue {
                crate::types::Prim::Int32
            } else {
                crate::types::Prim::F32
            };
            crate::unify::unify(
                &Type::Var(variable),
                &Type::Tensor(vec![], crate::types::TensorPrec::Concrete(dtype)),
                &mut subst,
            )
            .unwrap();
            assert_eq!(
                builtin
                    .semantic_selection(&[Type::Var(variable)], &subst)
                    .unwrap(),
                BuiltinCaseSelection::Resolved("Boundary:to_string:ToStringTensor".into())
            );
        }
    }

    #[test]
    fn builtin_atom_discovery_scalar_bounds_choose_numeric_not_recursive_cases() {
        for bound in [
            TypeVarRestriction::ActiveFloat,
            TypeVarRestriction::ActiveInt,
            TypeVarRestriction::ActiveNumeric,
        ] {
            let variable = TypeVar(1234);
            let mut subst = Subst::new();
            subst.narrow_tvar_restriction(variable, bound).unwrap();
            for name in ["eq", "neq", "to_string"] {
                let expected = if name == "to_string" {
                    "Boundary:to_string:ToStringScalar".to_string()
                } else {
                    format!("Numeric:{name}:TableA")
                };
                assert_eq!(
                    crate::builtins::builtin_decl(name)
                        .unwrap()
                        .semantic_selection(&[Type::Var(variable)], &subst)
                        .unwrap(),
                    BuiltinCaseSelection::Resolved(expected)
                );
            }
            assert!(
                crate::builtins::builtin_decl("len")
                    .unwrap()
                    .semantic_selection(&[Type::Var(variable)], &subst)
                    .is_err()
            );
            assert!(
                subst
                    .insert_type(
                        variable,
                        Type::Adt("List".into(), vec![Type::Prim(Prim::F32)])
                    )
                    .is_err()
            );
        }
    }
}
