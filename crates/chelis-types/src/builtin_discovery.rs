//! Target-independent builtin case discovery. No backend capability cells.
use crate::builtins::{
    BUILTINS, BuiltinCapabilityDecl, BuiltinDecl, BuiltinSemanticDomain as Domain,
    BuiltinSiblingCaseId as Case,
};
use crate::types::{Prim, Type};
use std::collections::BTreeSet;

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
        self.capability.validate()?;
        let operand = |index| -> Result<&Type, String> {
            let mut ty = arguments
                .get(index)
                .ok_or("application has no governing operand")?;
            while let Type::Ref(inner) = ty {
                ty = inner;
            }
            Ok(ty)
        };
        let case = match self.name {
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
                _ => return Err("concat requires List or int32 axis".into()),
            }),
            "drop" => Some(match arguments.len() {
                1 => Case::DropValue,
                2 => Case::DropList,
                _ => return Err("drop requires one or two arguments".into()),
            }),
            "eq" | "neq" => match operand(0)? {
                Type::Tensor(..) | Type::Prim(Prim::Bool) => None,
                Type::Prim(prim) if prim.is_numeric() => None,
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
            None if self.capability.domains.contains(&Domain::Numeric) => {
                Ok(format!("Numeric:{}:TableA", self.name))
            }
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
                Ok(format!("{:?}:{}:{case:?}", matches[0].domain, self.name))
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
