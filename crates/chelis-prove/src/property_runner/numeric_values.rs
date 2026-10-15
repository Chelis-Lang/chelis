//! Canonical literal values and exact domains of typed property inputs.

use crate::solver::{CmpOp, SmtExpr};
use chelis_deep::ast::{Atom, Expr as DeepExpr};
use chelis_deep::{DeepTag, ExprCarrier};
use chelis_surf::ast::{Decl, Expr, Param, TypeExpr, UnaryOp};
use chelis_types::{scalar_from_f64, scalar_from_i64, types::Prim};

/// An immutable semantic view prepared once from the full declaration set.
/// Binding dtypes come from canonical whole-program desugaring, never a second
/// inference pass in the prover. Source declaration indices remain separate.
pub(super) struct SemanticSurfProgram {
    declarations: Vec<Decl>,
}

impl SemanticSurfProgram {
    pub(super) fn new(authored: &[Decl]) -> Result<Self, String> {
        let deep = chelis_surf::desugar::desugar_program(authored)
            .map_err(|error| format!("semantic Surf desugaring failed: {error}"))?;
        let semantic = chelis_surf::resugar::resugar_program_for_analysis(&deep)
            .map_err(|error| format!("semantic Surf reconstruction failed: {error}"))?;
        Ok(Self {
            declarations: super::flatten_module_decls(&semantic),
        })
    }

    pub(super) fn declarations(&self) -> &[Decl] {
        &self.declarations
    }

    pub(super) fn function_body(&self, name: &str) -> Result<&Expr, String> {
        let mut candidates = self.declarations.iter().filter_map(|decl| match decl {
            Decl::FunDef {
                name: candidate,
                body,
                ..
            } if candidate == name => Some(body),
            _ => None,
        });
        let body = candidates
            .next()
            .ok_or_else(|| format!("semantic function `{name}` is absent"))?;
        if candidates.next().is_some() {
            return Err(format!("semantic function `{name}` has ambiguous identity"));
        }
        Ok(body)
    }

    pub(super) fn property(&self, authored: &super::Property) -> Result<super::Property, String> {
        let mut candidates = self.declarations.iter().filter_map(|decl| match decl {
            Decl::Property {
                name,
                params,
                preconditions,
                body,
                ..
            } if name == &authored.name => Some((params, preconditions, body)),
            _ => None,
        });
        let Some((params, preconditions, body)) = candidates.next() else {
            return Err(format!("semantic property `{}` is absent", authored.name));
        };
        if candidates.next().is_some()
            || !params
                .iter()
                .map(|p| &p.name)
                .eq(authored.params.iter().map(|p| &p.name))
        {
            return Err(format!(
                "semantic property `{}` has ambiguous identity",
                authored.name
            ));
        }
        let mut property = authored.clone();
        property.params = params.clone();
        property.preconditions = preconditions.clone();
        property.body = body.clone();
        Ok(property)
    }
}

/// Ask the language's canonical boundary whether this spelling denotes one
/// literal. Converting/binder/named casts retain their Deep node and decline.
/// Normal unary arithmetic also retains its node; the i64 magnitude sentinel
/// is already one signed literal here, so its sign cannot be applied twice.
pub(super) fn surf_literal_value(expr: &Expr) -> Option<SmtExpr> {
    if !matches!(
        expr,
        Expr::Lit(..) | Expr::Unary(UnaryOp::Neg, ..) | Expr::Cast(..)
    ) {
        return None;
    }
    let canonical = chelis_surf::desugar::desugar_expr_only(expr).ok()?;
    deep_literal_value(&canonical)
}

/// One typed ingress shared by both surfaces and both substitution paths.
/// In particular, an integer-written float is finalized from its integer
/// payload directly; no integer is first routed through an f64 image.
pub(super) fn deep_literal_value(expr: &DeepExpr) -> Option<SmtExpr> {
    let ExprCarrier::DecodedNode(DeepTag::Lit, metadata, [DeepExpr::Atom(atom, _)]) =
        expr.carrier()
    else {
        return None;
    };
    let primitive = match metadata.ty() {
        Some(ty) => match ty.expression().carrier() {
            ExprCarrier::DecodedNode(DeepTag::TPrim, _, [DeepExpr::Atom(Atom::Name(name), _)]) => {
                Prim::parse_name(name)?
            }
            _ => return None,
        },
        None => match atom {
            Atom::Int(_) => Prim::Int32,
            Atom::Float(_) => Prim::F32,
            _ => return None,
        },
    };
    let value = match atom {
        Atom::Int(value) if primitive.is_integer() && metadata.literal_source().is_none() => {
            scalar_from_i64("lit", primitive, *value).ok()?
        }
        Atom::Int(value) if primitive.is_float() && metadata.literal_source().is_some() => {
            scalar_from_i64("lit", primitive, *value).ok()?
        }
        Atom::Float(value) if primitive.is_float() && metadata.literal_source().is_none() => {
            scalar_from_f64("lit", primitive, *value).ok()?
        }
        _ => return None,
    };
    if primitive.is_integer() {
        Some(SmtExpr::IntLit(value.as_i64_exact()?))
    } else {
        let value = value.as_f64_lossy();
        value.is_finite().then_some(SmtExpr::RealLit(value))
    }
}

/// These constraints describe quantified inputs, never arbitrary computed
/// arguments or results. Arithmetic itself keeps its existing ideal model.
pub(super) fn integer_input_domains(params: &[Param]) -> Vec<SmtExpr> {
    let mut domains = Vec::new();
    for param in params {
        let Some(TypeExpr::Named(name, _)) = &param.ty else {
            continue;
        };
        let Some((minimum, maximum)) = Prim::parse_name(name).and_then(|prim| prim.integer_range())
        else {
            continue;
        };
        domains.push(SmtExpr::Cmp(
            CmpOp::Ge,
            Box::new(SmtExpr::Var(param.name.clone())),
            Box::new(SmtExpr::IntLit(minimum)),
        ));
        domains.push(SmtExpr::Cmp(
            CmpOp::Le,
            Box::new(SmtExpr::Var(param.name.clone())),
            Box::new(SmtExpr::IntLit(maximum)),
        ));
    }
    domains
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_surf::{ast::Decl, parser::parse_str};

    fn expression(source: &str) -> chelis_surf::ast::Expr {
        let mut declarations = parse_str(&format!("value = {source}\n")).unwrap();
        let Decl::LetDef { value, .. } = declarations.remove(0) else {
            panic!("literal fixture is a value binding");
        };
        value
    }

    #[test]
    fn canonical_cast_literals_keep_exact_integer_values_and_one_sentinel_sign() {
        for (source, expected) in [
            ("cast(-128, i8)", -128),
            ("cast(-32768, i16)", -32768),
            ("cast(-2147483648, i32)", -2147483648),
            ("cast(-9223372036854775808, i64)", i64::MIN),
            ("-9223372036854775808i64", i64::MIN),
            ("cast(-9007199254740993, i64)", -9007199254740993),
            ("cast(-3, i32)", -3),
            ("42", 42),
            ("42i64", 42),
        ] {
            let expr = expression(source);
            assert_eq!(
                surf_literal_value(&expr),
                Some(SmtExpr::IntLit(expected)),
                "{source}"
            );
            let deep = chelis_surf::desugar::desugar_expr_only(&expr).unwrap();
            assert_eq!(
                deep_literal_value(&deep),
                Some(SmtExpr::IntLit(expected)),
                "{source}"
            );
        }
    }

    #[test]
    fn literals_finalize_at_their_float_dtype_before_ideal_lowering() {
        for (source, expected) in [
            ("cast(16777217, f32)", 16777216.0),
            ("16777217f32", 16777216.0),
            ("cast(1.1, f32)", f64::from(1.1f32)),
            ("1.1", f64::from(1.1f32)),
            ("cast(1.1, f64)", 1.1),
            ("1.1f64", 1.1),
            ("cast(2049, f16)", 2048.0),
            ("cast(257, bf16)", 256.0),
        ] {
            let expr = expression(source);
            assert_eq!(
                surf_literal_value(&expr),
                Some(SmtExpr::RealLit(expected)),
                "{source}"
            );
            let deep = chelis_surf::desugar::desugar_expr_only(&expr).unwrap();
            assert_eq!(
                deep_literal_value(&deep),
                Some(SmtExpr::RealLit(expected)),
                "{source}"
            );
        }
    }

    #[test]
    fn literal_recognition_does_not_erase_conversions_or_compute_expressions() {
        for source in [
            "cast(x, i32)",
            "cast(1i64, i32)",
            "cast(1.5, i32)",
            "cast(1.1f32, f64)",
            "cast_trunc(1.5, i32)",
            "cast(1 + 2, i64)",
            "neg(3i64)",
            "cast(2147483648, i32)",
            "cast(-129, i8)",
        ] {
            assert_eq!(surf_literal_value(&expression(source)), None, "{source}");
        }
    }

    #[test]
    fn integer_input_domains_use_exact_representable_endpoints_only() {
        for (dtype, minimum, maximum) in [
            ("i8", i8::MIN as i64, i8::MAX as i64),
            ("i16", i16::MIN as i64, i16::MAX as i64),
            ("i32", i32::MIN as i64, i32::MAX as i64),
            ("i64", i64::MIN, i64::MAX),
        ] {
            let mut declarations =
                parse_str(&format!("@property p forall(x: {dtype}): true\n")).unwrap();
            let Decl::Property { params, .. } = declarations.remove(0) else {
                panic!()
            };
            assert_eq!(
                integer_input_domains(&params),
                vec![
                    SmtExpr::Cmp(
                        crate::solver::CmpOp::Ge,
                        Box::new(SmtExpr::Var("x".into())),
                        Box::new(SmtExpr::IntLit(minimum))
                    ),
                    SmtExpr::Cmp(
                        crate::solver::CmpOp::Le,
                        Box::new(SmtExpr::Var("x".into())),
                        Box::new(SmtExpr::IntLit(maximum))
                    ),
                ]
            );
        }
        let mut declarations = parse_str("@property p forall(x: f32, b: bool): true\n").unwrap();
        let Decl::Property { params, .. } = declarations.remove(0) else {
            panic!()
        };
        assert!(integer_input_domains(&params).is_empty());
    }
}
