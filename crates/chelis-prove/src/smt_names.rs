//! Prover-name hygiene: solver symbols (chelis#3236) and injected evaluation
//! roots (chelis#3267).
//!
//! Every symbol the prover itself introduces into an SMT goal -- an
//! abstraction variable, a contract-abstraction symbol, an induction symbol,
//! an obligation input placeholder -- is chosen through a [`NameSupply`]
//! seeded with every name the goal already uses. A generated symbol is
//! therefore never spelled like a user binder, a free name, a quantifier
//! binder, or an earlier generated symbol, so it cannot alias any of them.
//!
//! The second half is the fail-closed guard: a goal that declares one name
//! twice is rejected by every name-keyed solver-variable map (see
//! [`duplicate_variable_reason`]) instead of silently overwriting the first
//! declaration, so a collision that slips past freshness can never become a
//! proof.
//!
//! The same supply names the top-level roots the prover adds to a user's
//! program to evaluate a property sample, a producer value, a generated
//! opaque value, or a constant: [`fresh_root_name`] is the one place every
//! such injection takes its name from, so a root is never spelled like a
//! definition or import the program already has.

use std::collections::BTreeSet;

use chelis_deep::ExprCarrier;
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr};

use crate::solver::{SmtExpr, SmtSort};
use crate::tier_b::SmtProperty;

/// The set of names a goal already uses, and the generator of names it does
/// not.
#[derive(Debug, Clone, Default)]
pub struct NameSupply {
    taken: BTreeSet<String>,
}

impl NameSupply {
    /// An empty supply: no name is taken yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// A supply seeded with every name atom the Deep `program` spells (see
    /// [`Self::reserve_deep`]): each top-level definition, type, constructor,
    /// module, import and export, and every name inside a body.
    pub fn for_deep_program(program: &[DeepExpr]) -> Self {
        let mut supply = Self::new();
        for expr in program {
            supply.reserve_deep(expr);
        }
        supply
    }

    /// A supply seeded with every name `property` uses: each declared
    /// variable, and every free name, quantifier binder, and applied function
    /// symbol in its preconditions and postcondition.
    pub fn for_property(property: &SmtProperty) -> Self {
        let mut supply = Self::new();
        supply.reserve_property(property);
        supply
    }

    /// Mark `name` as taken.
    pub fn reserve(&mut self, name: &str) {
        if !self.taken.contains(name) {
            self.taken.insert(name.to_owned());
        }
    }

    /// Mark every name `property` uses as taken.
    pub fn reserve_property(&mut self, property: &SmtProperty) {
        for (name, _) in &property.variables {
            self.reserve(name);
        }
        for pre in &property.preconditions {
            self.reserve_expr(pre);
        }
        self.reserve_expr(&property.postcondition);
    }

    /// Mark every variable, quantifier binder, and applied function symbol in
    /// `expr` as taken. Over-approximating the used set is sound: it can only
    /// make a generated name longer, never make it alias.
    pub fn reserve_expr(&mut self, expr: &SmtExpr) {
        let mut stack = vec![expr];
        while let Some(node) = stack.pop() {
            match node {
                SmtExpr::Var(name) => self.reserve(name),
                SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {}
                SmtExpr::Arith(_, left, right) | SmtExpr::Cmp(_, left, right) => {
                    stack.push(left);
                    stack.push(right);
                }
                SmtExpr::Bool(_, children) => stack.extend(children.iter()),
                SmtExpr::Not(inner) => stack.push(inner),
                SmtExpr::Forall(bindings, body) | SmtExpr::Exists(bindings, body) => {
                    for (name, _) in bindings {
                        self.reserve(name);
                    }
                    stack.push(body);
                }
                SmtExpr::Apply(name, args) => {
                    self.reserve(name);
                    stack.extend(args.iter());
                }
                SmtExpr::Ite(condition, then_branch, else_branch) => {
                    stack.push(condition);
                    stack.push(then_branch);
                    stack.push(else_branch);
                }
            }
        }
    }

    /// Mark every name atom in the Deep expression `expr` as taken: every
    /// identifier the source spells, whether referenced or bound. This is the
    /// widest avoid set a Deep-lowered goal can draw user names from.
    pub fn reserve_deep(&mut self, expr: &DeepExpr) {
        let mut stack = vec![expr];
        while let Some(node) = stack.pop() {
            match node.carrier() {
                ExprCarrier::Atom(DeepAtom::Name(name)) => self.reserve(name),
                ExprCarrier::Atom(_) | ExprCarrier::MetadataMap(_) => {}
                ExprCarrier::DecodedNode(_, _, children)
                | ExprCarrier::StructuralList(children)
                | ExprCarrier::UndecodableHead(_, _, children) => stack.extend(children.iter()),
                ExprCarrier::MetadataExpression(meta) => stack.push(&meta.expr),
            }
        }
    }

    /// Whether `name` is already taken.
    pub fn is_taken(&self, name: &str) -> bool {
        self.taken.contains(name)
    }

    /// A name not yet taken, which this call then takes. `stem` itself when it
    /// is free, else the first free `stem_1`, `stem_2`, ... The choice is a
    /// function of the taken set alone, so it is deterministic.
    pub fn fresh(&mut self, stem: &str) -> String {
        if !self.taken.contains(stem) {
            self.taken.insert(stem.to_owned());
            return stem.to_owned();
        }
        let mut suffix: u64 = 1;
        loop {
            let candidate = format!("{stem}_{suffix}");
            if !self.taken.contains(&candidate) {
                self.taken.insert(candidate.clone());
                return candidate;
            }
            suffix += 1;
        }
    }
}

/// The name of an evaluation root the prover declares as a new top-level
/// definition of `program` (chelis#3267): `stem` when `program` spells no
/// such name, else the first free `stem_1`, `stem_2`, ...
///
/// The avoid set is every name atom of the whole assembled program: the
/// definitions of the module the root joins, the names its imports spell,
/// and every other module the program carries. That over-approximates the
/// top-level names visible where the root is declared; a wider avoid set can
/// only lengthen the root's name, never let it redefine a user name. A stem
/// that is a linker-format binding name (`pkg__<module>__<terminal>`) stays
/// one, since the suffix extends the terminal.
pub fn fresh_root_name(program: &[DeepExpr], stem: &str) -> String {
    NameSupply::for_deep_program(program).fresh(stem)
}

/// The first name `variables` declares more than once, if any.
pub fn first_duplicate_variable(variables: &[(String, SmtSort)]) -> Option<&str> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    variables
        .iter()
        .map(|(name, _)| name.as_str())
        .find(|name| !seen.insert(name))
}

/// The fail-closed reason every name-keyed solver-variable map reports when a
/// name is declared twice in one scope. Overwriting the first declaration
/// would make two distinct variables one solver constant, so the goal is
/// rejected instead (routes to Tier C, never a proof).
pub fn duplicate_variable_reason(name: &str) -> String {
    format!(
        "solver variable `{name}` is declared more than once in one scope; two distinct \
         variables cannot share one solver symbol (chelis#3236, routes to Tier C)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::CmpOp;

    fn var(name: &str) -> SmtExpr {
        SmtExpr::Var(name.to_string())
    }

    #[test]
    fn fresh_keeps_a_free_stem_and_suffixes_a_taken_one() {
        let mut supply = NameSupply::new();
        supply.reserve("__erf_abs_0");
        supply.reserve("__erf_abs_0_1");
        assert_eq!(supply.fresh("__erf_abs_1"), "__erf_abs_1");
        assert_eq!(supply.fresh("__erf_abs_0"), "__erf_abs_0_2");
        // A name handed out is itself taken for later requests.
        assert_eq!(supply.fresh("__erf_abs_1"), "__erf_abs_1_1");
    }

    #[test]
    fn property_supply_takes_declared_free_bound_and_applied_names() {
        let property = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Forall(
                vec![("bound".to_string(), SmtSort::Real)],
                Box::new(SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(var("bound")),
                    Box::new(var("free")),
                )),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Apply("erf".to_string(), vec![var("x")])),
                Box::new(var("post_free")),
            ),
        };
        let supply = NameSupply::for_property(&property);
        for name in ["x", "bound", "free", "erf", "post_free"] {
            assert!(supply.is_taken(name), "{name} must be taken");
        }
        assert!(!supply.is_taken("unused"));
    }

    #[test]
    fn duplicate_variable_detection_reports_the_repeated_name_only() {
        let distinct = vec![
            ("x".to_string(), SmtSort::Real),
            ("y".to_string(), SmtSort::Real),
        ];
        assert_eq!(first_duplicate_variable(&distinct), None);
        let repeated = vec![
            ("x".to_string(), SmtSort::Real),
            ("y".to_string(), SmtSort::Real),
            ("x".to_string(), SmtSort::Int),
        ];
        assert_eq!(first_duplicate_variable(&repeated), Some("x"));
    }

    fn deep_program(surf: &str) -> Vec<DeepExpr> {
        let decls = chelis_surf::parser::parse_str(surf).expect("parse Surf fixture");
        chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture desugars")
    }

    /// Every stem an injection site uses (chelis#3267).
    const ROOT_STEMS: [&str; 6] = [
        "__chelis_property_probe",
        "__chelis_property_pre",
        "__chelis_value_probe",
        "__chelis_gen_probe",
        "__chelis_const_probe",
        "__chelis_prop_probe",
    ];

    #[test]
    fn root_name_keeps_the_stem_when_the_program_does_not_spell_it() {
        let program = deep_program("module M\ndef f(x: f32) -> f32 = x\n");
        for stem in ROOT_STEMS {
            assert_eq!(fresh_root_name(&program, stem), stem);
        }
    }

    #[test]
    fn root_name_avoids_a_module_definition_of_the_stem() {
        for stem in ROOT_STEMS {
            let program = deep_program(&format!("module M\n{stem} = true\n"));
            let root = fresh_root_name(&program, stem);
            assert_eq!(root, format!("{stem}_1"));
            // A program that also spells the first suffix gets the next one.
            let program = deep_program(&format!(
                "module M\n{stem} = true\ndef {stem}_1() -> f32 = 1.0\n"
            ));
            assert_eq!(fresh_root_name(&program, stem), format!("{stem}_2"));
        }
    }

    #[test]
    fn root_name_avoids_an_imported_name() {
        let program = deep_program(
            "module M\nimport Std.Scalar (__chelis_gen_probe)\ndef f(x: f32) -> f32 = x\n",
        );
        assert_eq!(
            fresh_root_name(&program, "__chelis_gen_probe"),
            "__chelis_gen_probe_1"
        );
    }

    #[test]
    fn linked_root_name_stays_in_its_module() {
        let program = vec![DeepExpr::Atom(
            DeepAtom::Name("pkg__m__M__chelis_prop_probe".to_string()),
            chelis_deep::Span::new(0, 0),
        )];
        let root = fresh_root_name(&program, "pkg__m__M__chelis_prop_probe");
        assert_eq!(root, "pkg__m__M__chelis_prop_probe_1");
        assert_eq!(
            chelis_types::linked_binding_in_module_of("Pkg__m__M__T", "chelis_prop_probe_1"),
            Some(root),
            "the suffixed name is the linker-format binding of the same module"
        );
    }
}
