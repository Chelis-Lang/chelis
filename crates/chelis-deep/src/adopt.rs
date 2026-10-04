//! Contextual literal adoption, decided once, on Deep (chelis#3114).
//!
//! `spec/04-type-system.md` §5.3 binds an unsuffixed numeric literal at its
//! default dtype, `i32` or `f32`. §5.6 (`spec/02-surf-syntax.md` §P10b) names
//! the closed set of positions where it takes another:
//!
//! 1. an element of a tensor literal whose own binding declares a tensor type;
//! 2. an element of a tensor literal passed where the top-level callee
//!    declares a tensor parameter;
//! 3. an element of a tensor literal that is the body of a function whose
//!    declared result is a tensor type;
//! 4. the operand of a checked cast, or an element of a tensor literal that is
//!    one, which takes the target.
//!
//! A pipe denotes its nested call, so a literal a pipe feeds to a stage stands
//! exactly where [`crate::pipe::fold_pipes`] puts it.
//!
//! Whether a position adopts depends on bindings: a lexical binding of the
//! callee's name is another callable, and a lexical binding of `to_tensor` or
//! `neg` is an ordinary function. Macro expansion moves a macro's body and its
//! arguments into the scope of each use site (`spec/02-surf-syntax.md` §P5b).
//! The rule is therefore decided here, on Deep, where every binder is visible,
//! and it is decided again on the expanded program. The desugarer marks every
//! unsuffixed literal `surf_literal_style: "unsuffixed"` at its default dtype
//! and runs [`adopt_program`] provisionally; macro expansion runs it on the
//! expanded program, which is final. A literal without the marker was written
//! with a suffix, or as Deep, and is never retyped.
//!
//! The resugarer prints from [`analyze`], the same walk: a literal prints
//! without its suffix exactly when an unsuffixed literal there derives its
//! dtype, so the printed spelling and the desugared dtype cannot disagree.

use std::collections::{BTreeMap, BTreeSet};

use crate::annotations::{
    BindingTypeOrigin, LiteralOrigin, LiteralStyle, MetadataKey, MetadataValue, Spanned,
    TypeSyntax,
};
use crate::literal_source::atom_bound_fit;
use crate::node::Node;
use crate::{
    Atom, CastMode, DeepTag, DtypeBound, Expr, LiteralFamilyFit, Metadata, Span, cast_mode_of,
    decode_dtype_bounds,
};

/// When [`adopt_program`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// On desugared Deep before macro expansion, which can still move a
    /// literal. Every unsuffixed literal keeps its marker.
    Provisional,
    /// On the expanded program. A literal no position adopted keeps its
    /// default dtype and loses its marker; an adopted literal keeps the
    /// marker, which the checker's dtype-binder rules read.
    Final,
}

/// Decide the dtype of every unsuffixed literal in a program.
pub fn adopt_program(exprs: &mut [Expr], stage: Stage) {
    let changes = {
        let globals = Globals::collect(exprs);
        let mut walk = Walk::new(&globals, Some(stage));
        walk.sequence(exprs, &mut Scope::default());
        walk.into_changes()
    };
    for expr in exprs.iter_mut() {
        changes.apply(expr);
    }
}

/// Decide the dtype of every unsuffixed literal in an expression standing in
/// `program`'s scope, under the local names `locals`.
pub fn adopt_expression(expr: &mut Expr, program: &[Expr], locals: &[String], stage: Stage) {
    let changes = {
        let globals = Globals::collect(program);
        let mut walk = Walk::new(&globals, Some(stage));
        let mut scope = Scope::default();
        scope.locals.extend(locals.iter().cloned());
        walk.expression(expr, Slot::PLAIN, &mut scope);
        walk.into_changes()
    };
    changes.apply(expr);
}

/// How every literal of a program prints, and which values a declaration
/// makes tensor literals. It borrows the program it analyzed.
pub struct Analysis<'p> {
    literals: BTreeMap<usize, Site<'p>>,
    declared: BTreeSet<usize>,
}

/// Analyze a program for printing.
pub fn analyze(exprs: &[Expr]) -> Analysis<'_> {
    let globals = Globals::collect(exprs);
    let mut walk = Walk::new(&globals, None);
    walk.sequence(exprs, &mut Scope::default());
    walk.into_analysis()
}

/// Analyze one expression, with no enclosing program, for printing.
pub fn analyze_expression(expr: &Expr) -> Analysis<'_> {
    let globals = Globals::default();
    let mut walk = Walk::new(&globals, None);
    walk.expression(expr, Slot::PLAIN, &mut Scope::default());
    walk.into_analysis()
}

/// How a numeric literal prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiteralSpelling {
    /// Without a suffix: an unsuffixed literal there derives this literal's
    /// dtype. `refolds` says that a negative value printed as a negated
    /// literal folds back into one literal there, so the minimum of a signed
    /// width may print as `-2147483648`.
    Unsuffixed { refolds: bool },
    /// An integer value at a float dtype, printed as a decimal without a
    /// suffix: an unsuffixed decimal there derives the literal's dtype.
    UnsuffixedDecimal,
    /// With every suffix, including a default one.
    Suffixed,
}

impl Analysis<'_> {
    /// How the literal node `literal` prints.
    ///
    /// A literal the analysis did not reach, or one written with a default
    /// suffix (`surf_literal_style: "explicit"`), keeps its suffix, which is
    /// faithful wherever it prints.
    pub fn literal_spelling(&self, literal: &Node) -> LiteralSpelling {
        let Some(site) = self.literals.get(&address(literal)) else {
            return LiteralSpelling::Suffixed;
        };
        let meta = literal.meta();
        if meta
            .surf_literal_style()
            .is_some_and(|style| *style.value() == LiteralStyle::Explicit)
        {
            return LiteralSpelling::Suffixed;
        }
        let Some(ty) = meta.ty().and_then(|ty| Dtype::of_type(ty.expression())) else {
            return LiteralSpelling::Suffixed;
        };
        let Some(atom) = numeric_atom(literal) else {
            return LiteralSpelling::Suffixed;
        };
        if site.negated {
            // `-x` folds into one literal where the position folds, so only an
            // unfolded negation reproduces this node, and only at the default.
            return if !site.slot.folds(atom) && Dtype::default_of(atom) == ty {
                LiteralSpelling::Unsuffixed { refolds: false }
            } else {
                LiteralSpelling::Suffixed
            };
        }
        if site.slot.rederives(atom) == ty {
            return LiteralSpelling::Unsuffixed {
                refolds: is_negative(atom) && site.slot.folds(atom),
            };
        }
        if let Atom::Int(value) = atom
            && ty.is_float()
            && site.slot.rederives(&Atom::Float(*value as f64)) == ty
        {
            return LiteralSpelling::UnsuffixedDecimal;
        }
        LiteralSpelling::Suffixed
    }

    /// Whether `value`, a binding's value or a function's body, stands where
    /// its own declaration states a tensor type (positions 1 and 3). There a
    /// bracket literal converts to a tensor without `to_tensor`.
    pub fn declares_tensor(&self, value: &Node) -> bool {
        self.declared.contains(&address(value))
    }
}

/// The type a position gives an element or a cast operand, as adoption reads
/// it.
#[derive(Clone, Debug, PartialEq)]
enum Target<'p> {
    /// A primitive, numeric or not.
    Prim(&'p str),
    /// A dtype binder of the enclosing declaration, with its bound if any.
    Binder(&'p str, Option<DtypeBound>),
    /// Any other type.
    Other,
}

/// The adopting position an expression stands in.
#[derive(Clone, Debug, PartialEq)]
enum Context<'p> {
    /// No adopting position.
    Plain,
    /// A value whose own binding or function result declares a tensor type
    /// (positions 1 and 3), with its element type.
    Declared(Target<'p>),
    /// An argument at a declared tensor parameter of the top-level callee
    /// (position 2), with its element type.
    Parameter(Target<'p>),
    /// The operand of a checked cast (position 4), with its target.
    Cast(Target<'p>),
}

/// Where in a context an expression stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// In the context itself.
    Value,
    /// As the argument of an intrinsic `to_tensor` call that stands in the
    /// context, where a finite `Cons`/`Nil` chain is a tensor literal.
    TensorSource,
    /// As an element of that tensor literal, at any nesting depth.
    Element,
    /// As the rest of the tensor literal's element chain, already known to be
    /// finite.
    Chain,
}

#[derive(Clone, Debug, PartialEq)]
struct Slot<'p> {
    context: Context<'p>,
    place: Place,
}

impl<'p> Slot<'p> {
    const PLAIN: Slot<'static> = Slot {
        context: Context::Plain,
        place: Place::Value,
    };

    fn new(context: Context<'p>, place: Place) -> Self {
        Self { context, place }
    }

    /// What an unsuffixed literal with value `atom` takes here, and whether
    /// that is an adoption rather than the §5.3 default.
    fn decide(&self, atom: &Atom) -> (Dtype<'p>, bool) {
        let adopted = match (self.place, &self.context) {
            (Place::Value, Context::Cast(Target::Prim(prim))) => {
                scalar_cast_adopts(atom, prim).then_some(Dtype::Prim(prim))
            }
            (Place::Value | Place::Element, Context::Cast(Target::Binder(binder, bound))) => bound
                .as_ref()
                .is_some_and(|bound| {
                    matches!(
                        atom_bound_fit(Some(atom), bound),
                        LiteralFamilyFit::Fits | LiteralFamilyFit::IntegerOutOfRange
                    )
                })
                .then_some(Dtype::Binder(binder)),
            (
                Place::Element,
                Context::Declared(Target::Prim(prim))
                | Context::Parameter(Target::Prim(prim))
                | Context::Cast(Target::Prim(prim)),
            ) => element_adopts(atom, prim).then_some(Dtype::Prim(prim)),
            _ => None,
        };
        match adopted {
            Some(dtype) => (dtype, true),
            None => (Dtype::default_of(atom), false),
        }
    }

    /// Whether an unsuffixed negated literal with value `-atom` folds into one
    /// signed literal here before it adopts. A cast operand always folds; a
    /// tensor element folds where it adopts.
    fn folds(&self, atom: &Atom) -> bool {
        match (self.place, &self.context) {
            (Place::Value, Context::Cast(_)) => true,
            (Place::Element, Context::Cast(Target::Binder(..))) => true,
            (
                Place::Element,
                Context::Declared(Target::Prim(prim))
                | Context::Parameter(Target::Prim(prim))
                | Context::Cast(Target::Prim(prim)),
            ) => element_adopts(atom, prim),
            _ => false,
        }
    }

    /// The dtype a literal with value `atom`, printed without a suffix,
    /// takes here when the program is desugared again. A negative value
    /// prints as a negated literal, which keeps the default where it does not
    /// fold.
    fn rederives(&self, atom: &Atom) -> Dtype<'p> {
        if is_negative(atom) && !self.folds(atom) {
            Dtype::default_of(atom)
        } else {
            self.decide(atom).0
        }
    }
}

/// `spec/04-type-system.md` §5.6 position 4 for a scalar: a decimal adopts a
/// float target, and an integer adopts a float or integer target.
fn scalar_cast_adopts(atom: &Atom, prim: &str) -> bool {
    match atom {
        Atom::Float(_) => is_float_prim(prim),
        Atom::Int(_) => is_float_prim(prim) || is_integer_prim(prim),
        _ => false,
    }
}

/// A tensor literal's element adopts the element type exactly as a scalar
/// cast operand adopts a primitive target. A literal that cannot bind there
/// keeps its §5.3 default and the checker reports any mismatch.
fn element_adopts(atom: &Atom, prim: &str) -> bool {
    scalar_cast_adopts(atom, prim)
}

fn is_float_prim(prim: &str) -> bool {
    matches!(prim, "f16" | "bf16" | "f32" | "f64")
}

fn is_integer_prim(prim: &str) -> bool {
    matches!(prim, "i8" | "i16" | "i32" | "i64")
}

/// A literal's dtype as adoption decides it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dtype<'a> {
    Prim(&'a str),
    Binder(&'a str),
}

impl<'a> Dtype<'a> {
    /// The §5.3 default for a numeric atom.
    fn default_of(atom: &Atom) -> Self {
        match atom {
            Atom::Int(_) => Dtype::Prim("i32"),
            _ => Dtype::Prim("f32"),
        }
    }

    fn of_type(ty: &'a Expr) -> Option<Self> {
        let (tag, _, children) = parts(ty)?;
        let [Expr::Atom(Atom::Name(name), _)] = children else {
            return None;
        };
        match tag {
            DeepTag::TPrim => Some(Dtype::Prim(name)),
            DeepTag::TVar => Some(Dtype::Binder(name)),
            _ => None,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Dtype::Prim(prim) if is_float_prim(prim))
    }

    fn type_expr(self) -> Expr {
        let (tag, name) = match self {
            Dtype::Prim(name) => (DeepTag::TPrim, name),
            Dtype::Binder(name) => (DeepTag::TVar, name),
        };
        Expr::node(
            tag,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name(name.to_string()), Span::new(0, 0))],
            Span::new(0, 0),
        )
    }
}

/// A literal's position, recorded for printing.
#[derive(Clone, Debug)]
struct Site<'p> {
    slot: Slot<'p>,
    /// The literal is the operand of a `neg` application.
    negated: bool,
}

/// What the program declares at top level, through nested modules.
#[derive(Default)]
struct Globals<'p> {
    /// Each declared type by name. A later declaration replaces an earlier one.
    signatures: BTreeMap<&'p str, &'p Expr>,
    /// Each declaration's dtype-binder bounds by name.
    bounds: BTreeMap<&'p str, Vec<(String, DtypeBound)>>,
    /// Every name a top-level `def` or `defsig` declares.
    names: BTreeSet<&'p str>,
}

impl<'p> Globals<'p> {
    fn collect(exprs: &'p [Expr]) -> Self {
        let mut globals = Self::default();
        globals.collect_into(exprs);
        globals
    }

    fn collect_into(&mut self, exprs: &'p [Expr]) {
        for expr in exprs {
            let Some((tag, meta, children)) = parts(expr) else {
                continue;
            };
            match tag {
                DeepTag::Module => self.collect_into(children.get(1..).unwrap_or_default()),
                DeepTag::Defsig => {
                    let (Some(name), Some(ty)) = (children.first().and_then(name), children.last())
                    else {
                        continue;
                    };
                    self.names.insert(name);
                    self.signatures.insert(name, ty);
                    self.bounds.insert(name, decode_dtype_bounds(meta));
                }
                DeepTag::Def => {
                    if let Some(name) = children.first().and_then(name) {
                        self.names.insert(name);
                    }
                }
                _ => {}
            }
        }
    }

    /// The element type of a declared tensor type.
    fn element(&self, ty: &'p Expr, binders: Option<&'p str>) -> Option<Target<'p>> {
        let (DeepTag::TTensor, _, children) = parts(ty)? else {
            return None;
        };
        Some(self.target(children.last()?, binders))
    }

    /// A type as an element type or a cast target, where a `t-var` names a
    /// dtype binder of the declaration `binders`.
    fn target(&self, ty: &'p Expr, binders: Option<&'p str>) -> Target<'p> {
        match Dtype::of_type(ty) {
            Some(Dtype::Prim(prim)) => Target::Prim(prim),
            Some(Dtype::Binder(binder)) => Target::Binder(
                binder,
                binders
                    .and_then(|owner| self.bounds.get(owner))
                    .and_then(|bounds| bounds.iter().find(|(name, _)| name == binder))
                    .map(|(_, bound)| bound.clone()),
            ),
            None => Target::Other,
        }
    }
}

/// The local names in scope, innermost last.
#[derive(Default)]
struct Scope {
    locals: Vec<String>,
}

impl Scope {
    fn binds(&self, name: &str) -> bool {
        self.locals.iter().any(|local| local == name)
    }
}

/// A change to one node of the analyzed program.
enum Change {
    /// Give a literal this dtype and marker.
    Retype { dtype: OwnedDtype, marked: bool },
    /// Replace a `neg` application of a literal by one signed literal.
    Fold { dtype: OwnedDtype, marked: bool, value: Atom },
    /// Replace a negative literal by a `neg` application of its magnitude.
    Unfold { marked: bool },
}

#[derive(Clone, Debug)]
enum OwnedDtype {
    Prim(String),
    Binder(String),
}

impl From<Dtype<'_>> for OwnedDtype {
    fn from(dtype: Dtype<'_>) -> Self {
        match dtype {
            Dtype::Prim(name) => OwnedDtype::Prim(name.to_string()),
            Dtype::Binder(name) => OwnedDtype::Binder(name.to_string()),
        }
    }
}

impl OwnedDtype {
    fn borrowed(&self) -> Dtype<'_> {
        match self {
            OwnedDtype::Prim(name) => Dtype::Prim(name),
            OwnedDtype::Binder(name) => Dtype::Binder(name),
        }
    }
}

/// The changes one walk decided, keyed by node address.
#[derive(Default)]
struct Changes {
    changes: BTreeMap<usize, Change>,
    /// Nodes with a changed node among their descendants.
    dirty: BTreeSet<usize>,
    /// Nodes with a changed node inside a metadata expression.
    dirty_meta: BTreeSet<usize>,
}

struct Walk<'g, 'p> {
    globals: &'g Globals<'p>,
    stage: Option<Stage>,
    analysis: Analysis<'p>,
    changes: Changes,
    /// Addresses of the nodes being visited, outermost first.
    ancestors: Vec<usize>,
}

impl<'g, 'p> Walk<'g, 'p> {
    fn new(globals: &'g Globals<'p>, stage: Option<Stage>) -> Self {
        Self {
            globals,
            stage,
            analysis: Analysis {
                literals: BTreeMap::new(),
                declared: BTreeSet::new(),
            },
            changes: Changes::default(),
            ancestors: Vec::new(),
        }
    }

    fn into_changes(self) -> Changes {
        self.changes
    }

    fn into_analysis(self) -> Analysis<'p> {
        self.analysis
    }

    /// A top-level declaration sequence.
    fn sequence(&mut self, exprs: &'p [Expr], scope: &mut Scope) {
        for expr in exprs {
            self.expression(expr, Slot::PLAIN, scope);
        }
    }

    fn expression(&mut self, expr: &'p Expr, slot: Slot<'p>, scope: &mut Scope) {
        self.expression_in(expr, slot, scope, None);
    }

    /// Visit `expr` standing in `slot`, inside the declaration `binders`.
    fn expression_in(
        &mut self,
        expr: &'p Expr,
        slot: Slot<'p>,
        scope: &mut Scope,
        binders: Option<&'p str>,
    ) {
        stacker::maybe_grow(
            crate::nesting::STACK_RED_ZONE_BYTES,
            4 * 1024 * 1024,
            || match expr {
                Expr::Node(node, _) => self.node(node, slot, scope, binders),
                Expr::MetaExpr(wrapper, _) => {
                    self.metadata(&wrapper.metadata, None, scope, binders);
                    self.expression_in(&wrapper.expr, Slot::PLAIN, scope, binders);
                }
                Expr::BareList(items, _) => {
                    // A macro definition decides nothing until it expands.
                    if !is_macro_definition(expr) {
                        for item in items {
                            self.expression_in(item, Slot::PLAIN, scope, binders);
                        }
                    }
                }
                Expr::UnknownForm(data) => {
                    if data.head != "defmacro" {
                        for child in &data.children {
                            self.expression_in(child, Slot::PLAIN, scope, binders);
                        }
                    }
                }
                Expr::Atom(..) | Expr::Map(..) => {}
            },
        );
    }

    fn node(
        &mut self,
        node: &'p Node,
        slot: Slot<'p>,
        scope: &mut Scope,
        binders: Option<&'p str>,
    ) {
        let tag = node.tag();
        let children = node.children_slice();
        if tag == DeepTag::Lit {
            self.literal(node, slot, false);
            return;
        }
        if let Some(operand) = self.negated_literal(node, scope) {
            self.negation(node, operand, slot);
            return;
        }
        if slot.context != Context::Plain && slot.place == Place::Value {
            if let Context::Declared(_) = slot.context {
                self.analysis.declared.insert(address(node));
            }
        }
        self.ancestors.push(address(node));
        match tag {
            DeepTag::Module => {
                self.metadata(node.meta(), Some(node), scope, binders);
                for child in children.get(1..).unwrap_or_default() {
                    self.expression_in(child, Slot::PLAIN, scope, None);
                }
            }
            DeepTag::Def => self.definition(node, scope),
            DeepTag::Fn => {
                self.metadata(node.meta(), Some(node), scope, binders);
                let depth = scope.locals.len();
                if let Some(params) = children.first() {
                    bind_parameters(params, scope);
                }
                for child in children.get(1..).unwrap_or_default() {
                    self.expression_in(child, Slot::PLAIN, scope, binders);
                }
                scope.locals.truncate(depth);
            }
            DeepTag::Let => {
                self.metadata(node.meta(), Some(node), scope, binders);
                let depth = scope.locals.len();
                if let Some(Expr::Node(bindings, _)) = children.first()
                    && bindings.tag() == DeepTag::Bind
                {
                    self.ancestors.push(address(bindings));
                    self.metadata(bindings.meta(), Some(bindings), scope, binders);
                    for pair in bindings.children_slice().chunks(2) {
                        let [name, value] = pair else {
                            continue;
                        };
                        let slot = declared_binding_slot(self.globals, value, binders);
                        self.expression_in(value, slot, scope, binders);
                        if let Some(name) = self::name(name) {
                            scope.locals.push(name.to_string());
                        }
                    }
                    self.ancestors.pop();
                }
                for child in children.get(1..).unwrap_or_default() {
                    self.expression_in(child, Slot::PLAIN, scope, binders);
                }
                scope.locals.truncate(depth);
            }
            DeepTag::Match => {
                self.metadata(node.meta(), Some(node), scope, binders);
                if let Some(scrutinee) = children.first() {
                    self.expression_in(scrutinee, Slot::PLAIN, scope, binders);
                }
                for arm in children.get(1..).unwrap_or_default() {
                    let depth = scope.locals.len();
                    if let Some((DeepTag::Arm, _, [pattern, ..])) = parts(arm) {
                        scope.locals.extend(crate::pattern_binder_names(pattern));
                    }
                    self.expression_in(arm, Slot::PLAIN, scope, binders);
                    scope.locals.truncate(depth);
                }
            }
            DeepTag::Pipe => {
                self.metadata(node.meta(), Some(node), scope, binders);
                if let Some((head, stages)) = children.split_first() {
                    let head_slot = self.pipe_head_slot(stages, slot, scope, binders);
                    self.expression_in(head, head_slot, scope, binders);
                    for stage in stages {
                        self.expression_in(stage, Slot::PLAIN, scope, binders);
                    }
                }
            }
            DeepTag::App if self.is_chain(node, &slot) => {
                self.chain(node, slot.context.clone(), scope, binders);
            }
            _ => {
                self.metadata(node.meta(), Some(node), scope, binders);
                for (index, child) in children.iter().enumerate() {
                    let child_slot =
                        self.child_slot(tag, node.meta(), children, index, &slot, scope, binders);
                    self.expression_in(child, child_slot, scope, binders);
                }
            }
        }
        self.ancestors.pop();
    }

    /// A top-level `def`: positions 1 and 3 read its declared type.
    fn definition(&mut self, node: &'p Node, scope: &mut Scope) {
        let children = node.children_slice();
        let Some(owner) = children.first().and_then(name) else {
            return;
        };
        let signature = self.globals.signatures.get(owner).copied();
        let depth = scope.locals.len();
        if let Some(value) = children.get(1) {
            match parts(value) {
                Some((DeepTag::Fn, fn_meta, [params, body])) => {
                    bind_parameters(params, scope);
                    // A property's preconditions name its parameters.
                    self.metadata(node.meta(), Some(node), scope, Some(owner));
                    let Expr::Node(function, _) = value else {
                        unreachable!("parts reads a node")
                    };
                    self.ancestors.push(address(function));
                    self.metadata(fn_meta, Some(function), scope, Some(owner));
                    let result = signature
                        .and_then(|ty| match parts(ty) {
                            Some((DeepTag::TFn, _, [.., result])) => Some(result),
                            _ => None,
                        })
                        .and_then(|result| self.globals.element(result, Some(owner)));
                    let slot = result.map_or(Slot::PLAIN, |element| {
                        Slot::new(Context::Declared(element), Place::Value)
                    });
                    self.expression_in(body, slot, scope, Some(owner));
                    self.ancestors.pop();
                }
                _ => {
                    self.metadata(node.meta(), Some(node), scope, Some(owner));
                    let slot = signature
                        .and_then(|ty| self.globals.element(ty, Some(owner)))
                        .map_or(Slot::PLAIN, |element| {
                            Slot::new(Context::Declared(element), Place::Value)
                        });
                    self.expression_in(value, slot, scope, Some(owner));
                }
            }
        }
        scope.locals.truncate(depth);
    }

    /// The slot of child `index` of a node with tag `tag`, metadata `meta`
    /// and children `children`, the node standing in `parent`. The child need
    /// not be read, which lets a pipe ask where the fold puts its accumulator.
    #[allow(clippy::too_many_arguments)]
    fn child_slot(
        &self,
        tag: DeepTag,
        meta: &'p Metadata,
        children: &'p [Expr],
        index: usize,
        parent: &Slot<'p>,
        scope: &Scope,
        binders: Option<&'p str>,
    ) -> Slot<'p> {
        match tag {
            DeepTag::App if index > 0 => {
                let Some(callee) = children.first().and_then(untyped_variable) else {
                    return Slot::PLAIN;
                };
                if callee == "to_tensor" && children.len() == 2 && self.intrinsic(callee, scope) {
                    let context = match parent.place {
                        Place::Value => parent.context.clone(),
                        _ => Context::Plain,
                    };
                    return Slot::new(context, Place::TensorSource);
                }
                if callee == "Cons"
                    && children.len() == 3
                    && parent.place != Place::Value
                    && (parent.place == Place::Chain || finite_chain(meta, children))
                {
                    return match index {
                        1 => Slot::new(parent.context.clone(), Place::Element),
                        _ => Slot::new(parent.context.clone(), Place::Chain),
                    };
                }
                if scope.binds(callee) {
                    return Slot::PLAIN;
                }
                let Some((DeepTag::TFn, _, signature)) =
                    self.globals.signatures.get(callee).copied().and_then(parts)
                else {
                    return Slot::PLAIN;
                };
                let parameters = signature.split_last().map_or(&[][..], |(_, rest)| rest);
                parameters
                    .get(index - 1)
                    .and_then(|parameter| self.globals.element(parameter, Some(callee)))
                    .map_or(Slot::PLAIN, |element| {
                        Slot::new(Context::Parameter(element), Place::Value)
                    })
            }
            DeepTag::Cast if index == 0 && cast_mode_of(children) == Ok(CastMode::Checked) => {
                children.get(1).map_or(Slot::PLAIN, |target| {
                    Slot::new(
                        Context::Cast(self.globals.target(target, binders)),
                        Place::Value,
                    )
                })
            }
            _ => Slot::PLAIN,
        }
    }

    /// Where a pipe's head stands: each stage receives the value before it as
    /// the fold applies it, from the last stage, which stands in the pipe's
    /// own slot, back to the head.
    fn pipe_head_slot(
        &self,
        stages: &'p [Expr],
        result: Slot<'p>,
        scope: &Scope,
        binders: Option<&'p str>,
    ) -> Slot<'p> {
        let mut slot = result;
        for stage in stages.iter().rev() {
            slot = match crate::pipe::forwarding_stage(stage) {
                Some((body, input)) => {
                    let Some((tag, meta, children)) = parts(body) else {
                        return Slot::PLAIN;
                    };
                    self.child_slot(tag, meta, children, input, &slot, scope, binders)
                }
                // The fold applies any other stage to the value as a callee.
                None => {
                    let synthetic = std::slice::from_ref(stage);
                    match untyped_variable(stage) {
                        Some(_) => self.applied_slot(synthetic, &slot, scope),
                        None => Slot::PLAIN,
                    }
                }
            };
        }
        slot
    }

    /// The slot of the argument of `(app <callee> <argument>)`, where the
    /// callee is `callee[0]` and the application stands in `parent`.
    fn applied_slot(&self, callee: &'p [Expr], parent: &Slot<'p>, scope: &Scope) -> Slot<'p> {
        // A one-argument application of a variable: `Cons` needs two
        // arguments, so only the callee's name and arity matter.
        let Some(name) = callee.first().and_then(untyped_variable) else {
            return Slot::PLAIN;
        };
        if name == "to_tensor" && self.intrinsic(name, scope) {
            let context = match parent.place {
                Place::Value => parent.context.clone(),
                _ => Context::Plain,
            };
            return Slot::new(context, Place::TensorSource);
        }
        if scope.binds(name) {
            return Slot::PLAIN;
        }
        let Some((DeepTag::TFn, _, signature)) =
            self.globals.signatures.get(name).copied().and_then(parts)
        else {
            return Slot::PLAIN;
        };
        signature
            .split_last()
            .and_then(|(_, parameters)| parameters.first())
            .and_then(|parameter| self.globals.element(parameter, Some(name)))
            .map_or(Slot::PLAIN, |element| {
                Slot::new(Context::Parameter(element), Place::Value)
            })
    }

    /// Whether `name` names the intrinsic: no local binder and no top-level
    /// declaration rebinds it.
    fn intrinsic(&self, name: &str, scope: &Scope) -> bool {
        !scope.binds(name) && !self.globals.names.contains(name)
    }

    /// Whether `node` is the first spine node of a tensor literal's finite
    /// element chain where `slot` makes one.
    fn is_chain(&self, node: &'p Node, slot: &Slot<'p>) -> bool {
        let children = node.children_slice();
        children.len() == 3
            && children.first().and_then(untyped_variable) == Some("Cons")
            && match slot.place {
                Place::Chain => true,
                Place::TensorSource | Place::Element => finite_chain(node.meta(), children),
                Place::Value => false,
            }
    }

    /// Visit a tensor literal's finite element chain without recursing along
    /// its spine.
    fn chain(
        &mut self,
        mut node: &'p Node,
        context: Context<'p>,
        scope: &mut Scope,
        binders: Option<&'p str>,
    ) {
        let depth = self.ancestors.len();
        loop {
            self.ancestors.push(address(node));
            self.metadata(node.meta(), Some(node), scope, binders);
            let children = node.children_slice();
            self.expression_in(
                &children[0],
                Slot::PLAIN,
                scope,
                binders,
            );
            self.expression_in(
                &children[1],
                Slot::new(context.clone(), Place::Element),
                scope,
                binders,
            );
            match &children[2] {
                Expr::Node(rest, _)
                    if rest.children_slice().len() == 3
                        && rest.children_slice().first().and_then(untyped_variable)
                            == Some("Cons") =>
                {
                    node = rest;
                }
                tail => {
                    self.expression_in(tail, Slot::PLAIN, scope, binders);
                    break;
                }
            }
        }
        self.ancestors.truncate(depth);
    }

    /// The operand of `node` when it is `(app (var neg) <literal>)` with a
    /// non-negative numeric literal and `neg` is the intrinsic.
    fn negated_literal(&self, node: &'p Node, scope: &Scope) -> Option<&'p Node> {
        if node.tag() != DeepTag::App || node.meta().ty().is_some() {
            return None;
        }
        let [callee, Expr::Node(operand, _)] = node.children_slice() else {
            return None;
        };
        (untyped_variable(callee) == Some("neg")
            && self.intrinsic("neg", scope)
            && operand.tag() == DeepTag::Lit
            && numeric_atom(operand).is_some_and(|atom| !is_negative(atom)))
        .then_some(&**operand)
    }

    fn literal(&mut self, node: &'p Node, slot: Slot<'p>, negated: bool) {
        let Some(atom) = numeric_atom(node) else {
            return;
        };
        if !negated
            && let Some(stage) = self.stage
            && is_marked(node)
        {
            let change = if is_negative(atom) && !slot.folds(atom) {
                Change::Unfold {
                    marked: stage == Stage::Provisional,
                }
            } else {
                let (dtype, adopted) = slot.decide(atom);
                Change::Retype {
                    dtype: dtype.into(),
                    marked: stage == Stage::Provisional || adopted,
                }
            };
            if change_alters(node, &change) {
                self.record(node, change);
            }
        }
        self.analysis.literals.insert(address(node), Site { slot, negated });
    }

    fn negation(&mut self, node: &'p Node, operand: &'p Node, slot: Slot<'p>) {
        let atom = numeric_atom(operand).expect("a negated literal is numeric");
        if let Some(stage) = self.stage
            && is_marked(operand)
        {
            let change = if slot.folds(atom) {
                let value = negate(atom);
                let (dtype, adopted) = slot.decide(&value);
                Some((
                    address(node),
                    Change::Fold {
                        dtype: dtype.into(),
                        marked: stage == Stage::Provisional || adopted,
                        value,
                    },
                ))
            } else {
                // The operand of an ordinary negation stands in no position.
                let (dtype, _) = Slot::PLAIN.decide(atom);
                let change = Change::Retype {
                    dtype: dtype.into(),
                    marked: stage == Stage::Provisional,
                };
                change_alters(operand, &change).then(|| (address(operand), change))
            };
            if let Some((key, change)) = change {
                self.ancestors.push(address(node));
                self.record_at(key, change);
                self.ancestors.pop();
            }
        }
        self.analysis.literals.insert(
            address(operand),
            Site {
                slot,
                negated: true,
            },
        );
    }

    fn record(&mut self, node: &Node, change: Change) {
        self.record_at(address(node), change);
    }

    fn record_at(&mut self, key: usize, change: Change) {
        self.changes.changes.insert(key, change);
        for ancestor in self.ancestors.iter().rev() {
            if !self.changes.dirty.insert(*ancestor) {
                break;
            }
        }
    }

    /// Visit the live expressions in a node's metadata, which stand in no
    /// position. `owner` is the node the metadata belongs to.
    fn metadata(
        &mut self,
        meta: &'p Metadata,
        owner: Option<&'p Node>,
        scope: &mut Scope,
        binders: Option<&'p str>,
    ) {
        let mut leaves = Vec::new();
        meta.visit_expressions(&mut |value, role| {
            if role == crate::metadata::MetadataRole::Expression {
                leaves.push(value);
            }
        });
        if leaves.is_empty() {
            return;
        }
        let before = self.changes.changes.len();
        if let Some(invariant) = meta.invariant() {
            scope.locals.push(invariant.binder().name().value().clone());
        }
        for leaf in leaves {
            self.expression_in(leaf, Slot::PLAIN, scope, binders);
        }
        if meta.invariant().is_some() {
            scope.locals.pop();
        }
        if self.changes.changes.len() != before
            && let Some(owner) = owner
        {
            self.changes.dirty_meta.insert(address(owner));
        }
    }
}

/// The slot of a block binding's value. A value carrying its binding's
/// declared tensor type is position 1; a type the value merely has (an
/// `inferred` origin) declares nothing.
fn declared_binding_slot<'p>(
    globals: &Globals<'p>,
    value: &'p Expr,
    binders: Option<&'p str>,
) -> Slot<'p> {
    let Some((_, meta, _)) = parts(value) else {
        return Slot::PLAIN;
    };
    if meta
        .surf_binding_type()
        .is_some_and(|origin| *origin.value() == BindingTypeOrigin::Inferred)
    {
        return Slot::PLAIN;
    }
    meta.ty()
        .and_then(|ty| globals.element(ty.expression(), binders))
        .map_or(Slot::PLAIN, |element| {
            Slot::new(Context::Declared(element), Place::Value)
        })
}

/// Whether a change makes any difference to the literal node it names.
fn change_alters(node: &Node, change: &Change) -> bool {
    match change {
        Change::Retype { dtype, marked } => {
            let meta = node.meta();
            let current = meta.ty().and_then(|ty| Dtype::of_type(ty.expression()));
            let integer_source = meta.literal_source().is_some();
            current != Some(dtype.borrowed())
                || is_marked(node) != *marked
                || integer_source != needs_integer_source(node, dtype.borrowed())
        }
        Change::Fold { .. } | Change::Unfold { .. } => true,
    }
}

fn needs_integer_source(node: &Node, dtype: Dtype<'_>) -> bool {
    matches!(numeric_atom(node), Some(Atom::Int(_))) && dtype.is_float()
}

impl Changes {
    /// Apply the changes to the expression they were decided on.
    fn apply(&self, expr: &mut Expr) {
        if self.changes.is_empty() {
            return;
        }
        stacker::maybe_grow(
            crate::nesting::STACK_RED_ZONE_BYTES,
            4 * 1024 * 1024,
            || match expr {
                Expr::Node(node, span) => {
                    let key = address(node);
                    if let Some(change) = self.changes.get(&key) {
                        *expr = changed(node, *span, change);
                        return;
                    }
                    if !self.dirty.contains(&key) && !self.dirty_meta.contains(&key) {
                        return;
                    }
                    let span = *span;
                    let Expr::Node(node, _) =
                        std::mem::replace(expr, Expr::Atom(Atom::Bool(false), span))
                    else {
                        unreachable!("matched a node")
                    };
                    let rebuild_meta = self.dirty_meta.contains(&key);
                    let (tag, meta, mut children) = node.into_parts();
                    let meta = if rebuild_meta {
                        self.rebuilt_metadata(&meta)
                    } else {
                        meta
                    };
                    for child in &mut children {
                        self.apply(child);
                    }
                    *expr = Expr::Node(Box::new(Node::new(tag, meta, children)), span);
                }
                Expr::MetaExpr(wrapper, _) => self.apply(&mut wrapper.expr),
                Expr::BareList(items, _) => {
                    for item in items {
                        self.apply(item);
                    }
                }
                Expr::UnknownForm(data) => {
                    for child in &mut data.children {
                        self.apply(child);
                    }
                }
                Expr::Atom(..) | Expr::Map(..) => {}
            },
        );
    }

    /// Metadata whose expression leaves are rebuilt with the changes.
    fn rebuilt_metadata(&self, meta: &Metadata) -> Metadata {
        meta.try_map_leaves::<crate::metadata::MetadataError>(&mut |leaf, _| {
            Ok(self.rebuilt(leaf))
        })
        .expect("an adopted literal keeps its metadata admissible")
    }

    /// A copy of `expr`, an expression of the analyzed program, with the
    /// changes applied.
    fn rebuilt(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::Node(node, span) => {
                let key = address(node);
                if let Some(change) = self.changes.get(&key) {
                    return changed(node, *span, change);
                }
                if !self.dirty.contains(&key) && !self.dirty_meta.contains(&key) {
                    return expr.clone();
                }
                let meta = if self.dirty_meta.contains(&key) {
                    self.rebuilt_metadata(node.meta())
                } else {
                    node.meta().clone()
                };
                let children = node
                    .children_slice()
                    .iter()
                    .map(|child| self.rebuilt(child))
                    .collect();
                Expr::Node(Box::new(Node::new(node.tag(), meta, children)), *span)
            }
            Expr::MetaExpr(wrapper, span) => Expr::MetaExpr(
                crate::ast::MetaExpr {
                    metadata: wrapper.metadata.clone(),
                    expr: Box::new(self.rebuilt(&wrapper.expr)),
                },
                *span,
            ),
            Expr::BareList(items, span) => {
                Expr::BareList(items.iter().map(|item| self.rebuilt(item)).collect(), *span)
            }
            Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
                head: data.head.clone(),
                meta: data.meta.clone(),
                children: data
                    .children
                    .iter()
                    .map(|child| self.rebuilt(child))
                    .collect(),
                span: data.span,
            })),
            Expr::Atom(..) | Expr::Map(..) => expr.clone(),
        }
    }
}

/// The node a change makes of `node`, whose expression span is `span`.
fn changed(node: &Node, span: Span, change: &Change) -> Expr {
    match change {
        Change::Retype { dtype, marked } => {
            literal(node.meta(), dtype.borrowed(), *marked, atom_expr(node).clone(), span)
        }
        Change::Fold {
            dtype,
            marked,
            value,
        } => {
            // The signed literal takes the operand's annotations and the
            // negation's location.
            let [_, Expr::Node(operand, _)] = node.children_slice() else {
                unreachable!("a folded negation applies `neg` to a literal")
            };
            let mut meta = operand.meta().clone();
            match node.meta().span_id() {
                Some(id) => {
                    meta.replace(MetadataValue::Span(id.clone()));
                }
                None => {
                    meta.remove(MetadataKey::Span);
                }
            }
            let atom = Expr::Atom(value.clone(), atom_expr(operand).span());
            literal(&meta, dtype.borrowed(), *marked, atom, span)
        }
        Change::Unfold { marked } => {
            let original = atom_expr(node);
            let Expr::Atom(value, atom_span) = original else {
                unreachable!("a literal holds an atom")
            };
            let magnitude = negate(value);
            let dtype = Dtype::default_of(&magnitude);
            let mut meta = node.meta().clone();
            meta.remove(MetadataKey::Span);
            let operand = literal(&meta, dtype, *marked, Expr::Atom(magnitude, *atom_span), span);
            let mut application = Metadata::default();
            if let Some(id) = node.meta().span_id() {
                application
                    .insert(MetadataValue::Span(id.clone()))
                    .expect("one span");
            }
            let neg = Expr::node(
                DeepTag::Var,
                Metadata::default(),
                vec![Expr::Atom(Atom::Name("neg".to_string()), Span::new(0, 0))],
                Span::new(0, 0),
            );
            Expr::node(DeepTag::App, application, vec![neg, operand], span)
        }
    }
}

/// A literal of `atom` at `dtype`, keeping `meta`'s other annotations and
/// carrying the unsuffixed marker when `marked`.
fn literal(meta: &Metadata, dtype: Dtype<'_>, marked: bool, atom: Expr, span: Span) -> Expr {
    let mut meta = meta.clone();
    meta.replace(MetadataValue::Type(
        TypeSyntax::try_new(dtype.type_expr()).expect("a dtype is type syntax"),
    ));
    if matches!(atom, Expr::Atom(Atom::Int(_), _)) && dtype.is_float() {
        meta.replace(MetadataValue::LiteralSource(Spanned::new(
            LiteralOrigin::Integer,
            Span::new(0, 0),
        )));
    } else {
        meta.remove(MetadataKey::LiteralSource);
    }
    if marked {
        meta.replace(MetadataValue::SurfLiteralStyle(Spanned::new(
            LiteralStyle::Unsuffixed,
            Span::new(0, 0),
        )));
    } else {
        meta.remove(MetadataKey::SurfLiteralStyle);
    }
    Expr::node(DeepTag::Lit, meta, vec![atom], span)
}

/// A literal node's atom expression.
fn atom_expr(node: &Node) -> &Expr {
    &node.children_slice()[0]
}

fn address(node: &Node) -> usize {
    node as *const Node as usize
}

fn parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr {
        Expr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        _ => None,
    }
}

fn name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

/// The name of a `var` node without a type, the only spelling of a reference
/// a position reads.
fn untyped_variable(expr: &Expr) -> Option<&str> {
    match parts(expr)? {
        (DeepTag::Var, meta, [Expr::Atom(Atom::Name(name), _)]) if meta.ty().is_none() => {
            Some(name)
        }
        _ => None,
    }
}

/// Whether `(app <meta> (var Cons) <element> <rest>)` starts a finite chain
/// of untyped `Cons` applications ending in an untyped `Nil`: the chain a
/// bracket literal desugars to.
fn finite_chain(meta: &Metadata, children: &[Expr]) -> bool {
    if meta.ty().is_some() {
        return false;
    }
    let mut tail = &children[2];
    loop {
        if untyped_variable(tail) == Some("Nil") {
            return true;
        }
        match parts(tail) {
            Some((DeepTag::App, meta, [cons, _, rest]))
                if meta.ty().is_none() && untyped_variable(cons) == Some("Cons") =>
            {
                tail = rest;
            }
            _ => return false,
        }
    }
}

/// Bind the names a `params` node introduces.
fn bind_parameters(params: &Expr, scope: &mut Scope) {
    let Some((DeepTag::Params, _, parameters)) = parts(params) else {
        return;
    };
    scope.locals.extend(
        parameters
            .iter()
            .filter_map(|parameter| match parameter {
                Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
                Expr::BareList(elements, _) => elements.first().and_then(name),
                Expr::MetaExpr(wrapper, _) => name(&wrapper.expr),
                _ => None,
            })
            .map(str::to_string),
    );
}

fn is_macro_definition(expr: &Expr) -> bool {
    matches!(expr, Expr::BareList(elements, _)
        if matches!(elements.as_slice(), [Expr::Atom(Atom::Name(head), _), Expr::Map(..), ..]
            if head == "defmacro"))
}

fn numeric_atom(node: &Node) -> Option<&Atom> {
    match node.children_slice() {
        [Expr::Atom(atom @ (Atom::Int(_) | Atom::Float(_)), _)] => Some(atom),
        _ => None,
    }
}

fn is_marked(node: &Node) -> bool {
    node.meta()
        .surf_literal_style()
        .is_some_and(|style| *style.value() == LiteralStyle::Unsuffixed)
}

fn is_negative(atom: &Atom) -> bool {
    match atom {
        Atom::Int(value) => *value < 0,
        Atom::Float(value) => value.is_sign_negative(),
        _ => false,
    }
}

/// The negation of a literal's value. The magnitude of the `i64` minimum is
/// its own lexer sentinel.
fn negate(atom: &Atom) -> Atom {
    match atom {
        Atom::Int(value) => Atom::Int(value.checked_neg().unwrap_or(i64::MIN)),
        Atom::Float(value) => Atom::Float(-*value),
        other => other.clone(),
    }
}
