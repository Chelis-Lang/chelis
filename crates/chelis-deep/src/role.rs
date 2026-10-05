//! Child-role classification for the Deep vocabulary (chelis#908).
//!
//! `child_stamp_role` classifies every (tag, child_index) pair into a
//! role that determines how the stamp pass and consumers handle that
//! child. `arity_contract` gives the legal child count per tag.
//! `bypass_child_expectation` gives the stamp-time expectation for
//! children at bypass slots.
//!
//! All three are total over `DeepTag` — adding a variant without an
//! entry is a compile error (deny-lint on wildcard matches).

use crate::tag::DeepTag;

/// The role a child occupies relative to its parent node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChildStampRole {
    /// Traversed by ordinary expression inference.
    RuntimeExpr,
    /// Compiler/source syntax preserved verbatim.
    Syntax,
    /// A field, axis, projection, or transform selector.
    Selector,
    /// Handler payload whose literal-form contract is owned by
    /// `chelis-effects`, not expression inference.
    EffectHandler,
    /// A declaration, parameter, or binding name.
    Binder,
    /// Type/dimension syntax resolved by its owning type consumer.
    Type,
    /// Traversed by a dedicated inference owner rather than `infer_expr`
    /// on the structural parent.
    ExplicitInferenceBypass,
}

/// The recursive grammar role expected while stamping a serialized type.
///
/// Unlike [`ChildStampRole::Type`], this table distinguishes actual types,
/// tensor element types, tensor axes, dimensions, and rank spreads. It is the
/// single structural authority used by the public type-fragment ingress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypeSyntaxRole {
    Type,
    NominalArgument,
    TensorElement,
    TensorAxis,
    Dimension,
    Rank,
    Name,
    Integer,
}

/// The intrinsic grammar role of a Deep node in a serialized type tree.
///
/// This match is deliberately total over the closed vocabulary. Adding a new
/// Deep tag cannot accidentally make it type syntax: it must be classified
/// here and in [`type_syntax_child_role`] in the same change.
pub fn type_syntax_node_role(tag: DeepTag) -> Option<TypeSyntaxRole> {
    use TypeSyntaxRole::{Dimension, Rank, Type};

    match tag {
        DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple => Some(Type),
        DeepTag::DName | DeepTag::DVar | DeepTag::DLit => Some(Dimension),
        DeepTag::DRank => Some(Rank),

        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Def
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Fn
        | DeepTag::App
        | DeepTag::Let
        | DeepTag::Match
        | DeepTag::Arm
        | DeepTag::If
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Record
        | DeepTag::Access
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::HandleEffect
        | DeepTag::Borrow
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Kv
        | DeepTag::Effects
        | DeepTag::Resource => None,
    }
}

/// The grammar role of one child of a type/dimension/rank node.
///
/// Callers first prove that `tag` is in [`type_syntax_node_role`]. Returning
/// `None` for every other vocabulary tag keeps the grammar fail-closed.
pub fn type_syntax_child_role(tag: DeepTag, index: usize, arity: usize) -> Option<TypeSyntaxRole> {
    use TypeSyntaxRole::{Integer, Name, NominalArgument, TensorAxis, TensorElement, Type};

    match tag {
        DeepTag::TPrim | DeepTag::TVar => Some(Name),
        DeepTag::TFn | DeepTag::TRef | DeepTag::TTuple => Some(Type),
        DeepTag::TTensor => {
            if index + 1 == arity {
                Some(TensorElement)
            } else {
                Some(TensorAxis)
            }
        }
        DeepTag::TAdt => {
            if index == 0 {
                Some(Name)
            } else {
                Some(NominalArgument)
            }
        }
        // `t-unit` has no legal children. Returning a role lets Node's
        // ordinary arity gate own any extra-child diagnostic.
        DeepTag::TUnit => Some(Type),
        DeepTag::DName | DeepTag::DVar | DeepTag::DRank => Some(Name),
        DeepTag::DLit => Some(Integer),

        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Def
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Fn
        | DeepTag::App
        | DeepTag::Let
        | DeepTag::Match
        | DeepTag::Arm
        | DeepTag::If
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Record
        | DeepTag::Access
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::HandleEffect
        | DeepTag::Borrow
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Kv
        | DeepTag::Effects
        | DeepTag::Resource => None,
    }
}

/// Whether `tag` is legal at the requested recursive type-syntax role.
pub fn type_syntax_role_accepts_tag(role: TypeSyntaxRole, tag: DeepTag) -> bool {
    use TypeSyntaxRole::{Dimension, NominalArgument, Rank, TensorAxis, TensorElement, Type};

    match role {
        Type => type_syntax_node_role(tag) == Some(Type),
        NominalArgument => matches!(type_syntax_node_role(tag), Some(Type | Dimension)),
        TensorElement => matches!(tag, DeepTag::TPrim | DeepTag::TVar),
        TensorAxis => matches!(type_syntax_node_role(tag), Some(Dimension | Rank)),
        Dimension => type_syntax_node_role(tag) == Some(Dimension),
        Rank => type_syntax_node_role(tag) == Some(Rank),
        TypeSyntaxRole::Name | TypeSyntaxRole::Integer => false,
    }
}

/// Legal child count for a vocabulary tag.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AritySpec {
    Fixed(usize),
    AtLeast(usize),
    Range(usize, usize),
}

/// What the stamp pass expects at a bypass slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BypassExpectation {
    /// Requires a vocabulary head from the given classifier.
    /// Undecodable head → StampError.
    RequiresDeclaration,
    /// Requires a specific single tag.
    RequiresTag(DeepTag),
    /// Requires a pattern-vocabulary head.
    RequiresPattern,
    /// Form-expecting (same as RuntimeExpr rule — undecodable → UnknownForm).
    FormExpecting,
}

/// The content shape a form's semantics reads out of one child slot.
///
/// `spec/04-type-system.md` §10 [04-TOT-4] requires a form that reads a child
/// through a partial extraction to name, on failure, "the form and the shape it
/// expected". This enum is that vocabulary: a closed set of shapes with one
/// spelling each, so no consumer composes the phrase itself and no two
/// diagnostics describe the same expectation differently. `Display` renders the
/// shape as a noun phrase, which the consumer places after "expected".
///
/// It is separate from [`BypassExpectation`], which classifies what the STAMP
/// pass requires of a bypass child. This one classifies what a form's checker
/// disposition reads, which is why it names families of value rather than
/// vocabulary heads.
///
/// The set holds exactly the shapes a consumer reads today. A slot that joins
/// the seam brings its own variant in the same change, so the enum never
/// carries a spelling nothing produces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotShape {
    /// A symbol naming a declared record field (`access`, `kv`).
    FieldName,
    /// A symbol naming a constructor (`pat-ctor`, `pat-record`).
    ConstructorName,
    /// A symbol naming a value binding (`pat-var`, `pat-as`).
    BindingName,
    /// A symbol naming a named operation mode (`cast`).
    ModeSelector,
    /// An integer axis (`vmap`).
    IntegerAxis,
    /// A non-negative integer projection index (`tuple-get`).
    TupleIndex,
    /// An integer parameter index, or a tuple of them (`grad`'s `wrt`).
    ParameterIndices,
    /// A scalar literal value (`pat-lit`). spec/03-deep-syntax.md section 6.3
    /// fixes this as a value rather than an expression node: "patterns do not
    /// contain expression nodes".
    LiteralValue,
}

impl core::fmt::Display for SlotShape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let text = match self {
            SlotShape::FieldName => "a symbol field name",
            SlotShape::ConstructorName => "a symbol constructor name",
            SlotShape::BindingName => "a symbol binding name",
            SlotShape::ModeSelector => "a symbol mode selector",
            SlotShape::IntegerAxis => "an integer axis",
            SlotShape::TupleIndex => "a non-negative integer index",
            SlotShape::ParameterIndices => {
                "an integer parameter index or a tuple of integer parameter indices"
            }
            SlotShape::LiteralValue => "a scalar literal value",
        };
        f.write_str(text)
    }
}

/// Exhaustive child-role table for the closed Deep vocabulary.
///
/// Total over `DeepTag` — no wildcard arm.
pub fn child_stamp_role(tag: DeepTag, index: usize, _arity: usize) -> ChildStampRole {
    use ChildStampRole::*;

    match tag {
        DeepTag::Module => {
            if index == 0 {
                Binder
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Import | DeepTag::ImportAll | DeepTag::Export => Syntax,

        DeepTag::Def => {
            if index == 0 {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Defsig => match (index, _arity) {
            (0, _) => Binder,
            (1, 3..) => Syntax,
            _ => Type,
        },
        DeepTag::Deftype | DeepTag::Typealias => {
            if index == 0 {
                Binder
            } else if index == 1 {
                // The numbered Deep spec makes this a structural list of
                // binder names, not a type expression. Treating `(a b)` as a
                // type asks the decoder to interpret `a` as a type head.
                Syntax
            } else {
                Type
            }
        }
        DeepTag::Variant | DeepTag::Field => {
            if index == 0 {
                Binder
            } else {
                Type
            }
        }
        DeepTag::Defdim => Binder,

        DeepTag::Fn => {
            if index == 0 {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::App
        | DeepTag::If
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::Par
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Copy
        | DeepTag::Borrow
        | DeepTag::Unquote
        | DeepTag::Splice => RuntimeExpr,

        DeepTag::HandleEffect => {
            if index == 0 {
                EffectHandler
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Let => {
            if index == 0 {
                ExplicitInferenceBypass
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Match => {
            if index == 0 {
                RuntimeExpr
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Arm => {
            if index == 0 {
                ExplicitInferenceBypass
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Var | DeepTag::Lit => Syntax,
        DeepTag::Record => {
            if index == 0 {
                Type
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Access => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::TupleGet => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::RecordUpdate => {
            if index == 0 {
                RuntimeExpr
            } else {
                ExplicitInferenceBypass
            }
        }

        DeepTag::PatVar => Binder,
        DeepTag::PatLit => Syntax,
        DeepTag::PatCtor | DeepTag::PatRecord => {
            if index == 0 {
                Selector
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::PatTuple => ExplicitInferenceBypass,
        DeepTag::PatWild => Syntax,
        DeepTag::PatAs => {
            if index == 0 {
                Binder
            } else {
                ExplicitInferenceBypass
            }
        }

        DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TRef
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank => Type,

        DeepTag::Grad | DeepTag::Vmap => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::Cast => match index {
            0 => RuntimeExpr,
            1 => Type,
            // The optional [05-OP-6] mode child: a bare selector symbol
            // (`trunc`), read exactly like `grad`'s index selector. The
            // mode is structural, not metadata, so `strip_metadata` (the
            // canonical-display path) cannot silently turn a truncating
            // cast back into the checked default.
            _ => Selector,
        },

        DeepTag::Quote | DeepTag::Effects | DeepTag::Resource => Syntax,

        DeepTag::Params => Binder,
        DeepTag::Bind => {
            if index.is_multiple_of(2) {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Kv => {
            if index == 0 {
                Selector
            } else {
                ExplicitInferenceBypass
            }
        }
    }
}

/// Legal child count for each vocabulary tag.
///
/// Total over `DeepTag` — no wildcard arm.
pub fn arity_contract(tag: DeepTag) -> AritySpec {
    use AritySpec::*;
    match tag {
        DeepTag::Module => AtLeast(0),  // [name] + declarations
        DeepTag::Import => Fixed(2),    // module-path, name-list
        DeepTag::ImportAll => Fixed(1), // module-path
        DeepTag::Export => AtLeast(1),  // names

        DeepTag::Def => Fixed(2),       // name, body
        DeepTag::Defsig => Range(2, 3), // name, [binders], type
        DeepTag::Deftype => AtLeast(2), // name, params [+ variants]
        DeepTag::Typealias => Fixed(3), // name, params, type
        DeepTag::Variant => AtLeast(1), // name [+ fields]
        DeepTag::Field => Fixed(2),     // name, type
        DeepTag::Defdim => Fixed(1),    // name

        DeepTag::Fn => Fixed(2),             // params, body
        DeepTag::App => AtLeast(1),          // callee + args
        DeepTag::Let => Fixed(2),            // bindings, body
        DeepTag::Match => AtLeast(2),        // scrutinee + arms
        DeepTag::Arm => Fixed(3),            // pattern, guard, body
        DeepTag::If => Fixed(3),             // cond, then, else
        DeepTag::Var => Fixed(1),            // name
        DeepTag::Lit => Fixed(1),            // value
        DeepTag::Record => AtLeast(1),       // type-name + fields
        DeepTag::Access => Fixed(2),         // expr, field
        DeepTag::Block => AtLeast(1),        // expressions
        DeepTag::Tuple => AtLeast(0),        // elements (unit = empty)
        DeepTag::TupleGet => Fixed(2),       // expr, index
        DeepTag::RecordUpdate => AtLeast(2), // expr + fields
        DeepTag::Par => AtLeast(1),          // expressions
        DeepTag::HandleEffect => Fixed(2),   // handler-expr, body
        DeepTag::Borrow => Fixed(1),         // expr

        DeepTag::PatVar => Fixed(1),      // name
        DeepTag::PatLit => Fixed(1),      // value
        DeepTag::PatCtor => AtLeast(1),   // name [+ sub-patterns]
        DeepTag::PatTuple => AtLeast(0),  // sub-patterns
        DeepTag::PatRecord => AtLeast(1), // name [+ field patterns]
        DeepTag::PatWild => Fixed(0),     // no children
        DeepTag::PatAs => Fixed(2),       // name, sub-pattern

        DeepTag::TPrim => Fixed(1),     // name
        DeepTag::TFn => AtLeast(1),     // arg-types... + return-type (0-arg fn has just return)
        DeepTag::TTensor => AtLeast(1), // dtype + dims
        DeepTag::TRef => Fixed(1),      // inner type
        DeepTag::TAdt => AtLeast(1),    // name [+ type-params]
        DeepTag::TVar => Fixed(1),      // name
        DeepTag::TUnit => Fixed(0),     // no children
        DeepTag::TTuple => AtLeast(0),  // element types

        DeepTag::DName => Fixed(1), // name
        DeepTag::DVar => Fixed(1),  // name
        DeepTag::DLit => Fixed(1),  // value
        DeepTag::DRank => Fixed(1), // rank-expr

        DeepTag::Grad => Range(1, 2), // expr [+ selector]
        DeepTag::Vmap => AtLeast(1),  // expr [+ selectors]
        DeepTag::Jit => Fixed(1),     // expr
        DeepTag::Realize => Fixed(1), // expr
        DeepTag::Cast => Range(2, 3), // expr, type, optional mode selector
        DeepTag::Copy => Fixed(1),    // expr

        DeepTag::Quote => Fixed(1),   // expr
        DeepTag::Unquote => Fixed(1), // expr
        DeepTag::Splice => Fixed(1),  // expr

        DeepTag::Params => AtLeast(0), // param names
        // An empty bind is the canonical representation of a let with no
        // local bindings. Pair parity is a structural validation rule, not a
        // minimum-arity rule.
        DeepTag::Bind => AtLeast(0),    // zero or more name-value pairs
        DeepTag::Kv => Fixed(2),        // key, value
        DeepTag::Effects => AtLeast(0), // effect names
        DeepTag::Resource => Fixed(1),  // device name
    }
}

/// Whether a tag is a declaration (used by bypass_child_expectation for Module).
pub fn is_declaration_tag(tag: DeepTag) -> bool {
    match tag {
        DeepTag::Def
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Defdim
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export => true,

        DeepTag::Module
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Fn
        | DeepTag::App
        | DeepTag::Let
        | DeepTag::Match
        | DeepTag::Arm
        | DeepTag::If
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Record
        | DeepTag::Access
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::HandleEffect
        | DeepTag::Borrow
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Kv
        | DeepTag::Effects
        | DeepTag::Resource => false,
    }
}

/// Whether a tag is a pattern (used by bypass_child_expectation for Match arms).
pub fn is_pattern_tag(tag: DeepTag) -> bool {
    match tag {
        DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs => true,

        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Def
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Fn
        | DeepTag::App
        | DeepTag::Let
        | DeepTag::Match
        | DeepTag::Arm
        | DeepTag::If
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Record
        | DeepTag::Access
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::HandleEffect
        | DeepTag::Borrow
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Kv
        | DeepTag::Effects
        | DeepTag::Resource => false,
    }
}

/// Stamp-time expectation for children at bypass slots.
///
/// Only called for (tag, index) pairs where `child_stamp_role` returns
/// `ExplicitInferenceBypass`. Panics on non-bypass pairs (consumer bug).
pub fn bypass_child_expectation(tag: DeepTag, index: usize) -> BypassExpectation {
    use BypassExpectation::*;
    match tag {
        DeepTag::Module => RequiresDeclaration,
        DeepTag::Match => RequiresTag(DeepTag::Arm),
        DeepTag::Record | DeepTag::RecordUpdate => RequiresTag(DeepTag::Kv),
        DeepTag::Let => RequiresTag(DeepTag::Bind),
        DeepTag::Arm => RequiresPattern,
        DeepTag::PatCtor => RequiresPattern,
        DeepTag::PatRecord => RequiresTag(DeepTag::Kv),
        DeepTag::PatTuple => RequiresPattern,
        DeepTag::PatAs => RequiresPattern,
        DeepTag::Kv => FormExpecting,
        // Tags with no bypass slot, enumerated explicitly so this function
        // keeps the module's totality promise: adding a `DeepTag` variant is
        // a compile error here until the new tag is classified, the same
        // mutation-oracle contract `child_stamp_role` carries
        // (`spec/design/checker_totality.md` §Phase 3 oracle). Reaching this
        // arm at runtime is still a consumer bug: `child_stamp_role` never
        // reports `ExplicitInferenceBypass` for these tags.
        DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Def
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Fn
        | DeepTag::App
        | DeepTag::If
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Access
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::Par
        | DeepTag::HandleEffect
        | DeepTag::Borrow
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatWild
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Effects
        | DeepTag::Resource => panic!(
            "bypass_child_expectation called for non-bypass (tag={:?}, index={index})",
            tag
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_stamp_role_is_total_over_deeptag() {
        // Every tag returns a role for at least index 0 without panicking.
        for tag in DeepTag::ALL {
            let _ = child_stamp_role(tag, 0, 1);
        }
    }

    #[test]
    fn arity_contract_is_total_over_deeptag() {
        for tag in DeepTag::ALL {
            let spec = arity_contract(tag);
            // Every spec must have a non-negative minimum.
            match spec {
                AritySpec::Fixed(n) => assert!(n <= 100, "{:?}", tag),
                AritySpec::AtLeast(n) => assert!(n <= 100, "{:?}", tag),
                AritySpec::Range(lo, hi) => assert!(lo <= hi, "{:?}", tag),
            }
        }
    }

    #[test]
    fn defsig_binder_role_preserves_the_arity_owner_for_overlong_forms() {
        assert_eq!(
            child_stamp_role(DeepTag::Defsig, 1, 2),
            ChildStampRole::Type
        );
        assert_eq!(
            child_stamp_role(DeepTag::Defsig, 1, 3),
            ChildStampRole::Syntax
        );
        assert_eq!(
            child_stamp_role(DeepTag::Defsig, 1, 4),
            ChildStampRole::Syntax
        );
    }

    #[test]
    fn serialized_type_grammar_classifies_the_exact_type_dimension_and_rank_tags() {
        let classified = DeepTag::ALL
            .into_iter()
            .filter_map(|tag| type_syntax_node_role(tag).map(|role| (tag, role)))
            .collect::<Vec<_>>();
        assert_eq!(
            classified,
            vec![
                (DeepTag::TPrim, TypeSyntaxRole::Type),
                (DeepTag::TFn, TypeSyntaxRole::Type),
                (DeepTag::TTensor, TypeSyntaxRole::Type),
                (DeepTag::TRef, TypeSyntaxRole::Type),
                (DeepTag::TAdt, TypeSyntaxRole::Type),
                (DeepTag::TVar, TypeSyntaxRole::Type),
                (DeepTag::TUnit, TypeSyntaxRole::Type),
                (DeepTag::TTuple, TypeSyntaxRole::Type),
                (DeepTag::DName, TypeSyntaxRole::Dimension),
                (DeepTag::DVar, TypeSyntaxRole::Dimension),
                (DeepTag::DLit, TypeSyntaxRole::Dimension),
                (DeepTag::DRank, TypeSyntaxRole::Rank),
            ]
        );

        for tag in DeepTag::ALL {
            assert_eq!(
                type_syntax_node_role(tag).is_some(),
                type_syntax_child_role(tag, 0, 1).is_some(),
                "type grammar node and child tables disagree for {tag:?}"
            );
        }
    }

    #[test]
    fn serialized_type_grammar_keeps_tensor_axes_and_elements_in_their_namespaces() {
        assert_eq!(
            type_syntax_child_role(DeepTag::TTensor, 0, 3),
            Some(TypeSyntaxRole::TensorAxis)
        );
        assert_eq!(
            type_syntax_child_role(DeepTag::TTensor, 2, 3),
            Some(TypeSyntaxRole::TensorElement)
        );
        for tag in [DeepTag::DName, DeepTag::DVar, DeepTag::DLit, DeepTag::DRank] {
            assert!(type_syntax_role_accepts_tag(
                TypeSyntaxRole::TensorAxis,
                tag
            ));
            assert!(!type_syntax_role_accepts_tag(
                TypeSyntaxRole::TensorElement,
                tag
            ));
        }
        for tag in [DeepTag::TPrim, DeepTag::TVar] {
            assert!(type_syntax_role_accepts_tag(
                TypeSyntaxRole::TensorElement,
                tag
            ));
            assert!(!type_syntax_role_accepts_tag(
                TypeSyntaxRole::TensorAxis,
                tag
            ));
        }
    }

    #[test]
    fn serialized_nominal_arguments_admit_types_and_dimensions_but_not_rank_spreads() {
        assert_eq!(
            type_syntax_child_role(DeepTag::TAdt, 1, 2),
            Some(TypeSyntaxRole::NominalArgument)
        );
        for tag in [
            DeepTag::TPrim,
            DeepTag::TAdt,
            DeepTag::DName,
            DeepTag::DVar,
            DeepTag::DLit,
        ] {
            assert!(type_syntax_role_accepts_tag(
                TypeSyntaxRole::NominalArgument,
                tag
            ));
        }
        assert!(!type_syntax_role_accepts_tag(
            TypeSyntaxRole::NominalArgument,
            DeepTag::DRank
        ));
        assert!(!type_syntax_role_accepts_tag(
            TypeSyntaxRole::Type,
            DeepTag::DLit
        ));
    }

    #[test]
    fn bypass_child_expectation_covers_all_bypass_slots() {
        // For every (tag, index) where the role is Bypass, the expectation
        // function returns without panicking.
        for tag in DeepTag::ALL {
            for index in 0..10 {
                if child_stamp_role(tag, index, 10) == ChildStampRole::ExplicitInferenceBypass {
                    let _ = bypass_child_expectation(tag, index);
                }
            }
        }
    }

    #[test]
    fn declaration_classifier_is_exhaustive() {
        // Exercises every variant (deny-lint enforces no wildcard).
        for tag in DeepTag::ALL {
            let _ = is_declaration_tag(tag);
        }
    }

    #[test]
    fn pattern_classifier_is_exhaustive() {
        for tag in DeepTag::ALL {
            let _ = is_pattern_tag(tag);
        }
    }

    #[test]
    fn known_declarations() {
        assert!(is_declaration_tag(DeepTag::Def));
        assert!(is_declaration_tag(DeepTag::Defsig));
        assert!(is_declaration_tag(DeepTag::Deftype));
        assert!(is_declaration_tag(DeepTag::Import));
        assert!(!is_declaration_tag(DeepTag::App));
        assert!(!is_declaration_tag(DeepTag::Var));
    }

    #[test]
    fn known_patterns() {
        assert!(is_pattern_tag(DeepTag::PatVar));
        assert!(is_pattern_tag(DeepTag::PatCtor));
        assert!(is_pattern_tag(DeepTag::PatWild));
        assert!(!is_pattern_tag(DeepTag::Var));
        assert!(!is_pattern_tag(DeepTag::App));
    }

    #[test]
    fn module_children_expect_declarations() {
        assert_eq!(
            bypass_child_expectation(DeepTag::Module, 1),
            BypassExpectation::RequiresDeclaration
        );
    }

    #[test]
    fn match_children_expect_arm() {
        assert_eq!(
            bypass_child_expectation(DeepTag::Match, 1),
            BypassExpectation::RequiresTag(DeepTag::Arm)
        );
    }

    #[test]
    fn kv_value_is_form_expecting() {
        assert_eq!(
            bypass_child_expectation(DeepTag::Kv, 1),
            BypassExpectation::FormExpecting
        );
    }

    #[test]
    #[should_panic(expected = "non-bypass")]
    fn non_bypass_tag_panics_loudly() {
        // `def`'s children are Binder/RuntimeExpr; asking for a bypass
        // expectation is a consumer bug and must stay a loud panic, not a
        // silent default.
        let _ = bypass_child_expectation(DeepTag::Def, 1);
    }

    #[test]
    fn bypass_expectation_agrees_with_role_table_in_both_directions() {
        // The role table and the expectation table must name the same tag
        // set: a tag has a bypass slot iff `bypass_child_expectation`
        // answers for it. Divergence in either direction means one table
        // was edited without the other.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for tag in DeepTag::ALL {
            let has_bypass_slot = (0..10).any(|index| {
                child_stamp_role(tag, index, 10) == ChildStampRole::ExplicitInferenceBypass
            });
            let answers =
                std::panic::catch_unwind(|| bypass_child_expectation(tag, usize::MAX)).is_ok();
            assert_eq!(
                has_bypass_slot, answers,
                "role table and bypass expectation disagree for {tag:?}"
            );
        }
        std::panic::set_hook(previous_hook);
    }
}
