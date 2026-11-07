//! [04-LIN-9] and spec/04 section 1.1: the closed allow-list of the
//! operations a key-carrying value may reach.
//!
//! A key reaches an operation only when this list names it; every other
//! operation refuses it, so an operation nobody listed can never admit a key.
//! The linearity checker reads the list through [`tag_keys`] (each Deep tag)
//! and [`builtin_key_operand`] (each builtin operand), and the IR verifier's
//! key rules read it through each graph operation's input slots
//! (`chelis_dag_core::verify::SlotRead::admission`), so the two cannot disagree
//! about which operations take a key.

use chelis_deep::DeepTag;

use crate::builtins::{BuiltinSiblingCaseId, CaseKeys, KeyRouting, case_keys};
use chelis_numeric_core::key_admission::{KeyAdmission, KeyPrimitive};

/// Why an operation refuses a key-carrying operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyRefusal {
    /// The operation's atom names no `key` (spec/04 section 1.1).
    NotNamed,
    /// The operation reads its operand and leaves it live: a borrow, a
    /// copy, an observation of anything but its extent ([04-LIN-9]).
    Read,
    /// The builtin hands each value of `parameter` to its callback and also
    /// keeps it in its result, so one key would be used twice ([04-LIN-9]).
    CallbackAndResult { parameter: &'static str },
}

/// What a Deep tag does with a key-carrying value among its runtime
/// operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKeys {
    Admits(KeyAdmission),
    /// An application: its callee decides, operand by operand
    /// ([`builtin_key_operand`] for a builtin, [`KeyAdmission::KeyParameter`]
    /// for anything else).
    ByCallee,
    Refuses(KeyRefusal),
    /// Not a runtime expression, or a runtime expression with no runtime
    /// operand.
    NoOperand,
}

/// The allow-list entry of every Deep tag. Exhaustive, with no wildcard arm:
/// a new tag is a compile error here until it decides.
pub const fn tag_keys(tag: DeepTag) -> TagKeys {
    use KeyAdmission::{Aggregate, Join, Move};
    match tag {
        DeepTag::App => TagKeys::ByCallee,
        // A handler region (`with device(..) { .. }`) lowers to its body, so
        // its result moves out as a block's does; its handler is a literal
        // the effects gate admits, never a runtime operand.
        DeepTag::Let | DeepTag::Block | DeepTag::HandleEffect | DeepTag::Fn => {
            TagKeys::Admits(Move)
        }
        DeepTag::If | DeepTag::Match => TagKeys::Admits(Join),
        DeepTag::Record
        | DeepTag::Access
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::RecordUpdate => TagKeys::Admits(Aggregate),
        DeepTag::Borrow | DeepTag::Copy => TagKeys::Refuses(KeyRefusal::Read),
        // `grad`, `vmap` and `jit` take a function, which carries no key;
        // their application passes keys to its target's key parameters.
        DeepTag::Par
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice => TagKeys::Refuses(KeyRefusal::NotNamed),
        DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Module
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
        | DeepTag::Arm
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
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Kv
        | DeepTag::Effects
        | DeepTag::Resource => TagKeys::NoOperand,
    }
}

/// The builtins that read their tensor operand's extent and nothing else:
/// [`KeyAdmission::ExtentObservation`] at operand 0.
pub const EXTENT_OBSERVATIONS: [&str; 2] = ["shape", "numel"];

/// The verdict of builtin `name`, under the sibling cases a call selects, on
/// a key-carrying operand at zero-based `index`. A key primitive admits its
/// key operand, and an extent observation its tensor operand; any other
/// builtin admits an operand only when every selected case routes that
/// operand's type parameter to one consumer. A builtin with no sibling case
/// (the Numeric domain alone) admits none.
pub fn builtin_key_operand(
    name: &str,
    cases: &[BuiltinSiblingCaseId],
    index: usize,
) -> Result<KeyAdmission, KeyRefusal> {
    if let Some(primitive) = KeyPrimitive::of_builtin(name) {
        return if index == KeyPrimitive::KEY_OPERAND {
            Ok(KeyAdmission::Primitive(primitive))
        } else {
            Err(KeyRefusal::NotNamed)
        };
    }
    if EXTENT_OBSERVATIONS.contains(&name) {
        return if index == 0 {
            Ok(KeyAdmission::ExtentObservation)
        } else {
            Err(KeyRefusal::NotNamed)
        };
    }
    let mut admitted = Err(KeyRefusal::NotNamed);
    for case in cases {
        admitted = Ok(case_key_operand(*case, index)?);
    }
    admitted
}

/// One sibling case's verdict on a key-carrying operand at `index`: the
/// type parameter declared at that operand decides by its routing, and an
/// operand with no declared type parameter admits no key.
pub fn case_key_operand(
    case: BuiltinSiblingCaseId,
    index: usize,
) -> Result<KeyAdmission, KeyRefusal> {
    let parameters = match case_keys(case) {
        CaseKeys::NoKeyOperand | CaseKeys::Refused(_) => return Err(KeyRefusal::NotNamed),
        CaseKeys::Values(parameters) => parameters,
    };
    let Some(parameter) = parameters
        .iter()
        .find(|parameter| parameter.site.operand() == index)
    else {
        return Err(KeyRefusal::NotNamed);
    };
    match parameter.routing {
        KeyRouting::OneConsumer if case == BuiltinSiblingCaseId::DropValue => {
            Ok(KeyAdmission::Drop)
        }
        KeyRouting::OneConsumer => Ok(KeyAdmission::RoutingCase),
        KeyRouting::Borrowed => Err(KeyRefusal::Read),
        KeyRouting::CallbackAndResult => Err(KeyRefusal::CallbackAndResult {
            parameter: parameter.name,
        }),
    }
}
