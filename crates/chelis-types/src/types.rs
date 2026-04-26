//! Internal type representation for the Chelis type checker.
//!
//! These are the checker's working types — NOT the Deep AST nodes.
//! They mirror the Deep t-* tags but are Rust-native for efficient unification.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A unique identifier for a type variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeVar(pub u32);

/// A unique identifier for a dimension variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DimVar(pub u32);

/// Numeric precision types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Prim {
    F32,
    F64,
    F16,
    Bf16,
    F8e4m3,
    Int8,
    Int32,
    Int64,
    Bool,
    String,
}

impl Prim {
    /// Parse a primitive type from its canonical name.
    pub fn parse_name(s: &str) -> Option<Prim> {
        match s {
            "f32" => Some(Prim::F32),
            "f64" => Some(Prim::F64),
            "f16" => Some(Prim::F16),
            "bf16" => Some(Prim::Bf16),
            "f8e4m3" => Some(Prim::F8e4m3),
            "int8" => Some(Prim::Int8),
            "int32" => Some(Prim::Int32),
            "int64" => Some(Prim::Int64),
            "bool" => Some(Prim::Bool),
            "string" => Some(Prim::String),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Prim::F32 => "f32",
            Prim::F64 => "f64",
            Prim::F16 => "f16",
            Prim::Bf16 => "bf16",
            Prim::F8e4m3 => "f8e4m3",
            Prim::Int8 => "int8",
            Prim::Int32 => "int32",
            Prim::Int64 => "int64",
            Prim::Bool => "bool",
            Prim::String => "string",
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Prim::F32 | Prim::F64 | Prim::F16 | Prim::Bf16 | Prim::F8e4m3
        )
    }

    pub fn is_numeric(&self) -> bool {
        !matches!(self, Prim::Bool | Prim::String)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, Prim::Int8 | Prim::Int32 | Prim::Int64)
    }

    /// Whether this precision is valid as the element type of a tensor in
    /// the current Phase 0f backend. The C/HIP backends support f32, f64,
    /// bool, and integer precisions; reduced floats (f16, bf16, f8e4m3) are
    /// not supported as tensor element types and must be rejected at check
    /// time per the "no implicit precision promotion" rule.
    ///
    /// This intentionally does not affect host scalar precisions — only
    /// tensor precisions are constrained here.
    pub fn is_valid_tensor_precision(&self) -> bool {
        matches!(
            self,
            Prim::F32 | Prim::F64 | Prim::Bool | Prim::Int8 | Prim::Int32 | Prim::Int64
        )
    }

    /// Whether this precision is a legitimate target for `cast(scalar, p)`.
    /// The host scalar lane supports full f64 plus everything tensors can
    /// hold; reduced floats (f16, bf16, f8e4m3) have no scalar
    /// representation and must be rejected at check time.
    pub fn is_valid_scalar_cast_target(&self) -> bool {
        matches!(
            self,
            Prim::F32 | Prim::F64 | Prim::Bool | Prim::Int8 | Prim::Int32 | Prim::Int64
        )
    }
}

/// A tensor dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    /// Concrete named dimension (e.g., batch, hidden).
    Name(String),
    /// Polymorphic dimension variable (e.g., a, b).
    Var(DimVar),
    /// Fixed numeric size (e.g., 512).
    Lit(i64),
    /// Wildcard — unknown/dynamic dimension.
    Wildcard,
}

/// Chelis type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    /// Primitive type (f32, int32, bool, etc.).
    Prim(Prim),
    /// Function type: args → return.
    Fn(Vec<Type>, Box<Type>),
    /// Tensor type: dimensions + precision.
    Tensor(Vec<Dim>, Prim),
    /// Algebraic data type: name + type arguments.
    Adt(String, Vec<Type>),
    /// Type variable (for inference).
    Var(TypeVar),
    /// Tuple type.
    Tuple(Vec<Type>),
    /// Unit type.
    Unit,
    /// Error sentinel — used for partial inference past errors.
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Effect {
    Random,
    Accum,
    Io,
    /// Chelis-native testing effect. Pinned at the root, no handler.
    /// Any function that (directly or transitively) calls a `test_assert_*`
    /// builtin acquires this effect, preventing assertions from silently
    /// leaking into `! {}` production code.
    Test,
    Resource(String),
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random => f.write_str("Random"),
            Self::Accum => f.write_str("Accum"),
            Self::Io => f.write_str("IO"),
            Self::Test => f.write_str("Test"),
            Self::Resource(device) => write!(f, "Resource(\"{device}\")"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectSet {
    effects: BTreeSet<Effect>,
}

impl EffectSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, effect: Effect) {
        self.effects.insert(effect);
    }

    pub fn remove(&mut self, effect: &Effect) {
        self.effects.remove(effect);
    }

    pub fn contains(&self, effect: &Effect) -> bool {
        self.effects.contains(effect)
    }

    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    pub fn extend(&mut self, other: &EffectSet) {
        self.effects.extend(other.effects.iter().cloned());
    }

    pub fn iter(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter()
    }
}

impl FromIterator<Effect> for EffectSet {
    fn from_iter<T: IntoIterator<Item = Effect>>(iter: T) -> Self {
        let mut set = Self::new();
        for effect in iter {
            set.insert(effect);
        }
        set
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rendered: Vec<String> = self.effects.iter().map(ToString::to_string).collect();
        write!(f, "{{{}}}", rendered.join(", "))
    }
}

/// A polymorphic type scheme: ∀ tvars, dvars. body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scheme {
    pub tvars: Vec<TypeVar>,
    pub dvars: Vec<DimVar>,
    pub body: Type,
}

impl Scheme {
    /// A monomorphic scheme (no quantified variables).
    pub fn mono(ty: Type) -> Scheme {
        Scheme {
            tvars: vec![],
            dvars: vec![],
            body: ty,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Prim(p) => write!(f, "{}", p.name()),
            Type::Fn(args, ret) => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "({}) -> {}", arg_strs.join(", "), ret)
            }
            Type::Tensor(dims, prec) => {
                let dim_strs: Vec<String> = dims.iter().map(|d| format!("{d:?}")).collect();
                write!(f, "tensor[{}, {}]", dim_strs.join(", "), prec.name())
            }
            Type::Adt(name, args) if args.is_empty() => write!(f, "{name}"),
            Type::Adt(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{name} {}", arg_strs.join(" "))
            }
            Type::Var(v) => write!(f, "?{}", v.0),
            Type::Tuple(ts) => {
                let strs: Vec<String> = ts.iter().map(|t| t.to_string()).collect();
                write!(f, "({})", strs.join(", "))
            }
            Type::Unit => write!(f, "unit"),
            Type::Error => write!(f, "<error>"),
        }
    }
}

/// Counter for generating fresh type and dimension variables.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct VarGen {
    next_tvar: u32,
    next_dvar: u32,
}

impl VarGen {
    pub fn fresh_tvar(&mut self) -> TypeVar {
        let v = TypeVar(self.next_tvar);
        self.next_tvar += 1;
        v
    }

    pub fn fresh_dvar(&mut self) -> DimVar {
        let v = DimVar(self.next_dvar);
        self.next_dvar += 1;
        v
    }

    pub fn fresh_type(&mut self) -> Type {
        Type::Var(self.fresh_tvar())
    }

    pub fn fresh_dim(&mut self) -> Dim {
        Dim::Var(self.fresh_dvar())
    }
}
