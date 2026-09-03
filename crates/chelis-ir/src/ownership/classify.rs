use std::fmt;

use chelis_types::types::Prim;

use crate::host_type_state::ConcreteHostType;

/// Closed heap universe from compiled-value-ownership C4. Tensor storage has
/// no `ConcreteHostType`; it remains listed so downstream planners cannot grow
/// a second heap-kind vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum HeapKind {
    String,
    Tensor,
    TensorStorage,
    List,
    Tuple,
    Dict,
    Adt,
    Option,
    MappedFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum NonHeapKind {
    Scalar(Prim),
    Unit,
    ContextualCallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ValueClass {
    NonHeap(NonHeapKind),
    Heap(HeapKind),
}

impl ValueClass {
    pub(crate) fn is_heap(self) -> bool {
        matches!(self, Self::Heap(_))
    }
}

impl fmt::Display for ValueClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placement {
    Parameter,
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClassifyError {
    FirstClassFunction,
    FunctionContainer,
}

impl fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirstClassFunction => f.write_str("first-class function value (chelis#879)"),
            Self::FunctionContainer => {
                f.write_str("container recursively holds a function value (chelis#879)")
            }
        }
    }
}

pub(crate) fn classify(
    ty: &ConcreteHostType,
    placement: Placement,
) -> Result<ValueClass, ClassifyError> {
    use ConcreteHostType as T;
    Ok(match ty {
        T::Scalar(prim) => classify_prim(*prim),
        T::Function(_, _) if placement == Placement::Parameter => {
            ValueClass::NonHeap(NonHeapKind::ContextualCallback)
        }
        T::Function(_, _) => return Err(ClassifyError::FirstClassFunction),
        T::Adt(_, children) => container(children.iter(), HeapKind::Adt)?,
        T::List(child) => container(std::iter::once(child.as_ref()), HeapKind::List)?,
        T::Dict(key, value) => {
            container([key.as_ref(), value.as_ref()].into_iter(), HeapKind::Dict)?
        }
        T::Tuple(children) => container(children.iter(), HeapKind::Tuple)?,
        T::Tensor(_) => ValueClass::Heap(HeapKind::Tensor),
        T::Option(child) => container(std::iter::once(child.as_ref()), HeapKind::Option)?,
        T::MappedFile => ValueClass::Heap(HeapKind::MappedFile),
        T::Unit => ValueClass::NonHeap(NonHeapKind::Unit),
    })
}

fn classify_prim(prim: Prim) -> ValueClass {
    match prim {
        Prim::String => ValueClass::Heap(HeapKind::String),
        Prim::F32
        | Prim::F64
        | Prim::F16
        | Prim::Bf16
        | Prim::F8e4m3
        | Prim::Int8
        | Prim::Int16
        | Prim::Int32
        | Prim::Int64
        | Prim::Bool => ValueClass::NonHeap(NonHeapKind::Scalar(prim)),
    }
}

fn container<'a>(
    children: impl Iterator<Item = &'a ConcreteHostType>,
    kind: HeapKind,
) -> Result<ValueClass, ClassifyError> {
    if children.into_iter().any(contains_function) {
        Err(ClassifyError::FunctionContainer)
    } else {
        Ok(ValueClass::Heap(kind))
    }
}

fn contains_function(ty: &ConcreteHostType) -> bool {
    use ConcreteHostType as T;
    match ty {
        T::Function(_, _) => true,
        T::Adt(_, xs) | T::Tuple(xs) => xs.iter().any(contains_function),
        T::List(x) | T::Option(x) => contains_function(x),
        T::Dict(k, v) => contains_function(k) || contains_function(v),
        T::Scalar(_) | T::Tensor(_) | T::MappedFile | T::Unit => false,
    }
}

pub(crate) fn render_type(ty: &ConcreteHostType) -> String {
    match ty {
        ConcreteHostType::Scalar(prim) => prim.name().to_string(),
        other => format!("{other:?}"),
    }
}
