//! Borrow and copy result decisions.

use crate::model::Type;

pub fn settled_borrow_type<E>(resolved: Type<E>) -> Option<Type<E>> {
    match resolved {
        Type::Ref(_) => Some(resolved),
        Type::Tensor(_, _)
        | Type::Adt(_, _)
        | Type::KindedAdt(_, _)
        | Type::Tuple(_)
        | Type::Error(_) => Some(Type::reference(resolved)),
        _ => None,
    }
}

pub fn copy_result_from_source<E: Clone>(settled: &Type<E>) -> Type<E> {
    match settled {
        Type::Ref(referent) => referent.as_ref().clone(),
        owned => owned.clone(),
    }
}
