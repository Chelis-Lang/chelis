//! `AsRef<T> for Cow<'_, T>` (Aeneas.Std Core/Borrow.lean) as chelis-ir `host::lower_app_host_expr`
//! uses it: a `Cow` of a cloneable tree, `Borrowed` or `Owned`, read through `as_ref()`; then a
//! `Cow<str>` (owned as a `String`) and a `Cow<[u32]>` (owned as a `Vec<u32>`).
use std::borrow::Cow;

/// A cloneable node (its `Cow` is owned as itself: `ToOwned` through `Clone`).
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub head: u32,
    pub args: Vec<u32>,
}

fn size(expr: &Expr) -> u32 {
    let mut total = 1;
    for arg in &expr.args {
        total += *arg;
    }
    total + expr.head
}

/// As `lower_app_host_expr`: the flattened chain's expression when there is one (owned), the
/// original expression otherwise (borrowed), then read through `as_ref()`.
pub fn bound_size(expr: &Expr, flatten: bool) -> (u32, Expr) {
    let bind_expr: Cow<'_, Expr> = if flatten {
        let mut args = expr.args.clone();
        args.push(10);
        Cow::Owned(Expr { head: expr.head + 1, args })
    } else {
        Cow::Borrowed(expr)
    };
    let view: &Expr = bind_expr.as_ref();
    (size(view), view.clone())
}

pub fn text_len(text: &str, owned: bool) -> (usize, String) {
    let cow: Cow<'_, str> = if owned { Cow::Owned(format!("{text}!")) } else { Cow::Borrowed(text) };
    let view: &str = cow.as_ref();
    (view.len(), view.to_string())
}

pub fn slice_sum(values: &[u32], owned: bool) -> u32 {
    let cow: Cow<'_, [u32]> = if owned { Cow::Owned(vec![7; values.len()]) } else { Cow::Borrowed(values) };
    let view: &[u32] = cow.as_ref();
    let mut total = 0;
    for v in view {
        total += *v;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exprs() {
        let e = Expr { head: 1, args: vec![2, 3] };
        assert_eq!(bound_size(&e, false), (7, e.clone()));
        assert_eq!(bound_size(&e, true), (18, Expr { head: 2, args: vec![2, 3, 10] }));
    }

    #[test]
    fn texts() {
        assert_eq!(text_len("héllo", false), (6, "héllo".to_string()));
        assert_eq!(text_len("héllo", true), (7, "héllo!".to_string()));
        assert_eq!(slice_sum(&[1, 2, 3], false), 6);
        assert_eq!(slice_sum(&[1, 2, 3], true), 21);
    }
}
