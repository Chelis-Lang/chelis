//! Closures taking a mutable borrow. Charon's partial monomorphization
//! specializes `FnMut<Self, (&'a mut T, U)>` into a trait with a region
//! parameter `'a`, and lays out its `call_mut` method as `<'self, 'a, ..>`,
//! while the closure's implementation of `call_mut` is laid out as usual,
//! `<'a, .., 'self>`. The region order determines the order of the given back
//! values: the trait method field and its implementations must agree on it
//! (the closure state and the argument have the same type in `counted`, so a
//! disagreement would silently swap them).

pub fn apply2(x: &mut u32, k: &u32, f: impl Fn(&mut u32, &u32) -> u32) -> u32 {
    f(x, k)
}

pub fn scaled(x: &mut u32, k: &u32) -> u32 {
    apply2(x, k, |x, k| {
        *x = x.wrapping_mul(*k);
        *x
    })
}

pub fn apply_twice(x: &mut u32, k: &u32, mut f: impl FnMut(&mut u32, &u32) -> u32) -> u32 {
    let a = f(x, k);
    let b = f(x, k);
    a.wrapping_add(b)
}

/// The closure mutably borrows `total`.
pub fn accumulate(x: &mut u32, k: &u32) -> (u32, u32) {
    let mut total = 0u32;
    let r = apply_twice(x, k, |x, k| {
        total = total.wrapping_add(*k);
        *x = x.wrapping_add(total);
        *x
    });
    (r, total)
}

/// The closure owns its state, which the first call updates for the second.
pub fn counted(x: &mut u32, k: &u32) -> u32 {
    let mut n = 0u32;
    apply_twice(x, k, move |x, k| {
        n = n.wrapping_add(1);
        *x = x.wrapping_mul(*k).wrapping_add(n);
        *x
    })
}

pub fn apply_once(x: &mut u32, k: &u32, f: impl FnOnce(&mut u32, &u32) -> u32) -> u32 {
    f(x, k)
}

pub fn once(x: &mut u32, k: &u32) -> u32 {
    let mut total = 5u32;
    let r = apply_once(x, k, |x, k| {
        total = total.wrapping_add(*k);
        *x = x.wrapping_add(total);
        *x
    });
    r.wrapping_add(total)
}

pub fn bump_twice(x: &mut u32, mut f: impl FnMut(&mut u32) -> u32) -> u32 {
    let a = f(x);
    let b = f(x);
    a.wrapping_add(b)
}

pub fn bumped(x: &mut u32) -> (u32, u32) {
    let mut step = 1u32;
    let r = bump_twice(x, |x| {
        step = step.wrapping_mul(3);
        *x = x.wrapping_add(step);
        *x
    });
    (r, step)
}

/// The closure owns its state and takes a single mutable borrow.
pub fn stepped(x: &mut u32) -> u32 {
    let mut step = 2u32;
    bump_twice(x, move |x| {
        step = step.wrapping_mul(step);
        *x = x.wrapping_add(step);
        *x
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_native() {
        for (x, k, r) in [(3u32, 7u32, 21u32), (0, 5, 0), (u32::MAX, 2, u32::MAX - 1)] {
            let mut x = x;
            assert_eq!(scaled(&mut x, &k), r);
            assert_eq!(x, r);
        }
    }

    #[test]
    fn accumulate_native() {
        for (x, k, out, x_after) in [(1u32, 2u32, (10u32, 4u32), 7u32), (0, 0, (0, 0), 0)] {
            let mut x = x;
            assert_eq!(accumulate(&mut x, &k), out);
            assert_eq!(x, x_after);
        }
    }

    #[test]
    fn counted_native() {
        for (x, k, r, x_after) in [(2u32, 3u32, 30u32, 23u32), (5, 0, 3, 2)] {
            let mut x = x;
            assert_eq!(counted(&mut x, &k), r);
            assert_eq!(x, x_after);
        }
    }

    #[test]
    fn once_native() {
        let mut x = 1u32;
        assert_eq!(once(&mut x, &2), 15);
        assert_eq!(x, 8);
    }

    #[test]
    fn bumped_native() {
        let mut x = 10u32;
        assert_eq!(bumped(&mut x), (35, 9));
        assert_eq!(x, 22);
    }

    #[test]
    fn stepped_native() {
        let mut x = 1u32;
        assert_eq!(stepped(&mut x), 5 + 21);
        assert_eq!(x, 21);
    }
}
