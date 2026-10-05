//! Independent concrete contract models. Linked against each erased mutant.
use chelis_axis_core::{checked_inverse, is_permutation, normalize_axis, reduction_survivors};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn permutation(axes: &[usize], rank: usize) -> bool {
    axes.len() == rank
        && axes.iter().all(|a| *a < rank)
        && axes.iter().enumerate().all(|(i, a)| !axes[..i].contains(a))
}

fn check(function: &str, rank: usize, axes: &[usize], raw: i64, axis: usize) -> bool {
    let expected = match function {
        "is_permutation" => format!("{:?}", permutation(axes, rank)),
        "checked_inverse" => {
            let inverse = if permutation(axes, rank) {
                Some(
                    (0..rank)
                        .map(|a| axes.iter().position(|v| *v == a).unwrap())
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            format!("{inverse:?}")
        }
        "normalize_axis" => {
            let position = if raw < 0 {
                raw as i128 + rank as i128
            } else {
                raw as i128
            };
            let value =
                if rank as u128 <= i64::MAX as u128 && position >= 0 && position < rank as i128 {
                    Some(position as usize)
                } else {
                    None
                };
            format!("{value:?}")
        }
        "reduction_survivors" => {
            let value = if axis < rank {
                Some((0..rank).filter(|p| *p != axis).collect::<Vec<_>>())
            } else {
                None
            };
            format!("{value:?}")
        }
        _ => panic!("unknown function"),
    };
    let actual = catch_unwind(AssertUnwindSafe(|| match function {
        "is_permutation" => format!("{:?}", is_permutation(axes, rank)),
        "checked_inverse" => format!("{:?}", checked_inverse(axes, rank)),
        "normalize_axis" => format!("{:?}", normalize_axis(rank, raw)),
        "reduction_survivors" => format!("{:?}", reduction_survivors(rank, axis)),
        _ => unreachable!(),
    }))
    .unwrap_or_else(|_| "panic".to_owned());
    if expected == actual {
        return false;
    }
    println!(
        "{{\"function\":{function:?},\"rank\":{rank},\"axes\":{axes:?},\"raw\":{raw},\"axis\":{axis},\"expected\":{expected:?},\"actual\":{actual:?},\"within_abi_rank\":{},\"within_axis_i32_domain\":{}}}",
        rank <= i32::MAX as usize,
        if function == "normalize_axis" {
            i32::try_from(raw).is_ok()
        } else {
            axes.iter().all(|a| *a <= i32::MAX as usize) && axis <= i32::MAX as usize
        }
    );
    true
}

fn lists(function: &str, rank: usize, len: usize, axes: &mut Vec<usize>) -> bool {
    if axes.len() == len {
        return check(function, rank, axes, 0, 0);
    }
    for value in (0..=rank).chain([usize::MAX]) {
        axes.push(value);
        if lists(function, rank, len, axes) {
            return true;
        }
        axes.pop();
    }
    false
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let function = std::env::args().nth(1).expect("function required");
    if function == "normalize_axis" {
        for rank in (0..=16).chain([
            i32::MAX as usize,
            i64::MAX as usize,
            i64::MAX as usize + 1,
            usize::MAX,
        ]) {
            let signed = i64::try_from(rank).unwrap_or(i64::MAX);
            for raw in [
                -2,
                -1,
                0,
                1,
                -signed,
                (-signed).saturating_sub(1),
                signed.saturating_sub(1),
                signed,
                signed.saturating_add(1),
                i64::MIN,
                i64::MIN + 1,
                i64::MAX,
            ] {
                if check(&function, rank, &[], raw, 0) {
                    return;
                }
            }
        }
    } else if function == "reduction_survivors" {
        for rank in 0..=16 {
            for axis in (0..=rank + 1).chain([usize::MAX]) {
                if check(&function, rank, &[], 0, axis) {
                    return;
                }
            }
        }
    } else {
        for rank in 0..=4 {
            for len in 0..=rank + 1 {
                if lists(&function, rank, len, &mut Vec::new()) {
                    return;
                }
            }
        }
    }
}
