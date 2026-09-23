// A returned shared symbolic ADT is expanded by field accesses. Its outer
// borrow can end at a branch while field ownership survives into a loop join.
#[derive(Clone, Copy)]
pub enum Op {
    First,
    Second,
}

pub struct Node {
    pub op: Op,
    pub values: [u32; 2],
}

pub fn identity(node: &Node) -> &Node {
    node
}

pub fn selected_field(node: &Node, take_second: bool, mut repeats: u32) -> u32 {
    let returned = identity(node);
    let first = &returned.values[0];
    let second = &returned.values[1];
    let mut selected = first;
    if take_second {
        match &returned.op {
            Op::First => selected = first,
            Op::Second => selected = second,
        }
    }
    while repeats > 0 {
        if *selected == *first {
            selected = second;
        } else {
            selected = first;
        }
        repeats -= 1;
    }
    *selected
}

// The quantified Fn bound introduces binder-local region declarations inside
// generic arguments. They must not be refreshed like actual free lifetimes.
pub fn apply_predicate<F: for<'a> Fn(&'a u32) -> bool>(f: F, value: &u32) -> bool {
    f(value)
}

pub fn quantified_iterator(values: &[u32], threshold: &u32) -> usize {
    values.iter().filter(|value| {
        apply_predicate(|candidate| *candidate > *threshold, value)
    }).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_payload_and_quantified_predicate_survive_joins() {
        let node = Node { op: Op::Second, values: [7, 11] };
        assert_eq!(selected_field(&node, false, 0), 7);
        assert_eq!(selected_field(&node, true, 0), 11);
        assert_eq!(selected_field(&node, true, 1), 7);
        assert_eq!(selected_field(&node, false, 3), 11);
        assert_eq!(quantified_iterator(&node.values, &8), 1);
    }
}
