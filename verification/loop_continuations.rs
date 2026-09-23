use std::vec::Vec;

pub fn scoped_return(values: Vec<u32>, enabled: bool) -> Result<u32, u32> {
    let mut total = 0;
    if enabled {
        let weights = vec![1u32; values.len()];
        let mut index = 0;
        while index < values.len() {
            if values[index] == 0 {
                return Err(index as u32);
            }
            total += weights[index];
            index += 1;
        }
    }
    Ok(total)
}

pub fn serial(values: &[u32]) -> Result<u32, u32> {
    let mut index = 0;
    while index < values.len() {
        if values[index] == 0 {
            return Err(10);
        }
        index += 1;
    }
    let mut index = 0;
    while index < values.len() {
        if values[index] == 1 {
            return Err(20);
        }
        index += 1;
    }
    Ok(30)
}

pub fn scoped_borrow(values: &mut [u32], enabled: bool) -> Option<&mut u32> {
    if enabled {
        let mut index = 0;
        while index < values.len() {
            if values[index] == 0 {
                return Some(&mut values[index]);
            }
            index += 1;
        }
    }
    None
}

pub fn update_selected(values: &mut [u32], enabled: bool) {
    if let Some(value) = scoped_borrow(values, enabled) {
        *value = 7;
    }
}

// Independent conditionals must remain joined before the returning loop.
pub fn branch_prefix(mut value: u32, choices: [bool; 12]) -> u32 {
    if choices[0] {
        value += 1;
    }
    if choices[1] {
        value += 1;
    }
    if choices[2] {
        value += 1;
    }
    if choices[3] {
        value += 1;
    }
    if choices[4] {
        value += 1;
    }
    if choices[5] {
        value += 1;
    }
    if choices[6] {
        value += 1;
    }
    if choices[7] {
        value += 1;
    }
    if choices[8] {
        value += 1;
    }
    if choices[9] {
        value += 1;
    }
    if choices[10] {
        value += 1;
    }
    if choices[11] {
        value += 1;
    }
    let mut index = 0;
    while index < value {
        if index == 5 {
            return index;
        }
        index += 1;
    }
    value
}

enum Branch {
    Left,
    Right,
    Exit,
}

#[inline(never)]
fn branch(code: u32) -> Branch {
    match code {
        0 => Branch::Left,
        1 => Branch::Right,
        _ => Branch::Exit,
    }
}

// Each independent match has two ordinary paths and an abrupt exit.
// Mutations in the body and after the loop distinguish all three exits.
pub fn mixed_exits(state: &mut u32, first: u32, second: u32, third: u32) -> Result<u32, u32> {
    let mut round = 0;
    while round < 2 {
        round += 1;
        match branch(first) {
            Branch::Left => *state += 1,
            Branch::Right => *state += 2,
            Branch::Exit => {
                *state += 3;
                continue;
            }
        }
        match branch(second) {
            Branch::Left => *state += 10,
            Branch::Right => *state += 20,
            Branch::Exit => {
                *state += 30;
                break;
            }
        }
        match branch(third) {
            Branch::Left => *state += 100,
            Branch::Right => *state += 200,
            Branch::Exit => return Err(*state),
        }
        *state += 1000;
    }
    *state += 10000;
    Ok(*state)
}

pub fn mixed_exit_observation(
    initial: u32,
    first: u32,
    second: u32,
    third: u32,
) -> (Result<u32, u32>, u32) {
    let mut state = initial;
    let result = mixed_exits(&mut state, first, second, third);
    (result, state)
}

pub fn nested_exit(state: &mut u32, outer: u32, inner: u32) -> Result<u32, u32> {
    match branch(outer) {
        Branch::Left => {
            match branch(inner) {
                Branch::Left => *state += 1,
                Branch::Right => *state += 2,
                Branch::Exit => return Err(*state),
            }
            *state += 10;
        }
        Branch::Right => *state += 20,
        Branch::Exit => return Ok(*state),
    }
    *state += 100;
    Ok(*state)
}

pub fn nested_exit_observation(initial: u32, outer: u32, inner: u32) -> (Result<u32, u32>, u32) {
    let mut state = initial;
    let result = nested_exit(&mut state, outer, inner);
    (result, state)
}

// The iterator borrows the temporary vector and the selected elements through
// different lifetimes. Joining its loop must not widen either lender's mask.
pub fn nested_shared_scan(values: &[u32], indices: &[usize], threshold: u32) -> u32 {
    let selected = indices
        .iter()
        .filter_map(|index| values.get(*index))
        .collect::<Vec<_>>();
    let mut count = 0;
    for input in &selected {
        if **input > threshold {
            count += 1;
        }
    }
    count
}

pub enum SharedRoute<'a> {
    Selected(&'a u32),
    Borrowed(&'a u32),
    Owned(u32),
    Pair(&'a u32, &'a u32),
    Empty,
}

// Sharing a wildcard arm must preserve the original constructor and all of its
// payloads when a later match reads the same borrowed value.
pub fn shared_default_route(route: &SharedRoute<'_>) -> (u32, u32) {
    let mut defaults = 0;
    let first = match route {
        SharedRoute::Selected(value) => **value,
        _ => {
            defaults += 1;
            0
        }
    };
    let second = match route {
        SharedRoute::Selected(value) | SharedRoute::Borrowed(value) => **value,
        SharedRoute::Owned(value) => *value,
        SharedRoute::Pair(left, right) => **left + **right,
        SharedRoute::Empty => 0,
    };
    (first + second, defaults)
}

// Recursive calls from the loop body put the wrapper, body, and search in one
// recursive component. A failed branch must continue searching its siblings.
pub fn recursive_search(values: &[u32], start: usize, target: u32) -> Option<usize> {
    if start >= values.len() {
        return None;
    }
    if values[start] == target {
        return Some(start);
    }
    let mut next = start + 1;
    while next < values.len() {
        if values[next] > values[start] {
            if let Some(found) = recursive_search(values, next, target) {
                return Some(found);
            }
        }
        next += 1;
    }
    None
}

pub struct OptionalPayload {
    pub steps: Vec<usize>,
    pub source: Vec<u32>,
    pub generation: u32,
}

pub struct OptionalOwner {
    pub full: Option<OptionalPayload>,
    pub legacy: Vec<u32>,
    pub recorded: usize,
}

// As in Spine::complete, iteration only builds a candidate replacement. Both
// early errors must return the complete original optional payload and owner.
pub fn complete_optional(owner: &mut OptionalOwner, limit: usize) -> Result<(), u32> {
    if let Some(full) = &mut owner.full {
        let mut next = 0;
        let mut steps = Vec::new();
        for node in &full.steps {
            if *node < next || *node >= limit {
                return Err(7);
            }
            while next <= *node {
                steps.push(next);
                next += 1;
            }
        }
        while next < limit {
            steps.push(next);
            next += 1;
        }
        full.steps = steps;
    }
    owner.recorded = limit;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_exit_observations() {
        assert_eq!(scoped_return(vec![2, 0, 3], true), Err(1));
        assert_eq!(scoped_return(vec![2, 3], true), Ok(2));
        assert_eq!(scoped_return(vec![2, 0, 3], false), Ok(0));
        assert_eq!(serial(&[1, 0]), Err(10));
        assert_eq!(serial(&[2, 1]), Err(20));
        assert_eq!(serial(&[2, 3]), Ok(30));
        assert_eq!(serial(&[]), Ok(30));
        let mut values = [2, 0, 0];
        update_selected(&mut values, false);
        assert_eq!(values, [2, 0, 0]);
        update_selected(&mut values, true);
        assert_eq!(values, [2, 7, 0]);
        let mut nonzero = [2, 3];
        update_selected(&mut nonzero, true);
        assert_eq!(nonzero, [2, 3]);
        assert_eq!(branch_prefix(2, [false; 12]), 2);
        assert_eq!(branch_prefix(2, [true; 12]), 5);
    }

    #[test]
    fn mixed_exit_observations() {
        let cases = [
            (0, 0, 0, (Ok(12229), 12229)),
            (1, 1, 1, (Ok(12451), 12451)),
            (2, 2, 2, (Ok(10013), 10013)),
            (0, 2, 2, (Ok(10038), 10038)),
            (0, 0, 2, (Err(18), 18)),
            (1, 1, 2, (Err(29), 29)),
        ];
        for (first, second, third, expected) in cases {
            let mut state = 7;
            let result = mixed_exits(&mut state, first, second, third);
            assert_eq!((result, state), expected);
        }
    }

    #[test]
    fn nested_exit_observations() {
        for (outer, inner, expected) in [
            (0, 0, (Ok(118), 118)),
            (0, 1, (Ok(119), 119)),
            (0, 2, (Err(7), 7)),
            (1, 2, (Ok(127), 127)),
            (2, 0, (Ok(7), 7)),
        ] {
            assert_eq!(nested_exit_observation(7, outer, inner), expected);
        }
        assert!(std::panic::catch_unwind(|| nested_exit_observation(u32::MAX, 0, 0)).is_err());
    }

    #[test]
    fn nested_shared_scan_observations() {
        assert_eq!(nested_shared_scan(&[3, 1, 4], &[2, 99, 0, 2, 1], 2), 3);
        assert_eq!(nested_shared_scan(&[3, 1, 4], &[2, 99, 0, 2, 1], 4), 0);
        assert_eq!(nested_shared_scan(&[3, 1, 4], &[], 2), 0);
    }

    #[test]
    fn shared_default_arm_retains_constructor_and_payloads() {
        assert_eq!(shared_default_route(&SharedRoute::Selected(&7)), (14, 0));
        assert_eq!(shared_default_route(&SharedRoute::Pair(&7, &11)), (18, 1));
        assert_eq!(shared_default_route(&SharedRoute::Owned(11)), (11, 1));
    }

    #[test]
    fn recursive_loop_search_observations() {
        let values = [0, 9, 1, 4, 4];
        assert_eq!(recursive_search(&values, 0, 4), Some(3));
        assert_eq!(recursive_search(&values, 0, 9), Some(1));
        assert_eq!(recursive_search(&values, 0, 2), None);
        assert_eq!(recursive_search(&values, 1, 4), None);
        assert_eq!(recursive_search(&values, 3, 4), Some(3));
        assert_eq!(recursive_search(&values, values.len(), 4), None);
        assert_eq!(recursive_search(&[], 0, 4), None);
    }

    #[test]
    fn optional_completion_observations() {
        for (steps, limit, expected_result, expected_steps, recorded) in [
            (vec![5, 1], 5, Err(7), vec![5, 1], 31),
            (vec![1, 0], 5, Err(7), vec![1, 0], 31),
            (vec![1, 5], 5, Err(7), vec![1, 5], 31),
            (vec![1, 3], 5, Ok(()), vec![0, 1, 2, 3, 4], 5),
            (vec![], 0, Ok(()), vec![], 0),
        ] {
            let mut owner = OptionalOwner {
                full: Some(OptionalPayload {
                    steps,
                    source: vec![13, 17],
                    generation: 19,
                }),
                legacy: vec![23, 29],
                recorded: 31,
            };
            let result = complete_optional(&mut owner, limit);
            let full = owner.full.unwrap();
            assert_eq!(
                (
                    result,
                    full.steps,
                    full.source,
                    full.generation,
                    owner.legacy,
                    owner.recorded
                ),
                (
                    expected_result,
                    expected_steps,
                    vec![13, 17],
                    19,
                    vec![23, 29],
                    recorded
                )
            );
        }
        let mut absent = OptionalOwner {
            full: None,
            legacy: vec![23, 29],
            recorded: 31,
        };
        assert_eq!(complete_optional(&mut absent, 5), Ok(()));
        assert!(absent.full.is_none());
        assert_eq!((absent.legacy, absent.recorded), (vec![23, 29], 5));
    }
}
