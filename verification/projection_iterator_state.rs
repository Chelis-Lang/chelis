// A Map owns two disjoint mutable closure captures alongside shared captures.
// Ending collect's projections must preserve the iterator and untouched captures,
// and return both mutable updates rather than selecting one complete Map state.
pub fn mapped_state(
    values: &[Option<u32>],
    total: &mut u32,
    slots: &mut Vec<u32>,
    source: &u32,
) -> Vec<u32> {
    values
        .iter()
        .enumerate()
        .map(|(axis, value)| match value {
            Some(value) => *value,
            None => {
                *total += *source + axis as u32;
                let slot = slots.len() as u32;
                slots.push(*total);
                slot
            }
        })
        .collect()
}

pub fn recovered() -> (Vec<u32>, u32, Vec<u32>, u32) {
    let mut total = 10;
    let mut slots = vec![5];
    let source = 7;
    let output = mapped_state(
        &[Some(40), None, Some(60), None],
        &mut total,
        &mut slots,
        &source,
    );
    (output, total, slots, source)
}

// any must stop at its matching item, then next resumes with both the mapper
// captures and the underlying iterator advanced exactly once per mapped item.
pub fn resumed() -> (bool, Option<u32>, Option<u32>, u32, Vec<u32>, u32) {
    let mut total = 10;
    let mut slots = Vec::new();
    let source = 7;
    let values = [2, 3, 5];
    let mut mapped = values.iter().map(|value| {
        total += *value + source;
        slots.push(total);
        total
    });
    let found = mapped.any(|value| value >= 29);
    let next = mapped.next();
    let exhausted = mapped.next();
    (found, next, exhausted, total, slots, source)
}

// Reusing the captured shared state across two mutable iterator loops must not
// replace the second loop's projection with the first loop's ended region mask.
pub fn sequential_projected_loops(
    first: &mut [Option<u32>],
    second: &mut [Option<u32>],
    offset: &u32,
) {
    let mapped = |value| value + *offset;
    for slot in first {
        if let Some(value) = slot {
            *value = mapped(*value);
        }
    }
    for slot in second {
        if let Some(value) = slot {
            *value = mapped(*value);
        }
    }
}

fn advance_optional(slot: Option<&mut u32>) -> Option<&mut u32> {
    let value = slot?;
    *value += 1;
    Some(value)
}

pub fn optional_reference_transfer() -> (u32, u32, bool) {
    let mut callee_updated = 41;
    let _ = advance_optional(Some(&mut callee_updated));
    let mut caller_updated = 41;
    if let Some(returned) = advance_optional(Some(&mut caller_updated)) {
        *returned = 50;
    }
    let missing = advance_optional(None).is_none();
    (callee_updated, caller_updated, missing)
}

fn reborrow_optional<'a, 'b>(slot: &'a mut Option<&'b mut u32>) -> Option<&'a mut u32> {
    match slot {
        Some(value) => Some(&mut **value),
        None => None,
    }
}

fn update_nested_optional(slot: &mut Option<&mut u32>) {
    let mut index = 0;
    while index < 2 {
        if let Some(value) = reborrow_optional(slot) {
            *value += index + 1;
        }
        index += 1;
    }
}

// Each child reborrow must restore its parent before the next iteration, then
// the retained outer borrow must return both writes to the original owner.
pub fn nested_reference_restoration() -> (u32, bool) {
    let mut value = 10;
    let mut present = Some(&mut value);
    update_nested_optional(&mut present);
    let mut absent = None;
    update_nested_optional(&mut absent);
    (value, absent.is_none())
}

fn reborrow_step(execution: Option<&mut u32>, value: u32) -> Result<u32, ()> {
    if let Some(state) = execution {
        *state += value;
    }
    if value == 0 {
        Err(())
    } else {
        Ok(value)
    }
}

pub fn fallible_reborrow_loop(mut execution: Option<&mut u32>, values: &[u32]) -> Result<u32, ()> {
    let mut total = 0;
    for &value in values.iter().rev() {
        total += reborrow_step(execution.as_deref_mut(), value)?;
    }
    Ok(total)
}

// Both normal completion and a ? exit must return the real accumulated writes.
pub fn observe_fallible_reborrow(initial: u32, values: &[u32]) -> (Result<u32, ()>, u32) {
    let mut state = initial;
    let result = fallible_reborrow_loop(Some(&mut state), values);
    (result, state)
}

// Rejected elements still update the callback; exhaustion must return those
// writes along with the advanced iterator rather than the last yielded state.
pub fn filtered_callback_state(values: &[u32]) -> (Option<u32>, Option<u32>, Option<u32>, u32) {
    let mut calls = 0;
    let (first, second, exhausted) = {
        let mut selected = values.iter().filter_map(|value| {
            calls += 1;
            if *value % 2 == 0 {
                Some(*value + calls)
            } else {
                None
            }
        });
        (selected.next(), selected.next(), selected.next())
    };
    (first, second, exhausted, calls)
}

pub fn filtered_shared_capture(rank: usize, reduced: &usize) -> Vec<usize> {
    (0..rank).filter(|axis| axis != reduced).collect()
}

pub fn filtered_shared_mutable_captures(values: &[u32], excluded: &u32) -> (Vec<u32>, u32) {
    let mut calls = 0;
    let selected = values
        .iter()
        .copied()
        .filter(|value| {
            calls += 1;
            value != excluded
        })
        .collect();
    (selected, calls)
}

#[cfg(test)]
mod tests {
    #[test]
    fn nested_shared_capture_preserves_filter_decisions() {
        assert_eq!(super::filtered_shared_capture(4, &1), vec![0, 2, 3]);
        assert_eq!(super::filtered_shared_capture(0, &1), Vec::<usize>::new());
    }

    #[test]
    fn nested_shared_capture_does_not_discard_mutable_callback_updates() {
        assert_eq!(
            super::filtered_shared_mutable_captures(&[1, 2, 2, 4], &2),
            (vec![1, 4], 4)
        );
    }

    #[test]
    fn recovers_disjoint_updates_and_preserves_shared_state() {
        assert_eq!(
            super::recovered(),
            (vec![40, 1, 60, 2], 28, vec![5, 18, 28], 7)
        );
    }

    #[test]
    fn resumes_mapper_and_iterator_after_short_circuit() {
        assert_eq!(
            super::resumed(),
            (true, Some(41), None, 41, vec![19, 29, 41], 7)
        );
    }

    #[test]
    fn preserves_projected_state_across_sequential_loops() {
        let mut first = [Some(2), None, Some(5)];
        let mut second = [None, Some(9)];
        let offset = 3;
        super::sequential_projected_loops(&mut first, &mut second, &offset);
        assert_eq!(first, [Some(5), None, Some(8)]);
        assert_eq!(second, [None, Some(12)]);
        assert_eq!(offset, 3);
    }

    #[test]
    fn option_transfer_restores_callee_and_caller_mutations() {
        assert_eq!(super::optional_reference_transfer(), (42, 50, true));
    }

    #[test]
    fn nested_reborrows_restore_parent_state_between_iterations() {
        assert_eq!(super::nested_reference_restoration(), (13, true));
    }

    #[test]
    fn reborrowed_loop_exits_restore_completed_mutations() {
        assert_eq!(super::observe_fallible_reborrow(5, &[2, 3]), (Ok(5), 10));
        assert_eq!(
            super::observe_fallible_reborrow(5, &[2, 0, 3]),
            (Err(()), 8)
        );
        assert_eq!(super::fallible_reborrow_loop(None, &[2, 3]), Ok(5));
    }

    #[test]
    fn filtered_callback_keeps_rejected_and_terminal_updates() {
        assert_eq!(
            super::filtered_callback_state(&[1, 2, 3, 4, 5]),
            (Some(4), Some(8), None, 5)
        );
        assert_eq!(super::filtered_callback_state(&[]), (None, None, None, 0));
    }
}
