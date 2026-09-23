pub enum Payload<T> {
    First(T),
    Second(T),
}

pub fn into_value<T>(payload: Payload<T>) -> T {
    match payload {
        Payload::First(value) => value,
        Payload::Second(value) => value,
    }
}

fn singleton(value: u32) -> Vec<u32> {
    let mut values = Vec::new();
    values.push(value);
    values
}

pub fn append_selected(first: bool) -> Vec<u32> {
    let payload = if first {
        Payload::First(singleton(17))
    } else {
        Payload::Second(singleton(29))
    };
    let mut values = into_value(payload);
    values.push(99);
    values
}

pub fn append_or_default(success: bool) -> Vec<u32> {
    let result: Result<Vec<u32>, Vec<u32>> = if success {
        Ok(singleton(17))
    } else {
        Err(singleton(29))
    };
    let mut values = result.unwrap_or(singleton(41));
    values.push(99);
    values
}

pub fn copy_with_live_borrow<T: Copy>(value: T) -> (T, T) {
    let borrowed = &value;
    let copied = value;
    (*borrowed, copied)
}

pub fn copy_after_borrowed_push(value: u32) -> (u32, u32) {
    let mut borrows = Vec::new();
    borrows.push(&value);
    let copied = value;
    (*borrows[0], copied)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_cleanup_retains_owned_mutation() {
        assert_eq!(append_selected(true), vec![17, 99]);
        assert_eq!(append_selected(false), vec![29, 99]);
    }

    #[test]
    fn result_cleanup_selects_only_returned_buffer() {
        assert_eq!(append_or_default(true), vec![17, 99]);
        assert_eq!(append_or_default(false), vec![41, 99]);
    }

    #[test]
    fn last_direct_copy_keeps_an_indirect_reader_alive() {
        assert_eq!(copy_with_live_borrow(37_u32), (37, 37));
    }

    #[test]
    fn copied_lender_survives_mutating_borrow_storage() {
        assert_eq!(copy_after_borrowed_push(41), (41, 41));
    }
}
