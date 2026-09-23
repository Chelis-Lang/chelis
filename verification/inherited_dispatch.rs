struct State {
    remaining: u32,
}

struct Counter {
    state: State,
}

struct Token;

static STEP: u32 = 2;

fn advance(state: &mut State) -> Option<Token> {
    if state.remaining == 0 {
        None
    } else {
        state.remaining -= STEP;
        Some(Token)
    }
}

impl Iterator for Counter {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        advance(&mut self.state)
    }
}

pub fn count_visible(limit: u32) -> usize {
    Counter {
        state: State { remaining: limit },
    }
    .count()
}

#[cfg(test)]
mod tests {
    use super::count_visible;

    #[test]
    fn inherited_dispatch_restores_method_dependencies() {
        assert_eq!(count_visible(0), 0);
        assert_eq!(count_visible(6), 3);
    }

    #[test]
    fn inherited_dispatch_preserves_method_failure() {
        assert!(std::panic::catch_unwind(|| count_visible(3)).is_err());
    }
}
